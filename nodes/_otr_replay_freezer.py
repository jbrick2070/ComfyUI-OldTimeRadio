"""Freeze a finished episode into a REPLAY BUNDLE: the library half.

Moved here from ``scripts/otr_freeze_replay_bundle.py`` (Scene Producer chunk 1,
2026-10-22) so a node can freeze a stage inside the ComfyUI server. The script
stays as the command-line front end and re-exports everything below.

WHAT IS FROZEN: the durable ledger, the master WAV, ``episode_canon.json``, every
file under ``stills/`` and ``portraits/``. With ``include_render_outputs=True``,
also every file under ``clips/`` and ``audio/foley/`` -- the Scene Producer's
Video freeze asks for that; the command line never does, so an ordinary bundle
is exactly what it was. The manifest is a safe import format: relative paths
only, sizes and SHA-256 per file, the source episode id, root and commit; built
in a temporary sibling directory and renamed into place only after every file
was verified by the same validator the import runs.

TWO RULES THIS MODULE KEEPS BECAUSE IT RUNS INSIDE THE SERVER:

* It raises ``ReplayBundleError``, never ``SystemExit``. ComfyUI's prompt worker
  catches ``Exception``; a ``SystemExit`` raised in a node escapes it and stops
  the worker for every prompt queued after it.
* It never spawns a process. The source commit is an ARGUMENT; only the
  command-line wrapper asks git for it. A process spawn under ``nodes/`` is a
  registry-scan finding, and this file ships.

UTF-8, no BOM. Standard library only.
"""
from __future__ import annotations

import contextlib
import functools
import hashlib
import json
import logging
import os
import pathlib
import shutil
from typing import Optional, Union

try:
    from .production_ledger import (
        REPLAY_MANIFEST_NAME, REPLAY_MANIFEST_SCHEMA, ReplayBundleError,
        load_replay_manifest,
    )
    from ._otr_producer_direction import MANIFEST_PRODUCER_KEY, META_IMAGE_ENGINES
except ImportError:  # pragma: no cover -- flat (sys.path) load
    from production_ledger import (  # type: ignore
        REPLAY_MANIFEST_NAME, REPLAY_MANIFEST_SCHEMA, ReplayBundleError,
        load_replay_manifest,
    )
    from _otr_producer_direction import (  # type: ignore
        MANIFEST_PRODUCER_KEY, META_IMAGE_ENGINES)

log = logging.getLogger("OTR.replay_freezer")

PathLike = Union[str, "os.PathLike[str]"]

#: Always frozen: what a replay needs to show the same pictures.
FROZEN_DIRS = ("stills", "portraits")
#: Frozen only with ``include_render_outputs=True``: what a replay needs to
#: reuse filmed clips and their foley stems instead of filming them again.
RENDER_OUTPUT_DIRS = ("clips", "audio/foley")

__all__ = [
    "FROZEN_DIRS", "RENDER_OUTPUT_DIRS", "ReplayBundleError",
    "sha256_file", "find_ledger", "master_digest", "find_master",
    "freeze", "derive_engine_bundle",
]


def sha256_file(path: PathLike) -> str:
    """SHA-256 of a file's bytes, read in 1 MiB blocks."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_ledger(ep_dir: PathLike) -> pathlib.Path:
    """The episode's durable ledger under ``audio/``; the renamed one wins."""
    ep_dir = pathlib.Path(ep_dir)
    hits = sorted((ep_dir / "audio").glob("*_ledger.json"))
    if not hits:
        raise ReplayBundleError("no *_ledger.json under %s" % (ep_dir / "audio"))
    if len(hits) > 1:
        # the renamed episode's ledger carries the episode id; prefer it
        named = [h for h in hits if h.name.startswith(ep_dir.name)]
        hits = named or hits
    return hits[0]


def master_digest(ledger: dict) -> str:
    """The assembler master's digest the ledger stamps, or ""."""
    audio = ledger.get("audio") if isinstance(ledger.get("audio"), dict) else {}
    return str(audio.get("master_audio_sha256") or "").strip().lower()


def find_master(ep_dir: PathLike, ledger: dict) -> pathlib.Path:
    """The ASSEMBLER's master WAV, chosen by the digest the ledger stamps.

    NOT ``final_audio_path`` (plan item 18, 2026-10-14). That pointer is
    truthful about what was DELIVERED, and on a foley episode the delivered
    file is the terminal foley mix. Freezing it made the replay mix an
    already-mixed master a second time. The replay needs the pre-loudness
    assembler stem, which the ledger identifies by digest.

    Every ``audio/*_master.wav`` inside THIS episode is a candidate, pending or
    not -- a renamed episode keeps its ``pending_<ts>_master.wav`` name, and a
    run whose video failed has only that file -- and neither filename nor mtime
    decides: only the digest does. No match refuses, and names the digest.
    """
    ep_dir = pathlib.Path(ep_dir)
    want = master_digest(ledger)
    if not want:
        raise ReplayBundleError(
            "the ledger stamps no audio.master_audio_sha256; it cannot "
            "name its assembler master. Re-render the episode on a "
            "current tree, then freeze it.")
    audio_dir = ep_dir / "audio"
    matches = [p for p in sorted(audio_dir.glob("*_master.wav"))
               if p.is_file() and sha256_file(p) == want]
    if not matches:
        raise ReplayBundleError(
            "no *_master.wav under %s matches the ledger's assembler master %s "
            "(was it cleaned up after publish? freeze with cleanup off)"
            % (audio_dir, want[:12]))
    named = [p for p in matches if p.name.startswith(ep_dir.name)]
    return (named or matches)[0].resolve()


def _building_dir(final: pathlib.Path) -> pathlib.Path:
    """The temporary sibling a bundle is built in before its atomic rename."""
    return final.parent / ("." + final.name + ".building")


def _single_folder_name(name: object, what: str) -> str:
    """``name`` as ONE folder name, or a refusal. The ledger's episode id becomes
    part of a path this module deletes and renames, so a corrupted ledger must
    not be able to point it anywhere else (Codex QA, 2026-10-22)."""
    text = str(name or "")
    if (not text or text in (".", "..") or text != text.strip()
            or any(ch in text for ch in "/\\:\x00")):
        raise ReplayBundleError("%s is not a single safe folder name: %r" % (what, text))
    return text


def _require_bundle_target(final: pathlib.Path, out_root: Optional[PathLike]) -> None:
    """Refuse a bundle directory that is no directory name at all, or that would
    land anywhere but directly inside ``out_root`` -- checked BEFORE the building
    directory is removed or the bundle renamed into place."""
    if not final.name or final.name in (".", ".."):
        raise ReplayBundleError("not a usable bundle directory: %s" % final)
    if out_root is not None:
        root = pathlib.Path(out_root).resolve()
        if final.resolve().parent != root:
            raise ReplayBundleError("bundle %s would land outside %s" % (final, root))


def _missing_ancestors(folder: PathLike) -> list:
    """The folders on the way to ``folder``, itself included, that are not on
    disk yet, outermost first: what ``mkdir(parents=True)`` is about to create.
    Judged by ``lexists``, so a link -- even a dangling one -- counts as present
    and is never listed for removal."""
    missing = []
    probe = pathlib.Path(os.path.abspath(folder))
    while not os.path.lexists(probe):
        missing.append(probe)
        if probe.parent == probe:
            break
        probe = probe.parent
    missing.reverse()
    return missing


def _remove_if_empty(folders: list) -> None:
    """Remove ``folders`` innermost first while each is empty, and stop at the
    first that is not: something else has written into it meanwhile."""
    for folder in reversed(folders):
        try:
            os.rmdir(folder)
        except OSError:
            return


@contextlib.contextmanager
def _building(final: pathlib.Path):
    """Build a bundle in a temporary sibling of ``final``, then rename it into
    place once the block has written it and the import's own validator has
    passed it.

    A refusal anywhere -- the block, the validator, the rename -- removes the
    sibling AND every folder this call created on the way to it, so a retry
    finds the disk exactly as it was (Sonnet QA, 2026-10-22). A refused first
    freeze used to leave the empty project folder it had created, and the Scene
    Producer read that folder as a project already started, refusing the very
    retry its plan promises. The rename is inside the same guard: a failed
    rename used to leave a complete ``.building`` folder behind.
    """
    tmp = _building_dir(final)
    created = _missing_ancestors(tmp.parent)
    try:
        if os.path.lexists(tmp):
            shutil.rmtree(tmp)                    # a stale sibling from a crashed freeze
        tmp.mkdir(parents=True)
        yield tmp
        load_replay_manifest(str(tmp))            # the same validator the import runs
        os.replace(tmp, final)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        _remove_if_empty(created)
        raise


def _refusals_as_bundle_errors(func):
    """Every way a freeze can fail surfaces as ``ReplayBundleError``.

    The module promises that exception, and its callers catch only it: the
    command line turns it into a clean exit, the Scene Producer into a refusal.
    An unreadable ledger (``ValueError``), a target outside the folders a
    replay may be read from (``OtrPathContractError``, also a ``ValueError``)
    or a failed copy or rename (``OSError``) used to escape raw (Sonnet QA,
    2026-10-22). The original stays chained for the traceback.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (OSError, ValueError) as exc:
            raise ReplayBundleError("%s refused: %s" % (func.__name__, exc)) from exc
    return wrapper


@_refusals_as_bundle_errors
def freeze(ep_dir: PathLike, out_root: Optional[PathLike], *,
           allow_no_plan: bool = False,
           dest: Optional[PathLike] = None,
           source_commit: str = "",
           include_render_outputs: bool = False,
           producer: Optional[dict] = None) -> pathlib.Path:
    """Freeze ``ep_dir`` into an immutable bundle and return its directory.

    The bundle lands at ``out_root/<episode_id>`` unless ``dest`` names the
    bundle directory itself; either way it must not exist yet, because a
    bundle is never edited. Given both, ``dest`` must sit directly inside
    ``out_root``. ``source_commit`` is recorded as given (an empty string when
    unknown) -- this function never asks git. Clips and foley stems are frozen
    only with ``include_render_outputs=True``.

    ``producer`` is given only by a Scene Producer stage end: the direction's
    identity, recorded as the manifest's ``producer`` section together with
    the image engines the episode's stills were painted with, read from the
    ledger. A producer freeze of a ledger with no engine stamp is refused. An
    ordinary freeze writes no producer section.

    The episode path is made absolute first: the manifest records it as
    ``source_episode_root``, and a replay rebases its image paths from it, which
    a relative root cannot do.
    """
    ep_dir = pathlib.Path(os.path.abspath(ep_dir))
    ledger_path = find_ledger(ep_dir)
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    if not isinstance(ledger, dict) or not ledger.get("lines"):
        raise ReplayBundleError("ledger carries no lines: %s" % ledger_path)
    video = ledger.get("video") if isinstance(ledger.get("video"), dict) else {}
    if not video.get("shots"):
        msg = ("ledger %s carries no planned video.shots -- a replay of it would re-author "
               "prompts through the LLM, which is not a replay. Render the episode on a tree "
               "where ShotLock stamps the plan durably (2026-09-02+), then freeze that."
               % ledger_path.name)
        if not allow_no_plan:
            raise ReplayBundleError(msg)
        log.warning("[replay freeze] allow_no_plan: %s", msg)
    master = find_master(ep_dir, ledger)
    episode_id = str(ledger.get("episode_id") or ep_dir.name)
    producer_section = None
    if producer is not None:
        meta = ledger.get("meta") if isinstance(ledger.get("meta"), dict) else {}
        engines = meta.get(META_IMAGE_ENGINES)
        if not isinstance(engines, dict) or not engines:
            raise ReplayBundleError(
                "a producer freeze needs the image engines its stills were painted "
                "with, and ledger %s records none (meta %s)" % (ledger_path.name,
                                                                META_IMAGE_ENGINES))
        producer_section = dict(producer, image_engines=dict(engines))

    if dest is not None:
        final = pathlib.Path(dest)
        _require_bundle_target(final, out_root)
    elif out_root is not None:
        final = pathlib.Path(out_root) / _single_folder_name(episode_id,
                                                             "the ledger's episode_id")
        _require_bundle_target(final, out_root)
    else:
        raise ReplayBundleError("freeze needs out_root or dest")
    if final.exists():
        raise ReplayBundleError("bundle already exists: %s (a bundle is immutable; pick --out)"
                                % final)

    files = []

    def _add(src: pathlib.Path, rel: str) -> None:
        if not src.is_file() or src.stat().st_size <= 0:
            raise ReplayBundleError("refusing to freeze a missing or empty file: %s" % src)
        dst = tmp / pathlib.Path(rel)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        files.append({"path": rel, "bytes": dst.stat().st_size, "sha256": sha256_file(dst)})

    with _building(final) as tmp:
        ledger_rel = "audio/" + ledger_path.name
        master_rel = "audio/" + master.name
        _add(ledger_path, ledger_rel)
        _add(master, master_rel)
        if files[-1]["sha256"] != master_digest(ledger):
            raise ReplayBundleError(
                "the copied master %s does not match the ledger's assembler "
                "master %s" % (files[-1]["sha256"][:12], master_digest(ledger)[:12]))
        canon = ep_dir / "episode_canon.json"
        if canon.is_file():
            _add(canon, "episode_canon.json")
        dirs = FROZEN_DIRS + (RENDER_OUTPUT_DIRS if include_render_outputs else ())
        for d in dirs:
            root = ep_dir / pathlib.Path(d)
            if not root.is_dir():
                continue
            for f in sorted(root.rglob("*")):
                if f.is_file():
                    _add(f, f.relative_to(ep_dir).as_posix())
        # every image row the ledger names must be inside the bundle by its basename
        rows = (((ledger.get("images") or {}).get("images") or [])
                if isinstance(ledger.get("images"), dict) else [])
        bundled = {pathlib.Path(r["path"]).name for r in files}
        missing = [str(r.get("path")) for r in rows
                   if isinstance(r, dict) and r.get("path")
                   and pathlib.Path(str(r["path"])).name not in bundled]
        if missing:
            raise ReplayBundleError("ledger image rows not found under stills/ or portraits/: %s"
                                    % missing[:5])

        manifest = {
            "schema_version": REPLAY_MANIFEST_SCHEMA,
            "source_episode_id": episode_id,
            "source_episode_root": str(ep_dir),
            "source_commit": str(source_commit or ""),
            "ledger": ledger_rel,
            "master_audio": master_rel,
            "planned_shots": len(video.get("shots") or []),
            "files": files,
        }
        if producer_section is not None:
            manifest[MANIFEST_PRODUCER_KEY] = producer_section
        (tmp / REPLAY_MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return final


@_refusals_as_bundle_errors
def derive_engine_bundle(bundle_dir: PathLike, engine_id: str,
                         out_root: Optional[PathLike] = None) -> pathlib.Path:
    """A BUNDLE-TO-BUNDLE derivation: the same frozen files, a new manifest that
    names the engine the replay must render on (still-in lab peer, campaign
    item 2, 2026-09-02).

    Never routes through :func:`freeze` -- that function is hard-wired to a live
    episode directory. This one reads the source manifest through the same
    validator the import uses, copies every listed file byte for byte into
    ``<bundle>__engine_<id>``, re-verifies each SHA-256, and writes the source
    manifest plus ``engine_override`` and ``derived_from``. The derived bundle
    is self-contained and immutable like any other;
    ``production_ledger.import_replay_bundle`` stamps the override raw and
    ``OTR_ShotLock``'s replay branch validates and applies it to the whole plan.
    """
    bundle_dir = pathlib.Path(bundle_dir).resolve()
    engine_id = str(engine_id or "").strip()
    if not engine_id:
        raise ReplayBundleError("derive: an engine id is required")
    manifest = load_replay_manifest(str(bundle_dir))
    if str(manifest.get("engine_override") or ""):
        raise ReplayBundleError(
            "derive: %s already carries engine_override=%r -- derive "
            "from the ORIGINAL bundle" % (bundle_dir, manifest["engine_override"]))
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in engine_id)
    out_root = pathlib.Path(out_root) if out_root else bundle_dir.parent
    final = out_root / ("%s__engine_%s" % (bundle_dir.name, safe))
    _require_bundle_target(final, out_root)
    if final.exists():
        raise ReplayBundleError("derived bundle already exists: %s (a bundle is immutable)" % final)
    with _building(final) as tmp:
        for row in manifest["files"]:
            rel = str(row["path"])
            src = bundle_dir / pathlib.Path(rel)
            dst = tmp / pathlib.Path(rel)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            if sha256_file(dst) != str(row.get("sha256") or ""):
                raise ReplayBundleError(
                    "derive: copied file %s does not match its manifest sha256" % rel)
        derived = dict(manifest)
        derived["engine_override"] = engine_id
        derived["derived_from"] = str(bundle_dir)
        (tmp / REPLAY_MANIFEST_NAME).write_text(
            json.dumps(derived, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return final
