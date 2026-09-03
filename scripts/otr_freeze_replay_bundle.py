"""Freeze a finished episode into a REPLAY BUNDLE (campaign item 0, 2026-09-02).

    python scripts/otr_freeze_replay_bundle.py <episode dir or episode id> [--out DIR]

The bundle is what the canonical graph replays: the writer's ``replay_from`` widget
(or ``otr_canonical_api_run.py --replay-from``) points at it, the writer short-circuits
authorship, the audio chain passes through, node 7 copies the frozen master, ShotLock
reuses the PLANNED video section, node 91 verifies the imported stills, and only the
video phase and the publish tail run -- the same seeds, the same audio, a new episode.

WHAT IS FROZEN: the durable ledger, the master WAV, ``episode_canon.json``, every file
under ``stills/`` and ``portraits/``. NOT frozen: clips, composites, the published MP4 --
the replay makes its own. The manifest is a safe import format: relative paths only,
sizes and SHA-256 per file, the source episode id, root and commit; built in a temporary
sibling directory and renamed into place only after every file was verified. Freezing is
explicit and operator-visible; nothing freezes itself.

The ledger MUST carry the planned ``video.shots`` (stamped durably by ShotLock since
2026-09-02) -- a replay without the plan would re-author prompts through the LLM, which is
not a replay. Refused otherwise unless ``--allow-no-plan`` is passed for a diagnostic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nodes.production_ledger import (  # noqa: E402
    REPLAY_MANIFEST_NAME, REPLAY_MANIFEST_SCHEMA, load_replay_manifest,
)

FROZEN_DIRS = ("stills", "portraits")


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=str(REPO_ROOT), text=True).strip()
    except Exception:  # noqa: BLE001
        return ""


def _episodes_root() -> pathlib.Path:
    """The live episodes tree, through the same resolver the ledger uses."""
    from nodes import _otr_paths
    return pathlib.Path(_otr_paths.otr_episodes_root())


def resolve_episode_dir(arg: str) -> pathlib.Path:
    p = pathlib.Path(arg)
    if p.is_dir():
        return p.resolve()
    cand = _episodes_root() / arg
    if cand.is_dir():
        return cand.resolve()
    raise SystemExit("episode dir not found: %r (also tried %s)" % (arg, cand))


def find_ledger(ep_dir: pathlib.Path) -> pathlib.Path:
    hits = sorted((ep_dir / "audio").glob("*_ledger.json"))
    if not hits:
        raise SystemExit("no *_ledger.json under %s" % (ep_dir / "audio"))
    if len(hits) > 1:
        # the renamed episode's ledger carries the episode id; prefer it
        named = [h for h in hits if h.name.startswith(ep_dir.name)]
        hits = named or hits
    return hits[0]


def find_master(ep_dir: pathlib.Path, ledger: dict) -> pathlib.Path:
    fap = str(ledger.get("final_audio_path") or "")
    if fap and pathlib.Path(fap).is_file():
        return pathlib.Path(fap).resolve()
    # the fallback: never a `pending_*` master (the un-renamed provisional file
    # can outlive the rename with a newer mtime); prefer the one carrying the
    # episode id, else the newest
    hits = [p for p in (ep_dir / "audio").glob("*_master.wav")
            if not p.name.startswith("pending_")]
    named = [p for p in hits if p.name.startswith(ep_dir.name)]
    hits = sorted(named or hits, key=lambda p: p.stat().st_mtime)
    if not hits:
        raise SystemExit("no *_master.wav under %s" % (ep_dir / "audio"))
    return hits[-1].resolve()


def freeze(ep_dir: pathlib.Path, out_root: pathlib.Path, *, allow_no_plan: bool = False) -> pathlib.Path:
    ledger_path = find_ledger(ep_dir)
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    if not isinstance(ledger, dict) or not ledger.get("lines"):
        raise SystemExit("ledger carries no lines: %s" % ledger_path)
    video = ledger.get("video") if isinstance(ledger.get("video"), dict) else {}
    if not video.get("shots"):
        msg = ("ledger %s carries no planned video.shots -- a replay of it would re-author "
               "prompts through the LLM, which is not a replay. Render the episode on a tree "
               "where ShotLock stamps the plan durably (2026-09-02+), then freeze that."
               % ledger_path.name)
        if not allow_no_plan:
            raise SystemExit(msg)
        print("WARNING (--allow-no-plan): " + msg)
    master = find_master(ep_dir, ledger)
    episode_id = str(ledger.get("episode_id") or ep_dir.name)

    final = out_root / episode_id
    if final.exists():
        raise SystemExit("bundle already exists: %s (a bundle is immutable; pick --out)" % final)
    tmp = out_root / ("." + episode_id + ".building")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    files = []

    def _add(src: pathlib.Path, rel: str) -> None:
        if not src.is_file() or src.stat().st_size <= 0:
            raise SystemExit("refusing to freeze a missing or empty file: %s" % src)
        dst = tmp / pathlib.Path(rel)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        files.append({"path": rel, "bytes": dst.stat().st_size, "sha256": _sha256(dst)})

    ledger_rel = "audio/" + ledger_path.name
    master_rel = "audio/" + master.name
    _add(ledger_path, ledger_rel)
    _add(master, master_rel)
    canon = ep_dir / "episode_canon.json"
    if canon.is_file():
        _add(canon, "episode_canon.json")
    for d in FROZEN_DIRS:
        root = ep_dir / d
        if not root.is_dir():
            continue
        for f in sorted(root.rglob("*")):
            if f.is_file():
                rel = f.relative_to(ep_dir).as_posix()
                _add(f, rel)
    # every image row the ledger names must be inside the bundle by its basename
    rows = ((ledger.get("images") or {}).get("images") or []) if isinstance(ledger.get("images"), dict) else []
    bundled = {pathlib.Path(r["path"]).name for r in files}
    missing = [str(r.get("path")) for r in rows
               if isinstance(r, dict) and r.get("path") and pathlib.Path(str(r["path"])).name not in bundled]
    if missing:
        raise SystemExit("ledger image rows not found under stills/ or portraits/: %s" % missing[:5])

    manifest = {
        "schema_version": REPLAY_MANIFEST_SCHEMA,
        "source_episode_id": episode_id,
        "source_episode_root": str(ep_dir),
        "source_commit": _git_commit(),
        "ledger": ledger_rel,
        "master_audio": master_rel,
        "planned_shots": len(video.get("shots") or []),
        "files": files,
    }
    (tmp / REPLAY_MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                                            encoding="utf-8")
    load_replay_manifest(str(tmp))            # the same validator the import runs
    os.replace(tmp, final)
    return final


def derive_engine_bundle(bundle_dir: pathlib.Path, engine_id: str,
                         out_root: pathlib.Path | None = None) -> pathlib.Path:
    """A BUNDLE-TO-BUNDLE derivation: the same frozen files, a new manifest that
    names the engine the replay must render on (still-in lab peer, campaign
    item 2, 2026-09-02).

    Never routes through :func:`freeze` -- that function is hard-wired to a live
    episode directory and names its output after the episode id, which would
    collide with the immutability guard on the very bundle being derived. This
    one reads the source manifest through the same validator the import uses,
    copies every listed file byte for byte into ``<bundle>__engine_<id>``,
    re-verifies each SHA-256, and writes the source manifest plus
    ``engine_override`` and ``derived_from``. The derived bundle is self-contained
    and immutable like any other; ``production_ledger.import_replay_bundle``
    stamps the override raw and ``OTR_ShotLock``'s replay branch validates and
    applies it to the whole plan.
    """
    bundle_dir = pathlib.Path(bundle_dir).resolve()
    engine_id = str(engine_id or "").strip()
    if not engine_id:
        raise SystemExit("derive: an engine id is required")
    manifest = load_replay_manifest(str(bundle_dir))
    if str(manifest.get("engine_override") or ""):
        raise SystemExit("derive: %s already carries engine_override=%r -- derive "
                         "from the ORIGINAL bundle" % (bundle_dir, manifest["engine_override"]))
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in engine_id)
    out_root = pathlib.Path(out_root) if out_root else bundle_dir.parent
    final = out_root / ("%s__engine_%s" % (bundle_dir.name, safe))
    if final.exists():
        raise SystemExit("derived bundle already exists: %s (a bundle is immutable)" % final)
    tmp = out_root / ("." + final.name + ".building")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for row in manifest["files"]:
        rel = str(row["path"])
        src = bundle_dir / pathlib.Path(rel)
        dst = tmp / pathlib.Path(rel)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        if _sha256(dst) != str(row.get("sha256") or ""):
            shutil.rmtree(tmp, ignore_errors=True)
            raise SystemExit("derive: copied file %s does not match its manifest sha256" % rel)
    derived = dict(manifest)
    derived["engine_override"] = engine_id
    derived["derived_from"] = str(bundle_dir)
    (tmp / REPLAY_MANIFEST_NAME).write_text(json.dumps(derived, indent=2, ensure_ascii=False) + "\n",
                                            encoding="utf-8")
    load_replay_manifest(str(tmp))            # the same validator the import runs
    os.replace(tmp, final)
    return final


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episode", help="episode directory, or an episode id under the live episodes root; "
                                    "with --derive-engine, an EXISTING bundle directory")
    ap.add_argument("--out", default="", help="bundle root (default: <episodes root>/_replay)")
    ap.add_argument("--allow-no-plan", action="store_true",
                    help="freeze a ledger without planned video.shots (diagnostic only)")
    ap.add_argument("--derive-engine", default="", dest="derive_engine",
                    help="derive a sibling bundle from an existing bundle that replays on the "
                         "named engine (a registered Ghost sibling of the frozen plan's engine)")
    args = ap.parse_args(argv)
    if args.derive_engine:
        final = derive_engine_bundle(pathlib.Path(args.episode), args.derive_engine,
                                     pathlib.Path(args.out) if args.out else None)
        man = json.loads((final / REPLAY_MANIFEST_NAME).read_text(encoding="utf-8"))
        print("[derive] %s -> %s (engine_override=%s, %d file(s))"
              % (args.episode, final, man["engine_override"], len(man["files"])))
        return 0
    ep_dir = resolve_episode_dir(args.episode)
    out_root = pathlib.Path(args.out) if args.out else (_episodes_root() / "_replay")
    out_root.mkdir(parents=True, exist_ok=True)
    final = freeze(ep_dir, out_root, allow_no_plan=args.allow_no_plan)
    man = json.loads((final / REPLAY_MANIFEST_NAME).read_text(encoding="utf-8"))
    print("[freeze] %s -> %s (%d file(s), %d planned shot(s), commit %s)"
          % (ep_dir.name, final, len(man["files"]), man["planned_shots"], man["source_commit"] or "?"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
