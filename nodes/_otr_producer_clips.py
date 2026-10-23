"""The Scene Producer's clip reuse: the causal key, the project's clip store and
the reuse records (planning/scene-producer/PLAN.md, "Video" and "Clips survive
a crash").

A FILMED SHOT IS FOUND BY WHAT MADE IT. Before a producer Video run renders a
shot, the render driver hashes everything that decides the shot's clip -- its
prebuilt segment requests, with every file they name replaced by that file's
hash, its coverage plan, its engine and render policy, and the settings outside
the request that change bytes -- into a versioned key (:func:`clip_reuse_key`).
A clip already filmed under that key is used instead of rendering it again:
the bundle's own record of the shot first (``video.producer_clip_reuse``), then
the project's store (``_replay/p/<project>/clips/<key>/``). Every file is
verified by size and sha256 on use.

A SHOT IS COMMITTED THE MOMENT IT IS WHOLE. Its clip goes into the episode's
``clips/`` folder and the store before the next shot starts, so a run that dies
at shot 80 of 100 keeps the 79 it finished, and the next run films only the
rest. A store entry is built in a staging folder, its record written last, and
the folder renamed into place: an entry named for its key is whole or absent.
A committed entry is never changed; one that fails verification is moved
aside, not overwritten.

THE ENVIRONMENT IS IN THE KEY. Engines read their sampler settings -- steps,
guidance, a negative prompt, the folder a checkpoint is in -- from ``OTR_``
settings at render time, outside the request, so every one of them enters the
key (:func:`causal_env`) except those that cannot change a clip: the ones the
pack itself writes while a prompt runs, and those that decide whether, when,
how often or how fast a render runs, what it may spend, where its scratch and
logs go or where credentials are kept. A spend cap raised after a halt films
nothing again.

WHAT THE KEY DOES NOT SEE, honestly: a setting outside the pack's ``OTR_``
names, and a model file an engine does not declare (``model_artifacts``).
Swapping either under the same engine id and version calls for a new
project.

Standard library only. UTF-8, no BOM.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import os
import re
import shutil
import threading
import time
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Optional

try:
    from . import _otr_producer_direction as _PD
    from ._otr_shared import env as otr_env
except ImportError:  # pragma: no cover -- flat (sys.path) load
    import _otr_producer_direction as _PD  # type: ignore
    from _otr_shared import env as otr_env  # type: ignore

log = logging.getLogger("OTR.producer_clips")

#: Bumped whenever what the key hashes changes, so a store entry is never found
#: by a key that has come to mean something else.
CLIP_KEY_VERSION = 2
RECORD_VERSION = 1
RECORD_NAME = "record.json"
#: The video section's map from shot id to the record of the clip this
#: workspace delivered for it, and the map from shot id to the payload of the
#: floor a shot fell to. Their paths are relative to the episode folder. The
#: names belong to the shared direction module, which cards.json reads them by.
MAP_KEY = _PD.CLIP_MAP_KEY
FLOOR_KEY = _PD.FLOOR_MAP_KEY
#: The store's folder inside a project, beside its bundles; no bundle name can
#: take it (``_otr_producer_direction.OUT_RE``).
STORE_DIR = "clips"
#: The files a record can hold, and the names they have inside a store entry.
_STORE_NAMES = {"clip": "clip.mp4", "foley": "foley.wav"}
#: Where each file lives in an episode folder.
_EPISODE_DIRS = {"clip": ("clips",), "foley": ("audio", "foley")}
#: The clip fields that name a file. A record keeps the files, not the paths.
_PATH_FIELDS = {"clip": "path", "foley": "foley_path"}
#: Request fields that never reach the pixels: the request's own name, the
#: trace stamps, and the folder an engine writes its plate into.
_NOT_CAUSAL = frozenset({"request_id", "observability", "plate_path"})
#: ``OTR_`` settings that never change a clip's bytes. The pack writes the
#: first four itself while a prompt runs -- the workflow validator's two
#: exports, the output folder, the writer's style-grammar switch -- so they can
#: differ between two runs of one project; the last two say where credentials
#: are kept, and rotating a key must not film a project again.
_ENV_NOT_CAUSAL = frozenset({
    "OTR_ACTIVE_PROFILE", "OTR_SNAPSHOT_HASH", "OTR_OUTPUT_DIR", "OTR_ENABLE_STYLE_GRAMMAR",
    "OTR_ALLOW_KEY_FILES", "OTR_API_KEY_PACK_ROOT"})
#: The shapes of ``OTR_`` settings that never change a clip's bytes: where
#: credentials are kept and which endpoint is called; how long to wait and how
#: often to retry; how many calls run at once; what a run may spend, in money
#: or memory; and where scratch, caches and logs go. Folders and roots that
#: choose model files are not among them: they stay in the key.
_ENV_NOT_CAUSAL_RE = re.compile(
    r"(_API_KEY|_KEY_FILES?|_TOKEN_FILE|_SECRETS_FILE|_API_BASE|_URL|_HOST|_PORT"
    r"|_TIMEOUT|_TIMEOUT_S|_RETRIES|_FANOUT|_WORKERS"
    r"|_BUDGET_USD|_BUDGET_MARGIN|_EST_USD|_EST_USD_PER_S|_HEADROOM_MB"
    r"|_CACHE|_CACHE_DIR|_TMP|_LOG_PATH|_STDERR_DIR|_MAX_AGE_S|_STRICT)$"
    r"|_MAX_CONCURRENCY|_COST_|_SWEEP_")
_KEY_RE = re.compile(r"^[0-9a-f]{64}$")
#: How long a commit waits, in turn, before it renames a held staging folder
#: into place again: on Windows a scanner opening a file it has just seen
#: appear holds the folder for a moment.
_RENAME_WAITS_S = (0.05, 0.1, 0.2, 0.4, 0.8)
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$")

__all__ = ["CLIP_KEY_VERSION", "FLOOR_KEY", "MAP_KEY", "ClipStoreError", "ProducerClips",
           "causal_env", "clip_reuse_key", "commit", "file_sha256", "lookup",
           "request_projection", "store_dir"]


class ClipStoreError(RuntimeError):
    """A store entry or bundle record that cannot be trusted, and why."""


# --------------------------------------------------------------------------- #
# the key
# --------------------------------------------------------------------------- #

def file_sha256(path: Any) -> Optional[str]:
    """The sha256 of a file's bytes, read in full on every call, or None when it
    cannot be read. Never remembered: a file replaced in place by another of
    the same size, its modification time put back, must not pass as the file
    it replaced."""
    digest = hashlib.sha256()
    try:
        with open(str(path), "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def causal_env() -> Dict[str, str]:
    """The pack's ``OTR_`` settings in the environment, as the key hashes them.

    Engines read their own at render time, outside the request -- HuMo its
    steps, guidance and negative prompt, LTX its sampler and schedule -- so all
    of them enter the key, and a setting added tomorrow is in it the day it
    lands. Left out: the names in :data:`_ENV_NOT_CAUSAL` and those of a shape
    :data:`_ENV_NOT_CAUSAL_RE` matches, none of which changes what a clip
    shows."""
    return {name: value for name, value in sorted(otr_env.snapshot().items())
            if name.startswith("OTR_") and name not in _ENV_NOT_CAUSAL
            and not _ENV_NOT_CAUSAL_RE.search(name)}


def _as_file(value: Any) -> Any:
    """A request value as what it holds. A path names a file, so it enters the
    key as that file's sha256 -- never as a workspace path, which changes on
    every replay. Anything else is kept as it is."""
    if not isinstance(value, str) or not value or not os.path.isabs(value):
        return value
    digest = file_sha256(value) if os.path.isfile(value) else None
    return {"sha256": digest} if digest else {"absent": os.path.basename(value)}


def request_projection(request: Any) -> Dict[str, Any]:
    """What one prebuilt request asks its engine for, as the key hashes it: the
    whole request but its id, its trace stamps and its plate folder, with each
    file it names -- init image, references, audio, base clip -- as the file's
    hash."""
    if hasattr(request, "model_dump"):
        req = request.model_dump()
    elif isinstance(request, dict):
        req = dict(request)
    else:
        raise TypeError("a render request is a dict, got %s" % type(request).__name__)
    out: Dict[str, Any] = {}
    for name in sorted(req):
        value = req[name]
        if name in _NOT_CAUSAL:
            continue
        if name in ("asset_refs", "conditioning_refs") and isinstance(value, dict):
            value = {str(k): _as_file(v) for k, v in sorted(value.items())}
        elif name == "audio_ref" and isinstance(value, dict):
            value = dict(value, path=_as_file(value.get("path")))
        elif name == "base_clip_ref":
            value = _as_file(value)
        out[name] = value
    return out


def clip_reuse_key(*, shot_id: str, requests: Iterable[Any], coverage_plan: Any,
                   engine: Dict[str, Any], render_policy: Dict[str, Any],
                   settings: Dict[str, Any]) -> str:
    """The versioned causal key of one shot's clip.

    ``requests`` are the shot's prebuilt segment requests in play order -- one
    for a single-clip shot -- taken before any of them renders. A CHAIN
    successor's init image is its predecessor's last frame, which does not
    exist until that segment renders, so the chain enters the key as the
    coverage plan's join and the predecessor's own request, never as a frame
    hash. ``engine`` is the engine's identity (id, versions, recipe, declared
    model files), ``render_policy`` the episode's device and precision policy,
    ``settings`` the environment settings that change bytes. Workspace paths,
    run ids and trace stamps never enter it, so the same shot keys the same in
    a fresh replay workspace."""
    body = {"v": CLIP_KEY_VERSION, "shot_id": str(shot_id),
            "requests": [request_projection(r) for r in requests],
            "coverage_plan": coverage_plan, "engine": engine,
            "render_policy": render_policy, "settings": settings}
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=True,
                                     separators=(",", ":"), default=str)
                          .encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# the store
# --------------------------------------------------------------------------- #

def store_dir(project_id: str) -> Path:
    """The project's clip store, beside its bundles and confined like them."""
    return _PD.project_dir(project_id) / STORE_DIR


def _file_entry(path: Path, name: str) -> Dict[str, Any]:
    digest = file_sha256(path)
    if not digest:
        raise ClipStoreError("%s cannot be read" % path)
    return {"name": name, "bytes": path.stat().st_size, "sha256": digest}


def _check_record(record: Any, where: str) -> Dict[str, Any]:
    """A record's shape, refused before any of its names reaches a path."""
    if not isinstance(record, dict) or record.get("version") != RECORD_VERSION:
        raise ClipStoreError("%s is not a version-%d record" % (where, RECORD_VERSION))
    if not _KEY_RE.fullmatch(str(record.get("key") or "")):
        raise ClipStoreError("%s names no valid key" % where)
    if not isinstance(record.get("clip"), dict) or not isinstance(record.get("attempts"), list):
        raise ClipStoreError("%s carries no clip" % where)
    files = record.get("files")
    if not isinstance(files, dict) or "clip" not in files:
        raise ClipStoreError("%s lists no clip file" % where)
    _check_files(files, where)
    if set(record["clip"]) & set(_PATH_FIELDS.values()):
        raise ClipStoreError("%s names a path in its clip: a record keeps files, not paths"
                             % where)
    _check_clip_shape(record["clip"], where)
    # The mux finds a beat's foley bed by its stem: a receipt for a stem the
    # record does not list would drop the bed without a word.
    foley_sha = record["clip"].get("foley_sha256")
    if foley_sha and (files.get("foley") or {}).get("sha256") != foley_sha:
        raise ClipStoreError("%s carries a foley receipt for a stem it does not list" % where)
    return record


def _check_floor(record: Any, where: str) -> Dict[str, Any]:
    """A floor payload's shape: its marks, and the floor clip when one was
    drawn -- a black floor has none."""
    if not isinstance(record, dict) or record.get("version") != RECORD_VERSION:
        raise ClipStoreError("%s is not a version-%d floor" % (where, RECORD_VERSION))
    if not isinstance(record.get("marks"), dict) or not isinstance(record.get("files"), dict):
        raise ClipStoreError("%s carries no floor marks" % where)
    clip = record.get("clip")
    if record["files"] and not isinstance(clip, dict):
        raise ClipStoreError("%s lists a file but no clip" % where)
    if isinstance(clip, dict) and set(clip) & set(_PATH_FIELDS.values()):
        raise ClipStoreError("%s names a path in its clip: a floor keeps files, not paths"
                             % where)
    if isinstance(clip, dict):
        _check_clip_shape(clip, where)
    _check_files(record["files"], where)
    return record


#: A clip body's numbers, read as numbers wherever the clip is used.
_CLIP_NUMBERS = ("frame_count", "target_frame_count", "fps", "width", "height",
                 "duration_s", "vram_peak_mb")


def _is_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return not isinstance(value, float) or math.isfinite(value)


def _check_clip_shape(clip: Dict[str, Any], where: str) -> None:
    """A clip body's receipts and segments, of the shapes a render gives
    them, and its numbers numbers, or a refusal -- so a corrupt record is
    passed over for the store, never a crash of the run that reads it."""
    if ((clip.get("receipt") is not None and not isinstance(clip.get("receipt"), dict))
            or (clip.get("receipts") is not None and not isinstance(clip.get("receipts"), list))
            or (clip.get("segments") is not None
                and not isinstance(clip.get("segments"), list))):
        raise ClipStoreError("%s carries a clip whose receipts or segments are not "
                             "receipts or segments" % where)
    bad = [k for k in _CLIP_NUMBERS if clip.get(k) is not None and not _is_number(clip[k])]
    if bad:
        raise ClipStoreError("%s carries a clip whose %s is not a number"
                             % (where, ", ".join(bad)))


def _check_files(files: Dict[str, Any], where: str) -> None:
    if not set(files) <= set(_STORE_NAMES):
        raise ClipStoreError("%s lists a file it cannot hold" % where)
    for role, item in files.items():
        if (not isinstance(item, dict) or not _NAME_RE.fullmatch(str(item.get("name") or ""))
                or not isinstance(item.get("bytes"), int) or isinstance(item.get("bytes"), bool)
                or not _KEY_RE.fullmatch(str(item.get("sha256") or ""))):
            raise ClipStoreError("%s lists its %s file badly" % (where, role))


def _verify(path: Path, item: Dict[str, Any]) -> None:
    try:
        size = path.stat().st_size
    except OSError:
        raise ClipStoreError("%s is missing" % path) from None
    if size != item["bytes"] or file_sha256(path) != item["sha256"]:
        raise ClipStoreError("%s is not the file its record names" % path)


def lookup(store: Path, key: str) -> Optional[Dict[str, Any]]:
    """The verified record of the store's entry for ``key``, or None when it has
    none. A staging folder is never an entry; an entry whose record or files do
    not verify is refused, never taken on its folder's word."""
    if not _KEY_RE.fullmatch(str(key or "")):
        raise ClipStoreError("not a clip key: %r" % (key,))
    entry = Path(store) / key
    if not entry.is_dir():
        return None
    try:
        record = json.loads((entry / RECORD_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError) as exc:
        raise ClipStoreError("store entry %s has no readable record: %s"
                             % (key[:12], exc)) from None
    _check_record(record, "store entry %s" % key[:12])
    if record["key"] != key:
        raise ClipStoreError("store entry %s records key %s" % (key[:12], record["key"][:12]))
    for role, item in record["files"].items():
        _verify(entry / _STORE_NAMES[role], item)
    return record


def _replace_held(src: Path, dst: Path, settled: Any = None) -> None:
    """``os.replace``, again while a scanner holds the file or folder for a
    moment. ``settled()``, when given, says a refusal needs no wait: another
    writer has put the same thing there."""
    for wait in _RENAME_WAITS_S + (None,):
        try:
            os.replace(str(src), str(dst))
            return
        except PermissionError:
            if wait is None or (settled is not None and settled()):
                raise
            time.sleep(wait)


def _rename_into_place(staging: Path, entry: Path) -> None:
    """Rename a staged entry onto its key, again while the folder is held. A
    refusal with the entry already there is another writer's commit, which
    the caller settles; that one is not waited out."""
    _replace_held(staging, entry, settled=entry.exists)


def commit(store: Path, key: str, body: Dict[str, Any],
           files: Dict[str, str]) -> Dict[str, Any]:
    """Commit one filmed shot to the store and return its record.

    ``files`` maps ``clip`` (and ``foley`` when the shot has a stem) to the
    file in the episode. They are copied into a unique staging folder beside
    the entry and hashed, the record goes in last, and the folder is renamed
    to ``key``. An entry already there is verified and kept -- the same key
    is the same clip, so a second writer has nothing to add -- and one that
    fails verification is moved aside, never written over."""
    if not _KEY_RE.fullmatch(str(key or "")):
        raise ClipStoreError("not a clip key: %r" % (key,))
    store = Path(store)
    entry = store / key
    try:
        existing = lookup(store, key)
    except ClipStoreError as exc:
        aside = store / (".%s.refused-%s" % (key, uuid.uuid4().hex[:8]))
        log.error("[OTR.producer_clips] store entry %s refused (%s); moved aside to %s "
                  "and committed again", key[:12], exc, aside.name)
        os.replace(str(entry), str(aside))
        existing = None
    if existing is not None:
        return existing
    store.mkdir(parents=True, exist_ok=True)
    staging = store / (".%s.staging-%s" % (key, uuid.uuid4().hex[:8]))
    staging.mkdir()
    try:
        entries = {}
        for role, src in files.items():
            name = os.path.basename(str(src))
            target = staging / _STORE_NAMES[role]
            shutil.copyfile(str(src), str(target))
            entries[role] = _file_entry(target, name)
        record = _check_record(dict(body, version=RECORD_VERSION, key=key, files=entries),
                               "the record for %s" % key[:12])
        (staging / RECORD_NAME).write_text(
            json.dumps(record, ensure_ascii=True, sort_keys=True, indent=1), encoding="utf-8")
        try:
            _rename_into_place(staging, entry)
        except OSError:
            # Another writer committed the same key first: the same clip.
            settled = lookup(store, key)
            if settled is None:
                raise
            shutil.rmtree(str(staging), ignore_errors=True)
            return settled
    except BaseException:
        shutil.rmtree(str(staging), ignore_errors=True)
        raise
    return record


# --------------------------------------------------------------------------- #
# one run's reuse
# --------------------------------------------------------------------------- #

class ProducerClips:
    """One producer Video run's clip reuse: what it found, what it filmed, and
    the map its bundle carries. Built by OTR_VideoRenderBatch only when a
    direction is on the wire; the render driver takes every kept card with
    :meth:`take_kept` before any shot is filmed, calls :meth:`find` for each
    other shot before its weights load, and :meth:`commit` once a shot is
    whole. Safe to call from the cloud fan-out's worker threads."""

    def __init__(self, *, project_id: str, episode_root: Any, bundle_map: Any,
                 bundle_floors: Any = None, kept: Iterable[Any] = (),
                 takes: Any = None, stage: str = "video") -> None:
        #: "video" films what it does not find; "publish" films nothing, and
        #: refuses a shot it finds no clip for (plan, "Publish").
        self.stage = str(stage)
        self.store = store_dir(project_id)
        self.episode_root = Path(str(episode_root))
        self.clips_dir = self.episode_root.joinpath(*_EPISODE_DIRS["clip"])
        self.foley_dir = self.episode_root.joinpath(*_EPISODE_DIRS["foley"])
        self._bundle = {str(sid): record for sid, record in dict(bundle_map or {}).items()
                        if isinstance(record, dict)}
        self._bundle_floors = {str(sid): record
                               for sid, record in dict(bundle_floors or {}).items()
                               if isinstance(record, dict)}
        #: The cards whose clip -- or floor -- is kept as it is (kept.clips).
        self.kept = frozenset(str(sid) for sid in kept or ())
        #: The clip take each shot is filmed with (takes.clips); 0 when absent.
        self.takes = {str(sid): int(take or 0) for sid, take in dict(takes or {}).items()}
        self._lock = threading.Lock()
        self._keys: Dict[str, str] = {}
        self._records: Dict[str, Dict[str, Any]] = {}
        self._floors: Dict[str, Dict[str, Any]] = {}
        #: Shot id -> where its clip came from this run: "bundle", "store",
        #: "kept" or "kept floor".
        self.reused: Dict[str, str] = {}

    # -- finding ---------------------------------------------------------- #
    def find(self, shot_id: str, key: str) -> Optional[Dict[str, Any]]:
        """The clip filmed under ``key`` for this shot, in this workspace, as
        ``{"clip": ..., "attempts": [...]}`` -- or None, and the shot renders.
        The bundle's own record is used in place when it names this key and
        its files verify; otherwise the store's entry is copied in. A PUBLISH
        run renders nothing: a shot it finds no clip for refuses the run,
        before any weights load (plan, "Publish"). A kept card is never looked
        up by key: :meth:`take_kept` takes it."""
        found = self._find(str(shot_id), key)
        if found is None and self.stage == "publish":
            raise ClipStoreError(
                "Publish has no filmed clip for shot %s as it stands now: film it in a "
                "Video run first, or keep the clip it has" % shot_id)
        return found

    def check_gap(self, shot_id: str) -> None:
        """A shot skipped as a refused-still gap refuses a Publish run. A kept
        card never gets here: the render takes every keep before any gap."""
        if self.stage == "publish":
            raise ClipStoreError(
                "Publish refuses shot %s: its still was refused, so it has no clip; "
                "paint it again, or keep the card to publish the gap" % shot_id)

    def take_kept(self, shot_id: str) -> Dict[str, Any]:
        """A KEPT card's clip, or the floor it fell to, taken by its shot id
        alone: a keep is used whatever has changed since, so no key is asked
        for and none is computed (:meth:`_keep`)."""
        return self._keep(str(shot_id))

    def _find(self, shot_id: str, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            self._keys[shot_id] = key
        record = self._bundle.get(shot_id)
        if record is not None and record.get("key") == key:
            try:
                return self._take(shot_id, record, self._in_bundle(record), "bundle")
            except ClipStoreError as exc:
                log.warning("[OTR.producer_clips] shot %s: the bundle's clip does not "
                            "verify (%s); trying the store", shot_id, exc)
        try:
            entry = lookup(self.store, key)
        except ClipStoreError as exc:
            log.error("[OTR.producer_clips] shot %s: %s; filming it again", shot_id, exc)
            return None
        if entry is None:
            return None
        try:
            return self._take(shot_id, entry, self._from_store(key, entry), "store")
        except (ClipStoreError, OSError) as exc:
            log.error("[OTR.producer_clips] shot %s: the stored clip could not be copied "
                      "in (%s); filming it again", shot_id, exc)
            return None

    def _keep(self, shot_id: str) -> Dict[str, Any]:
        """A KEPT card's clip, used as it is whatever has changed since: the
        clip the bundle's record names -- or the floor the shot fell to, which
        stays visibly a floor, its marks returned with it. Keep never vouches
        for a missing or tampered file: either refuses the run."""
        record = self._bundle.get(shot_id)
        if record is not None:
            try:
                return self._take(shot_id, record, self._in_bundle(record), "kept")
            except ClipStoreError as exc:
                raise ClipStoreError("card %s keeps its clip, but %s" % (shot_id, exc)) from None
        floor = self._bundle_floors.get(shot_id)
        if floor is None:
            raise ClipStoreError("card %s keeps a clip it does not have" % shot_id)
        try:
            _check_floor(floor, "the bundle's floor for %s" % shot_id)
            paths = self._bundle_paths(floor["files"])
        except ClipStoreError as exc:
            raise ClipStoreError("card %s keeps its floor, but %s" % (shot_id, exc)) from None
        clip = copy.deepcopy(floor["clip"]) if isinstance(floor.get("clip"), dict) else {}
        for role, path in paths.items():
            clip[_PATH_FIELDS[role]] = str(path)
        with self._lock:
            self.reused[shot_id] = "kept floor"
            self._floors[shot_id] = floor
        return {"clip": clip, "attempts": [], "marks": dict(floor["marks"])}

    def _in_bundle(self, record: Dict[str, Any]) -> Dict[str, Path]:
        """A bundle record's files where they sit in this workspace, verified."""
        _check_record(record, "the bundle's record")
        return self._bundle_paths(record["files"])

    def _bundle_paths(self, files: Dict[str, Any]) -> Dict[str, Path]:
        """Bundle-map files where they sit in this workspace, verified: each in
        its own episode folder, under its own name, never anywhere else."""
        paths = {}
        for role, item in files.items():
            rel = PurePosixPath(str(item.get("path") or ""))
            if (rel.is_absolute() or ".." in rel.parts
                    or tuple(rel.parts[:-1]) != _EPISODE_DIRS[role]
                    or rel.name != item["name"]):
                raise ClipStoreError("its %s path %r is not an episode %s file"
                                     % (role, str(rel), role))
            path = self.episode_root.joinpath(*rel.parts)
            _verify(path, item)
            paths[role] = path
        return paths

    def _from_store(self, key: str, entry: Dict[str, Any]) -> Dict[str, Path]:
        """Copy a verified store entry's files into this workspace."""
        paths = {}
        for role, item in entry["files"].items():
            folder = self.episode_root.joinpath(*_EPISODE_DIRS[role])
            folder.mkdir(parents=True, exist_ok=True)
            target = folder / item["name"]
            if not (target.is_file() and target.stat().st_size == item["bytes"]
                    and file_sha256(target) == item["sha256"]):
                partial = folder / (".%s.%s.part" % (item["name"], uuid.uuid4().hex[:8]))
                try:
                    shutil.copyfile(str(self.store / key / _STORE_NAMES[role]), str(partial))
                    _replace_held(partial, target)
                finally:
                    if partial.exists():
                        partial.unlink()
                _verify(target, item)
            paths[role] = target
        return paths

    def _take(self, shot_id: str, record: Dict[str, Any], paths: Dict[str, Path],
              source: str) -> Dict[str, Any]:
        # A copy all the way down: the run marks the receipts on the clip it
        # hands back, and the record rides on into the next bundle as filmed.
        clip = copy.deepcopy(record["clip"])
        for role, path in paths.items():
            clip[_PATH_FIELDS[role]] = str(path)
        with self._lock:
            self.reused[shot_id] = source
            self._records[shot_id] = self._map_record(record)
        return {"clip": clip, "attempts": list(record["attempts"])}

    # -- committing ------------------------------------------------------- #
    def commit(self, shot_id: str, clip: Dict[str, Any], attempts: list) -> None:
        """Keep a shot this run filmed: its record goes into this run's map,
        and its clip and foley stem into the store. The record is made from
        the files as they sit in the episode folder and checked as any record
        is, so the map describes the episode's own files -- the bundle frozen
        with them verifies on its own, whatever the store already holds under
        the key. A store that cannot take the shot is reported, not fatal: the
        clip is in the episode, and the stage's bundle carries it."""
        shot_id = str(shot_id)
        with self._lock:
            key = self._keys.get(shot_id)
        if not key:
            return
        files = {role: str(clip.get(field) or "") for role, field in _PATH_FIELDS.items()
                 if clip.get(field)}
        # JSON-native from here on, so the record on disk, in the map and in
        # the next run's ledger are the same record.
        body = json.loads(json.dumps(
            {"shot_id": shot_id, "attempts": [str(a) for a in attempts or ()],
             "clip_take": self.takes.get(shot_id, 0),
             "clip": {k: v for k, v in clip.items() if k not in _PATH_FIELDS.values()}},
            ensure_ascii=True, default=str))
        try:
            record = _check_record(dict(body, version=RECORD_VERSION, key=key, files={
                role: _file_entry(Path(path), os.path.basename(path))
                for role, path in files.items()}), "shot %s's record" % shot_id)
        except ClipStoreError as exc:
            log.error("[OTR.producer_clips] shot %s keeps no reuse record: %s", shot_id, exc)
            return
        try:
            commit(self.store, key, body, files)
        except (OSError, ClipStoreError) as exc:
            log.error("[OTR.producer_clips] shot %s was filmed but not committed to the "
                      "store (%s); this stage's bundle still carries it", shot_id, exc)
        with self._lock:
            self._records[shot_id] = self._map_record(record)

    def _map_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """A record as the bundle's map keeps it: each file with its place in
        the episode folder beside its name, size and hash."""
        files = {role: dict(item, path="/".join(_EPISODE_DIRS[role] + (item["name"],)))
                 for role, item in record["files"].items()}
        return dict(record, files=files)

    def records_for(self, shot_ids: Iterable[Any]) -> Dict[str, Dict[str, Any]]:
        """The map for the shots this run delivered a clip for, in plan order."""
        with self._lock:
            return {str(sid): self._records[str(sid)] for sid in shot_ids
                    if str(sid) in self._records}

    # -- floors ----------------------------------------------------------- #
    def floored(self, shot_id: str, marks: Dict[str, Any], clip: Any) -> None:
        """Record a shot this run floored (plan, "Floors are failed cards"):
        its floor marks, and the floor clip when one was drawn -- already in
        the episode's clips folder. A floor is never a reuse record, so the
        next Video run films the shot again unless its card is kept."""
        shot_id = str(shot_id)
        files: Dict[str, Any] = {}
        body_clip = None
        path = str(clip.get("path") or "") if isinstance(clip, dict) else ""
        if path and os.path.isfile(path):
            try:
                entry = _file_entry(Path(path), os.path.basename(path))
                files = {"clip": dict(entry, path="/".join(
                    _EPISODE_DIRS["clip"] + (entry["name"],)))}
                body_clip = {k: v for k, v in clip.items() if k not in _PATH_FIELDS.values()}
            except ClipStoreError as exc:
                log.error("[OTR.producer_clips] shot %s floored, and its floor clip cannot "
                          "be read (%s); it is recorded as a floor with no clip", shot_id, exc)
        record = json.loads(json.dumps(
            {"version": RECORD_VERSION, "shot_id": shot_id, "marks": dict(marks or {}),
             "clip": body_clip, "files": files}, ensure_ascii=True, default=str))
        with self._lock:
            self._floors[shot_id] = record
            self._records.pop(shot_id, None)

    def floors_for(self, shot_ids: Iterable[Any]) -> Dict[str, Dict[str, Any]]:
        """The floor payloads of the shots this run floored, or kept floored,
        in plan order."""
        with self._lock:
            return {str(sid): self._floors[str(sid)] for sid in shot_ids
                    if str(sid) in self._floors}
