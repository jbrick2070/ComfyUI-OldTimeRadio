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

WHAT THE KEY DOES NOT SEE, honestly: an engine setting read from the
environment at render time other than those the render driver hands
:func:`clip_reuse_key`, and a model file an engine does not declare
(``model_artifacts``). Swapping either under the same engine id and version
calls for a new project.

Standard library only. UTF-8, no BOM.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import threading
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Optional

try:
    from . import _otr_producer_direction as _PD
except ImportError:  # pragma: no cover -- flat (sys.path) load
    import _otr_producer_direction as _PD  # type: ignore

log = logging.getLogger("OTR.producer_clips")

#: Bumped whenever what the key hashes changes, so a store entry is never found
#: by a key that has come to mean something else.
CLIP_KEY_VERSION = 1
RECORD_VERSION = 1
RECORD_NAME = "record.json"
#: The video section's map from shot id to the record of the clip this
#: workspace delivered for it. Its paths are relative to the episode folder.
MAP_KEY = "producer_clip_reuse"
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
_KEY_RE = re.compile(r"^[0-9a-f]{64}$")
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$")

__all__ = ["CLIP_KEY_VERSION", "MAP_KEY", "ClipStoreError", "ProducerClips",
           "clip_reuse_key", "commit", "file_sha256", "lookup", "request_projection",
           "store_dir"]


class ClipStoreError(RuntimeError):
    """A store entry or bundle record that cannot be trusted, and why."""


# --------------------------------------------------------------------------- #
# the key
# --------------------------------------------------------------------------- #

_DIGESTS: Dict[tuple, str] = {}
_DIGEST_LOCK = threading.Lock()


def file_sha256(path: Any) -> Optional[str]:
    """The sha256 of a file, computed once per process for each path, size and
    modification time, or None when it cannot be read."""
    try:
        st = os.stat(str(path))
    except OSError:
        return None
    memo = (os.path.abspath(str(path)), st.st_size, st.st_mtime_ns)
    with _DIGEST_LOCK:
        hit = _DIGESTS.get(memo)
    if hit:
        return hit
    digest = hashlib.sha256()
    try:
        with open(str(path), "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
    except OSError:
        return None
    value = digest.hexdigest()
    with _DIGEST_LOCK:
        _DIGESTS[memo] = value
    return value


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
    if not isinstance(files, dict) or "clip" not in files \
            or not set(files) <= set(_STORE_NAMES):
        raise ClipStoreError("%s lists no clip file" % where)
    for role, item in files.items():
        if (not isinstance(item, dict) or not _NAME_RE.fullmatch(str(item.get("name") or ""))
                or not isinstance(item.get("bytes"), int) or isinstance(item.get("bytes"), bool)
                or not _KEY_RE.fullmatch(str(item.get("sha256") or ""))):
            raise ClipStoreError("%s lists its %s file badly" % (where, role))
    return record


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
            os.replace(str(staging), str(entry))
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
    direction is on the wire; the render driver calls :meth:`find` before a
    shot's weights load and :meth:`commit` once the shot is whole. Safe to
    call from the cloud fan-out's worker threads."""

    def __init__(self, *, project_id: str, episode_root: Any, bundle_map: Any) -> None:
        self.store = store_dir(project_id)
        self.episode_root = Path(str(episode_root))
        self.clips_dir = self.episode_root.joinpath(*_EPISODE_DIRS["clip"])
        self.foley_dir = self.episode_root.joinpath(*_EPISODE_DIRS["foley"])
        self._bundle = {str(sid): record for sid, record in dict(bundle_map or {}).items()
                        if isinstance(record, dict)}
        self._lock = threading.Lock()
        self._keys: Dict[str, str] = {}
        self._records: Dict[str, Dict[str, Any]] = {}
        #: Shot id -> where its clip came from this run: "bundle" or "store".
        self.reused: Dict[str, str] = {}

    # -- finding ---------------------------------------------------------- #
    def find(self, shot_id: str, key: str) -> Optional[Dict[str, Any]]:
        """The clip filmed under ``key`` for this shot, in this workspace, as
        ``{"clip": ..., "attempts": [...]}`` -- or None, and the shot renders.
        The bundle's own record is used in place when it names this key and
        its files verify; otherwise the store's entry is copied in."""
        shot_id = str(shot_id)
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

    def _in_bundle(self, record: Dict[str, Any]) -> Dict[str, Path]:
        """A bundle record's files where they sit in this workspace, verified."""
        _check_record(record, "the bundle's record")
        paths = {}
        for role, item in record["files"].items():
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
                shutil.copyfile(str(self.store / key / _STORE_NAMES[role]), str(partial))
                os.replace(str(partial), str(target))
                _verify(target, item)
            paths[role] = target
        return paths

    def _take(self, shot_id: str, record: Dict[str, Any], paths: Dict[str, Path],
              source: str) -> Dict[str, Any]:
        clip = dict(record["clip"])
        for role, path in paths.items():
            clip[_PATH_FIELDS[role]] = str(path)
        with self._lock:
            self.reused[shot_id] = source
            self._records[shot_id] = self._map_record(record)
        return {"clip": clip, "attempts": list(record["attempts"])}

    # -- committing ------------------------------------------------------- #
    def commit(self, shot_id: str, clip: Dict[str, Any], attempts: list) -> None:
        """Keep a shot this run filmed: its clip, already in the episode's
        clips folder, and its foley stem go into the store, and the record into
        this run's map. A store that cannot take it is reported, not fatal: the
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
             "clip": {k: v for k, v in clip.items() if k not in _PATH_FIELDS.values()}},
            ensure_ascii=True, default=str))
        try:
            record = commit(self.store, key, body, files)
        except (OSError, ClipStoreError) as exc:
            log.error("[OTR.producer_clips] shot %s was filmed but not committed to the "
                      "store (%s); this stage's bundle still carries it", shot_id, exc)
            try:
                record = dict(body, version=RECORD_VERSION, key=key, files={
                    role: _file_entry(Path(path), os.path.basename(path))
                    for role, path in files.items()})
            except ClipStoreError as unread:
                log.error("[OTR.producer_clips] shot %s keeps no reuse record: %s",
                          shot_id, unread)
                return
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
