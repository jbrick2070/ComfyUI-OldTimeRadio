"""The broadcast feed: the folder the 24/7 OBS broadcast watches. (PLAN row
4.1, planning/creatoros-concepts/PLAN.md; operator decision D1 = A, 2026-10-24:
air by default, pull from air.)

WHY THIS EXISTS. OBS used to watch ``otr/obs/`` itself, so every leg aired,
probe legs included. ``otr/obs/`` must stay the dailies and the success signal,
so the fix cannot touch it. Instead every published episode is ALSO hardlinked
into ``otr/episodes/_shared/broadcast/`` (same volume, no extra disk), OBS
watches that folder, and an episode can be pulled from air (its link removed)
or put back.

THE RULES.
  * Nothing here writes, renames or deletes anything in ``otr/obs/``. The feed
    only ever adds or removes links inside its own folder.
  * Air by default: ``sync`` links every published ``*_final.mp4`` that is not
    on the pulled list. It runs after every publish, at server boot and from
    ``scripts/otr_broadcast.py``, so a publish whose sync failed, a
    hand-deleted obs file or a re-publish (a NEW file under the same name) is
    put right on the next one.
  * A pull is an intent kept in ``otr/episodes/_shared/state/broadcast_pulled.json``.
    On Windows OBS holds the file it is playing, so removing that link can
    fail: the episode is then reported "pull pending" (never "pulled" while the
    link exists), and each later sync tries again.
  * Episodes are named by their obs FILE NAME, accepted only when it is exactly
    a ``*_final.mp4`` file listed in the obs folder. Never a path.
  * A feed that cannot hold a hardlink to obs (``OTR_OBS_DIR`` pinned on
    another volume) is reported loudly, once per sync, and nothing else
    changes. Every function here reports; none raises into a publish.
  * NO ROUTE CHANGES THE FEED. ComfyUI routes need no login, and an
    unauthenticated route with a side effect is a registry ban class (the note
    above the routes in ``__init__.py``). The sidebar tab reads
    ``GET /otr/episodes`` (``list_episodes``, read-only); pulling and putting
    back are ``scripts/otr_broadcast.py``.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, Dict, List, Optional

try:
    from .. import _otr_paths
except ImportError:  # pragma: no cover -- flat load
    import _otr_paths  # type: ignore

log = logging.getLogger("OTR.broadcast")

FEED_DIRNAME = "broadcast"
PULLED_FILENAME = "broadcast_pulled.json"
PUBLISHED_SUFFIX = "_final.mp4"
_LOCK = threading.Lock()


class BroadcastError(ValueError):
    """A request the feed refuses, in words safe to send back to a browser."""


def obs_dir() -> str:
    return str(_otr_paths.otr_obs_dir())


def feed_dir() -> str:
    return str(_otr_paths.otr_shared_root() / FEED_DIRNAME)


def _pulled_path() -> str:
    return str(_otr_paths.otr_state_dir() / PULLED_FILENAME)


def published_names() -> List[str]:
    """Every published episode in the obs folder, by file name."""
    try:
        names = os.listdir(obs_dir())
    except OSError:
        return []
    base = obs_dir()
    return sorted(n for n in names
                  if n.endswith(PUBLISHED_SUFFIX) and os.path.isfile(os.path.join(base, n)))


def _checked_name(name: Any) -> str:
    """``name`` if it is exactly a published episode's file name, else refused."""
    text = str(name or "")
    if (not text or text != os.path.basename(text) or "/" in text or "\\" in text
            or text in (".", "..") or not text.endswith(PUBLISHED_SUFFIX)):
        raise BroadcastError("not an episode name")
    if text not in published_names():
        raise BroadcastError("no published episode by that name")
    return text


def load_pulled() -> List[str]:
    try:
        with open(_pulled_path(), "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return []
    except (OSError, ValueError) as exc:
        log.warning("[OTR.broadcast] the pulled list cannot be read (%s); "
                    "treating nothing as pulled", type(exc).__name__)
        return []
    pulled = data.get("pulled") if isinstance(data, dict) else None
    return sorted({str(n) for n in pulled if isinstance(n, str)}) if isinstance(pulled, list) else []


def _save_pulled(pulled: List[str]) -> None:
    path = _pulled_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    partial = path + ".part"
    with open(partial, "w", encoding="utf-8") as fh:
        json.dump({"pulled": sorted(set(pulled))}, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    os.replace(partial, path)


def _is_link_to(link: str, target: str) -> bool:
    try:
        return os.path.samefile(link, target)
    except OSError:
        return False


def _unlink(path: str) -> Optional[str]:
    """Remove ``path``; None when it is gone, else why it is still there."""
    try:
        os.remove(path)
        return None
    except FileNotFoundError:
        return None
    except OSError as exc:
        return "in use (%s)" % type(exc).__name__


def _same_dir(a: str, b: str) -> bool:
    try:
        return os.path.samefile(a, b)
    except OSError:
        return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def sync(published_into: Optional[str] = None) -> Dict[str, Any]:
    """Make the feed match obs minus the pulled list. Never raises.

    ``published_into`` is the obs folder a publish just wrote to. The feed
    mirrors the canonical obs folder only, so a publish anywhere else (a test's
    own folder, say) is reported and leaves the feed alone."""
    report: Dict[str, Any] = {"linked": [], "relinked": [], "removed": [],
                              "pending": [], "errors": []}
    if published_into is not None and not _same_dir(published_into, obs_dir()):
        report["errors"].append(
            "this publish went to a folder that is not the canonical obs "
            "folder, which is the only one the feed mirrors")
        return report
    try:
        with _LOCK:
            _sync_locked(report)
    except Exception as exc:  # noqa: BLE001 -- reported, never raised
        report["errors"].append("%s: the feed could not be synced" % type(exc).__name__)
    if report["errors"]:
        log.warning("[OTR.broadcast] feed NOT fully synced: %s",
                    "; ".join(report["errors"][:3]))
    if report["pending"]:
        log.warning("[OTR.broadcast] pull pending, the file is in use (OBS "
                    "may be playing it); retried on the next sync: %s",
                    ", ".join(report["pending"][:5]))
    return report


def _sync_locked(report: Dict[str, Any]) -> None:
    source = obs_dir()
    feed = feed_dir()
    os.makedirs(feed, exist_ok=True)
    published = set(published_names())
    pulled = set(load_pulled())
    cross_volume_reported = False
    for name in sorted(published - pulled):
        target = os.path.join(source, name)
        link = os.path.join(feed, name)
        if os.path.exists(link):
            if _is_link_to(link, target):
                continue
            # A re-publish replaced the obs file with a NEW file of the same
            # name; the old link still holds the old picture.
            why = _unlink(link)
            if why:
                report["errors"].append("%s: the old link is %s" % (name, why))
                continue
            kind = "relinked"
        else:
            kind = "linked"
        try:
            # The REAL file (BUG-12.177): os.link on Windows links a symlink
            # itself, and a relative one would resolve nowhere from the feed.
            os.link(os.path.realpath(target), link)
            report[kind].append(name)
        except OSError as exc:
            if not cross_volume_reported:
                report["errors"].append(
                    "%s: a hardlink from the obs folder into the feed failed "
                    "(%s); the obs folder and the feed must be on one volume "
                    "(is OTR_OBS_DIR pinned elsewhere?)" % (name, type(exc).__name__))
                cross_volume_reported = True
    for name in sorted(os.listdir(feed)):
        if not name.endswith(PUBLISHED_SUFFIX):
            continue
        if name in published and name not in pulled:
            continue
        why = _unlink(os.path.join(feed, name))
        if why:
            report["pending" if name in pulled else "errors"].append(
                name if name in pulled else "%s: %s" % (name, why))
        else:
            report["removed"].append(name)


def pull(name: Any) -> Dict[str, Any]:
    """Take an episode off air. Its obs file is untouched."""
    episode = _checked_name(name)
    with _LOCK:
        pulled = load_pulled()
        if episode not in pulled:
            pulled.append(episode)
            _save_pulled(pulled)
        why = _unlink(os.path.join(feed_dir(), episode))
    if why:
        return {"episode": episode, "state": "pull pending",
                "reason": "the feed file is %s; it is removed on the next sync" % why}
    return {"episode": episode, "state": "off air"}


def put_back(name: Any) -> Dict[str, Any]:
    """Put a pulled episode back on air."""
    episode = _checked_name(name)
    with _LOCK:
        pulled = [n for n in load_pulled() if n != episode]
        _save_pulled(pulled)
    report = sync()
    on_air = os.path.exists(os.path.join(feed_dir(), episode))
    out = {"episode": episode, "state": "on air" if on_air else "not on air"}
    if not on_air and report["errors"]:
        out["reason"] = report["errors"][0]
    return out


def _title_of(name: str) -> str:
    stem = name[: -len(PUBLISHED_SUFFIX)]
    return stem.split("__", 1)[0]


def list_episodes() -> List[Dict[str, Any]]:
    """Every published episode with its air state, newest first. Read-only."""
    feed = feed_dir()
    pulled = set(load_pulled())
    rows = []
    for name in published_names():
        try:
            st = os.stat(os.path.join(obs_dir(), name))
        except OSError:
            continue
        on_air = os.path.exists(os.path.join(feed, name))
        state = ("on air" if on_air and name not in pulled
                 else "pull pending" if on_air else "off air")
        rows.append({"episode": name, "title": _title_of(name), "bytes": st.st_size,
                     "published_at": int(st.st_mtime), "state": state,
                     "on_air": state == "on air"})
    rows.sort(key=lambda r: (r["published_at"], r["episode"]), reverse=True)
    return rows


def act(episode: Any, action: Any) -> Dict[str, Any]:
    """One action by name, for ``scripts/otr_broadcast.py``."""
    if action == "pull":
        return pull(episode)
    if action == "put_back":
        return put_back(episode)
    if action == "sync":
        report = sync()
        return {"state": "synced", **{k: len(v) for k, v in report.items()}}
    raise BroadcastError("action must be pull, put_back or sync")


__all__ = ["FEED_DIRNAME", "BroadcastError", "feed_dir", "sync", "pull", "put_back",
           "list_episodes", "act", "published_names", "load_pulled"]
