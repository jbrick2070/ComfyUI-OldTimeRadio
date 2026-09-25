"""Asset cleanup after publish -- the writer's `asset_cleanup` choice, carried
out by the terminal mux (GO_FORWARD_PLAN row 0b, decided 2026-09-25).

Three settings, chosen per run on OTR_LedgerScriptWriter, default off:

* ``off``     -- keep everything (nothing is stamped on the ledger).
* ``partial`` -- delete the sound and the pictures, keep every text file.
* ``full``    -- delete the whole episode folder.

All three leave ``otr/obs`` alone: the published copy is the deliverable, and
this module refuses to run at all unless that copy is proven to exist OUTSIDE
the folder it is about to clean.

WHY EVERY GUARD HERE EXISTS. The previous space-saver (``perfect_run_spacesaver``,
2026-05-02) wiped the WRONG episode on its first day (BUG-LOCAL-014, commit
``d2c2df81``) because it found its ledger by an mtime walk -- and that walker is
still live as the ledger singleton's last-resort fallback. So the folder comes
only from the in-flight singleton, and the ledger found for it must carry this
run's delivery token: a ledger the walker wandered to from another run cannot.

SPLIT IN TWO ON PURPOSE. :func:`plan_asset_cleanup` is pure -- it reads the
tree and the facts the mux gathered and returns what it WOULD do, or why it
refuses. :func:`execute_asset_cleanup` does it. Both are stdlib only, so the
tests drive them on a ``tmp_path`` tree without ComfyUI, torch or a ledger.

LINKS ARE NEVER FOLLOWED. A linked DIRECTORY (symlink or Windows junction)
anywhere inside the episode folder is a refusal: the episode tree never holds
one, so finding one means the tree is not what this module thinks it is. A
linked FILE is unlinked as an entry and never resolved -- the rule
``_otr_janitor.py`` already uses.

UTF-8, no BOM, ASCII-only source. Dependency-free (stdlib only).
"""
from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

#: The three slugs, in dropdown order. The ledger stamp is always one of the
#: last two; ``off`` is never stamped (an absent key IS off).
MODES = ("off", "partial", "full")

#: The dropdown labels. Every one says what it KEEPS, in full, because a
#: stranger reads this dropdown cold (operator, 2026-09-25: "use the full text
#: so we make it easy for people"). The stamp is the FIRST WORD, so these can
#: be reworded later without touching a stamp.
LABELS = (
    "off (keep everything)",
    "partial (keep only the text files)",
    "full (keep only the published video)",
)
DEFAULT_LABEL = LABELS[0]

#: What ``partial`` deletes, by lower-cased extension. It is a DELETE list and
#: not a keep list on purpose: an extension nobody thought of is KEPT, because
#: the safe mistake is a kept file.
PARTIAL_DELETE_EXTENSIONS = frozenset({
    ".mp4", ".mkv", ".mov", ".webm",
    ".wav", ".flac", ".mp3", ".m4a",
    ".png", ".jpg", ".jpeg", ".webp", ".gif", ".exr",
    ".npy",
})

TOOLTIP = (
    "What to delete from this episode's folder once it has been published.\n\n"
    "off keeps everything.\n\n"
    "partial deletes the sound and the pictures (video, audio, images) and "
    "keeps every text file: the ledger, the canon, the treatment, the "
    "manifests, the captions and the QA reports.\n\n"
    "full deletes the whole episode folder.\n\n"
    "All three leave otr/obs alone: the published video there is never "
    "touched, and nothing is deleted unless that published copy is on disk. "
    "An episode whose publication was withheld is never cleaned.\n\n"
    "A partial or full episode can no longer be frozen into a replay bundle "
    "(the freeze needs the master WAV and the stills). Freeze first, or run "
    "it off."
)


def slug_from_label(label) -> str:
    """The slug a dropdown label stands for: its first word, lower-cased.

    Raises ValueError on a label whose first word is not a mode -- a widget
    value is untrusted input, and an unknown one fails at zero cost rather than
    being guessed at. An empty value (a graph saved before this widget existed)
    is ``off``.
    """
    text = str(label or "").strip()
    if not text:
        return "off"
    slug = text.split()[0].lower()
    if slug not in MODES:
        raise ValueError(
            "asset_cleanup: %r is not one of %s" % (text, ", ".join(LABELS)))
    return slug


def mode_from_meta(meta) -> Tuple[str, Optional[str]]:
    """The mode a ledger's meta asks for, and a note when it was not readable.

    Absent is ``off`` with no note. An unknown value is ``off`` WITH a note, so
    the caller logs it once: a value this module does not understand never
    deletes anything.
    """
    raw = (meta or {}).get("asset_cleanup") if isinstance(meta, dict) else None
    if raw is None:
        return "off", None
    slug = str(raw).strip().lower()
    if slug in MODES:
        return slug, None
    return "off", "unknown asset_cleanup value %r on the ledger; treated as off" % (raw,)


# --------------------------------------------------------------------------- #
# the planner
# --------------------------------------------------------------------------- #


def _is_link(path) -> bool:
    """A symlink or a Windows junction. Never raises."""
    try:
        if os.path.islink(path):
            return True
        isjunction = getattr(os.path, "isjunction", None)
        return bool(isjunction and isjunction(path))
    except OSError:
        return True


def _norm(path) -> str:
    """The real, case-folded path. REAL and not lexical: the episode folder
    arrives already resolved from the singleton, while the video paths arrive
    as the nodes wrote them, and an output tree reached through a junction
    would otherwise never compare equal -- a refusal on every run, which is
    the safe direction and still a feature that never fires."""
    return os.path.normcase(os.path.realpath(os.path.abspath(str(path))))


def _inside(path, directory) -> bool:
    """Is ``path`` the directory itself or below it?"""
    p, d = _norm(path), _norm(directory)
    return p == d or p.startswith(d.rstrip("\\/") + os.sep)


def _walk(episode_dir: Path):
    """(files, dirs, linked_dirs) under ``episode_dir``, never descending a
    link. ``files`` includes linked files, as entries."""
    files: List[Path] = []
    dirs: List[Path] = []
    linked_dirs: List[Path] = []
    stack = [episode_dir]
    while stack:
        current = stack.pop()
        with os.scandir(current) as entries:
            for entry in entries:
                path = Path(entry.path)
                if _is_link(path):
                    if entry.is_dir():          # follows the link to ask
                        linked_dirs.append(path)
                    else:
                        files.append(path)
                    continue
                if entry.is_dir(follow_symlinks=False):
                    dirs.append(path)
                    stack.append(path)
                else:
                    files.append(path)
    return sorted(files), sorted(dirs), sorted(linked_dirs)


def plan_asset_cleanup(
    episode_dir,
    mode: str,
    *,
    episodes_root,
    obs_copy,
    video_paths: Iterable = (),
    obs_dir=None,
    wire_token=None,
    ledger_token=None,
    ledger_obs_path=None,
) -> Tuple[List[Path], List[Path], Optional[str]]:
    """What a cleanup of ``episode_dir`` would delete and keep, or why not.

    Returns ``(delete, keep, refusal)``. ``refusal`` is None when the plan may
    run; otherwise it names the first fact that did not agree and both lists
    are empty. ``off`` is always ``([], [], None)``. ``full`` lists the folder
    itself as its one deletion; ``partial`` lists files.

    Every fact is passed in by the caller, so this reads the tree and nothing
    else -- no ledger, no singleton, no ComfyUI:

    * ``episode_dir`` -- the in-flight singleton's folder, or None.
    * ``episodes_root`` -- ``otr_episodes_root()``; the folder must be a
      direct child of it and not ``_``-prefixed.
    * ``obs_copy`` -- the published copy. None (a BLOCKED episode) refuses:
      the archival final is then its only copy.
    * ``video_paths`` -- the silent video and the archival final; each must
      resolve INSIDE the folder, or the folder is not the one this run wrote.
    * ``obs_dir`` -- ``otr_obs_dir()``; the folder may not be it or contain it.
    * ``wire_token`` / ``ledger_token`` -- the delivery token on the
      ``script_json`` wire and on the ledger found for this stem. Both must be
      present and equal: that binding is what makes a ledger found by the mtime
      fallback harmless.
    * ``ledger_obs_path`` -- ``meta.obs_final_path`` RE-READ from disk after
      the terminal stamp; it must name ``obs_copy``.
    """
    refused = lambda why: ([], [], why)  # noqa: E731 -- one shape, many exits
    slug = str(mode or "").strip().lower()
    if slug not in MODES:
        return refused("unknown mode %r" % (mode,))
    if slug == "off":
        return [], [], None

    # --- the folder is the one this run wrote -------------------------------
    if episode_dir is None:
        return refused("no in-flight episode answers for this video")
    episode_dir = Path(episode_dir)
    if not str(episode_dir.name) or episode_dir.name.startswith("_"):
        return refused("%s is a reserved folder, not an episode" % episode_dir.name)
    if _is_link(episode_dir):
        return refused("%s is a link; links are never followed" % episode_dir)
    if not episode_dir.is_dir():
        return refused("%s is not a folder on disk" % episode_dir)
    if episodes_root is None or _norm(episode_dir.parent) != _norm(episodes_root):
        return refused("%s is not a direct child of the episodes root" % episode_dir)
    if obs_dir is not None and _inside(obs_dir, episode_dir):
        return refused("the obs folder is inside %s" % episode_dir)

    # --- the run token -------------------------------------------------------
    wire = str(wire_token or "").strip()
    held = str(ledger_token or "").strip()
    if not wire:
        return refused("the script_json wire carries no delivery token")
    if not held:
        return refused("the ledger for this episode carries no delivery token")
    if wire != held:
        return refused("the ledger's delivery token is not this run's")

    # --- the videos this run made live in the folder -------------------------
    videos = [v for v in (video_paths or ()) if str(v or "").strip()]
    if not videos:
        return refused("no video path to bind the folder to")
    for video in videos:
        if not _inside(video, episode_dir):
            return refused("%s is outside %s" % (video, episode_dir))

    # --- the publish proof ---------------------------------------------------
    if not obs_copy:
        return refused("nothing was published (a withheld episode is never cleaned)")
    if _inside(obs_copy, episode_dir):
        return refused("the published copy is inside the folder it would clean")
    try:
        if not os.path.isfile(obs_copy) or os.path.getsize(obs_copy) <= 0:
            return refused("the published copy %s is missing or empty" % obs_copy)
    except OSError as exc:
        return refused("the published copy cannot be read: %s" % exc)
    if not ledger_obs_path or _norm(ledger_obs_path) != _norm(obs_copy):
        return refused("the ledger does not record %s as the published copy" % obs_copy)

    # --- the tree ------------------------------------------------------------
    try:
        files, _dirs, linked_dirs = _walk(episode_dir)
    except OSError as exc:
        return refused("the folder cannot be read: %s" % exc)
    if linked_dirs:
        return refused("a linked folder sits inside the episode: %s" % linked_dirs[0])

    if slug == "full":
        return [episode_dir], [], None
    delete = [f for f in files if f.suffix.lower() in PARTIAL_DELETE_EXTENSIONS]
    keep = [f for f in files if f.suffix.lower() not in PARTIAL_DELETE_EXTENSIONS]
    return delete, keep, None


# --------------------------------------------------------------------------- #
# the executor
# --------------------------------------------------------------------------- #


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _size(path: Path) -> int:
    try:
        return 0 if _is_link(path) else os.path.getsize(path)
    except OSError:
        return 0


def _remove_empty_dirs(episode_dir: Path) -> List[str]:
    """Remove folders left empty, deepest first. Never the episode folder
    itself, never a link. Returns what went."""
    removed = []
    for root, dirnames, _files in os.walk(episode_dir, topdown=False):
        for name in dirnames:
            path = os.path.join(root, name)
            if _is_link(path):
                continue
            try:
                os.rmdir(path)             # refuses a folder that still holds anything
                removed.append(path)
            except OSError:
                continue
    return removed


def execute_asset_cleanup(
    mode: str,
    episode_dir,
    delete: List[Path],
    keep: List[Path],
    *,
    write_receipt: Callable[[dict], bool],
    obs_copy: str = "",
) -> str:
    """Carry out a plan from :func:`plan_asset_cleanup`. Returns one report line.

    ``write_receipt(receipt) -> bool`` stamps ``meta.asset_cleanup_receipt`` on
    the in-flight ledger and saves it. It is called TWICE for ``partial``:
    ``state: started`` BEFORE the first unlink, so a crash mid-delete leaves a
    record, and ``state: done`` after. If the started receipt cannot be saved
    nothing is deleted. ``full`` writes the started receipt and then removes
    the folder, receipt and all; its done receipt lands only when a locked
    file kept the ledger alive.

    A file that will not go (a Windows handle, a permission) is SKIPPED and
    named; the rest still goes. Only an OSError is handled here -- anything
    else propagates, so the caller's own ordering (re-raise a cancel first,
    then report) stays the caller's.
    """
    slug = str(mode or "").strip().lower()
    if slug not in ("partial", "full"):
        return "asset_cleanup %s: nothing to do" % slug
    episode_dir = Path(episode_dir)
    started = {"state": "started", "mode": slug, "when": _now()}
    if not write_receipt(started):
        return ("asset_cleanup %s: skipped -- the started receipt could not be "
                "saved, so nothing was deleted" % slug)

    removed: List[str] = []
    skipped: List[str] = []
    removed_bytes = 0
    if slug == "full":
        try:
            files, _dirs, _linked = _walk(episode_dir)
        except OSError:
            files = []
        sizes = {str(f): _size(f) for f in files}

        def _skip_entry(_func, _path, _exc):
            # Keep going. Which FILES survived is read off the disk below; a
            # folder that refused is only refusing because a file in it did.
            return None

        if sys.version_info >= (3, 12):
            shutil.rmtree(episode_dir, onexc=_skip_entry)
        else:  # pragma: no cover -- the pack runs on 3.12
            shutil.rmtree(episode_dir, onerror=_skip_entry)
        for name, nbytes in sizes.items():
            if os.path.lexists(name):
                skipped.append(name)
            else:
                removed.append(name)
                removed_bytes += nbytes
        kept: List[str] = []
    else:
        for path in delete:
            nbytes = _size(path)
            try:
                os.unlink(path)            # a link goes as an entry, never resolved
            except OSError:
                skipped.append(str(path))
                continue
            removed.append(str(path))
            removed_bytes += nbytes
        _remove_empty_dirs(episode_dir)
        kept = [str(k) for k in keep]

    # THE DONE RECEIPT. For `partial` the ledger is still there and this is
    # the tombstone that explains the missing files. For `full` it normally
    # went with the folder and the save simply reports False; when a locked
    # file kept the folder (and possibly the ledger) alive, it lands.
    done_saved = write_receipt({
        "state": "done",
        "mode": slug,
        "when": started["when"],
        "finished": _now(),
        "removed": removed,
        "removed_bytes": removed_bytes,
        "kept": kept,
        "skipped": skipped,
        "replay_freeze_possible": False,
    })
    line = ("asset_cleanup %s: removed %d files (%.1f MB), kept %d, skipped %d"
            % (slug, len(removed), removed_bytes / (1024.0 * 1024.0),
               len(kept), len(skipped)))
    if slug == "partial" and not done_saved:
        line += "; the done receipt did not save"
    if skipped:
        line += " [skipped: %s]" % ", ".join(skipped)
    if obs_copy:
        line += "; the published copy stays at %s" % obs_copy
    return line


__all__ = [
    "MODES", "LABELS", "DEFAULT_LABEL", "PARTIAL_DELETE_EXTENSIONS", "TOOLTIP",
    "slug_from_label", "mode_from_meta",
    "plan_asset_cleanup", "execute_asset_cleanup",
]
