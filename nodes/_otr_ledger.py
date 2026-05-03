"""
_otr_ledger.py  --  shared ledger I/O helpers for OTR nodes
============================================================

Every OTR node that writes diagnostic fields back to the per-episode
``*_ledger.json`` re-implements roughly the same load-mutate-save
dance (BUG-090 director_raw, BUG-094 line timings, BUG-095 music
wav_paths, BUG-096 line dur_s, BUG-098 cast dedup, BUG-102 clips
warmup_pad_ms, ...). This module consolidates the boilerplate so:

  * Schema version bumps land in exactly one place.
  * sha256-of-first-1KB gate hashing is consistent across nodes.
  * git-commit lookup is identical regardless of caller.
  * Failures degrade silently with a warning rather than crashing
    a 4-hour render -- a missing ledger field is never worth
    aborting the run.

All helpers are best-effort: any I/O exception is caught and logged
at WARNING level. Callers should NEVER let a ledger write-back
failure abort their main work.

Schema version
--------------
Current canonical: ``l3-2026-04-28`` (post-BUG-100/101/102 diagnostic
expansion). Previous: ``l2-2026-04-25``. The version string is
written to ``ledger["schema_version"]`` AND ``ledger["meta"]["schema_version"]``.
Consumers should accept either location for back-compat with l2 ledgers.
"""
from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Iterable, Optional

log = logging.getLogger("OTR")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CURRENT_SCHEMA_VERSION = "l3-2026-05-02"
"""Set on every ledger write performed via this module. Bump when
adding any field that downstream consumers must check for. Keep the
date suffix so the lineage is greppable.

Lineage:
  l1-2026-04-24 -- baseline (cast, scenes, shots, lines, sfx, music, clips)
  l2-2026-04-25 -- adds beats[] hierarchy
  l3-2026-04-28 -- diagnostic expansion (meta.phase_ms, audio_gates,
                   text_for_tts, bark_render_ms, warmup_pad_ms,
                   transitions, radio_bookend_path)
  l3-2026-05-02 -- ADDITIVE: meta.paths block resolved at write time so
                   downstream nodes can look up canonical episode dirs
                   without reconstructing them from episode_id (Phase E,
                   BUG-LOCAL-018). All readers must continue to use
                   meta.get(...) with default-None to stay back-compat
                   with l3-2026-04-28 ledgers.
"""

GATE_HASH_BYTES = 1024
"""How many leading bytes of a master audio waveform to feed into
sha256 for the per-gate integrity check. 1 KB is enough to detect
sample-rate / channel / encoding drift but cheap enough to compute
inside a render's hot path. The full audio is NOT hashed -- this is
a tripwire, not a verification suite."""


# ---------------------------------------------------------------------------
# Load / save (best-effort)
# ---------------------------------------------------------------------------

def load_ledger_safe(path: Path) -> Optional[dict]:
    """Read a ledger JSON file. Returns None on any error.

    Callers should treat None as "the ledger isn't usable; skip
    write-back". Errors are logged at WARNING.
    """
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        log.warning("[OTR_Ledger] ledger not found: %s", path)
        return None
    except json.JSONDecodeError as exc:
        log.warning("[OTR_Ledger] ledger JSON parse failed (%s): %s", path, exc)
        return None
    except Exception as exc:  # OSError, permissions, etc.
        log.warning("[OTR_Ledger] ledger load failed (%s): %s", path, exc)
        return None


def _build_meta_paths(ledger_path: Path, episode_id: str) -> dict:
    """Resolve the canonical absolute paths for this episode's per-dir
    workspace, derived from the on-disk ledger location.

    Layout (post 2026-05-02 EVENING reorg):
        output/otr/
          obs/
            <episode_id>.mp4
          episodes/
            <episode_id>/
              audio/<episode_id>_ledger.json
              stills/
              portraits/
              videos/
              composited/

    Detects layout by checking if the ledger sits two levels under a
    directory literally named "episodes". When detected, returns a fully
    populated paths block with absolute paths for every per-episode
    subdir + the OBS final mp4. When not detected (legacy flat layout
    or test fixtures with arbitrary parents), returns a minimal block
    with just episode_root and audio_dir resolved from the ledger
    location -- no fabricated subdirs.

    BUG-LOCAL-018 (Phase E, 2026-05-02): adding this block lets
    downstream nodes look up canonical paths via
    ``ledger["meta"]["paths"]["audio_dir"]`` instead of reconstructing
    them from episode_id (which would re-introduce the slug-mismatch
    risk Phase C closed). All readers must use ``.get(...)`` with
    default-None for back-compat with older ledgers that lack this
    block.
    """
    ledger_path = Path(ledger_path).resolve()
    audio_dir = ledger_path.parent
    ep_dir = audio_dir.parent

    paths: dict = {
        "ledger_path": str(ledger_path),
        "episode_root": str(ep_dir),
        "audio_dir": str(audio_dir),
    }

    # Per-episode workspace layout detection: ep_dir's parent must be
    # named "episodes" (case-insensitive on Windows but stored as-is).
    parent_of_ep = ep_dir.parent
    if parent_of_ep.name.lower() == "episodes":
        paths["stills_dir"] = str(ep_dir / "stills")
        paths["portraits_dir"] = str(ep_dir / "portraits")
        paths["videos_dir"] = str(ep_dir / "videos")
        paths["composited_dir"] = str(ep_dir / "composited")
        # OBS final lives at output/otr/obs/<ep>.mp4 -- a sibling of
        # episodes/, NOT a child. Only stamp it if the grandparent of
        # ep_dir has a sibling named "obs"; otherwise omit (we don't
        # fabricate a path that may not exist).
        otr_root = parent_of_ep.parent
        obs_root = otr_root / "obs"
        if otr_root.name.lower() == "otr":
            paths["obs_final"] = str(obs_root / f"{episode_id}.mp4")
            paths["obs_dir"] = str(obs_root)
        paths["layout"] = "per-episode-workspace"
    else:
        paths["layout"] = "legacy-flat"

    return paths


def save_ledger_safe(path: Path, ledger: dict) -> bool:
    """Write a ledger dict back to disk with schema_version + meta.paths
    stamped.

    Always sets:
      - ``ledger["schema_version"]``
      - ``ledger["meta"]["schema_version"]``
      - ``ledger["meta"]["paths"]``  (BUG-LOCAL-018, Phase E)

    The ``meta.paths`` block is resolved fresh on every save from the
    actual on-disk ``path`` argument. This makes it self-correcting --
    if the per-episode dir was renamed mid-pipeline (Phase B
    rename_episode), the next ledger save reflects the new location
    without any caller having to update it explicitly.

    Returns True on success, False on failure (always logs WARNING).
    """
    try:
        ledger["schema_version"] = CURRENT_SCHEMA_VERSION
        meta = ledger.setdefault("meta", {})
        meta["schema_version"] = CURRENT_SCHEMA_VERSION
        episode_id = ledger.get("episode_id") or ""
        meta["paths"] = _build_meta_paths(Path(path), str(episode_id))
        Path(path).write_text(
            json.dumps(ledger, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return True
    except Exception as exc:
        log.warning("[OTR_Ledger] ledger save failed (%s): %s", path, exc)
        return False


# ---------------------------------------------------------------------------
# Auto-discover most recent ledger in the canonical audio dir
# ---------------------------------------------------------------------------

def in_flight_ledger_path() -> Optional[Path]:
    """Return the in-flight Ledger singleton's on-disk path.

    BUG-LOCAL-021 (Phase G, 2026-05-03): replaces ``find_most_recent_ledger``
    for in-flight write-back paths. The mtime walker can return a stale
    leftover ledger across queue boundaries (proven by the FLUX radio
    bookend stamping to a 6-day-old episode on the 2026-05-02 soak). The
    in-flight singleton's ``path`` property advances correctly through
    ``Ledger.rename_episode`` (Phase B), so it tracks the active episode
    by construction.

    ComfyUI sequential queue + ``LLMScriptWriter.IS_CHANGED = time.time()``
    prevent the singleton from going stale across queued runs. Falls back
    to ``find_most_recent_ledger`` for headless/standalone scenarios where
    no LLMScriptWriter has run in this process.

    Returns the path on success, None on miss. Never raises.
    """
    try:
        # Late import: production_ledger imports _otr_ledger at class-init
        # for the SCHEMA_VERSION; doing the reverse import at module load
        # would cycle. Late-binding inside the function is safe.
        try:
            from . import production_ledger as _PL  # type: ignore
        except ImportError:
            import production_ledger as _PL  # type: ignore
        led = _PL.get_ledger()
        p = Path(led.path)
        if p.exists():
            return p
        # Singleton's path doesn't exist on disk (rare: rename failed
        # silently, or singleton initialized but never .save()'d). Fall
        # through to legacy walker.
        log.debug(
            "[OTR_Ledger] in_flight singleton path %s not on disk; "
            "falling back to mtime walker", p,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "[OTR_Ledger] in_flight singleton lookup failed (%s); "
            "falling back to mtime walker", exc,
        )
    # Last-resort fallback: legacy mtime walker. Standalone/test paths
    # without an active singleton can still find a ledger this way.
    try:
        try:
            from . import _otr_paths as _P
        except ImportError:
            import _otr_paths as _P  # type: ignore
        return find_most_recent_ledger(
            [_P.otr_episodes_root(), _P.otr_legacy_audio_dir()]
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("[OTR_Ledger] fallback mtime walker failed: %s", exc)
        return None


def find_most_recent_ledger(audio_dirs: Iterable[Path]) -> Optional[Path]:
    """Search the supplied dirs for ``*_ledger.json`` and return the
    newest by mtime. Returns None if no candidates found.

    Walks each given dir at TWO levels:
      1. ``<dir>/*_ledger.json``                   (legacy flat layout)
      2. ``<dir>/<episode_id>/audio/*_ledger.json``  (per-episode workspace
         layout, post 2026-05-02 EVENING reorg -- see
         ``otr_episodes_root()`` in ``_otr_paths``)

    BUG-LOCAL-103 (2026-04-29 morning): we used to filter out
    ``pending_*_ledger.json`` here, which silently broke every
    audio-node ledger write. BatchBark / SceneSequencer /
    AudioEnhance / EpisodeAssembler all run while the ledger is
    still ``pending_<title>_ledger.json`` (LLMScriptWriter creates
    the pending file; SignalLostVideo renames it once the audio
    title is finalized). Filtering pending_* meant those four
    nodes' write-backs no-op'd because they couldn't find the
    in-flight ledger. Fix: include pending_* in the glob; mtime
    sort still prefers the newest, so a renamed canonical ledger
    naturally wins after rename.

    Use ``otr_episodes_root()`` (per-episode workspace) +
    ``otr_legacy_audio_dir()`` (pre-cutover legacy) from
    ``_otr_paths`` as the canonical search list.
    """
    candidates: list[Path] = []
    for d in audio_dirs:
        try:
            d = Path(d)
            if not d.exists():
                continue
            # Legacy flat layout: ledgers directly in the dir.
            candidates.extend(d.glob("*_ledger.json"))
            # Per-episode workspace layout (post 2026-05-02 EVENING):
            # ledgers under <dir>/<episode_id>/audio/*_ledger.json.
            candidates.extend(d.glob("*/audio/*_ledger.json"))
        except Exception as exc:
            log.warning("[OTR_Ledger] ledger glob failed in %s: %s", d, exc)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


# ---------------------------------------------------------------------------
# Per-line / per-clip patching
# ---------------------------------------------------------------------------

def patch_line_fields(
    ledger: dict,
    line_id: str,
    fields: dict,
) -> bool:
    """Merge ``fields`` into ``ledger["lines"][i]`` for the line whose
    ``line_id`` matches. Returns True if a row was updated, False if
    no matching line was found.
    """
    lines = ledger.get("lines") or []
    for ln in lines:
        if ln.get("line_id") == line_id:
            ln.update(fields)
            return True
    return False


def patch_clip_fields(
    ledger: dict,
    line_id: str,
    fields: dict,
) -> bool:
    """Merge ``fields`` into ``ledger["clips"][i]`` for the clip whose
    ``line_id`` matches. Returns True if a row was updated, False if
    no matching clip was found.
    """
    clips = ledger.get("clips") or []
    for cl in clips:
        if cl.get("line_id") == line_id:
            cl.update(fields)
            return True
    return False


# ---------------------------------------------------------------------------
# Audio gate integrity hashing (CLAUDE.md C7)
# ---------------------------------------------------------------------------

def audio_gate_record(
    gate_name: str,
    waveform_bytes: bytes,
    dur_s: float,
    sample_count: int,
    sample_rate: int,
) -> dict:
    """Build a single ``audio_gates[]`` entry. Caller passes the
    already-extracted leading bytes (typically waveform.cpu().numpy()
    .tobytes()[:GATE_HASH_BYTES]) so this helper does no torch I/O.
    """
    h = hashlib.sha256(waveform_bytes).hexdigest()[:32]
    return {
        "gate": str(gate_name),
        "dur_s": float(dur_s),
        "sample_count": int(sample_count),
        "sample_rate": int(sample_rate),
        "sha256_first_kb": h,
    }


def append_audio_gate(
    ledger: dict,
    gate_record: dict,
) -> None:
    """Append a gate record to ``ledger["audio_gates"]``, creating
    the array if absent. No-op-safe (never raises).
    """
    try:
        gates = ledger.setdefault("audio_gates", [])
        gates.append(gate_record)
    except Exception as exc:
        log.warning("[OTR_Ledger] append_audio_gate failed: %s", exc)


# ---------------------------------------------------------------------------
# Meta block (git commit, phase timing, schema version)
# ---------------------------------------------------------------------------

_GIT_COMMIT_CACHE: Optional[str] = None
"""Resolve once per process. The OTR repo's HEAD doesn't change mid-run;
re-shelling out to git for every ledger save wastes a few ms."""


def lookup_git_commit(repo_root: Path) -> Optional[str]:
    """Return the short SHA of HEAD for the repo containing this
    module. Caches the result for the lifetime of the process. Returns
    None if git lookup fails (uninstalled, not a repo, etc.).
    """
    global _GIT_COMMIT_CACHE
    if _GIT_COMMIT_CACHE is not None:
        return _GIT_COMMIT_CACHE
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(repo_root)),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if out.returncode == 0:
            _GIT_COMMIT_CACHE = out.stdout.strip() or None
            return _GIT_COMMIT_CACHE
    except Exception as exc:
        log.warning("[OTR_Ledger] git rev-parse failed: %s", exc)
    return None


def set_meta(ledger: dict, key: str, value: Any) -> None:
    """Set ``ledger["meta"][key] = value``, creating ``meta`` if
    absent. No-op-safe.
    """
    try:
        meta = ledger.setdefault("meta", {})
        meta[key] = value
    except Exception as exc:
        log.warning("[OTR_Ledger] set_meta failed (%s): %s", key, exc)


def record_phase_ms(ledger: dict, phase_name: str, ms: int) -> None:
    """Record a node's wall-clock duration into
    ``ledger["meta"]["phase_ms"][phase_name]``. Multiple writes for
    the same phase overwrite (last write wins). Used to surface the
    bottleneck-phase per episode.
    """
    try:
        meta = ledger.setdefault("meta", {})
        phase_ms = meta.setdefault("phase_ms", {})
        phase_ms[str(phase_name)] = int(ms)
    except Exception as exc:
        log.warning("[OTR_Ledger] record_phase_ms failed (%s): %s", phase_name, exc)


# ---------------------------------------------------------------------------
# Transitions (consecutive line boundaries with crossfade)
# ---------------------------------------------------------------------------

def append_transition(
    ledger: dict,
    from_line_id: str,
    to_line_id: str,
    crossfade_ms: int,
    boundary_s: float,
) -> None:
    """Append a transition record to ``ledger["transitions"]``. Used
    by SceneSequencer / EpisodeAssembler to capture the exact
    crossfade overlap between consecutive lines so post-mortem can
    diagnose audio-tail-eating-next-clip-start regressions.
    """
    try:
        transitions = ledger.setdefault("transitions", [])
        transitions.append({
            "from_line_id": str(from_line_id) if from_line_id else None,
            "to_line_id": str(to_line_id) if to_line_id else None,
            "crossfade_ms": int(crossfade_ms),
            "boundary_s": float(boundary_s),
        })
    except Exception as exc:
        log.warning("[OTR_Ledger] append_transition failed: %s", exc)


__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "GATE_HASH_BYTES",
    "load_ledger_safe",
    "save_ledger_safe",
    "find_most_recent_ledger",
    "in_flight_ledger_path",
    "patch_line_fields",
    "patch_clip_fields",
    "audio_gate_record",
    "append_audio_gate",
    "lookup_git_commit",
    "set_meta",
    "record_phase_ms",
    "append_transition",
]
