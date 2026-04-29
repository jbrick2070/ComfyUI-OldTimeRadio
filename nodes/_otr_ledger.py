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

CURRENT_SCHEMA_VERSION = "l3-2026-04-28"
"""Set on every ledger write performed via this module. Bump when
adding any field that downstream consumers must check for. Keep the
date suffix so the lineage is greppable."""

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


def save_ledger_safe(path: Path, ledger: dict) -> bool:
    """Write a ledger dict back to disk with schema_version stamped.

    Always sets ``ledger["schema_version"]`` and
    ``ledger.setdefault("meta", {})["schema_version"]`` to
    ``CURRENT_SCHEMA_VERSION`` so any node that touches the ledger
    keeps the version field current.

    Returns True on success, False on failure (always logs WARNING).
    """
    try:
        ledger["schema_version"] = CURRENT_SCHEMA_VERSION
        meta = ledger.setdefault("meta", {})
        meta["schema_version"] = CURRENT_SCHEMA_VERSION
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

def find_most_recent_ledger(audio_dirs: Iterable[Path]) -> Optional[Path]:
    """Search the supplied dirs for ``*_ledger.json`` and return the
    newest by mtime. Returns None if no candidates found.

    BUG-LOCAL-103 (2026-04-29 morning): we used to filter out
    ``pending_*_ledger.json`` here, which silently broke every
    audio-node ledger write. BatchBark / SceneSequencer /
    AudioEnhance / EpisodeAssembler all run while the ledger is
    still ``pending_<title>_ledger.json`` (LLMScriptWriter creates
    the pending file; SignalLostVideo renames it once the audio
    title is finalized). Filtering pending_* meant those four
    nodes' write-backs no-op'd because they couldn't find the
    in-flight ledger. The deep_earth_echoes 2026-04-28 overnight
    run shipped with audio_gates=[] and meta.phase_ms missing 4
    of the 5 phases as a result. Fix: include pending_* in the
    glob; mtime sort still prefers the newest, so a renamed
    canonical ledger naturally wins after rename.

    Use ``otr_audio_dir()`` + ``otr_legacy_audio_dir()`` from
    ``_otr_paths`` as the canonical search list.
    """
    candidates: list[Path] = []
    for d in audio_dirs:
        try:
            d = Path(d)
            if not d.exists():
                continue
            candidates.extend(d.glob("*_ledger.json"))
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
    "patch_line_fields",
    "patch_clip_fields",
    "audio_gate_record",
    "append_audio_gate",
    "lookup_git_commit",
    "set_meta",
    "record_phase_ms",
    "append_transition",
]
