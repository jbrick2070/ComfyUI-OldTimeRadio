"""
_otr_probe.py  --  measure-don't-assume helpers for OTR pipeline
==================================================================

Single source of truth for "what is the real duration / mtime / shape
of this artifact on disk?". Use these at every phase boundary instead
of trusting metadata fields like ``ledger.clips[].dur_s``.

Rationale (BUG-LOCAL-033 forensics, 2026-05-03 EVENING):
``ledger.clips[].dur_s`` for LTX broadcast units stamps the LINE's
audio duration (e.g. 12.72s) instead of the actual rendered VIDEO
duration (e.g. 7.72s). A downstream consumer that trusts that field
will compute the wrong drift correction. The ONLY reliable way to
know what's on disk is to ffprobe it.

Rule: metadata describes intent, ffprobe describes reality.
When they disagree, reality wins.

Two helpers:

  * ``probe_duration_s(path)``  -- returns actual on-disk duration in
                                   seconds via ffprobe -show_entries
                                   format=duration. Returns 0.0 on
                                   any failure (caller decides what
                                   to do with that).

  * ``stage_output_is_fresh(out_path, in_path)`` -- returns True iff
                                   ``out_path`` mtime is strictly
                                   greater than ``in_path`` mtime.
                                   Use at top of smoke / live runs to
                                   confirm the stage actually re-ran
                                   (not served from cache).

Both are best-effort: any I/O exception returns a safe degraded value
so the caller can decide whether to log + abort or log + continue.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger("OTR")


def probe_duration_s(path: Path | str, *, ffprobe: str = "ffprobe") -> float:
    """Return the actual on-disk duration of an audio/video file in
    seconds via ``ffprobe -show_entries format=duration``.

    Returns 0.0 on any failure (file missing, ffprobe missing, parse
    error, container without duration metadata, etc.). Caller MUST
    decide how to handle the 0.0 sentinel (typically: skip the
    drift-correction logic for this clip and log a warning).
    """
    p = Path(path)
    if not p.exists():
        return 0.0
    if not shutil.which(ffprobe):
        log.warning("[OTR_Probe] ffprobe binary not on PATH; cannot measure %s", p.name)
        return 0.0
    try:
        out = subprocess.check_output(
            [
                ffprobe, "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=nw=1:nk=1",
                str(p),
            ],
            stderr=subprocess.PIPE,
        ).decode("utf-8", errors="replace").strip()
    except Exception as exc:  # noqa: BLE001
        log.warning("[OTR_Probe] ffprobe failed on %s: %s", p.name, exc)
        return 0.0
    if not out:
        return 0.0
    try:
        return float(out.splitlines()[0])
    except Exception:
        return 0.0


def stage_output_is_fresh(out_path: Path | str, in_path: Path | str) -> tuple[bool, str]:
    """Check whether ``out_path`` is newer than ``in_path`` on disk.

    Use at the top of a smoke / live run to confirm the stage's
    output was actually regenerated (not served from cache or left
    stale by a partial prior run). Returns ``(is_fresh, reason)``.

    ``is_fresh=True`` means ``out_path`` mtime is strictly greater
    than ``in_path`` mtime.

    ``is_fresh=False`` reasons:
      * out_path missing (stage didn't write)
      * in_path missing (caller bug)
      * out_path mtime <= in_path mtime (stale; cache served)

    Both paths must exist; if either is missing, returns
    ``(False, "<which> missing")``.
    """
    out = Path(out_path)
    inp = Path(in_path)
    if not inp.exists():
        return False, f"input {inp.name} missing"
    if not out.exists():
        return False, f"output {out.name} missing (stage did not write)"
    try:
        out_m = out.stat().st_mtime
        in_m = inp.stat().st_mtime
    except OSError as exc:
        return False, f"stat() failed: {exc}"
    if out_m > in_m:
        return True, (
            f"{out.name} mtime {out_m:.3f} > {inp.name} mtime {in_m:.3f} "
            f"(delta {out_m - in_m:.1f}s)"
        )
    return False, (
        f"STALE: {out.name} mtime {out_m:.3f} <= {inp.name} mtime {in_m:.3f} "
        f"(delta {out_m - in_m:.1f}s) -- cached or partially-regenerated output"
    )


# Frame-quantized duration tolerance. One frame at 25fps = 40ms.
# Use as the +/- bound for "this segment matches the audio target".
DEFAULT_DURATION_TOL_S = 0.04


def duration_gap_s(
    audio_target_s: float,
    video_actual_s: float,
    tol_s: float = DEFAULT_DURATION_TOL_S,
) -> tuple[float, str]:
    """Compute the per-clip duration gap and classify the action.

    Returns ``(gap_seconds, action)`` where action is one of:
      * ``"extend"``   -- gap > +tol; need ``tpad`` to freeze-extend
      * ``"truncate"`` -- gap < -tol; need ``-t`` to truncate
      * ``"noop"``     -- |gap| <= tol; no action needed

    gap = audio_target_s - video_actual_s. Positive means audio is
    longer than video (need to pad video). Negative means video is
    longer than audio (need to truncate video).
    """
    gap = float(audio_target_s) - float(video_actual_s)
    if gap > tol_s:
        return gap, "extend"
    if gap < -tol_s:
        return gap, "truncate"
    return gap, "noop"


__all__ = [
    "probe_duration_s",
    "stage_output_is_fresh",
    "duration_gap_s",
    "DEFAULT_DURATION_TOL_S",
]
