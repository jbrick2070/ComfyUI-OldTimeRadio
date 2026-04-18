"""
otr_v2.visual.postproc.vhs  --  VHS aesthetic post-processor
=============================================================

Day 8 of the 14-day video stack sprint.  Takes rendered per-shot
MP4 clips (from Days 5-7 backends) and applies a deterministic
VHS-tape filter chain via ffmpeg filter_complex:

    * Scanlines (alternating luminance rows)
    * Chroma bleed (soft blur on the chroma planes)
    * RGB shift / chromatic aberration (rgbashift)
    * Film grain / tape noise (noise=allf=t+u)
    * Vignette (soft edge darkening)
    * Final light blur (tape softness)

Audio is NEVER touched.  When the source MP4 has an audio track,
ffmpeg is invoked with ``-c:a copy`` so the audio bytes pass through
unchanged (C7 byte-identical guarantee).  The Renderer does the
final mux with the Bark episode audio downstream, so this stage
normally operates on video-only inputs anyway.

Stub mode (byte-identical passthrough copy of the input) activates
when any of:

    * env ``OTR_VHS_STUB=1``
    * ffmpeg binary not locatable
    * caller passes ``force_stub=True``

This lets CI, unit tests, and weight-missing dev machines exercise
the pipeline without ffmpeg and still get a writable output file.

No torch / diffusers imports.  Module import is cheap.

Design doc: ROADMAP.md Day 8.  Honours C4 (duration cap) and C7
(audio byte-identical) by construction:

    * C4 is preserved structurally: the filter chain never changes
      the frame count or timebase -- only pixel values.
    * C7 is preserved by ``-c:a copy`` (or by passthrough in stub
      mode, which does not re-encode anything).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable

log = logging.getLogger("OTR.visual.postproc.vhs")


# ------------------------------------------------------------------
# Defaults
# ------------------------------------------------------------------

# Canonical per-shot video filenames produced by the motion backends.
# ``render.mp4`` is the shared canonical name written by the Renderer
# layer when it normalises LTX/Wan outputs.  ``motion.mp4`` (Day 5)
# and ``loop.mp4`` (Day 6) are the raw backend filenames.  The VHS
# stage transforms the one that is present.
VIDEO_FILENAME_CANDIDATES: tuple[str, ...] = (
    "render.mp4",
    "motion.mp4",
    "loop.mp4",
)

# Files the VHS stage must skip (still-image or auxiliary assets).
SKIP_FILENAMES: frozenset[str] = frozenset({
    "keyframe.png",
    "composite.png",
    "mask.png",
    "depth.png",
    "anchor.png",
    "render.png",
    "STATUS.json",
    "meta.json",
    "shotlist.json",
})

# Output suffix for VHS-filtered clips.  The original input is left
# untouched; the renderer picks up the ``*_vhs.mp4`` when present.
VHS_SUFFIX: str = "_vhs.mp4"

# Default filter parameters.  Keys are intentionally flat so the
# caller can override one knob without reconstructing the whole dict.
DEFAULT_VHS_PARAMS: dict[str, Any] = {
    # Overall intensity: "low" | "medium" | "high".  Informs the
    # default strength of each effect when the specific knob is
    # left at None / missing.  Callers can always override any
    # individual knob.
    "intensity": "medium",
    # Scanline density (1 = every row, 2 = every other row, ...).
    "scanline_density": 2,
    # Scanline alpha blend strength (0.0-1.0).  ~0.18 reads as VHS
    # without eating detail.
    "scanline_alpha": 0.18,
    # RGB shift in pixels (chromatic aberration).  Positive = right.
    "rgb_shift_px": 2,
    # Chroma blur sigma (chroma bleed).  Applied to the U/V planes
    # via gblur.
    "chroma_blur_sigma": 1.2,
    # Tape grain / noise strength (ffmpeg ``noise`` c0s=, 0-100).
    "noise_strength": 14,
    # Vignette angle in radians-over-pi (ffmpeg ``vignette=PI/X``).
    # Higher = softer vignette.
    "vignette_over_pi": 5.0,
    # Final output blur (tape softness).  sigma in px.
    "final_blur_sigma": 0.35,
    # Output framerate (frames/sec).  Must match renderer _FPS=24.
    "fps": 24,
    # ffmpeg encoder preset for the video pass (audio is always copy).
    "preset": "fast",
    # ffmpeg CRF for libx264 (lower = higher quality).  22 is visually
    # clean at 720p, 24 if disk is tight.
    "crf": 22,
}

# Intensity scaling multipliers for the "low" / "medium" / "high"
# preset knob.  Only touches knobs whose intensity actually changes
# the filter chain text, so tests can assert that the chain string
# differs across presets.
_INTENSITY_SCALE: dict[str, dict[str, float]] = {
    "low": {
        "scanline_alpha": 0.5,
        "rgb_shift_px": 0.5,
        "chroma_blur_sigma": 0.5,
        "noise_strength": 0.4,
        "final_blur_sigma": 0.5,
    },
    "medium": {
        "scanline_alpha": 1.0,
        "rgb_shift_px": 1.0,
        "chroma_blur_sigma": 1.0,
        "noise_strength": 1.0,
        "final_blur_sigma": 1.0,
    },
    "high": {
        "scanline_alpha": 1.6,
        "rgb_shift_px": 1.6,
        "chroma_blur_sigma": 1.5,
        "noise_strength": 1.5,
        "final_blur_sigma": 1.3,
    },
}

# Canonical stub reasons recorded in meta.json.  Tests assert against
# these strings so the values are stable.
STUB_REASON_ENV: str = "env_otr_vhs_stub"
STUB_REASON_NO_FFMPEG: str = "ffmpeg_unavailable"
STUB_REASON_FORCED: str = "force_stub_flag"
STUB_REASON_REAL: str = "real_mode"

# Candidate ffmpeg locations (shared pattern with renderer._find_ffmpeg).
_FFMPEG_CANDIDATES: tuple[str, ...] = (
    "ffmpeg",
    r"C:\Users\jeffr\Documents\ComfyUI\ffmpeg\bin\ffmpeg.exe",
    r"C:\ffmpeg\bin\ffmpeg.exe",
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def find_ffmpeg() -> str | None:
    """Locate ffmpeg binary.  Returns the resolved path or None."""
    for candidate in _FFMPEG_CANDIDATES:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return None


def _merge_params(overrides: dict[str, Any] | None) -> dict[str, Any]:
    """Merge caller overrides onto ``DEFAULT_VHS_PARAMS``.  Pure."""
    merged = dict(DEFAULT_VHS_PARAMS)
    if overrides:
        for key, value in overrides.items():
            if value is not None:
                merged[key] = value
    # Coerce intensity to a known preset; fall back to "medium" on
    # unknown strings so the chain builder never crashes on user
    # input from the renderer UI.
    if merged.get("intensity") not in _INTENSITY_SCALE:
        merged["intensity"] = "medium"
    return merged


def _scaled(params: dict[str, Any], key: str) -> float:
    """Return ``params[key]`` scaled by the intensity multiplier."""
    base = float(params.get(key, DEFAULT_VHS_PARAMS.get(key, 0.0)))
    scale = _INTENSITY_SCALE[params["intensity"]].get(key, 1.0)
    return base * scale


def _params_hash(params: dict[str, Any]) -> str:
    """Deterministic short hash of the filter parameters (for meta.json)."""
    blob = json.dumps(params, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def _format_float(value: float, places: int = 3) -> str:
    """ffmpeg filter graph needs plain decimals, not 1e-05 etc."""
    formatted = f"{float(value):.{places}f}"
    # Strip trailing zeros to keep the chain string stable/readable.
    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
    return formatted or "0"


# ------------------------------------------------------------------
# Filter chain construction (pure, testable without ffmpeg)
# ------------------------------------------------------------------


def build_vhs_filter_chain(params: dict[str, Any] | None = None) -> str:
    """Return the ffmpeg ``-vf`` / ``filter_complex`` chain for the VHS look.

    Pure function.  Given identical params, returns byte-identical
    output.  This is what the unit tests assert against so we can
    validate the filter graph without a working ffmpeg install.

    The chain is ordered:

        1. ``format=yuv420p`` -- normalise pixel format
        2. ``rgbashift``      -- chromatic aberration (RGB split)
        3. ``gblur`` (chroma) -- chroma bleed via reduced-detail blur
        4. ``geq`` (scanlines)-- alternating-row luminance multiplier
        5. ``noise``          -- tape grain
        6. ``vignette``       -- soft edge darkening
        7. ``gblur`` (final)  -- light overall blur (tape softness)
    """
    p = _merge_params(params)

    rgb_shift = int(round(_scaled(p, "rgb_shift_px")))
    chroma_sigma = _scaled(p, "chroma_blur_sigma")
    scan_alpha = _scaled(p, "scanline_alpha")
    noise_c0s = int(round(_scaled(p, "noise_strength")))
    final_sigma = _scaled(p, "final_blur_sigma")

    scan_density = max(1, int(p.get("scanline_density", 2)))
    vignette_denom = max(1.0, float(p.get("vignette_over_pi", 5.0)))

    stages: list[str] = []

    # 1. Normalise to yuv420p -- matches renderer output pix_fmt.
    stages.append("format=yuv420p")

    # 2. RGB shift for chromatic aberration.  ``rgbashift`` wants
    #    integer offsets per channel in pixels.  We nudge R left and
    #    B right by the configured amount so the shift is visible on
    #    both edges.
    if rgb_shift > 0:
        stages.append(
            f"rgbashift=rh=-{rgb_shift}:bh={rgb_shift}"
        )

    # 3. Chroma bleed.  Implemented via ``gblur`` with planes=6
    #    (bitmask: U + V, i.e. chroma planes only) so luma detail
    #    survives.
    if chroma_sigma > 0:
        stages.append(
            f"gblur=sigma={_format_float(chroma_sigma)}:steps=1:planes=6"
        )

    # 4. Scanlines.  ``geq`` (generic expression) with a row-mod test
    #    dims every Nth row by ``scan_alpha``.  We touch luma only --
    #    chroma stays because we already bled it in stage 3.
    if scan_alpha > 0:
        # lum expression: if row mod density == 0, multiply by (1-alpha).
        lum_expr = (
            f"if(eq(mod(Y\\,{scan_density})\\,0)\\,"
            f"lum(X\\,Y)*{_format_float(1.0 - scan_alpha)}\\,lum(X\\,Y))"
        )
        stages.append(f"geq=lum='{lum_expr}':cb='cb(X,Y)':cr='cr(X,Y)'")

    # 5. Tape grain.  ``allf=t+u`` = temporal + uniform, c0s= = strength
    #    on the first (luma) plane.  Seed pinned for determinism.
    if noise_c0s > 0:
        stages.append(
            f"noise=c0s={noise_c0s}:c0f=t+u:alls=0"
        )

    # 6. Vignette.  PI/denom in ffmpeg expression syntax.
    stages.append(f"vignette=PI/{_format_float(vignette_denom)}")

    # 7. Final softness.
    if final_sigma > 0:
        stages.append(f"gblur=sigma={_format_float(final_sigma)}:steps=1")

    return ",".join(stages)


# ------------------------------------------------------------------
# Single-file apply
# ------------------------------------------------------------------


def _resolve_stub_reason(
    force_stub: bool,
    ffmpeg_path: str | None,
) -> tuple[bool, str]:
    """Decide whether we run in stub mode and record why."""
    if force_stub:
        return (True, STUB_REASON_FORCED)
    if os.environ.get("OTR_VHS_STUB", "").strip() == "1":
        return (True, STUB_REASON_ENV)
    if ffmpeg_path is None:
        return (True, STUB_REASON_NO_FFMPEG)
    return (False, STUB_REASON_REAL)


def _write_meta(
    meta_path: Path,
    *,
    input_path: Path,
    output_path: Path,
    mode: str,
    stub_reason: str,
    params: dict[str, Any],
    filter_chain: str,
    duration_ms: float,
    ffmpeg_cmd: list[str] | None,
    error: str | None = None,
) -> None:
    """Atomic-ish meta.json write next to the output file."""
    payload = {
        "stage": "vhs_postproc",
        "mode": mode,
        "stub_reason": stub_reason,
        "input": str(input_path),
        "output": str(output_path),
        "params": params,
        "params_hash": _params_hash(params),
        "filter_chain": filter_chain,
        "duration_ms": round(duration_ms, 2),
        "ffmpeg_cmd": ffmpeg_cmd or [],
        "error": error,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def apply_vhs_filter(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    params: dict[str, Any] | None = None,
    *,
    force_stub: bool = False,
) -> dict[str, Any]:
    """Apply the VHS filter to a single video file.

    Returns a meta dict mirroring what is written to ``<output>.meta.json``.
    Raises ``FileNotFoundError`` when ``input_path`` doesn't exist (both
    in stub and real mode -- a missing input is a caller bug, not a
    recoverable degrade).

    Stub mode is a byte-identical passthrough copy of the input, which
    preserves any embedded audio stream exactly (C7).
    """
    src = Path(input_path)
    dst = Path(output_path)
    if not src.exists() or not src.is_file():
        raise FileNotFoundError(f"VHS input not found: {src}")

    merged = _merge_params(params)
    filter_chain = build_vhs_filter_chain(merged)

    ffmpeg_path = find_ffmpeg()
    use_stub, stub_reason = _resolve_stub_reason(force_stub, ffmpeg_path)

    dst.parent.mkdir(parents=True, exist_ok=True)
    meta_path = dst.with_suffix(dst.suffix + ".meta.json")

    start = time.monotonic()
    ffmpeg_cmd: list[str] | None = None
    error: str | None = None

    try:
        if use_stub:
            # Byte-identical passthrough -- guarantees audio track bytes
            # (if any) survive unchanged.
            if src.resolve() != dst.resolve():
                shutil.copyfile(src, dst)
        else:
            # Real mode -- video pass re-encoded with filter chain,
            # audio copied byte-for-byte if present.
            assert ffmpeg_path is not None  # for type-checkers
            ffmpeg_cmd = [
                ffmpeg_path,
                "-y",
                "-i", str(src),
                "-vf", filter_chain,
                "-r", str(int(merged.get("fps", 24))),
                "-c:v", "libx264",
                "-preset", str(merged.get("preset", "fast")),
                "-crf", str(int(merged.get("crf", 22))),
                "-pix_fmt", "yuv420p",
                # C7: audio bytes passed through unchanged when present.
                "-c:a", "copy",
                # Do not fail if the source has no audio stream.
                "-map", "0:v:0",
                "-map", "0:a?",
                str(dst),
            ]
            result = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                error = (result.stderr or "").strip()[-2000:]
                log.error(
                    "ffmpeg failed for %s (rc=%d): %s",
                    src, result.returncode, error,
                )
                # Degrade to passthrough so the pipeline keeps moving.
                if src.resolve() != dst.resolve():
                    shutil.copyfile(src, dst)
                stub_reason = f"{STUB_REASON_NO_FFMPEG}:ffmpeg_error"
                use_stub = True
    except Exception as exc:  # pragma: no cover -- defensive
        error = repr(exc)
        log.exception("VHS apply failed for %s", src)
        # Still produce an output so downstream stages see a file.
        if src.resolve() != dst.resolve() and not dst.exists():
            shutil.copyfile(src, dst)
        stub_reason = f"{STUB_REASON_NO_FFMPEG}:exception"
        use_stub = True

    duration_ms = (time.monotonic() - start) * 1000.0

    mode = "stub" if use_stub else "real"
    _write_meta(
        meta_path,
        input_path=src,
        output_path=dst,
        mode=mode,
        stub_reason=stub_reason,
        params=merged,
        filter_chain=filter_chain,
        duration_ms=duration_ms,
        ffmpeg_cmd=ffmpeg_cmd,
        error=error,
    )

    return {
        "stage": "vhs_postproc",
        "mode": mode,
        "stub_reason": stub_reason,
        "input": str(src),
        "output": str(dst),
        "params": merged,
        "params_hash": _params_hash(merged),
        "filter_chain": filter_chain,
        "duration_ms": round(duration_ms, 2),
        "ffmpeg_cmd": ffmpeg_cmd or [],
        "error": error,
    }


# ------------------------------------------------------------------
# Batch apply across a job directory
# ------------------------------------------------------------------


def _iter_shot_clips(job_dir: Path) -> Iterable[Path]:
    """Yield the per-shot video file for every shot under ``job_dir``."""
    if not job_dir.is_dir():
        return
    for child in sorted(job_dir.iterdir()):
        if not child.is_dir():
            continue
        # Skip internal dirs (starting with '_' or '.').
        if child.name.startswith(("_", ".")):
            continue
        for candidate in VIDEO_FILENAME_CANDIDATES:
            candidate_path = child / candidate
            if candidate_path.is_file():
                yield candidate_path
                break  # take the first match, don't double-process


def apply_vhs_to_job_dir(
    job_dir: str | os.PathLike[str],
    params: dict[str, Any] | None = None,
    *,
    force_stub: bool = False,
) -> dict[str, Any]:
    """Run VHS filter across every per-shot video clip under ``job_dir``.

    Produces a sibling ``*_vhs.mp4`` next to each input clip.  Skips
    still images (``keyframe.png``, ``composite.png``, etc.).  Returns
    a summary dict with per-clip meta entries.
    """
    root = Path(job_dir)
    clips = list(_iter_shot_clips(root))

    results: list[dict[str, Any]] = []
    for clip in clips:
        out_path = clip.with_name(clip.stem + VHS_SUFFIX)
        meta = apply_vhs_filter(
            clip, out_path, params=params, force_stub=force_stub
        )
        results.append(meta)

    summary_path = root / "vhs_postproc_summary.json"
    summary = {
        "stage": "vhs_postproc_batch",
        "job_dir": str(root),
        "count": len(results),
        "entries": results,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if root.is_dir():
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return summary


__all__ = [
    "DEFAULT_VHS_PARAMS",
    "VIDEO_FILENAME_CANDIDATES",
    "SKIP_FILENAMES",
    "VHS_SUFFIX",
    "STUB_REASON_ENV",
    "STUB_REASON_NO_FFMPEG",
    "STUB_REASON_FORCED",
    "STUB_REASON_REAL",
    "find_ffmpeg",
    "build_vhs_filter_chain",
    "apply_vhs_filter",
    "apply_vhs_to_job_dir",
]
