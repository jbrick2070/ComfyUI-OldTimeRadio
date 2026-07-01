"""Per-beat CONTENT ORACLE for the all-engines x all-slots soak (C5, 2026-06-30).

The soak proves every engine renders REAL content in every slot -- not just that
a clip file exists. This module is the pure, dependency-free verdict layer it
calls on each per-beat MANIFEST clip (``clips[].path`` -- never the obs final):

  * LUMA floor (EVERY beat): ffmpeg ``signalstats`` mean YAVG must clear a NAMED
    floor (default 16/255). A dark/black clip is the D2 BLACK-floor symptom this
    whole sprint kills, so it is a HARD failure for every engine.
  * MOTION (motion engines only): over a bounded window, ffmpeg ``freezedetect``
    must NOT report the whole clip frozen. STATIC engines (still_flat / the
    static_image_gen + abstract families) are EXEMPT -- a held still is correct
    for them, not a failure ("static is OK for static engines").

Engine -> motion-required is decided by the engine's FAMILY (read from the live
video registry, with a static fallback table so this stays importable without the
registry). Cold-import clean: only ``os`` / ``subprocess`` / ``json`` at module
scope; ffmpeg is shelled out lazily inside the check. UTF-8, no BOM, SFW.
"""
from __future__ import annotations

import json
import os
import subprocess
from typing import Optional

#: Engine FAMILIES that must show temporal motion (the genuine moving-video
#: lanes). Everything else -- static_image_gen (still_pan / still_flat),
#: static_motion (still_motion / still_parallax: a slow pan that freezedetect can
#: false-flag), abstract (visualizer) -- is motion-EXEMPT: only the luma floor
#: applies. (Operator: "static is OK for static engines, not a failure.")
MOTION_FAMILIES: frozenset = frozenset({
    "image_to_video",
    "text_to_video",
    "audio_driven_face",
    "audio_conditioned_video",
})

#: Fallback engine_id -> family for the common engines, used ONLY when the live
#: video registry is not importable (keeps the oracle usable from a bare soak
#: script). The registry is the source of truth when present.
_FAMILY_FALLBACK: dict = {
    "humo": "audio_driven_face",
    "humo_1.7B": "audio_driven_face",
    "humo_1.7B_169": "audio_driven_face",
    "humo_14B_169": "audio_driven_face",
    "ltx_video": "text_to_video",
    "ltx_audio_in": "audio_conditioned_video",
    "wan_i2v": "image_to_video",
    "wan_ti2v": "image_to_video",
    "mesh_stage": "image_to_video",
    "still_parallax": "static_motion",
    "still_motion": "static_motion",
    "still_pan": "static_image_gen",
    "still_flat": "static_image_gen",
    "visualizer": "abstract",
    "viz_mxc_cpu": "abstract",       # OTR rainbow visualizer -- motion-exempt (2026-06-30)
    "viz_mxc_mandala": "abstract",   # Cosmic Radio Mandala -- motion-exempt (2026-06-30)
}

#: Default mean-luma floor (0..255). A clip whose mean YAVG is at/below this reads
#: as a dark/black floor -- the D2 symptom -- and FAILS for every engine.
DEFAULT_LUMA_FLOOR = 16.0


class ContentOracleError(AssertionError):
    """A per-beat clip failed the content oracle (dark floor, or a motion engine
    produced a frozen clip). Raised LOUD so the soak scores the leg RED."""


def _ffmpeg(ffmpeg: Optional[str] = None) -> str:
    return ffmpeg or os.environ.get("OTR_FFMPEG", "ffmpeg")


def family_for_engine(engine_id: str) -> str:
    """The engine's family: the live video registry first, then the fallback
    table, then '' (unknown). Pure attr read; never raises."""
    try:
        from nodes._otr_video_engines import registry as _vreg  # lazy
        if _vreg.is_registered(engine_id):
            return str(getattr(_vreg.get_engine(engine_id), "family", "") or "")
    except Exception:  # noqa: BLE001 -- registry not importable from a bare script
        pass
    return _FAMILY_FALLBACK.get(engine_id, "")


def motion_required_for_engine(engine_id: str) -> bool:
    """True iff a clip from ``engine_id`` must show temporal motion (its family is
    a genuine moving-video lane). Static engines are motion-exempt."""
    return family_for_engine(engine_id) in MOTION_FAMILIES


def mean_yavg(clip_path: str, *, ffmpeg: Optional[str] = None,
              frames: int = 0) -> float:
    """Mean luma (YAVG, 0..255) over the clip via ffmpeg ``signalstats``.

    Reads ``lavfi.signalstats.YAVG`` from ``metadata=print`` and averages it
    across the analyzed frames. ``frames`` > 0 caps the window (faster). Raises
    ``ContentOracleError`` if ffmpeg yields no YAVG samples (unreadable clip)."""
    # NB: the metadata `print` filter writes YAVG to the ffmpeg log at INFO level,
    # so `-v info` is REQUIRED (under `-v error` the samples are suppressed).
    vf = "signalstats,metadata=mode=print:key=lavfi.signalstats.YAVG"
    cmd = [_ffmpeg(ffmpeg), "-v", "info", "-i", clip_path]
    if frames and frames > 0:
        cmd += ["-frames:v", str(int(frames))]
    cmd += ["-vf", vf, "-f", "null", "-"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    vals = []
    for line in (proc.stderr or "").splitlines():
        marker = "lavfi.signalstats.YAVG="
        idx = line.find(marker)
        if idx >= 0:
            try:
                vals.append(float(line[idx + len(marker):].strip()))
            except ValueError:
                continue
    if not vals:
        raise ContentOracleError(
            f"content oracle: no YAVG samples from {clip_path!r} "
            f"(unreadable clip or ffmpeg error: {(proc.stderr or '').strip()[:200]})")
    return sum(vals) / len(vals)


def is_frozen(clip_path: str, *, ffmpeg: Optional[str] = None,
              noise_db: float = -60.0, freeze_seconds: float = 0.5) -> bool:
    """True iff ffmpeg ``freezedetect`` reports a freeze spanning (effectively)
    the whole clip -- i.e. NO temporal motion. Tuned permissive (a long
    ``freeze_seconds`` + a low noise floor) so a real moving clip never trips."""
    vf = (f"freezedetect=n=-{abs(noise_db)}dB:d={float(freeze_seconds)}"
          if noise_db < 0 else
          f"freezedetect=n={float(noise_db)}:d={float(freeze_seconds)}")
    cmd = [_ffmpeg(ffmpeg), "-v", "info", "-i", clip_path,
           "-vf", vf, "-f", "null", "-"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return "freeze_start" in (proc.stderr or "")


def check_clip(clip_path: str, engine_id: str, *, luma_floor: float = DEFAULT_LUMA_FLOOR,
               ffmpeg: Optional[str] = None, frames: int = 0) -> dict:
    """Run the oracle on ONE per-beat clip. Returns a verdict dict
    ``{engine_id, family, mean_yavg, motion_required, frozen, ok, reason}``;
    raises :class:`ContentOracleError` on failure (dark floor, or a motion engine
    that produced a frozen clip). ``frames`` caps the analyzed window."""
    if not clip_path or not os.path.exists(clip_path):
        raise ContentOracleError(
            f"content oracle: clip for engine {engine_id!r} missing at {clip_path!r}")
    fam = family_for_engine(engine_id)
    motion_req = fam in MOTION_FAMILIES
    yavg = mean_yavg(clip_path, ffmpeg=ffmpeg, frames=frames)
    verdict = {
        "engine_id": engine_id, "family": fam, "mean_yavg": yavg,
        "motion_required": motion_req, "frozen": None, "ok": True, "reason": "ok",
    }
    if yavg <= luma_floor:
        verdict.update(ok=False, reason="dark_floor")
        raise ContentOracleError(
            f"content oracle: engine {engine_id!r} clip is DARK "
            f"(mean YAVG {yavg:.1f} <= floor {luma_floor}) -- the D2 black-floor "
            f"symptom: {clip_path!r}")
    if motion_req:
        frozen = is_frozen(clip_path, ffmpeg=ffmpeg)
        verdict["frozen"] = frozen
        if frozen:
            verdict.update(ok=False, reason="frozen_motion_engine")
            raise ContentOracleError(
                f"content oracle: motion engine {engine_id!r} (family {fam!r}) "
                f"produced a FROZEN clip (no temporal motion): {clip_path!r}")
    return verdict


def check_manifest(manifest: dict, *, luma_floor: float = DEFAULT_LUMA_FLOOR,
                   ffmpeg: Optional[str] = None, frames: int = 0) -> list:
    """Run the oracle over every per-beat clip in a render manifest
    (``manifest['clips']`` with ``path`` + ``engine_id`` rows). Skips rows with
    no path / 0 frames (a legitimately empty beat). Returns the verdict list;
    raises on the first failing clip."""
    clips = (manifest or {}).get("clips") or []
    verdicts = []
    for row in clips:
        path = row.get("path") or ""
        if not path:
            continue
        if int(row.get("frame_count") or row.get("target_frame_count") or 0) <= 0 \
                and not os.path.exists(path):
            continue
        engine_id = str(row.get("engine_id") or row.get("delivered_engine") or "")
        verdicts.append(check_clip(path, engine_id, luma_floor=luma_floor,
                                   ffmpeg=ffmpeg, frames=frames))
    return verdicts


def load_manifest(path: str) -> dict:
    """Load a render manifest JSON (node_episode_manifest.json / the
    clip_manifest_json string dumped to disk)."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
