"""
otr_v2.hyworld.wall_clock  --  per-backend wall-clock estimator (Day 11)
========================================================================

Given a PlannerResult (or an iterable of PlannerJob-like dicts), estimate
projected wall-clock time for the full sidecar run in both stub mode and
real mode.  Used by the 3-min continuous scene gate to assert the Day 11
ROADMAP bar: "wall clock < 45 min" for a 3-min of final episode content.

Per-backend estimates are point estimates based on the platform pins
(RTX 5080 Laptop, 16 GB VRAM, Blackwell sm_120, torch 2.10.0, CUDA 13.0,
FP8 e4m3fn, SageAttention + SDPA, no Flash Attention) and the stack
selection locked in ROADMAP.md Stage 1-7.  These numbers are intended to
be *conservative upper bounds* for a single shot, not medians; the gate
is meant to catch catastrophic regressions (OOM loops, FA chasing that
fell back to eager attention), not to be a precise render-time predictor.

No torch / diffusers / cuda imports.  Pure stdlib.  Safe to import from
unit tests and from the planner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

# ------------------------------------------------------------------
# Per-backend wall-clock point estimates (seconds per shot)
# ------------------------------------------------------------------

# Stub mode: sidecar stub writes a single solid-color PNG or a tiny MP4
# marker.  These are bounded by disk I/O; we assume ~100 ms per shot
# with headroom for Windows NTFS + Python overhead.
STUB_WALL_CLOCK_S: dict[str, float] = {
    "placeholder_test":     0.05,
    "flux_anchor":          0.10,
    "pulid_portrait":       0.10,
    "flux_keyframe":        0.10,
    "ltx_motion":           0.10,
    "wan21_loop":           0.10,
    "florence2_sdxl_comp":  0.10,
}

# Real mode: measured / projected times for a 1024-ish frame on a warm
# pipeline (cold-load first-shot penalty is tracked separately).  Numbers
# are conservative upper bounds, not medians.
REAL_WALL_CLOCK_S: dict[str, float] = {
    "placeholder_test":      0.1,
    # Stage 1: FLUX.1-dev FP8 + ControlNet Union Pro 2.0, 1024x1024,
    # 20 steps.  FP8 e4m3fn on Blackwell with SageAttention is the fast
    # path; eager attention would be ~2x slower and that's what this
    # estimate catches.
    "flux_anchor":          28.0,
    # Stage 3: FLUX + PuLID identity adapter.  Same base cost as anchor
    # + the adapter forward pass.
    "pulid_portrait":       32.0,
    # Stage 2: FLUX + Depth/Canny ControlNet on a pre-computed depth
    # map.  The depth estimate is cached per-shot (Day 4), so only the
    # diffusion step counts here.
    "flux_keyframe":        25.0,
    # Stage 4: LTX-Video 2.3, 257 frames @ 24 fps, 10 s clip.  FP8 e4m3fn
    # path, SageAttention enabled.  Cold-start penalty (~60 s first call)
    # handled separately.
    "ltx_motion":           95.0,
    # Stage 5: Wan2.1 1.3B I2V, 10 s loop at 24 fps.  Smaller model than
    # LTX; offload keeps peak VRAM under 10 GB.
    "wan21_loop":           65.0,
    # Stage 6: Florence-2 mask + SDXL inpaint, single 1024x1024 pass.
    # Florence is small and fast; SDXL at 20 steps is the dominant cost.
    "florence2_sdxl_comp":  18.0,
}

# One-time pipeline-warmup penalty per backend (weight load + first-shot
# graph capture).  This hits once per sidecar process; if the sidecar is
# held warm between shots of the same backend, subsequent shots only pay
# the per-shot cost above.
COLD_LOAD_PENALTY_S: dict[str, float] = {
    "placeholder_test":     0.0,
    "flux_anchor":         45.0,
    "pulid_portrait":      50.0,
    "flux_keyframe":       40.0,
    "ltx_motion":          70.0,
    "wan21_loop":          30.0,
    "florence2_sdxl_comp": 25.0,
}

# VHS post-processor: ffmpeg filter chain across every motion+loop clip.
# Real-mode ffmpeg with the Day 8 filter chain runs ~0.5x clip length.
REAL_VHS_PER_CLIP_S: float = 5.0
STUB_VHS_PER_CLIP_S: float = 0.02

# Day 11 ROADMAP bar: a 3-minute continuous scene must render in under
# 45 minutes of wall clock, including all cold-loads and VHS postproc.
DAY_11_WALL_CLOCK_CEILING_S: float = 45 * 60.0  # 2700 s

# Day 11 ROADMAP sub-bar: stub-mode 3-min scene must complete in well
# under a minute -- this is a CI-safety floor, not a rendering bar.
DAY_11_STUB_CEILING_S: float = 60.0


# ------------------------------------------------------------------
# Estimator
# ------------------------------------------------------------------


@dataclass
class WallClockEstimate:
    """Projected wall-clock breakdown for a planned episode."""
    mode: str                           # "stub" or "real"
    total_s: float                      # full projected wall clock
    render_s: float                     # per-shot render total (no cold-load, no vhs)
    cold_load_s: float                  # one-time warmup penalties summed
    vhs_s: float                        # post-processor cost
    per_backend_s: dict[str, float] = field(default_factory=dict)
    per_backend_shots: dict[str, int] = field(default_factory=dict)
    unknown_backends: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "total_s": round(self.total_s, 3),
            "total_minutes": round(self.total_s / 60.0, 2),
            "render_s": round(self.render_s, 3),
            "cold_load_s": round(self.cold_load_s, 3),
            "vhs_s": round(self.vhs_s, 3),
            "per_backend_s": {
                k: round(v, 3) for k, v in self.per_backend_s.items()
            },
            "per_backend_shots": dict(self.per_backend_shots),
            "unknown_backends": list(self.unknown_backends),
        }


def _job_backends(jobs: Iterable[Any]) -> list[str]:
    """Pull backend names out of a PlannerJob-or-dict iterable.

    Accepts either dataclass instances with a ``.backend`` attribute or
    plain dicts with a ``"backend"`` key.  Unknown types are skipped
    silently to keep the estimator robust to test fixtures.
    """
    names: list[str] = []
    for j in jobs:
        name = getattr(j, "backend", None)
        if name is None and isinstance(j, dict):
            name = j.get("backend")
        if name:
            names.append(str(name))
    return names


def _video_clip_count(jobs: Iterable[Any]) -> int:
    """Count motion + loop jobs -- those feed the VHS post-processor."""
    n = 0
    for name in _job_backends(jobs):
        if name in ("ltx_motion", "wan21_loop"):
            n += 1
    return n


def estimate(
    jobs: Iterable[Any],
    *,
    mode: str = "real",
    include_vhs: bool = True,
    include_cold_load: bool = True,
) -> WallClockEstimate:
    """Project wall clock for an iterable of PlannerJob-like entries.

    Parameters
    ----------
    jobs : iterable of PlannerJob or dict
        The planned sidecar jobs.  Order does not affect the estimate.
    mode : "real" | "stub"
        Which per-backend table to use.
    include_vhs : bool
        Add VHS post-processor cost for every motion + loop clip.
    include_cold_load : bool
        Add one-time per-backend warmup penalties.  Each backend that
        appears at least once contributes its cold-load cost.

    Returns
    -------
    WallClockEstimate
        Dataclass with total_s plus a breakdown.  Unknown backend names
        land in ``unknown_backends`` and contribute 0 to the estimate.
    """
    mode = (mode or "real").strip().lower()
    if mode not in ("real", "stub"):
        raise ValueError(f"mode must be 'real' or 'stub', got {mode!r}")

    table = STUB_WALL_CLOCK_S if mode == "stub" else REAL_WALL_CLOCK_S
    vhs_per_clip = STUB_VHS_PER_CLIP_S if mode == "stub" else REAL_VHS_PER_CLIP_S

    names = _job_backends(jobs)

    per_backend_s: dict[str, float] = {}
    per_backend_shots: dict[str, int] = {}
    unknown: list[str] = []
    render_s = 0.0

    for name in names:
        per_shot = table.get(name)
        if per_shot is None:
            if name not in unknown:
                unknown.append(name)
            continue
        per_backend_shots[name] = per_backend_shots.get(name, 0) + 1
        per_backend_s[name] = per_backend_s.get(name, 0.0) + per_shot
        render_s += per_shot

    cold_load_s = 0.0
    if include_cold_load and mode == "real":
        # Only real mode pays cold-load; stubs don't load weights.
        for name in per_backend_shots:
            cold_load_s += COLD_LOAD_PENALTY_S.get(name, 0.0)

    vhs_s = 0.0
    if include_vhs:
        clip_count = _video_clip_count(jobs)
        vhs_s = clip_count * vhs_per_clip

    total_s = render_s + cold_load_s + vhs_s

    return WallClockEstimate(
        mode=mode,
        total_s=total_s,
        render_s=render_s,
        cold_load_s=cold_load_s,
        vhs_s=vhs_s,
        per_backend_s=per_backend_s,
        per_backend_shots=per_backend_shots,
        unknown_backends=unknown,
    )


__all__ = [
    "STUB_WALL_CLOCK_S",
    "REAL_WALL_CLOCK_S",
    "COLD_LOAD_PENALTY_S",
    "REAL_VHS_PER_CLIP_S",
    "STUB_VHS_PER_CLIP_S",
    "DAY_11_WALL_CLOCK_CEILING_S",
    "DAY_11_STUB_CEILING_S",
    "WallClockEstimate",
    "estimate",
]
