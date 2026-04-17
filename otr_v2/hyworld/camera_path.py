"""
camera_path.py  --  Deterministic camera trajectory engine
==========================================================

Narrative-first runtime contract: given a camera adjective (from
``shotlist.py``), a target duration, an fps, and a seed, produce a
reproducible per-frame ``(zoom, cx, cy)`` trajectory.  Same inputs -->
byte-identical pose stream --> byte-identical ffmpeg filter chain.

Why this module exists
----------------------
Before Phase C, camera motion was scattered ffmpeg zoompan expressions
inside ``worker.py``.  The active 2026-04-16 spec ("image-to-splat +
headless renderer") needs the same camera-path abstraction to feed
gsplat / SplatFusion as a future emit backend, so we extract a named,
testable subsystem now -- regardless of which Phase C render stack ships.

Design decisions (from 2026-04-16 round-robin consult, stack pick #4):
  * Pose is `(zoom, cx, cy)` with `cx/cy` in [0, 1] where 0.5 is centered.
    `cx=0.0` means the crop is anchored to the left edge of the input,
    `cx=1.0` to the right edge.  This matches ffmpeg zoompan's
    `x=(iw-iw/zoom)*cx` parametrization and is stack-agnostic enough to
    feed a splat fly-through camera later (cx/cy become yaw/pitch or
    lateral offsets there).
  * Easing curves are analytic (no Python RNG, no handheld jitter in
    v1 -- deferred per gpt-4.1 "simpler baseline" advice).  Handheld
    drift is a v1.1 add-on behind the ``seed`` parameter.
  * Frame count is exact: ``round(duration_sec * fps)``.  No off-by-one
    drift across long durations (gpt-4.1 "determinism is broader than
    camera math" note).
  * Catalog is the existing ``shotlist.py`` phrase list.  First
    substring match wins (matches the pre-Phase-C behavior in
    ``worker._camera_to_motion`` so existing episodes re-render
    identically).

Determinism contract
--------------------
  * No wall-clock time.  No unseeded RNG.  No network.  No torch.
  * Same ``(adjective, duration_sec, fps, seed)`` => same pose list
    => same SHA-256 of the quantized pose stream.
  * ``hash_trajectory()`` quantizes to 10 decimal places so a 1-ULP
    ``cos()`` difference across libm builds does not crack the test.
  * Tests in ``tests/test_camera_path_determinism.py`` lock this in.

Emit backends
-------------
  * ``sample(traj) -> List[Pose]``        -- canonical Python poses
  * ``to_zoompan_filter(traj, w, h)``     -- ffmpeg MVP (ships today)
  * (future) splat camera path emitter   -- gsplat / SplatFusion adapter
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

Pose = Tuple[float, float, float]  # (zoom, cx_norm, cy_norm)


@dataclass(frozen=True)
class Trajectory:
    """Canonical camera trajectory for a single shot.

    All fields are data; the engine is pure.  Frozen so a Trajectory can
    be used as a cache key and so no downstream accidentally mutates
    the pose stream.
    """

    label: str           # resolved catalog key (e.g. "clean push")
    duration_sec: float
    fps: int
    zoom_start: float
    zoom_end: float
    cx_start: float      # normalized [0,1], 0.5 = centered
    cx_end: float
    cy_start: float
    cy_end: float
    easing: str          # "linear" | "ease_in_out" | "ease_out"
    seed: int = 0        # reserved for v1.1 handheld jitter

    @property
    def frame_count(self) -> int:
        """Exact frame count for this duration.  min 1."""
        return max(1, int(round(self.duration_sec * self.fps)))


# ---------------------------------------------------------------------------
# Easing primitives (pure functions over u in [0,1])
# ---------------------------------------------------------------------------

def _linear(u: float) -> float:
    return u


def _ease_in_out(u: float) -> float:
    # Cosine smoothstep.  C1-continuous start and end; the consult's
    # recommended "simple easing baseline".
    return 0.5 - 0.5 * math.cos(math.pi * u)


def _ease_out(u: float) -> float:
    # Cubic ease-out.  Sharp start, gentle landing -- reads as "dolly
    # slowing into frame".
    return 1.0 - (1.0 - u) ** 3


_EASINGS = {
    "linear": _linear,
    "ease_in_out": _ease_in_out,
    "ease_out": _ease_out,
}


# ---------------------------------------------------------------------------
# Adjective catalog
# ---------------------------------------------------------------------------
# Keys are lowercase substrings.  First match wins (preserves the
# pre-Phase-C behavior in worker._camera_to_motion).  Each entry:
#   zoom   = (start, end)  zoom factors, end >= 1.0
#   cx     = (start, end)  normalized crop center x in [0,1]
#   cy     = (start, end)  normalized crop center y in [0,1]
#   easing = one of _EASINGS keys
#
# The values here were ported from worker.py Phase A so existing
# episodes re-render visually identical motion.  Tweaks go here, not
# in worker.py.

_ADJECTIVE_CATALOG: List[Tuple[str, dict]] = [
    ("locked off",    {"zoom": (1.00, 1.00), "cx": (0.50, 0.50), "cy": (0.50, 0.50), "easing": "linear"}),
    ("clean push",    {"zoom": (1.00, 1.40), "cx": (0.50, 0.50), "cy": (0.50, 0.50), "easing": "ease_in_out"}),
    ("slow handheld", {"zoom": (1.00, 1.30), "cx": (0.50, 0.50), "cy": (0.50, 0.50), "easing": "linear"}),
    ("fast dolly",    {"zoom": (1.00, 1.55), "cx": (0.45, 0.45), "cy": (0.55, 0.55), "easing": "ease_out"}),
    ("whip-pan",      {"zoom": (1.30, 1.30), "cx": (0.00, 1.00), "cy": (0.50, 0.50), "easing": "ease_in_out"}),
    ("low angle",     {"zoom": (1.25, 1.25), "cx": (0.50, 0.50), "cy": (1.00, 0.00), "easing": "linear"}),
    ("macro detail",  {"zoom": (1.00, 1.15), "cx": (0.50, 0.50), "cy": (0.50, 0.50), "easing": "ease_in_out"}),
    ("slow drift",    {"zoom": (1.00, 1.20), "cx": (0.40, 0.60), "cy": (0.50, 0.50), "easing": "linear"}),
]

_DEFAULT_KEY = "slow drift"


def known_adjectives() -> List[str]:
    """All catalog keys in order.  Used by tests and shotlist validators."""
    return [k for (k, _) in _ADJECTIVE_CATALOG]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def from_adjective(
    adjective: str,
    duration_sec: float,
    fps: int = 24,
    seed: int = 0,
) -> Trajectory:
    """Resolve (adjective, duration, fps, seed) to a deterministic Trajectory.

    Raises ValueError on non-positive duration or fps.  Unknown
    adjective falls back to the default catalog entry (slow drift)
    -- preserves the pre-Phase-C "never fail, just drift" behavior.
    """
    if duration_sec <= 0:
        raise ValueError(f"duration_sec must be > 0 (got {duration_sec})")
    if fps <= 0:
        raise ValueError(f"fps must be > 0 (got {fps})")

    adj_lower = (adjective or "").lower()

    # Pull the default spec up front so "no-match" is explicit.
    default_spec = next(s for (k, s) in _ADJECTIVE_CATALOG if k == _DEFAULT_KEY)

    label = _DEFAULT_KEY
    spec = default_spec
    for needle, candidate_spec in _ADJECTIVE_CATALOG:
        if needle in adj_lower:
            label = needle
            spec = candidate_spec
            break

    return Trajectory(
        label=label,
        duration_sec=float(duration_sec),
        fps=int(fps),
        zoom_start=spec["zoom"][0],
        zoom_end=spec["zoom"][1],
        cx_start=spec["cx"][0],
        cx_end=spec["cx"][1],
        cy_start=spec["cy"][0],
        cy_end=spec["cy"][1],
        easing=spec["easing"],
        seed=int(seed),
    )


def sample(traj: Trajectory) -> List[Pose]:
    """Produce the canonical per-frame pose stream.

    Returns exactly ``traj.frame_count`` poses.  Frame 0 is the start
    pose, the final frame is the end pose.  For frame_count == 1 the
    single pose is the start pose (u = 0).
    """
    n = traj.frame_count
    easing = _EASINGS.get(traj.easing, _linear)
    poses: List[Pose] = []
    for i in range(n):
        u = 0.0 if n == 1 else i / (n - 1)
        e = easing(u)
        z = traj.zoom_start + (traj.zoom_end - traj.zoom_start) * e
        cx = traj.cx_start + (traj.cx_end - traj.cx_start) * e
        cy = traj.cy_start + (traj.cy_end - traj.cy_start) * e
        poses.append((z, cx, cy))
    return poses


def hash_trajectory(traj: Trajectory) -> str:
    """SHA-256 of the quantized sampled pose stream.

    Used by the determinism regression test.  Quantizes floats to 10
    decimal places so a 1-ULP ``cos()`` drift across libm builds cannot
    crack the hash.  10 dp is well below any perceptible camera motion
    difference (sub-pixel on a 1920-wide input).
    """
    poses = sample(traj)
    lines = [f"{z:.10f},{cx:.10f},{cy:.10f}" for (z, cx, cy) in poses]
    blob = "\n".join(lines).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


# ---------------------------------------------------------------------------
# ffmpeg zoompan emit backend
# ---------------------------------------------------------------------------

def to_zoompan_filter(
    traj: Trajectory,
    width: int,
    height: int,
) -> str:
    """Emit an ffmpeg zoompan filter chain for this trajectory.

    Produces the same behavior as ``sample()`` to within the precision
    of ffmpeg's expression evaluator.  zoompan driver notes:

      * ``z`` is the zoom factor, >= 1.0
      * ``x``, ``y`` are the top-left of the crop in input-pixel space
      * ``on`` is the current output frame index (0..d-1)
      * ``d`` is the total animation length in output frames
      * ``s=WxH`` is the output resolution

    Center parametrization: for normalized center ``(cx, cy)`` in
    [0, 1] where 0.5 is centered,

        x = (iw - iw/zoom) * cx
        y = (ih - ih/zoom) * cy

    Feeds through ``-loop 1 -framerate 1 -t 1 -frames:v d`` in the
    caller so zoompan's frame multiplier does not blow up the duration
    (see BUG-LOCAL-014 guard in ``worker._make_motion_clip``).
    """
    d = traj.frame_count
    zs, ze = traj.zoom_start, traj.zoom_end
    cx0, cx1 = traj.cx_start, traj.cx_end
    cy0, cy1 = traj.cy_start, traj.cy_end

    # Normalized progress expression.  For d == 1 we collapse to 0 to
    # avoid a zero-divide in the zoompan evaluator.
    u_expr = "0" if d == 1 else f"(on/{d - 1})"

    # Easing expression in zoompan's math dialect (PI, cos, pow all ok).
    if traj.easing == "linear":
        e_expr = u_expr
    elif traj.easing == "ease_in_out":
        e_expr = f"(0.5-0.5*cos(PI*{u_expr}))"
    elif traj.easing == "ease_out":
        e_expr = f"(1-pow(1-{u_expr},3))"
    else:
        e_expr = u_expr

    # Zoom expression.  If start == end we emit a scalar -- keeps the
    # filter string short and side-steps any FP rounding in the delta.
    if abs(ze - zs) < 1e-12:
        z_expr = f"{zs}"
    else:
        z_expr = f"({zs}+({ze - zs})*{e_expr})"

    cx_expr = f"({cx0}+({cx1 - cx0})*{e_expr})"
    cy_expr = f"({cy0}+({cy1 - cy0})*{e_expr})"

    x_expr = f"(iw-iw/zoom)*{cx_expr}"
    y_expr = f"(ih-ih/zoom)*{cy_expr}"

    return (
        f"zoompan=z='{z_expr}':"
        f"x='{x_expr}':y='{y_expr}':"
        f"d={d}:s={width}x{height}:fps={traj.fps}"
    )


# ---------------------------------------------------------------------------
# Worker-facing convenience wrapper
# ---------------------------------------------------------------------------

def zoompan_for_shot(
    camera: str,
    duration_sec: float,
    width: int,
    height: int,
    fps: int = 24,
    seed: int = 0,
) -> Tuple[str, str]:
    """One-call path for worker.py: camera adjective -> (label, filter_chain).

    Keeps the existing ``_camera_to_motion`` return shape so the
    worker.py swap is a single-line substitution.
    """
    traj = from_adjective(camera, duration_sec, fps=fps, seed=seed)
    return (traj.label, to_zoompan_filter(traj, width, height))
