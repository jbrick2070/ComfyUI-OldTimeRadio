"""LTX-AV frame/dimension validator (NEW shared helper; LTX-AV lane only).

The LTX-2.3 audio-conditioned lane (``eng_ltx_av``) needs its OWN dimension rule
that DELIBERATELY diverges from the golden prompt-only ``eng_ltx_video``: the AV
lane snaps the frame count UP to the next valid ``8n+1`` (so audio is never
truncated), whereas ``eng_ltx_video._ltx_frame_length`` snaps DOWN. The two lanes
must never share this math -- copying one into the other silently desyncs a lane.

LTX latents are 8x temporally compressed + a single leading frame, so a valid
frame count is ``8n+1`` (9, 17, 25, ...); spatial dims must be multiples of 32.
Upstream ComfyUI/LTXVideo silently ROUNDS bad dims; OTR fails LOUD instead
(Bug Bible row at ship). ``next_8n1`` snaps UP; ``assert_ltx_dims`` RAISES with a
message that names the nearest valid value in BOTH directions (the message format
is pinned for tests).

Cold-import clean (V-12): stdlib only, no torch/comfy. UTF-8, no BOM, ASCII-only.
"""
from __future__ import annotations

#: LTX spatial stride (W and H must both be multiples of this).
_LTX_SPATIAL_MULTIPLE = 32
#: LTX temporal rule: frames == 8n + 1.
_LTX_TEMPORAL_BASE = 8
#: Smallest valid frame count grounded for the 2.3 graph (one base latent frame).
_LTX_MIN_FRAMES = 9


def next_8n1(n: int) -> int:
    """Smallest ``8k+1`` that is >= ``n`` (snap UP), floored at ``_LTX_MIN_FRAMES``.

    DELIBERATELY snaps UP -- never reuse ``eng_ltx_video``'s snap-DOWN formula.
    ``next_8n1(9)==9``, ``next_8n1(10)==17``, ``next_8n1(25)==25``,
    ``next_8n1(26)==33``.
    """
    n = int(n)
    if n <= _LTX_MIN_FRAMES:
        return _LTX_MIN_FRAMES
    return ((n + (_LTX_TEMPORAL_BASE - 2)) // _LTX_TEMPORAL_BASE) * _LTX_TEMPORAL_BASE + 1


def _nearest_multiples(value: int, multiple: int):
    """(floor, ceil) multiples of ``multiple`` bracketing ``value`` (both > 0)."""
    v = int(value)
    floor = max(multiple, (v // multiple) * multiple)
    ceil = floor if v % multiple == 0 else floor + multiple
    return floor, ceil


def _nearest_8n1(frames: int):
    """(floor, ceil) valid ``8n+1`` frame counts bracketing ``frames``."""
    f = int(frames)
    if f <= _LTX_MIN_FRAMES:
        return _LTX_MIN_FRAMES, _LTX_MIN_FRAMES
    ceil = next_8n1(f)
    floor = ceil if (f - 1) % _LTX_TEMPORAL_BASE == 0 else ceil - _LTX_TEMPORAL_BASE
    return max(_LTX_MIN_FRAMES, floor), ceil


def assert_ltx_dims(w: int, h: int, frames: int):
    """Validate LTX-AV dims; RAISE ``ValueError`` (never round) on a violation.

    Rules: ``w % 32 == 0``, ``h % 32 == 0``, ``frames == 8n+1``,
    ``frames >= 9``. The message names the nearest valid value in BOTH
    directions (floor/ceil) so the caller can pick. ``1472x832`` passes as-is;
    ``1450x832`` raises. Returns ``(w, h, frames)`` when valid.
    """
    w, h, frames = int(w), int(h), int(frames)
    if w <= 0 or h <= 0 or frames <= 0:
        raise ValueError(
            "ltx_av dims must be positive, got w=%d h=%d frames=%d" % (w, h, frames))
    if w % _LTX_SPATIAL_MULTIPLE != 0:
        lo, hi = _nearest_multiples(w, _LTX_SPATIAL_MULTIPLE)
        raise ValueError(
            "ltx_av width %d is not a multiple of %d (nearest valid: %d or %d)"
            % (w, _LTX_SPATIAL_MULTIPLE, lo, hi))
    if h % _LTX_SPATIAL_MULTIPLE != 0:
        lo, hi = _nearest_multiples(h, _LTX_SPATIAL_MULTIPLE)
        raise ValueError(
            "ltx_av height %d is not a multiple of %d (nearest valid: %d or %d)"
            % (h, _LTX_SPATIAL_MULTIPLE, lo, hi))
    if frames < _LTX_MIN_FRAMES:
        raise ValueError(
            "ltx_av frames %d below the minimum %d (nearest valid: %d)"
            % (frames, _LTX_MIN_FRAMES, _LTX_MIN_FRAMES))
    if (frames - 1) % _LTX_TEMPORAL_BASE != 0:
        lo, hi = _nearest_8n1(frames)
        raise ValueError(
            "ltx_av frames %d is not 8n+1 (nearest valid: %d or %d)"
            % (frames, lo, hi))
    return w, h, frames


__all__ = ["assert_ltx_dims", "next_8n1", "_LTX_MIN_FRAMES",
           "_LTX_SPATIAL_MULTIPLE", "_LTX_TEMPORAL_BASE"]
