"""NOTHING RUNS BACKWARDS. THE MIRROR IS GONE (operator directive 2026-08-02).

This file used to hold a boundary from BOTH sides: mirroring stayed legal on
scene lanes, and was terminal only on audio-synced ones. That boundary is
retired. The operator's rule is unconditional -- "kill mirrors and ping-pong,
true video for every second of audio" -- so
``wrapper_bridge.extend_frames_to_target`` was DELETED and
``fit_frames_to_target`` lost its ``allow_mirror`` parameter.

The distinction the old tests defended was real, but it does not survive the
rule. The back half of every mirror cycle is the render played in REVERSE. On a
talking head that is a mouth unspeaking; on a scene lane it is motion running
backwards under dialogue that keeps going forward. The second case is subtler,
not absent.

WHAT COVERS THE BEAT NOW: coverage planning splits a beat the engine cannot
afford in one pass into multiple NATIVE segments, each rendered forward. Every
frame is original and no second of audio is covered twice -- which is what the
mirror was approximating, and what these tests pin instead.

Pure/CPU; no GPU, no wrapper nodes.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from nodes._otr_video_engines import wrapper_bridge as wb


def _frames(n, h=4, w=4, c=3):
    """``n`` distinguishable uint8 frames -- frame i is filled with value i.

    ``n == 0`` builds the empty ``[0,H,W,C]`` batch directly; ``np.stack`` of
    an empty list raises, and the empty batch is a case under test.
    """
    if n <= 0:
        return np.zeros((0, h, w, c), dtype=np.uint8)
    return np.stack([np.full((h, w, c), i, dtype=np.uint8) for i in range(n)])


# --------------------------------------------------------------------------- #
# The mirror does not exist, and cannot be asked for.
# --------------------------------------------------------------------------- #
def test_the_mirror_extender_is_gone():
    """Deleted, not defaulted off. A capability that still exists behind a flag
    is one forgotten keyword argument away from shipping backwards video -- and
    this one had already been re-armed once by a default flipping the other
    way."""
    assert not hasattr(wb, "extend_frames_to_target")
    assert "extend_frames_to_target" not in getattr(wb, "__all__", ())


def test_fit_frames_to_target_takes_no_mirror_flag():
    """``allow_mirror`` is gone from the signature, so no caller can re-arm the
    mirror by passing True -- deliberately, or by copying an older call."""
    assert "allow_mirror" not in inspect.signature(
        wb.fit_frames_to_target).parameters
    with pytest.raises(TypeError):
        wb.fit_frames_to_target(_frames(2), 4, allow_mirror=True)


# --------------------------------------------------------------------------- #
# A short render is terminal on EVERY lane, not only the lip-synced ones.
# --------------------------------------------------------------------------- #
def test_short_render_is_terminal():
    with pytest.raises(wb.MirrorExtensionForbidden) as exc:
        wb.fit_frames_to_target(_frames(3), 9)
    assert exc.value.rendered == 3
    assert exc.value.target == 9
    assert "backwards" in str(exc.value)


def test_one_frame_short_still_raises():
    """No "close enough" tolerance. One frame short is 40 ms of audio with no
    video, and inventing that frame is exactly what was removed."""
    with pytest.raises(wb.MirrorExtensionForbidden):
        wb.fit_frames_to_target(_frames(7), 8)


# --------------------------------------------------------------------------- #
# Trimming never reverses time, so it is untouched.
# --------------------------------------------------------------------------- #
def test_trim_is_still_legal():
    """A beat quantized UP to a legal render rung still has to fit back down --
    this is the normal path for every engine whose ladder misses the target."""
    out = wb.fit_frames_to_target(_frames(9), 4)
    assert [int(f[0, 0, 0]) for f in out] == [0, 1, 2, 3], (
        "trim keeps the FIRST n frames, in forward order")


def test_exact_match_is_identity():
    src = _frames(5)
    out = wb.fit_frames_to_target(src, 5)
    assert len(out) == 5
    assert np.array_equal(out, src)


def test_degenerate_inputs_do_not_raise():
    """A zero-length batch or a non-positive target is not a coverage defect --
    it is a different failure, and this guard must not steal it."""
    assert len(wb.fit_frames_to_target(_frames(0), 5)) == 0
    out = wb.fit_frames_to_target(_frames(3), 0)
    assert len(out) == 3
    assert len(wb.fit_frames_to_target(_frames(3), -2)) == 3


def test_forbidden_error_is_a_valueerror_subclass():
    """Callers catch it by name to add routing advice; it stays a ValueError so
    a caller that does not know the name still fails closed rather than sailing
    past a coverage defect."""
    assert issubclass(wb.MirrorExtensionForbidden, ValueError)
