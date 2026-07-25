"""LIP SYNC NEVER RUNS BACKWARDS (operator directive 2026-07-25).

``wrapper_bridge.extend_frames_to_target`` builds the mirror cycle
``[0,1,..,N-1,N-2,..,1]`` so a VRAM-bounded short render can fill a beat with
seamless motion. On a scene lane that is decorative and correct -- it is the
fix for the 0.68s-then-freeze bug. On an AUDIO-SYNCED lane it is a defect:
the back half of every cycle is the render in REVERSE, so a talking head's
mouth speaks forwards and then unspeaks backwards against forward speech.

Operator, verbatim: "lip synch needs to have continuity, no render backwards,
that doesn't work."

These pins hold the boundary from both sides -- mirroring stays legal where it
was designed to be, and is terminal where it corrupts sync. Pure/CPU; no GPU,
no wrapper nodes.
"""
from __future__ import annotations

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
# The mirror is still exactly what it was on the lanes that want it.
# --------------------------------------------------------------------------- #
def test_mirror_still_allowed_by_default_and_reverses():
    """Default behaviour is UNCHANGED -- and the reversal is real, which is
    precisely why an audio lane must not use it."""
    out = wb.fit_frames_to_target(_frames(3), 5)
    assert len(out) == 5
    # cycle = [0,1,2,1] -> tiled/trimmed to 5 = [0,1,2,1,0]
    assert [int(f[0, 0, 0]) for f in out] == [0, 1, 2, 1, 0]


def test_extend_frames_to_target_untouched_by_the_flag():
    """The scene-lane extender keeps its own signature and behaviour."""
    out = wb.extend_frames_to_target(_frames(3), 5)
    assert [int(f[0, 0, 0]) for f in out] == [0, 1, 2, 1, 0]


# --------------------------------------------------------------------------- #
# On an audio-synced lane a SHORT render is terminal, never mirrored.
# --------------------------------------------------------------------------- #
def test_short_render_is_terminal_when_mirror_forbidden():
    with pytest.raises(wb.MirrorExtensionForbidden) as exc:
        wb.fit_frames_to_target(_frames(3), 9, allow_mirror=False)
    assert exc.value.rendered == 3
    assert exc.value.target == 9
    assert "backwards" in str(exc.value)


def test_trim_is_still_legal_when_mirror_forbidden():
    """Trimming never reverses time, so the flag must not block it -- a beat
    quantized UP to the render floor still has to fit back down."""
    out = wb.fit_frames_to_target(_frames(9), 4, allow_mirror=False)
    assert [int(f[0, 0, 0]) for f in out] == [0, 1, 2, 3]


def test_exact_match_is_identity_when_mirror_forbidden():
    out = wb.fit_frames_to_target(_frames(5), 5, allow_mirror=False)
    assert len(out) == 5


def test_degenerate_inputs_do_not_raise_when_mirror_forbidden():
    """A zero-length batch or a non-positive target is not a sync defect --
    it is a different failure, and this guard must not steal it."""
    assert len(wb.fit_frames_to_target(_frames(0), 5, allow_mirror=False)) == 0
    out = wb.fit_frames_to_target(_frames(3), 0, allow_mirror=False)
    assert len(out) == 3


def test_forbidden_error_is_a_valueerror_subclass():
    """Callers that only catch ValueError still fail closed rather than
    sailing past a sync defect."""
    assert issubclass(wb.MirrorExtensionForbidden, ValueError)
