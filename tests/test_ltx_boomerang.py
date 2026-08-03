"""CPU tests for the LTX boomerang restore (BUG-LOCAL-117d, loop_via_reverse).

Covers the pure helpers (_boomerang_frames mirror semantics, _ltx_loop_source_length
math incl. the freeze-shortfall the roundtable caught) and the env + per-engine
class gate (_loop_via_reverse). No GPU.
UTF-8, no BOM, ASCII-only source.
"""
from __future__ import annotations

import numpy as np
import pytest

from nodes._otr_video_engines.eng_ltx_video import (
    LtxVideoEngine,
    _boomerang_frames,
    _ltx_loop_source_length,
)


# ---- mirror semantics -------------------------------------------------------
def test_boomerang_index_order():
    # [0,1,2,3] -> [0,1,2,3,2,1,0]; the duplicate turnaround (LAST) frame drops.
    frames = np.arange(4, dtype=np.uint8).reshape(4, 1, 1, 1)
    out = _boomerang_frames(frames)
    assert [int(x) for x in out.reshape(-1)] == [0, 1, 2, 3, 2, 1, 0]
    assert out.shape[0] == 2 * 4 - 1


def test_boomerang_preserves_8n1():
    # An 8n+1 source stays 8n+1 after the mirror (9 -> 17).
    frames = np.zeros((9, 2, 2, 3), dtype=np.uint8)
    out = _boomerang_frames(frames)
    assert out.shape[0] == 17 and out.shape[0] % 8 == 1


def test_boomerang_short_arrays_passthrough():
    for n in (0, 1):
        frames = np.zeros((n, 2, 2, 3), dtype=np.uint8)
        assert _boomerang_frames(frames).shape[0] == n


# ---- source-length math -----------------------------------------------------
def test_loop_source_length_193_roundtrips():
    # b005's target: 193 -> source 97 -> mirror back to exactly 193.
    src = _ltx_loop_source_length(193, 25)
    assert src == 97 and 2 * src - 1 == 193


def test_loop_source_length_no_freeze_shortfall(monkeypatch):
    # The 169 -> half 85 -> snap 81 -> 161 < 169 FREEZE the roundtable caught:
    # the mirror must always COVER the target (2*src-1 >= target), src is 8n+1.
    monkeypatch.delenv("OTR_LTX_LOOP_MIN_DECODE_FRAMES", raising=False)
    monkeypatch.setenv("OTR_LTX_MAX_FRAMES", "705")
    for target in (97, 121, 169, 177, 193, 233, 305):
        src = _ltx_loop_source_length(target, 25)
        assert src % 8 == 1, target
        assert 2 * src - 1 >= target, (target, src)


def test_loop_source_length_floors_at_97(monkeypatch):
    # A short beat still renders at least the PROVEN-safe 97 source @ 832x480.
    monkeypatch.delenv("OTR_LTX_LOOP_MIN_DECODE_FRAMES", raising=False)
    assert _ltx_loop_source_length(40, 25) == 97


def test_loop_min_env_override(monkeypatch):
    monkeypatch.setenv("OTR_LTX_LOOP_MIN_DECODE_FRAMES", "49")
    assert _ltx_loop_source_length(193, 25) == 97   # half already >= 49
    assert _ltx_loop_source_length(20, 25) == 49     # short beat floors at 49


# ---- env + per-engine gate --------------------------------------------------
def test_video_does_NOT_loop_by_default(monkeypatch):
    """DEFAULT FLIPPED 2026-08-01 (operator): "no boomerangs, we need to remove
    all boomerangs in place of native clips". The boomerang was the SECOND,
    independent mirror in the tree -- the WAN ping-pong is a VRAM fallback, this
    one fired on every single-clip LTX beat as a style device."""
    monkeypatch.delenv("OTR_LTX_LOOP_VIA_REVERSE", raising=False)
    assert LtxVideoEngine()._loop_via_reverse() is False


@pytest.mark.parametrize("val", ["on", "1", "true", "yes", "off", "0",
                                 "false", "no", "banana", ""])
def test_the_env_hatch_can_no_longer_arm_the_boomerang(monkeypatch, val):
    """RETIRED 2026-08-02. The knob is inert for EVERY value.

    These cases used to parse ``on/1/true/yes`` into True. The operator's
    directive is unconditional, so the capability was removed rather than
    defaulted off: a rule that survives only while nobody sets a variable is not
    a rule, and this one had already been re-armed once by a default flipping
    the other way.
    """
    monkeypatch.setenv("OTR_LTX_LOOP_VIA_REVERSE", val)
    assert LtxVideoEngine()._loop_via_reverse() is False, val


def test_the_boomerang_is_off_with_no_env_at_all(monkeypatch):
    monkeypatch.delenv("OTR_LTX_LOOP_VIA_REVERSE", raising=False)
    assert LtxVideoEngine()._loop_via_reverse() is False


# ---- THE DEFERRAL TRIPWIRE (chunk 7b, 2026-07-27) ---------------------------
#
# ltx_video declares max_frames=169 (eng_ltx_video.py:396-403) and, with NO
# environment variable set at all, returns 193 frames for a 169-frame ask. The
# declaration is therefore false TODAY, before any operator touches anything.
#
# 7b deliberately does NOT fix it, and this test is what makes that a conscious
# choice rather than an oversight nobody wrote down. The obvious repair --
# clamping 2*src-1 down to the ceiling -- was proposed by an r2 kibitz seat and
# REJECTED, because test_loop_source_length_no_freeze_shortfall above pins the
# opposite direction for exactly this target: the mirror must always COVER the
# beat window or the composite freeze-fills the tail, which is the 169 -> 161
# shortfall the roundtable already caught once. Clamping trades a declared-
# ceiling violation for a returning visible-freeze bug.
#
# The boomerang is a loop-fill fallback, so it belongs to 7c's rip. When 7c
# deletes it, THIS TEST WILL FAIL -- that is intended. The 7c author should
# delete it in the same commit that removes the boomerang, and should not
# "fix" it by loosening the assertion.

def test_the_boomerang_violates_its_own_declared_ceiling_TODAY():
    """A documented, deliberately-deferred contract violation. See the note above.

    Not an xfail: it passes now because it asserts what the code currently
    does. It is a tripwire for 7c, not a known-fail.
    """
    from nodes._otr_video_engines import frame_contract as fc
    from nodes._otr_video_engines import registry as vreg

    declared = fc.frame_contract_for(vreg.get_engine("ltx_video")).max_frames
    assert declared == 169, (
        "this tripwire is written against a declared 169 ceiling; the "
        "declaration moved to %r, so re-derive the numbers below" % (declared,))

    # No env set anywhere -- this is the DEFAULT path, not an operator override.
    src = _ltx_loop_source_length(declared, 25)
    output_frames = 2 * src - 1

    assert src == 97, src
    assert output_frames == 193, output_frames
    assert output_frames > declared, (
        "the boomerang no longer overshoots the declared ceiling -- if 7c "
        "removed the loop-fill, DELETE this tripwire in that same commit "
        "rather than relaxing it")


def test_the_boomerang_is_OFF_by_default_so_the_overshoot_is_opt_in_only():
    """The tripwire above now measures an OPT-IN path, which is the point.

    ``_LOOP_VIA_REVERSE_DEFAULT`` was True until 2026-08-01; every ltx_video beat
    on the loop path rendered 2N-1 and overshot its declared max_frames=169. With
    the default OFF that overshoot is no longer reachable unless an operator sets
    OTR_LTX_LOOP_VIA_REVERSE=on, so the declaration is honest by default and the
    tripwire documents what opting back in costs.
    """
    import os

    if os.environ.get("OTR_LTX_LOOP_VIA_REVERSE"):
        pytest.skip("box pins OTR_LTX_LOOP_VIA_REVERSE; the default is what "
                    "this test is about")
    assert LtxVideoEngine()._loop_via_reverse() is False


# ---- THE COVERAGE-PLAN NARROWING (live fix, 2026-07-29) ---------------------
#
# The tripwire above stays TRUE and stays PASSING: a SINGLE-clip render still
# overshoots its declared ceiling, and retiring the boomerang outright is still
# 7c's job. What follows closes the OTHER half -- the one a coverage plan can
# reach, which took down the live otr_w45_ltx_video leg on 2026-07-29 with
# "shot shot_music_opening_001 segment 0 rendered 193 frame(s) but its plan
# asked for 169".
#
# Inside a coverage plan the boomerang's own justification evaporates. The
# PLAN covers the beat, and it partitions against this engine's declared 169
# ceiling, so no loop-fill is owed. And render_beat_coverage refuses to
# truncate -- it proves each segment against segment.render_frames -- so the
# surplus the boomerang assumes will be absorbed is instead a hard failure.
# eng_wan_ti2v reached this conclusion first for its own ping-pong and narrows
# on the same discriminator; LTX never got the narrowing.

def _prepared(multi_clip):
    return {"session_ctx": {"multi_clip": multi_clip}}


def test_loop_fill_suppressed_inside_a_coverage_plan():
    """THE LIVE BUG. A multi-clip beat must render its planned length."""
    assert LtxVideoEngine()._loop_fill_allowed(_prepared(True)) is False


def test_loop_fill_still_runs_for_a_single_clip_beat():
    """The COVERAGE narrowing is still scoped to coverage: it is not what turns
    the loop off now. `_loop_fill_allowed` reports the engine default, which is
    False since 2026-08-01 -- so a single-clip beat is 1/1 and done, no fill."""
    assert LtxVideoEngine()._loop_fill_allowed(_prepared(False)) is False


def test_loop_fill_allowed_when_the_caller_prepared_nothing():
    """A caller with no session (the bare-adapter path) is not a coverage plan,
    so it keeps the shipped default rather than silently gaining or losing a
    loop. That default is now False."""
    for prepared in (None, {}, "not-a-dict", {"session_ctx": None}):
        assert LtxVideoEngine()._loop_fill_allowed(prepared) is False, prepared


def test_env_off_still_wins_inside_and_outside_a_coverage_plan(monkeypatch):
    """The narrowing ADDS a suppression; it never revives a disabled loop."""
    monkeypatch.setenv("OTR_LTX_LOOP_VIA_REVERSE", "off")
    eng = LtxVideoEngine()
    assert eng._loop_fill_allowed(_prepared(False)) is False
    assert eng._loop_fill_allowed(_prepared(True)) is False


def test_the_169_segment_now_renders_169_not_193():
    """The exact live numbers, from the other side of the fix.

    The tripwire above proves _ltx_loop_source_length(169) still yields a
    193-frame mirror. This proves a coverage-plan segment never reaches it:
    loop-fill is suppressed, so the length in play is _ltx_frame_length, which
    returns the ask EXACTLY -- which is what render_beat_coverage demands.
    """
    from nodes._otr_video_engines.eng_ltx_video import _ltx_frame_length

    planned = 169
    assert LtxVideoEngine()._loop_fill_allowed(_prepared(True)) is False
    assert _ltx_frame_length(planned, 25) == planned
    # and the path NOT taken is still the overshoot the tripwire pins
    assert 2 * _ltx_loop_source_length(planned, 25) - 1 == 193


def test_the_shorter_second_segment_was_doomed_too():
    """Segment 1 of the same live beat planned 89 frames; the loop floor of 97
    would have mirrored it to 193 as well -- the same 193 for a different ask,
    which is what proves the loop-fill ignores the plan rather than rounding
    it."""
    assert 2 * _ltx_loop_source_length(89, 25) - 1 == 193
    assert LtxVideoEngine()._loop_fill_allowed(_prepared(True)) is False


def test_there_is_no_opting_back_in(monkeypatch):
    """Was "opting back in actually works after the default flip".

    That guard existed because the opt-in had genuinely broken: a truthy env
    value returned the DEFAULT, which was indistinguishable from True only
    while the default was True, so the moment it flipped, "on" silently meant
    OFF -- an opt-in that could not opt in.

    The opt-in is now gone entirely, which makes that whole class of bug
    unreachable. Inverted rather than deleted, because the reason this knob
    needed a guard at all is the reason it should not exist: it was subtle
    enough to break silently, and what it turned on was backwards video.
    """
    for val in ("on", "1", "true", "yes"):
        monkeypatch.setenv("OTR_LTX_LOOP_VIA_REVERSE", val)
        assert LtxVideoEngine()._loop_via_reverse() is False, val
