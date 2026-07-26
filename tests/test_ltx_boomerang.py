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
def test_video_loops_by_default(monkeypatch):
    monkeypatch.delenv("OTR_LTX_LOOP_VIA_REVERSE", raising=False)
    assert LtxVideoEngine()._loop_via_reverse() is True


@pytest.mark.parametrize("val,expect", [
    ("on", True), ("1", True), ("true", True), ("yes", True),
    ("off", False), ("0", False), ("false", False), ("no", False),
])
def test_env_parse(monkeypatch, val, expect):
    monkeypatch.setenv("OTR_LTX_LOOP_VIA_REVERSE", val)
    assert LtxVideoEngine()._loop_via_reverse() is expect


def test_env_invalid_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("OTR_LTX_LOOP_VIA_REVERSE", "banana")
    assert LtxVideoEngine()._loop_via_reverse() is True   # LOUD warn + default


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


def test_the_boomerang_is_on_by_default_which_is_why_the_violation_is_reachable():
    """The overshoot above is not an opt-in path; it is the shipped default.

    ``_LOOP_VIA_REVERSE_DEFAULT`` is True (eng_ltx_video.py:412), so every
    ltx_video beat that takes the loop path renders 2N-1. If this ever becomes
    False the violation stops being reachable by default and the tripwire above
    is measuring a path nobody runs.
    """
    import os

    if os.environ.get("OTR_LTX_LOOP_VIA_REVERSE"):
        pytest.skip("box pins OTR_LTX_LOOP_VIA_REVERSE; the default is what "
                    "this test is about")
    assert LtxVideoEngine()._loop_via_reverse() is True
