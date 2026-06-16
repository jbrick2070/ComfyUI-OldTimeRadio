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
