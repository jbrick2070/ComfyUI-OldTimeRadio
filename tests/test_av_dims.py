"""Unit tests for the LTX-AV dims validator (pure; no torch/comfy/network)."""
import pytest

from nodes._otr_shared import av_dims


def test_next_8n1_snaps_up():
    assert av_dims.next_8n1(9) == 9
    assert av_dims.next_8n1(10) == 17
    assert av_dims.next_8n1(17) == 17
    assert av_dims.next_8n1(25) == 25
    assert av_dims.next_8n1(26) == 33
    # below the floor clamps up to the minimum, never below
    assert av_dims.next_8n1(1) == av_dims._LTX_MIN_FRAMES
    assert av_dims.next_8n1(0) == av_dims._LTX_MIN_FRAMES


def test_assert_ltx_dims_accepts_canonical():
    assert av_dims.assert_ltx_dims(1472, 832, 97) == (1472, 832, 97)
    assert av_dims.assert_ltx_dims(512, 288, 97) == (512, 288, 97)
    assert av_dims.assert_ltx_dims(1472, 832, 9) == (1472, 832, 9)


def test_assert_ltx_dims_rejects_bad_width_naming_both_directions():
    with pytest.raises(ValueError) as exc:
        av_dims.assert_ltx_dims(1450, 832, 97)
    msg = str(exc.value)
    assert "width" in msg and "nearest valid" in msg
    assert "1440" in msg and "1472" in msg


def test_assert_ltx_dims_rejects_bad_height():
    with pytest.raises(ValueError) as exc:
        av_dims.assert_ltx_dims(1472, 830, 97)
    assert "height" in str(exc.value)


def test_assert_ltx_dims_rejects_non_8n1_naming_both_directions():
    with pytest.raises(ValueError) as exc:
        av_dims.assert_ltx_dims(512, 288, 100)
    msg = str(exc.value)
    assert "8n+1" in msg and "97" in msg and "105" in msg


def test_assert_ltx_dims_rejects_below_min_and_nonpositive():
    with pytest.raises(ValueError):
        av_dims.assert_ltx_dims(512, 288, 8)
    with pytest.raises(ValueError):
        av_dims.assert_ltx_dims(0, 288, 97)
    with pytest.raises(ValueError):
        av_dims.assert_ltx_dims(512, 288, 0)


def test_divergence_from_lane_a_snap_down():
    # The AV lane snaps UP; a snap-DOWN of 26 would give 25. Prove we never do that.
    assert av_dims.next_8n1(26) != 25
    assert av_dims.next_8n1(26) == 33
