"""GO_FORWARD section 4A M7 unit tests -- Wan silent-clip contract proof.

The Phase-2 clip contract used to be SELF-DECLARED: _clip_from_raw hardcoded
has_audio=False / h264 / yuv420p / bt709 / fps25 in a dict, and nothing probed
the emitted mp4. M7 makes render_clip ffprobe the silent clip and PROVE those
fields before the mux trusts them.

The validator is pure (synthetic field dicts -> deterministic, no ffmpeg). One
integration check encodes a real clip via the shared encoder and round-trips it
through ffprobe + the validator, gated on ffmpeg/ffprobe being installed.
"""

from __future__ import annotations

import os
import shutil
import tempfile

import pytest

from nodes._otr_video_engines import wrapper_bridge as wb
from nodes._otr_video_engines.eng_wan_i2v import (
    _parse_fps, ffprobe_clip_fields, validate_silent_clip_contract,
)

_HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _ok_fields(**over):
    base = {
        "codec_types": ["video"],
        "video_codec": "h264",
        "pix_fmt": "yuv420p",
        "color_space": "bt709",
        "color_primaries": None,     # ffprobe omits these for libx264+yuv420p
        "color_transfer": None,
        "fps": 25,
    }
    base.update(over)
    return base


# --------------------------------------------------------------------------- #
# _parse_fps
# --------------------------------------------------------------------------- #
def test_parse_fps_basic():
    assert _parse_fps("25/1") == 25
    assert _parse_fps("30000/1001") == 30   # ~29.97 -> 30
    assert _parse_fps("0/0") == 0
    assert _parse_fps(None) == 0
    assert _parse_fps("garbage") == 0


# --------------------------------------------------------------------------- #
# validate_silent_clip_contract -- happy path
# --------------------------------------------------------------------------- #
def test_validate_passes_canonical_contract():
    validate_silent_clip_contract(_ok_fields(), 25)


def test_validate_allows_bt709_primaries_when_reported():
    validate_silent_clip_contract(
        _ok_fields(color_primaries="bt709", color_transfer="bt709"), 25)


def test_validate_allows_unknown_primaries():
    validate_silent_clip_contract(
        _ok_fields(color_primaries="unknown", color_transfer="unknown"), 25)


# --------------------------------------------------------------------------- #
# validate_silent_clip_contract -- fail-closed
# --------------------------------------------------------------------------- #
def test_validate_fails_on_audio_stream():
    with pytest.raises(wb.GraphExecutionError) as exc:
        validate_silent_clip_contract(
            _ok_fields(codec_types=["video", "audio"]), 25)
    assert "AUDIO" in str(exc.value)


def test_validate_fails_on_zero_video_streams():
    with pytest.raises(wb.GraphExecutionError):
        validate_silent_clip_contract(_ok_fields(codec_types=[]), 25)


def test_validate_fails_on_two_video_streams():
    with pytest.raises(wb.GraphExecutionError):
        validate_silent_clip_contract(
            _ok_fields(codec_types=["video", "video"]), 25)


def test_validate_fails_on_wrong_codec():
    with pytest.raises(wb.GraphExecutionError) as exc:
        validate_silent_clip_contract(_ok_fields(video_codec="hevc"), 25)
    assert "video_codec" in str(exc.value)


def test_validate_fails_on_wrong_pix_fmt():
    with pytest.raises(wb.GraphExecutionError):
        validate_silent_clip_contract(_ok_fields(pix_fmt="yuv444p"), 25)


def test_validate_fails_on_wrong_colorspace():
    with pytest.raises(wb.GraphExecutionError):
        validate_silent_clip_contract(_ok_fields(color_space="bt2020nc"), 25)


def test_validate_fails_on_nonbt709_primaries():
    with pytest.raises(wb.GraphExecutionError):
        validate_silent_clip_contract(_ok_fields(color_primaries="bt2020"), 25)


def test_validate_fails_on_fps_mismatch():
    with pytest.raises(wb.GraphExecutionError) as exc:
        validate_silent_clip_contract(_ok_fields(fps=30), 25)
    assert "fps" in str(exc.value)


# --------------------------------------------------------------------------- #
# integration: a REAL encoded clip round-trips through ffprobe + the validator
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg/ffprobe not installed")
def test_real_encoded_clip_satisfies_contract():
    import numpy as np
    frames = (np.random.rand(4, 480, 832, 3) * 255).astype("uint8")
    out = os.path.join(tempfile.gettempdir(), "otr_m7_contract_probe.mp4")
    try:
        path, n = wb.encode_frames_to_silent_mp4(frames, out, 25)
        fields = ffprobe_clip_fields(path)
        assert "audio" not in fields["codec_types"]
        assert fields["video_codec"] == "h264"
        assert fields["pix_fmt"] == "yuv420p"
        assert fields["color_space"] == "bt709"
        assert fields["fps"] == 25
        # The whole-contract assert the render path runs.
        validate_silent_clip_contract(fields, 25)
    finally:
        if os.path.exists(out):
            os.remove(out)
