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


# --------------------------------------------------------------------------- #
# Multi-clip coverage chunk 6: geometry + the DECODED frame count
# --------------------------------------------------------------------------- #
from nodes._otr_video_engines.wan_shared import (  # noqa: E402
    ffprobe_counted_frames,
)
from nodes._otr_video_engines import wrapper_bridge as _wb_mod  # noqa: E402


@pytest.mark.skipif(not _HAS_FFMPEG, reason="needs ffmpeg + ffprobe")
def test_the_probe_reports_the_clip_GEOMETRY():
    """Segments of one beat that differ in canvas concatenate into a clip that
    changes size partway through, and the mux will not mention it."""
    import numpy as np

    frames = [np.zeros((64, 96, 3), dtype=np.uint8) for _ in range(4)]
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "geom.mp4")
        path, _n = wb.encode_frames_to_silent_mp4(frames, out, 25)
        fields = ffprobe_clip_fields(path)
    assert (fields["width"], fields["height"]) == (96, 64)


@pytest.mark.skipif(not _HAS_FFMPEG, reason="needs ffmpeg + ffprobe")
def test_counted_frames_counts_what_a_decoder_actually_reads():
    import numpy as np

    frames = [np.zeros((64, 96, 3), dtype=np.uint8) for _ in range(7)]
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "counted.mp4")
        path, n = wb.encode_frames_to_silent_mp4(frames, out, 25)
        assert ffprobe_counted_frames(path) == 7 == n


def test_counted_frames_REFUSES_an_unreadable_count(monkeypatch):
    """NO GUESS: an unreadable count is a failed verification, not a zero.

    A zero would sail straight into an assembly check as 'the beat produced no
    frames', which is a different and much more confusing failure than 'the
    count could not be read'.
    """
    class _Proc:
        returncode = 0
        stdout = b'{"streams": [{"nb_read_frames": "N/A"}]}'
        stderr = b""

    monkeypatch.setattr("subprocess.run", lambda *a, **k: _Proc())
    with pytest.raises(_wb_mod.GraphExecutionError, match="not a count"):
        ffprobe_counted_frames("whatever.mp4")


def test_counted_frames_REFUSES_a_file_with_no_video_stream(monkeypatch):
    class _Proc:
        returncode = 0
        stdout = b'{"streams": []}'
        stderr = b""

    monkeypatch.setattr("subprocess.run", lambda *a, **k: _Proc())
    with pytest.raises(_wb_mod.GraphExecutionError, match="NO video stream"):
        ffprobe_counted_frames("whatever.mp4")


def test_counted_frames_raises_NAMED_when_ffprobe_fails(monkeypatch):
    class _Proc:
        returncode = 1
        stdout = b""
        stderr = b"moov atom not found"

    monkeypatch.setattr("subprocess.run", lambda *a, **k: _Proc())
    with pytest.raises(_wb_mod.GraphExecutionError, match="count_frames failed"):
        ffprobe_counted_frames("whatever.mp4")


# --------------------------------------------------------------------------- #
# EVERY ffprobe failure is a GraphExecutionError -- the whole contract, pinned
#
# Both functions have always DOCUMENTED "a NAMED GraphExecutionError on an
# ffprobe failure", and until 2026-08-23 two shapes escaped that promise: a
# truncated JSON body came out as a raw json.JSONDecodeError, and any OSError
# that was not FileNotFoundError came out raw as well. Lean-mean order 8 closed
# both, deliberately. These tests exist so the next person to touch the probe
# boundary cannot reopen them by accident -- a caller that documents one exit
# and has three is a trap, and the suite was green through it for months.
# --------------------------------------------------------------------------- #
from nodes._otr_shared import ffprobe as _ffp_mod  # noqa: E402


def _binary(tmp_path):
    """A file that EXISTS, so binary resolution succeeds and the test is about
    what happens when it RUNS."""
    path = tmp_path / "ffprobe.exe"
    path.write_bytes(b"")
    return str(path)


def _raises_from_run(monkeypatch, exc):
    def _boom(*a, **k):
        raise exc
    monkeypatch.setattr(_ffp_mod.subprocess, "run", _boom)


@pytest.mark.parametrize("probe", ["clip_fields", "counted_frames"])
def test_a_garbled_json_body_is_a_named_refusal_not_a_json_error(
        monkeypatch, tmp_path, probe):
    """ffprobe exited 0 and wrote something that is not the document."""
    class _Proc:
        returncode = 0
        stdout = "{not json at all"
        stderr = ""

    monkeypatch.setattr(_ffp_mod.subprocess, "run", lambda *a, **k: _Proc())
    call = (ffprobe_clip_fields if probe == "clip_fields"
            else ffprobe_counted_frames)
    with pytest.raises(_wb_mod.GraphExecutionError):
        call("whatever.mp4", ffprobe=_binary(tmp_path))


@pytest.mark.parametrize("probe", ["clip_fields", "counted_frames"])
def test_a_binary_that_cannot_be_launched_is_a_named_refusal(
        monkeypatch, tmp_path, probe):
    """A permission block or a corrupt binary -- an OSError that is NOT
    FileNotFoundError, which used to escape both functions raw."""
    _raises_from_run(monkeypatch, PermissionError(13, "Permission denied"))
    call = (ffprobe_clip_fields if probe == "clip_fields"
            else ffprobe_counted_frames)
    with pytest.raises(_wb_mod.GraphExecutionError):
        call("whatever.mp4", ffprobe=_binary(tmp_path))


@pytest.mark.parametrize("probe", ["clip_fields", "counted_frames"])
def test_a_vanished_binary_still_says_not_found_by_name(
        monkeypatch, tmp_path, probe):
    """The oldest of the contracts, and it must survive the boundary move: a
    missing ffprobe is a broken install and says so in those words."""
    _raises_from_run(monkeypatch, FileNotFoundError(2, "No such file"))
    call = (ffprobe_clip_fields if probe == "clip_fields"
            else ffprobe_counted_frames)
    with pytest.raises(_wb_mod.GraphExecutionError, match="not found"):
        call("whatever.mp4", ffprobe=_binary(tmp_path))


@pytest.mark.parametrize("probe", ["clip_fields", "counted_frames"])
def test_no_ffprobe_on_the_box_at_all_is_a_named_refusal(
        monkeypatch, probe):
    monkeypatch.delenv("OTR_FFPROBE", raising=False)
    monkeypatch.delenv("OTR_FFMPEG", raising=False)
    monkeypatch.setattr(_ffp_mod.shutil, "which", lambda name: None)
    call = (ffprobe_clip_fields if probe == "clip_fields"
            else ffprobe_counted_frames)
    with pytest.raises(_wb_mod.GraphExecutionError, match="not found"):
        call("whatever.mp4")


def test_counted_frames_still_names_count_frames_in_its_refusals(
        monkeypatch, tmp_path):
    """The two functions must stay TELLABLE APART in a log. `-count_frames`
    decodes; the cheap probe reads a header. A reader who cannot tell which one
    refused cannot tell an expensive verification failure from a per-clip one."""
    class _Proc:
        returncode = 1
        stdout = ""
        stderr = "moov atom not found"

    monkeypatch.setattr(_ffp_mod.subprocess, "run", lambda *a, **k: _Proc())
    with pytest.raises(_wb_mod.GraphExecutionError, match="count_frames failed"):
        ffprobe_counted_frames("whatever.mp4", ffprobe=_binary(tmp_path))
