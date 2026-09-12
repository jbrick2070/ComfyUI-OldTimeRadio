"""Cold install, part 2: the composite, the mux and the scopes measure through
the boundary, and the composite's gates say UNPROVEN rather than raise when
nothing on the box can measure.

Measured 2026-09-11 on the 5080 with ffprobe made unresolvable (PATH without
the WinGet dir, OTR_FFMPEG pinned to the imageio wheel binary): the canonical
Shakespeare leg cleared every engine, every per-beat probe, the counted frames
and the composite ENCODE, then OTR_SilentComposite asked its private
`_ffprobe_bin()` how many audio streams the output had, got -1, and its V-1
gate read -1 as a violation -- the empty path reached the caption burn as
"input video missing" and a 514-second render was lost with nothing in obs.
"""
from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path

import pytest

av = pytest.importorskip("av")

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OTR_TEST_MODE", "1")

from nodes._otr_shared import ffprobe as ffp  # noqa: E402
from nodes import otr_silent_composite as composite  # noqa: E402
from nodes import otr_master_audio_mux as mux  # noqa: E402
from nodes import otr_scene_aware_scopes as scopes  # noqa: E402


def _write_clip(path, *, with_audio, portrait=False):
    """A 12-frame mpeg4 clip at 24 fps (64x48, or 48x64 portrait), with an
    optional 16 kHz aac track -- written by PyAV, so the file needs no tool."""
    width, height = (48, 64) if portrait else (64, 48)
    with av.open(str(path), "w") as out:
        video = out.add_stream("mpeg4", rate=24)
        video.width, video.height, video.pix_fmt = width, height, "yuv420p"
        audio = out.add_stream("aac", rate=16000) if with_audio else None
        for i in range(12):
            frame = av.VideoFrame(width, height, "rgb24")
            frame.planes[0].update(bytes([i * 20]) * frame.planes[0].buffer_size)
            for packet in video.encode(frame):
                out.mux(packet)
        for packet in video.encode():
            out.mux(packet)
        if audio is not None:
            for i in range(8):
                frame = av.AudioFrame(format="fltp", layout="mono", samples=1024)
                frame.sample_rate = 16000
                for plane in frame.planes:
                    plane.update(bytes(plane.buffer_size))
                frame.pts = i * 1024
                for packet in audio.encode(frame):
                    out.mux(packet)
            for packet in audio.encode():
                out.mux(packet)
    return str(path)


@pytest.fixture(scope="module")
def clip_av(tmp_path_factory):
    return _write_clip(tmp_path_factory.mktemp("cold2") / "av.mp4", with_audio=True)


@pytest.fixture(scope="module")
def clip_silent(tmp_path_factory):
    return _write_clip(tmp_path_factory.mktemp("cold2") / "v.mp4", with_audio=False)


@pytest.fixture(scope="module")
def clip_portrait(tmp_path_factory):
    return _write_clip(tmp_path_factory.mktemp("cold2") / "p.mp4",
                       with_audio=False, portrait=True)


@pytest.fixture()
def no_binary(monkeypatch):
    monkeypatch.setattr(ffp, "resolve_ffprobe", lambda *a, **k: None)
    return monkeypatch


@pytest.fixture()
def nothing_measures(monkeypatch):
    """Neither a binary nor PyAV: the only shape in which -1 / 0.0 remain."""
    def refuse(*a, **k):
        raise ffp.FFprobeMissing("ffprobe not found; the PyAV fallback is unavailable too")
    monkeypatch.setattr(ffp, "probe_json", refuse)
    return monkeypatch


# --------------------------------------------------------------------------- #
# the composite measures without a binary
# --------------------------------------------------------------------------- #
def test_the_composite_counts_streams_and_frames_without_a_binary(
        no_binary, clip_av, clip_silent):
    assert composite.count_audio_streams(clip_av) == 1
    assert composite.count_audio_streams(clip_silent) == 0
    assert composite.count_video_frames(clip_silent) == 12
    assert composite.count_video_frames(clip_av) == 12


def test_the_composite_reads_the_video_shape_as_strings_like_before(
        no_binary, clip_silent):
    info = composite.probe_video(clip_silent)
    assert info == {"width": "64", "height": "48", "pix_fmt": "yuv420p",
                    "avg_frame_rate": "24/1", "r_frame_rate": "24/1"}
    assert all(isinstance(v, str) for v in info.values()), "callers parse text"


def test_the_composite_reads_durations_without_a_binary(no_binary, clip_av):
    assert composite._probe_duration(clip_av) == pytest.approx(0.512, abs=0.002)
    assert composite._probe_audio_duration(clip_av) == pytest.approx(0.512, abs=0.002)


def test_a_missing_file_still_answers_the_documented_sentinels(no_binary, tmp_path):
    gone = str(tmp_path / "gone.mp4")
    assert composite.count_audio_streams(gone) == -1
    assert composite.count_video_frames(gone) == -1
    assert composite.probe_video(gone) == {}
    assert composite._probe_duration(gone) == 0.0
    assert composite._probe_audio_duration(gone) == 0.0
    assert mux._probe_float(gone, "v:0") == -1.0


def test_when_nothing_can_measure_the_sentinels_hold_and_nothing_raises(
        nothing_measures, clip_av):
    assert composite.count_audio_streams(clip_av) == -1
    assert composite.count_video_frames(clip_av) == -1
    assert composite.probe_video(clip_av) == {}
    assert composite._probe_duration(clip_av) == 0.0
    assert composite._probe_audio_duration(clip_av) == 0.0
    assert mux._probe_float(clip_av, "a:0") == -1.0


# --------------------------------------------------------------------------- #
# the composite's gates: UNPROVEN is a warning, a real violation still raises
# --------------------------------------------------------------------------- #
def test_the_v1_gate_says_unproven_instead_of_killing_the_episode(
        monkeypatch, caplog, clip_silent, tmp_path):
    from nodes._otr_shared.ffmpeg import resolve_ffmpeg
    if not resolve_ffmpeg():
        pytest.skip("no ffmpeg on this box; the gate runs after a real encode")
    monkeypatch.setattr(composite, "count_audio_streams", lambda _p: -1)
    out = tmp_path / "silent_canonical.mp4"
    with caplog.at_level("WARNING"):
        path, report = composite.normalize_to_silent_canonical(
            clip_silent, str(out), w=64, h=48, fps=24)
    assert path == str(out) and os.path.isfile(path)
    assert any("V-1 UNPROVEN" in r.getMessage() for r in caplog.records)
    assert any("audio_streams=0 OK" in line for line in report)


def test_a_real_audio_stream_still_fails_the_v1_gate(monkeypatch, clip_silent, tmp_path):
    from nodes._otr_shared.ffmpeg import resolve_ffmpeg
    if not resolve_ffmpeg():
        pytest.skip("no ffmpeg on this box; the gate runs after a real encode")
    monkeypatch.setattr(composite, "count_audio_streams", lambda _p: 1)
    with pytest.raises(ValueError, match="must be 0"):
        composite.normalize_to_silent_canonical(
            clip_silent, str(tmp_path / "x.mp4"), w=64, h=48, fps=24)


def test_the_assemble_gates_read_a_negative_count_as_unproven_not_a_mismatch():
    source = inspect.getsource(composite)
    for gate in ("if na > 0:", "if na < 0:", "if got < 0:", "elif got != total:",
                 "if got_frames < 0:", "elif got_frames != int(n_frames):"):
        assert gate in source, gate
    assert "max(0, count_video_frames(base_video_path))" in source
    assert "A/V sync guard UNPROVEN" in source


# --------------------------------------------------------------------------- #
# the mux and the scopes measure without a binary
# --------------------------------------------------------------------------- #
def test_the_mux_measures_both_durations_without_a_binary(no_binary, clip_av, clip_silent):
    assert mux._probe_float(clip_silent, "v:0") == pytest.approx(0.5, abs=0.002)
    assert mux._probe_float(clip_av, "a:0") == pytest.approx(0.512, abs=0.002)
    # the STREAM duration, not the container: the video track of the
    # audio-carrying clip is 0.5 s inside a 0.512 s container
    assert mux._probe_float(clip_av, "v:0") == pytest.approx(0.5, abs=0.002)
    # a stream with no duration of its own falls back to the container
    monkeypatch_doc = {"streams": [{"index": 0}]}
    real = ffp.probe_json

    def stream_without_duration(path, entries=None, **kw):
        if entries == "stream=duration":
            return monkeypatch_doc
        return real(path, entries, **kw)
    no_binary.setattr(ffp, "probe_json", stream_without_duration)
    assert mux._probe_float(clip_av, "v:0") == pytest.approx(0.512, abs=0.002)


def test_the_blend_reads_real_dimensions_and_rate_without_a_binary(
        no_binary, clip_silent, tmp_path):
    from pathlib import Path as _P
    from nodes import otr_post_upscale_procgen_blend as blend
    assert blend._probe_dims(_P(clip_silent), "") == (64, 48)
    assert blend._probe_fps(_P(clip_silent), "") == 24.0
    assert blend._probe_dims(_P(tmp_path / "gone.mp4"), "") is None
    assert blend._probe_fps(_P(tmp_path / "gone.mp4"), "") == 25.0


def test_the_blend_keeps_its_own_fallbacks_when_nothing_can_measure(
        nothing_measures, clip_silent):
    from pathlib import Path as _P
    from nodes import otr_post_upscale_procgen_blend as blend
    assert blend._probe_dims(_P(clip_silent), "") is None
    assert blend._probe_fps(_P(clip_silent), "") == 25.0


def test_the_scopes_planner_detects_portrait_without_a_binary(
        no_binary, clip_silent, clip_portrait, tmp_path):
    cache = {}
    assert scopes._probe_is_portrait(clip_silent, "ffprobe", cache) is False
    assert scopes._probe_is_portrait(clip_portrait, "ffprobe", cache) is True
    assert scopes._probe_is_portrait(str(tmp_path / "gone.mp4"), "ffprobe", cache) is None
    assert cache[clip_portrait] is True, "the answer is cached per path"


# --------------------------------------------------------------------------- #
# wiring: the private resolvers are gone, the boundary is used
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("module", [composite, mux])
def test_the_private_ffprobe_resolver_is_gone(module):
    assert not hasattr(module, "_ffprobe_bin"), module.__name__
    assert "probe_json(" in inspect.getsource(module)
    assert "resolve_ffprobe() or" not in inspect.getsource(module)


def test_the_scopes_planner_goes_through_probe_json():
    source = inspect.getsource(scopes._probe_is_portrait)
    assert "probe_json(" in source and "probe_raw(" not in source
