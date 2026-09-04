"""The 2026-08-29 audio-bearing beat clip, pinned.

Operator ruling: a joint-AV beat clip he cannot HEAR is a failed render. After
the coverage assembler cuts the beat stem, the render driver muxes that same
audio INTO the beat mp4 as an AAC preview track
(``foley_stems.mux_native_audio_into_beat_clip``) and flips the beat row's
``has_audio`` to True. The WAV sidecar stays the authoritative mix source and
``OTRSilentComposite`` strips every row with ``-an``, so the preview track can
never reach the episode master.

These tests run REAL ffmpeg on tiny synthetic files -- the helper's whole job
is container surgery, and a mocked ffmpeg would prove nothing about it.
"""
from __future__ import annotations

from nodes._otr_shared.ffmpeg import resolve_ffmpeg as _resolve_ffmpeg  # the pack's one ffmpeg owner
import os
import shutil
import subprocess

import numpy as np
import pytest

from nodes._otr_video_engines import foley_stems as fs

FPS = 25
RATE = 48000

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None and not os.environ.get("OTR_FFMPEG"),
    reason="ffmpeg is not on PATH and OTR_FFMPEG is unset")


def _tiny_mp4(path, seconds=1.0):
    """A real, silent, tiny h264 mp4 -- the shape the beat assembler emits."""
    subprocess.run(
        [_resolve_ffmpeg() or "ffmpeg",
         "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", "color=c=black:s=64x64:r=%d:d=%s" % (FPS, seconds),
         "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
        check=True)
    return str(path)


def _tiny_wav(path, seconds=1.0, value=0.25):
    n = int(RATE * seconds)
    fs.write_pcm16_wav(path, np.full((2, n), value, dtype=np.float32), RATE)
    return str(path)


def _stream_kinds(path):
    out = subprocess.run(
        [shutil.which("ffprobe") or "ffprobe", "-v", "error",
         "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True).stdout
    return [ln.strip().strip(",") for ln in out.splitlines() if ln.strip()]


def test_the_beat_clip_gains_exactly_one_aac_track_in_place(tmp_path):
    video = _tiny_mp4(tmp_path / "beat.mp4")
    wav = _tiny_wav(tmp_path / "beat_foley.wav")
    receipt = fs.mux_native_audio_into_beat_clip(video, wav, fps=FPS)
    kinds = _stream_kinds(video)
    assert kinds.count("video") == 1
    assert kinds.count("audio") == 1
    assert receipt["audio_duration_s"] > 0.0
    assert abs(receipt["video_duration_s"] - receipt["audio_duration_s"]) \
        <= (1.0 / FPS) + 0.10
    # in place: no _av sibling left behind
    assert not os.path.exists(str(tmp_path / "beat_av.mp4"))


def test_a_missing_wav_is_a_loud_refusal_and_the_mp4_survives(tmp_path):
    video = _tiny_mp4(tmp_path / "beat.mp4")
    before = os.path.getsize(video)
    with pytest.raises(fs.FoleyStemError):
        fs.mux_native_audio_into_beat_clip(
            video, str(tmp_path / "nope.wav"), fps=FPS)
    assert os.path.getsize(video) == before, "the original must be untouched"
    assert _stream_kinds(video) == ["video"]


def test_a_stem_longer_than_the_picture_is_refused(tmp_path):
    """The stem was cut to the picture upstream; a gross length disagreement
    here means the cut plan and the clip describe different beats."""
    video = _tiny_mp4(tmp_path / "beat.mp4", seconds=1.0)
    wav = _tiny_wav(tmp_path / "beat_foley.wav", seconds=3.0)
    with pytest.raises(fs.FoleyStemError):
        fs.mux_native_audio_into_beat_clip(video, wav, fps=FPS)
    # transactional: the failed sibling is gone, the silent original stands
    assert _stream_kinds(video) == ["video"]
    assert not os.path.exists(str(tmp_path / "beat_av.mp4"))


def test_the_render_driver_wires_the_mux_at_the_beat_seam():
    """Structural pin: the beat-assembly path calls the mux right where the
    foley receipts fold in, and flips has_audio there -- beat scope, never the
    per-segment rows, whose silent contract is unchanged."""
    import inspect

    from nodes._otr_video_engines import render_driver as rd
    src = inspect.getsource(rd)
    seam = src[src.index("assemble_beat_foley_segments"):]
    assert "mux_native_audio_into_beat_clip" in seam
    assert 'beat_clip["has_audio"] = True' in seam
