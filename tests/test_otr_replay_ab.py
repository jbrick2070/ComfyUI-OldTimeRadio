"""scripts/otr_replay_ab.py scores replays of one episode against each other.

It is the measurement that caught recipe v4 of the 8 GB LTX lane leaving its
still (PBUG-20260928-05), so the three things it reads must read true on clips
whose answer is known: a clip that holds its still, one that cuts away, and one
whose every 8th frame is soft. Synthetic clips, written with PyAV; no GPU.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))
av = pytest.importorskip("av")
from PIL import Image, ImageFilter  # noqa: E402

import otr_replay_ab as ab  # noqa: E402

W, H, N = 512, 288, 60


def _textured(seed):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, (H // 8, W // 8, 3), dtype=np.uint8)
    return np.asarray(Image.fromarray(base).resize((W, H), Image.BICUBIC))


def _write_clip(path, frames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as box:
        stream = box.add_stream("mpeg4", rate=25)
        stream.width, stream.height, stream.pix_fmt = W, H, "yuv420p"
        # Every frame a keyframe: a GOP would make its keyframes sharper than
        # the frames between, a pulse of the test's own making.
        stream.gop_size = 1
        stream.options = {"qscale": "2"}
        for rgb in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
                box.mux(packet)
        for packet in stream.encode():
            box.mux(packet)


@pytest.fixture
def episode(tmp_path):
    still = _textured(1)
    stills = tmp_path / "bundle" / "stills"
    stills.mkdir(parents=True)
    Image.fromarray(still).save(stills / "still_shot_000_b1_0123456789ab.png")
    elsewhere = _textured(2)
    soft = np.asarray(Image.fromarray(still).filter(ImageFilter.GaussianBlur(3)))
    name = "shot_shot_000_b1_character_video_ltx_8gb.mp4"
    _write_clip(tmp_path / "holds" / "clips" / name, [still] * N)
    _write_clip(tmp_path / "cuts" / "clips" / name, [still] * 8 + [elsewhere] * (N - 8))
    _write_clip(tmp_path / "pulses" / "clips" / name,
                [soft if i % 8 == 0 and i else still for i in range(N)])
    # A b10 clip beside b1 must not be taken for it.
    _write_clip(tmp_path / "holds" / "clips" / "shot_shot_000_b10_character_video_ltx_8gb.mp4",
                [elsewhere] * 9)
    return tmp_path


def test_a_clip_that_cuts_away_is_counted_as_leaving_its_still(episode):
    scores = ab.score_arms(str(episode / "bundle" / "stills"),
                           [("holds", str(episode / "holds")), ("cuts", str(episode / "cuts"))])
    holds, cuts = scores["holds"]["shot_000_b1"], scores["cuts"]["shot_000_b1"]
    assert holds["distance"][24] < ab.LEFT_THE_STILL < cuts["distance"][24]
    out = io.StringIO()
    ab.report(scores, out)
    assert "cuts sits further from the still than holds at frame 24 in 1 of 1 beats" in out.getvalue()


def test_a_soft_8th_frame_reads_as_a_pulse(episode):
    scores = ab.score_arms(str(episode / "bundle" / "stills"),
                           [("holds", str(episode / "holds")), ("pulses", str(episode / "pulses"))])
    assert scores["holds"]["shot_000_b1"]["pulse"] == pytest.approx(1.0, abs=0.1)
    assert scores["pulses"]["shot_000_b1"]["pulse"] < 0.8


def test_the_beat_is_matched_by_its_whole_id(episode):
    clips = episode / "holds" / "clips"
    assert ab.clip_for(str(clips), "shot_000_b1").endswith("shot_shot_000_b1_character_video_ltx_8gb.mp4")
    assert ab.clip_for(str(clips), "shot_000_b10").endswith("shot_shot_000_b10_character_video_ltx_8gb.mp4")
    assert ab.still_beats(str(episode / "bundle" / "stills")) == {
        "shot_000_b1": str(episode / "bundle" / "stills" / "still_shot_000_b1_0123456789ab.png")}
