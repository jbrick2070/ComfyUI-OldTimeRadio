"""Node 12's floor is what the composite shows wherever a clip is missing.

The early in-frame scopes were retired on 2026-10-14. A fully covered episode
never shows node 12's pixels, so it cannot prove the floor still works; this
test renders a REAL floor (the CRT renderer's frames through node 12's own
encoder) and runs the REAL silent-timeline assembler over manifests that
expose it: no clips at all, leading / inter-beat / trailing gaps, a clip whose
file vanished, and a fully transparent frame-directory clip with no plate.

Every gap frame must show the floor's always-drawn "=== SIGNAL LOST ===" ident
(green, top-left) -- black fill would not -- every clip frame must show the
clip, and the assembled length must cover the floor's duration.
"""
from __future__ import annotations

import math
import subprocess
import wave

import numpy as np
import pytest

from nodes import otr_silent_composite as sc
from nodes import video_engine as ve

W, H, FPS = 832, 480, 25
FLOOR_SECONDS = 4.0          # 100 frames


def _ffmpeg():
    exe = sc._ffmpeg_bin("ffmpeg")
    if not exe:
        pytest.skip("ffmpeg not available")
    return exe


@pytest.fixture(scope="module")
def floor(tmp_path_factory):
    _ffmpeg()
    d = tmp_path_factory.mktemp("floor")
    sr = 24000
    t = np.arange(int(sr * FLOOR_SECONDS)) / sr
    audio = (0.6 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    total = int(math.ceil(FLOOR_SECONDS * FPS))
    renderer = ve._CRTRenderer(W, H, "Floor Test", ve._analyze_audio(audio, sr, total, FPS), FPS)
    wav = d / "floor.wav"
    with wave.open(str(wav), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((audio * 32767).astype(np.int16).tobytes())
    out = d / "floor.mp4"
    ve._encode_mp4((renderer.render(fi) for fi in range(total)), total,
                   str(wav), str(out), W, H, FPS)
    assert sc.count_video_frames(str(out)) == total
    return out, total


def _red_clip(path, frames):
    subprocess.run([_ffmpeg(), "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    "color=c=red:s=%dx%d:r=%d" % (W, H, FPS), "-frames:v", str(frames),
                    "-pix_fmt", "yuv420p", "-an", str(path)], check=True)
    return path


def _transparent_dir(path, frames):
    from PIL import Image
    path.mkdir()
    for i in range(frames):
        Image.new("RGBA", (W, H), (0, 0, 0, 0)).save(path / ("%06d.png" % i))
    return path


def _frame(path, n):
    raw = subprocess.run(
        [_ffmpeg(), "-loglevel", "error", "-i", str(path), "-vf",
         "select=eq(n\\,%d)" % n, "-vframes", "1", "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-"], check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape(H, W, 3).astype(np.int16)


def _shows_floor_ident(img):
    pad = W // 48
    box = img[pad // 2: pad // 2 + H // 18, pad: pad + W // 3]
    r, g, b = box[..., 0], box[..., 1], box[..., 2]
    return int(((g > 100) & (g > r + 40) & (g > b + 20)).sum()) > 20


def _is_red(img):
    return float(img[..., 0].mean()) > 180 and float(img[..., 1].mean()) < 60


def _row(i, frames, path, exists=True, kind=None):
    row = {"order": i, "shot_id": "s%d" % i, "beat_id": "b%03d" % i,
           "target_frame_count": frames, "frame_count": frames,
           "path": str(path) if path else "", "exists": exists, "start_s": None}
    if kind:
        row["type"] = kind
    return row


def _assemble(tmp_path, floor_path, clips):
    out = tmp_path / "assembled.mp4"
    res, _report = sc.assemble_silent_timeline(
        {"fps": FPS, "clips": clips}, str(floor_path), str(out), w=W, h=H, fps=FPS)
    assert res == str(out)
    assert sc.count_audio_streams(str(out)) == 0
    return out


def test_no_clips_at_all_is_the_floor_end_to_end(floor, tmp_path):
    floor_path, total = floor
    out = _assemble(tmp_path, floor_path, [])
    assert abs(sc.count_video_frames(str(out)) - total) <= 2
    for n in (0, 50, total - 3):
        assert _shows_floor_ident(_frame(out, n)), n


def test_leading_inter_and_trailing_gaps_show_the_floor(floor, tmp_path):
    floor_path, total = floor
    red = _red_clip(tmp_path / "red.mp4", 20)
    out = _assemble(tmp_path, floor_path, [
        _row(0, 20, None, exists=False),        # leading gap   0..19
        _row(1, 20, red),                       # clip         20..39
        _row(2, 20, None, exists=False),        # inter gap    40..59
        _row(3, 20, red),                       # clip         60..79
    ])                                          # trailing     80..99 (floor tail)
    assert abs(sc.count_video_frames(str(out)) - total) <= 2
    for n in (10, 50, 90):
        assert _shows_floor_ident(_frame(out, n)), ("gap frame", n)
    for n in (30, 70):
        img = _frame(out, n)
        assert _is_red(img) and not _shows_floor_ident(img), ("clip frame", n)


def test_a_clip_whose_file_vanished_falls_back_to_the_floor(floor, tmp_path):
    floor_path, total = floor
    gone = tmp_path / "gone.mp4"                # never written: exists=True lies
    out = _assemble(tmp_path, floor_path, [_row(0, 40, gone, exists=True)])
    assert abs(sc.count_video_frames(str(out)) - total) <= 2
    assert _shows_floor_ident(_frame(out, 20))


def test_a_transparent_directory_clip_with_no_plate_shows_the_floor(floor, tmp_path):
    floor_path, total = floor
    frames = _transparent_dir(tmp_path / "frames", 20)
    out = _assemble(tmp_path, floor_path, [_row(0, 20, frames, kind="directory")])
    assert abs(sc.count_video_frames(str(out)) - total) <= 2
    assert _shows_floor_ident(_frame(out, 10)), "the floor must show through the empty frames"
