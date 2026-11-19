"""The floor's frame count for planning skips the full decode (audit O5).

`count_video_frames` decodes every frame of the 1080p floor -- 33.3 s for a
409 s episode against 0.28 s to count packets, same answer. The fast reading
is trusted only when the container's own frame count agrees, and the decoded
count stays the A/V gate. UTF-8, no BOM, SFW.
"""
from __future__ import annotations

import shutil
import subprocess

import pytest

from nodes import otr_silent_composite as sc


def _floor(tmp_path, seconds=2, fps=25):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg not on this box")
    out = tmp_path / "floor.mp4"
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    "color=c=gray:s=64x36:r=%d" % fps, "-t", str(seconds),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)],
                   check=True)
    return str(out)


def test_the_fast_count_agrees_with_the_decoded_count(tmp_path):
    path = _floor(tmp_path)
    assert sc.count_floor_frames(path) == sc.count_video_frames(path) == 50


def test_a_disagreeing_header_falls_back_to_decoding(tmp_path, monkeypatch):
    path = _floor(tmp_path)
    monkeypatch.setattr(sc._ffp, "probe_json", lambda *a, **k: {
        "streams": [{"nb_read_packets": "50", "nb_frames": "49"}]}
        if "-count_packets" in k.get("extra_args", ()) else
        {"streams": [{"nb_read_frames": "48"}]})
    assert sc.count_floor_frames(path) == 48


def test_a_missing_floor_is_minus_one(tmp_path):
    assert sc.count_floor_frames(str(tmp_path / "nope.mp4")) == -1


def test_assembly_plans_with_the_fast_count():
    import inspect
    src = inspect.getsource(sc)
    assert "floor_frames = max(0, count_floor_frames(base_video_path))" in src
