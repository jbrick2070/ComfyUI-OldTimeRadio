"""BUG 4 (2026-06-20): the ALWAYS-ON audio-reactive bottom bars are a SEPARATE
overlay layer (decoupled from the scene-aware floor), DEFAULT ON, captions above,
manual 'off' switch. Tests the pure geometry/command pieces + a real-ffmpeg paint
proof that the green strip lands in the bottom band."""
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from nodes import otr_post_upscale_procgen_blend as pb


def _ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def test_audio_bars_widget_default_on():
    spec = pb.PostUpscaleProcgenBlend.INPUT_TYPES()["optional"]["audio_bars"]
    choices, cfg = spec
    assert list(choices) == ["bottom", "off"]      # 'bottom' first = default ON
    assert cfg["default"] == "bottom"


def test_bars_widget_is_last_optional_widget():
    # BUG-LOCAL-097: a new widget must be APPENDED (positional widgets_values).
    opt = list(pb.PostUpscaleProcgenBlend.INPUT_TYPES()["optional"].keys())
    assert opt[-1] == "audio_bars"


def test_bars_strip_geom_above_caption_safe():
    for w, h in [(1920, 1080), (1472, 832), (832, 1216)]:  # incl. PORTRAIT
        x, y, sw, sh = pb._bars_strip_geom(w, h)
        assert x == int(w * 0.05) and sw == w - 2 * int(w * 0.05)
        # strip sits ENTIRELY above the lower-15% caption safe-area
        safe_top = h - int(h * 0.15)
        assert y + sh <= safe_top
        assert sh >= 8


def test_build_bars_caption_cmd_is_green_lighten_at_60():
    cmd = pb._build_bars_caption_cmd(
        Path("comp.mp4"), Path("bars.mp4"), Path("out.mp4"),
        pb._BARS_OPACITY, "ffmpeg", None, (1920, 1080))
    fc = cmd[cmd.index("-filter_complex") + 1]
    assert "blend=all_mode=lighten" in fc
    assert "all_opacity=0.600" in fc                # the operator's ~60%
    assert "colorchannelmixer=rr=0:rg=0:rb=0:gr=0:gg=1:gb=0" in fc  # green-only
    assert "tpad=" in fc                            # never clamp the credits tail
    assert "-c:a" in cmd and "copy" in cmd          # audio byte-identical
    assert "ass=" not in fc                         # no captions here -> no ass


def test_build_bars_caption_cmd_captions_burn_after_bars():
    cmd = pb._build_bars_caption_cmd(
        Path("comp.mp4"), Path("bars.mp4"), Path("out.mp4"),
        pb._BARS_OPACITY, "ffmpeg", "sub.ass", (1920, 1080))
    fc = cmd[cmd.index("-filter_complex") + 1]
    # the bars blend produces [vpre]; the ass (caption) filter consumes it AFTER
    assert "[vpre]" in fc and fc.index("blend=all_mode=lighten") < fc.index("ass=")


def test_find_master_audio_prefers_audio_dir_wav(tmp_path):
    # At the blend stage the source video is SILENT (audio muxes later); the bars
    # must read the MASTER WAV from the episode's audio/ dir, not the silent mp4.
    ep = tmp_path / "signal_lost_demo"
    (ep / "audio").mkdir(parents=True)
    silent = ep / "signal_lost_demo_silent.mp4"
    silent.write_bytes(b"\x00")
    master = ep / "audio" / "pending_123_master.wav"
    master.write_bytes(b"\x00")
    assert pb._find_master_audio(silent) == master
    # no master wav -> falls back to the source itself (back-compat)
    lonely = tmp_path / "x" / "y_silent.mp4"
    lonely.parent.mkdir(parents=True)
    lonely.write_bytes(b"\x00")
    assert pb._find_master_audio(lonely) == lonely


@pytest.mark.skipif(_ffmpeg() is None, reason="ffmpeg unavailable")
def test_bars_layer_paints_green_in_bottom_strip(tmp_path):
    ff = _ffmpeg()
    src = tmp_path / "src.mp4"
    bars = tmp_path / "bars.mp4"
    # broadband (white-noise-ish) loud audio over black video so MANY freq bins
    # light up -> the green strip is unambiguous.
    subprocess.run(
        [ff, "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", "color=c=black:s=640x360:r=25:d=1",
         "-f", "lavfi", "-i", "anoisesrc=d=1:c=pink:a=0.8",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
         "-shortest", str(src)], check=True)
    out = pb._render_bars_only_mp4(src, bars, 640, 360, 25.0, ff)
    assert out is not None and bars.is_file()
    # decode one frame to raw RGB and check the strip band has green, top is black.
    raw = subprocess.run(
        [ff, "-loglevel", "error", "-i", str(bars), "-frames:v", "1",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        check=True, stdout=subprocess.PIPE).stdout
    fr = np.frombuffer(raw[: 640 * 360 * 3], dtype=np.uint8).reshape(360, 640, 3)
    x, y, sw, sh = pb._bars_strip_geom(640, 360)
    strip = fr[y:y + sh, x:x + sw, :]
    top = fr[0:y, :, :]
    assert int(strip[:, :, 1].max()) > 40          # green present in the strip
    # near-black above (a tiny yuv420p<->rgb roundtrip floor is lighten-harmless)
    assert int(top.max()) < 16
