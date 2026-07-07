"""LTX-AV quality wire (2026-06-27) -- the bakeoff-winner upgrade tests.

Covers the four contracts from roundtables/2026-06-27-ltx-upgrade-wire/final.md
section 5, all CPU:

1. ``_scale_filter`` contract -- the shared composite scale chain: the plain
   ``-vf`` (clip/floor) and the labeled ``[in]...[out]`` (bg/fg) forms; the order
   scale -> unsharp(iff sharpen) -> pad(iff pad) -> fps; lanczos + unsharp present
   IFF sharpen; pad absent IFF pad=False; the unsharp amount env knob.
2. ALPHA preservation -- source-over BLEND MATH through ``_encode_segment_from_dir``
   (the output is flattened yuv420p, NO alpha channel): a semi-transparent RGBA
   foreground over a CONTRASTING still plate flattens to the expected fg/bg blend,
   proving the alpha matte was honored (pad=False) and not destroyed.
3. DECODE env override -- ``_decode_temporal_knobs`` resolves the whole-clip
   default 4096/8, honors a runtime env override (monkeypatch, no module reload),
   threads into ``_build_graph``'s decode node, and FAILS LOUD (no clamp) on a
   non-int / <=0 / overlap>=size value.
4. ffmpeg ``unsharp`` capability on the resolved binary (sharpen path) -- run in
   the suite, never at import/startup.

Plus the VRAM-peak threading (_clip_from_raw -> render_shot) the upgrade adds.
UTF-8, no BOM, ASCII-only source.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import pathlib

import pytest

from nodes.otr_silent_composite import (
    _scale_filter, _unsharp_amount, _encode_segment_from_dir, _ffmpeg_bin,
    count_audio_streams, probe_video,
)
from nodes._otr_video_engines.eng_ltx_av import (
    _decode_temporal_knobs, LtxAudioInEngine,
)
from nodes._otr_video_engines import wrapper_bridge as wb

HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
needs_ffmpeg = pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg/ffprobe required")


# --------------------------------------------------------------------------- #
# 1. _scale_filter contract
# --------------------------------------------------------------------------- #
def test_scale_filter_vf_clip_sharpen_order():
    """clip path: plain -vf chain, scale(lanczos) -> unsharp -> pad -> fps."""
    vf = _scale_filter(100, 50, 25, sharpen=True)
    assert not vf.startswith("[")            # plain chain (no labels)
    assert vf == (
        "scale=100:50:force_original_aspect_ratio=decrease:flags=lanczos,"
        "unsharp=5:5:0.4:5:5:0.0,"
        "pad=100:50:(ow-iw)/2:(oh-ih)/2:color=black,fps=25")


def test_scale_filter_vf_floor_no_sharpen_byte_identical_legacy():
    """floor/normalize path: sharpen=False -> the legacy bilinear chain, no
    lanczos, no unsharp (must stay byte-identical to the pre-upgrade floor)."""
    vf = _scale_filter(1472, 832, 25, sharpen=False)
    assert "lanczos" not in vf and "unsharp" not in vf
    assert vf == (
        "scale=1472:832:force_original_aspect_ratio=decrease,"
        "pad=1472:832:(ow-iw)/2:(oh-ih)/2:color=black,fps=25")


def test_scale_filter_labeled_fg_no_pad_alpha_safe():
    """fg path: labeled form, sharpen=True but pad=False (a black pad would
    paint opaque borders over the alpha edges)."""
    fg = _scale_filter(100, 50, 25, sharpen=True, pad=False,
                       in_label="1:v", out_label="fg", pre="format=rgba,")
    assert fg.startswith("[1:v]format=rgba,") and fg.endswith("[fg]")
    assert "pad=" not in fg                  # pad absent IFF pad=False
    assert ":flags=lanczos" in fg and "unsharp=5:5:" in fg
    assert fg == (
        "[1:v]format=rgba,"
        "scale=100:50:force_original_aspect_ratio=decrease:flags=lanczos,"
        "unsharp=5:5:0.4:5:5:0.0,fps=25[fg]")


def test_scale_filter_labeled_bg_floor_trim_tpad_no_sharpen():
    """bg floor path: labeled form, sharpen=False, trim prefix + tpad suffix
    preserved (byte-identical to the legacy floor bg)."""
    bg = _scale_filter(
        100, 50, 25, sharpen=False, in_label="0:v", out_label="bg",
        pre="trim=start_frame=5,setpts=PTS-STARTPTS,",
        post=",tpad=stop_mode=clone:stop_duration=3600")
    assert bg == (
        "[0:v]trim=start_frame=5,setpts=PTS-STARTPTS,"
        "scale=100:50:force_original_aspect_ratio=decrease,"
        "pad=100:50:(ow-iw)/2:(oh-ih)/2:color=black,fps=25,"
        "tpad=stop_mode=clone:stop_duration=3600[bg]")


def test_scale_filter_labeled_bg_still_sharpened():
    """bg still-plate path: labeled, sharpen=True (a real still plate IS
    composited content -- lanczos + unsharp), pad=True."""
    bg = _scale_filter(64, 64, 25, sharpen=True, in_label="0:v", out_label="bg")
    assert bg == (
        "[0:v]scale=64:64:force_original_aspect_ratio=decrease:flags=lanczos,"
        "unsharp=5:5:0.4:5:5:0.0,"
        "pad=64:64:(ow-iw)/2:(oh-ih)/2:color=black,fps=25[bg]")


def test_scale_filter_unsharp_amount_env(monkeypatch):
    monkeypatch.setenv("OTR_COMPOSITE_UNSHARP_AMOUNT", "0.8")
    assert _unsharp_amount() == 0.8
    assert "unsharp=5:5:0.8:5:5:0.0" in _scale_filter(10, 10, 25, sharpen=True)
    # a bad value falls back to the default 0.4 (cosmetic knob, not fail-loud)
    monkeypatch.setenv("OTR_COMPOSITE_UNSHARP_AMOUNT", "not-a-float")
    assert _unsharp_amount() == 0.4
    assert "unsharp=5:5:0.4:5:5:0.0" in _scale_filter(10, 10, 25, sharpen=True)


def test_scale_filter_no_sharpen_never_emits_unsharp_or_lanczos(monkeypatch):
    """The floor/credits roll must NEVER be sharpened, even with the env set."""
    monkeypatch.setenv("OTR_COMPOSITE_UNSHARP_AMOUNT", "0.8")
    vf = _scale_filter(10, 10, 25, sharpen=False)
    assert "unsharp" not in vf and "lanczos" not in vf


# --------------------------------------------------------------------------- #
# 2. ALPHA -- source-over blend math (output flattened yuv420p, NO alpha)
# --------------------------------------------------------------------------- #
@needs_ffmpeg
def test_alpha_foreground_blends_over_still_plate(tmp_path):
    """A straight-alpha RGBA fg (opaque-red | 50%-red | transparent columns) over
    a CONTRASTING BLUE still plate flattens to yuv420p: the opaque column reads
    red, the transparent column reads the BLUE background (the alpha matte was
    honored), the output carries NO alpha channel."""
    Image = pytest.importorskip("PIL.Image")

    w = h = 64
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    fg = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    px = fg.load()
    for y in range(h):
        for x in range(w):
            if x < 20:
                px[x, y] = (255, 0, 0, 255)      # opaque red
            elif x < 42:
                px[x, y] = (255, 0, 0, 128)      # 50% red
            else:
                px[x, y] = (0, 0, 0, 0)          # transparent
    for i in range(3):                            # 3 sorted RGBA frames
        fg.save(frame_dir / ("frame_%06d.png" % i))

    plate = tmp_path / "plate.png"
    Image.new("RGBA", (w, h), (0, 0, 255, 255)).save(plate)   # solid blue bg

    seg = tmp_path / "seg.mp4"
    fb = _ffmpeg_bin("ffmpeg")
    _encode_segment_from_dir(fb, str(frame_dir), 3, str(seg), w=w, h=h, fps=25,
                             bg_path=str(plate), bg_is_still=True)
    assert seg.is_file() and seg.stat().st_size > 0

    # the flatten produced a SILENT yuv420p clip -- NO alpha channel survives
    assert count_audio_streams(str(seg)) == 0
    assert probe_video(str(seg)).get("pix_fmt") == "yuv420p"

    out_png = tmp_path / "f1.png"
    subprocess.run([fb, "-y", "-loglevel", "error", "-i", str(seg),
                    "-frames:v", "1", str(out_png)],
                   check=True, capture_output=True)
    got = Image.open(out_png).convert("RGB").load()
    # opaque-red column -> red (fg replaces bg)
    r_l, g_l, b_l = got[10, 32]
    assert r_l > 150 and b_l < 110, (r_l, g_l, b_l)
    # transparent column -> the BLUE plate shows through (alpha honored, not a
    # carried alpha channel)
    r_r, g_r, b_r = got[54, 32]
    assert b_r > 150 and r_r < 110, (r_r, g_r, b_r)


# --------------------------------------------------------------------------- #
# 3. DECODE env knobs -- runtime read, fail-loud, no clamp
# --------------------------------------------------------------------------- #
def _clear_decode_env(monkeypatch):
    monkeypatch.delenv("OTR_LTX_AV_DECODE_TEMPORAL_SIZE", raising=False)
    monkeypatch.delenv("OTR_LTX_AV_DECODE_TEMPORAL_OVERLAP", raising=False)


def test_decode_knobs_default_whole_clip(monkeypatch):
    _clear_decode_env(monkeypatch)
    assert _decode_temporal_knobs() == (4096, 8)   # operator winner pick


def test_decode_knobs_env_override_runtime(monkeypatch):
    monkeypatch.setenv("OTR_LTX_AV_DECODE_TEMPORAL_SIZE", "128")
    monkeypatch.setenv("OTR_LTX_AV_DECODE_TEMPORAL_OVERLAP", "32")
    assert _decode_temporal_knobs() == (128, 32)   # the LOUD tiled fallback


def test_decode_knobs_non_int_raises(monkeypatch):
    _clear_decode_env(monkeypatch)
    monkeypatch.setenv("OTR_LTX_AV_DECODE_TEMPORAL_SIZE", "not-int")
    with pytest.raises(ValueError):
        _decode_temporal_knobs()


def test_decode_knobs_overlap_ge_size_raises(monkeypatch):
    monkeypatch.setenv("OTR_LTX_AV_DECODE_TEMPORAL_SIZE", "100")
    monkeypatch.setenv("OTR_LTX_AV_DECODE_TEMPORAL_OVERLAP", "100")
    with pytest.raises(ValueError):
        _decode_temporal_knobs()


def test_decode_knobs_nonpositive_size_raises(monkeypatch):
    monkeypatch.setenv("OTR_LTX_AV_DECODE_TEMPORAL_SIZE", "0")
    monkeypatch.setenv("OTR_LTX_AV_DECODE_TEMPORAL_OVERLAP", "0")
    with pytest.raises(ValueError):
        _decode_temporal_knobs()


def test_decode_knobs_threaded_into_build_graph(monkeypatch):
    """The decode node reads the env at _build_graph call time (monkeypatch, no
    module reload) -- spatial tile stays fixed at 512/64."""
    monkeypatch.setenv("OTR_LTX_AV_DECODE_TEMPORAL_SIZE", "256")
    monkeypatch.setenv("OTR_LTX_AV_DECODE_TEMPORAL_OVERLAP", "16")
    monkeypatch.delenv("OTR_LTX_AV_RECIPE", raising=False)
    eng = LtxAudioInEngine()
    graph = eng._build_graph(
        {"text_prompt": "t", "seed": 0}, 105, 832, 480, "a.wav", "s.png")
    dec = graph["decode"]["inputs"]
    assert dec["temporal_size"] == 256 and dec["temporal_overlap"] == 16
    assert dec["tile_size"] == 512 and dec["overlap"] == 64


def test_build_graph_default_decode_is_whole_clip(monkeypatch):
    _clear_decode_env(monkeypatch)
    monkeypatch.delenv("OTR_LTX_AV_RECIPE", raising=False)
    eng = LtxAudioInEngine()
    graph = eng._build_graph(
        {"text_prompt": "t", "seed": 0}, 105, 832, 480, "a.wav", "s.png")
    dec = graph["decode"]["inputs"]
    assert dec["temporal_size"] == 4096 and dec["temporal_overlap"] == 8


# --------------------------------------------------------------------------- #
# 4. ffmpeg unsharp capability (resolved bin; sharpen path; in-suite, not import)
# --------------------------------------------------------------------------- #
@needs_ffmpeg
def test_ffmpeg_unsharp_capability():
    fb = _ffmpeg_bin("ffmpeg")
    assert fb
    p = subprocess.run(
        [fb, "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", "color=c=red:s=64x64:d=0.1", "-frames:v", "1",
         "-vf", "scale=64:64:force_original_aspect_ratio=decrease:flags=lanczos,"
                "unsharp=5:5:0.4:5:5:0.0",
         "-f", "null", "-"],
        capture_output=True, text=True)
    assert p.returncode == 0, p.stderr


# --------------------------------------------------------------------------- #
# 5. VRAM peak threading (render-window peak -> clip -> render_shot)
# --------------------------------------------------------------------------- #
def test_clip_from_raw_threads_vram_peak():
    eng = LtxAudioInEngine()
    clip = eng._clip_from_raw(
        {"out_path": "x.mp4", "frame_count": 10, "vram_peak_mb": 12345},
        {"shot_id": "s1"})
    assert clip["vram_peak_mb"] == 12345


def test_clip_from_raw_vram_peak_none_off_box():
    eng = LtxAudioInEngine()
    clip = eng._clip_from_raw({"out_path": "x.mp4", "frame_count": 10},
                              {"shot_id": "s1"})
    assert clip["vram_peak_mb"] is None


def test_render_shot_prefers_clip_peak(monkeypatch):
    import nodes._otr_video_engines.render_driver as rd
    monkeypatch.setattr(
        rd, "_render_one",
        lambda eng, req, force_oom=False: {"vram_peak_mb": 9999})
    _clip, _shot, _att, used = rd.render_shot(
        {"shot_id": "s1", "engine_id": "ltx_audio_in"}, {})
    assert used == 9999


def test_render_shot_falls_back_when_no_clip_peak(monkeypatch):
    import nodes._otr_video_engines.render_driver as rd
    monkeypatch.setattr(
        rd, "_render_one", lambda eng, req, force_oom=False: {"path": "x"})
    monkeypatch.setattr(rd._mc, "vram_used_mb", lambda: 777)
    _clip, _shot, _att, used = rd.render_shot(
        {"shot_id": "s1", "engine_id": "ltx_audio_in"}, {})
    assert used == 777
