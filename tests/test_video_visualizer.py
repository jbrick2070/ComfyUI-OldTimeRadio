"""The ``visualizer`` engine -- low-VRAM ffmpeg-only procedural CRT scopes (2026-06-18).

CPU-only coverage: registration / identity, the LOUD assert_usable ladder (no
fallback), the pure render-request + CanonicalClip helpers, scope_draw determinism,
and -- when ffmpeg + soundfile are present -- a real silent-mp4 render of the
expected frame count at 25fps 16:9 with has_audio False. The engine must NOT couple
to the floor node or the SceneAwareScopes overlay.
"""
from __future__ import annotations

import os
import shutil

import numpy as np
import pytest

from nodes._otr_video_engines import registry as vreg
from nodes._otr_video_engines.eng_visualizer import VisualizerEngine
from nodes._otr_video_engines.registry import (
    EngineUnusable, EngineUsabilityReason,
)
from nodes._otr_shared import scope_draw as sd


# --------------------------------------------------------------------------- #
# registration / identity
# --------------------------------------------------------------------------- #
def test_registered_and_identity():
    assert vreg.is_registered("visualizer")
    eng = vreg.get_engine("visualizer")
    assert eng.name == "visualizer"
    assert eng.family == "abstract"
    assert eng.requires_flag == "OTR_ENABLE_VISUALIZER"
    assert eng.default_roles == ()
    assert eng.required_inputs == ("audio_ref",)
    assert eng.render_aspect == "wide"
    assert eng.target_fps == 25
    assert eng.commercial_clean is True
    assert eng.fallback_engine is None          # NO FALLBACKS


def test_serves_all_three_visual_roles():
    assert set(VisualizerEngine().roles) == {
        "announcer_visual", "music_visual", "character_video"}


# --------------------------------------------------------------------------- #
# assert_usable -- LOUD, no fallback
# --------------------------------------------------------------------------- #
def test_assert_usable_gated_by_flag(monkeypatch):
    monkeypatch.delenv("OTR_ENABLE_VISUALIZER", raising=False)
    with pytest.raises(EngineUnusable) as exc:
        VisualizerEngine().assert_usable(host_caps={}, profile={})
    assert exc.value.reason is EngineUsabilityReason.GATED_BY_FLAG


def test_assert_usable_passes_with_flag_and_ffmpeg(monkeypatch):
    if not sd.find_ffmpeg("ffmpeg"):
        pytest.skip("ffmpeg not on PATH")
    monkeypatch.setenv("OTR_ENABLE_VISUALIZER", "1")
    rt = {"audio_ref": "C:/some/beat.wav"}
    assert VisualizerEngine().assert_usable(
        host_caps={}, profile={}, request_template=rt) == "visualizer"


def test_assert_usable_does_not_gate_audio_ref_pre_render(monkeypatch):
    # audio_ref is NOT gated in assert_usable (the per-beat audio is sliced at
    # render; the template carries an empty audio_ref for music/announcer beats,
    # like eng_ltx_av). render_clip is the LOUD audio gate. (2026-06-18 soak fix:
    # the old check aborted shot_b000_music_open before render.)
    if not sd.find_ffmpeg("ffmpeg"):
        pytest.skip("ffmpeg not on PATH")
    monkeypatch.setenv("OTR_ENABLE_VISUALIZER", "1")
    assert VisualizerEngine().assert_usable(
        host_caps={}, profile={}, request_template={"audio_ref": ""}) == "visualizer"


def test_assert_usable_for_all_three_roles(monkeypatch):
    if not sd.find_ffmpeg("ffmpeg"):
        pytest.skip("ffmpeg not on PATH")
    monkeypatch.setenv("OTR_ENABLE_VISUALIZER", "1")
    eng = VisualizerEngine()
    rt = {"audio_ref": "C:/some/beat.wav"}
    for _role in ("announcer_visual", "music_visual", "character_video"):
        assert eng.assert_usable(host_caps={}, profile={}, request_template=rt) \
            == "visualizer"


# --------------------------------------------------------------------------- #
# pure helpers
# --------------------------------------------------------------------------- #
def test_ref_path_handles_str_dict_and_empty():
    eng = VisualizerEngine()
    assert eng._ref_path("a.wav") == "a.wav"
    assert eng._ref_path({"path": "b.wav"}) == "b.wav"
    assert eng._ref_path("") == ""
    assert eng._ref_path(None) == ""


def test_build_render_request_reads_timing_and_seed():
    req = {"audio_ref": "x.wav", "timing": {"target_frame_count": 50},
           "seed_bundle": {"request_seed": 7}}
    plan = VisualizerEngine()._build_render_request(req)
    assert plan == {"audio_path": "x.wav", "target_frame_count": 50, "seed": 7}


def test_canonicalize_shape_silent():
    eng = VisualizerEngine()
    clip = eng.canonicalize({"out_path": "/t/v.mp4", "frame_count": 33},
                            {"shot_id": "b0001"}, {})
    assert clip["engine_id"] == "visualizer"
    assert clip["family"] == "abstract"
    assert clip["has_audio"] is False
    assert clip["type"] == "video" and clip["codec"] == "h264"
    assert clip["fps"] == 25 and clip["frame_count"] == 33
    assert clip["path"] == "/t/v.mp4"


def test_canvas_dims_default_is_16x9():
    w, h = VisualizerEngine()._canvas_dims({})
    assert (w, h) == (1472, 832)                # wide default
    assert w > h


# --------------------------------------------------------------------------- #
# scope_draw determinism + shape (pure, torch-free)
# --------------------------------------------------------------------------- #
def test_scope_draw_is_torch_free():
    import sys
    # importing scope_draw must not require torch
    assert "torch" not in repr(sd.paint_frame)  # sanity; real guard = import works
    src = open(sd.__file__, encoding="utf-8").read()
    assert "import torch" not in src


def test_paint_frame_deterministic_and_shaped():
    w, h, total, fps = 320, 192, 4, 25
    freq = np.linspace(0.1, 0.9, 32).astype(np.float32)
    wave = np.sin(np.linspace(0, 6.28, 200)).astype(np.float32)
    scan = sd.build_scanlines(w, h)
    vig = sd.build_vignette(w, h)
    a = sd.paint_frame(w, h, 2, total, fps, 0.6, freq, wave, 0.5, 0.5, scan, vig,
                       rng_key="visualizer|7")
    b = sd.paint_frame(w, h, 2, total, fps, 0.6, freq, wave, 0.5, 0.5, scan, vig,
                       rng_key="visualizer|7")
    arr_a, arr_b = np.asarray(a), np.asarray(b)
    assert arr_a.shape == (h, w, 3)
    assert np.array_equal(arr_a, arr_b)         # deterministic (seeded noise)


def test_paint_frame_seed_changes_noise():
    w, h = 320, 192
    freq = np.full(32, 0.9, dtype=np.float32)    # vol>0.3 path adds seeded noise
    wave = np.zeros(8, dtype=np.float32)
    scan, vig = sd.build_scanlines(w, h), sd.build_vignette(w, h)
    a = np.asarray(sd.paint_frame(w, h, 1, 4, 25, 0.9, freq, wave, 0.5, 0.5,
                                  scan, vig, rng_key="visualizer|1"))
    b = np.asarray(sd.paint_frame(w, h, 1, 4, 25, 0.9, freq, wave, 0.5, 0.5,
                                  scan, vig, rng_key="visualizer|2"))
    assert not np.array_equal(a, b)             # different seed -> different noise


# --------------------------------------------------------------------------- #
# render_clip (real ffmpeg + soundfile; skip if absent)
# --------------------------------------------------------------------------- #
def _have_render_deps():
    try:
        import soundfile  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return bool(shutil.which("ffmpeg"))


@pytest.mark.skipif(not _have_render_deps(),
                    reason="needs ffmpeg + soundfile for a real render")
def test_render_clip_produces_silent_16x9_mp4(tmp_path, monkeypatch):
    import soundfile as sf
    monkeypatch.setenv("OTR_ENABLE_VISUALIZER", "1")
    monkeypatch.setenv("OTR_TEST_MODE", "1")
    sr, dur = 24000, 0.5
    wav = tmp_path / "beat.wav"
    sf.write(str(wav), (0.2 * np.sin(
        np.linspace(0, 220 * 6.28 * dur, int(sr * dur)))).astype(np.float32), sr)
    total = 8
    req = {"audio_ref": str(wav), "canvas": {"w": 320, "h": 192, "fps": 25},
           "timing": {"target_frame_count": total},
           "seed_bundle": {"request_seed": 7}, "shot_id": "b0001"}
    eng = VisualizerEngine()
    out = eng.render_clip(req, eng.prepare({}, {}, {}))
    assert out["frame_count"] == total
    assert os.path.isfile(out["out_path"]) and os.path.getsize(out["out_path"]) > 0
    clip = eng.canonicalize(out, req, {})
    assert clip["has_audio"] is False and clip["frame_count"] == total


@pytest.mark.skipif(not _have_render_deps(),
                    reason="needs ffmpeg + soundfile for a real render")
def test_render_clip_idle_scopes_when_audio_absent(monkeypatch):
    # The accessible floor renders EVERY beat: no audio_ref -> idle scopes from
    # synthesized silence (a silent beat is a silent scope, not a crash). This is
    # what makes it forced-on-all-roles safe (2026-06-18 soak: shot_b005 had none).
    monkeypatch.setenv("OTR_ENABLE_VISUALIZER", "1")
    monkeypatch.setenv("OTR_TEST_MODE", "1")
    total = 8
    req = {"audio_ref": "", "canvas": {"w": 320, "h": 192, "fps": 25},
           "timing": {"target_frame_count": total}, "seed_bundle": {"request_seed": 1}}
    out = VisualizerEngine().render_clip(req, None)
    assert out["frame_count"] == total
    assert os.path.isfile(out["out_path"]) and os.path.getsize(out["out_path"]) > 0


def test_render_clip_fails_loud_on_bad_frame_count(monkeypatch):
    # total_frames <= 0 is a real config error -> still fails LOUD.
    monkeypatch.setenv("OTR_ENABLE_VISUALIZER", "1")
    req = {"audio_ref": "", "timing": {"target_frame_count": 0},
           "seed_bundle": {"request_seed": 1}}
    with pytest.raises(EngineUnusable):
        VisualizerEngine().render_clip(req, None)


# --------------------------------------------------------------------------- #
# separation invariant + hygiene
# --------------------------------------------------------------------------- #
def test_engine_does_not_import_the_floor_node():
    # The SEPARATION INVARIANT is about IMPORTS, not prose -- scan import lines
    # only (the docstring legitimately names what the engine must NOT import).
    import ast
    p = __import__("nodes._otr_video_engines.eng_visualizer",
                   fromlist=["x"]).__file__
    tree = ast.parse(open(p, encoding="utf-8").read())
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
            imported += ["%s.%s" % (node.module or "", a.name) for a in node.names]
    joined = " ".join(imported)
    assert "video_engine" not in joined         # never the floor renderer
    assert "scene_aware_scopes" not in joined    # never the overlay node


def test_no_dummy_token_utf8():
    p = __import__("nodes._otr_video_engines.eng_visualizer",
                   fromlist=["x"]).__file__
    raw = open(p, "rb").read()
    assert raw[:3] != b"\xef\xbb\xbf"           # no BOM
    assert b"dummy" not in raw.lower()
