"""The ``viz_green`` engine -- low-VRAM ffmpeg-only procedural CRT scopes
(2026-06-18; renamed from ``visualizer`` 2026-06-30, item 2).

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
    assert vreg.is_registered("viz_green")
    eng = vreg.get_engine("viz_green")
    assert eng.name == "viz_green"
    assert eng.family == "abstract"
    assert eng.requires_flag is None            # registry IS the menu (no flag gate)
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
def test_assert_usable_no_flag_gate(monkeypatch):
    # No flag gate (registry IS the menu): OTR_ENABLE_VISUALIZER=0 no longer
    # disables the floor -- ffmpeg presence is the only real gate.
    if not sd.find_ffmpeg("ffmpeg"):
        pytest.skip("ffmpeg not on PATH")
    monkeypatch.setenv("OTR_ENABLE_VISUALIZER", "0")
    assert VisualizerEngine().assert_usable(host_caps={}, profile={}) == "viz_green"


def test_assert_usable_default_on_when_flag_unset(monkeypatch):
    if not sd.find_ffmpeg("ffmpeg"):
        pytest.skip("ffmpeg not on PATH")
    monkeypatch.delenv("OTR_ENABLE_VISUALIZER", raising=False)
    assert VisualizerEngine().assert_usable(host_caps={}, profile={}) == "viz_green"


def test_assert_usable_passes_with_flag_and_ffmpeg(monkeypatch):
    if not sd.find_ffmpeg("ffmpeg"):
        pytest.skip("ffmpeg not on PATH")
    monkeypatch.setenv("OTR_ENABLE_VISUALIZER", "1")
    rt = {"audio_ref": "C:/some/beat.wav"}
    assert VisualizerEngine().assert_usable(
        host_caps={}, profile={}, request_template=rt) == "viz_green"


def test_assert_usable_does_not_gate_audio_ref_pre_render(monkeypatch):
    # audio_ref is NOT gated in assert_usable (the per-beat audio is sliced at
    # render; the template carries an empty audio_ref for music/announcer beats,
    # like eng_ltx_av). render_clip is the LOUD audio gate. (2026-06-18 soak fix:
    # the old check aborted shot_b000_music_open before render.)
    if not sd.find_ffmpeg("ffmpeg"):
        pytest.skip("ffmpeg not on PATH")
    monkeypatch.setenv("OTR_ENABLE_VISUALIZER", "1")
    assert VisualizerEngine().assert_usable(
        host_caps={}, profile={}, request_template={"audio_ref": ""}) == "viz_green"


def test_assert_usable_for_all_three_roles(monkeypatch):
    if not sd.find_ffmpeg("ffmpeg"):
        pytest.skip("ffmpeg not on PATH")
    monkeypatch.setenv("OTR_ENABLE_VISUALIZER", "1")
    eng = VisualizerEngine()
    rt = {"audio_ref": "C:/some/beat.wav"}
    for _role in ("announcer_visual", "music_visual", "character_video"):
        assert eng.assert_usable(host_caps={}, profile={}, request_template=rt) \
            == "viz_green"


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
    assert clip["engine_id"] == "viz_green"
    assert clip["family"] == "abstract"
    assert clip["has_audio"] is False
    assert clip["type"] == "video" and clip["codec"] == "h264"
    assert clip["fps"] == 25 and clip["frame_count"] == 33
    assert clip["path"] == "/t/v.mp4"


def test_canvas_dims_default_is_16x9():
    w, h = VisualizerEngine()._canvas_dims({})
    assert (w, h) == (1472, 832)                # wide default
    assert w > h


def test_the_lane_DECLARES_NO_canvas_and_honours_ANY_request_size():
    """G2 (lane 11, 2026-08-11) -- THE PROPERTY, WHICH IS THE OPPOSITE OF A PIN.

    This lane's first draft DECLARED `render_canvas = (1472, 832)`, on the
    measured argument that `build_request_from_shot` already hands it exactly
    that while `render_single` hands it 832x480. A Codex consult broke that
    framing and was right: 1472x832 is not a property of this lane at all, it
    is the default of `OTR_VIDEO_LANDSCAPE_CANVAS` -- an OPERATOR LEVER --
    applied by the driver to every non-face family. Since
    `declared_render_canvas` is applied LAST and overrules every earlier
    channel, declaring would have made viz_green the one visualizer that
    silently ignores that lever, and would have pinned the smoke path too.

    Lesson L2's own precision note is the check that catches this: a canvas
    declaration must agree with the OVERRIDE PATH or state that the overrides
    are unsupported. Never quietly disable them.

    So the assertion is that this lane declares NOTHING and paints whatever it
    is given -- which is exactly what makes its profile canvas channel INERT
    rather than reconcilable (see PROFILE_CANVAS_DOCUMENTED_DEAD).
    """
    from nodes._otr_video_engines import render_driver as rd

    eng = vreg.get_engine("viz_green")
    assert getattr(eng, "render_canvas", None) is None
    assert rd.declared_render_canvas("viz_green") is None
    # It paints at whatever size the request carries -- including sizes no
    # profile or default would ever produce.
    for size in ((1472, 832), (832, 480), (640, 384), (1024, 576)):
        got = VisualizerEngine()._canvas_dims(
            {"canvas": {"w": size[0], "h": size[1], "fps": 25}})
        assert got == size


def test_the_landscape_lever_still_reaches_this_lane(monkeypatch):
    """The half that would otherwise rot silently.

    Asserting "declares nothing" proves the attribute is absent, not that the
    operator's lever still works. If a future change declares a canvas here,
    this is the test that says what it costs.
    """
    from nodes._otr_video_engines import render_driver as rd

    ledger = {
        "episode_id": "ep_lane11",
        "images": {"images": [{"beat_id": "b001", "kind": "scene_beat",
                               "path": "C:/tmp/scene_b001.png"}]},
        "video": {"fps": 25, "canonical_canvas": None,
                  "shots": [{"shot_id": "shot_b001", "role": "music_visual",
                             "group_id": "grp_music_visual",
                             "engine_id": "viz_green", "family": "",
                             "target_frame_count": 25}]},
    }
    shot = ledger["video"]["shots"][0]

    req = rd.build_request_from_shot(shot, ledger)
    assert (req["canvas"]["w"], req["canvas"]["h"]) == (1472, 832)   # the default

    monkeypatch.setenv("OTR_VIDEO_LANDSCAPE_CANVAS", "1024x576")
    req = rd.build_request_from_shot(shot, ledger)
    assert (req["canvas"]["w"], req["canvas"]["h"]) == (1024, 576), (
        "the operator's landscape lever no longer reaches viz_green -- if a "
        "render_canvas declaration was added, that is what it cost")


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
                       rng_key="viz_green|7")
    b = sd.paint_frame(w, h, 2, total, fps, 0.6, freq, wave, 0.5, 0.5, scan, vig,
                       rng_key="viz_green|7")
    arr_a, arr_b = np.asarray(a), np.asarray(b)
    assert arr_a.shape == (h, w, 3)
    assert np.array_equal(arr_a, arr_b)         # deterministic (seeded noise)


def test_paint_frame_seed_changes_noise():
    w, h = 320, 192
    freq = np.full(32, 0.9, dtype=np.float32)    # vol>0.3 path adds seeded noise
    wave = np.zeros(8, dtype=np.float32)
    scan, vig = sd.build_scanlines(w, h), sd.build_vignette(w, h)
    a = np.asarray(sd.paint_frame(w, h, 1, 4, 25, 0.9, freq, wave, 0.5, 0.5,
                                  scan, vig, rng_key="viz_green|1"))
    b = np.asarray(sd.paint_frame(w, h, 1, 4, 25, 0.9, freq, wave, 0.5, 0.5,
                                  scan, vig, rng_key="viz_green|2"))
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


@pytest.mark.skipif(not _have_render_deps(),
                    reason="needs ffmpeg + soundfile for a real render")
def test_render_clip_zero_frames_defaults_to_one_second(monkeypatch):
    # A degenerate 0-length beat defaults to one second (fps frames), like the
    # cheap floor -- the accessible floor renders every beat, never crashes.
    # (2026-06-18 soak: shot_b005 had target_frame_count=0.)
    monkeypatch.setenv("OTR_ENABLE_VISUALIZER", "1")
    monkeypatch.setenv("OTR_TEST_MODE", "1")
    req = {"audio_ref": "", "canvas": {"w": 320, "h": 192, "fps": 25},
           "timing": {"target_frame_count": 0}, "seed_bundle": {"request_seed": 1}}
    out = VisualizerEngine().render_clip(req, None)
    assert out["frame_count"] == 25            # fps = 1 second floor
    assert os.path.isfile(out["out_path"])


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
