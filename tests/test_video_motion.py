"""A-S5 / CW-6 CPU tests -- the in-process motion adapters (the LTX lanes).

The LTX lanes run IN-PROCESS in the main cu130 venv (their node classes are
absent in the pytest sandbox), so every test here exercises the COLD path: the
adapters are registered and selectable, fail closed without the install, an LTX
lane fails closed when SageAttention is resident (BUG-070), the init_image
aspect plan never stretches, the AS-3 lease is taken + released, and the pure
request / clip helpers are deterministic. The live load + the VRAM<=14.5 GB
boundary + render-twice pixels are the GPU smoke (operator), NOT covered here.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys
import types

import pytest

from nodes._otr_shared import gpu_residency as gr
from nodes._otr_video_engines import motion_common as mc
from nodes._otr_video_engines import registry as vreg
from nodes._otr_video_engines.eng_ltx25 import Ltx25VideoEngine
from nodes._otr_video_engines.eng_ltx_8gb import Ltx8gbEngine

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Registration + the ordered usability gate
# --------------------------------------------------------------------------- #
def test_registry_ltx_selectable_no_flag(monkeypatch):
    # ltx25_video claims no DEFAULT role. Registry IS the menu: it is
    # SELECTABLE for the bookend roles with NO flag gate.
    eng = vreg.get_engine("ltx25_video")
    assert eng.default_roles == ()
    assert eng.requires_flag is None
    for role in ("music_visual", "announcer_visual"):
        assert vreg.assert_usable("ltx25_video", role) == "ltx25_video"


def test_ltx_assert_usable_sage_then_install(monkeypatch):
    eng = vreg.get_engine("ltx25_video")
    # No flag gate (registry IS the menu). The first gate is the SageAttention
    # check: resident Sage -> BUG-070 INCOMPATIBLE_PROFILE (fail closed BEFORE
    # the node-class / weight checks and any forward).
    monkeypatch.setitem(sys.modules, "sageattention",
                        types.ModuleType("sageattention"))
    with pytest.raises(vreg.EngineUnusable) as e2:
        eng.assert_usable(host_caps={}, profile={})
    assert e2.value.reason == vreg.EngineUsabilityReason.INCOMPATIBLE_PROFILE
    # Sage clear but the install absent -> MISSING_MODEL (still closed).
    monkeypatch.delitem(sys.modules, "sageattention", raising=False)
    monkeypatch.delenv("OTR_SAGEATTENTION_PATCHED", raising=False)
    monkeypatch.setenv("OTR_LTX25_NATIVE_DIT", "ltx-2.5-NO-SUCH.safetensors")
    with pytest.raises(vreg.EngineUnusable) as e3:
        eng.assert_usable(host_caps={}, profile={})
    assert e3.value.reason == vreg.EngineUsabilityReason.MISSING_MODEL


def test_sage_gate_helper_pure():
    mc.assert_sage_not_patched("ltx25_video", "image_to_video", modules={}, env={})
    with pytest.raises(vreg.EngineUnusable) as ei:
        mc.assert_sage_not_patched("ltx25_video", "image_to_video",
                                   modules={"sageattention": object()}, env={})
    assert ei.value.reason == vreg.EngineUsabilityReason.INCOMPATIBLE_PROFILE
    assert mc.sageattention_patched(
        modules={}, env={"OTR_SAGEATTENTION_PATCHED": "1"}) is True
    assert mc.sageattention_patched(modules={}, env={}) is False


def test_aspect_transform_uniform_no_stretch():
    plan = mc.resolve_aspect_transform(480, 832, 1280, 720, "pad")   # portrait->landscape
    sx = plan["scaled_w"] / plan["src_w"]
    sy = plan["scaled_h"] / plan["src_h"]
    assert abs(sx - sy) < 0.01                       # one uniform scale (no stretch)
    assert plan["scaled_w"] % 2 == 0 and plan["scaled_h"] % 2 == 0   # mod-2
    assert plan["scaled_w"] <= 1280 and plan["scaled_h"] <= 720      # pad fits inside
    cplan = mc.resolve_aspect_transform(480, 832, 1280, 720, "crop")
    assert cplan["scaled_w"] >= 1280 and cplan["scaled_h"] >= 720    # crop covers
    with pytest.raises(ValueError):
        mc.resolve_aspect_transform(480, 832, 1280, 720, "stretch")  # unknown policy
    with pytest.raises(ValueError):                  # hand-built stretch is caught
        mc.assert_no_silent_stretch(
            {"src_w": 480, "src_h": 832, "scaled_w": 1280, "scaled_h": 720})


def test_ltx_prepare_releases_lease_on_load_failure(monkeypatch, tmp_path):
    monkeypatch.setenv("OTR_GPU_LEASE_DIR", str(tmp_path))
    # No ComfyUI node classes registered -> load() resolves the graph classes
    # and raises (WrapperNodeMissing, a RuntimeError); the AS-3 lease must be
    # released, never stranded.
    monkeypatch.setattr(
        "nodes._otr_video_engines.wrapper_bridge.node_class_mappings",
        lambda mapping=None: {} if mapping is None else mapping)
    eng = Ltx25VideoEngine()
    assert not gr.is_held()
    with pytest.raises(RuntimeError):
        eng.prepare(host_caps={}, profile={}, session_ctx={})
    assert not gr.is_held()                       # lease released, never stranded


def test_teardown_idempotent_no_lease(monkeypatch, tmp_path):
    monkeypatch.setenv("OTR_GPU_LEASE_DIR", str(tmp_path))
    for cls in (Ltx25VideoEngine, Ltx8gbEngine):
        eng = cls()
        eng.unload()                              # no residency -> no crash
        eng.teardown(None)                        # no lease -> no raise
        eng.teardown({"lease": None})
        eng.teardown({"lease": None, "patchers": []})


# --------------------------------------------------------------------------- #
# Determinism contract (V-7): build the request twice -> identical; seed routes
# --------------------------------------------------------------------------- #
def test_ltx_build_render_request_deterministic():
    eng = vreg.get_engine("ltx25_video")
    req = {"text_prompt": "a neon diner",
           "timing": {"target_frame_count": 50}, "seed_bundle": {"request_seed": 7}}
    a, b = eng._build_render_request(req), eng._build_render_request(req)
    assert a == b                                  # render-twice request stability
    assert a == {"init_image": "", "text_prompt": "a neon diner",
                 "fps": 25, "target_frame_count": 50, "seed": 7}
    req2 = dict(req, seed_bundle={"request_seed": 8})
    assert eng._build_render_request(req2)["seed"] == 8 and \
        eng._build_render_request(req2) != a       # seed routes deterministically


def test_canonicalize_silent_bt709():
    # The two lanes whose canonicalize is PURE. ltx25_video re-probes the file
    # on disk at this seam, so a stub path cannot reach it; its pure half is
    # walked by tests/test_frame_receipt_conformance.py.
    for name in ("ltx_8gb", "razzle_ltx_8gb"):
        eng = vreg.get_engine(name)
        clip = eng.canonicalize({"out_path": "C:/o.mp4", "frame_count": 40},
                                {"shot_id": "s1"}, {})
        assert clip["has_audio"] is False              # V-1: mux owns audio
        assert clip["pixel_format"] == "yuv420p"
        assert clip["matrix"] == clip["transfer"] == clip["color_primaries"] == "bt709"
        assert clip["frame_count"] == 40 and clip["fps"] == 25
        assert clip["engine_id"] == name and clip["path"] == "C:/o.mp4"
        assert clip["clip_id"] == "s1"


# --------------------------------------------------------------------------- #
# Cold-import + ASCII source (V-12 / CLAUDE.md)
# --------------------------------------------------------------------------- #
def test_cold_import_motion_adapters_no_heavy_libs():
    code = (
        "import sys;"
        "import nodes._otr_video_engines.eng_ltx25;"
        "import nodes._otr_video_engines.eng_ltx_8gb;"
        "import nodes._otr_video_engines.motion_common;"
        "heavy=[m for m in ('torch','transformers','diffusers') if m in sys.modules];"
        "print('HEAVY', heavy);"
        "sys.exit(1 if heavy else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=str(REPO_ROOT),
                       capture_output=True, text=True)
    assert r.returncode == 0, f"heavy libs pulled at import:\n{r.stdout}\n{r.stderr}"


def test_new_motion_source_is_ascii_no_em_dash():
    for name in ("motion_common.py", "eng_ltx25.py", "eng_ltx_8gb.py"):
        src_path = REPO_ROOT / "nodes" / "_otr_video_engines" / name
        src = src_path.read_text(encoding="utf-8")
        assert "—" not in src, f"em-dash forbidden in {name} (CLAUDE.md)"
        src.encode("ascii")                            # ASCII-only source


# --------------------------------------------------------------------------- #
# The shared scene-init path: an image_to_video LTX lane gets the beat's still
# --------------------------------------------------------------------------- #
def test_i2v_driver_attaches_scene_still_with_trace(monkeypatch, tmp_path):
    """A scene still in the ledger -> the ltx request shows
    init_source=scene_still; no still -> LOUD structural failure."""
    from nodes._otr_video_engines import render_driver as rd
    monkeypatch.delenv("OTR_LTX_RADIO_PROMPT", raising=False)
    still = tmp_path / "scene_b001.png"
    still.write_bytes(b"\x89PNG\r\n\x1a\n")
    ledger = {
        "meta": {"story_brief_status": "ok", "story_brief": "a relay station",
                 "story_brief_terms": {"setting": ["a relay station"]}},
        "lines": [{"line_id": "b001", "char_id": "announcer",
                   "speaker_role": "announcer", "text": "Tonight...",
                   "start_s": 0.0, "dur_s": 5.0}],
        "images": {"images": [{"object_id": "scene_b001", "kind": "scene_wide",
                               "beat_id": "b001", "path": str(still)}]},
    }
    shot = {"shot_id": "shot_b001", "source_line_ids": ["b001"],
            "role": "announcer_visual", "engine_id": "ltx25_video",
            "group_id": "grp_announcer_visual", "target_frame_count": 50,
            "creative": {}}
    req = rd.build_request_from_shot(shot, ledger)
    assert req["observability"]["init_source"] == "scene_still"
    assert req["asset_refs"]["init_image"] == str(still)
    # missing still -> structural failure; no hidden text-only degradation
    ledger2 = dict(ledger, images={"images": []})
    # The refusal comes from the SHARED scene-init path, keyed on the lane's
    # declared family = "image_to_video". Same policy, one owner.
    with pytest.raises(rd.DeferredImageGapError, match="NO scene still"):
        rd.build_request_from_shot(shot, ledger2)
