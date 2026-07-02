"""LTX audio-in lane tests for the ONE `ltx_audio_in` engine (the ltx_av_talk /
ltx_av_music split was removed 2026-06-26 -- the talk-vs-scene routing is now
role-driven in render_driver). Registration, role-fit, schema family, assert_usable
flag gate, ref extraction, silent canonicalize. CPU-only; no torch, no comfy nodes,
no network, no real forward (render_clip's graph is GPU-only -- M4 smoke only)."""
import pytest

from nodes._otr_video_engines import registry
from nodes._otr_video_engines import schemas
from nodes._otr_video_engines.registry import EngineUsabilityReason, EngineUnusable
from nodes._otr_video_engines.eng_ltx_av import LtxAudioInEngine
from nodes._otr_shared import role_compat


def test_engine_registered_and_legacy_gone():
    names = set(registry.all_engine_names())
    assert "ltx_audio_in" in names
    assert "ltx_av_talk" not in names
    assert "ltx_av_music" not in names


def test_default_roles_are_the_bookends():
    # ltx_audio_in is the per-role DEFAULT for music + announcer (inheriting the
    # slot the deleted ltx_av_music held). Registry IS the menu: no flag gate.
    assert LtxAudioInEngine.default_roles == ("music_visual", "announcer_visual")
    assert LtxAudioInEngine.requires_flag is None


def test_family_and_required_inputs():
    assert LtxAudioInEngine.family == "audio_conditioned_video"
    assert set(LtxAudioInEngine.required_inputs) == {
        "text_prompt", "audio_ref", "init_image"}
    assert LtxAudioInEngine.accepts_still is True
    assert LtxAudioInEngine._is_talk is True


def test_family_registered_in_schemas_and_in_sync():
    assert "audio_conditioned_video" in schemas.FAMILIES
    # the FAMILY-level requirement stays text_prompt + audio_ref (the engine adds
    # init_image on top of the family minimum).
    assert schemas.FAMILY_REQUIRED_INPUTS["audio_conditioned_video"] == (
        "text_prompt", "audio_ref")
    assert set(schemas.FAMILIES) == set(schemas.FAMILY_REQUIRED_INPUTS)


def test_capabilities_row_present():
    assert "ltx_audio_in" in registry.CAPABILITIES
    assert "ltx_av_talk" not in registry.CAPABILITIES
    assert "ltx_av_music" not in registry.CAPABILITIES
    # M0-measured Q3_K_M peak is under the 14500 ceiling
    assert registry.CAPABILITIES["ltx_audio_in"]["vram_estimate_mb"] <= 14500


def _desc(engine_cls):
    return {"engine_id": engine_cls.name, "roles": engine_cls.roles,
            "required_inputs": engine_cls.required_inputs}


def test_role_fit_serves_bookends_and_character():
    d = _desc(LtxAudioInEngine)
    # text+audio+init -> fits every role that supplies all three (capability-only,
    # operator 2026-06-22). ltx_audio_in serves music + announcer (bookends) AND
    # character (the role-driven driver picks the right still per beat).
    assert role_compat.engine_fits_role(d, "music_visual")
    assert role_compat.engine_fits_role(d, "announcer_visual")
    assert role_compat.engine_fits_role(d, "character_video")
    # rip-sfx-broll (2026-07-01): the dead roles are unknown tokens and raise.
    import pytest as _pytest
    for dead in ("scene_broll", "background_abstract"):
        with _pytest.raises(role_compat.RoleCompatError):
            role_compat.engine_fits_role(d, dead)


def test_assert_usable_no_flag_gate(monkeypatch):
    # No flag gate (registry IS the menu): OTR_ENABLE_LTX_AV=0 no longer disables
    # the lane. The ladder still fails closed on the REAL gates -- here NVML, which
    # the heavy LTX-AV lane requires to enforce the VRAM ceiling. NEVER GATED_BY_FLAG.
    monkeypatch.setenv("OTR_ENABLE_LTX_AV", "0")
    monkeypatch.setattr(
        "nodes._otr_shared.gpu_residency.nvml_available", lambda *a, **k: False)
    with pytest.raises(EngineUnusable) as exc:
        LtxAudioInEngine().assert_usable(host_caps=None, profile=None)
    assert exc.value.reason == EngineUsabilityReason.INCOMPATIBLE_PROFILE
    assert exc.value.reason != EngineUsabilityReason.GATED_BY_FLAG


def test_ref_path_extraction():
    eng = LtxAudioInEngine()
    assert eng._ref_path({"path": "/x/a.wav"}) == "/x/a.wav"
    assert eng._ref_path("/x/b.wav") == "/x/b.wav"
    assert eng._ref_path(None) == ""

    class _R:
        path = "/x/c.wav"
    assert eng._ref_path(_R()) == "/x/c.wav"


def test_build_render_request_from_dict():
    eng = LtxAudioInEngine()
    req = {
        "shot_id": "shot_b001",
        "text_prompt": "a person at a period microphone",
        "audio_ref": {"path": "/tmp/slice.wav"},
        "asset_refs": {"init_image": "/tmp/scene_still.png"},
        "timing": {"target_frame_count": 120},
        "seed_bundle": {"request_seed": 4242},
    }
    plan = eng._build_render_request(req)
    assert plan["audio_path"] == "/tmp/slice.wav"
    assert plan["init_image"] == "/tmp/scene_still.png"
    assert plan["target_frame_count"] == 120
    assert plan["seed"] == 4242
    assert plan["text_prompt"]


def test_canonicalize_is_silent_with_identity_stamps():
    eng = LtxAudioInEngine()
    req = {"shot_id": "shot_b002"}
    clip = eng.canonicalize({"out_path": "/tmp/x.mp4", "frame_count": 97}, req, None)
    assert clip["has_audio"] is False          # V-1: only the mux adds audio
    assert clip["engine_id"] == "ltx_audio_in"
    assert clip["family"] == "audio_conditioned_video"
    assert clip["color_primaries"] == "bt709"
    assert clip["frame_count"] == 97
    assert clip["clip_id"] == "shot_b002"


def test_no_brief_helper_imports_in_engine_source():
    # PROMPT-THIN: the adapter must not import any brief/story helpers.
    import os
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(here, "nodes", "_otr_video_engines", "eng_ltx_av.py"),
               encoding="utf-8").read()
    assert "story_brief" not in src
    assert "brief_helpers" not in src
    assert "finish_visual_prompt" not in src
