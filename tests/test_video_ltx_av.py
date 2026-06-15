"""LTX-AV (audio-input) lane tests: registration/dark, role-fit, schema family,
assert_usable flag gate, ref extraction, silent canonicalize. CPU-only; no torch,
no comfy nodes, no network, no real forward (render_clip's graph is GPU-only and
is exercised by the M4 smoke, never here)."""
import pytest

from nodes._otr_video_engines import registry
from nodes._otr_video_engines import schemas
from nodes._otr_video_engines.registry import EngineUsabilityReason, EngineUnusable
from nodes._otr_video_engines.eng_ltx_av import LtxAvTalkEngine, LtxAvMusicEngine
from nodes._otr_shared import role_compat


def test_both_engines_registered():
    names = set(registry.all_engine_names())
    assert "ltx_av_talk" in names
    assert "ltx_av_music" in names


def test_both_engines_dark_default_roles_empty():
    assert LtxAvTalkEngine.default_roles == ()
    assert LtxAvMusicEngine.default_roles == ()
    assert LtxAvTalkEngine.requires_flag == "OTR_ENABLE_LTX_AV"
    assert LtxAvMusicEngine.requires_flag == "OTR_ENABLE_LTX_AV"


def test_families_and_required_inputs():
    assert LtxAvTalkEngine.family == "audio_driven_face"
    assert LtxAvMusicEngine.family == "audio_conditioned_video"
    assert set(LtxAvTalkEngine.required_inputs) == {
        "text_prompt", "audio_ref", "init_image"}
    assert set(LtxAvMusicEngine.required_inputs) == {"text_prompt", "audio_ref"}


def test_new_family_registered_in_schemas_and_in_sync():
    assert "audio_conditioned_video" in schemas.FAMILIES
    assert schemas.FAMILY_REQUIRED_INPUTS["audio_conditioned_video"] == (
        "text_prompt", "audio_ref")
    # the in-module guard (sets must stay equal) still holds after the addition
    assert set(schemas.FAMILIES) == set(schemas.FAMILY_REQUIRED_INPUTS)


def test_capabilities_rows_present():
    assert "ltx_av_talk" in registry.CAPABILITIES
    assert "ltx_av_music" in registry.CAPABILITIES
    # M0-measured Q3_K_M peak is under the 14500 ceiling
    assert registry.CAPABILITIES["ltx_av_talk"]["vram_estimate_mb"] <= 14500
    assert registry.CAPABILITIES["ltx_av_music"]["vram_estimate_mb"] <= 14500


def _desc(engine_cls):
    return {"engine_id": engine_cls.name, "roles": engine_cls.roles,
            "required_inputs": engine_cls.required_inputs}


def test_role_fit_talk():
    d = _desc(LtxAvTalkEngine)
    assert role_compat.engine_fits_role(d, "announcer_visual")
    assert role_compat.engine_fits_role(d, "character_video")
    # talk is not offered for music_visual (role membership gate)
    assert not role_compat.engine_fits_role(d, "music_visual")


def test_role_fit_music_requires_audio_ref_supply():
    d = _desc(LtxAvMusicEngine)
    # music_visual now supplies audio_ref (M1 role_compat edit) so music fits
    assert role_compat.engine_fits_role(d, "music_visual")
    # and not the talk roles (membership gate)
    assert not role_compat.engine_fits_role(d, "announcer_visual")


def test_assert_usable_gated_by_flag(monkeypatch):
    monkeypatch.delenv("OTR_ENABLE_LTX_AV", raising=False)
    for cls in (LtxAvTalkEngine, LtxAvMusicEngine):
        with pytest.raises(EngineUnusable) as exc:
            cls().assert_usable(host_caps=None, profile=None)
        assert exc.value.reason == EngineUsabilityReason.GATED_BY_FLAG


def test_ref_path_extraction():
    eng = LtxAvMusicEngine()
    assert eng._ref_path({"path": "/x/a.wav"}) == "/x/a.wav"
    assert eng._ref_path("/x/b.wav") == "/x/b.wav"
    assert eng._ref_path(None) == ""

    class _R:
        path = "/x/c.wav"
    assert eng._ref_path(_R()) == "/x/c.wav"


def test_build_render_request_from_dict():
    eng = LtxAvTalkEngine()
    req = {
        "shot_id": "shot_b001",
        "text_prompt": "a person at a period microphone",
        "audio_ref": {"path": "/tmp/slice.wav"},
        "asset_refs": {"init_image": "/tmp/portrait.png"},
        "timing": {"target_frame_count": 120},
        "seed_bundle": {"request_seed": 4242},
    }
    plan = eng._build_render_request(req)
    assert plan["audio_path"] == "/tmp/slice.wav"
    assert plan["init_image"] == "/tmp/portrait.png"
    assert plan["target_frame_count"] == 120
    assert plan["seed"] == 4242
    assert plan["text_prompt"]


def test_canonicalize_is_silent_with_identity_stamps():
    eng = LtxAvMusicEngine()
    req = {"shot_id": "shot_b002"}
    clip = eng.canonicalize({"out_path": "/tmp/x.mp4", "frame_count": 97}, req, None)
    assert clip["has_audio"] is False          # V-1: only the mux adds audio
    assert clip["engine_id"] == "ltx_av_music"
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
