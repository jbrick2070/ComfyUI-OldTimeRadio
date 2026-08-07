"""Model-slot audit pins for the 2026-07-09 GO_FORWARD gate.

These tests keep the compact audit doc honest: the saved canonical local path
must resolve to registered/cataloged engines, while retired or unknown engines
fail loud instead of slipping into a fallback.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from nodes import _otr_audio_engines as audio_engines
import nodes._otr_image_engines  # noqa: F401 - register image adapters
import nodes._otr_video_engines  # noqa: F401 - register video adapters
from nodes import _otr_model_catalog as catalog
from nodes import otr_video_director as video_director
from nodes._otr_image_engines import registry as image_registry
from nodes._otr_video_engines import registry as video_registry


REPO_ROOT = Path(__file__).resolve().parent.parent
CANONICAL = REPO_ROOT / "workflows" / "otr_canonical.json"


def _workflow() -> dict:
    return json.loads(CANONICAL.read_text(encoding="utf-8"))


def _node(workflow: dict, node_type: str) -> dict:
    matches = [node for node in workflow["nodes"] if node.get("type") == node_type]
    assert len(matches) == 1, node_type
    return matches[0]


def _widgets(node: dict) -> dict:
    names = [
        inp["widget"]["name"]
        for inp in node.get("inputs", [])
        if isinstance(inp, dict) and isinstance(inp.get("widget"), dict)
    ]
    return dict(zip(names, node.get("widgets_values") or []))


def test_canonical_kept_local_slots_are_registered_or_cataloged(tmp_path, monkeypatch):
    workflow = _workflow()

    writer = _widgets(_node(workflow, "OTR_LedgerScriptWriter"))
    monkeypatch.setenv("OTR_MODEL_CATALOG_AUTO_DOWNLOAD", "1")
    # 2026-07-20: canonical uses the in-process Transformers safetensors row,
    # measured in NF4 on the 16 GB card. The separate GGUF Q8 row retains its
    # own context limitations and is not selected here.
    assert catalog.validate_model_id(
        writer["creative_writing_model"], hub_root=tmp_path
    ) == "google/gemma-4-12b-it"
    assert catalog.validate_model_id(
        writer["technical_model"], hub_root=tmp_path
    ) == "google/gemma-4-12b-it"

    audio_nodes = {
        "char_voice": _widgets(_node(workflow, "OTR_BatchCharacterVoices"))["engine"],
        "announcer_voice": _widgets(_node(workflow, "OTR_AnnouncerVoice"))["engine"],
        "music": _widgets(_node(workflow, "OTR_StableAudioTheme"))["engine"],
    }
    assert audio_nodes == {
        "char_voice": "indextts2",
        "announcer_voice": "kokoro",
        "music": "stable_audio_3",
    }
    for role, engine in audio_nodes.items():
        assert audio_engines.assert_usable(engine, role) == engine

    director = _widgets(_node(workflow, "OTR_VideoDirector"))
    video_slots = {
        "announcer_visual": "announcer_video_model",
        "music_visual": "music_video_model",
        "character_video": "character_video_model",
    }
    expected_video = {
        "announcer_visual": "viz_mxc_cpu",
        "music_visual": "viz_mxc_mandala",
        "character_video": "viz_camera",
    }
    for role, widget in video_slots.items():
        engine = video_director._engine_id_from_pick(director[widget])
        assert engine == expected_video[role]
        assert video_registry.assert_usable(engine, role) == engine
        assert getattr(video_registry.get_engine(engine), "invocable", True) is True

    image_slots = {
        "announcer_visual": "announcer_image_model",
        "music_visual": "music_image_model",
        "character_video": "character_image_model",
    }
    for role, widget in image_slots.items():
        engine = director[widget]
        assert engine == "z_image_turbo"
        assert image_registry.assert_usable(engine, role) == engine


def test_retired_and_unknown_engines_fail_loud():
    for engine in ("hidream_i1", "sd35_large"):
        assert not image_registry.is_registered(engine)
        with pytest.raises(image_registry.EngineUnusable) as excinfo:
            image_registry.assert_usable(engine, "character_video")
        assert excinfo.value.reason is image_registry.EngineUsabilityReason.MALFORMED_CONFIG

    for engine in (
        "still_parallax",
        "triposr",
        "triposg_talk",
        "hunyuan3d_talk",
        "trellis_talk",
    ):
        assert not video_registry.is_registered(engine)
        with pytest.raises(video_registry.EngineUnusable) as excinfo:
            video_registry.assert_usable(engine, "character_video")
        assert excinfo.value.reason is video_registry.EngineUsabilityReason.MALFORMED_CONFIG

    with pytest.raises(audio_engines.EngineUnusable) as excinfo:
        audio_engines.assert_usable("stable_audio_3", "char_voice")
    assert excinfo.value.reason is audio_engines.EngineUsabilityReason.INCOMPATIBLE_PROFILE


def test_requested_local_smoke_candidate_contracts_are_inspected(monkeypatch):
    """Pre-live-smoke contract check for the operator's 2026-07-09 queue."""
    chatterbox = audio_engines.get_engine("chatterbox")
    assert chatterbox.roles == ("char_voice", "announcer_voice")
    assert chatterbox.interface == "per_line"
    assert chatterbox.requires_voice_ref is True
    assert chatterbox.missing_ref_fallback is None
    assert chatterbox.sample_rate == 24000

    dia = audio_engines.get_engine("dia")
    assert dia.roles == ("char_voice", "announcer_voice")
    assert dia.interface == "per_line"
    assert dia.requires_voice_ref is True
    assert dia.missing_ref_fallback is None
    assert dia.sample_rate == 44100

    wan_i2v = video_registry.get_engine("wan_i2v")
    assert wan_i2v.family == "image_to_video"
    assert wan_i2v.required_inputs == ("init_image",)
    assert wan_i2v.default_roles == ()
    assert wan_i2v._node_candidates()["wan"] == ("WanImageToVideo",)
    assert wan_i2v._loader_names()["clip"] == "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
    assert wan_i2v._loader_names()["vae"] == "wan_2.1_vae.safetensors"

    for env in ("OTR_WAN_TI2V_SAMPLER", "OTR_WAN_TI2V_SCHEDULER"):
        monkeypatch.delenv(env, raising=False)
    wan_ti2v = video_registry.get_engine("wan_ti2v")
    assert wan_ti2v.family == "image_to_video"
    assert wan_ti2v.required_inputs == ("init_image",)
    assert wan_ti2v.default_roles == ()
    assert wan_ti2v.commercial_clean is True
    assert wan_ti2v._node_candidates()["latent"] == ("Wan22ImageToVideoLatent",)
    assert wan_ti2v._loader_names()["clip"] == "umt5-xxl-encoder-Q5_K_M.gguf"
    assert wan_ti2v._loader_names()["vae"] == "wan2.2_vae.safetensors"
    assert wan_ti2v._resolve_render_config()["sampler"] == "euler"


def test_requested_cloud_smoke_candidate_contracts_are_inspected():
    cloud_still_candidates = {
        "cloud_nano_banana_2",
        "cloud_seedream_2",
        "cloud_krea_2_turbo",
        "cloud_luma_photon_flash",
    }
    for engine_id in cloud_still_candidates:
        engine = image_registry.get_engine(engine_id)
        assert engine.required_inputs == ("text_prompt",)
        assert engine.default_roles == ()
        assert "cpu" in image_registry.CAPABILITIES[engine_id]["device_backends"]
        assert image_registry.CAPABILITIES[engine_id]["model_requirements"] == []

    vidu = video_registry.get_engine("cloud_vidu_q2_pro_fast_720p")
    assert vidu.node_key == "cloud_vidu_q2_i2v"
    assert vidu.family == "image_to_video"
    assert vidu.required_inputs == ("init_image", "text_prompt")
    assert vidu.reactivity == "mute_only"
    assert vidu.must_strip_audio is True
    # rip-sfx 2026-08-06: wants_provider_sfx is DELETED, not False -- a dead
    # attribute kept to appease a stale assert is what the rip removes.
    assert not hasattr(vidu, "wants_provider_sfx")
