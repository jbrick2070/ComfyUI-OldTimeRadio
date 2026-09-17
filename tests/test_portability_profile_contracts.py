"""Shipping LTX/HuMo profiles name every artifact and safe launch fact."""
from __future__ import annotations

import json
from pathlib import Path

from nodes._otr_video_engines import eng_humo
from nodes._otr_video_engines import ltx25_recipe as ltx


ROOT = Path(__file__).resolve().parents[1]
PROFILES = ROOT / "config" / "profiles"

LTX_IDS = (
    "otr_ltx25_foley_lumina",
    "otr_ltx25_high_foley_plus",
    "otr_ltx25_high_mime",
    "otr_ltx25_high_video",
    "otr_w45_ltx25_foley_plus",
    "otr_w45_ltx25_mime",
    "otr_w45_ltx25_video",
)
HUMO14_IDS = (
    "otr_g4_humo", "otr_w45_humo", "otr_w45_humo_14b_169",
)
HUMO17_IDS = (
    "otr_w45_humo_1_7b", "otr_w45_humo_1_7b_169",
)


def _profile(profile_id: str) -> dict:
    with (PROFILES / (profile_id + ".json")).open(encoding="utf-8") as handle:
        return json.load(handle)


def test_every_shipping_ltx_profile_has_complete_model_and_launch_contract():
    expected = [
        ltx.LTX25_DIT_GGUF,
        ltx.LTX25_TEXT_ENCODER_GGUF,
        ltx.LTX25_VIDEO_VAE,
        ltx.LTX25_AUDIO_VAE,
        ltx.LTX25_UPSCALER_MODEL,
    ]
    for profile_id in LTX_IDS:
        profile = _profile(profile_id)
        assert profile["status"] == "shipping"
        models = profile["preflight"]["required_models"]
        assert models[:5] == expected
        assert profile["launch"] == {
            "boot_contract": "default",
            "sage_attention": False,
            "extra_args": [],
            "env": {},
        }


def test_ltx_foley_lumina_profile_keeps_ltx_weights_only():
    models = _profile("otr_ltx25_foley_lumina")["preflight"][
        "required_models"]
    assert "flux-2-klein-4b-Q4_K_M.gguf" not in models
    assert models[:5] == [
        "LTX-2.5-Distilled-Q3_K_M.gguf",
        "gemma4-12b-with-proj-ltx-2.5-Q5_K_M.gguf",
        "ltx-2.5-video-vae-bf16.safetensors",
        "ltx-2.5-audio-vae-bf16.safetensors",
        "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
    ]
    assert len(models) == len(set(models)) == 5


def test_humo_14b_profiles_follow_engine_loader_order(monkeypatch):
    for name in (
        "OTR_HUMO_CKPT", "OTR_HUMO_UNET_NAME", "OTR_HUMO_LORA_NAME",
        "OTR_HUMO_CLIP_NAME", "OTR_HUMO_VAE_NAME",
        "OTR_HUMO_AUDIO_ENCODER_NAME",
    ):
        monkeypatch.delenv(name, raising=False)
    expected = list(eng_humo.HuMoEngine()._loader_names().values())

    for profile_id in HUMO14_IDS:
        profile = _profile(profile_id)
        assert profile["status"] == "shipping"
        assert profile["preflight"]["required_models"] == expected


def test_humo_17b_profiles_omit_the_incompatible_14b_lora(monkeypatch):
    for name in (
        "OTR_HUMO_17B_CKPT", "OTR_HUMO_17B_UNET_NAME",
        "OTR_HUMO_17B_LORA_NAME", "OTR_HUMO_CLIP_NAME",
        "OTR_HUMO_VAE_NAME", "OTR_HUMO_AUDIO_ENCODER_NAME",
    ):
        monkeypatch.delenv(name, raising=False)
    expected = [
        value for value in eng_humo.HuMo17BEngine()._loader_names().values()
        if value != "none"
    ]

    for profile_id in HUMO17_IDS:
        profile = _profile(profile_id)
        assert profile["status"] == "shipping"
        models = profile["preflight"]["required_models"]
        assert models == expected
        assert not any("lightx2v" in name.lower() for name in models)
