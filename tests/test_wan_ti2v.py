"""GO_FORWARD section 4 / 4A (M8 + S2) -- the wan_ti2v 5B TI2V engine.

CPU-only coverage for the new 8GB-tier engine: registration + roles, the GGUF
loader-mode inference, the 5B node-candidate set (Wan22ImageToVideoLatent, NOT
WanImageToVideo), the 5B graph wiring (KSampler takes the CLIPTextEncode
positive/negative DIRECTLY + the latent from Wan22ImageToVideoLatent), the M8
Wan2.2-VAE fail-closed guard, the assert_usable fail-closed ladder, and that the
shared pure helpers (build_render_request / staged name / clip dict) bind to this
engine. No GPU, no model load.
"""

from __future__ import annotations

import pytest

from nodes._otr_video_engines import registry as vreg
from nodes._otr_video_engines import wrapper_bridge as wb
from nodes._otr_video_engines.eng_wan_ti2v import WanTi2vEngine
from nodes._otr_video_engines.registry import (
    EngineUnusable, EngineUsabilityReason,
)


# --------------------------------------------------------------------------- #
# registration / identity
# --------------------------------------------------------------------------- #
def test_registered_and_identity():
    assert vreg.is_registered("wan_ti2v")
    eng = vreg.get_engine("wan_ti2v")
    assert eng.name == "wan_ti2v"
    assert eng.family == "image_to_video"
    assert eng.requires_flag == "OTR_ENABLE_WAN_TI2V"
    assert eng.default_roles == ()              # dark / selectable-not-default
    assert eng.required_inputs == ("init_image",)
    assert eng.commercial_clean is True         # Apache-2.0 (manifest)


def test_serves_the_image_init_roles():
    eng = WanTi2vEngine()
    assert set(eng.roles) == {"scene_broll", "music_visual", "character_video"}


def test_distinct_from_wan_i2v_not_a_subclass():
    # Shares only the pure mixin -- it must NOT be a WanI2VEngine subclass.
    from nodes._otr_video_engines.eng_wan_i2v import WanI2VEngine
    assert not issubclass(WanTi2vEngine, WanI2VEngine)


# --------------------------------------------------------------------------- #
# loader mode + node candidates (the 5B difference)
# --------------------------------------------------------------------------- #
def test_loader_mode_infers_gguf_from_default_unet(monkeypatch):
    monkeypatch.delenv("OTR_WAN_TI2V_LOADER", raising=False)
    monkeypatch.delenv("OTR_WAN_TI2V_UNET_NAME", raising=False)
    monkeypatch.delenv("OTR_WAN_TI2V_CKPT", raising=False)
    assert WanTi2vEngine()._loader_mode() == "gguf"


def test_loader_mode_explicit_safetensors(monkeypatch):
    monkeypatch.setenv("OTR_WAN_TI2V_LOADER", "safetensors")
    assert WanTi2vEngine()._loader_mode() == "safetensors"


def test_node_candidates_use_the_5b_latent_node(monkeypatch):
    monkeypatch.delenv("OTR_WAN_TI2V_LOADER", raising=False)
    monkeypatch.delenv("OTR_WAN_TI2V_UNET_NAME", raising=False)
    monkeypatch.delenv("OTR_WAN_TI2V_CKPT", raising=False)
    cands = WanTi2vEngine()._node_candidates()
    assert cands["latent"] == ("Wan22ImageToVideoLatent",)
    assert cands["unet"] == ("UnetLoaderGGUF",)         # gguf default
    assert "wan" not in cands                           # not the 14B WanImageToVideo


def test_node_candidates_unet_switches_to_core_loader_on_safetensors(monkeypatch):
    monkeypatch.setenv("OTR_WAN_TI2V_LOADER", "safetensors")
    assert WanTi2vEngine()._node_candidates()["unet"] == ("UNETLoader",)


def test_loader_names_default_to_umt5_and_wan22_vae(monkeypatch):
    for k in ("OTR_WAN_TI2V_CLIP_NAME", "OTR_WAN_TI2V_VAE_NAME"):
        monkeypatch.delenv(k, raising=False)
    names = WanTi2vEngine()._loader_names()
    assert names["clip"] == "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
    assert names["vae"] == "wan2.2_vae.safetensors"


# --------------------------------------------------------------------------- #
# the 5B graph wiring -- KSampler takes pos/neg DIRECT + latent from the 5B node
# --------------------------------------------------------------------------- #
def _graph(monkeypatch):
    monkeypatch.delenv("OTR_WAN_TI2V_LOADER", raising=False)
    monkeypatch.delenv("OTR_WAN_TI2V_UNET_NAME", raising=False)
    monkeypatch.delenv("OTR_WAN_TI2V_CKPT", raising=False)
    eng = WanTi2vEngine()
    req = {"text_prompt": "a slow pan", "canvas": {"w": 832, "h": 480}}
    return eng._build_graph(req, "init.png", {"seed": 7}, 81, 832, 480)


def test_graph_ksampler_takes_pos_neg_directly_and_5b_latent(monkeypatch):
    g = _graph(monkeypatch)
    ks = g["ksampler"]["inputs"]
    assert tuple(ks["positive"]) == ("pos", 0)
    assert tuple(ks["negative"]) == ("neg", 0)
    assert tuple(ks["latent_image"]) == ("latent", 0)


def test_graph_latent_node_takes_vae_and_start_image(monkeypatch):
    g = _graph(monkeypatch)
    lat = g["latent"]["inputs"]
    assert tuple(lat["vae"]) == ("vae", 0)
    assert tuple(lat["start_image"]) == ("loadimage", 0)
    assert lat["width"] == 832 and lat["height"] == 480 and lat["length"] == 81
    assert "positive" not in lat and "negative" not in lat   # 5B latent has no cond


def test_graph_gguf_unet_inputs_have_no_weight_dtype(monkeypatch):
    g = _graph(monkeypatch)
    assert "weight_dtype" not in g["unet"]["inputs"]
    assert g["unet"]["inputs"]["unet_name"].endswith(".gguf")


def test_graph_modelsampling_shift_default_5(monkeypatch):
    monkeypatch.delenv("OTR_WAN_TI2V_SHIFT", raising=False)
    g = _graph(monkeypatch)
    assert g["modelsampling"]["inputs"]["shift"] == 5.0


# --------------------------------------------------------------------------- #
# M8 -- the Wan2.2-VAE fail-closed guard
# --------------------------------------------------------------------------- #
def _stage_full_install(tmp_path, monkeypatch, *, vae_name="wan2.2_vae.safetensors"):
    ckpt = tmp_path / "Wan2.2-TI2V-5B-Q5_K_M.gguf"
    ckpt.write_bytes(b"unet-placeholder")
    monkeypatch.setenv("OTR_ENABLE_WAN_TI2V", "1")
    monkeypatch.setenv("OTR_WAN_TI2V_CKPT", str(ckpt))
    clip_dir = tmp_path / "text_encoders"
    clip_dir.mkdir()
    vae_dir = tmp_path / "vae"
    vae_dir.mkdir()
    monkeypatch.setenv("OTR_WAN_TI2V_CLIP_DIR", str(clip_dir))
    monkeypatch.setenv("OTR_WAN_TI2V_VAE_DIR", str(vae_dir))
    if vae_name is not None:
        monkeypatch.setenv("OTR_WAN_TI2V_VAE_NAME", vae_name)
    eng = WanTi2vEngine()
    names = eng._loader_names()
    (clip_dir / names["clip"]).write_bytes(b"clip-placeholder")
    if names["vae"]:
        (vae_dir / names["vae"]).write_bytes(b"vae-placeholder")
    return eng


def test_assert_usable_passes_with_wan22_vae(tmp_path, monkeypatch):
    eng = _stage_full_install(tmp_path, monkeypatch)
    assert eng.assert_usable(host_caps={}, profile={}) == "wan_ti2v"


def test_m8_rejects_the_21_vae(tmp_path, monkeypatch):
    eng = _stage_full_install(tmp_path, monkeypatch,
                              vae_name="wan_2.1_vae.safetensors")
    with pytest.raises(EngineUnusable) as exc:
        eng.assert_usable(host_caps={}, profile={})
    assert exc.value.reason is EngineUsabilityReason.MISSING_MODEL
    assert "M8" in str(exc.value)


def test_m8_rejects_empty_vae_name(tmp_path, monkeypatch):
    eng = _stage_full_install(tmp_path, monkeypatch, vae_name="")
    with pytest.raises(EngineUnusable) as exc:
        eng.assert_usable(host_caps={}, profile={})
    assert exc.value.reason is EngineUsabilityReason.MISSING_MODEL
    assert "M8" in str(exc.value)


# --------------------------------------------------------------------------- #
# assert_usable fail-closed ladder
# --------------------------------------------------------------------------- #
def test_flag_gate_precedes_everything(tmp_path, monkeypatch):
    eng = _stage_full_install(tmp_path, monkeypatch)
    monkeypatch.delenv("OTR_ENABLE_WAN_TI2V", raising=False)
    with pytest.raises(EngineUnusable) as exc:
        eng.assert_usable(host_caps={}, profile={})
    assert exc.value.reason is EngineUsabilityReason.GATED_BY_FLAG


def test_missing_unet_fails_closed(tmp_path, monkeypatch):
    eng = _stage_full_install(tmp_path, monkeypatch)
    monkeypatch.setenv("OTR_WAN_TI2V_CKPT", str(tmp_path / "nope.gguf"))
    monkeypatch.setenv("OTR_WAN_TI2V_UNET_DIR", str(tmp_path / "empty"))
    with pytest.raises(EngineUnusable) as exc:
        eng.assert_usable(host_caps={}, profile={})
    assert exc.value.reason is EngineUsabilityReason.MISSING_MODEL
    assert "UNET not found" in str(exc.value)


def test_missing_clip_fails_closed(tmp_path, monkeypatch):
    eng = _stage_full_install(tmp_path, monkeypatch)
    # delete the staged clip so only the VAE remains
    import os
    clip = os.path.join(os.environ["OTR_WAN_TI2V_CLIP_DIR"],
                        eng._loader_names()["clip"])
    os.remove(clip)
    with pytest.raises(EngineUnusable) as exc:
        eng.assert_usable(host_caps={}, profile={})
    assert exc.value.reason is EngineUsabilityReason.MISSING_MODEL
    assert "CLIP/umt5" in str(exc.value)


# --------------------------------------------------------------------------- #
# shared pure helpers bind to this engine
# --------------------------------------------------------------------------- #
def test_shared_clip_dict_carries_this_engine_id():
    eng = WanTi2vEngine()
    clip = eng._clip_from_raw({"out_path": "/x/y.mp4", "frame_count": 81},
                              {"shot_id": "b004"})
    assert clip["engine_id"] == "wan_ti2v"
    assert clip["family"] == "image_to_video"
    assert clip["has_audio"] is False
    assert clip["clip_id"] == "b004"


def test_shared_clip_dict_fallback_id_uses_engine_name():
    eng = WanTi2vEngine()
    clip = eng._clip_from_raw({}, {})
    assert clip["clip_id"] == "wan_ti2v_clip"


def test_shared_build_render_request_is_deterministic():
    eng = WanTi2vEngine()
    req = {"asset_refs": {"init_image": "/p.png"},
           "seed_bundle": {"request_seed": 5},
           "timing": {"target_frame_count": 50}}
    a, b = eng._build_render_request(req), eng._build_render_request(req)
    assert a == b
    assert a["seed"] == 5 and a["init_image"] == "/p.png"


def test_shared_staged_name_keys_on_shot_seed_dims():
    eng = WanTi2vEngine()
    name = eng._staged_init_name(
        {"shot_id": "b009", "seed_bundle": {"request_seed": 3}}, 832, 480)
    assert name == "otr_wan_init_b009_s3_832x480.png"
