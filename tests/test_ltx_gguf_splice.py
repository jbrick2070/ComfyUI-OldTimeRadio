"""CPU tests for the LTX 22B-GGUF clean-break splice (Phase 1).

Locks the frozen mini recipe (workflows/ltx_bookend_mini_repro_gguf_mit.json) into
LtxVideoEngine: the GGUF graph shape PER sampler mode, the banned-class license
guard, the mechanical cleanup of the old 2B / checkpoint / T5 symbols, the
distilled-mode value pin (the #1 invariant), the weight floors + resolver
defaults, and the i2v full-frame-still contract (BUG-413). No GPU.
UTF-8, no BOM, ASCII-only source.
"""
from __future__ import annotations

import pathlib

from nodes._otr_video_engines import eng_ltx_video as elv
from nodes._otr_video_engines import wrapper_bridge as wb
from nodes._otr_video_engines.eng_ltx_video import (
    LTX_DISTILLED_SIGMAS,
    LtxVideoEngine,
)

_ENG_SRC = pathlib.Path(elv.__file__).read_text(encoding="utf-8")


def _plan(seed=2):
    return {"text_prompt": "a relay station", "negative_prompt": "",
            "fps": 25, "target_frame_count": 49, "seed": seed}


# --- graph shape PER mode (SPLICE_PLAN 12.8) ------------------------------- #
def test_graph_shape_distilled(monkeypatch):
    monkeypatch.delenv("OTR_LTX_SAMPLER", raising=False)   # distilled is the default
    g = LtxVideoEngine()._build_graph(_plan(), 49, 832, 480)
    # GGUF loaders only -- NO 2B checkpoint / CLIP encoder graph keys.
    assert "checkpoint" not in g and "encoder" not in g
    assert "lora" in g
    # unet -> lora -> guider.model (the LoRA-wrapped GGUF unet IS the model)
    assert g["lora"]["inputs"]["model"] == wb.Wire("unet", 0)
    assert g["guider"]["inputs"]["model"] == wb.Wire("lora", 0)
    assert g["sampleradv"]["inputs"]["latent_image"] == wb.Wire("latent", 0)
    # te -> both CLIPTextEncode clips
    assert g["pos"]["inputs"]["clip"] == wb.Wire("te", 0)
    assert g["neg"]["inputs"]["clip"] == wb.Wire("te", 0)
    # the only VAE input is the video VAE
    assert g["vaedecode"]["inputs"]["vae"] == wb.Wire("videovae", 0)


def test_graph_shape_ksampler(monkeypatch):
    monkeypatch.setenv("OTR_LTX_SAMPLER", "ksampler")
    g = LtxVideoEngine()._build_graph(_plan(), 49, 832, 480)
    assert "checkpoint" not in g and "encoder" not in g
    assert "lora" in g
    assert g["ksampler"]["inputs"]["model"] == wb.Wire("lora", 0)
    assert g["ksampler"]["inputs"]["latent_image"] == wb.Wire("latent", 0)
    assert g["vaedecode"]["inputs"]["vae"] == wb.Wire("videovae", 0)


def test_all_vae_inputs_are_videovae_in_i2v(monkeypatch):
    monkeypatch.delenv("OTR_LTX_SAMPLER", raising=False)
    g = LtxVideoEngine()._build_graph_i2v(_plan(), 49, 832, 480, "scene.png")
    assert g["img2vid"]["inputs"]["vae"] == wb.Wire("videovae", 0)
    assert g["vaedecode"]["inputs"]["vae"] == wb.Wire("videovae", 0)
    assert "checkpoint" not in g


# --- license / banned + old node classes (SPLICE_PLAN 6, 8) --------------- #
def test_no_banned_or_old_node_classes():
    eng = LtxVideoEngine()
    seen = set()
    for getter in (eng._node_candidates, eng._node_candidates_sampling,
                   eng._node_candidates_i2v):
        for cands in getter().values():
            seen.update(cands)
    # the old 2B recipe classes are GONE
    for old in ("CheckpointLoaderSimple", "CLIPLoader", "VAEDecode"):
        assert old not in seen, old
    # AGPL/GPL classes are NEVER introduced (RES4LYF / VHS)
    for banned in ("ClownSampler_Beta", "MultimodalGuider", "VHS_VideoCombine"):
        assert banned not in seen, banned
    # the GGUF recipe classes ARE present
    for need in ("UnetLoaderGGUF", "LoraLoaderModelOnly", "VAELoader",
                 "LTXAVTextEncoderLoader", "VAEDecodeTiled"):
        assert need in seen, need


# --- mechanical cleanup of the old contract (SPLICE_PLAN 12.10) ------------ #
def test_old_symbols_gone_from_engine_source():
    for sym in ("_ckpt_path", "_ckpt_name", "_text_encoder_name",
                "_use_distilled_lora", "OTR_LTX_VIDEO_CKPT",
                "OTR_LTX_VIDEO_CKPT_NAME", "OTR_LTX_T5_ENCODER",
                "CheckpointLoaderSimple", "CLIPLoader"):
        assert sym not in _ENG_SRC, sym


# --- distilled value pin = the mini (the #1 invariant; SPLICE_PLAN 12.6) --- #
def test_distilled_pins_mini_values_ignoring_env(monkeypatch):
    monkeypatch.delenv("OTR_LTX_SAMPLER", raising=False)   # distilled default
    # junk env knobs MUST be ignored in distilled mode -- the mini wins
    monkeypatch.setenv("OTR_LTX_CFG", "9.0")
    monkeypatch.setenv("OTR_LTX_SAMPLER_NAME", "dpmpp_2m")
    monkeypatch.setenv("OTR_LTX_DISTILLED_LORA_STRENGTH", "0.1")
    monkeypatch.setenv("OTR_LTX_I2V_STRENGTH", "0.2")
    eng = LtxVideoEngine()
    g = eng._build_graph(_plan(), 49, 832, 480)
    assert g["guider"]["inputs"]["cfg"] == 1.0                          # mini
    assert g["samplersel"]["inputs"]["sampler_name"] == "euler_cfg_pp"  # mini
    assert g["lora"]["inputs"]["strength_model"] == 0.7                 # mini
    assert g["sigmas"]["inputs"]["values"] == list(LTX_DISTILLED_SIGMAS)
    iv = eng._build_graph_i2v(_plan(), 49, 832, 480, "s.png")
    assert iv["img2vid"]["inputs"]["strength"] == 0.75                  # mini i2v


def test_ksampler_honors_env_knobs(monkeypatch):
    monkeypatch.setenv("OTR_LTX_SAMPLER", "ksampler")
    monkeypatch.setenv("OTR_LTX_CFG", "4.5")
    monkeypatch.setenv("OTR_LTX_DISTILLED_LORA_STRENGTH", "0.55")
    g = LtxVideoEngine()._build_graph(_plan(), 49, 832, 480)
    assert g["ksampler"]["inputs"]["cfg"] == 4.5
    assert g["lora"]["inputs"]["strength_model"] == 0.55


# --- weight floors + resolver defaults (SPLICE_PLAN 12.5, 4A) -------------- #
def test_weight_paths_five_artifacts():
    wp = LtxVideoEngine()._weight_paths()
    labels = [w[0] for w in wp]
    assert len(wp) == 5
    assert "transformer GGUF" in labels and "projection ckpt" in labels
    assert elv._FLOOR_UNET == 8 * elv._GiB
    assert elv._FLOOR_ENCODER == 6 * elv._GiB
    assert elv._FLOOR_VIDEO_VAE == 1 * elv._GiB
    assert elv._FLOOR_LORA == 5 * elv._GiB
    assert elv._FLOOR_PROJECTION_CKPT == 30 * elv._GiB


def test_resolver_defaults_match_the_mini():
    eng = LtxVideoEngine()
    assert eng._unet_name() == "ltx-2.3-22b-dev-Q3_K_M.gguf"
    assert eng._encoder_name() == "gemma_3_12B_it_fp4_mixed.safetensors"
    assert eng._projection_ckpt() == "ltx-2.3-22b-dev.safetensors"
    assert eng._video_vae_name() == "ltx-2.3-22b-dev_video_vae.safetensors"
    name, _path = eng._distilled_lora_file()
    assert name.endswith("ltx-2.3-22b-distilled-lora-384-1.1.safetensors")


# --- i2v init = the FULL-FRAME scene still, NEVER a portrait (BUG-413 / 5) - #
def test_i2v_init_prefers_scene_still_over_portrait():
    # asset_refs carries BOTH a portrait and a full-frame scene still; the LTX
    # init resolver MUST pick the scene still (key "still"), never the portrait.
    req = {"asset_refs": {"still": "C:/out/scene_b001.png",
                          "image": "C:/out/portraits/announcer.png",
                          "init_image": "C:/out/portraits/announcer.png"}}
    assert LtxVideoEngine._init_image_path(req) == "C:/out/scene_b001.png"


def test_canvas_default_is_832x480_and_commercial_clean():
    assert (elv._LTX_DEFAULT_W, elv._LTX_DEFAULT_H) == (832, 480)
    assert LtxVideoEngine.commercial_clean is True
