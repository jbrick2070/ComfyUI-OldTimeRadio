"""CPU topology lock for the S5 SILENT two-stage HQ recipe (2026-07-02).

eng_ltx_video hq_two_stage = the ia2v_canonical transplant with the ENTIRE
audio lane removed (operator ratified: port the two-stage HQ recipe to silent
ltx_video; 2 LTX rows, NO ltx_lowvram). These tests pin:

  * recipe detection mirrors eng_ltx_av (dev -> hq_two_stage; distilled ->
    single_pass; unknown family RAISES; explicit invalid RAISES; hq on a
    distilled unet RAISES);
  * TWO-STAGE topology: base motion at HALF canvas -> LTXVLatentUpsampler x2
    -> hard re-anchor -> 3-step refine -> decode; NO audio nodes anywhere;
  * Inplace anchors 0.7 base / 1.0 refine; ancestral base + plain refine
    samplers; front-loaded base ladder + 0.85->0 refine ladder; CropGuides on
    the refine only, anchored on the BASE latent; seed/seed+1 noise streams;
  * distilled-lora-384 (original, non-1.1) @ 0.5;
  * guide chain ImageScale(1920x1088) -> ResizeImagesByLongerEdge(1536) ->
    LTXVPreprocess(18) -- canvas-independent fixed numbers;
  * BUG-414 videovae enc/dec split;
  * init image REQUIRED (NO FALLBACKS -- no silent t2v downgrade);
  * single_pass stays byte-identical (the frozen GGUF-mini graph unchanged).
UTF-8, no BOM, ASCII-only source.
"""
from __future__ import annotations

import pytest

from nodes._otr_video_engines import wrapper_bridge as wb
from nodes._otr_video_engines.eng_ltx_video import (
    LTX_DISTILLED_SIGMAS,
    LTX_HQ_REFINE_SIGMAS,
    LtxVideoEngine,
    RECIPE_HQ_TWO_STAGE,
    RECIPE_SINGLE_PASS,
)
from nodes._otr_video_engines.registry import EngineUnusable

W = wb.Wire
DEV_UNET = "ltx-2.3-22b-dev-Q3_K_M.gguf"
DISTILLED_UNET = "ltx-2.3-22b-distilled-1.1-Q3_K_M.gguf"

_AUDIO_NODES = ("audiovae", "loadaudio", "audioenc", "solidmask",
                "noisemask", "concat", "concat_refine", "separate",
                "separate_base")


@pytest.fixture()
def hq_env(monkeypatch):
    monkeypatch.delenv("OTR_LTX_VIDEO_RECIPE", raising=False)
    monkeypatch.delenv("OTR_LTX_HQ_LORA", raising=False)
    monkeypatch.setenv("OTR_LTX_VIDEO_UNET", DEV_UNET)
    return LtxVideoEngine()


def _graph(eng, image="still.png", w=832, h=480, length=121, seed=7):
    plan = {"text_prompt": "a vintage radio station interior", "seed": seed,
            "target_frame_count": length}
    return eng._build_graph_hq(plan, length, w, h, image)


# -- recipe detection (mirrors eng_ltx_av) ----------------------------------


def test_dev_default_recipe_is_hq(hq_env):
    assert hq_env._recipe() == RECIPE_HQ_TWO_STAGE


def test_distilled_default_recipe_is_single_pass(hq_env, monkeypatch):
    monkeypatch.setenv("OTR_LTX_VIDEO_UNET", DISTILLED_UNET)
    assert hq_env._recipe() == RECIPE_SINGLE_PASS


def test_unknown_family_raises(hq_env, monkeypatch):
    monkeypatch.setenv("OTR_LTX_VIDEO_UNET", "mystery-model.gguf")
    with pytest.raises(EngineUnusable):
        hq_env._recipe()


def test_invalid_recipe_env_raises(hq_env, monkeypatch):
    monkeypatch.setenv("OTR_LTX_VIDEO_RECIPE", "turbo_mode")
    with pytest.raises(EngineUnusable):
        hq_env._recipe()


def test_hq_on_distilled_unet_raises(hq_env, monkeypatch):
    monkeypatch.setenv("OTR_LTX_VIDEO_UNET", DISTILLED_UNET)
    monkeypatch.setenv("OTR_LTX_VIDEO_RECIPE", "hq_two_stage")
    with pytest.raises(EngineUnusable):
        hq_env._recipe()


def test_explicit_single_pass_on_dev_ok(hq_env, monkeypatch):
    monkeypatch.setenv("OTR_LTX_VIDEO_RECIPE", "single_pass")
    assert hq_env._recipe() == RECIPE_SINGLE_PASS


# -- the silent two-stage topology ------------------------------------------


def test_no_audio_nodes_anywhere(hq_env):
    g = _graph(hq_env)
    for node in _AUDIO_NODES:
        assert node not in g, node
    cands = hq_env._node_candidates_hq()
    for node in _AUDIO_NODES:
        assert node not in cands, node


def test_two_stage_topology(hq_env):
    g = _graph(hq_env)
    # STAGE A: half-canvas empty latent + soft inplace anchor; the sampler's
    # latent is the inplace output DIRECT (pure video latent -- no AV concat)
    assert g["emptylatent"]["inputs"]["width"] == 416      # 832 // 2
    assert g["emptylatent"]["inputs"]["height"] == 240     # 480 // 2
    assert g["emptylatent"]["inputs"]["length"] == 121
    assert g["inplace_base"]["inputs"]["strength"] == 0.7
    assert g["inplace_base"]["inputs"]["latent"] == W("emptylatent", 0)
    assert g["sampler"]["inputs"]["latent_image"] == W("inplace_base", 0)
    # STAGE B: upsample the BASE sample x2 -> hard re-anchor -> refine
    assert g["upscaler"]["inputs"]["samples"] == W("sampler", 0)
    assert g["inplace_refine"]["inputs"]["latent"] == W("upscaler", 0)
    assert g["inplace_refine"]["inputs"]["strength"] == 1.0
    assert g["sampler_refine"]["inputs"]["latent_image"] == W("inplace_refine", 0)
    # final: decode the refine output straight (no separate -- video-only)
    assert g["decode"]["inputs"]["samples"] == W("sampler_refine", 0)


def test_samplers_and_ladders(hq_env):
    g = _graph(hq_env)
    assert g["ksel"]["inputs"]["sampler_name"] == "euler_ancestral_cfg_pp"
    assert g["sigmas"]["inputs"]["values"] == list(LTX_DISTILLED_SIGMAS)
    assert g["ksel_refine"]["inputs"]["sampler_name"] == "euler_cfg_pp"
    assert g["sigmas_refine"]["inputs"]["values"] == list(LTX_HQ_REFINE_SIGMAS)
    # two INDEPENDENT deterministic noise streams per beat
    assert g["noise"]["inputs"]["noise_seed"] == 7
    assert g["noise_refine"]["inputs"]["noise_seed"] == 8


def test_cropguides_refine_only(hq_env):
    g = _graph(hq_env)
    assert g["guider"]["inputs"]["positive"] == W("cond", 0)
    assert g["guider"]["inputs"]["negative"] == W("cond", 1)
    # the refine guider goes THROUGH CropGuides anchored on the BASE latent
    assert g["cropguides"]["inputs"]["latent"] == W("sampler", 0)
    assert g["guider_refine"]["inputs"]["positive"] == W("cropguides", 0)
    assert g["guider_refine"]["inputs"]["negative"] == W("cropguides", 1)


def test_half_distilled_lora(hq_env):
    g = _graph(hq_env)
    assert g["lora"]["inputs"]["model"] == W("unet", 0)
    assert g["lora"]["inputs"]["strength_model"] == 0.5
    assert g["lora"]["inputs"]["lora_name"].endswith(
        "ltx-2.3-22b-distilled-lora-384.safetensors")
    assert "1.1" not in g["lora"]["inputs"]["lora_name"]
    assert g["guider"]["inputs"]["model"] == W("lora", 0)
    assert g["guider_refine"]["inputs"]["model"] == W("lora", 0)


def test_guide_chain_canvas_independent(hq_env):
    g = _graph(hq_env, w=512, h=288)   # tiny canvas: guide numbers UNCHANGED
    assert g["imagescale"]["inputs"]["width"] == 1920
    assert g["imagescale"]["inputs"]["height"] == 1088
    assert g["resizelonger"]["inputs"]["longer_edge"] == 1536
    assert g["preprocess"]["inputs"]["img_compression"] == 18
    # both inplace anchors consume the SAME preprocessed guide
    assert g["inplace_base"]["inputs"]["image"] == W("preprocess", 0)
    assert g["inplace_refine"]["inputs"]["image"] == W("preprocess", 0)


def test_videovae_enc_dec_split(hq_env):
    g = _graph(hq_env)
    # BUG-414: encode-side VAE feeds inplace/upscaler; decode-side feeds decode
    assert g["inplace_base"]["inputs"]["vae"] == W("videovae_enc", 0)
    assert g["inplace_refine"]["inputs"]["vae"] == W("videovae_enc", 0)
    assert g["upscaler"]["inputs"]["vae"] == W("videovae_enc", 0)
    assert g["decode"]["inputs"]["vae"] == W("videovae_dec", 0)


def test_init_image_required_no_fallback(hq_env):
    with pytest.raises(wb.GraphExecutionError):
        _graph(hq_env, image="")


# -- single_pass stays frozen ------------------------------------------------


def test_single_pass_graph_unchanged(hq_env, monkeypatch):
    """The frozen GGUF-mini graph is byte-identical under the recipe layer."""
    monkeypatch.setenv("OTR_LTX_VIDEO_RECIPE", "single_pass")
    plan = {"text_prompt": "x", "seed": 1, "target_frame_count": 121}
    g = hq_env._build_graph(plan, 121, 832, 480)
    assert g["lora"]["inputs"]["strength_model"] == 0.7
    assert g["lora"]["inputs"]["lora_name"].endswith(
        "ltx-2.3-22b-distilled-lora-384-1.1.safetensors")
    assert g["sampleradv"]["inputs"]["latent_image"] == W("latent", 0)
    assert "upscaler" not in g and "inplace_base" not in g


def test_weight_paths_recipe_aware(hq_env, monkeypatch):
    hq_labels = [p[0] for p in hq_env._weight_paths()]
    assert "latent spatial upscaler" in hq_labels
    assert "HQ distilled LoRA (384)" in hq_labels
    monkeypatch.setenv("OTR_LTX_VIDEO_RECIPE", "single_pass")
    sp_labels = [p[0] for p in hq_env._weight_paths()]
    assert "latent spatial upscaler" not in sp_labels
    assert "distilled LoRA" in sp_labels
