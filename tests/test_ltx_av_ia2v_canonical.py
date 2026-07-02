"""CPU topology lock for the ia2v_canonical recipe (2026-07-02 transplant).

The canonical comfy.org "LTX-2.3: Image Audio to Video" LIP-SYNC graph,
transplanted into eng_ltx_av behind RECIPE_IA2V (the new DEV-family default;
operator GO after the live isolation smoke proved articulation on OUR radio
still -- docs/2026-07-02-canonical-ia2v/). These tests pin every mechanism
the smoke proved mattered, so a future edit cannot silently regress back to
the frozen-mouth single-pass:

  * TWO-STAGE: base motion pass at HALF canvas -> LTXVLatentUpsampler x2 ->
    audio RE-CONCAT -> 3-step refine;
  * Inplace i2v anchors: 0.7 base (soft -- lets audio actuate the mouth),
    1.0 refine (hard re-anchor);
  * audio latent FROZEN under SetLatentNoiseMask(SolidMask(0));
  * ancestral base sampler, plain refine sampler; front-loaded base ladder,
    0.85->0 refine ladder; CropGuides on the REFINE only;
  * distilled-lora-384 (original, non-1.1) @ 0.5 on the DEV unet;
  * guide chain ImageScale(1.5x) -> ResizeImagesByLongerEdge(1536) ->
    LTXVPreprocess(18);
  * BUG-414 videovae enc/dec split preserved across both stages;
  * i2v REQUIRED (NO FALLBACKS -- no silent t2v downgrade).
UTF-8, no BOM, ASCII-only source.
"""
from __future__ import annotations

import pytest

from nodes._otr_video_engines import wrapper_bridge as wb
from nodes._otr_video_engines.eng_ltx_av import (
    IA2V_REFINE_SIGMAS,
    LTX_DISTILLED_SIGMAS,
    LtxAudioInEngine,
    RECIPE_IA2V,
    _recipe_config,
)

W = wb.Wire
DEV_UNET = "ltx-2.3-22b-dev-Q3_K_M.gguf"


@pytest.fixture()
def ia2v_env(monkeypatch):
    monkeypatch.delenv("OTR_LTX_AV_RECIPE", raising=False)
    monkeypatch.delenv("OTR_LTX_AV_DISTILLED_LORA", raising=False)
    monkeypatch.setenv("OTR_LTX_AV_UNET", DEV_UNET)
    return LtxAudioInEngine()


def _graph(eng, image="still.png", w=832, h=480, length=121, seed=7):
    plan = {"text_prompt": "the radio is talking as it speaks", "seed": seed,
            "target_frame_count": length}
    return eng._build_graph(plan, length, w, h, "voice.wav", image)


def test_dev_default_recipe_is_ia2v(ia2v_env):
    assert ia2v_env._recipe() == RECIPE_IA2V


def test_two_stage_topology(ia2v_env):
    g = _graph(ia2v_env)
    # STAGE A: half-canvas empty latent + soft inplace anchor + masked audio
    assert g["emptylatent"]["inputs"]["width"] == 416      # 832 // 2
    assert g["emptylatent"]["inputs"]["height"] == 240     # 480 // 2
    assert g["emptylatent"]["inputs"]["length"] == 121
    assert g["inplace_base"]["inputs"]["strength"] == 0.7
    assert g["inplace_base"]["inputs"]["latent"] == W("emptylatent", 0)
    assert g["concat"]["inputs"]["video_latent"] == W("inplace_base", 0)
    assert g["concat"]["inputs"]["audio_latent"] == W("noisemask", 0)
    # the audio latent rides FROZEN under a ZERO mask
    assert g["solidmask"]["inputs"]["value"] == 0.0
    assert g["noisemask"]["inputs"]["samples"] == W("audioenc", 0)
    assert g["noisemask"]["inputs"]["mask"] == W("solidmask", 0)
    # STAGE B: separate -> upsample x2 -> hard re-anchor -> audio RE-CONCAT
    assert g["separate_base"]["inputs"]["av_latent"] == W("sampler", 0)
    assert g["upscaler"]["inputs"]["samples"] == W("separate_base", 0)
    assert g["inplace_refine"]["inputs"]["latent"] == W("upscaler", 0)
    assert g["inplace_refine"]["inputs"]["strength"] == 1.0
    assert g["concat_refine"]["inputs"]["video_latent"] == W("inplace_refine", 0)
    assert g["concat_refine"]["inputs"]["audio_latent"] == W("separate_base", 1)
    assert g["sampler_refine"]["inputs"]["latent_image"] == W("concat_refine", 0)
    # final: separate the REFINE output; decode VIDEO only (audio = mux-LAST)
    assert g["separate"]["inputs"]["av_latent"] == W("sampler_refine", 0)
    assert g["decode"]["inputs"]["samples"] == W("separate", 0)
    # the single-pass node must be GONE (no plain LTXVImgToVideo in ia2v)
    assert "i2v" not in g


def test_samplers_and_ladders(ia2v_env):
    g = _graph(ia2v_env)
    # base = ancestral over the front-loaded 8-step ladder (motion decisions)
    assert g["ksel"]["inputs"]["sampler_name"] == "euler_ancestral_cfg_pp"
    assert g["sigmas"]["inputs"]["values"] == list(LTX_DISTILLED_SIGMAS)
    # refine = plain euler_cfg_pp over the 3-step 0.85->0 ladder (detail only)
    assert g["ksel_refine"]["inputs"]["sampler_name"] == "euler_cfg_pp"
    assert g["sigmas_refine"]["inputs"]["values"] == list(IA2V_REFINE_SIGMAS)
    # two INDEPENDENT deterministic noise streams per beat
    assert g["noise"]["inputs"]["noise_seed"] == 7
    assert g["noise_refine"]["inputs"]["noise_seed"] == 8


def test_cropguides_refine_only(ia2v_env):
    g = _graph(ia2v_env)
    # base guider takes the conditioning DIRECT...
    assert g["guider"]["inputs"]["positive"] == W("cond", 0)
    assert g["guider"]["inputs"]["negative"] == W("cond", 1)
    # ...the refine guider goes THROUGH CropGuides anchored on the base latent
    assert g["cropguides"]["inputs"]["latent"] == W("separate_base", 0)
    assert g["guider_refine"]["inputs"]["positive"] == W("cropguides", 0)
    assert g["guider_refine"]["inputs"]["negative"] == W("cropguides", 1)


def test_half_distilled_lora_on_dev(ia2v_env):
    g = _graph(ia2v_env)
    assert g["lora"]["inputs"]["model"] == W("unet", 0)
    assert g["lora"]["inputs"]["strength_model"] == 0.5
    assert g["lora"]["inputs"]["lora_name"].endswith(
        "ltx-2.3-22b-distilled-lora-384.safetensors")
    assert "1.1" not in g["lora"]["inputs"]["lora_name"]
    # both guiders ride the SAME LoRA-wrapped model
    assert g["guider"]["inputs"]["model"] == W("lora", 0)
    assert g["guider_refine"]["inputs"]["model"] == W("lora", 0)


def test_guide_image_conditioning_chain(ia2v_env):
    g = _graph(ia2v_env, w=832, h=480)
    assert g["imagescale"]["inputs"]["image"] == W("loadimage", 0)
    assert g["imagescale"]["inputs"]["width"] == 1248     # 832 * 1.5
    assert g["imagescale"]["inputs"]["height"] == 720     # 480 * 1.5
    assert g["resizelonger"]["inputs"]["longer_edge"] == 1536
    assert g["preprocess"]["inputs"]["img_compression"] == 18
    # BOTH inplace anchors condition on the SAME preprocessed guide
    assert g["inplace_base"]["inputs"]["image"] == W("preprocess", 0)
    assert g["inplace_refine"]["inputs"]["image"] == W("preprocess", 0)


def test_videovae_split_survives_two_stage(ia2v_env):
    # BUG-414 lock, two-stage edition: encode-side VAE is a distinct node whose
    # last consumer runs BEFORE each sampler peak; decode rides its own node.
    g = _graph(ia2v_env)
    assert "videovae" not in g
    for consumer in ("inplace_base", "inplace_refine", "upscaler"):
        assert g[consumer]["inputs"]["vae"] == W("videovae_enc", 0)
    assert g["decode"]["inputs"]["vae"] == W("videovae_dec", 0)


def test_ia2v_requires_init_image(ia2v_env):
    with pytest.raises(Exception):
        _graph(ia2v_env, image="")


def test_node_candidates_and_weights_gated_by_recipe(ia2v_env, monkeypatch):
    cands = ia2v_env._node_candidates()
    for logical in ("inplace_base", "upscaler", "cropguides", "noisemask",
                    "preprocess", "sampler_refine", "concat_refine"):
        assert logical in cands
    labels = [lbl for lbl, _p, _f in ia2v_env._weight_paths()]
    assert any("upscaler" in lbl for lbl in labels)
    assert any("LoRA" in lbl for lbl in labels)
    # distilled_native stays lean: none of the two-stage classes/weights
    monkeypatch.setenv("OTR_LTX_AV_RECIPE", "distilled_native")
    monkeypatch.setenv(
        "OTR_LTX_AV_UNET",
        r"distilled-1.1\ltx-2.3-22b-distilled-1.1-Q3_K_M.gguf")
    lean = LtxAudioInEngine()
    assert "upscaler" not in lean._node_candidates()
    lean_labels = [lbl for lbl, _p, _f in lean._weight_paths()]
    assert not any("upscaler" in lbl for lbl in lean_labels)


def test_keep_set_keeps_lora(ia2v_env):
    rcfg = _recipe_config(RECIPE_IA2V)
    keep = ia2v_env._keep_set("decode", rcfg)
    assert keep == {"unet", "lora", "decode"}
