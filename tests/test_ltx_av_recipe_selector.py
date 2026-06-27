"""CPU tests for the LTX-AV recipe selector (the recipe FOLLOWS THE MODEL).

Pins the 2026-06-26 refactor of the old binary OTR_LTX_AV_SHARP flag into the
tri-state resolver eng_ltx_av._recipe() consulted at all four sites
(_weight_paths / _node_candidates / _build_graph / render_clip). NO FALLBACKS:
an ambiguous unet, a bad override, or a retired flag must RAISE. Graph-wiring
assertions verify LoRA / ModelSampling / scheduler / sigmas per recipe + the
residency keep-set. UTF-8, no BOM, ASCII-only source.
"""
from __future__ import annotations

import pytest

from nodes._otr_video_engines import wrapper_bridge as wb
from nodes._otr_video_engines.registry import EngineUnusable
from nodes._otr_video_engines.eng_ltx_av import (
    LtxAudioInEngine,
    _recipe_config,
    RECIPE_SHARP_LORA,
    RECIPE_DISTILLED_NATIVE,
    RECIPE_M0_BASE,
)

DEV_UNET = "ltx-2.3-22b-dev-Q3_K_M.gguf"
DISTILLED_UNET = r"distilled-1.1\ltx-2.3-22b-distilled-1.1-Q3_K_M.gguf"


def _clean(monkeypatch):
    monkeypatch.delenv("OTR_LTX_AV_SHARP", raising=False)
    monkeypatch.delenv("OTR_LTX_AV_RECIPE", raising=False)


# --------------------------------------------------------------------------- #
# recipe resolution (fail-loud)
# --------------------------------------------------------------------------- #
def test_auto_distilled_filename_resolves_distilled_native(monkeypatch):
    _clean(monkeypatch)
    monkeypatch.setenv("OTR_LTX_AV_UNET", DISTILLED_UNET)
    assert LtxAudioInEngine()._recipe() == RECIPE_DISTILLED_NATIVE


def test_auto_dev_filename_resolves_sharp_lora(monkeypatch):
    _clean(monkeypatch)
    monkeypatch.setenv("OTR_LTX_AV_UNET", DEV_UNET)
    assert LtxAudioInEngine()._recipe() == RECIPE_SHARP_LORA


def test_explicit_override_wins_over_filename(monkeypatch):
    _clean(monkeypatch)
    monkeypatch.setenv("OTR_LTX_AV_UNET", DISTILLED_UNET)  # would auto -> native
    monkeypatch.setenv("OTR_LTX_AV_RECIPE", "sharp_lora")
    assert LtxAudioInEngine()._recipe() == RECIPE_SHARP_LORA


def test_ambiguous_unet_auto_raises(monkeypatch):
    _clean(monkeypatch)
    monkeypatch.setenv("OTR_LTX_AV_UNET", "some-unknown-model-Q4.gguf")
    with pytest.raises(EngineUnusable):
        LtxAudioInEngine()._recipe()


def test_unknown_distilled_version_auto_raises(monkeypatch):
    # a distilled GGUF that does NOT match the proven distilled-1.1 family prefix
    # (e.g. a future distilled-2.0) matches NEITHER strict prefix -> fail loud,
    # forcing the operator to confirm the recipe with OTR_LTX_AV_RECIPE rather
    # than silently assuming the 1.1 recipe still applies.
    _clean(monkeypatch)
    monkeypatch.setenv("OTR_LTX_AV_UNET", "ltx-2.3-22b-distilled-2.0-Q3_K_M.gguf")
    with pytest.raises(EngineUnusable):
        LtxAudioInEngine()._recipe()


def test_bad_recipe_env_raises(monkeypatch):
    _clean(monkeypatch)
    monkeypatch.setenv("OTR_LTX_AV_UNET", DEV_UNET)
    monkeypatch.setenv("OTR_LTX_AV_RECIPE", "distilled")  # not a valid value
    with pytest.raises(EngineUnusable):
        LtxAudioInEngine()._recipe()


def test_retired_sharp_flag_raises(monkeypatch):
    _clean(monkeypatch)
    monkeypatch.setenv("OTR_LTX_AV_UNET", DEV_UNET)
    monkeypatch.setenv("OTR_LTX_AV_SHARP", "1")  # retired -> must raise
    with pytest.raises(EngineUnusable):
        LtxAudioInEngine()._recipe()


# --------------------------------------------------------------------------- #
# graph wiring per recipe (override pins the recipe; unet name then irrelevant)
# --------------------------------------------------------------------------- #
def _build(monkeypatch, recipe):
    _clean(monkeypatch)
    monkeypatch.setenv("OTR_LTX_AV_UNET", DEV_UNET)
    monkeypatch.setenv("OTR_LTX_AV_RECIPE", recipe)
    eng = LtxAudioInEngine()
    plan = {"text_prompt": "x", "seed": 0}
    # image_name non-empty -> i2v branch (so the strength widget is present)
    g = eng._build_graph(plan, 105, 832, 480, "a.wav", "i.png")
    cand = eng._node_candidates()
    return eng, g, cand


def test_sharp_lora_wiring(monkeypatch):
    eng, g, cand = _build(monkeypatch, "sharp_lora")
    assert "lora" in cand and "modelsampling" not in cand and "sched" not in cand
    assert "lora" in g and "modelsampling" not in g
    assert "sigmas" in g and "sched" not in g
    assert g["guider"]["inputs"]["model"] == wb.Wire("lora", 0)
    assert g["i2v"]["inputs"]["strength"] == 0.75
    assert any("LoRA" in p[0] for p in eng._weight_paths())
    assert eng._keep_set("decode", _recipe_config(RECIPE_SHARP_LORA)) == {
        "unet", "decode", "lora"}


def test_distilled_native_wiring(monkeypatch):
    eng, g, cand = _build(monkeypatch, "distilled_native")
    assert "lora" not in cand and "modelsampling" not in cand and "sched" not in cand
    assert "lora" not in g and "modelsampling" not in g
    assert "sigmas" in g and "sched" not in g
    # the distilled unet feeds the guider DIRECTLY (no LoRA, no ModelSampling)
    assert g["guider"]["inputs"]["model"] == wb.Wire("unet", 0)
    assert g["i2v"]["inputs"]["strength"] == 0.75
    assert not any("LoRA" in p[0] for p in eng._weight_paths())
    assert eng._keep_set("decode", _recipe_config(RECIPE_DISTILLED_NATIVE)) == {
        "unet", "decode"}


def test_m0_base_wiring(monkeypatch):
    eng, g, cand = _build(monkeypatch, "m0_base")
    assert "modelsampling" in cand and "sched" in cand and "lora" not in cand
    assert "modelsampling" in g and "lora" not in g
    assert "sched" in g and "sigmas" not in g
    assert g["guider"]["inputs"]["model"] == wb.Wire("modelsampling", 0)
    assert g["i2v"]["inputs"]["strength"] == 1.0
    assert not any("LoRA" in p[0] for p in eng._weight_paths())
    assert eng._keep_set("decode", _recipe_config(RECIPE_M0_BASE)) == {
        "unet", "decode", "modelsampling"}


def test_recipe_config_distilled_native_is_sharp_minus_lora():
    sharp = _recipe_config(RECIPE_SHARP_LORA)
    native = _recipe_config(RECIPE_DISTILLED_NATIVE)
    # identical sampling recipe; the ONLY difference is the LoRA
    assert native["use_lora"] is False and sharp["use_lora"] is True
    for k in ("use_modelsampling", "manual_sigmas", "sampler", "cfg", "i2v_strength"):
        assert native[k] == sharp[k]
