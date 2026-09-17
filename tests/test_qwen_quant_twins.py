"""One Qwen identity -- Quant is baked from the machine, not a twin.

NVIDIA/cuda loads NF4; Mac/CPU loads full. The retired ``:nf4`` spelling
still validates onto the same row. Gemma 4 12B bakes NF4 with one identity.
"""
from __future__ import annotations

import pytest

from nodes import _otr_model_catalog as catalog
from nodes._otr_loader_backends import chat_template_kwargs
from nodes._otr_model_loader import request_slot
from nodes._otr_shared import llm_policy as lp
from nodes._otr_workflow_apply import _llm_option_value


def test_retired_nf4_spelling_collapses_to_the_one_qwen():
    full = catalog.DEFAULT_LLM
    nf4 = catalog.DEFAULT_LLM_NF4
    assert full != nf4
    assert catalog._canonical_qwen_id(nf4) == full
    assert catalog._canonical_qwen_id(nf4 + catalog.vram_badge_for(full)) == full
    assert catalog.hf_weights_id(full) == "Qwen/Qwen3.5-4B"
    assert catalog.hf_weights_id(nf4) == "Qwen/Qwen3.5-4B"


def test_one_qwen_fit_tags_cover_nv8_and_mac():
    tags = catalog.fit_tags_for(catalog.DEFAULT_LLM)
    assert "nv8" in tags
    assert "nv16" in tags and "nv24" in tags
    assert "mac16-tight" in tags or "mac16" in tags
    # Retired :nf4 spelling aliases onto the same curated row.
    assert catalog.fit_tags_for(catalog.DEFAULT_LLM_NF4) == tags


def test_resident_estimate_is_the_nf4_half_for_platform_qwen():
    est = catalog._estimate_resident_gb(catalog.DEFAULT_LLM)
    assert est == pytest.approx(4.34, abs=0.05)
    assert catalog._estimate_resident_gb(catalog.DEFAULT_LLM_NF4) == est


def test_resolve_pick_keeps_the_one_qwen_for_any_quant_widget():
    assert catalog.resolve_pick_for_quant(
        catalog.DEFAULT_LLM, "bnb_nf4",
    ) == catalog.DEFAULT_LLM
    assert catalog.resolve_pick_for_quant(
        catalog.DEFAULT_LLM, "none",
    ) == catalog.DEFAULT_LLM
    assert catalog.resolve_pick_for_quant(
        "google/gemma-4-12b-it", "bnb_nf4",
    ) == "google/gemma-4-12b-it"


def test_mismatch_does_not_crash_a_leftover_quant_widget():
    assert catalog.quant_pick_mismatch(catalog.DEFAULT_LLM, "bnb_nf4") is None
    assert catalog.quant_pick_mismatch(catalog.DEFAULT_LLM_NF4, "none") is None
    assert catalog.quant_pick_mismatch(catalog.DEFAULT_LLM, "none") is None
    assert catalog.quant_pick_mismatch("google/gemma-4-12b-it", "none") is None


def test_platform_and_gemma_bake_quant_from_the_pick():
    assert catalog.effective_quant_policy(
        "google/gemma-4-12b-it", "none",
    ) == "bnb_nf4"
    assert catalog.effective_quant_policy(
        "google/gemma-4-12b-it (23.9 GB, nv16 nv24)", "none",
    ) == "bnb_nf4"
    assert catalog.effective_quant_policy(
        "google/gemma-4-12b-it", "bnb_8bit",
    ) == "bnb_nf4"
    assert catalog.effective_quant_policy(
        catalog.DEFAULT_LLM, "none", device="cuda",
    ) == "bnb_nf4"
    assert catalog.effective_quant_policy(
        catalog.DEFAULT_LLM, "bnb_nf4", device="cuda",
    ) == "bnb_nf4"
    assert catalog.effective_quant_policy(
        catalog.DEFAULT_LLM_NF4, "none", device="cuda",
    ) == "bnb_nf4"
    assert catalog.effective_quant_policy(
        catalog.DEFAULT_LLM, "bnb_nf4", device="cpu",
    ) == "none"
    assert catalog.effective_quant_policy(
        catalog.DEFAULT_LLM, "none", device="mps",
    ) == "none"


def test_chat_template_kwargs_fire_for_the_one_qwen():
    assert chat_template_kwargs(catalog.DEFAULT_LLM) == {"enable_thinking": False}
    assert chat_template_kwargs(catalog.DEFAULT_LLM_NF4) == {
        "enable_thinking": False,
    }
    assert chat_template_kwargs("google/gemma-4-12b-it") == {}


def test_fresh_and_default_llm_options_are_the_one_qwen():
    assert catalog.fresh_llm_option() == catalog.default_llm_option()
    assert catalog.fresh_llm_option().startswith(catalog.DEFAULT_LLM + " (")
    assert "nv8" in catalog.fresh_llm_option()
    assert catalog.DEFAULT_LLM_NF4 not in catalog.dropdown_choices()


def test_applier_keeps_the_one_qwen_for_any_profile_quant():
    schemas = {
        "OTR_LedgerScriptWriter": {
            "input": {
                "required": {
                    "creative_writing_model": (
                        catalog.dropdown_choices(),
                        {},
                    ),
                },
            },
        },
    }
    nf4_label = _llm_option_value(
        "OTR_LedgerScriptWriter",
        "creative_writing_model",
        catalog.DEFAULT_LLM,
        schemas,
        quant_policy="bnb_nf4",
    )
    full_label = _llm_option_value(
        "OTR_LedgerScriptWriter",
        "creative_writing_model",
        catalog.DEFAULT_LLM,
        schemas,
        quant_policy="none",
    )
    assert catalog._strip_label_suffix(nf4_label) == catalog.DEFAULT_LLM
    assert catalog._strip_label_suffix(full_label) == catalog.DEFAULT_LLM


@pytest.mark.parametrize(
    "model_id,stale,baked",
    [
        ("google/gemma-4-12b-it", "none", "bnb_nf4"),
        (catalog.DEFAULT_LLM_NF4, "none", "bnb_nf4"),
        (catalog.DEFAULT_LLM, "none", "bnb_nf4"),
        (catalog.DEFAULT_LLM, "bnb_nf4", "bnb_nf4"),
    ],
)
def test_request_slot_bakes_quant_from_the_pick(monkeypatch, model_id, stale, baked):
    from nodes import _otr_model_loader as loader

    seen = {}

    def capture_load(mid, **kwargs):
        seen["model_id"] = mid
        seen["policy"] = kwargs.get("policy")
        return {
            "model": object(),
            "tokenizer": object(),
            "model_id": mid,
        }

    monkeypatch.setattr(loader, "load_llm", capture_load)
    monkeypatch.setattr(
        loader, "_require_transformers_model_support", lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        catalog, "auto_download_if_missing", lambda *_a, **_k: None,
    )
    loader.LLM_CACHE.update({
        "model_id": None, "slot": None, "cache_entry": None,
    })
    try:
        entry = request_slot(
            "creative",
            model_id,
            policy=lp.LLMRuntimePolicy(vram_ceiling_gb=0, quant_policy=stale),
        )
    finally:
        loader.LLM_CACHE.update({
            "model_id": None, "slot": None, "cache_entry": None,
        })
    assert entry["model_id"] == catalog._canonical_qwen_id(
        catalog._strip_label_suffix(model_id),
    )
    assert seen["policy"].quant_policy == baked
