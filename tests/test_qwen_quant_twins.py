"""Two Qwen 3.5 dropdown identities -- NF4 vs full -- must stay honest.

A single Hugging Face snapshot cannot wear both ``nv8`` (the NF4 load) and
Quant ``none`` (canonical / Mac). The picker therefore carries two ids that
survive ``_strip_label_suffix``. Users can still pick Gemma, Llama, a
cache-discovered CausalLM, or a cloud slot; those rows do not own Quant.
"""
from __future__ import annotations

import pytest

from nodes import _otr_model_catalog as catalog
from nodes._otr_loader_backends import chat_template_kwargs
from nodes._otr_model_loader import ModelLoaderError, request_slot
from nodes._otr_shared import llm_policy as lp
from nodes._otr_workflow_apply import _llm_option_value


def test_the_two_qwen_ids_do_not_collapse_under_strip():
    full = catalog.DEFAULT_LLM
    nf4 = catalog.DEFAULT_LLM_NF4
    assert full != nf4
    assert catalog._strip_label_suffix(full + catalog.vram_badge_for(full)) == full
    assert catalog._strip_label_suffix(nf4 + catalog.vram_badge_for(nf4)) == nf4
    assert catalog.hf_weights_id(full) == "Qwen/Qwen3.5-4B"
    assert catalog.hf_weights_id(nf4) == "Qwen/Qwen3.5-4B"


def test_fit_tags_tell_the_truth():
    full_tags = catalog.fit_tags_for(catalog.DEFAULT_LLM)
    nf4_tags = catalog.fit_tags_for(catalog.DEFAULT_LLM_NF4)
    assert "nv8" not in full_tags
    assert "mac16-tight" in full_tags or "mac16" in full_tags
    assert "nv16" in full_tags and "nv24" in full_tags
    assert "nv8" in nf4_tags
    assert "nv16" in nf4_tags and "nv24" in nf4_tags
    assert not any(t.startswith("mac16") for t in nf4_tags)


def test_resident_estimate_follows_the_pick():
    full = catalog._estimate_resident_gb(catalog.DEFAULT_LLM)
    nf4 = catalog._estimate_resident_gb(catalog.DEFAULT_LLM_NF4)
    assert full == pytest.approx(8.68, abs=0.05)
    assert nf4 == pytest.approx(4.34, abs=0.05)


def test_resolve_pick_for_quant_maps_the_family_not_gemma():
    assert catalog.resolve_pick_for_quant(
        catalog.DEFAULT_LLM, "bnb_nf4"
    ) == catalog.DEFAULT_LLM_NF4
    assert catalog.resolve_pick_for_quant(
        catalog.DEFAULT_LLM_NF4, "none"
    ) == catalog.DEFAULT_LLM
    assert catalog.resolve_pick_for_quant(
        "google/gemma-4-12b-it", "bnb_nf4"
    ) == "google/gemma-4-12b-it"


def test_mismatch_fails_loud_only_when_the_pick_owns_quant():
    assert catalog.quant_pick_mismatch(catalog.DEFAULT_LLM, "bnb_nf4")
    assert catalog.quant_pick_mismatch(catalog.DEFAULT_LLM_NF4, "none")
    assert catalog.quant_pick_mismatch(catalog.DEFAULT_LLM, "none") is None
    assert catalog.quant_pick_mismatch(catalog.DEFAULT_LLM_NF4, "bnb_nf4") is None
    assert catalog.quant_pick_mismatch("google/gemma-4-12b-it", "none") is None


def test_chat_template_kwargs_fire_for_both_identities():
    assert chat_template_kwargs(catalog.DEFAULT_LLM) == {"enable_thinking": False}
    assert chat_template_kwargs(catalog.DEFAULT_LLM_NF4) == {"enable_thinking": False}
    assert chat_template_kwargs(catalog.DEFAULT_LLM_NF4 + catalog.vram_badge_for(
        catalog.DEFAULT_LLM_NF4
    )) == {"enable_thinking": False}
    assert chat_template_kwargs("google/gemma-4-12b-it") == {}


def test_fresh_node_default_is_the_nf4_pick():
    assert catalog.fresh_llm_option().startswith(catalog.DEFAULT_LLM_NF4 + " (")
    assert catalog.default_llm_option().startswith(catalog.DEFAULT_LLM + " (")
    assert "nv8" not in catalog.default_llm_option()
    assert "nv8" in catalog.fresh_llm_option()


def test_applier_composes_profile_family_plus_quant():
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
    assert catalog._strip_label_suffix(nf4_label) == catalog.DEFAULT_LLM_NF4
    assert catalog._strip_label_suffix(full_label) == catalog.DEFAULT_LLM


def test_request_slot_refuses_full_pick_with_nf4_quant(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("mismatch must fail before load")

    monkeypatch.setattr("nodes._otr_model_loader.load_llm", forbidden)
    with pytest.raises(ModelLoaderError, match="owns the load|matching Qwen"):
        request_slot(
            "creative",
            catalog.DEFAULT_LLM,
            policy=lp.LLMRuntimePolicy(vram_ceiling_gb=0),
        )
