"""Source Banks v2 `source_ref` surface guardrails.

This chunk only adds the inert, append-only source-reference surface. Bank
fetchers consume it in later chunks; until then blank is byte-stable and
nonblank is just preserved for downstream fail-loud consumers.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
_SCRIPTS = _REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from nodes.OTR_LedgerScriptWriter import (  # noqa: E402
    OTR_LedgerScriptWriter,
    _resolve_inputs,
)

_CANONICAL_WORKFLOW = _REPO / "workflows" / "otr_canonical.json"


def test_source_ref_slot_pinned_with_llm_policy_tail():
    """S5 platform-portability (2026-07-10): source_ref is NO LONGER the
    final append-only widget -- OLD pin: order[27] == "source_ref" AND
    len(order) == 28 (the tail). NEW pin: source_ref stays fixed at
    order[27] (never moves), but the six explicit LLM runtime-policy
    widgets (llm_device .. gguf_quant) were appended after it at
    order[28:34], and the gate_in forceInput socket is now the actual
    final declared entry at order[34] (len(order) == 35). Renamed from
    test_source_ref_is_final_append_only_widget since that name would now
    lie -- source_ref is pinned mid-vector, not the tail.
    """
    spec = OTR_LedgerScriptWriter.INPUT_TYPES()
    order = list(spec["required"].keys()) + list(spec["optional"].keys())

    assert order[23] == "source_bank"
    assert order[24] == "visual_style"
    assert order[25] == "google_api_slot_a_model"
    assert order[26] == "google_api_slot_b_model"
    assert order[27] == "source_ref"
    assert order[28] == "llm_device"
    assert order[29] == "llm_attn_impl"
    assert order[30] == "llm_quant_policy"
    assert order[31] == "llm_vram_ceiling_gb"
    assert order[32] == "gguf_n_ctx"
    assert order[33] == "gguf_quant"
    assert order[34] == "gate_in"
    assert len(order) == 35

    source_ref_type, meta = spec["optional"]["source_ref"]
    assert source_ref_type == "STRING"
    assert meta["default"] == ""

    # gate_in is declared (order[34]) but is a forceInput socket -- it
    # consumes NO widgets_values slot (verified against the live 34-wide
    # vector in test_patch_widget_by_name_lands_source_ref_slot_27 and
    # tests/test_otr_api_companions.py::test_round_trip_canonical_node1_
    # inputs_correct).
    gate_type, gate_meta = spec["optional"]["gate_in"]
    assert gate_type == "STRING"
    assert gate_meta["forceInput"] is True


def test_source_ref_is_real_run_parameter_and_resolved_value():
    params = inspect.signature(OTR_LedgerScriptWriter.run).parameters
    assert "source_ref" in params

    resolved = _resolve_inputs(custom_premise="seed")
    assert resolved["source_ref"] == ""

    resolved2 = _resolve_inputs(
        custom_premise="seed",
        source_ref="https://example.invalid/source.txt",
    )
    assert resolved2["source_ref"] == "https://example.invalid/source.txt"



def test_source_ref_on_both_headless_whitelists():
    from nodes._otr_workflow_apply import CREATIVE_WHITELIST as pkg_wl
    import otr_api

    assert "source_ref" in pkg_wl
    assert "source_ref" in otr_api.CREATIVE_WHITELIST


def test_patch_widget_by_name_lands_source_ref_slot_27():
    import otr_api

    spec = OTR_LedgerScriptWriter.INPUT_TYPES()
    schemas = {
        "OTR_LedgerScriptWriter": {
            "input": {
                "required": spec["required"],
                "optional": spec["optional"],
            },
        },
    }
    workflow = otr_api.load_workflow(str(_CANONICAL_WORKFLOW))
    otr_api.patch_widget_by_name(
        workflow,
        1,
        "source_ref",
        "https://example.invalid/source.txt",
        schemas,
    )
    node1 = next(n for n in workflow["nodes"] if n["id"] == 1)

    # S5 platform-portability (2026-07-10): OLD pin 28 -> NEW pin 34 (the
    # six llm_device..gguf_quant widgets appended at 28-33; source_ref
    # stays at 27). schemas comes from the LIVE INPUT_TYPES() above, so
    # this already reflects the new vector -- only the pin needed updating.
    assert len(node1["widgets_values"]) == 34
    assert node1["widgets_values"][23] == "scifi_news"
    assert node1["widgets_values"][24] == "sci_fi_radio"
    assert node1["widgets_values"][25] == "(select Google API model)"
    assert node1["widgets_values"][26] == "(select Google API model)"
    assert node1["widgets_values"][27] == "https://example.invalid/source.txt"


def test_patch_creative_allows_source_ref():
    import otr_api

    spec = OTR_LedgerScriptWriter.INPUT_TYPES()
    schemas = {
        "OTR_LedgerScriptWriter": {
            "input": {
                "required": spec["required"],
                "optional": spec["optional"],
            },
        },
    }
    workflow = otr_api.load_workflow(str(_CANONICAL_WORKFLOW))
    otr_api.patch_creative(
        workflow,
        1,
        "source_ref",
        "pd://sherlock/case-001",
        schemas,
    )
    node1 = next(n for n in workflow["nodes"] if n["id"] == 1)
    assert node1["widgets_values"][27] == "pd://sherlock/case-001"
