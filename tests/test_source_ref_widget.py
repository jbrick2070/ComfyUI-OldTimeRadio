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


def test_source_ref_is_final_append_only_widget():
    spec = OTR_LedgerScriptWriter.INPUT_TYPES()
    order = list(spec["required"].keys()) + list(spec["optional"].keys())

    assert order[23] == "source_bank"
    assert order[24] == "visual_style"
    assert order[25] == "google_api_slot_a_model"
    assert order[26] == "google_api_slot_b_model"
    assert order[27] == "source_ref"
    assert len(order) == 28

    source_ref_type, meta = spec["optional"]["source_ref"]
    assert source_ref_type == "STRING"
    assert meta["default"] == ""


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


def test_refine_core_carries_source_ref(monkeypatch):
    from nodes import _otr_story_select as sel

    class _Rcfg:
        target_grade = "B"
        bar = 0
        effective_passes = 2

    monkeypatch.setattr(sel, "resolve_refine_passes", lambda *_a, **_k: _Rcfg())
    captured = {}

    def _fake_loop(self, _rcfg, _core):
        captured.update(_core)
        return ("", "", "", 0, "")

    monkeypatch.setattr(OTR_LedgerScriptWriter, "_refine_loop", _fake_loop)
    node = OTR_LedgerScriptWriter()
    out = node.run(source_ref="archive://fixture-001", refine_target_grade="B")

    assert out == ("", "", "", 0, "")
    assert captured["source_ref"] == "archive://fixture-001"
    assert "os" not in captured and "_scaffold" not in captured


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

    assert len(node1["widgets_values"]) == 28
    assert node1["widgets_values"][23] == "science_news"
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
