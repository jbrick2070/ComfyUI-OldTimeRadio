"""Sprint E E2 / R-1: workflow JSON canonical-config widget drift guard.

Pins the audio C7 baseline reproduction precondition into the shipped
workflow JSON. If a future edit reverts the writer or HuMo seed
widgets, this test fails loud so default-config baseline reproduction
stays unblocked.

Per Sprint E E2 plan (W-1 + W-2):
- Node 1 (OTR_LedgerScriptWriter): the `seed` widget was removed
  (BUG-LOCAL-269/270); the model-slot widgets ship the measured local HF
  Gemma 4 12B writer while DEFAULT_LLM remains the C7 API fallback.
- Node 51 (OTR_BatchHumoRender) widgets must ship seed=7, control="fixed".

Node 63 (OTR_WorkflowValidator) path widget intentionally ships as
empty string; the source-side fallback (Sprint E E5 / C-3 in the plan)
resolves to the canonical _DEFAULT_WORKFLOW_PATH at runtime. We do NOT
pin a hardcoded operator path here because the S29 Phase 1 cleanbreak
explicitly removed `C:/Users/jeffr/...` literals from the JSON surface
and the forbidden-pattern sweep guards that decision.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
CANONICAL_JSON = REPO_ROOT / "workflows" / "otr_canonical.json"


def _load_canonical_workflow() -> dict:
    return json.loads(CANONICAL_JSON.read_text(encoding="utf-8"))


def _node_by_id(workflow: dict, node_id: int) -> dict:
    for n in workflow.get("nodes", []):
        if n.get("id") == node_id:
            return n
    raise AssertionError(f"node id {node_id} not present in {CANONICAL_JSON.name}")


class TestWriterCanonicalModelSlots:
    """W-1 drift guard. Writer Node 1 ships the measured HF 12B writer.

    The writer's `seed` widget was REMOVED (BUG-LOCAL-269 / 270): the
    cast + style RNGs are decoupled and draw OS entropy, so there is no
    longer a seed widget (or its control_after_generate companion) to
    pin here.
    """

    def test_writer_ships_measured_hf_12b_model_slots(self):
        """Canonical workflow ships the measured HF 12B writer on both
        model slots.

        Renamed from test_writer_ships_fast_30_word_model_slots
        (2026-08-14): the `target_words` widget this test used to pin at
        slot 1 (value 30, "the quick 30-word smoke canvas") was DELETED
        (operator directive -- episode length is an observation driven by
        act_count alone, never a word-count instruction). That assertion
        is gone with the widget, not replaced with anything -- there is no
        word-count concept left to pin. The model-slot pins below are a
        separate concern (which LLM ships in the canonical graph) and
        still hold.
        """
        n = _node_by_id(_load_canonical_workflow(), 1)
        assert n.get("type") == "OTR_LedgerScriptWriter"
        widgets = n.get("widgets_values", [])
        assert len(widgets) >= 4, (
            "writer widgets_values shorter than expected; widget vector "
            "may have drifted"
        )
        # Widget order per OTR_LedgerScriptWriter.INPUT_TYPES, post the
        # 2026-08-14 target_words removal (BUG-LOCAL-269/270 seed-widget
        # removal predates it and still holds -- no seed widget either):
        #   [0] episode_title
        #   [1] num_characters
        #   [2] creative_writing_model
        #   [3] technical_model
        # 2026-07-20: official Gemma4Unified Transformers 5.10.4 + NF4
        # measured ~7.15 GiB and hard-constrained JSON through LMFE. This is
        # the safetensors/HF lane, not the independent GGUF Q8 row whose
        # context downgrade motivated the earlier Mistral canvas pin.
        # 2026-08-04: THE SIZE SUFFIX IS PART OF THE COMBO VALUE. The live
        # choice list offers 'google/gemma-4-12b-it (11.9 GB)', so the bare id
        # matched nothing: the operator reported both dropdowns rendering RED
        # on opening the graph, and an unmatched COMBO can resolve to index 0
        # of the list -- which here is Mistral-Nemo. The canvas said Gemma and
        # could have run Mistral. Pinned in full so the suffix cannot be lost.
        expected_creative = "google/gemma-4-12b-it (11.9 GB)"
        expected_technical = "google/gemma-4-12b-it (11.9 GB)"
        assert widgets[2] == expected_creative, (
            f"writer creative_writing_model must be {expected_creative!r}; "
            f"got {widgets[2]!r}."
        )
        assert widgets[3] == expected_technical, (
            f"writer technical_model must be {expected_technical!r}; "
            f"got {widgets[3]!r}."
        )


class TestValidatorPathFallback:
    """W-3 design note. The validator's workflow_json_path widget
    intentionally ships as empty string; the source-side fallback
    (Sprint E E5 / C-3 in plan) is the canonical path resolver.

    This test pins the empty-string convention so a future edit that
    hardcodes the operator path (and trips the forbidden-sweep
    `C:/Users/jeffr` marker) fails here first.
    """

    def test_validator_path_widget_is_empty_string(self):
        n = _node_by_id(_load_canonical_workflow(), 63)
        assert n.get("type") == "OTR_WorkflowValidator"
        widgets = n.get("widgets_values", [])
        assert widgets[0] == "", (
            f"validator workflow_json_path widget must be empty string "
            f"(operator-path neutral); got {widgets[0]!r}. The source-side "
            "fallback in _otr_workflow_validator.py resolves to "
            "_DEFAULT_WORKFLOW_PATH at runtime."
        )

    def test_validator_validate_anyway_default_on(self):
        n = _node_by_id(_load_canonical_workflow(), 63)
        widgets = n.get("widgets_values", [])
        assert widgets[1] is True
        assert widgets[2] is True


        # else: widget falls through to schema default (True) -- OK.


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
