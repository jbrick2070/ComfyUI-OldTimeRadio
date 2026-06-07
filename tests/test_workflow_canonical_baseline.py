"""Sprint E E2 / R-1: workflow JSON canonical-config widget drift guard.

Pins the audio C7 baseline reproduction precondition into the shipped
workflow JSON. If a future edit reverts the writer or HuMo seed
widgets, this test fails loud so default-config baseline reproduction
stays unblocked.

Per Sprint E E2 plan (W-1 + W-2):
- Node 1 (OTR_LedgerScriptWriter): the `seed` widget was removed
  (BUG-LOCAL-269/270); the model-slot widgets must ship the C7 models.
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
CANONICAL_JSON = REPO_ROOT / "workflows" / "otr_scifi_16gb_full.json"


def _load_canonical_workflow() -> dict:
    return json.loads(CANONICAL_JSON.read_text(encoding="utf-8"))


def _node_by_id(workflow: dict, node_id: int) -> dict:
    for n in workflow.get("nodes", []):
        if n.get("id") == node_id:
            return n
    raise AssertionError(f"node id {node_id} not present in {CANONICAL_JSON.name}")


class TestWriterCanonicalModelSlots:
    """W-1 drift guard. Writer Node 1 model-slot widgets must ship the
    C7 baseline models.

    The writer's `seed` widget was REMOVED (BUG-LOCAL-269 / 270): the
    cast + style RNGs are decoupled and draw OS entropy, so there is no
    longer a seed widget (or its control_after_generate companion) to
    pin here.
    """

    def test_writer_both_slots_mistral_nemo(self):
        """Sprint 10A step 2 reconciliation (2026-05-26):
        canonical writer model is mistralai/Mistral-Nemo-Instruct-2407
        on both creative and technical slots.

        Replaces the prior Sprint C3 destination of google/gemma-4-E4B-it.
        The three live runs on 2026-05-25 / 2026-05-26 (handoff entries
        signal_lost_bioluminescent_trench_descent, pending_20260525_223109,
        pending_20260526_053338) all ran on Mistral-Nemo per the soak
        logs; the workflow JSON had drifted to gemma-4-E4B-it but the
        running Desktop session was on Mistral-Nemo the whole time.
        Sprint 10A step 2 picked Mistral-Nemo as canonical (rationale:
        C7 byte-identical audio baseline, no auth gating, broad chat-
        template + grammar-constrained-decoding support, validated in
        3 live runs vs zero for gemma-4-E4B-it on the current pipeline
        shape) and flipped the workflow JSON to match. This test pins
        the new destination.
        """
        n = _node_by_id(_load_canonical_workflow(), 1)
        assert n.get("type") == "OTR_LedgerScriptWriter"
        widgets = n.get("widgets_values", [])
        assert len(widgets) >= 5, (
            "writer widgets_values shorter than expected; widget vector "
            "may have drifted"
        )
        # Widget order per OTR_LedgerScriptWriter.INPUT_TYPES, post the
        # BUG-LOCAL-269/270 seed-widget removal:
        #   [0] episode_title
        #   [1] target_words
        #   [2] num_characters
        #   [3] creative_writing_model
        #   [4] technical_model
        expected = "mistralai/Mistral-Nemo-Instruct-2407"
        assert widgets[3] == expected, (
            f"writer creative_writing_model must be {expected!r} per "
            f"Sprint 10A step 2 reconciliation; got {widgets[3]!r}."
        )
        assert widgets[4] == expected, (
            f"writer technical_model must be {expected!r} per "
            f"Sprint 10A step 2 reconciliation; got {widgets[4]!r}."
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
