"""Sprint E E2 / R-1: workflow JSON canonical-config widget drift guard.

Pins the audio C7 baseline reproduction precondition into the shipped
workflow JSON. If a future edit reverts the writer or HuMo seed
widgets, this test fails loud so default-config baseline reproduction
stays unblocked.

Per Sprint E E2 plan (W-1 + W-2):
- Node 1 (OTR_LedgerScriptWriter) widgets must ship seed=42, control="fixed".
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


class TestWriterCanonicalSeed:
    """W-1 drift guard. Writer Node 1 widgets must ship the C7 baseline."""

    def test_writer_seed_is_42(self):
        n = _node_by_id(_load_canonical_workflow(), 1)
        assert n.get("type") == "OTR_LedgerScriptWriter"
        widgets = n.get("widgets_values", [])
        assert len(widgets) >= 5, (
            "writer widgets_values shorter than expected; widget vector "
            "may have drifted"
        )
        # Widget order per OTR_LedgerScriptWriter.INPUT_TYPES:
        #   [0] episode_title
        #   [1] target_words
        #   [2] num_characters
        #   [3] seed
        #   [4] seed control (ComfyUI auto-inserts after every INT seed)
        assert widgets[3] == 42, (
            f"writer seed widget must be 42 for C7 baseline; got {widgets[3]!r}. "
            "Drift fails the audio byte-identity contract per "
            "tests/fixtures/baseline_v1.5.wav."
        )

    def test_writer_seed_control_is_fixed(self):
        n = _node_by_id(_load_canonical_workflow(), 1)
        widgets = n.get("widgets_values", [])
        assert widgets[4] == "fixed", (
            f"writer seed control must be 'fixed' for C7 baseline reproduction; "
            f"got {widgets[4]!r}. 'randomize' rerolls on every Queue Prompt "
            "and blocks runtime b3sum reproduction."
        )

    def test_writer_both_slots_gemma_4(self):
        """Sprint C3 destination + 2026-05-18 retest #11 flip:
        canonical writer model is google/gemma-4-E4B-it on both
        creative and technical slots. The Mistral-Nemo baseline
        was the pre-Sprint-C3 default; reconcile commit 0ce8d2b
        repointed the harness at _full.json but did NOT flip the
        writer widgets, so Mistral-Nemo was reactivated as a
        side effect of the workflow consolidation. This test
        guards the Sprint C3 destination.
        """
        n = _node_by_id(_load_canonical_workflow(), 1)
        widgets = n.get("widgets_values", [])
        # Widget order:
        #   [5] creative_writing_model
        #   [6] technical_model
        expected = "google/gemma-4-E4B-it"
        assert widgets[5] == expected, (
            f"writer creative_writing_model must be {expected!r} for C7 "
            f"baseline (Sprint C3 destination); got {widgets[5]!r}."
        )
        assert widgets[6] == expected, (
            f"writer technical_model must be {expected!r} for C7 "
            f"baseline (Sprint C3 destination); got {widgets[6]!r}."
        )


class TestHumoCanonicalSeed:
    """W-2 drift guard. HuMo Node 51 widgets must ship a fixed seed."""

    def test_humo_seed_is_7(self):
        n = _node_by_id(_load_canonical_workflow(), 51)
        assert n.get("type") == "OTR_BatchHumoRender"
        widgets = n.get("widgets_values", [])
        assert len(widgets) >= 6, (
            "HuMo widgets_values shorter than expected; widget vector "
            "may have drifted"
        )
        # Widget order per BatchHumoRender.INPUT_TYPES:
        #   [0] ledger_json
        #   [1] portraits_dir
        #   [2] clip_length
        #   [3] max_clips
        #   [4] seed
        #   [5] seed control
        assert widgets[4] == 7, (
            f"HuMo seed widget must be 7 (schema default); got {widgets[4]!r}"
        )

    def test_humo_seed_control_is_fixed(self):
        n = _node_by_id(_load_canonical_workflow(), 51)
        widgets = n.get("widgets_values", [])
        assert widgets[5] == "fixed", (
            f"HuMo seed control must be 'fixed' to preserve cross-run visual "
            f"stability during soak comparisons; got {widgets[5]!r}."
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


class TestSkipEnvStillsDefault:
    """W-4 drift guard. BatchFluxRender skip_env_stills is the
    BUG-LOCAL-078 follow-up optimization. Schema default is True;
    the JSON may omit the trailing optional widgets and let ComfyUI
    fall through to the default. If the widget IS present in the
    vector, it must be True."""

    def test_batchfluxrender_skip_env_stills_true_if_present(self):
        n = _node_by_id(_load_canonical_workflow(), 23)
        assert n.get("type") == "OTR_BatchFluxRender"
        widgets = n.get("widgets_values", [])
        # skip_env_stills is the LAST optional widget per
        # BatchFluxRender.INPUT_TYPES. If the widget vector reaches
        # that position, the value must be True.
        # Widget positions documented at visual/batch_flux_render.py
        # INPUT_TYPES (model/clip/vae are typed inputs not widgets;
        # script_json is a forceInput socket post-PRIORITY-3 quick-win
        # #1 -- 2026-05-23 -- so it is NOT a widget either and the
        # vector shifted left by one):
        #   required: [0..9] = batch_limit, seed, ctrl, steps, cfg,
        #             sampler_name, scheduler, width, height, guidance
        #   optional: [10..16] = fallback_prompt, style_suffix,
        #             freeze_seed, fast_batch, radio_bookend_prompt,
        #             radio_bookend_seed, skip_env_stills
        if len(widgets) >= 17:
            assert widgets[16] is True, (
                f"BatchFluxRender skip_env_stills must be True per "
                f"BUG-LOCAL-078 follow-up; got {widgets[16]!r}. "
                "False re-enables the dead env-still pass and burns "
                "2-3 minutes of FLUX time per episode."
            )
        # else: widget falls through to schema default (True) -- OK.


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
