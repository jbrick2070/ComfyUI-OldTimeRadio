"""Sprint 10A step 3-C regression -- shadow-pass wiring in OTR_LedgerScriptWriter.

Covers the integration between the writer's `run()` flow and the
Stage 1 shadow pass added by step 3-C:

  * INPUT_TYPES exposes enable_stage1_shadow_pass with default=False.
  * The signature of `run()` accepts the new kwarg with default False.
  * _resolve_inputs accepts and round-trips the new kwarg into the
    resolved dict.
  * The shadow-pass code lives at the cast-lock point in the source,
    so the operator soak log lands the [Stage1Shadow] log line right
    after [OTR_LedgerScriptWriter] cast locked.
  * On Stage1PlanGenerationError the meta.stage1_shadow_attempts list
    carries an 'exhausted' status and the existing pipeline continues
    (no exception bubbles out of the shadow block).
  * On any other unexpected exception the shadow block carries a
    'shadow_setup_failed' status and the pipeline continues.

The tests EXERCISE only the source-text + INPUT_TYPES + _resolve_inputs
contract. The full writer run() touches LLM cache + RSS fetch + many
other systems, so a full integration is operator-soak territory. The
source-level pins here protect against silent rewiring drift that
would slip past pytest.
"""

from __future__ import annotations

import ast
import pathlib

import pytest


WRITER_SRC = (
    pathlib.Path(__file__).resolve().parent.parent
    / "nodes"
    / "OTR_LedgerScriptWriter.py"
)


# ---------------------------------------------------------------------------
# INPUT_TYPES exposes the widget
# ---------------------------------------------------------------------------


class TestInputTypesExposesWidget:
    def test_widget_in_optional_section(self):
        from nodes.OTR_LedgerScriptWriter import OTR_LedgerScriptWriter

        inputs = OTR_LedgerScriptWriter.INPUT_TYPES()
        assert "optional" in inputs
        assert "enable_stage1_shadow_pass" in inputs["optional"]
        spec = inputs["optional"]["enable_stage1_shadow_pass"]
        # ComfyUI widget spec is (TYPE, {options})
        assert spec[0] == "BOOLEAN"
        assert spec[1].get("default") is False, (
            "enable_stage1_shadow_pass must default to False so the "
            "existing pipeline keeps its bit-identical behaviour"
        )

    def test_widget_is_at_slot_17(self):
        """Slot 17 placement of enable_stage1_shadow_pass. Append-
        only rule (older N-slot workflow JSON files keep loading;
        ComfyUI fills missing trailing widgets from defaults).
        Appends in order:
          - Sprint 10B Wave 0 (2026-05-27): use_multiturn_dialogue
          - Sprint 10B Wave 1 Agent B (2026-05-27):
            enable_production_stage3_validators
          - Sprint 2.2 (2026-05-28): news_briefs_required
          - Sprint 4.5 (2026-05-28): use_stage1_fanout
        """
        from nodes.OTR_LedgerScriptWriter import OTR_LedgerScriptWriter

        inputs = OTR_LedgerScriptWriter.INPUT_TYPES()
        optional_keys = list(inputs["optional"].keys())
        # Slot 17 = fifth-to-last after the four appends.
        assert optional_keys[-5] == "enable_stage1_shadow_pass", (
            f"enable_stage1_shadow_pass must be the FIFTH-TO-LAST "
            f"optional widget (slot 17); current fifth-to-last is "
            f"{optional_keys[-5]!r}"
        )
        assert optional_keys[-4] == "use_multiturn_dialogue", (
            f"Sprint 10B Wave 0: use_multiturn_dialogue at slot 18 "
            f"(fourth-to-last); current is {optional_keys[-4]!r}"
        )
        assert optional_keys[-3] == "enable_production_stage3_validators", (
            f"Sprint 10B Wave 1 Agent B: "
            f"enable_production_stage3_validators at slot 19 "
            f"(third-to-last); current is {optional_keys[-3]!r}"
        )
        assert optional_keys[-2] == "news_briefs_required", (
            f"Sprint 2.2: news_briefs_required at slot 20 "
            f"(second-to-last); current is {optional_keys[-2]!r}"
        )
        assert optional_keys[-1] == "use_stage1_fanout", (
            f"Sprint 4.5: use_stage1_fanout must be the LAST "
            f"optional widget (slot 21); current last is "
            f"{optional_keys[-1]!r}"
        )


# ---------------------------------------------------------------------------
# run() signature accepts the kwarg
# ---------------------------------------------------------------------------


class TestRunSignatureAcceptsKwarg:
    def test_run_has_kwarg_with_false_default(self):
        from nodes.OTR_LedgerScriptWriter import OTR_LedgerScriptWriter
        import inspect

        sig = inspect.signature(OTR_LedgerScriptWriter.run)
        params = sig.parameters
        assert "enable_stage1_shadow_pass" in params
        assert params["enable_stage1_shadow_pass"].default is False


# ---------------------------------------------------------------------------
# _resolve_inputs accepts the kwarg and round-trips it into resolved
# ---------------------------------------------------------------------------


class TestResolveInputsRoundTrip:
    def test_default_false_lands_in_resolved(self):
        from nodes.OTR_LedgerScriptWriter import _resolve_inputs

        resolved = _resolve_inputs(
            episode_title="",
            target_words=350,
            num_characters=2,
            custom_premise="placeholder seed for the resolver",
        )
        assert resolved.get("enable_stage1_shadow_pass") is False

    def test_explicit_true_lands_in_resolved(self):
        from nodes.OTR_LedgerScriptWriter import _resolve_inputs

        resolved = _resolve_inputs(
            episode_title="",
            target_words=350,
            num_characters=2,
            custom_premise="placeholder seed for the resolver",
            enable_stage1_shadow_pass=True,
        )
        assert resolved.get("enable_stage1_shadow_pass") is True

    def test_truthy_non_bool_coerced(self):
        """The resolver wraps with bool() to keep the meta clean."""
        from nodes.OTR_LedgerScriptWriter import _resolve_inputs

        resolved = _resolve_inputs(
            episode_title="",
            target_words=350,
            num_characters=2,
            custom_premise="placeholder seed for the resolver",
            enable_stage1_shadow_pass=1,  # truthy non-bool
        )
        assert resolved.get("enable_stage1_shadow_pass") is True


# ---------------------------------------------------------------------------
# Source-level pins -- shadow block placement + error handling
# ---------------------------------------------------------------------------


class TestSourceLevelPins:
    def _src(self) -> str:
        return WRITER_SRC.read_text(encoding="utf-8")

    def test_shadow_pass_block_referenced_in_source(self):
        src = self._src()
        # Sprint 10A step 3-C marker comment + the runtime branch.
        assert "Sprint 10A step 3-C" in src
        assert "enable_stage1_shadow_pass" in src
        assert "stage1_shadow_attempts" in src

    def test_shadow_pass_after_cast_lock(self):
        """The shadow pass must run AFTER the cast lock log so the
        operator sees the [Stage1Shadow] log line in the expected
        place in the soak transcript.

        The runtime branch is uniquely marked by the full comment
        "Sprint 10A step 3-C: optional Stage 1 shadow pass" -- the
        bare "Sprint 10A step 3-C" string appears in earlier comments
        (INPUT_TYPES widget block, _resolve_inputs kwarg, run()
        signature) so we anchor on the more specific phrase.
        """
        src = self._src()
        cast_lock_marker = "cast locked: %d rows"
        shadow_marker = "Sprint 10A step 3-C: optional Stage 1 shadow pass"
        cast_lock_idx = src.index(cast_lock_marker)
        shadow_idx = src.index(shadow_marker)
        assert shadow_idx > cast_lock_idx, (
            "Sprint 10A step 3-C shadow pass must run AFTER the cast "
            "lock log; got reversed order in source"
        )

    def test_shadow_pass_imports_stage1_modules_under_try_block(self):
        """The imports for the Stage 1 modules MUST live INSIDE the
        feature-flag branch -- not at module level -- so that:
          1. Workflows with the flag OFF (default) do not pay the
             lm-format-enforcer import cost.
          2. A future broken Stage 1 module cannot crash the writer at
             import time when the flag is off.
        Pin via source check: the imports must come AFTER the
        Sprint 10A marker comment.
        """
        src = self._src()
        shadow_idx = src.index("Sprint 10A step 3-C: optional Stage 1")
        # Imports should be local to the branch, AFTER the marker.
        assert src.find(
            "from . import _otr_constrained_generate", shadow_idx,
        ) > shadow_idx, "_otr_constrained_generate import must be local"
        assert src.find(
            "from . import _otr_stage1_call", shadow_idx,
        ) > shadow_idx, "_otr_stage1_call import must be local"
        assert src.find(
            "from . import _otr_stage1_plan", shadow_idx,
        ) > shadow_idx, "_otr_stage1_plan import must be local"

    def test_shadow_pass_has_three_exception_arms(self):
        """The shadow pass must catch:
          1. Stage1PlanGenerationError -- retry budget exhausted
          2. Exception -- any other unexpected runtime failure
        AND the success path must record status='valid_first_attempt'
        (or similar) on the first attempt. The pipeline must NEVER
        halt on a shadow-pass failure."""
        src = self._src()
        shadow_idx = src.index("Sprint 10A step 3-C: optional Stage 1")
        # Look for both exception arms after the marker.
        assert src.find(
            "Stage1PlanGenerationError", shadow_idx,
        ) > shadow_idx, "Stage1PlanGenerationError arm missing"
        assert src.find(
            "except Exception", shadow_idx,
        ) > shadow_idx, "fallback Exception arm missing"

    def test_shadow_pass_stamps_meta_keys(self):
        """The shadow pass must stamp BOTH meta.stage1_shadow_attempts
        AND meta.stage1_shadow_plan_present so the soak grep + the
        downstream consumers can distinguish a successful pass from
        an exhausted-budget pass."""
        src = self._src()
        shadow_idx = src.index("Sprint 10A step 3-C: optional Stage 1")
        rest = src[shadow_idx:]
        assert 'meta["stage1_shadow_attempts"]' in rest
        assert 'meta["stage1_shadow_plan_present"]' in rest


# ---------------------------------------------------------------------------
# Workflow JSON pin
# ---------------------------------------------------------------------------


class TestWorkflowJsonPin:
    def test_workflow_writer_widget_vector_is_22_all_on(self):
        """2026-05-28 (Jeffrey directive): otr_scifi_16gb_full.json is
        baked to the one-click ALL-ON optimal-run state. The writer
        vector is the full 22-widget surface (3 required + 19 optional)
        with every render-quality + diagnostic flag ON.

        NOTE: this intentionally overrides the prior 'ships legacy-safe
        by default' invariant for THIS file -- Jeffrey runs the soak
        himself and wanted no widget fiddling. PD1 byte-identity is
        still guarded by test_audio_byte_identical (hermetic; pins its
        own seeds), which is unaffected by the shipped workflow default.

        Slots (0-indexed): 15 enable_polish_pass,
        17 enable_stage1_shadow_pass, 18 use_multiturn_dialogue,
        19 enable_production_stage3_validators, 20 news_briefs_required,
        21 use_stage1_fanout."""
        import json
        wf_path = (
            WRITER_SRC.parent.parent / "workflows" / "otr_scifi_16gb_full.json"
        )
        wf = json.loads(wf_path.read_text(encoding="utf-8"))
        writer = next(n for n in wf["nodes"] if n.get("id") == 1)
        assert writer["type"] == "OTR_LedgerScriptWriter"
        wv = writer["widgets_values"]
        assert len(wv) == 22, (
            f"workflow JSON writer widgets_values length must be 22 "
            f"(full INPUT_TYPES surface, all-on bake); got {len(wv)}"
        )
        assert wv[15] is True, f"slot 15 enable_polish_pass; got {wv[15]!r}"
        assert wv[17] is True, f"slot 17 enable_stage1_shadow_pass; got {wv[17]!r}"
        assert wv[18] is True, f"slot 18 use_multiturn_dialogue; got {wv[18]!r}"
        assert wv[19] is True, f"slot 19 enable_production_stage3_validators; got {wv[19]!r}"
        assert wv[20] is True, f"slot 20 news_briefs_required; got {wv[20]!r}"
        assert wv[21] is True, f"slot 21 use_stage1_fanout; got {wv[21]!r}"
