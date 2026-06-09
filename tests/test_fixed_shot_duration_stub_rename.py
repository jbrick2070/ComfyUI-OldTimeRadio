"""Chunk E cleanbreak: OTR_ShotDurationCalculator + OTR_FixedShotDurationStub retired.

Sprint E E8 renamed OTR_ShotDurationCalculator -> OTR_FixedShotDurationStub.
Chunk E (2026-06-08) fully retires both: OTR_ShotLock owns all per-episode
budget / shot-duration logic, so neither node is needed on the wire.

Retirement contract:
  - NODE_CLASS_MAPPINGS is empty (node not visible in ComfyUI picker)
  - Both old and new type names are in DELETED_NODE_TYPES (stale workflow
    JSONs fail validation loudly)
  - The class + pure-Python helpers are kept for unit tests
  - No workflow JSON contains either type name
"""
from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE = REPO_ROOT / "nodes" / "otr_shot_duration_calculator.py"
CANONICAL_JSON = REPO_ROOT / "workflows" / "otr_scifi_16gb_full.json"
VALIDATION_MOD = REPO_ROOT / "nodes" / "_workflow_validation.py"


def _src() -> str:
    return SOURCE.read_text(encoding="utf-8")


class TestSourceClassRetained:
    """Class must survive for unit-test helpers; registration is gone."""

    def test_stub_class_still_present_for_tests(self):
        assert "class OTRFixedShotDurationStub" in _src()

    def test_old_class_name_absent(self):
        assert "class OTRShotDurationCalculator" not in _src()


class TestRegistrationCleared:
    """Both names must be absent from NODE_CLASS_MAPPINGS (Chunk E retire)."""

    def test_node_class_mappings_empty(self):
        from nodes.otr_shot_duration_calculator import (
            NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS,
        )
        assert "OTR_FixedShotDurationStub" not in NODE_CLASS_MAPPINGS, (
            "OTR_FixedShotDurationStub must be unregistered (Chunk E retire)"
        )
        assert "OTR_ShotDurationCalculator" not in NODE_CLASS_MAPPINGS
        assert "OTR_FixedShotDurationStub" not in NODE_DISPLAY_NAME_MAPPINGS
        assert "OTR_ShotDurationCalculator" not in NODE_DISPLAY_NAME_MAPPINGS


class TestTombstoned:
    """Both type names must appear in DELETED_NODE_TYPES."""

    def test_old_name_tombstoned(self):
        from nodes._workflow_validation import DELETED_NODE_TYPES
        assert "OTR_ShotDurationCalculator" in DELETED_NODE_TYPES

    def test_stub_name_tombstoned(self):
        from nodes._workflow_validation import DELETED_NODE_TYPES
        assert "OTR_FixedShotDurationStub" in DELETED_NODE_TYPES

    def test_videos_plan_tombstoned(self):
        from nodes._workflow_validation import DELETED_NODE_TYPES
        assert "OTR_VideoPlan" in DELETED_NODE_TYPES

    def test_render_plan_tombstoned(self):
        from nodes._workflow_validation import DELETED_NODE_TYPES
        assert "OTR_RenderPlan" in DELETED_NODE_TYPES


class TestWorkflowJsonClean:

    def test_workflow_json_has_no_old_type(self):
        raw = CANONICAL_JSON.read_text(encoding="utf-8")
        assert '"OTR_ShotDurationCalculator"' not in raw
        assert '"OTR_FixedShotDurationStub"' not in raw
        assert '"OTR_VideoPlan"' not in raw
        assert '"OTR_RenderPlan"' not in raw


class TestNodeStillFunctional:
    """Pure-Python helpers must survive the registration cleanup."""

    def test_input_types_intact(self):
        from nodes.otr_shot_duration_calculator import OTRFixedShotDurationStub
        schema = OTRFixedShotDurationStub.INPUT_TYPES()
        assert "pass3_compose_prompts_json" in schema["required"]
        assert "shot_durations_json" in schema["required"]

    def test_calculate_method_exists(self):
        from nodes.otr_shot_duration_calculator import OTRFixedShotDurationStub
        node = OTRFixedShotDurationStub()
        assert callable(getattr(node, "calculate", None))
