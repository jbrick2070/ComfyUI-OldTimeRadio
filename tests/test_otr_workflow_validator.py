"""tests/test_otr_workflow_validator.py -- coverage for the
OTR_WorkflowValidator node (S26 Sprint 3 / T1.2).

ADR: docs/2026-05-13-S14_2-active-validation-ADR.md (Option B).

Two canonical scenarios:
  - canonical workflow loads and validates cleanly (the same file the
    S16.6 CI test asserts on).
  - adversarial broken workflow raises a typed exception (deliberately
    placing a deleted-class node type at known retirement to fire
    WorkflowDeletedNodeError, or attaching an unknown OTR_ type to fire
    WorkflowUnknownNodeTypeError).

The tests instantiate `WorkflowValidator` directly and call .validate(),
the same execute() ComfyUI calls. No ComfyUI boot required.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from nodes._otr_workflow_validator import WorkflowValidator, _DEFAULT_WORKFLOW_PATH
from nodes._workflow_validation import (
    WorkflowDeletedNodeError,
    WorkflowUnknownNodeTypeError,
    WorkflowValidationError,
)


# ---------------------------------------------------------------------------
# canonical workflow -- the production fixture must validate cleanly
# ---------------------------------------------------------------------------

class TestCanonicalWorkflow:
    def test_default_workflow_path_points_at_canonical(self):
        """Sanity: the node's default path resolves at the canonical
        workflow fixture this repo ships with."""
        assert _DEFAULT_WORKFLOW_PATH.exists()
        assert _DEFAULT_WORKFLOW_PATH.name == "otr_scifi_16gb_full.json"

    def test_validate_canonical_workflow_passes(self):
        """The canonical workflow JSON must pass the same contract
        check that test_workflow_live_passes_validator asserts."""
        node = WorkflowValidator()
        # strict_unknown_types=False matches the CI runner default
        # (lets the test bypass NODE_CLASS_MAPPINGS introspection in
        # environments where not every OTR class is fully importable).
        (msg,) = node.validate(
            workflow_json_path=str(_DEFAULT_WORKFLOW_PATH),
            validate_anyway=True,
            strict_unknown_types=False,
        )
        assert msg.startswith("OTR_WorkflowValidator: OK -- ")
        assert "nodes" in msg and "links" in msg

    def test_validate_anyway_false_short_circuits(self):
        """validate_anyway=False returns the skip message without
        running the contract check (diagnostic loads)."""
        node = WorkflowValidator()
        (msg,) = node.validate(
            workflow_json_path=str(_DEFAULT_WORKFLOW_PATH),
            validate_anyway=False,
            strict_unknown_types=True,
        )
        assert "validate_anyway=False" in msg
        assert "skipped" in msg

    def test_is_changed_picks_up_path_change(self):
        """IS_CHANGED returns a string that varies on path / flags /
        mtime. Distinct inputs produce distinct signatures."""
        sig_a = WorkflowValidator.IS_CHANGED(
            str(_DEFAULT_WORKFLOW_PATH), True, True,
        )
        sig_b = WorkflowValidator.IS_CHANGED(
            str(_DEFAULT_WORKFLOW_PATH), False, True,
        )
        sig_c = WorkflowValidator.IS_CHANGED(
            "", True, True,
        )
        assert sig_a != sig_b
        assert sig_a != sig_c


# ---------------------------------------------------------------------------
# adversarial workflows -- contract violations must raise typed errors
# ---------------------------------------------------------------------------

class TestAdversarialWorkflow:
    def test_missing_workflow_file_raises_file_not_found(self, tmp_path):
        """An invalid workflow path raises FileNotFoundError -- the
        node refuses to silently no-op on a misconfigured widget."""
        node = WorkflowValidator()
        with pytest.raises(FileNotFoundError, match="workflow JSON not found"):
            node.validate(
                workflow_json_path=str(tmp_path / "does_not_exist.json"),
                validate_anyway=True,
                strict_unknown_types=False,
            )

    def test_malformed_json_raises_value_error(self, tmp_path):
        """A workflow file that isn't valid JSON raises ValueError, not
        a silent json.JSONDecodeError bubbling through."""
        bad = tmp_path / "broken.json"
        bad.write_text("{ not valid json", encoding="utf-8")
        node = WorkflowValidator()
        with pytest.raises(ValueError, match="failed to parse"):
            node.validate(
                workflow_json_path=str(bad),
                validate_anyway=True,
                strict_unknown_types=False,
            )

    def test_unknown_otr_node_type_raises(self, tmp_path):
        """An OTR_-prefixed node that isn't in NODE_CLASS_MAPPINGS
        fires WorkflowUnknownNodeTypeError when strict_unknown_types
        is True. This is the deliberate adversarial path the ADR
        documents."""
        adversarial = {
            "last_node_id": 1,
            "last_link_id": 0,
            "nodes": [
                {
                    "id": 1,
                    "type": "OTR_DefinitelyNotARegisteredClass_XYZ",
                    "inputs": [],
                    "outputs": [],
                    "widgets_values": [],
                }
            ],
            "links": [],
            "groups": [],
            "version": 0.4,
        }
        wf = tmp_path / "adversarial.json"
        wf.write_text(json.dumps(adversarial), encoding="utf-8")
        node = WorkflowValidator()
        with pytest.raises(WorkflowUnknownNodeTypeError):
            node.validate(
                workflow_json_path=str(wf),
                validate_anyway=True,
                strict_unknown_types=True,
            )

    def test_deleted_node_type_raises(self, tmp_path):
        """A workflow that still references a known-deleted node class
        (e.g. OTR_LedgerScriptReviewer from S15.5-S19 cleanbreak) raises
        WorkflowDeletedNodeError. The DELETED_NODE_TYPES list in
        _workflow_validation pins each retirement."""
        from nodes._workflow_validation import DELETED_NODE_TYPES
        # Take the first registered deleted-type sentinel to drive the
        # adversarial case so the test stays in sync with the registry.
        if not DELETED_NODE_TYPES:
            pytest.skip(
                "DELETED_NODE_TYPES is empty -- nothing to assert against."
            )
        deleted_type = next(iter(DELETED_NODE_TYPES))
        adversarial = {
            "last_node_id": 1,
            "last_link_id": 0,
            "nodes": [
                {
                    "id": 1,
                    "type": deleted_type,
                    "inputs": [],
                    "outputs": [],
                    "widgets_values": [],
                }
            ],
            "links": [],
            "groups": [],
            "version": 0.4,
        }
        wf = tmp_path / "adversarial_deleted.json"
        wf.write_text(json.dumps(adversarial), encoding="utf-8")
        node = WorkflowValidator()
        with pytest.raises((WorkflowDeletedNodeError, WorkflowValidationError)):
            node.validate(
                workflow_json_path=str(wf),
                validate_anyway=True,
                strict_unknown_types=False,
            )
