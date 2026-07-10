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
        assert _DEFAULT_WORKFLOW_PATH.name == "otr_canonical.json"

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
        # Adversarial smoke for the DELETED_NODE_TYPES sentinel list.
        # OTR_LedgerScriptReviewer (retired S15.5-S19 cleanbreak) is
        # one of the pinned entries; the validator must raise
        # WorkflowDeletedNodeError when a workflow JSON still names
        # it. _workflow_validation.DELETED_NODE_TYPES holds the full
        # retirement registry.
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


# ---------------------------------------------------------------------------
# GATE B S2: stamp widgets + startup assertion + runtime env export
# ---------------------------------------------------------------------------

def _host(has_cuda=True, vram_mb=16384, platform="win", has_mps=False,
          vendor="nvidia"):
    # S5 platform-portability (2026-07-10): OLD shape had no "has_mps" /
    # "vendor" keys. _assert_stamp now also asserts device_backend=mps
    # needs host has_mps, and a profile's gpu_vendor pin must match the
    # detected host vendor (docs/2026-07-09-platform-portability-final.md
    # section 1). The real _detect_host() always returns both keys; a stub
    # missing them made host.get("vendor") resolve to None, which is
    # "not in ('none', 'nvidia')" -- a false-positive vendor mismatch
    # against the 16gb_full fixture profile (gpu_vendor="nvidia"). NEW
    # shape adds both keys with defaults matching that fixture's cuda/
    # nvidia profile so the pre-existing has_cuda=False call sites are
    # unaffected (they short-circuit before the vendor check runs).
    return {"has_cuda": has_cuda, "vram_mb": vram_mb, "platform": platform,
            "has_mps": has_mps, "vendor": vendor}


class TestStampAssertion:
    def test_input_types_carries_exactly_three_optional_stamp_widgets(self):
        it = WorkflowValidator.INPUT_TYPES()
        assert list(it["optional"]) == ["profile_id", "master_hash",
                                        "generated_by"], (
            "the three stamp widgets are the ONLY optional fields allowed "
            "on node 63 (decision doc section 4)")

    def test_master_ships_unstamped_with_padded_vector(self):
        wf = json.loads(_DEFAULT_WORKFLOW_PATH.read_text(encoding="utf-8"))
        n63 = next(n for n in wf["nodes"]
                   if n["type"] == "OTR_WorkflowValidator")
        assert n63["widgets_values"] == ["", True, True, "", "", ""], (
            "master must pad the three stamp slots EMPTY (unstamped) in the "
            "same commit as the INPUT_TYPES change")

    def test_unstamped_run_exports_nothing(self, monkeypatch):
        monkeypatch.delenv("OTR_ACTIVE_PROFILE", raising=False)
        node = WorkflowValidator()
        (msg,) = node.validate(str(_DEFAULT_WORKFLOW_PATH), True, False)
        assert "stamp OK" not in msg
        assert "OTR_ACTIVE_PROFILE" not in __import__("os").environ

    def test_stamped_run_exports_env(self, monkeypatch):
        # Post-VRAM-rip: the validator exports ONLY the active-profile id +
        # snapshot hash -- the OOM budget is owned by the operator's tier JSON,
        # so there is no VRAM-budget env export any more.
        #
        # S5 platform-portability (2026-07-10): _assert_stamp now also runs
        # the semantic master_hash tripwire whenever master_hash is
        # non-empty (nodes._otr_workflow_apply.semantic_master_hash vs the
        # live file) -- pass "" explicitly (matches the default) so this
        # test exercises the unstamped-hash / stamped-profile env-export
        # path it is actually about, not the drift tripwire. _host()'s
        # nvidia/cuda default now satisfies the new gpu_vendor-match assert
        # too (see _host()'s docstring comment above).
        import os
        monkeypatch.setattr(WorkflowValidator, "_detect_host",
                            staticmethod(lambda: _host()))
        monkeypatch.delenv("OTR_ACTIVE_PROFILE", raising=False)
        monkeypatch.delenv("OTR_SNAPSHOT_HASH", raising=False)
        node = WorkflowValidator()
        (msg,) = node.validate(str(_DEFAULT_WORKFLOW_PATH), True, False,
                               profile_id="16gb_full", master_hash="")
        assert "stamp OK" in msg
        assert os.environ["OTR_ACTIVE_PROFILE"] == "16gb_full"
        assert len(os.environ["OTR_SNAPSHOT_HASH"]) == 64

    def test_no_cuda_aborts_suggesting_cpu_floor(self, monkeypatch):
        monkeypatch.setattr(WorkflowValidator, "_detect_host",
                            staticmethod(lambda: _host(has_cuda=False)))
        node = WorkflowValidator()
        with pytest.raises(ValueError, match="cpu_floor"):
            node.validate(str(_DEFAULT_WORKFLOW_PATH), True, False,
                          profile_id="16gb_full")

    def test_validate_anyway_false_never_skips_the_assertion(self, monkeypatch):
        monkeypatch.setattr(WorkflowValidator, "_detect_host",
                            staticmethod(lambda: _host(has_cuda=False)))
        node = WorkflowValidator()
        with pytest.raises(ValueError, match="validate_anyway never skips"):
            node.validate(str(_DEFAULT_WORKFLOW_PATH), False, False,
                          profile_id="16gb_full")

    def test_unknown_profile_id_aborts_naming_known(self, monkeypatch):
        monkeypatch.setattr(WorkflowValidator, "_detect_host",
                            staticmethod(lambda: _host()))
        node = WorkflowValidator()
        with pytest.raises(ValueError, match="16gb_full"):
            node.validate(str(_DEFAULT_WORKFLOW_PATH), True, False,
                          profile_id="no_such_tier")

    def test_is_changed_varies_on_stamp(self):
        a = WorkflowValidator.IS_CHANGED(str(_DEFAULT_WORKFLOW_PATH), True,
                                         True)
        b = WorkflowValidator.IS_CHANGED(str(_DEFAULT_WORKFLOW_PATH), True,
                                         True, profile_id="16gb_full")
        assert a != b

    def test_gate_wired_63_to_87_and_image_lane_downstream(self):
        """S2 ordering guarantee + the S0 topology verification: node 63's
        validation_report feeds OTR_VideoDirector.gate_in, and the image
        DISPATCH lane (node 91) is transitively downstream of the gate via
        ShotLock's policy input -- no dispatcher gate input needed."""
        wf = json.loads(_DEFAULT_WORKFLOW_PATH.read_text(encoding="utf-8"))
        links = {l[0]: l for l in wf["links"]}
        n87 = next(n for n in wf["nodes"] if n["type"] == "OTR_VideoDirector")
        gate = next(i for i in n87["inputs"] if i["name"] == "gate_in")
        lnk = links[gate["link"]]
        assert (lnk[1], lnk[2]) == (63, 0), "gate_in must come from node 63"
        # transitivity: 87 -> (video_policy_json) -> 90 -> (script_json) -> 91
        n90 = next(n for n in wf["nodes"] if n["type"] == "OTR_ShotLock")
        pol = next(i for i in n90["inputs"] if i["name"] == "video_policy_json")
        assert links[pol["link"]][1] == 87
        n91 = next(n for n in wf["nodes"]
                   if n["type"] == "OTR_ImageGenDispatcher")
        sj = next(i for i in n91["inputs"] if i["name"] == "script_json")
        assert links[sj["link"]][1] == 90
