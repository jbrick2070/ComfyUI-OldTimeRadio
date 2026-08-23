"""Chunk E cleanbreak: OTR_ShotDurationCalculator + OTR_FixedShotDurationStub retired.

Sprint E E8 renamed OTR_ShotDurationCalculator -> OTR_FixedShotDurationStub.
Chunk E (2026-06-08) unregistered both: OTR_ShotLock owns all per-episode budget
and shot-duration logic, so neither node is needed on the wire.

LEAN-MEAN 2026-08-23 -- the implementation is now GONE too.
`nodes/otr_shot_duration_calculator.py` (461 lines) is deleted, along with
`tests/test_otr_shot_duration_calculator.py`, which tested only that dead
implementation. What the retirement contract asserted for two months was that
the class was "kept for unit tests" -- i.e. the sole remaining consumer of the
module was the test suite proving the module still existed. That is a closed
loop, not coverage.

WHAT SURVIVES IS THE HALF THAT PROTECTS A USER, and it is why this file is
rewritten rather than deleted with the rest. Someone out there may hold a
workflow JSON naming one of these node types. The tombstones in
DELETED_NODE_TYPES are what turn that into a LOUD validation failure with a
migration message instead of a silent unknown-node. Deleting the code must never
delete the tombstone: the code is what nobody runs, the tombstone is what
everybody hits.

So the contract is now:
  - the module and its class are GONE (nothing imports them)
  - BOTH type names remain in DELETED_NODE_TYPES
  - the canonical workflow names neither
"""
from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE = REPO_ROOT / "nodes" / "otr_shot_duration_calculator.py"
CANONICAL_JSON = REPO_ROOT / "workflows" / "otr_canonical.json"


class TestImplementationRetired:
    """The dead implementation is gone -- and nothing reaches for it."""

    def test_module_file_is_deleted(self):
        assert not SOURCE.exists(), (
            "nodes/otr_shot_duration_calculator.py is back. It was retired on "
            "2026-08-23; OTR_ShotLock owns shot-duration logic. If a real "
            "consumer needs it again, that is a design decision, not a restore.")

    def test_nothing_imports_the_retired_module(self):
        """A revived import would fail at runtime, so catch it at suite time."""
        offenders = []
        for folder in ("nodes", "scripts", "tests"):
            base = REPO_ROOT / folder
            if not base.is_dir():
                continue
            for path in base.rglob("*.py"):
                if ".claude" in path.parts or "__pycache__" in path.parts:
                    continue
                if path.resolve() == Path(__file__).resolve():
                    continue          # this file names it only in prose
                text = path.read_text(encoding="utf-8", errors="replace")
                if "otr_shot_duration_calculator" in text:
                    offenders.append(path.relative_to(REPO_ROOT).as_posix())
        assert not offenders, (
            f"these still reference the retired module: {offenders}")


class TestTombstoned:
    """THE LOAD-BEARING HALF. A stale user workflow must fail LOUD, not quietly.

    These four assertions are the reason this file survived the deletion of
    everything it used to test.
    """

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

    def test_a_tombstoned_type_actually_fails_validation(self):
        """The tombstone is only worth keeping if it still REFUSES.

        Membership in a frozenset proves nothing on its own -- the validator has
        to act on it. This walks the real check.
        """
        from nodes._workflow_validation import DELETED_NODE_TYPES
        assert "OTR_FixedShotDurationStub" in DELETED_NODE_TYPES
        import nodes._workflow_validation as wv
        assert hasattr(wv, "WorkflowDeletedNodeError"), (
            "the deleted-node error class is gone; the tombstone list would "
            "then be inert data rather than a gate")


class TestWorkflowJsonClean:

    def test_workflow_json_has_no_old_type(self):
        raw = CANONICAL_JSON.read_text(encoding="utf-8")
        assert '"OTR_ShotDurationCalculator"' not in raw
        assert '"OTR_FixedShotDurationStub"' not in raw
        assert '"OTR_VideoPlan"' not in raw
        assert '"OTR_RenderPlan"' not in raw
