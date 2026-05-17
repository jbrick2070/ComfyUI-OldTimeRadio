"""Sprint E E8: OTR_ShotDurationCalculator -> OTR_FixedShotDurationStub rename.

Closes M10. The old name implied dynamic computation; the current
implementation operates on a hand-crafted JSON array of durations
(8s stub default) until the Bark audio-timeline wiring lands. The
rename makes the stub-nature surface-visible.

Per CLAUDE.md no-back-compat directive: NO alias entry for the old
name. Workflow JSONs are rewritten clean; pre-Sprint-E saved
workflow JSONs with the old type name will fail to load with a
"missing OTR_ShotDurationCalculator" surface, which is the
intentional greenfield behavior.
"""
from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE = REPO_ROOT / "nodes" / "otr_shot_duration_calculator.py"
INIT = REPO_ROOT / "__init__.py"
CANONICAL_JSON = REPO_ROOT / "workflows" / "otr_scifi_16gb_full.json"


def _src() -> str:
    return SOURCE.read_text(encoding="utf-8")


class TestSourceClassRenamed:
    def test_new_class_name_present(self):
        src = _src()
        assert "class OTRFixedShotDurationStub" in src

    def test_old_class_name_absent(self):
        src = _src()
        assert "class OTRShotDurationCalculator" not in src, (
            "Old class name OTRShotDurationCalculator must be deleted; "
            "CLAUDE.md no-back-compat rule"
        )

    def test_module_class_mappings_use_new_id(self):
        from nodes.otr_shot_duration_calculator import (
            NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS,
            OTRFixedShotDurationStub,
        )
        assert "OTR_FixedShotDurationStub" in NODE_CLASS_MAPPINGS
        assert NODE_CLASS_MAPPINGS["OTR_FixedShotDurationStub"] is OTRFixedShotDurationStub
        assert "OTR_ShotDurationCalculator" not in NODE_CLASS_MAPPINGS
        assert "OTR_FixedShotDurationStub" in NODE_DISPLAY_NAME_MAPPINGS
        assert "OTR_ShotDurationCalculator" not in NODE_DISPLAY_NAME_MAPPINGS

    def test_module_all_exports_new_name(self):
        from nodes import otr_shot_duration_calculator as mod
        assert "OTRFixedShotDurationStub" in mod.__all__
        assert "OTRShotDurationCalculator" not in mod.__all__


class TestRegistryRenamed:
    def test_init_registry_uses_new_id(self):
        src = INIT.read_text(encoding="utf-8")
        assert '"OTR_FixedShotDurationStub"' in src
        assert "OTRFixedShotDurationStub" in src

    def test_init_registry_drops_old_id(self):
        src = INIT.read_text(encoding="utf-8")
        # The old key must not appear as a NODE_CLASS_MAPPINGS key.
        # Forensic mentions in comments are allowed; the registry
        # entry itself must use the new name.
        # Check the registry tuple line is gone.
        assert '"OTR_ShotDurationCalculator":  (' not in src


class TestWorkflowJsonRenamed:
    def test_workflow_json_node_type_renamed(self):
        wf = json.loads(CANONICAL_JSON.read_text(encoding="utf-8"))
        n21 = [n for n in wf["nodes"] if n["id"] == 21][0]
        assert n21["type"] == "OTR_FixedShotDurationStub"
        # Search-and-replace cousin
        assert n21["properties"]["Node name for S&R"] == "OTR_FixedShotDurationStub"

    def test_workflow_json_drops_old_type(self):
        raw = CANONICAL_JSON.read_text(encoding="utf-8")
        assert '"OTR_ShotDurationCalculator"' not in raw, (
            "Old node type must be gone from canonical workflow JSON; "
            "CLAUDE.md no-back-compat rule"
        )

    def test_workflow_json_links_preserved(self):
        """The rename touches the node type + properties name. Links
        in/out of node 21 must be unchanged."""
        wf = json.loads(CANONICAL_JSON.read_text(encoding="utf-8"))
        links = {l[0]: l for l in wf["links"]}
        # L40 source = 20.2 -> target = 21.0
        assert 40 in links
        assert links[40][3] == 21  # target node still 21
        # L41 source = 21.0 -> target = 23.3
        assert 41 in links
        assert links[41][1] == 21  # source node still 21


class TestNodeStillFunctional:
    """Defensive: the rename must not break the node's INPUT_TYPES
    or its calculate() entrypoint."""

    def test_input_types_intact(self):
        from nodes.otr_shot_duration_calculator import OTRFixedShotDurationStub
        schema = OTRFixedShotDurationStub.INPUT_TYPES()
        assert "pass3_compose_prompts_json" in schema["required"]
        assert "shot_durations_json" in schema["required"]

    def test_calculate_method_exists(self):
        from nodes.otr_shot_duration_calculator import OTRFixedShotDurationStub
        node = OTRFixedShotDurationStub()
        assert callable(getattr(node, "calculate", None))
