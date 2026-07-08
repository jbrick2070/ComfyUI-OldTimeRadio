"""Sprint D D-final -- FreezeCascade node title pin.

The default workflow's FreezeCascade node title was historically
"1b. Ledger Freeze Cascade (Phase 0..10)" with a literal double-
dot in the parenthetical. The "0..10" form parses ambiguously --
it reads as either "phases 0 through 10" (range notation) or as
a typo for "0.10" (version label) -- and drifts grep + screenshot
review. D-final renames to the unambiguous "Phase 0.10" form and
pins it here.

Single assertion: the FreezeCascade node's title contains
"Phase 0.10" and does NOT contain the double-dot "Phase 0..10"
form.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

WORKFLOW_PATH = REPO_ROOT / "workflows" / "otr_canonical.json"


def test_freeze_cascade_node_title_is_phase_0_10_not_double_dot() -> None:
    workflow = json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))
    freeze_nodes = [
        n for n in workflow["nodes"]
        if n.get("type") == "OTR_LedgerFreezeCascade"
    ]
    assert freeze_nodes, (
        "default workflow has no OTR_LedgerFreezeCascade node"
    )
    for node in freeze_nodes:
        title = node.get("title") or ""
        assert "Phase 0..10" not in title, (
            f"FreezeCascade node id={node['id']} title contains the "
            f"old double-dot form 'Phase 0..10'; D-final renamed to "
            f"'Phase 0.10'. Title: {title!r}"
        )
        assert "Phase 0.10" in title, (
            f"FreezeCascade node id={node['id']} title does NOT "
            f"contain 'Phase 0.10' label; D-final pin. Title: "
            f"{title!r}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
