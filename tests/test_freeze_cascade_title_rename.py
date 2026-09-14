"""Sprint D D-final -- the FreezeCascade double-dot title typo stays dead.

The default workflow's FreezeCascade node title was historically
"1b. Ledger Freeze Cascade (Phase 0..10)" with a literal double-
dot in the parenthetical. The "0..10" form parses ambiguously --
it reads as either "phases 0 through 10" (range notation) or as
a typo for "0.10" (version label) -- and drifts grep + screenshot
review. D-final renamed it to the unambiguous "Phase 0.10" form,
and this file pinned BOTH halves: the old form forbidden, the new
one required.

THE REQUIRED HALF IS GONE AS OF 2026-09-13, and the ambiguity it
was protecting against is gone with it. The canvas relayout
6234e44c retitled all 23 nodes into a numbered "N - name" form
along the path a reader follows, dropping every parenthetical:
node 62 is now "2 - Ledger Freeze". There is no phase label left
to be ambiguous about, so demanding one would pin a naming scheme
the graph deliberately left behind.

What survives is the half that was always the real subject -- the
double-dot must never come back. That guard costs nothing and
still fires if a future edit reintroduces "Phase 0..10" in any
title.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

WORKFLOW_PATH = REPO_ROOT / "workflows" / "otr_canonical.json"


def test_freeze_cascade_node_title_never_carries_the_double_dot() -> None:
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
            f"old double-dot form 'Phase 0..10', which reads as both a "
            f"range and a typo. Title: {title!r}"
        )
        assert ".." not in title, (
            f"FreezeCascade node id={node['id']} title carries a double "
            f"dot, the shape D-final removed. Title: {title!r}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
