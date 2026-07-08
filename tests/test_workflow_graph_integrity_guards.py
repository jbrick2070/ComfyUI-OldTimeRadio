"""Implementation-addendum bug-prevention guards: W1 (widget-vector drift,
full node coverage) + N2 (graph-integrity gaps the existing suite misses).

Verified 2026-06-02 against the shipped tests: widget_vector_drift is run as a
HARD GATE only inside OTR_WorkflowValidator.validate(), which no CI test
invokes (test_workflow_live_passes_validator calls validate_workflow_contract
directly). And while link-id uniqueness / endpoint existence / last_link_id are
covered (test_workflow_json_guardrails, test_workflow_contract_validation),
these three are NOT: node-id uniqueness, link source-slot bounds, and
output<->central-link fan-out reconciliation (the addendum N2 dropped-consumer
class). Pure-JSON, byte-neutral.
"""
from __future__ import annotations

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

from nodes._otr_workflow_validator import widget_vector_drift  # noqa: E402

CANONICAL = REPO / "workflows" / "otr_canonical.json"
WF = json.loads(CANONICAL.read_text(encoding="utf-8"))
NODES = WF.get("nodes") or []
LINKS = WF.get("links") or []


def _otr_ncm() -> dict:
    """Reuse the AST-walk + importlib mapping helper (heavy optional deps are
    silently skipped, same as the live-validator test)."""
    import test_workflow_contract_validation as twv

    return twv._otr_node_class_mappings()


# -- W1: widget-vector drift across the full resolvable OTR node set ----------


def test_widget_vector_drift_all_nodes_except_known_musicgen():
    drift = widget_vector_drift(WF, _otr_ncm())
    # node-14 OTR_MusicGenTheme's forceInput slot is the documented open
    # BUG-281; every OTHER node must carry zero widgets_values drift. Existing
    # tests pin only AudioGen length -- this extends the gate to all nodes.
    unexpected = [d for d in drift if "OTR_MusicGenTheme" not in d]
    assert unexpected == [], (
        "widget-vector drift on non-MusicGen node(s): " + "; ".join(unexpected)
    )


# -- N2: graph-integrity gaps ------------------------------------------------


def test_node_ids_unique():
    ids = [n.get("id") for n in NODES]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    assert not dupes, f"duplicate node ids in canonical workflow: {dupes}"


def test_link_source_slot_in_bounds():
    by_id = {n.get("id"): n for n in NODES}
    bad = []
    for lk in LINKS:
        if not isinstance(lk, (list, tuple)) or len(lk) < 5:
            continue
        link_id, src_node, src_slot = lk[0], lk[1], lk[2]
        node = by_id.get(src_node)
        if node is None:
            bad.append(f"link {link_id}: source node {src_node} missing")
            continue
        outs = node.get("outputs") or []
        if not (isinstance(src_slot, int) and 0 <= src_slot < len(outs)):
            bad.append(
                f"link {link_id}: src_slot {src_slot} out of range for node "
                f"{src_node} ({len(outs)} outputs)"
            )
    assert not bad, "link source-slot bounds: " + "; ".join(bad)


def test_output_links_reconcile_with_central_links():
    """Each output slot's `links` set must equal the set of central link ids
    whose source is (node, slot) -- catches a hand-patch that updates the
    central links[] but forgets the source outputs[].links (silently dropping
    a downstream consumer) or leaves a stale fan-out id behind."""
    central: dict = {}
    for lk in LINKS:
        if not isinstance(lk, (list, tuple)) or len(lk) < 3:
            continue
        central.setdefault((lk[1], lk[2]), set()).add(lk[0])
    mismatches = []
    for n in NODES:
        nid = n.get("id")
        for slot, out in enumerate(n.get("outputs") or []):
            declared = set(out.get("links") or [])
            expected = central.get((nid, slot), set())
            if declared != expected:
                mismatches.append(
                    f"node {nid} out[{slot}] {out.get('name')!r}: "
                    f"outputs.links={sorted(declared)} != central={sorted(expected)}"
                )
    assert not mismatches, (
        "output<->central link fan-out reconciliation:\n  " + "\n  ".join(mismatches)
    )


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
