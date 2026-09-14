"""`perfect_run_spacesaver` is GONE, and must not come back (2026-09-13).

THE HISTORY, because the inversion matters. This file used to assert the
opposite: that the widget MUST REMAIN on `OTR_LedgerScriptWriter`. That was
correct when it was written (queue item 8, 2026-08-08) -- the widget had been a
no-op since its consumer `_spacesaver_cleanup_if_flagged` went with the retired
RTX-VSR node, and removing it would have shifted every later widget index in
every saved graph (BUG-LOCAL-097).

What changed is not the risk, it is who does the work. Operator ruling
2026-08-28: *"why not delete an inert widget and just make the adjustments so
it's ok -- that's being lazy not to remove an inert widget"*. The index shift
is WORK, not a veto, and the work is the three-part removal in CLAUDE.md
section 0 -- `widgets_values`, the `inputs` descriptor array, and every link
whose `dst_slot` indexes past it -- executed across all 17 shipped graphs by
`scripts/otr_widget_surgery.py`. Measured on the canonical: `gate_in` moved
from descriptor 32 to 31 and link 279 was repaired BY IDENTITY, never by
subtracting one.

THIS FILE WAS REPLACED RATHER THAN DELETED, deliberately. A deleted guard
leaves nothing to stop the widget being re-added by a future window reading an
old doc -- and the repo is full of old docs that describe it as load-bearing. A
guard that asserts its ABSENCE keeps the decision enforced instead of merely
recorded.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NODES_DIR = REPO / "nodes"
WORKFLOWS = REPO / "workflows"

WIDGET = "perfect_run_spacesaver"


def _writer_class():
    import nodes.OTR_LedgerScriptWriter as m

    for name in dir(m):
        obj = getattr(m, name)
        if not callable(getattr(obj, "INPUT_TYPES", None)):
            continue
        if "LedgerScriptWriter" in name:
            return obj
    raise AssertionError("could not resolve the writer class")


def test_the_widget_is_no_longer_declared():
    """The removal, asserted at the source of truth -- the live class."""
    spec = _writer_class().INPUT_TYPES()
    for section in ("required", "optional"):
        block = spec.get(section) or {}
        assert WIDGET not in block, (
            "%s is declared again in INPUT_TYPES[%r]. It was removed on "
            "2026-09-13 because it had done nothing since 2026-08-08. "
            "Re-adding it does not restore a feature -- its consumer was "
            "deleted with the RTX-VSR node -- it only shifts every widget "
            "after it in all 17 shipped graphs." % (WIDGET, section))


def test_no_shipped_graph_still_carries_the_descriptor_or_its_value():
    """All 17 graphs, because the canonical alone is not the shipping set.

    A migration that updates the canonical and misses a variant leaves the
    variant with a descriptor the class no longer declares -- which
    `test_widget_schema_order_matches_live_input_types` would also catch, but
    this names the widget, so the failure reads as itself.
    """
    graphs = sorted(WORKFLOWS.glob("*.json")) + sorted(
        (WORKFLOWS / "variants").glob("*.json"))
    checked = 0
    offenders = []
    for path in graphs:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(data.get("nodes"), list):
            continue
        checked += 1
        for node in data["nodes"]:
            for inp in node.get("inputs") or []:
                if (inp.get("widget") or {}).get("name") == WIDGET:
                    offenders.append("%s node %s" % (path.name, node.get("id")))
    assert checked >= 17, (
        "expected the 17 shipped graphs, read %d -- this guard is looking at "
        "the wrong place" % checked)
    assert not offenders, (
        "%s descriptor survives in: %r" % (WIDGET, offenders))


def test_the_writer_still_has_exactly_one_widget_per_saved_value():
    """The three-part removal, asserted at its consequence on the canonical.

    Dropping the descriptor without the value (or the other way round) leaves
    counts that disagree, and every widget after the removal reading its
    neighbour's value.
    """
    data = json.loads((WORKFLOWS / "otr_canonical.json").read_text(encoding="utf-8"))
    node = next(n for n in data["nodes"]
                if str(n.get("type")) == "OTR_LedgerScriptWriter")
    descriptors = [i for i in (node.get("inputs") or []) if i.get("widget")]
    assert len(descriptors) == len(node["widgets_values"]), (
        "the writer carries %d widget descriptor(s) and %d saved value(s)"
        % (len(descriptors), len(node["widgets_values"])))
    assert len(descriptors) == 36, (
        "the writer should carry 36 widgets after the removal (37 before); "
        "got %d" % len(descriptors))


def test_gate_in_kept_its_link_through_the_removal():
    """THE PART THAT IS INVISIBLE TO A WIDGET-COUNT CHECK.

    `dst_slot` indexes the same `inputs` array that holds the widget
    descriptors, so removing descriptor 8 moved `gate_in` from 32 to 31 and
    link 279 had to move with it. Repaired by identity -- matching
    `inputs[i].link` to the row's id -- which is self-correcting and
    impossible to double-apply.
    """
    data = json.loads((WORKFLOWS / "otr_canonical.json").read_text(encoding="utf-8"))
    node = next(n for n in data["nodes"]
                if str(n.get("type")) == "OTR_LedgerScriptWriter")
    slot = next(i for i, inp in enumerate(node["inputs"])
                if inp.get("name") == "gate_in")
    link_id = node["inputs"][slot].get("link")
    assert link_id is not None, "gate_in lost its link in the removal"
    row = next(r for r in data["links"] if r[0] == link_id)
    assert row[3] == node["id"] and row[4] == slot, (
        "link %s points at node %s slot %s but gate_in is node %s slot %s -- "
        "the link table was not repaired with the descriptor array"
        % (link_id, row[3], row[4], node["id"], slot))


def test_no_production_module_reads_the_flag_any_more():
    """The plumbing went with the widget, not just the declaration.

    An inert widget removed from the UI while its parameter, its `resolved`
    entry and its ledger stamp stay behind is a rename of the problem. The
    three modules that legitimately mentioned it -- the writer,
    `_otr_writer_inputs`, `_otr_writer_tail` -- are NOT excluded here any
    more, which is the whole difference between this guard and the one it
    replaced.
    """
    hits = []
    pattern = re.compile(r"\b%s\b" % WIDGET)
    for path in NODES_DIR.rglob("*.py"):
        src = path.read_text(encoding="utf-8", errors="replace")
        for line in src.splitlines():
            if not pattern.search(line):
                continue
            if line.lstrip().startswith("#"):
                continue  # the comment recording what was removed, and why
            hits.append("%s: %s" % (path.relative_to(REPO), line.strip()))
    assert not hits, (
        "%s is still read or written in production code:\n  %s\nThe widget is "
        "gone; a surviving consumer is a flag no user can set."
        % (WIDGET, "\n  ".join(hits)))


def test_the_profile_mapper_no_longer_exempts_a_widget_that_does_not_exist():
    """`exempt_widget_names` listing a removed widget is harmless but it is a
    lie the next reader has to disprove, and this repo has been bitten by
    exactly that (a stale count propagating for weeks)."""
    mapping = json.loads(
        (REPO / "config" / "profiles" / "widget_mapping.json").read_text(
            encoding="utf-8"))
    exempt = (mapping.get("exempt_widget_names") or {}).get(
        "OTR_LedgerScriptWriter") or []
    assert WIDGET not in exempt, (
        "widget_mapping.json still exempts %s from profile patching, but the "
        "widget no longer exists" % WIDGET)
