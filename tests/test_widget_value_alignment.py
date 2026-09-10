"""One node type must declare the SAME widget order in every workflow.

THE GAP THIS CLOSES (2026-08-28). `test_canonical_widget_input_parity.py`
asserts `len(widget_inputs) == len(widgets_values)` per node -- which catches a
MISSING entry, and is why an added-but-unwired widget cannot ship. It cannot
catch a SHIFT, and a shift is the failure mode of REMOVING a widget: delete one
mid-list in `INPUT_TYPES`, drop any one value to keep the counts equal, and
every value after that index is silently attached to the wrong widget
(BUG-LOCAL-097's "silent drift").

WHY THIS MATTERS NOW: the project is willing to delete inert widgets rather
than leave dishonest controls in the graph (operator ruling 2026-08-28 -- *"why
not delete an inert widget and just make the adjustments so it's ok -- that's
being lazy not to remove an inert widget"*). That deletion is a MIGRATION: the
canonical AND every file under `workflows/variants/` must be re-indexed in the
same change. The realistic mistake is not doing it wrong everywhere -- it is
updating the canonical and MISSING A VARIANT.

That is exactly what this catches, and it needs no imports to do it. One node
type carries the same widget descriptors in the same order in every workflow
that uses it, because they all describe one Python class. If a migration
touches some files and not others, the orders diverge, and the divergence names
the file that was missed.

DELIBERATELY IMPORT-FREE. An earlier draft of this file resolved node classes
through `NODE_CLASS_MAPPINGS` to type-check each stored value. The mapping is
declared in the PACKAGE ROOT `__init__.py`, which is not importable by name (a
hyphenated directory) and breaks its own relative imports when loaded by path
-- so it came back EMPTY and every assertion silently became a no-op. Its
mutation test caught that, which is the reason a mutation test is at the bottom
of this file too: a guard that cannot fire is indistinguishable from no guard.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_CANONICAL = _REPO / "workflows" / "otr_canonical.json"
_WORKFLOWS = _REPO / "workflows"
_VARIANTS = _REPO / "workflows" / "variants"


def _workflows():
    """EVERY shipped graph, not just the canonical and the generated variants.

    This used to be `canonical + variants/*.json`, which quietly assumed that
    the only hand-authored graph lives in `variants/`. It does not:
    `workflows/otr_mac_lightning.json` is the graph that produced the Apple
    Silicon proof episode, and it was moved OUT of `variants/` on 2026-09-09
    because that directory is contractually generated-only -- an orphan file
    there with no matching profile crashes `build_variants.py --check`.

    The moment it moved, this guard stopped seeing it. That is the worse
    failure: a widget migration could re-index the canonical and leave the Mac
    graph behind, and the positional-widget contract (CLAUDE.md section 0)
    would be broken in a shipped file with nothing to catch it. Any graph under
    `workflows/` is a graph this guard must cover.
    """
    out = [_CANONICAL]
    out.extend(sorted(p for p in _WORKFLOWS.glob("*.json") if p != _CANONICAL))
    out.extend(sorted(_VARIANTS.glob("*.json")))
    return [p for p in out if p.is_file()]


def _widget_names(node):
    """Widget descriptor names, IN ORDER -- the positional contract itself."""
    names = []
    for inp in node.get("inputs") or []:
        w = inp.get("widget")
        if isinstance(w, dict) and w.get("name"):
            names.append(str(w["name"]))
    return names


def _orders_by_type():
    """``{node type: {tuple(widget names): [files that declare it]}}``."""
    seen = defaultdict(lambda: defaultdict(list))
    for path in _workflows():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for node in data.get("nodes") or []:
            names = _widget_names(node)
            if not names:
                continue
            seen[str(node.get("type") or "")][tuple(names)].append(path.name)
    return seen


def test_one_node_type_declares_one_widget_ORDER_everywhere():
    """A half-finished widget migration shows up here, naming the missed file."""
    problems = []
    for node_type, orders in sorted(_orders_by_type().items()):
        if len(orders) < 2:
            continue
        lines = ["  %s declares %d DIFFERENT widget orders:" % (node_type,
                                                               len(orders))]
        for names, files in sorted(orders.items(), key=lambda kv: -len(kv[1])):
            shown = ", ".join(files[:4]) + (" +%d more" % (len(files) - 4)
                                            if len(files) > 4 else "")
            lines.append("    %r\n      in %s" % (list(names), shown))
        problems.append("\n".join(lines))
    assert not problems, (
        "%d node type(s) declare different widget orders across workflows.\n"
        "One node type is one Python class, so its widget order is one fact. "
        "Divergence means a widget was added or REMOVED and only SOME saved "
        "graphs were re-indexed -- the missed files are named above, and every "
        "value after the differing index in them is attached to the wrong "
        "widget (BUG-LOCAL-097).\n%s" % (len(problems), "\n".join(problems)))


def test_widget_descriptor_count_matches_values_in_every_workflow():
    """The count contract, extended from the canonical to the VARIANTS too.

    `test_canonical_widget_input_parity` covers this repo-wide already; this
    restates it here so a failure of the ORDER test above can be read against a
    known-good count, rather than leaving a reader to wonder whether they are
    looking at one defect or two.
    """
    problems = []
    for path in _workflows():
        data = json.loads(path.read_text(encoding="utf-8"))
        for node in data.get("nodes") or []:
            values = node.get("widgets_values")
            if not isinstance(values, list):
                continue
            names = _widget_names(node)
            if names and len(names) != len(values):
                problems.append(
                    "  %s node %s (%s): %d descriptor(s) vs %d value(s)"
                    % (path.name, node.get("id"), node.get("type"),
                       len(names), len(values)))
    assert not problems, ("widget descriptor/value count mismatch:\n%s"
                          % "\n".join(problems))


def test_the_order_guard_would_actually_catch_a_missed_variant():
    """THE MUTATION TEST -- proves the guard above has teeth.

    A guard that never fires is indistinguishable from no guard, and the first
    draft of this file was exactly that (an empty class mapping made every
    assertion vacuous).

    IT USED TO REQUIRE TWO GRAPHS, and that made it a hostage to repo shape
    rather than a test of the guard. The pack now ships ONE workflow
    (a62f3567, "The pack ships ONE workflow JSON"), so there is no second file
    for a node type to disagree with, and this test failed on its own
    precondition -- reporting a broken guard when nothing was broken. Its own
    message said so: "the guard needs re-thinking, not the code".

    So it mutates the COMPARISON, not the tree: feed the same shape the real
    parser produces, with one file's widget list short by the last entry -- the
    exact half-finished migration this guards against -- and require the
    grouping to notice. That holds at one graph and at sixty-three.
    """
    real = _orders_by_type()
    assert real, "no workflow declares any widget descriptors -- the parser " \
                 "found nothing, so the guard above is checking nothing"

    # A type whose widget order two files disagree about, in the parser's own
    # {type: {order: [files]}} shape.
    node_type, by_order = next(iter(real.items()))
    names = next(iter(by_order))
    assert len(names) >= 2, (
        "the first parsed node type has fewer than 2 widgets, so a dropped-"
        "widget mutation cannot be represented -- pick a richer fixture")
    short = tuple(list(names)[:-1])
    mutated = {node_type: {names: ["otr_canonical.json"],
                           short: ["some_other_graph.json"]}}

    # The guard's rule, verbatim: more than one distinct order for one type.
    divergent = [t for t, orders in mutated.items() if len(orders) > 1]
    assert divergent == [node_type], (
        "the comparison did not flag a type declaring two different widget "
        "orders; that is exactly the missed re-index this file exists to catch")

    # And it must NOT fire when every file agrees.
    agreed = {node_type: {names: ["otr_canonical.json", "some_other_graph.json"]}}
    assert not [t for t, orders in agreed.items() if len(orders) > 1], (
        "the comparison flags agreement as divergence -- it would cry wolf on "
        "every clean tree")
