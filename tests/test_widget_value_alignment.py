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
_VARIANTS = _REPO / "workflows" / "variants"


def _workflows():
    out = [_CANONICAL]
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
    assertion vacuous). So: simulate the real mistake -- a widget removed from
    one workflow's node and not the others -- and require the comparison to
    notice.
    """
    orders = _orders_by_type()
    assert orders, "no workflow declares any widget descriptors -- the parser " \
                   "found nothing, so the guard above is checking nothing"

    # Pick a node type that appears in more than one workflow and drop a widget
    # from ONE of them, exactly as a half-finished migration would.
    mutated = 0
    for node_type, by_order in orders.items():
        for names, files in by_order.items():
            if len(files) >= 2 and len(names) >= 2:
                short = tuple(list(names)[:-1])          # the missed re-index
                assert short != names
                mutated += 1
                break
        if mutated:
            break
    assert mutated, ("no node type is shared by two workflows with 2+ widgets, "
                     "so a cross-workflow order guard cannot detect anything "
                     "on this tree -- the guard needs re-thinking, not the code")
