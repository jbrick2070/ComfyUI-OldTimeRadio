"""A widget migration must carry every VALUE with its own NAME, in all 17 graphs.

THE HOLE THIS CLOSES (2026-09-13 QA verdict, guard 2 of 2). The obvious way to
check a reorder or a removal is to compare `widgets_values` before and after.
On this repo that check is close to worthless, because the real saved values
are overwhelmingly `""`, `False` and `0`: swap two widgets that both hold `""`
and every value-based assertion still passes while the graph is corrupt. The
writer alone carries fourteen empty strings.

So this guard does not compare values to values. It stamps each widget with a
marker derived from its own NAME, performs the migration through the real tool,
and then requires each surviving marker to be sitting on the widget it names.
A value that moves to a neighbour is caught even when the two values were
identical to begin with.

IT RUNS ON ALL SEVENTEEN SHIPPED GRAPHS, not just the canonical.
`tests/test_widget_surgery_tool.py` proves the tool against the canonical --
that is a test OF THE TOOL. This is a test of the MIGRATION: the variants are
generated, they are the files an installed user actually loads, and the
historical failure here was never "the tool is wrong", it was "the canonical
was re-indexed and a variant was missed".

WHAT EACH TEST BELOW COVERS
  * the reorder path -- every node with two or more widgets, reversed, the
    worst case for pairing;
  * the removal path -- each widget of each node removed in turn, with the
    survivors required to keep their own markers;
  * the RESTORE ComfyUI will actually perform -- a positional zip of
    `widgets_values` onto the live `INPUT_TYPES` widget list, which is what
    frontend 1.51.10 does with named restore off;
  * a mutation test proving the pairing check fires on the exact mistake the
    three-part removal rule exists to prevent (dropping the value at an index
    computed by arithmetic rather than found by identity).
"""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

# `tests/` is a package (it has an __init__.py), so a sibling test module is
# imported by its package path -- the same route
# `test_original_model_credit` uses for `test_openrouter_model_gone`. Guard 1
# owns these helpers because its docstring is where the widget/socket
# distinction is explained; re-deriving them here would let the two guards
# drift apart and disagree about what a widget is.
from tests.test_widget_schema_order_matches_live_input_types import (
    GRAPHS,
    NODE_CLASS_MAPPINGS,
    live_widget_order,
    saved_widget_order,
)

_REPO = Path(__file__).resolve().parents[1]
_TOOL = _REPO / "scripts" / "otr_widget_surgery.py"

# A marker no real widget value could collide with, and one that names the
# widget it belongs to so a mispairing reads as itself in the failure message.
_MARK = "OTR-MIGRATION-MARKER::%s"


def _tool():
    spec = importlib.util.spec_from_file_location("otr_widget_surgery", _TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _graph(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _stamp(node) -> list:
    """Replace this node's saved values with one marker per widget NAME."""
    names = saved_widget_order(node)
    node["widgets_values"] = [_MARK % n for n in names]
    return names


def _pairs(node) -> list:
    """`(widget name, value)` as the positional restore would pair them."""
    return list(zip(saved_widget_order(node), node.get("widgets_values") or []))


def _migratable_nodes(data) -> list:
    """Nodes this guard can meaningfully migrate.

    A node whose saved values are already out of step with its descriptors is
    a different defect, owned by `test_canonical_widget_input_parity`; the
    surgery tool refuses such a node by design, so skip rather than double-
    report it here.
    """
    out = []
    for node in data.get("nodes") or []:
        names = saved_widget_order(node)
        values = node.get("widgets_values")
        if len(names) >= 2 and isinstance(values, list) and len(values) == len(names):
            out.append(node)
    return out


@pytest.mark.parametrize("path", GRAPHS, ids=lambda p: p.name)
def test_a_reorder_carries_every_value_with_its_own_widget(path):
    """Reverse each node's widget order and require every marker to follow."""
    ws = _tool()
    data = _graph(path)
    nodes = _migratable_nodes(data)
    assert nodes, "%s has no node with two or more widgets to migrate" % path.name

    checked = 0
    for original in nodes:
        wf = copy.deepcopy(data)
        node = next(n for n in wf["nodes"] if n.get("id") == original.get("id"))
        names = _stamp(node)
        node_type = str(node.get("type") or "")

        # A type can appear more than once in a graph; the tool works by type,
        # so stamp every sibling too or the untouched ones fail the pairing.
        for sibling in wf["nodes"]:
            if sibling is not node and str(sibling.get("type") or "") == node_type:
                if len(saved_widget_order(sibling)) == len(
                        sibling.get("widgets_values") or []):
                    _stamp(sibling)
                else:
                    pytest.skip("%s: a sibling %s is already count-mismatched"
                                % (path.name, node_type))

        ws.reorder_widgets(wf, node_type, list(reversed(names)))

        for after in wf["nodes"]:
            if str(after.get("type") or "") != node_type:
                continue
            for name, value in _pairs(after):
                assert value == _MARK % name, (
                    "%s node %s (%s): after reversing the widget order, widget "
                    "%r holds %r -- the value belongs to widget %r.\nEvery "
                    "saved value is restored by POSITION, so a value that "
                    "drifts to a neighbour is applied to the wrong control at "
                    "load with nothing to complain.\npairs: %r"
                    % (path.name, after.get("id"), node_type, name, value,
                       str(value).split("::")[-1], _pairs(after)))
        assert ws.verify(wf, path.name) == [], ws.verify(wf, path.name)
        checked += 1

    assert checked == len(nodes)


@pytest.mark.parametrize("path", GRAPHS, ids=lambda p: p.name)
def test_a_removal_leaves_every_survivor_holding_its_own_value(path):
    """Remove each widget in turn; the survivors must keep their own markers.

    This is the three-part removal (CLAUDE.md section 0) asserted at its real
    consequence rather than at its mechanics: drop the descriptor and the value
    at mismatched indexes and the survivors shift by one, which no count check
    can see because the counts stay equal.
    """
    ws = _tool()
    data = _graph(path)
    nodes = _migratable_nodes(data)
    # One node per TYPE is enough: the tool operates by type and every node of
    # a type carries the same descriptor list.
    by_type = {}
    for node in nodes:
        by_type.setdefault(str(node.get("type") or ""), node)
    assert by_type, "%s has nothing to remove from" % path.name

    removals = 0
    for node_type, original in sorted(by_type.items()):
        for victim in saved_widget_order(original):
            wf = copy.deepcopy(data)
            skip = False
            for n in wf["nodes"]:
                if str(n.get("type") or "") != node_type:
                    continue
                if len(saved_widget_order(n)) != len(n.get("widgets_values") or []):
                    skip = True
                    break
                _stamp(n)
            if skip:
                continue

            ws.remove_widget(wf, node_type, victim)

            for after in wf["nodes"]:
                if str(after.get("type") or "") != node_type:
                    continue
                survivors = saved_widget_order(after)
                assert victim not in survivors, (
                    "%s node %s (%s): removing %r left its descriptor behind"
                    % (path.name, after.get("id"), node_type, victim))
                for name, value in _pairs(after):
                    assert value == _MARK % name, (
                        "%s node %s (%s): removing %r left widget %r holding "
                        "%r, which belongs to %r. The descriptor and the value "
                        "were dropped at different indexes, so every widget "
                        "after the removal is off by one -- invisible to a "
                        "count check, because the counts still match.\n"
                        "pairs: %r"
                        % (path.name, after.get("id"), node_type, victim, name,
                           value, str(value).split("::")[-1], _pairs(after)))
            assert ws.verify(wf, path.name) == [], ws.verify(wf, path.name)
            removals += 1

    assert removals, "%s: no removal was exercised" % path.name


@pytest.mark.parametrize("path", GRAPHS, ids=lambda p: p.name)
def test_the_positional_restore_lands_every_value_on_its_own_widget(path):
    """The restore ComfyUI will really perform, spelled out.

    Frontend 1.51.10 zips `widgets_values` onto the live widget list by index.
    So stamp by SAVED name, zip onto the LIVE names, and require the pairing to
    survive. This is the same fact guard 1 asserts as an order comparison,
    stated as the operation it actually breaks -- if the two ever disagree, the
    order comparison is the one that is wrong.
    """
    data = _graph(path)
    problems = []
    checked = 0
    for node in data.get("nodes") or []:
        cls = NODE_CLASS_MAPPINGS.get(str(node.get("type") or ""))
        if cls is None or not hasattr(cls, "INPUT_TYPES"):
            continue
        saved = saved_widget_order(node)
        if not saved or len(saved) != len(node.get("widgets_values") or []):
            continue
        values = [_MARK % n for n in saved]
        checked += 1
        for live_name, value in zip(live_widget_order(cls), values):
            if value != _MARK % live_name:
                problems.append(
                    "  node %s (%s): live widget %r would be restored with the "
                    "value saved for %r"
                    % (node.get("id"), node.get("type"), live_name,
                       str(value).split("::")[-1]))
                break
    assert checked, "%s compared no node against a live class" % path.name
    assert not problems, (
        "%s would restore values onto the wrong widgets on frontend 1.51.10:\n%s"
        % (path.name, "\n".join(problems)))


def test_the_pairing_check_catches_an_arithmetic_removal():
    """THE MUTATION TEST -- the exact mistake section 0 was written after.

    Remove a descriptor but pop the VALUE at the wrong index, which is what
    "subtract one" produces and what a count check cannot see. The pairing must
    fail, and it must name the widget that inherited someone else's value.
    """
    data = _graph(_REPO / "workflows" / "otr_canonical.json")
    node = next(n for n in data["nodes"]
                if len(saved_widget_order(n)) >= 4
                and len(n.get("widgets_values") or []) == len(saved_widget_order(n)))
    _stamp(node)
    assert all(v == _MARK % n for n, v in _pairs(node)), "precondition: aligned"

    names = saved_widget_order(node)
    victim = names[1]
    desc_index = next(i for i, inp in enumerate(node["inputs"])
                      if (inp.get("widget") or {}).get("name") == victim)
    node["inputs"].pop(desc_index)
    node["widgets_values"].pop(0)          # the off-by-one: 0 instead of 1

    assert len(saved_widget_order(node)) == len(node["widgets_values"]), (
        "the mutation must keep the COUNTS equal, or it is caught by an "
        "easier guard and proves nothing about this one")
    mispaired = [(n, v) for n, v in _pairs(node) if v != _MARK % n]
    assert mispaired, (
        "a descriptor and a value were dropped at different indexes and the "
        "pairing check still calls the node aligned -- this guard has no teeth")
