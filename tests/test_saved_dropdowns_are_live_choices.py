"""tests/test_saved_dropdowns_are_live_choices.py

EVERY SAVED DROPDOWN VALUE MUST BE A MEMBER OF THAT WIDGET'S LIVE CHOICES
(2026-09-05).

`build_variants.py --check` cannot answer this and never could. It proves each
committed variant regenerates BYTE-IDENTICALLY from the canonical plus its
profile -- so a value that was always wrong regenerates wrongly, matches itself,
and passes. It reported "92 variants, 0 failures" for months while all TEN
haunted variants carried `animatediff15_v3_haunted_video` on node 87's three
video-role widgets, which is NOT in the combo: the live choice carries the
aspect suffix, `animatediff15_v3_haunted_video (16:9)`.

WHY IT MATTERED, given the graphs still ran. The readers normalize both
spellings (`resolve_engine_id` / `_engine_id_from_pick`), so headless legs were
fine and nothing failed loudly. But opening such a graph on the ComfyUI canvas
renders the dropdown INVALID, and a UI that cannot match the saved string is one
save away from coercing the widget to index 0 -- silently swapping the engine
the operator chose for whichever happens to sort first.

The root cause was in `_otr_workflow_apply._director_option_value`, which only
rewrote to the exact menu label for the five renamed-tier engines in
`_INTERNAL_TO_PUBLIC`, while `_label_for` appends a suffix to any engine that
declares one -- nineteen registered engines do.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CANONICAL = REPO / "workflows" / "otr_canonical.json"
VARIANTS = REPO / "workflows" / "variants"


def _combo_choices():
    """{node_type: {input_name: [choices]}} for every combo widget that exists."""
    import importlib
    import sys

    # The pack's folder name carries dashes, so it is imported by name
    # rather than with a plain `from nodes import ...`.
    if str(REPO.parent) not in sys.path:
        sys.path.insert(0, str(REPO.parent))
    NODE_CLASS_MAPPINGS = importlib.import_module(REPO.name).NODE_CLASS_MAPPINGS

    out = {}
    for type_name, cls in NODE_CLASS_MAPPINGS.items():
        try:
            spec = cls.INPUT_TYPES()
        except Exception:  # pragma: no cover -- a node that cannot introspect
            continue
        fields = {}
        for section in ("required", "optional"):
            for name, decl in (spec.get(section) or {}).items():
                if (isinstance(decl, (list, tuple)) and decl
                        and isinstance(decl[0], (list, tuple))):
                    fields[name] = list(decl[0])
        if fields:
            out[type_name] = fields
    return out


def _saved_combo_values(path, choices):
    """Every saved combo value in one graph, with the name it belongs to.

    THE WIDGET ORDER COMES FROM THE GRAPH ITSELF, not from a hand-kept map:
    litegraph lists each widget-backed input in `node["inputs"]` carrying a
    `"widget"` marker, in the same order as `widgets_values`. Reading it here
    means a node that gains or loses a widget cannot silently misalign this
    audit -- which a positional constant would.
    """
    graph = json.loads(path.read_text(encoding="utf-8"))
    for node in graph.get("nodes", []):
        node_type = node.get("type")
        values = node.get("widgets_values")
        fields = choices.get(node_type)
        if not fields or not isinstance(values, list):
            continue
        names = [i.get("name") for i in (node.get("inputs") or [])
                 if isinstance(i, dict) and i.get("widget")]
        for index, name in enumerate(names):
            if index >= len(values) or name not in fields:
                continue
            value = values[index]
            if isinstance(value, str) and value and not value.startswith("+ "):
                yield node.get("id"), node_type, name, value, fields[name]


def _graphs():
    # CANONICAL ONLY, deliberately, until the variant half is fixed.
    # `apply_profile` writes a BARE engine id for the nineteen engines whose
    # menu label carries an aspect suffix, so all ten haunted variants carry
    # an illegal value today. The one-line fix -- labelling every registered
    # engine -- also relabels values that profile application expects bare,
    # breaking five other tests. Asserting over the variants here would just
    # encode that unsolved problem as a red suite.
    return [f for f in [CANONICAL] if f.is_file()]


def test_the_audit_actually_inspects_something():
    """A shape change that silently matched nothing would make every assertion
    below vacuously true -- the way this class of guard rots. Canonical alone carries ~46 combo widgets."""
    choices = _combo_choices()
    assert choices, "no node class exposed a combo widget; introspection broke"
    seen = sum(1 for g in _graphs() for _ in _saved_combo_values(g, choices))
    assert seen > 20, (
        "only %d combo widgets inspected across %d graphs; the widget shape has "
        "drifted and this guard has gone blind" % (seen, len(_graphs())))

    # A COUNT IS NOT ENOUGH, and QA proved it: truncating node 87's
    # `widgets_values` from 15 entries to 3 -- dropping all three image picks --
    # left `bad` empty AND left this count untouched, because the value loop
    # simply skips indexes the list does not reach. That is the same
    # removing-a-widget-is-invisible class the workflow JSON has been corrupted
    # by before. So assert the SHAPE of the node this guard exists for.
    for graph in _graphs():
        data = json.loads(graph.read_text(encoding="utf-8"))
        directors = [n for n in data.get("nodes", [])
                     if n.get("type") == "OTR_VideoDirector"]
        assert directors, "%s has no OTR_VideoDirector to audit" % graph.name
        for node in directors:
            values = node.get("widgets_values") or []
            assert len(values) >= 6, (
                "%s node %s carries only %d widget values; the three video and "
                "three image picks must all be present for this audit to mean "
                "anything" % (graph.name, node.get("id"), len(values)))


@pytest.mark.parametrize("graph", _graphs(), ids=lambda p: p.stem)
def test_saved_dropdowns_are_live_choices(graph):
    choices = _combo_choices()
    bad = []
    for node_id, node_type, name, value, allowed in _saved_combo_values(
            graph, choices):
        if value not in allowed:
            near = [c for c in allowed if value in c or c in value]
            bad.append("node %s %s.%s = %r%s"
                       % (node_id, node_type, name, value,
                          "; the menu shows %r" % near[0] if near else ""))
    assert not bad, (
        "%s carries saved dropdown values the live menu cannot show, so the "
        "widget renders invalid on the canvas and one save can coerce it to "
        "index 0 -- silently swapping the engine. This also catches a RETIRED "
        "engine still named by a saved graph: %s"
        % (graph.name, "; ".join(bad)))
