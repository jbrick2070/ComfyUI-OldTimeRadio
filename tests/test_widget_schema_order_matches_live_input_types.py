"""Every shipped graph's widget ORDER must match the LIVE node class.

THE HOLE THIS CLOSES, and it is the one the existing gates structurally cannot
see. `test_widget_value_alignment` proves the 17 shipped graphs agree with EACH
OTHER, and `test_canonical_widget_input_parity` proves each node's descriptor
count equals its value count. Both are satisfied by 17 graphs that agree on the
same WRONG order -- which is exactly what a reorder of `INPUT_TYPES` produces
if the canonical is re-indexed by hand and the class is not, or the other way
round. The graphs stay consistent with one another, every one of them is now
misaligned against the Python class, and every value at or after the moved
index is restored into the wrong widget.

That failure is silent by construction. ComfyUI frontend 1.51.10 restores
`widgets_values` by POSITION: `Comfy.Workflow.NamedValuesRestore` is
`defaultValue: false, experimental: true`, and `fallbackWidgetsValuesNames`
exists in the frontend's node-def schema but backend 0.35.1 never emits it
(`server.py::node_info` has no such key). Names do not participate in the
restore at all, so nothing at runtime will notice or complain.

WHAT IS ASSERTED, and why this exact shape. The saved `inputs` array holds link
sockets and widget descriptors TOGETHER, and their interleaving is graph
authoring history, not contract: the writer carries `gate_in` at descriptor 32
where it is declared, while CastLock, both directors and VideoRenderBatch carry
their link sockets hoisted to the front. Asserting the full array order would
fail on 153 nodes that are perfectly correct. What IS contract is the WIDGET
SUBSEQUENCE -- strip the sockets from the saved array and it must equal the
widget-bearing entries of the live `INPUT_TYPES`, in declaration order.

Measured 2026-09-13, before any widget work: that holds 357 times across the 17
shipped graphs, with zero exceptions and zero unresolved node types.

REQUIRED BEFORE ANY REORDER (2026-09-13 QA verdict, guard 1 of 2). The sibling
guard is `tests/test_widget_migration_pairs_values_by_name.py`.
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_WORKFLOWS = _REPO / "workflows"
_VARIANTS = _WORKFLOWS / "variants"

# ComfyUI renders an input as a WIDGET when its declared type is a COMBO (a
# list of choices) or one of these primitives -- and not when the declaration
# carries forceInput, which demands a wire instead.
_WIDGET_PRIMITIVES = frozenset(("STRING", "INT", "FLOAT", "BOOLEAN"))


def _mappings() -> dict:
    """The pack's registered nodes, imported the way ComfyUI imports them.

    Same route as `test_input_types_signature_parity` -- the package root is a
    hyphenated directory, so it is importable only as a child of its parent.
    """
    sys.path.insert(0, str(_REPO.parent))
    try:
        pkg = importlib.import_module(_REPO.name)
    finally:
        sys.path.pop(0)
    return dict(getattr(pkg, "NODE_CLASS_MAPPINGS", {}) or {})


NODE_CLASS_MAPPINGS = _mappings()


def live_widget_order(cls) -> list:
    """The widget names the LIVE class declares, in declaration order."""
    spec = cls.INPUT_TYPES()
    out = []
    for section in ("required", "optional"):
        block = spec.get(section) or {}
        if not isinstance(block, dict):
            continue
        for name, decl in block.items():
            if not isinstance(decl, (list, tuple)) or not decl:
                continue
            declared_type = decl[0]
            opts = decl[1] if len(decl) > 1 and isinstance(decl[1], dict) else {}
            if opts.get("forceInput"):
                continue  # a wire, never a widget
            if isinstance(declared_type, (list, tuple)):
                out.append(str(name))            # COMBO
            elif isinstance(declared_type, str) and declared_type in _WIDGET_PRIMITIVES:
                out.append(str(name))
    return out


def saved_widget_order(node) -> list:
    """The widget descriptor names the SAVED graph carries, in order.

    Sockets are skipped; what survives is the subsequence the positional
    restore actually zips `widgets_values` onto.
    """
    out = []
    for inp in node.get("inputs") or []:
        widget = inp.get("widget")
        if isinstance(widget, dict) and widget.get("name"):
            out.append(str(widget["name"]))
    return out


def _graphs() -> list:
    """Every shipped graph -- the canonical, any hand-authored sibling, and the
    16 generated variants.

    `workflows/otr_mac_lightning.json` lived in `variants/` until 2026-09-09,
    so a glob of `variants/` alone would have stopped covering it the day it
    moved. That is the same shape of miss that left
    `test_workflow_link_target_indexes` collecting one item of seventeen.
    """
    out = list(sorted(_WORKFLOWS.glob("*.json")))
    out += sorted(_VARIANTS.glob("*.json"))
    keep = []
    for path in out:
        if "external_examples" in path.parts:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data.get("nodes"), list):
            keep.append(path)
    return keep


GRAPHS = _graphs()


def test_the_graphs_and_the_registry_both_actually_loaded():
    """A guard that compares nothing passes on everything.

    This file's whole value is the comparison COUNT, so pin the two inputs it
    needs: the shipped graphs, and a node registry that is not empty. The
    registry is the fragile one -- `test_widget_value_alignment` is deliberately
    import-free because an earlier draft of IT resolved classes through a
    mapping that came back empty and turned every assertion into a no-op.
    """
    assert len(GRAPHS) >= 17, "expected the 17 shipped graphs, found %d: %r" % (
        len(GRAPHS), [p.name for p in GRAPHS])
    # The pack declares 24 nodes (`node_list.json`).
    assert len(NODE_CLASS_MAPPINGS) >= 24, (
        "the node registry resolved %d classes -- if it is empty or short, the "
        "comparisons below silently skip and this file proves nothing"
        % len(NODE_CLASS_MAPPINGS))


def test_every_otr_node_in_a_shipped_graph_resolves_to_a_live_class():
    """Coverage must not be allowed to collapse quietly.

    Each unresolved type is a node this guard skips. Measured 2026-09-13: zero
    unresolved. A node that stops importing -- a missing dependency, a renamed
    module -- would otherwise remove itself from this check without failing it.
    """
    unresolved = set()
    for path in GRAPHS:
        data = json.loads(path.read_text(encoding="utf-8"))
        for node in data.get("nodes") or []:
            node_type = str(node.get("type") or "")
            if node_type.startswith("OTR_") and node_type not in NODE_CLASS_MAPPINGS:
                unresolved.add(node_type)
    assert not unresolved, (
        "shipped graphs use OTR node type(s) that do not resolve to a live "
        "class, so the order guard below skips them entirely: %r"
        % sorted(unresolved))


@pytest.mark.parametrize("path", GRAPHS, ids=lambda p: p.name)
def test_saved_widget_order_matches_the_live_class(path):
    """The guard itself, per graph, naming the node and both orders."""
    data = json.loads(path.read_text(encoding="utf-8"))
    compared = 0
    problems = []
    for node in data.get("nodes") or []:
        cls = NODE_CLASS_MAPPINGS.get(str(node.get("type") or ""))
        if cls is None or not hasattr(cls, "INPUT_TYPES"):
            continue
        saved = saved_widget_order(node)
        live = live_widget_order(cls)
        compared += 1
        if saved == live:
            continue
        first = next((i for i, (a, b) in enumerate(zip(saved, live)) if a != b),
                     min(len(saved), len(live)))
        problems.append(
            "  node %s (%s) diverges at widget index %d\n"
            "    saved: %r\n    live : %r"
            % (node.get("id"), node.get("type"), first, saved, live))
    assert compared, (
        "%s matched no registered node type -- the comparison ran on nothing"
        % path.name)
    assert not problems, (
        "%s declares a widget order its node class does not.\n"
        "widgets_values restores by POSITION on frontend 1.51.10 (named "
        "restore is off by default and the backend emits no name fallback), so "
        "every value at or after the divergent index lands in the wrong "
        "widget -- silently, at load, with nothing at runtime to complain.\n"
        "Fix the graph to match INPUT_TYPES (scripts/otr_widget_surgery.py, "
        "then build_variants.py --all), never the other way round.\n%s"
        % (path.name, "\n".join(problems)))


def test_the_order_comparison_would_catch_a_graph_that_agrees_with_its_siblings():
    """THE MUTATION TEST -- and it targets the precise blind spot.

    Take a real graph, move one widget descriptor, and require the comparison
    to fail. Crucially this mutation is INVISIBLE to the sibling gates if it is
    applied to all 17 files at once: they would still agree with each other and
    their counts would still match. Only a comparison against the live class
    sees it.
    """
    data = json.loads((_WORKFLOWS / "otr_canonical.json").read_text(encoding="utf-8"))
    node = next(n for n in data["nodes"]
                if len(saved_widget_order(n)) >= 2
                and str(n.get("type")) in NODE_CLASS_MAPPINGS)
    cls = NODE_CLASS_MAPPINGS[str(node["type"])]
    assert saved_widget_order(node) == live_widget_order(cls), (
        "precondition: the fixture node must start aligned")

    widget_positions = [i for i, inp in enumerate(node["inputs"])
                        if isinstance(inp.get("widget"), dict)]
    a, b = widget_positions[0], widget_positions[-1]
    node["inputs"][a], node["inputs"][b] = node["inputs"][b], node["inputs"][a]

    assert saved_widget_order(node) != live_widget_order(cls), (
        "two widget descriptors were swapped and the comparison still calls "
        "the graph aligned -- this guard has no teeth")
