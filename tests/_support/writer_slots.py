"""Resolve a node's widget positions BY NAME, never by a hardcoded index.

WHY THIS EXISTS (2026-09-13). Removing one widget from the writer shifted every
later slot by one and turned fourteen assertions across seven test files red --
all of them of the form ``order[21] == "source_bank"`` or
``widgets_values[25] == ""``. The obvious repair is to subtract one from each
literal. That is the exact move CLAUDE.md section 0 forbids for the link table,
for exactly the same reason: it is arithmetic performed on a layout the author
reconstructed from comments, and those comments were already stale. Three of the
four previous migrations had left at least one comment block behind.

The deeper problem is that a shifted index does not necessarily FAIL. The real
saved values are overwhelmingly ``""``, ``False`` and ``0``, so an assertion
that drifts onto its neighbour frequently still passes -- it just stops checking
what it claims to check, silently, forever.

So: resolve the position from the structure's own descriptors at assertion time.
The test then says what it means, survives any future reorder untouched, and
fails with the widget's own NAME when the widget is what actually changed.

THE ONE PLACE ABSOLUTE ORDER IS STILL PINNED is
``tests/test_openrouter_slot_widgets_s2.py::_EXPECTED_INPUT_ORDER``, which
states the writer's declared order once as a single list. Everything else asks
these helpers. Do not re-pin absolute indexes anywhere else; a second copy of
the order is a second thing to forget to update.
"""
from __future__ import annotations


def widget_names(node: dict) -> list:
    """The node's widget descriptor names, in saved order.

    This is the sequence ``widgets_values`` is positionally zipped onto, so
    ``widget_names(node)[i]`` names ``node["widgets_values"][i]``. Link sockets
    are skipped because they consume no saved value.
    """
    out = []
    for inp in node.get("inputs") or []:
        widget = inp.get("widget")
        if isinstance(widget, dict) and widget.get("name"):
            out.append(str(widget["name"]))
    return out


def slot(node: dict, name: str) -> int:
    """Index into ``widgets_values`` for the widget called ``name``.

    Raises AssertionError naming the widget -- and listing what the node does
    carry -- rather than returning a wrong index or an IndexError, so a test
    that outlives its widget says so in one line.
    """
    names = widget_names(node)
    assert name in names, (
        "node %s (%s) has no widget named %r -- this assertion is pinning a "
        "control that no longer exists. Widgets: %r"
        % (node.get("id"), node.get("type"), name, names))
    return names.index(name)


def value(node: dict, name: str):
    """The saved value of the widget called ``name``.

    Refuses a node whose descriptor and value counts disagree, because the
    positional pairing is meaningless there and a value read out of such a node
    is whatever happened to land at that offset.
    """
    names = widget_names(node)
    values = node.get("widgets_values")
    assert isinstance(values, list) and len(values) == len(names), (
        "node %s (%s) carries %d widget descriptor(s) and %s saved value(s); "
        "no value can be read by name until those agree"
        % (node.get("id"), node.get("type"), len(names),
           len(values) if isinstance(values, list) else "no"))
    return values[names.index(name)]


def assert_relative_order(order, names) -> None:
    """``names`` must appear in ``order``, in this sequence and adjacent.

    The assertion most of the converted tests actually wanted. They pinned
    absolute indexes to express "these widgets sit together, in this order" --
    a claim about the GROUP, which stays true when something earlier in the node
    is added or removed, where an absolute index does not.
    """
    order = list(order)
    missing = [n for n in names if n not in order]
    assert not missing, (
        "declared order is missing %r; it has %r" % (missing, order))
    first = order.index(names[0])
    actual = order[first:first + len(names)]
    assert actual == list(names), (
        "expected %r to sit together in this order; found %r at that position "
        "in %r" % (list(names), actual, order))
