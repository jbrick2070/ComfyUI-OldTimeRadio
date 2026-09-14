"""The positional-surgery tool is trusted for edits to the shipped graphs, so
it gets the same guard the graphs do.

`scripts/otr_widget_surgery.py` has no production caller by design -- it is the
procedure CLAUDE.md section 0 describes, made executable so a window under time
pressure does not re-derive it. That is exactly the shape this repo keeps
getting wrong (correct code nothing reaches), so the test is the wiring: it
proves the tool on the REAL canonical graph rather than on a fixture that
agrees with it.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CANONICAL = REPO / "workflows" / "otr_canonical.json"
TOOL = REPO / "scripts" / "otr_widget_surgery.py"


def _tool():
    spec = importlib.util.spec_from_file_location("otr_widget_surgery", TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _canonical():
    return json.loads(CANONICAL.read_text(encoding="utf-8"))


def test_the_canonical_is_clean_before_any_surgery():
    ws = _tool()
    assert ws.verify(_canonical(), "canonical") == []


def test_a_widget_descriptor_that_carries_a_live_link_survives_a_reorder():
    """THE BLOCKER THIS FILE WAS WRITTEN FOR (2026-09-13).

    A widget converted to an input and WIRED carries its own `link`, and three
    do today: OTR_SceneSequencer's script_json, and OTR_SignalLostVideo's
    script_json and news_used. Moving one used to leave its link row pointing
    at the slot it vacated, so ComfyUI would feed that wire into whichever
    widget landed there -- invisible to a widget-count check.

    reorder_widgets now repairs the link table itself rather than trusting the
    caller to remember.
    """
    ws = _tool()
    wf = _canonical()

    node = next(n for n in wf["nodes"] if n.get("type") == "OTR_SceneSequencer")
    names = ws.widget_names(node)
    linked = [nm for nm, i in zip(names, ws.widget_descriptor_indexes(node))
              if node["inputs"][i].get("link") is not None]
    assert linked, "OTR_SceneSequencer no longer has a linked widget descriptor"

    # Move every linked widget to the end -- the worst case for its link row.
    moved = [n for n in names if n not in linked] + linked
    ws.reorder_widgets(wf, "OTR_SceneSequencer", moved)

    problems = ws.verify(wf, "after-reorder")
    assert problems == [], problems


def test_the_reorder_repairs_without_being_asked():
    """The caller is not trusted to run part 3. Assert the tool returns its own
    repairs and that a second pass finds nothing left to do."""
    ws = _tool()
    wf = _canonical()
    node = next(n for n in wf["nodes"] if n.get("type") == "OTR_SceneSequencer")
    names = ws.widget_names(node)

    _, repairs = ws.reorder_widgets(wf, "OTR_SceneSequencer", list(reversed(names)))
    assert ws.verify(wf, "after") == []
    # Identity repair is idempotent: nothing is left over.
    assert ws.repair_dst_slots(wf) == []
    assert isinstance(repairs, list)


def test_every_value_travels_with_its_own_widget():
    """Real defaults are "", False and 0, so two swapped widgets can hold equal
    values and hide the swap. Pair by NAME with symbolic markers instead."""
    ws = _tool()
    wf = _canonical()
    node = next(n for n in wf["nodes"] if n.get("type") == "OTR_LedgerScriptWriter")
    names = ws.widget_names(node)
    node["widgets_values"] = ["MARK::%s" % n for n in names]

    ws.reorder_widgets(wf, "OTR_LedgerScriptWriter", list(reversed(names)))

    after = next(n for n in wf["nodes"] if n.get("type") == "OTR_LedgerScriptWriter")
    for nm, val in zip(ws.widget_names(after), after["widgets_values"]):
        assert val == "MARK::%s" % nm, (nm, val)


def test_removing_a_widget_repairs_the_link_that_follows_it():
    """perfect_run_spacesaver sits at descriptor 8; gate_in is 32 and carries
    link 279. Removing the first moves the second, and the repair is by
    IDENTITY -- match inputs[i].link to the row's id -- never by arithmetic."""
    ws = _tool()
    wf = _canonical()

    before = {r[0]: list(r) for r in wf["links"]}
    touched = ws.remove_widget(wf, "OTR_LedgerScriptWriter", "perfect_run_spacesaver")
    assert touched, "perfect_run_spacesaver is not on the writer any more"
    repairs = ws.repair_dst_slots(wf)

    assert ws.verify(wf, "after-removal") == []
    assert ws.repair_dst_slots(wf) == [], "the repair is not idempotent"
    # Only slots moved; no link changed which nodes it joins.
    after = {r[0]: list(r) for r in wf["links"]}
    for lid, row in after.items():
        assert row[1:4] == before[lid][1:4], (lid, before[lid], row)


def test_a_short_widgets_values_is_refused_rather_than_backfilled():
    """Writing None for a missing value turns a graph that is merely
    inconsistent into one that is confidently wrong, and None is a value
    ComfyUI will hand to a widget."""
    ws = _tool()
    wf = _canonical()
    node = next(n for n in wf["nodes"] if n.get("type") == "OTR_SceneSequencer")
    names = ws.widget_names(node)
    node["widgets_values"] = node["widgets_values"][:-1]

    with pytest.raises(ValueError, match="saved values"):
        ws.reorder_widgets(wf, "OTR_SceneSequencer", list(names))


def test_a_partial_order_is_refused():
    """An order that is not a permutation would silently drop a widget."""
    ws = _tool()
    wf = _canonical()
    names = ws.widget_names(
        next(n for n in wf["nodes"] if n.get("type") == "OTR_SceneSequencer"))
    with pytest.raises(ValueError, match="permutation"):
        ws.reorder_widgets(wf, "OTR_SceneSequencer", names[:-1])
