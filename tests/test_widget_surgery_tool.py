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
    # UNPACK BOTH. remove_widget returns (touched, repairs) since it took on
    # part 3 itself; binding the pair to one name made `assert touched` always
    # true, because a 2-tuple is truthy even when the widget was never found.
    touched, repairs = ws.remove_widget(
        wf, "OTR_LedgerScriptWriter", "perfect_run_spacesaver")
    assert touched, "perfect_run_spacesaver is not on the writer any more"
    assert touched[0]["dropped_value"] is False, touched

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


def test_removing_a_widget_that_is_not_there_reports_nothing():
    """Guards the shape of the return, not just its truthiness.

    `remove_widget` returns (touched, repairs) since it took on part 3, and a
    caller that binds the pair to a single name gets a 2-tuple that is truthy
    even when the widget was never found -- which silently turned the
    "is it still on the writer?" assertion above into a no-op.
    """
    ws = _tool()
    wf = _canonical()
    touched, repairs = ws.remove_widget(wf, "OTR_LedgerScriptWriter",
                                        "a_widget_that_never_existed")
    assert touched == [], touched
    assert repairs == [], repairs
    assert ws.verify(wf, "unchanged") == []


def test_removing_from_a_short_widgets_values_is_refused():
    """The refusal remove_widget gained had NO test, which makes a guarantee
    decoration -- it is only real if something goes red when it is gone.

    Skipping the value pop on an already-short list and removing the descriptor
    anyway widens a pre-existing mismatch by one, silently.
    """
    ws = _tool()
    wf = _canonical()
    node = next(n for n in wf["nodes"] if n.get("type") == "OTR_LedgerScriptWriter")
    node["widgets_values"] = node["widgets_values"][:-1]

    with pytest.raises(ValueError, match="saved values"):
        ws.remove_widget(wf, "OTR_LedgerScriptWriter", "perfect_run_spacesaver")


def test_removing_from_a_node_with_no_widgets_values_is_refused():
    """ABSENT is not the same as short, and it is worse: the node declares
    widget descriptors and saves no values, so there is nothing to drop and no
    way to know what was meant. The length check alone let this through,
    because a missing key is not a short list."""
    ws = _tool()
    wf = _canonical()
    node = next(n for n in wf["nodes"] if n.get("type") == "OTR_LedgerScriptWriter")
    node.pop("widgets_values", None)

    with pytest.raises(ValueError, match="no widgets_values"):
        ws.remove_widget(wf, "OTR_LedgerScriptWriter", "perfect_run_spacesaver")


def test_a_repair_reports_a_stale_node_as_well_as_a_stale_slot():
    """repair_dst_slots corrected a wrong dst_node in place and returned it in
    nothing, so a caller reading `repairs == []` as "the graph was already
    consistent" was wrong -- against a docstring promising the return value
    shows what the repair did."""
    ws = _tool()
    wf = _canonical()
    # Point one link row at a node that does not hold it, leaving the slot right.
    node = next(n for n in wf["nodes"] if any(i.get("link") is not None
                                              for i in (n.get("inputs") or [])))
    idx, inp = next((i, x) for i, x in enumerate(node["inputs"])
                    if x.get("link") is not None)
    row = next(r for r in wf["links"] if r[0] == inp["link"])
    row[3] = 9999

    repairs = ws.repair_dst_slots(wf)

    assert repairs, "a stale dst_node was corrected but reported in nothing"
    entry = next(r for r in repairs if r["link"] == inp["link"])
    assert entry["from"] == (9999, idx), entry
    assert entry["to"] == (node["id"], idx), entry
    assert ws.verify(wf, "after") == []


def test_rename_widget_changes_the_name_and_nothing_else():
    """rename_widget had no coverage at all, and the next window is told to use
    display_name INSTEAD of renaming -- so if this function is ever reached, it
    will be by someone who decided to rename anyway. It should at least be
    provably narrow: the name changes, the position does not, and no link moves.
    """
    ws = _tool()
    wf = _canonical()
    node = next(n for n in wf["nodes"] if n.get("type") == "OTR_LedgerScriptWriter")
    before_order = ws.widget_names(node)
    before_values = list(node["widgets_values"])
    before_links = {r[0]: list(r) for r in wf["links"]}

    touched = ws.rename_widget(wf, "OTR_LedgerScriptWriter", "creativity", "flair")

    assert touched, "creativity is no longer on the writer"
    after = next(n for n in wf["nodes"] if n.get("type") == "OTR_LedgerScriptWriter")
    names = ws.widget_names(after)
    assert names == ["flair" if n == "creativity" else n for n in before_order]
    assert after["widgets_values"] == before_values, "a rename moved a value"
    assert {r[0]: list(r) for r in wf["links"]} == before_links, "a rename moved a link"
    assert ws.verify(wf, "after-rename") == []


def test_save_preserves_each_file_s_own_shape(tmp_path):
    """The canonical is pretty-printed and the variants are single-line. save()
    takes a `compact` flag and a caller who passes the wrong one rewrites the
    whole file, producing a diff that hides the real change inside it."""
    ws = _tool()
    wf = _canonical()

    pretty = tmp_path / "pretty.json"
    ws.save(str(pretty), wf, compact=False)
    text = pretty.read_text(encoding="utf-8")
    assert text.endswith("\n"), "pretty form lost its trailing newline"
    assert text.count("\n") > 100, "pretty form is not multi-line"
    assert ws.load(str(pretty)) == wf, "pretty round-trip changed the graph"

    compact = tmp_path / "compact.json"
    ws.save(str(compact), wf, compact=True)
    ctext = compact.read_text(encoding="utf-8")
    assert ctext.count("\n") == 0, "compact form is not single-line"
    assert ", " not in ctext[:200], "compact form kept separator padding"
    assert ws.load(str(compact)) == wf, "compact round-trip changed the graph"


def test_the_shipped_files_match_the_shape_save_would_write():
    """The real canonical is pretty and the real variants are single-line -- so
    a caller using save() with the matching flag produces no incidental diff."""
    import glob
    canonical = CANONICAL.read_text(encoding="utf-8")
    assert canonical.count("\n") > 100 and canonical.endswith("\n")
    for p in glob.glob(str(REPO / "workflows" / "variants" / "*.json")):
        assert Path(p).read_text(encoding="utf-8").count("\n") <= 1, p
