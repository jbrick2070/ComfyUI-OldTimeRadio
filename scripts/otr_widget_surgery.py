"""Positional widget surgery on a litegraph workflow JSON, done the only way
this repo has found to be safe.

Removing or moving a widget touches THREE things, and the third is the one that
is invisible to every count-based check (CLAUDE.md section 0):

  1. `widgets_values` -- the saved VALUE at the widget's index.
  2. the `inputs` DESCRIPTOR array -- the `{"widget": {"name": ...}}` entry.
  3. every link whose `dst_slot` indexes past the descriptor that moved.

`dst_slot` is an index into that same `inputs` array, which holds link sockets
and widget descriptors together. Repair it BY IDENTITY -- set each dst_slot to
the index whose `inputs[i].link` equals that link's id -- because identity is
self-correcting and impossible to double-apply, where arithmetic is neither.

The widget VALUE index is NOT the descriptor index. `inputs` interleaves real
link sockets with widget descriptors, so the value at `widgets_values[k]`
belongs to the k-th entry that carries a "widget" key, not to `inputs[k]`.
"""
from __future__ import annotations

import json
from pathlib import Path


def widget_descriptor_indexes(node):
    """Positions in `inputs` that carry a widget descriptor, in order.

    The n-th of these corresponds to widgets_values[n]."""
    return [i for i, inp in enumerate(node.get("inputs") or [])
            if inp.get("widget")]


def widget_names(node):
    return [(node["inputs"][i].get("widget") or {}).get("name")
            for i in widget_descriptor_indexes(node)]


def repair_dst_slots(wf):
    """Set every link row's dst_slot to the slot that actually holds it.

    Returns the repairs made, as (link_id, old_slot, new_slot)."""
    links_by_id = {row[0]: row for row in wf.get("links", [])}
    repairs = []
    for node in wf.get("nodes", []):
        for idx, inp in enumerate(node.get("inputs") or []):
            lid = inp.get("link")
            if lid is None:
                continue
            row = links_by_id.get(lid)
            if row is None or len(row) < 5:
                continue
            if row[3] != node["id"]:
                row[3] = node["id"]
            if row[4] != idx:
                repairs.append((lid, row[4], idx))
                row[4] = idx
    return repairs


def remove_widget(wf, node_type, widget_name):
    """Drop one widget from every node of `node_type` in this graph.

    Does parts 1 and 2; the caller runs repair_dst_slots for part 3."""
    touched = []
    for node in wf.get("nodes", []):
        if node.get("type") != node_type:
            continue
        desc_idx = widget_descriptor_indexes(node)
        names = widget_names(node)
        if widget_name not in names:
            continue
        value_pos = names.index(widget_name)
        input_pos = desc_idx[value_pos]

        wv = node.get("widgets_values")
        dropped = None
        if isinstance(wv, list) and value_pos < len(wv):
            dropped = wv.pop(value_pos)
        node["inputs"].pop(input_pos)
        touched.append({
            "node_id": node["id"], "value_pos": value_pos,
            "input_pos": input_pos, "dropped_value": dropped,
            "widgets_values_len": len(node.get("widgets_values") or []),
            "inputs_len": len(node.get("inputs") or []),
        })
    return touched


def reorder_widgets(wf, node_type, new_name_order):
    """Reorder a node's widgets to `new_name_order` (a full permutation).

    Moves the widget DESCRIPTORS among themselves and permutes widgets_values
    to match, then REPAIRS THE LINK TABLE ITSELF before returning.

    The repair is not optional and the caller is not trusted to remember it. A
    widget descriptor can carry a live `link` of its own -- a widget converted
    to an input and wired -- and today three do: OTR_SceneSequencer's
    script_json (link 277), and OTR_SignalLostVideo's script_json (16) and
    news_used (110). Moving one of those leaves its link row pointing at the
    slot it used to occupy, so ComfyUI feeds that wire's value into whichever
    widget now sits there. That is invisible to a widget-count check and is
    exactly the class CLAUDE.md section 0 exists for.

    An earlier docstring said this "leaves non-widget link sockets exactly
    where they are", which is true and reads as a completeness guarantee it
    never was: the sockets OUTSIDE the widget group do not move, but a linked
    widget INSIDE it does.

    Returns (touched, repairs) so a caller can see what the repair did.
    """
    touched = []
    for node in wf.get("nodes", []):
        if node.get("type") != node_type:
            continue
        desc_idx = widget_descriptor_indexes(node)
        names = widget_names(node)
        if sorted(names) != sorted(new_name_order):
            raise ValueError(
                "node %s: proposed order is not a permutation of the saved "
                "widgets.\n  missing: %r\n  unexpected: %r"
                % (node["id"],
                   sorted(set(names) - set(new_name_order)),
                   sorted(set(new_name_order) - set(names))))
        by_name_desc = {n: node["inputs"][i] for n, i in zip(names, desc_idx)}
        wv = node.get("widgets_values") or []
        if len(wv) != len(names):
            # REFUSE rather than backfill. An earlier cut wrote None for a
            # missing value, which turns a graph that is merely inconsistent
            # into one that is confidently wrong -- and None is a value
            # ComfyUI will happily hand to a widget.
            raise ValueError(
                "node %s: %d widget descriptors but %d saved values. Fix the "
                "graph before reordering it; a reorder cannot invent the "
                "missing value." % (node["id"], len(names), len(wv)))
        by_name_val = {n: wv[k] for k, n in enumerate(names)}
        for slot, nm in zip(desc_idx, new_name_order):
            node["inputs"][slot] = by_name_desc[nm]
        node["widgets_values"] = [by_name_val[n] for n in new_name_order]
        touched.append({"node_id": node["id"], "from": names,
                        "to": list(new_name_order)})
    # Part 3, run here rather than left to the caller. Identity-based, so it is
    # safe even when nothing moved and impossible to double-apply.
    repairs = repair_dst_slots(wf)
    return touched, repairs


def rename_widget(wf, node_type, old_name, new_name):
    """Rename a widget in place -- descriptor name, localized_name and the
    widget.name backref. Position, values and links are all untouched."""
    touched = []
    for node in wf.get("nodes", []):
        if node.get("type") != node_type:
            continue
        for inp in node.get("inputs") or []:
            w = inp.get("widget") or {}
            if w.get("name") != old_name:
                continue
            w["name"] = new_name
            if inp.get("name") == old_name:
                inp["name"] = new_name
            if inp.get("localized_name") == old_name:
                inp["localized_name"] = new_name
            touched.append({"node_id": node["id"], "from": old_name,
                            "to": new_name})
    return touched


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, wf, compact):
    """Write back in the file's OWN shape: the canonical is pretty-printed,
    the generated variants are single-line."""
    text = (json.dumps(wf, separators=(",", ":"), ensure_ascii=False) if compact
            else json.dumps(wf, indent=2, ensure_ascii=False) + "\n")
    Path(path).write_text(text, encoding="utf-8", newline="\n")


def verify(wf, label):
    """Re-assert the property the dedicated backstop checks, plus count parity."""
    problems = []
    links_by_id = {row[0]: row for row in wf.get("links", [])}
    nodes_by_id = {n["id"]: n for n in wf.get("nodes", [])}
    for node in wf.get("nodes", []):
        wv = node.get("widgets_values")
        if isinstance(wv, list):
            n_desc = len(widget_descriptor_indexes(node))
            if n_desc != len(wv):
                problems.append(
                    "%s: node %s has %d widget descriptors but %d values"
                    % (label, node["id"], n_desc, len(wv)))
        for idx, inp in enumerate(node.get("inputs") or []):
            lid = inp.get("link")
            if lid is None:
                continue
            row = links_by_id.get(lid)
            if row is None:
                problems.append("%s: node %s input[%d] -> link %s has no row"
                                % (label, node["id"], idx, lid))
                continue
            if row[3] != node["id"] or row[4] != idx:
                problems.append(
                    "%s: link %s says (node %s, slot %s) but sits at "
                    "(node %s, slot %d)"
                    % (label, lid, row[3], row[4], node["id"], idx))
            tgt = nodes_by_id.get(row[3])
            if tgt is not None and row[4] >= len(tgt.get("inputs") or []):
                problems.append("%s: link %s dst_slot %s out of range on node %s"
                                % (label, lid, row[4], row[3]))
    return problems
