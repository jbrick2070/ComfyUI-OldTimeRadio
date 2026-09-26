"""The Start-here note (plan 0d, 2026-09-26): a canvas note rides in every
shipped workflow, and nothing that turns a workflow into server work may send
it. ComfyUI's /prompt refuses an unknown class_type, so ONE emitted note would
fail every headless run, soak and pod leg while the browser -- whose frontend
treats a note as virtual -- kept working.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from nodes import _otr_workflow_apply as wa
from scripts import otr_api
from tests._support.shipped_graphs import shipped_graphs

REPO = Path(__file__).resolve().parents[1]
CANONICAL = REPO / "workflows" / "otr_canonical.json"


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _notes(wf):
    return [n for n in wf["nodes"] if n["type"] in wa.NOTE_NODE_TYPES]


def test_the_two_converters_skip_the_same_types():
    assert otr_api.NOTE_NODE_TYPES == wa.NOTE_NODE_TYPES
    # Notes only: Reroute and PrimitiveNode carry links / baked values.
    assert wa.NOTE_NODE_TYPES == frozenset({"Note", "MarkdownNote"})


@pytest.mark.parametrize("convert", [wa.workflow_to_api_prompt,
                                     otr_api.workflow_to_api_prompt],
                         ids=["nodes", "scripts"])
def test_no_converter_emits_the_note(convert):
    wf = _load(CANONICAL)
    notes = _notes(wf)
    assert notes, "the canonical lost its Start-here note"
    prompt = convert(wf, wa.build_offline_schemas())
    assert not {n["class_type"] for n in prompt.values()} & wa.NOTE_NODE_TYPES
    assert len(prompt) == len(wf["nodes"]) - len(notes)
    for note in notes:
        assert str(note["id"]) not in prompt


@pytest.mark.parametrize("path", shipped_graphs(), ids=lambda p: p.stem)
def test_every_shipped_workflow_carries_the_note_unstamped(path):
    """The note reaches every generated workflow unchanged, and it is never
    labelled as this pack's node: it is ComfyUI core, and the frontend's
    conflict detection reads `cnr_id`."""
    canon = _notes(_load(CANONICAL))
    wf = _load(path)
    notes = _notes(wf)
    assert [n["widgets_values"] for n in notes] == \
        [n["widgets_values"] for n in canon]
    for note in notes:
        assert "cnr_id" not in (note.get("properties") or {})


def test_the_note_sits_in_the_start_group_and_covers_nothing():
    wf = _load(CANONICAL)
    group = next(g for g in wf["groups"] if g["title"] == "SCRIPT / START HERE")
    gx, gy, gw, gh = group["bounding"]
    (note,) = _notes(wf)
    x, y = note["pos"]
    w, h = note["size"]
    assert gx <= x and gy <= y and x + w <= gx + gw and y + h <= gy + gh
    for other in wf["nodes"]:
        if other is note:
            continue
        ox, oy = other["pos"]
        ow, oh = other["size"]
        overlaps = x < ox + ow and ox < x + w and y < oy + oh and oy < y + h
        assert not overlaps, other["id"]


def test_what_the_note_names_exists():
    """A note that names a bank, a widget or a node title the pack does not
    have sends a stranger looking for it. Checked against the live code."""
    from nodes import _otr_rolls, _otr_story_routing

    wf = _load(CANONICAL)
    (note,) = _notes(wf)
    text = note["widgets_values"][0]
    banks = set(_otr_story_routing.list_bank_ids()) | {_otr_rolls.BANK_SENTINEL}
    for name in ("original", "shakespeare", "public_domain", "my_story",
                 _otr_rolls.BANK_SENTINEL):
        assert "`%s`" % name in text and name in banks, name
    writer = next(n for n in wf["nodes"]
                  if n["type"] == "OTR_LedgerScriptWriter")
    widgets = {i["name"] for i in writer["inputs"] if i.get("widget")}
    for bold in re.findall(r"\*\*([a-z_]+)\*\*", text):
        assert bold in widgets, bold
    assert "**%s**" % writer["title"] in text
