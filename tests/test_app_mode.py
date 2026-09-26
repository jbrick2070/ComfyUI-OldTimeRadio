# -*- coding: utf-8 -*-
"""Every generated workflow carries a ComfyUI app form (plan row 0e).

ComfyUI's app view shows a workflow as a form: `extra.linearData.inputs` is the
form top to bottom, `outputs` names the node whose result the app pane shows,
and `extra.linearMode` opens the workflow straight into it. ONE FORM, in the
operator's order (2026-09-26, after seeing the story-only form live: "story,
models, My Story at the bottom"). The per-machine workflows open on the GRAPH
(his call) and reach the form with the App button; workflows/otr_app.json opens
as the app. The canonical carries neither; it is the workflow he edits on the
canvas.

The lists are hand-ordered in config/app_mode.json; `build_variants.py`
resolves them against each file's own node ids. These tests read the SHIPPED
files, so a regenerated workflow that lost its form fails here by name.

Headless. No ComfyUI server, no model, no GPU.
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))

import build_variants as bv  # noqa: E402
from nodes import _otr_workflow_apply as wa  # noqa: E402
from tests._support.shipped_graphs import APP, CANONICAL, shipped_graphs, variant_paths  # noqa: E402

CONFIG = json.loads((REPO / "config" / "app_mode.json").read_text(encoding="utf-8"))

#: Matrix keys the per-machine workflow tunes for its own card. The form never
#: shows them: they are not choices, they are what makes the card fit.
MACHINE_TUNING = {
    "llm.device", "llm.quant_policy", "llm.vram_ceiling_gb", "llm.attn_impl",
    "audio.voice_device", "image.dtype_policy", "video.dtype_policy",
    "video.device_policy", "video.max_render_frames",
}


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _pairs(workflow):
    nodes = {n["id"]: n for n in workflow["nodes"]}
    return [(nodes[row[0]]["type"], row[1])
            for row in workflow["extra"]["linearData"]["inputs"]]


def test_the_canonical_is_not_an_app():
    extra = _load(CANONICAL).get("extra") or {}
    assert "linearMode" not in extra and "linearData" not in extra


@pytest.mark.parametrize("path", variant_paths(), ids=lambda p: p.stem)
def test_every_card_opens_on_the_graph_with_the_full_form(path):
    wf = _load(path)
    assert wf["extra"]["linearMode"] is False
    assert _pairs(wf) == [tuple(e[:2]) for e in CONFIG["form"]]


def test_otr_app_opens_as_the_app_with_the_same_form():
    wf = _load(APP)
    assert wf["extra"]["linearMode"] is True
    assert _pairs(wf) == [tuple(e[:2]) for e in CONFIG["form"]]


def test_the_form_reads_story_then_models_then_my_story():
    """His order, stated as a rule rather than a copy of the list: every story
    row comes before every model picker, and the My Story rows come last
    (Space saver closes the form)."""
    rows = [tuple(e[:2]) for e in CONFIG["form"]]
    names = [w for _t, w in rows]
    assert names[-1] == "asset_cleanup"
    my_story = ["story_characters", "story_plot", "story_setting",
                "story_author", "music_style"]
    assert names[-6:-1] == my_story
    story_end = names.index("custom_premise")
    first_model = names.index("announcer_video_model")
    assert story_end < first_model
    for w in ("episode_language", "act_count", "source_bank", "episode_title"):
        assert names.index(w) < first_model, w


@pytest.mark.parametrize("path", [APP] + variant_paths(), ids=lambda p: p.stem)
def test_every_form_row_is_a_drawable_widget_and_the_output_is_the_mux(path):
    wf = _load(path)
    nodes = {n["id"]: n for n in wf["nodes"]}
    for row in wf["extra"]["linearData"]["inputs"]:
        slots = [s for s in nodes[row[0]].get("inputs") or []
                 if (s.get("widget") or {}).get("name") == row[1]]
        assert len(slots) == 1 and slots[0].get("link") is None, row
    (out,) = wf["extra"]["linearData"]["outputs"]
    assert nodes[out]["type"] == CONFIG["output_node_type"]


def test_every_shipped_workflow_has_its_own_id():
    ids = [_load(p)["id"] for p in shipped_graphs()]
    assert len(ids) == len(set(ids)), ids
    assert _load(CANONICAL)["id"] == "09a7142b-5ada-4855-bfcf-041a5e65c555"


def test_the_form_offers_every_matrix_choice_and_no_machine_tuning():
    """A new matrix knob a person chooses cannot be missing from the app, and
    a knob that only makes a card fit cannot leak into it."""
    managed = wa.load_widget_mapping()["managed"]
    matrix = json.loads((REPO / "config" / "workflow_matrix.json").read_text(encoding="utf-8"))
    keys = set()
    for row in matrix["rows"]:
        keys.update(row.get("deltas", {}))
    shown = {tuple(e[:2]) for e in CONFIG["form"]}
    for key in sorted(keys):
        targets = {tuple(t) for t in managed.get(key, {}).get("targets", [])}
        if not targets:
            continue
        if key in MACHINE_TUNING or key.startswith(("render.", "seed_policy.")):
            assert not (targets & shown), key
        else:
            assert targets & shown, key


def test_notes_are_app_only_descriptions_that_exist():
    for list_key in ("form",):
        for entry in CONFIG[list_key]:
            if len(entry) > 2:
                assert isinstance(CONFIG[entry[2]], str) and CONFIG[entry[2]].strip()


def test_a_missing_widget_is_refused_not_dropped():
    wf = _load(CANONICAL)
    bad = dict(CONFIG, form=[["OTR_LedgerScriptWriter", "no_such_widget"]])
    with pytest.raises(bv.EmitRefused, match="no widget 'no_such_widget'"):
        bv.app_linear_data(wf, "form", bad)


def test_a_row_listed_twice_is_refused():
    wf = _load(CANONICAL)
    row = ["OTR_LedgerScriptWriter", "act_count"]
    with pytest.raises(bv.EmitRefused, match="listed twice"):
        bv.app_linear_data(wf, "form", dict(CONFIG, form=[row, row]))


def test_a_linked_widget_is_refused():
    """A widget converted to a linked input cannot be drawn in the form."""
    wf = _load(CANONICAL)
    writer = next(n for n in wf["nodes"] if n["type"] == "OTR_LedgerScriptWriter")
    slot = next(s for s in writer["inputs"]
                if (s.get("widget") or {}).get("name") == "act_count")
    slot["link"] = 99999
    with pytest.raises(bv.EmitRefused, match="linked input"):
        bv.app_linear_data(wf, "form", CONFIG)


def test_a_node_type_that_is_not_unique_is_refused():
    wf = _load(CANONICAL)
    writer = next(n for n in wf["nodes"] if n["type"] == "OTR_LedgerScriptWriter")
    wf["nodes"].append(dict(writer, id=99998))
    with pytest.raises(bv.EmitRefused, match="expected ONE OTR_LedgerScriptWriter"):
        bv.app_linear_data(wf, "form", CONFIG)


@pytest.mark.parametrize("path", [APP] + variant_paths(), ids=lambda p: p.stem)
def test_every_form_row_carries_its_plain_english_label(path):
    wf = _load(path)
    nodes = {n["id"]: n for n in wf["nodes"]}
    for node_id, widget, *_ in wf["extra"]["linearData"]["inputs"]:
        node = nodes[node_id]
        slot = next(s for s in node["inputs"]
                    if (s.get("widget") or {}).get("name") == widget)
        assert slot.get("label") == CONFIG["labels"][f"{node['type']}.{widget}"]


def test_the_canonical_keeps_its_raw_widget_names():
    """The operator's canvas: labels live on the generated forms only."""
    for node in _load(CANONICAL)["nodes"]:
        for slot in node.get("inputs") or []:
            if slot.get("widget"):
                assert "label" not in slot, (node["type"], slot["widget"]["name"])


def test_a_row_without_a_label_is_refused():
    wf = _load(CANONICAL)
    labels = dict(CONFIG["labels"])
    labels.pop("OTR_LedgerScriptWriter.act_count")
    bad = dict(CONFIG, labels=labels)
    with pytest.raises(bv.EmitRefused, match="has no label"):
        bv.app_linear_data(wf, "form", bad)


NOTE_KEYS = ("premise_note", "title_note", "source_ref_note", "pickers_note")


def test_every_note_fits_the_one_line_the_form_draws():
    """The form draws a note on one truncated line (a fixed-height row with
    `truncate`); 30 characters is what showed in full at the pane's width."""
    for key in NOTE_KEYS:
        assert 0 < len(CONFIG[key]) <= 30, (key, CONFIG[key])


@pytest.mark.parametrize("path", [APP] + variant_paths(), ids=lambda p: p.stem)
def test_every_configured_note_is_on_its_shipped_row(path):
    wf = _load(path)
    nodes = {n["id"]: n for n in wf["nodes"]}
    shipped = {(nodes[r[0]]["type"], r[1]): r[2] if len(r) > 2 else None
               for r in wf["extra"]["linearData"]["inputs"]}
    form = "form"
    for entry in CONFIG[form]:
        want = {"description": CONFIG[entry[2]]} if len(entry) > 2 else None
        assert shipped[(entry[0], entry[1])] == want, entry


def test_the_notes_reach_the_form_as_descriptions():
    """ComfyUI 1.52.7 draws `description` under the widget
    (InputWidgetConfig {height?, description?}; AppModeWidgetList reads
    config.description), so the note must be in the SHIPPED row."""
    rows = {r[1]: r for r in _load(APP)["extra"]["linearData"]["inputs"]}
    assert rows["source_ref"][2] == {"description": CONFIG["source_ref_note"]}
    assert rows["custom_premise"][2] == {"description": CONFIG["premise_note"]}
