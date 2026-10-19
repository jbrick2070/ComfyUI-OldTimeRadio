"""The form: what is offered, what lands on the graph, and the real workflow it targets."""
from __future__ import annotations

import json
import pathlib

import pytest

from radio_play_studio.config import find_workflow
from radio_play_studio.form import (
    FIELDS, FIELDS_BY_KEY, apply_choices, describe_form, locate_nodes,
)
from radio_play_studio.graph import GraphError, workflow_to_api_prompt

REPO = pathlib.Path(__file__).resolve().parents[3]


def _graph(workflow, schemas):
    return workflow_to_api_prompt(workflow, schemas), locate_nodes(workflow)


def test_the_form_offers_the_servers_live_choices(schemas):
    rows = {r["key"]: r for r in describe_form(schemas)}
    banks = schemas["OTR_LedgerScriptWriter"]["input"]["required"]["source_bank"][0]
    assert rows["bank"]["choices"] == banks
    assert rows["cast_size"]["min"] == 1 and rows["cast_size"]["max"] == 6
    assert [r["key"] for r in describe_form(schemas)] == [f.key for f in FIELDS]


def test_a_row_starts_on_the_workflows_own_value(schemas, workflow):
    workflow["nodes"][0]["widgets_values"][0] = "shakespeare"
    graph, ids = _graph(workflow, schemas)
    rows = {r["key"]: r for r in describe_form(schemas, graph, ids)}
    assert rows["bank"]["default"] == "shakespeare"
    assert {r["key"]: r for r in describe_form(schemas)}["bank"]["default"] == "roll (any eligible bank)"


def test_choices_land_on_the_writer_node_and_nothing_else_moves(schemas, workflow):
    graph, ids = _graph(workflow, schemas)
    before = json.loads(json.dumps(graph))
    apply_choices(graph, schemas, ids, {"bank": "shakespeare", "acts": 2, "cast_size": "4",
                                        "my_plot": "A lighthouse goes quiet."})
    inputs = graph["7"]["inputs"]
    assert (inputs["source_bank"], inputs["act_count"], inputs["num_characters"]) == ("shakespeare", "2", 4)
    assert inputs["story_plot"] == "A lighthouse goes quiet."
    inputs.update({k: before["7"]["inputs"][k] for k in ("source_bank", "act_count", "num_characters", "story_plot")})
    assert graph == before


def test_an_empty_box_means_empty_text(schemas, workflow):
    graph, ids = _graph(workflow, schemas)
    apply_choices(graph, schemas, ids, {"source_ref": None})
    assert graph["7"]["inputs"]["source_ref"] == ""


def test_an_unknown_field_is_refused(schemas, workflow):
    graph, ids = _graph(workflow, schemas)
    with pytest.raises(ValueError, match="unknown field"):
        apply_choices(graph, schemas, ids, {"vram": 8})


def test_a_value_the_server_would_refuse_is_refused_here(schemas, workflow):
    graph, ids = _graph(workflow, schemas)
    with pytest.raises(ValueError, match="Story bank"):
        apply_choices(graph, schemas, ids, {"bank": "not a bank"})


def test_a_field_wired_to_another_node_cannot_be_set(schemas, workflow):
    graph, ids = _graph(workflow, schemas)
    graph["7"]["inputs"]["source_bank"] = ["3", 0]
    with pytest.raises(GraphError, match="wired to another node"):
        apply_choices(graph, schemas, ids, {"bank": "original"})


def test_every_field_is_a_row_in_the_packs_own_app_form():
    """The studio may not offer a control the pack's app view does not."""
    app_mode = REPO / "config" / "app_mode.json"
    if not app_mode.is_file():
        pytest.skip("not running from a checkout of the pack")
    pack_form = {(row[0], row[1]) for row in json.loads(app_mode.read_text(encoding="utf-8"))["form"]}
    assert {(f.node_type, f.input) for f in FIELDS} <= pack_form


def test_the_real_workflow_has_each_node_the_form_addresses():
    """Against the shipped otr_app.json: each type the form names is there exactly once."""
    try:
        path = find_workflow()
    except FileNotFoundError:
        pytest.skip("otr_app.json is not reachable from here")
    ids = locate_nodes(json.loads(path.read_text(encoding="utf-8")))
    assert set(ids) == {f.node_type for f in FIELDS} | {"OTR_MasterAudioMux"}
    assert FIELDS_BY_KEY["bank"].input == "source_bank"
