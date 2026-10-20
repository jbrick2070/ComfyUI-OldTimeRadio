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
    apply_choices(graph, schemas, ids, {"bank": "my_story", "acts": 2, "cast_size": "4",
                                        "my_plot": "A lighthouse goes quiet."})
    inputs = graph["7"]["inputs"]
    assert (inputs["source_bank"], inputs["act_count"], inputs["num_characters"]) == ("my_story", "2", 4)
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
    """Against the default shipped workflow: each type the form names is there exactly once."""
    try:
        path = find_workflow()
    except FileNotFoundError:
        pytest.skip("the default workflow is not reachable from here")
    ids = locate_nodes(json.loads(path.read_text(encoding="utf-8")))
    assert set(ids) == {f.node_type for f in FIELDS} | {"OTR_MasterAudioMux"}
    assert FIELDS_BY_KEY["bank"].input == "source_bank"


def test_a_bank_the_pack_cannot_run_is_not_offered_or_accepted(schemas, workflow):
    banks = schemas["OTR_LedgerScriptWriter"]["input"]["required"]["source_bank"]
    banks[0].append("custom_source_bank")
    rows = {r["key"]: r for r in describe_form(schemas)}
    assert "custom_source_bank" not in rows["bank"]["choices"]
    graph, ids = _graph(workflow, schemas)
    with pytest.raises(ValueError, match="Story bank"):
        apply_choices(graph, schemas, ids, {"bank": "custom_source_bank"})


def test_my_story_boxes_with_another_hand_picked_bank_are_refused_before_submitting(schemas, workflow):
    graph = workflow_to_api_prompt(workflow, schemas)
    ids = locate_nodes(workflow)
    with pytest.raises(ValueError, match="my_story"):
        apply_choices(graph, schemas, ids, {"bank": "shakespeare", "my_plot": "A storm cuts the island off."})


def test_my_story_boxes_go_with_my_story_or_a_roll(schemas, workflow):
    for bank in ("my_story", "roll (any eligible bank)"):
        graph = workflow_to_api_prompt(workflow, schemas)
        ids = locate_nodes(workflow)
        apply_choices(graph, schemas, ids, {"bank": bank, "my_characters": "Ada (she), the keeper."})
        assert graph[ids["OTR_LedgerScriptWriter"]]["inputs"]["story_characters"] == "Ada (she), the keeper."


def test_the_author_box_is_the_credit_not_a_style():
    # The pack's story_author is who the story is BY: the announcer says it and
    # the credits print it. It was once labelled "In the style of".
    f = FIELDS_BY_KEY["my_author"]
    assert f.input == "story_author" and f.label == "Written by"


def test_my_story_with_a_classic_named_is_refused_before_submitting(schemas, workflow):
    graph = workflow_to_api_prompt(workflow, schemas)
    ids = locate_nodes(workflow)
    with pytest.raises(ValueError, match="Source to adapt"):
        apply_choices(graph, schemas, ids, {"bank": "my_story", "source_ref": "Hamlet",
                                            "my_plot": "A storm cuts the island off."})


def test_a_classic_with_the_roll_is_refused_unless_the_pool_is_one_bank(schemas, workflow):
    roll = "roll (any eligible bank)"
    graph = workflow_to_api_prompt(workflow, schemas)
    ids = locate_nodes(workflow)
    with pytest.raises(ValueError, match="roll"):
        apply_choices(graph, schemas, ids, {"bank": roll, "source_ref": "Hamlet"})
    graph = workflow_to_api_prompt(workflow, schemas)
    graph[ids["OTR_LedgerScriptWriter"]]["inputs"]["bank_roll_pool"] = "shakespeare"
    apply_choices(graph, schemas, ids, {"bank": roll, "source_ref": "Hamlet"})   # one name is a pick


def test_a_repeated_or_respelled_bank_in_the_pool_is_still_one_pick(schemas, workflow):
    graph = workflow_to_api_prompt(workflow, schemas)
    ids = locate_nodes(workflow)
    graph[ids["OTR_LedgerScriptWriter"]]["inputs"]["bank_roll_pool"] = "Public-Domain, public__domain, _public_domain"
    apply_choices(graph, schemas, ids, {"bank": "roll (any eligible bank)", "source_ref": "Moby Dick"})


def test_the_still_picker_sets_all_three_still_dropdowns(schemas, workflow):
    graph = workflow_to_api_prompt(workflow, schemas)
    ids = locate_nodes(workflow)
    apply_choices(graph, schemas, ids, {"stills": "moss_image"})
    director = graph[ids["OTR_VideoDirector"]]["inputs"]
    assert [director[k] for k in ("character_image_model", "announcer_image_model",
                                  "music_image_model")] == ["moss_image"] * 3


def test_the_still_picker_offers_open_weights_only(schemas, workflow):
    graph = workflow_to_api_prompt(workflow, schemas)
    rows = {r["key"]: r for r in describe_form(schemas, graph, locate_nodes(workflow))}
    assert rows["stills"]["choices"] == ["flux3_klein", "moss_image", "z_image_turbo"]
    with pytest.raises(ValueError):
        apply_choices(graph, schemas, locate_nodes(workflow), {"stills": "cloud_flux_pro"})


def test_a_saved_cloud_still_engine_starts_the_picker_on_a_listed_one(schemas, workflow):
    for node in workflow["nodes"]:
        if node["type"] == "OTR_VideoDirector":
            node["widgets_values"] = ["cloud_flux_pro"] * 3
    graph = workflow_to_api_prompt(workflow, schemas)
    rows = {r["key"]: r for r in describe_form(schemas, graph, locate_nodes(workflow))}
    assert rows["stills"]["default"] in rows["stills"]["choices"]
