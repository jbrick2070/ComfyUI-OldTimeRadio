"""The workflow -> API graph conversion and the rules for a single input."""
from __future__ import annotations

import pytest

from radio_play_studio.graph import (
    GraphError, coerce_value, describe_input, node_id_for_type, serialized_slot_names,
    workflow_to_api_prompt,
)


def _schemas():
    return {
        "Sampler": {"input": {
            "required": {
                "model": ("MODEL",),
                "seed": ("INT", {"default": 0}),
                "mode": (["fast", "slow"], {"default": "fast"}),
                "prompt": ("STRING", {"forceInput": True}),
                "pools": ("COMBO", {"options": ["a", "b"]}),
            },
        }},
    }


def _node(values, links=()):
    return {"id": 3, "type": "Sampler", "widgets_values": values,
            "inputs": [{"name": n, "link": lid} for n, lid in links]}


def test_a_seed_widget_has_a_hidden_control_slot():
    assert serialized_slot_names("Sampler", _schemas()) == [
        "seed", "seed__control_after_generate", "mode", "pools"]


def test_widget_values_land_on_their_names_and_links_are_kept():
    wf = {"nodes": [_node([5, "fixed", "slow", "b"], links=[("model", 1), ("prompt", 2)])],
          "links": [[1, 10, 0, 3, 0, "MODEL"], [2, 11, 1, 3, 3, "STRING"]]}
    graph = workflow_to_api_prompt(wf, _schemas())
    assert graph["3"]["class_type"] == "Sampler"
    assert graph["3"]["inputs"] == {
        "model": ["10", 0], "prompt": ["11", 1], "seed": 5, "mode": "slow", "pools": "b"}


def test_a_linked_widget_takes_the_link_not_its_saved_value():
    wf = {"nodes": [_node([5, "fixed", "slow", "b"], links=[("mode", 4)])],
          "links": [[4, 20, 0, 3, 2, "STRING"]]}
    inputs = workflow_to_api_prompt(wf, _schemas())["3"]["inputs"]
    assert inputs["mode"] == ["20", 0] and inputs["pools"] == "b"


def test_a_list_value_goes_out_wrapped():
    schemas = {"Pool": {"input": {"required": {"picked": ("COMBO", {"options": ["a", "b"]})}}}}
    wf = {"nodes": [{"id": 1, "type": "Pool", "widgets_values": [["a", "b"]], "inputs": []}], "links": []}
    assert workflow_to_api_prompt(wf, schemas)["1"]["inputs"]["picked"] == {"__value__": ["a", "b"]}


def test_notes_are_never_sent():
    wf = {"nodes": [{"id": 1, "type": "MarkdownNote", "widgets_values": ["x"]}], "links": []}
    assert workflow_to_api_prompt(wf, {}) == {}


def test_a_node_type_the_server_lacks_names_the_pack():
    wf = {"nodes": [{"id": 1, "type": "OTR_Missing", "widgets_values": []}], "links": []}
    with pytest.raises(GraphError, match="Install the Old-Time Radio pack"):
        workflow_to_api_prompt(wf, {})


def test_a_workflow_from_another_pack_version_is_refused():
    with pytest.raises(GraphError, match="different versions"):
        workflow_to_api_prompt({"nodes": [_node([5, "fixed"])], "links": []}, _schemas())


def test_a_drifted_seed_control_is_refused():
    with pytest.raises(GraphError, match="widget order has drifted"):
        workflow_to_api_prompt({"nodes": [_node([5, "slow", "fast", "a"])], "links": []}, _schemas())


def test_node_id_for_type_wants_exactly_one():
    wf = {"nodes": [{"id": 1, "type": "A"}, {"id": 2, "type": "B"}, {"id": 3, "type": "B"}]}
    assert node_id_for_type(wf, "A") == "1"
    with pytest.raises(GraphError, match="found 2"):
        node_id_for_type(wf, "B")
    with pytest.raises(GraphError, match="found 0"):
        node_id_for_type(wf, "C")


def test_a_dropdown_is_read_in_either_shape():
    old = {"N": {"input": {"required": {"x": (["a", "b"], {"default": "a"})}}}}
    new = {"N": {"input": {"required": {"x": ("COMBO", {"options": ["a", "b"], "default": "a"})}}}}
    assert describe_input(old, "N", "x")["choices"] == describe_input(new, "N", "x")["choices"] == ["a", "b"]


def test_describe_input_names_what_is_missing():
    with pytest.raises(GraphError, match="no input named"):
        describe_input(_schemas(), "Sampler", "nope")
    with pytest.raises(GraphError, match="not on the server"):
        describe_input(_schemas(), "Nope", "x")


@pytest.mark.parametrize("info,value,expected", [
    ({"kind": "choice", "choices": ["1", "2"]}, 2, "2"),
    ({"kind": "int", "min": 1, "max": 4}, "3", 3),
    ({"kind": "text"}, "hello", "hello"),
    ({"kind": "bool"}, True, True),
])
def test_coerce_value_accepts_what_the_server_accepts(info, value, expected):
    assert coerce_value(info, value, "x") == expected


@pytest.mark.parametrize("info,value", [
    ({"kind": "choice", "choices": ["1", "2"]}, "3"),
    ({"kind": "int", "min": 1, "max": 4}, 9),
    ({"kind": "int", "min": 1, "max": 4}, True),
    ({"kind": "int"}, "many"),
    ({"kind": "text"}, 5),
    ({"kind": "bool"}, "yes"),
    ({"kind": "int"}, float("inf")),
    ({"kind": "int"}, 2.7),
    ({"kind": "float"}, float("nan")),
])
def test_coerce_value_refuses_what_the_server_would(info, value):
    with pytest.raises(ValueError):
        coerce_value(info, value, "x")
