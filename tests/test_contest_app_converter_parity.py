# -*- coding: utf-8 -*-
"""Radio Play Studio (contest/radio-play-studio) turns the shipped app workflow
into an API graph with its own standard-library converter, so the app runs
without this pack's Python on its path. A copy is only safe if it is pinned.

These tests pin it two ways, against the pack's REAL node definitions:

* the app's graph is byte-identical to the pack's own
  ``nodes._otr_workflow_apply.workflow_to_api_prompt`` for every shipped graph;
* the app's form resolves against the real nodes: each field exists on the
  node it names, with the choices and limits the form will show, and a
  representative choice lands on the graph.

Headless. No ComfyUI server, no model, no GPU. Needs torch to import the four
nodes that import it at module scope; without it the schemas are partial and
the tests skip rather than compare a half-built answer.
"""
from __future__ import annotations

import json
import sys

import pytest

from tests._support.shipped_graphs import APP, REPO, shipped_graphs

sys.path.insert(0, str(REPO / "contest" / "radio-play-studio"))

from nodes import _otr_workflow_apply as wa  # noqa: E402
from radio_play_studio import form as rps_form  # noqa: E402
from radio_play_studio import graph as rps_graph  # noqa: E402

NOTE_TYPES = wa.NOTE_NODE_TYPES


@pytest.fixture(scope="module")
def schemas():
    return wa.build_offline_schemas()


def _workflow(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _complete(workflow, schemas):
    """Skip unless every node type in the graph has a real definition: a node
    that failed to import leaves the offline schemas partial, and the pack's
    converter would pass it through silently where the app's refuses."""
    missing = sorted({n["type"] for n in workflow["nodes"]} - NOTE_TYPES - set(schemas))
    if missing:
        pytest.skip("offline schemas are missing %s (torch not importable here?)" % missing)


def test_the_note_types_are_the_packs():
    assert rps_graph.NOTE_NODE_TYPES == NOTE_TYPES


@pytest.mark.parametrize("path", shipped_graphs(), ids=lambda p: p.stem)
def test_the_app_converter_matches_the_packs_on_every_shipped_graph(path, schemas):
    workflow = _workflow(path)
    _complete(workflow, schemas)
    assert rps_graph.workflow_to_api_prompt(workflow, schemas) == wa.workflow_to_api_prompt(workflow, schemas)


def test_the_app_form_resolves_against_the_real_nodes(schemas):
    workflow = _workflow(APP)
    _complete(workflow, schemas)
    graph = rps_graph.workflow_to_api_prompt(workflow, schemas)
    node_ids = rps_form.locate_nodes(workflow)
    rows = {r["key"]: r for r in rps_form.describe_form(schemas, graph, node_ids)}
    assert list(rows) == [f.key for f in rps_form.FIELDS]
    banks = rows["bank"]["choices"]
    assert "shakespeare" in banks and "my_story" in banks
    assert all(rows[k]["default"] is not None for k in ("bank", "language", "acts", "cast_size", "style"))


def test_a_representative_choice_lands_on_the_real_writer(schemas):
    workflow = _workflow(APP)
    _complete(workflow, schemas)
    graph = rps_graph.workflow_to_api_prompt(workflow, schemas)
    node_ids = rps_form.locate_nodes(workflow)
    rows = {r["key"]: r for r in rps_form.describe_form(schemas, graph, node_ids)}
    choices = {"bank": "my_story", "acts": rows["acts"]["choices"][-1],
               "cast_size": rows["cast_size"]["min"] or 1, "my_plot": "A quiet lighthouse."}
    rps_form.apply_choices(graph, schemas, node_ids, choices)
    inputs = graph[node_ids[rps_form.WRITER]]["inputs"]
    assert inputs["source_bank"] == "my_story" and inputs["story_plot"] == "A quiet lighthouse."
