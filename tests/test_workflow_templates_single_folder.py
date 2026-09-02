"""The template gallery serves ONE folder per pack, and the floor template must
stay byte-equal to the variant it was copied from.

ComfyUI's app/custom_node_manager.py registers a static mount at
``/api/workflow_templates/<pack>`` for EVERY folder named example_workflows,
example, examples, workflow or workflows, and the first mount registered wins
for the whole prefix. With both ``example_workflows/`` and ``workflows/``
present (2026-08-29 to 2026-09-01) the gallery listed ``otr_canonical`` and
``otr_story_only`` from ``workflows/`` but served them from the
``example_workflows/`` mount, so clicking either was a silent 404. The repo's
own docs/2026-08-23-workflow-discoverability-PROBLEM.md had warned about
exactly this. This test keeps the pack to a single template folder.

The gallery template ``workflows/otr_4060_floor.json`` is a copy of the
generated ``workflows/variants/otr_4060_floor.json`` (scripts/build_variants.py
owns the variant). A regenerated variant that is not copied forward would ship
a stale template, so the two must match in structure and every saved widget.
"""
from __future__ import annotations

import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]
TEMPLATE_FOLDER_NAMES = ("example_workflows", "example", "examples", "workflow", "workflows")


def test_exactly_one_template_folder_exists():
    present = [name for name in TEMPLATE_FOLDER_NAMES if (REPO / name).is_dir()]
    assert present == ["workflows"], (
        "ComfyUI mounts every one of these folders at the same template URL and "
        "the first wins; keep a single folder: found %r" % (present,))


def test_gallery_lists_the_three_shipped_graphs():
    listed = sorted(p.stem for p in (REPO / "workflows").glob("*.json"))
    assert listed == ["otr_4060_floor", "otr_canonical", "otr_story_only"], listed


def _graph_shape(path: pathlib.Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    nodes = {(n["id"], n["type"]): n.get("widgets_values") for n in data["nodes"]}
    return nodes, data["links"]


def test_floor_template_matches_its_generated_variant():
    template = _graph_shape(REPO / "workflows" / "otr_4060_floor.json")
    variant = _graph_shape(REPO / "workflows" / "variants" / "otr_4060_floor.json")
    assert template[0].keys() == variant[0].keys(), "node set differs"
    for key in template[0]:
        assert template[0][key] == variant[0][key], (
            "widgets differ on node %r: regenerate the variant, then copy it "
            "over workflows/otr_4060_floor.json" % (key,))
    assert template[1] == variant[1], "links differ"
