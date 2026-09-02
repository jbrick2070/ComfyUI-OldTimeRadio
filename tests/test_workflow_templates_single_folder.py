"""The template gallery serves ONE folder per pack, and the shipped graph set is
exactly what the operator ruled.

ComfyUI's app/custom_node_manager.py registers a static mount at
``/api/workflow_templates/<pack>`` for EVERY folder named example_workflows,
example, examples, workflow or workflows, and the first mount registered wins
for the whole prefix. With both ``example_workflows/`` and ``workflows/``
present (2026-08-29 to 2026-09-01) the gallery listed ``otr_canonical`` and
``otr_story_only`` from ``workflows/`` but served them from the
``example_workflows/`` mount, so clicking either was a silent 404. The repo's
own docs/2026-08-23-workflow-discoverability-PROBLEM.md had warned about
exactly this. This test keeps the pack to a single template folder.

Operator ruling 2026-09-02: ONE JSON for now -- ``otr_canonical`` (kokoro on both
voice slots) -- plus the script-only graph. The 4060 floor template that shipped
2026-08-29 to 2026-09-02 is gone from the gallery; per-machine saved dropdowns
live in ``workflows/variants/`` (generated, never hand-edited), and a 4060
dropdown-friendly JSON is saved only after that testing is done.
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


def test_gallery_lists_exactly_the_ruled_graphs():
    listed = sorted(p.stem for p in (REPO / "workflows").glob("*.json"))
    assert listed == ["otr_canonical", "otr_story_only"], listed


def test_canonical_ships_kokoro_on_both_voice_slots():
    """Operator ruling 2026-09-01/02: the one shipped graph voices announcer AND
    characters on kokoro from the preset bank, so a fresh install needs no
    reference WAV, sidecar or key."""
    data = json.loads((REPO / "workflows" / "otr_canonical.json").read_text(encoding="utf-8"))
    by_id = {n["id"]: n for n in data["nodes"]}
    assert by_id[80]["type"] == "OTR_CastLock"
    assert by_id[80]["widgets_values"][0] == "kokoro_builtin"
    assert by_id[80]["widgets_values"][3] == "kokoro" and by_id[80]["widgets_values"][4] == "kokoro"
    assert by_id[81]["widgets_values"] == ["kokoro"] and by_id[82]["widgets_values"] == ["kokoro"]
