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

Operator ruling 2026-09-02 was ONE JSON in the gallery -- ``otr_canonical``
(kokoro on both voice slots) -- with the per-machine saved dropdowns in
``workflows/variants/``. REVERSED 2026-09-25 (operator: "we can't store the
variants in a subfolder"): ComfyUI's gallery globs one level, so those graphs
shipped and were never listed. They now sit beside the canonical and the gallery
lists all of them. Still ONE template folder -- the 404 above came from a second
template-named folder, never from how many graphs one folder holds.
"""
from __future__ import annotations

import json
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[1]
TEMPLATE_FOLDER_NAMES = ("example_workflows", "example", "examples", "workflow", "workflows")


def test_exactly_one_template_folder_exists():
    present = [name for name in TEMPLATE_FOLDER_NAMES if (REPO / name).is_dir()]
    assert present == ["workflows"], (
        "ComfyUI mounts every one of these folders at the same template URL and "
        "the first wins; keep a single folder: found %r" % (present,))


def test_gallery_lists_exactly_the_ruled_graphs():
    """The canonical plus exactly the rows the workflow matrix ships -- no
    hand-authored stray (it would list in the menu with no row behind it), and
    no shipping row whose graph is missing (the menu would be one short)."""
    from nodes._otr_shared.capability_profiles import shipping_ids
    listed = sorted(p.stem for p in (REPO / "workflows").glob("*.json"))
    expected = sorted(["otr_canonical"] + [
        pid if pid.startswith("otr_") else "otr_" + pid
        for pid in shipping_ids()])
    assert listed == expected, (
        "gallery drift -- extra: %r, missing: %r"
        % (sorted(set(listed) - set(expected)),
           sorted(set(expected) - set(listed))))
    assert not (REPO / "workflows" / "variants").exists(), (
        "workflows/variants/ is back; a graph there never reaches the gallery")


def test_canonical_ships_kokoro_on_both_voice_slots():
    """Operator ruling 2026-09-01/02: the one shipped graph voices announcer AND
    characters on kokoro from the preset bank, so a fresh install needs no
    reference WAV, sidecar or key."""
    data = json.loads((REPO / "workflows" / "otr_canonical.json").read_text(encoding="utf-8"))
    by_id = {n["id"]: n for n in data["nodes"]}
    assert by_id[80]["type"] == "OTR_CastLock"
    assert by_id[80]["widgets_values"] == [
        "auto_registry", True, "kokoro", "kokoro", "default"]
    assert by_id[81]["widgets_values"] == [] and by_id[82]["widgets_values"] == []


def test_the_boot_message_names_only_templates_that_ship():
    """The FIRST line a new install prints must not name a missing template.

    It did. From 2026-09-02, when the operator's ruling dropped
    ``otr_4060_floor`` from the gallery, until 2026-09-05, the boot banner told
    every first-time user to open
    ``Browse Templates > EXTENSIONS > comfyui-old-time-radio > otr_4060_floor``
    -- a template that no longer shipped. Verified against the PUBLISHED
    alpha.22 bundle, not the repo, because ``.comfyignore`` decides what ships:
    ``workflows/`` carried exactly ``otr_canonical.json`` and
    ``otr_story_only.json``.

    ``otr_4060_floor`` is still a valid PROFILE id for provisioning and the
    headless runner, which is precisely why the stale name looked plausible and
    survived three days. This test reads the banner and requires every
    template-shaped name in it to exist as a shipped graph.
    """
    source = (REPO / "__init__.py").read_text(encoding="utf-8")
    marker = "[OldTimeRadio] Load the show:"
    assert marker in source, "the boot banner was renamed; re-pin this test"
    start = source.index(marker)
    banner = source[start:source.index(")", source.index("print(", start)) + 1]

    shipped = {p.stem for p in (REPO / "workflows").glob("*.json")}
    assert shipped, "no shipped templates found"

    named = set(re.findall(r"\botr_[a-z0-9_]+\b", banner))
    # A path reference like workflows/otr_canonical.json is fine either way --
    # it resolves through the same set.
    missing = sorted(n for n in named if n not in shipped)
    assert not missing, (
        "the boot banner names %r, which workflows/ does not ship (it ships "
        "%r). A first boot must never point at a template that is not there."
        % (missing, sorted(shipped)))
