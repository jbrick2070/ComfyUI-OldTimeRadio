# -*- coding: utf-8 -*-
"""Every node in every shipped workflow names this pack and its live version.

WHY (0d, 2026-09-25). Measured that day: the canonical had 21 nodes and
exactly one (OTR_WorkflowValidator) carried `properties.cnr_id`; none carried
`ver`. The frontend's getCnrIdFromNode reads `cnr_id` (falling back to
`aux_id`) and it is how "Install Missing Nodes" finds the pack a node came
from -- the official templates carry both on most nodes. A stranger who opens
a shipped workflow on a box without the pack gets a menu-wide "missing nodes"
with no install target.

`ver` is read from pyproject.toml at build time, never a hardcoded copy: a
copy drifts on the next bump (the DEPENDENCIES.md generator lesson). The
canonical is hand-authored, so `build_variants.py --check` carries the same
gate and fails the moment a bump leaves the canonical behind.

`extra.info` said `version: "2.0-alpha"` and nothing in nodes/, scripts/,
tests/ or js/ ever read it. pyproject is the version authority; it is gone.

Headless. No engine, no model, no GPU.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
CANONICAL = REPO / "workflows" / "otr_canonical.json"

for _p in (REPO, REPO / "scripts", REPO / "nodes"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import build_variants as bv  # noqa: E402


def _shipped_paths():
    return [CANONICAL] + list(bv._committed_variant_paths())


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_live_pack_version_is_the_project_table_version():
    """The reader agrees with a direct regex of the [project] table, and
    would not be fooled by a `version =` under some later [tool.*] table."""
    text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    project = re.search(r"^\[project\]\s*$(.*?)(?=^\[|\Z)", text,
                        re.MULTILINE | re.DOTALL).group(1)
    direct = re.search(r'^version\s*=\s*"([^"]+)"', project, re.MULTILINE).group(1)
    assert bv.live_pack_version() == direct
    assert re.fullmatch(r"\d+\.\d+\.\d+([.-][0-9A-Za-z.]+)?", direct), direct


def test_stamp_sets_both_keys_on_every_node_and_is_idempotent():
    wf = {"nodes": [{"id": 1, "type": "A"},
                    {"id": 2, "type": "B", "properties": {"x": 1}}]}
    bv.stamp_pack_identity(wf, "9.9.9")
    bv.stamp_pack_identity(wf, "9.9.9")
    for node in wf["nodes"]:
        assert node["properties"]["cnr_id"] == bv.PACK_CNR_ID
        assert node["properties"]["ver"] == "9.9.9"
    assert wf["nodes"][1]["properties"]["x"] == 1  # existing keys survive


def test_a_canvas_note_is_never_stamped_as_this_pack():
    """A Note / MarkdownNote is a ComfyUI core node; the frontend's conflict
    detection reads `cnr_id`, so it keeps the empty properties it saves with."""
    wf = {"nodes": [{"id": 1, "type": "OTR_X"},
                    {"id": 2, "type": "MarkdownNote", "properties": {}}]}
    bv.stamp_pack_identity(wf, "9.9.9")
    assert wf["nodes"][0]["properties"]["cnr_id"] == bv.PACK_CNR_ID
    assert wf["nodes"][1]["properties"] == {}


@pytest.mark.parametrize("path", _shipped_paths(), ids=lambda p: p.stem)
def test_every_node_in_every_shipped_workflow_carries_the_stamp(path):
    version = bv.live_pack_version()
    wf = _load(path)
    assert wf["nodes"], path.name
    for node in wf["nodes"]:
        if node["type"] in bv.NOTE_NODE_TYPES:
            continue      # ComfyUI core; see the note test above
        props = node.get("properties") or {}
        assert props.get("cnr_id") == bv.PACK_CNR_ID, (path.name, node["id"], node["type"])
        assert props.get("ver") == version, (path.name, node["id"], node["type"], props.get("ver"))


@pytest.mark.parametrize("path", _shipped_paths(), ids=lambda p: p.stem)
def test_extra_info_is_gone(path):
    wf = _load(path)
    assert "info" not in (wf.get("extra") or {}), path.name


def test_check_gate_names_a_missing_stamp(tmp_path):
    """The --check helper actually catches the defect it exists for."""
    good = {"nodes": [{"id": 1, "type": "A",
                       "properties": {"cnr_id": bv.PACK_CNR_ID, "ver": "1.0.0"}}]}
    bad = {"nodes": [{"id": 7, "type": "OTR_X", "properties": {"cnr_id": bv.PACK_CNR_ID}}],
           "extra": {"info": {"version": "2.0-alpha"}}}
    gp = tmp_path / "good.json"
    bp = tmp_path / "bad.json"
    gp.write_text(json.dumps(good), encoding="utf-8")
    bp.write_text(json.dumps(bad), encoding="utf-8")
    assert bv._pack_identity_failures([gp], "1.0.0") == []
    failures = bv._pack_identity_failures([bp], "1.0.0")
    assert len(failures) == 2, failures
    assert "7:OTR_X" in failures[0]
    assert "extra.info" in failures[1]
