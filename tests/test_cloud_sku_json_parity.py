"""Cloud SKU graphs differ only by length, cast size, and costume dropdowns.

The 2026-09-16 JSON sweep of the five shipping cloud graphs found identical
node/link shape; the only widget/profile drift was act_count, num_characters,
video/writer dropdowns, and validator stamps. This test keeps that contract.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from nodes import _otr_workflow_apply as wa

REPO = Path(__file__).resolve().parents[1]
PROFILES = REPO / "config" / "profiles"

CHEAP = (
    "otr_cloud_low_1act",
    "otr_cloud_low",
    "otr_cloud_low_5act",
)
DELUXE = (
    "otr_cloud_deluxe_3act",
    "otr_cloud_deluxe_audio_in_3act",
)

PROFILE_OK = {
    "id",
    "display_name",
    "features.act_count",
    "features.num_characters",
    "llm.comfy_slot_a_model",
    "role_overrides.announcer_visual",
    "role_overrides.music_visual",
    "role_overrides.character_visual",
    "slot_overrides.video_render_engine",
}

WIDGET_OK = {
    "OTR_LedgerScriptWriter.act_count",
    "OTR_LedgerScriptWriter.num_characters",
    "OTR_LedgerScriptWriter.comfy_slot_a_model",
    "OTR_VideoRenderBatch.engine",
    "OTR_VideoDirector.announcer_video_model",
    "OTR_VideoDirector.music_video_model",
    "OTR_VideoDirector.character_video_model",
    "OTR_WorkflowValidator.profile_id",
    "OTR_WorkflowValidator.master_hash",
    "OTR_WorkflowValidator.workflow_json_path",
}


def _flatten(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else k))
    elif isinstance(obj, list):
        if all(not isinstance(x, (dict, list)) for x in obj):
            out[prefix] = obj
        else:
            for i, v in enumerate(obj):
                out.update(_flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = obj
    return out


def _load_profile(pid):
    return json.loads((PROFILES / f"{pid}.json").read_text(encoding="utf-8"))


def _profile_diffs(a, b):
    fa, fb = _flatten(_load_profile(a)), _flatten(_load_profile(b))
    miss = object()
    return sorted(
        k for k in set(fa) | set(fb)
        if fa.get(k, miss) != fb.get(k, miss)
    )


def _widget_map(wf, schemas):
    out = {}
    for n in wf["nodes"]:
        names = wa.serialized_slot_names(n["type"], schemas)
        wv = n.get("widgets_values") or []
        for i, val in enumerate(wv):
            name = names[i] if i < len(names) else f"EXTRA_{i}"
            out[f"{n['type']}.{name}"] = val
    return out


def _shape(wf):
    nodes = sorted(
        (int(n["id"]), n["type"],
         tuple((i.get("name"), i.get("link")) for i in (n.get("inputs") or [])))
        for n in wf["nodes"]
    )
    links = sorted(tuple(x) if isinstance(x, list) else x
                   for x in (wf.get("links") or []))
    return (wf.get("last_node_id"), wf.get("last_link_id"), nodes, links)


@pytest.fixture(scope="module")
def schemas():
    return wa.build_offline_schemas()


@pytest.mark.parametrize("a,b", [
    (CHEAP[0], CHEAP[1]),
    (CHEAP[1], CHEAP[2]),
    (CHEAP[0], CHEAP[2]),
])
def test_cheap_cloud_profiles_differ_only_by_length_fields(a, b):
    unexpected = [k for k in _profile_diffs(a, b) if k not in PROFILE_OK]
    assert unexpected == []


def test_deluxe_cloud_profiles_differ_only_by_video_costume():
    unexpected = [
        k for k in _profile_diffs(DELUXE[0], DELUXE[1]) if k not in PROFILE_OK
    ]
    assert unexpected == []


@pytest.mark.parametrize("a,b", [
    (CHEAP[0], CHEAP[1]),
    (CHEAP[1], CHEAP[2]),
    (DELUXE[0], DELUXE[1]),
    (CHEAP[1], DELUXE[0]),
])
def test_cloud_variant_graphs_same_shape_and_expected_widget_drift(
        a, b, schemas):
    import sys
    sys.path.insert(0, str(REPO / "scripts"))
    import build_variants as bv

    va, _, _ = bv.build_variant(a, schemas=schemas)
    vb, _, _ = bv.build_variant(b, schemas=schemas)
    assert _shape(va) == _shape(vb)
    ma, mb = _widget_map(va, schemas), _widget_map(vb, schemas)
    miss = object()
    unexpected = sorted(
        k for k in set(ma) | set(mb)
        if ma.get(k, miss) != mb.get(k, miss) and k not in WIDGET_OK
    )
    assert unexpected == []
