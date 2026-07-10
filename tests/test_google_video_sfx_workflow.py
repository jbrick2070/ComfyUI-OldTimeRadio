"""Canonical workflow wiring for Google video SFX-bed muxing."""
from __future__ import annotations

import json
import pathlib

from nodes.otr_master_audio_mux import OTRMasterAudioMux

WF = pathlib.Path(__file__).resolve().parent.parent / "workflows" / "otr_canonical.json"


def _workflow():
    return json.loads(WF.read_text(encoding="utf-8"))


def test_master_audio_mux_declares_connector_only_clip_manifest_input():
    it = OTRMasterAudioMux.INPUT_TYPES()
    spec = it["optional"]["clip_manifest_json"]
    assert spec[0] == "STRING"
    assert spec[1]["forceInput"] is True
    # Connector-only SFX input must not add to the saved widget vector.
    widget_backed = [
        name for name, field in it["optional"].items()
        if field[0] in ("INT", "FLOAT", "STRING", "BOOLEAN")
        and not field[1].get("forceInput")
    ]
    assert widget_backed == ["fps", "ffmpeg", "output_path"]


def test_canonical_workflow_wires_clip_manifest_to_master_audio_mux():
    wf = _workflow()
    nodes = {n["id"]: n for n in wf["nodes"]}
    n85 = nodes[85]
    n92 = nodes[92]
    # S5 platform-portability (2026-07-10): OLD pin 278 -> NEW pin 279 (link
    # 279 = OTR_WorkflowValidator.validation_report -> node-1 gate_in; this
    # file's link IDs 85/92-side are unaffected, only the vector's ceiling).
    assert wf["last_link_id"] == 279
    assert [i["name"] for i in n85["inputs"]] == [
        "silent_video_path",
        "master_audio_path",
        "audio_done",
        "declared_credits_tail_s",
        "clip_manifest_json",
        "fps",
        "ffmpeg",
        "output_path",
    ]
    assert [i.get("link") for i in n85["inputs"][:5]] == [274, 263, 249, 276, 278]
    assert n85["widgets_values"] == [25, "ffmpeg", ""]
    assert n92["outputs"][1]["name"] == "clip_manifest_json"
    assert n92["outputs"][1]["links"] == [261, 271, 275, 278]
    links = {l[0]: l for l in wf["links"]}
    assert links[278] == [278, 92, 1, 85, 4, "STRING"]
    assert links[261] == [261, 92, 1, 84, 2, "STRING"]
    assert links[271] == [271, 92, 1, 94, 1, "STRING"]
    assert links[275] == [275, 92, 1, 95, 1, "STRING"]
    assert links[274][3:5] == [85, 0]
    assert links[263][3:5] == [85, 1]
    assert links[249][3:5] == [85, 2]
    assert links[276][3:5] == [85, 3]
