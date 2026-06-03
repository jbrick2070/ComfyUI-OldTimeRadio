"""S14.3 / BUG-LOCAL-293 -- widget-vector contract check in OTR_WorkflowValidator.

Unit-tests the serialized-slot math (forceInput dropped, seed companion added,
sockets excluded, COMBO counted) and runs the drift scan against the canonical
workflow with a partial NCM. Also documents the live node-14 MusicGenTheme
forceInput slot-count question (BUG-281).
"""
from __future__ import annotations

import json
import pathlib

from nodes._otr_workflow_validator import (
    _expected_slot_count,
    widget_vector_drift,
)
from nodes.musicgen_theme import MusicGenTheme
from nodes.batch_audiogen_generator import BatchAudioGenGenerator

REPO = pathlib.Path(__file__).resolve().parent.parent
CANONICAL = REPO / "workflows" / "otr_scifi_16gb_full.json"
WF = json.loads(CANONICAL.read_text(encoding="utf-8"))


def _node(wf: dict, ntype: str):
    for n in wf.get("nodes") or []:
        if n.get("type") == ntype:
            return n
    return None


def test_expected_slot_count_rules():
    it = {
        "required": {
            "script_json": ("STRING", {"forceInput": True}),  # dropped
            "episode_seed": ("STRING", {}),                    # 1
            "seed": ("INT", {}),                               # 2 (+companion -> 3)
            "model": ("MODEL", {}),                            # socket, excluded
            "mode": (["a", "b"], {}),                          # COMBO -> 4
        }
    }
    assert _expected_slot_count(it) == 4


def _manifest_vector(node_type: str) -> list:
    """The FROZEN delegation widget vector for a legacy engine, from
    config/legacy_invocation_manifest.json. Wave 2b replaced the legacy node
    INSTANCES in full.json with the v2 audio lane, which delegates using these
    vectors -- so the drift guard reads the manifest, not the workflow JSON."""
    manifest = json.loads(
        (REPO / "config" / "legacy_invocation_manifest.json")
        .read_text(encoding="utf-8")
    )
    return [w["default"] for w in manifest["nodes"][node_type]["widgets"]]


def test_audiogen_no_drift():
    """The AudioGen frozen delegation vector must match the class INPUT_TYPES
    slot count (the drift that would break delegation)."""
    wv = _manifest_vector("OTR_BatchAudioGenGenerator")
    expected = _expected_slot_count(BatchAudioGenGenerator.INPUT_TYPES())
    actual = len(wv)
    print(f"\n[AUDIOGEN manifest] expected={expected} actual={actual}")
    assert actual == expected, (
        f"AudioGen manifest widget-vector drift: {actual} vs expected {expected}"
    )


def test_musicgen_manifest_documents_bug281():
    """Documents the forceInput slot-count question on the MusicGen frozen
    delegation vector (BUG-281). Non-asserting on equality."""
    wv = _manifest_vector("OTR_MusicGenTheme")
    expected = _expected_slot_count(MusicGenTheme.INPUT_TYPES())
    actual = len(wv)
    print(f"\n[MUSICGEN manifest] expected={expected} actual={actual} "
          f"(drift={'YES' if expected != actual else 'no'})")
    assert isinstance(expected, int) and isinstance(actual, int)


def test_drift_scan_runs_on_canonical():
    ncm = {
        "OTR_MusicGenTheme": MusicGenTheme,
        "OTR_BatchAudioGenGenerator": BatchAudioGenGenerator,
    }
    findings = widget_vector_drift(WF, ncm)
    print(f"\n[DRIFT SCAN] {findings}")
    assert isinstance(findings, list)
