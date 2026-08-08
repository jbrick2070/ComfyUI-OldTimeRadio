"""Queue item 8 (2026-08-08): SilentComposite widget positional guard.

BUG-LOCAL-097: widgets_values is POSITIONAL; new widgets MUST APPEND at the
end, and existing positions MUST NOT SHIFT. This test pins the node-84 shape
(5 shipped widgets + 2 upscale widgets = 7 total) and asserts:

* The two new upscale widgets land at positions 5 and 6 (the LAST two).
* The five shipped widgets stay at positions 0..4 with their original values.
* Every widget-backed input in inputs[] has a matching widgets_values entry.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CANONICAL = REPO / "workflows" / "otr_canonical.json"


@pytest.fixture(scope="module")
def canonical() -> dict:
    with open(CANONICAL, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def node84(canonical) -> dict:
    hits = [n for n in canonical.get("nodes", [])
            if n.get("type") == "OTR_SilentComposite"]
    assert len(hits) == 1, f"expected exactly one OTR_SilentComposite; got {len(hits)}"
    return hits[0]


def test_widget_count_is_seven(node84):
    """5 shipped (canvas_w/h, fps, ffmpeg, output_path) + 2 new (upscale_engine,
    upscale_device) = 7. If someone appends a 3rd upscale widget without
    updating this test, positional deserialization drift is exactly what
    BUG-LOCAL-097 guards against."""
    wv = node84.get("widgets_values") or []
    assert len(wv) == 7, f"expected 7 widgets_values on node 84; got {len(wv)}: {wv!r}"


def test_first_five_widgets_unchanged(node84):
    """The 5 shipped widget values must stay at their historical positions;
    a stale saved workflow relies on positional index."""
    wv = node84["widgets_values"]
    assert wv[0] == 1920, f"canvas_w moved: got {wv[0]!r}"
    assert wv[1] == 1080, f"canvas_h moved: got {wv[1]!r}"
    assert wv[2] == 25, f"fps moved: got {wv[2]!r}"
    assert wv[3] == "ffmpeg", f"ffmpeg moved: got {wv[3]!r}"
    assert wv[4] == "", f"output_path moved: got {wv[4]!r}"


def test_upscale_widgets_at_positions_5_and_6(node84):
    """Positional law: new widgets append at the end."""
    wv = node84["widgets_values"]
    assert wv[5] == "off", f"upscale_engine wrong slot/value: got {wv[5]!r}"
    assert wv[6] == "cpu", f"upscale_device wrong slot/value: got {wv[6]!r}"


def test_inputs_list_matches_widget_positions(node84):
    """Every widget-backed input in inputs[] must have a corresponding
    widgets_values entry AT THE SAME POSITIONAL INDEX. This is the
    referential integrity between the two lists that ComfyUI's live
    INPUT_TYPES resolves against."""
    inputs = node84.get("inputs") or []
    widget_inputs = [i for i in inputs if isinstance(i, dict) and i.get("widget")]
    wv = node84.get("widgets_values") or []
    assert len(widget_inputs) == len(wv), (
        f"widget-backed inputs ({len(widget_inputs)}) != widgets_values "
        f"({len(wv)}): inputs={[i.get('name') for i in widget_inputs]!r}, "
        f"widgets_values={wv!r}")
    # Names in the last two positions must match the r4 spec.
    assert widget_inputs[-2]["name"] == "upscale_engine"
    assert widget_inputs[-1]["name"] == "upscale_device"


def test_upscale_engine_input_type_is_combo(node84):
    """The upscale_engine widget is a COMBO (dropdown) whose choices come
    from the live upscale registry roster. In the workflow JSON its input
    type slot is 'COMBO'."""
    inputs = node84.get("inputs") or []
    hit = next((i for i in inputs
                if isinstance(i, dict) and i.get("name") == "upscale_engine"),
               None)
    assert hit is not None, "upscale_engine input missing from node 84 inputs[]"
    assert hit.get("type") == "COMBO", (
        f"upscale_engine input type should be COMBO; got {hit.get('type')!r}")
    assert hit.get("widget", {}).get("name") == "upscale_engine"


def test_upscale_device_input_type_is_string(node84):
    inputs = node84.get("inputs") or []
    hit = next((i for i in inputs
                if isinstance(i, dict) and i.get("name") == "upscale_device"),
               None)
    assert hit is not None, "upscale_device input missing from node 84 inputs[]"
    assert hit.get("type") == "STRING"
    assert hit.get("widget", {}).get("name") == "upscale_device"
