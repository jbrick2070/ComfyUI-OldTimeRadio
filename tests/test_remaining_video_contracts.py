"""Offline contracts for the confirmed 2026-07-23 video qualification bugs."""

from __future__ import annotations

import json
from pathlib import Path

from nodes._otr_shared import capability_profiles as cp


REPO = Path(__file__).resolve().parents[1]


def test_otr_8gb_wan_profile_pins_low_vram_contract():
    profile = cp.load_profile("otr_8gb_wan")
    assert profile["render"]["canvas_w"] == 832
    assert profile["render"]["canvas_h"] == 480
    assert profile["render"]["frame_budget"] == 17
    assert profile["launch"]["env"] == {
        "OTR_VIDEO_LANDSCAPE_CANVAS": "832x480",
        "OTR_WAN_TI2V_MAX_FRAMES": "17",
    }

    variant = json.loads(
        (REPO / "workflows" / "variants" / "otr_8gb_wan.json").read_text(
            encoding="utf-8"))
    validator = next(
        node for node in variant["nodes"]
        if node.get("type") == "OTR_WorkflowValidator")
    render_batch = next(
        node for node in variant["nodes"]
        if node.get("type") == "OTR_VideoRenderBatch")
    env_recipe = json.loads(
        (REPO / "workflows" / "variants" / "otr_8gb_wan.env.json").read_text(
            encoding="utf-8"))
    assert render_batch["widgets_values"][3] == 17
    assert validator["widgets_values"][4] == env_recipe["master_hash"]
    assert env_recipe["env"]["OTR_WAN_TI2V_MAX_FRAMES"] == "17"
