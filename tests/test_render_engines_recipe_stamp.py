"""tests/test_render_engines_recipe_stamp.py -- S-E5 ledger recipe-stamp.

The meta.render_engines payload gains a per-beat recipe receipt
(delivered_engine + recipe/quant/LoRA/canvas/peak) + a by-engine roll-up,
PRESERVING the existing histogram / by_role / video_revision / vram_peak_mb
keys. The receipt threads engine -> canonical clip -> build_clip_manifest row
-> payload. Pure-Python (no GPU / ledger I/O): the payload builder is split out.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes.otr_video_render_batch import _build_render_engines_payload  # noqa: E402
from nodes._otr_video_engines import render_driver as rd  # noqa: E402


def _manifest_with_recipe():
    return {
        "engine_histogram": {"ltx_audio_in": 1, "still_pan": 1},
        "video_revision": 3,
        "clips": [
            {"shot_id": "s1", "role": "music_visual", "engine_id": "ltx_audio_in",
             "recipe": "distilled_native", "quant": "Q2_K", "use_lora": False,
             "render_canvas": "512x288", "vram_peak_mb": 13900},
            {"shot_id": "s2", "role": "announcer_visual", "engine_id": "still_pan"},
        ],
    }


def test_payload_preserves_existing_keys():
    payload = _build_render_engines_payload(_manifest_with_recipe(), 13900)
    assert payload["histogram"] == {"ltx_audio_in": 1, "still_pan": 1}
    assert payload["video_revision"] == 3
    assert payload["vram_peak_mb"] == 13900
    assert payload["by_role"]["music_visual"] == {"ltx_audio_in": 1}


def test_payload_per_clip_recipe_receipt():
    payload = _build_render_engines_payload(_manifest_with_recipe(), 13900)
    pc = {p["shot_id"]: p for p in payload["per_clip"]}
    assert pc["s1"]["delivered_engine"] == "ltx_audio_in"
    assert pc["s1"]["recipe"] == "distilled_native"
    assert pc["s1"]["quant"] == "Q2_K"
    assert pc["s1"]["use_lora"] is False
    assert pc["s1"]["render_canvas"] == "512x288"
    # an engine that emits NO receipt stamps recipe=None (row never dropped)
    assert pc["s2"]["delivered_engine"] == "still_pan"
    assert pc["s2"]["recipe"] is None


def test_payload_by_engine_rollup():
    payload = _build_render_engines_payload(_manifest_with_recipe(), 13900)
    assert payload["by_engine"]["ltx_audio_in"]["quant"] == "Q2_K"
    assert payload["by_engine"]["still_pan"]["recipe"] is None


def test_payload_empty_manifest():
    payload = _build_render_engines_payload({}, None)
    assert payload["per_clip"] == [] and payload["by_engine"] == {}
    assert payload["by_role"] == {}


def test_build_clip_manifest_threads_recipe_receipt():
    # the recipe receipt rides engine -> canonical clip -> manifest row.
    result = {
        "ledger": {"video": {"shots": [
            {"shot_id": "s1", "role": "music_visual",
             "target_frame_count": 100, "engine_id": "ltx_audio_in"}]},
            "lines": []},
        "clips": {"s1": {"type": "video", "path": "", "engine_id": "ltx_audio_in",
                         "frame_count": 100, "recipe": "sharp_lora",
                         "quant": "Q3_K_M", "use_lora": True,
                         "render_canvas": "512x288", "vram_peak_mb": 15500}},
        "trace": [],
    }
    mft = rd.build_clip_manifest(result, episode_id="ep")
    row = mft["clips"][0]
    assert row["recipe"] == "sharp_lora" and row["quant"] == "Q3_K_M"
    assert row["use_lora"] is True and row["render_canvas"] == "512x288"
    payload = _build_render_engines_payload(mft, 15500)
    assert payload["per_clip"][0]["recipe"] == "sharp_lora"
