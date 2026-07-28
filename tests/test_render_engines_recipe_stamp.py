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
from nodes import otr_credits_roll as cr  # noqa: E402


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


# --------------------------------------------------------------------------- #
# The by_engine roll-up is PER FIELD, never "the first clip wins"
# (2026-07-28 -- the LANE 2 fan-out lens-C row).
#
# WHY THE DEFECT SURVIVED: test_payload_by_engine_rollup above defines exactly
# ONE clip per engine, so the roll-up was never asked what it does with a
# second one. Every fixture below drives the REAL builder and hands its REAL
# output to the REAL credits reader -- none of them hand-builds a by_engine row
# and then verifies it, which is this build's most reliable blind spot.
# --------------------------------------------------------------------------- #
def _clip(shot_id, engine_id, **over):
    row = {"shot_id": shot_id, "role": "music_visual", "engine_id": engine_id}
    row.update(over)
    return row


def _payload(*clips):
    return _build_render_engines_payload({"clips": list(clips)}, None)


def test_by_engine_no_longer_lets_the_first_clip_speak_for_the_rest():
    """THE DEFECT, as its own test: two clips, one engine, two recipes."""
    p = _payload(
        _clip("s1", "ltx_audio_in", recipe="RECIPE_LTX8_I2V_v2"),
        _clip("s2", "ltx_audio_in",
              recipe="RECIPE_LTX8_I2V_v2+prequalification[tiled_vae=off]"),
    )
    row = p["by_engine"]["ltx_audio_in"]
    assert row["recipe"] != "RECIPE_LTX8_I2V_v2"          # the OLD behaviour
    assert row["recipe"] is None
    assert row["varied"] == ["recipe"]
    assert row["clip_count"] == 2
    # ...and per_clip stayed lossless, which is what makes the None recoverable.
    assert [c["recipe"] for c in p["per_clip"]] == [
        "RECIPE_LTX8_I2V_v2",
        "RECIPE_LTX8_I2V_v2+prequalification[tiled_vae=off]"]


def test_by_engine_still_reports_a_field_every_clip_agreed_on():
    """Blanking the WHOLE row on any disagreement is the opposite defect: the
    roll-up would stop reporting facts it does know."""
    p = _payload(
        _clip("s1", "wan_i2v", recipe="WAN_I2V_RECIPE_v1", quant="Q8_0"),
        _clip("s2", "wan_i2v", quant="Q8_0",
              recipe="WAN_I2V_RECIPE_v1+prequalification[shift=3.0]"),
    )
    row = p["by_engine"]["wan_i2v"]
    assert row["quant"] == "Q8_0"
    assert row["recipe"] is None and row["varied"] == ["recipe"]


def test_by_engine_render_canvas_is_the_case_reachable_at_head():
    """Needs no sweep and no consent act -- any episode mixing aspect ratios
    on one engine reaches this today."""
    p = _payload(_clip("s1", "still_pan", render_canvas="832x480"),
                 _clip("s2", "still_pan", render_canvas="480x832"))
    row = p["by_engine"]["still_pan"]
    assert row["render_canvas"] is None
    assert row["varied"] == ["render_canvas"]


def test_by_engine_peak_is_the_worst_clip_not_the_first_or_the_last():
    p = _payload(_clip("s1", "wan_ti2v", vram_peak_mb=8241),
                 _clip("s2", "wan_ti2v", vram_peak_mb=16127),
                 _clip("s3", "wan_ti2v", vram_peak_mb=11062))
    row = p["by_engine"]["wan_ti2v"]
    assert row["vram_peak_mb"] == 16127
    assert isinstance(row["vram_peak_mb"], int)   # returned exactly as stamped
    assert row["varied"] == []                    # a measurement, not identity


def test_by_engine_peak_ignores_an_unmeasured_clip_without_blanking():
    p = _payload(_clip("s1", "humo", vram_peak_mb=None),
                 _clip("s2", "humo", vram_peak_mb=4096.5))
    assert p["by_engine"]["humo"]["vram_peak_mb"] == 4096.5


def test_by_engine_peak_is_none_when_no_clip_measured_one():
    """CONTROL -- passes under the pre-2026-07-28 code too. It pins that an
    unmeasured engine (humo stamps no peak) still reports None rather than
    acquiring one from the worst-of loop."""
    p = _payload(_clip("s1", "humo"), _clip("s2", "humo"))
    assert p["by_engine"]["humo"]["vram_peak_mb"] is None


def test_a_nan_peak_cannot_swallow_a_real_measurement():
    """LATENT, and closed at the boundary rather than argued about: NaN
    compares False against everything, so once it was the incumbent worst no
    later peak could beat it -- and the answer would depend on clip ORDER.
    No adapter stamps a float peak today (gpu_residency casts
    int(info.used) // _MB), so this is the A5-lite shape: two lines at the
    boundary, not a live bug. Both orders must give the real number."""
    nan = float("nan")
    assert _payload(
        _clip("s1", "wan_ti2v", vram_peak_mb=nan),
        _clip("s2", "wan_ti2v", vram_peak_mb=20000),
    )["by_engine"]["wan_ti2v"]["vram_peak_mb"] == 20000
    assert _payload(
        _clip("s1", "wan_ti2v", vram_peak_mb=20000),
        _clip("s2", "wan_ti2v", vram_peak_mb=nan),
    )["by_engine"]["wan_ti2v"]["vram_peak_mb"] == 20000


def test_a_uniform_engine_reports_varied_empty_and_every_field_intact():
    p = _payload(
        _clip("s1", "ltx_8gb", recipe="R", quant="Q8_0", use_lora=False,
              render_canvas="512x288", family="image_to_video"),
        _clip("s2", "ltx_8gb", recipe="R", quant="Q8_0", use_lora=False,
              render_canvas="512x288", family="image_to_video"),
    )
    row = p["by_engine"]["ltx_8gb"]
    assert row["varied"] == []
    assert row["recipe"] == "R" and row["quant"] == "Q8_0"
    assert row["use_lora"] is False and row["render_canvas"] == "512x288"
    assert row["family"] == "image_to_video" and row["clip_count"] == 2


def test_unstamped_and_false_are_a_disagreement_not_a_match():
    """A clip that never stamped use_lora and a clip that stamped False are
    two different statements; collapsing them reports a LoRA decision the
    first clip never made."""
    p = _payload(_clip("s1", "ltx_video"),
                 _clip("s2", "ltx_video", use_lora=False))
    assert p["by_engine"]["ltx_video"]["varied"] == ["use_lora"]


def test_varied_lists_fields_in_the_receipts_own_order():
    p = _payload(
        _clip("s1", "ltx_av", recipe="a", quant="q1", render_canvas="832x480"),
        _clip("s2", "ltx_av", recipe="b", quant="q2", render_canvas="480x832"),
    )
    assert p["by_engine"]["ltx_av"]["varied"] == [
        "recipe", "quant", "render_canvas"]


def test_three_distinct_values_on_one_field_is_still_one_varied_entry():
    p = _payload(_clip("s1", "ltx_av", render_canvas="832x480"),
                 _clip("s2", "ltx_av", render_canvas="512x288"),
                 _clip("s3", "ltx_av", render_canvas="480x832"))
    row = p["by_engine"]["ltx_av"]
    assert row["render_canvas"] is None
    assert row["varied"] == ["render_canvas"]     # named once, not per value
    assert row["clip_count"] == 3


def test_a_varied_family_is_named_in_the_payload_not_only_on_the_card():
    """The other four identity fields each get a payload-level assertion;
    family was reachable only through the credits reader."""
    p = _payload(_clip("s1", "ltx_video", family="text_to_video"),
                 _clip("s2", "ltx_video", family="image_to_video"))
    row = p["by_engine"]["ltx_video"]
    assert row["family"] is None and row["varied"] == ["family"]


def test_two_multi_clip_engines_in_one_payload_do_not_leak_into_each_other():
    """THE NORMAL PRODUCTION SHAPE, and the one the fixtures kept missing: a
    real episode routes clips across several engines, several clips each. The
    shipped fixture had one clip per engine; the first draft of THIS file had
    several clips but only ever one engine -- the same blind spot one level
    up. Found by the pre-push QA fan-out (lens C)."""
    p = _payload(
        _clip("s1", "ltx_8gb", recipe="LTX_v2", quant="Q8_0"),
        _clip("s2", "wan_i2v", recipe="WAN_I2V_v1", quant="fp8"),
        _clip("s3", "ltx_8gb", recipe="LTX_v2", quant="Q8_0"),
        _clip("s4", "wan_i2v", quant="fp8",
              recipe="WAN_I2V_v1+prequalification[shift=3.0]"),
        _clip("s5", "ltx_8gb", recipe="LTX_v2", quant="Q8_0"),
    )
    assert set(p["by_engine"]) == {"ltx_8gb", "wan_i2v"}
    ltx, wan = p["by_engine"]["ltx_8gb"], p["by_engine"]["wan_i2v"]
    # ltx agreed with itself -- wan's departure must not contaminate it.
    assert ltx["recipe"] == "LTX_v2" and ltx["varied"] == []
    assert ltx["clip_count"] == 3
    # ...and wan's disagreement must not be laundered by ltx's agreement.
    assert wan["recipe"] is None and wan["varied"] == ["recipe"]
    assert wan["quant"] == "fp8" and wan["clip_count"] == 2


# --------------------------------------------------------------------------- #
# ...and the two production readers of by_engine, which live on the credits
# card. This is why the roll-up had to be fixed BEFORE the card is wired to
# draw video_suffix: over a first-clip-wins roll-up, wiring it puts
# confidently-wrong per-engine data on screen instead of none.
# --------------------------------------------------------------------------- #
def test_the_credits_recipe_line_says_mixed_instead_of_one_clips_values():
    ren = _payload(
        _clip("s1", "ltx_8gb", recipe="RECIPE_LTX8_I2V_v2", quant="Q8_0",
              render_canvas="512x288"),
        _clip("s2", "ltx_8gb", quant="Q8_0", render_canvas="832x480",
              recipe="RECIPE_LTX8_I2V_v2+prequalification[tiled_vae=off]"),
    )
    line = cr._recipe_suffix(ren, "ltx_8gb")
    assert "RECIPE_LTX8_I2V_v2" not in line
    assert "512x288" not in line and "832x480" not in line
    assert line == "Q8_0 · mixed recipe, render_canvas"


def test_a_mixed_lora_decision_never_prints_lora():
    """The trap a truthy "(varied)" sentinel would have walked into:
    _recipe_suffix tests use_lora for TRUTH, so any non-empty marker parked in
    that bool field would print "lora" for an engine that mixed both."""
    ren = _payload(_clip("s1", "wan_i2v", use_lora=True),
                   _clip("s2", "wan_i2v", use_lora=False))
    assert cr._recipe_suffix(ren, "wan_i2v") == "mixed use_lora"


def test_a_uniform_lora_decision_still_prints_lora():
    """CONTROL -- passes under the pre-2026-07-28 code too, on purpose: it is
    what stops a "always say mixed" implementation from looking correct."""
    ren = _payload(_clip("s1", "wan_i2v", use_lora=True),
                   _clip("s2", "wan_i2v", use_lora=True))
    assert cr._recipe_suffix(ren, "wan_i2v") == "lora"


def test_the_family_slot_says_mixed_rather_than_going_blank():
    ren = _payload(_clip("s1", "ltx_video", family="text_to_video"),
                   _clip("s2", "ltx_video", family="image_to_video"))
    assert cr._video_role_rows(ren) == [("music_visual", "ltx_video", "mixed")]
    # ...and the recipe line does NOT repeat it -- family has its own slot.
    assert "family" not in cr._recipe_suffix(ren, "ltx_video")


def test_a_uniform_family_still_pretty_prints():
    """CONTROL -- passes under the pre-2026-07-28 code too, on purpose: the
    "mixed" branch must not have eaten the _FAMILY_PRETTY mapping."""
    ren = _payload(_clip("s1", "ltx_video", family="text_to_video"),
                   _clip("s2", "ltx_video", family="text_to_video"))
    assert cr._video_role_rows(ren) == [
        ("music_visual", "ltx_video", "text-to-video")]


def test_a_pre_rollup_ledger_reads_exactly_as_before():
    """CONTROL, and the one row here that is hand-built ON PURPOSE: episodes
    already on disk carry no ``varied`` key, and the builder can no longer
    emit this shape, so the only way to pin the compatibility is to write the
    old shape out. It is a fixture for the READERS, not for the roll-up. Same
    shape as tests/test_credits_roll_spec.py's ``_led()``."""
    legacy = {"by_role": {"music_visual": {"ltx_video": 1}},
              "by_engine": {"ltx_video": {"family": "text_to_video",
                                          "recipe": "ia2v", "quant": "q4_K_M",
                                          "use_lora": True}}}
    assert cr._recipe_suffix(legacy, "ltx_video") == "ia2v · q4_K_M · lora"
    assert cr._video_role_rows(legacy) == [
        ("music_visual", "ltx_video", "text-to-video")]


def test_the_whole_chain_carries_a_second_clip_on_one_engine():
    """engine -> canonical clip -> build_clip_manifest -> payload -> the card,
    with TWO clips on ONE engine: the case the shipped fixture never had."""
    result = {
        "ledger": {"video": {"shots": [
            {"shot_id": "s1", "role": "music_visual",
             "target_frame_count": 97, "engine_id": "ltx_8gb"},
            {"shot_id": "s2", "role": "music_visual",
             "target_frame_count": 97, "engine_id": "ltx_8gb"}]},
            "lines": []},
        "clips": {
            "s1": {"type": "video", "path": "", "engine_id": "ltx_8gb",
                   "frame_count": 97, "recipe": "RECIPE_LTX8_I2V_v2",
                   "render_canvas": "512x288", "vram_peak_mb": 8278},
            "s2": {"type": "video", "path": "", "engine_id": "ltx_8gb",
                   "frame_count": 97, "recipe": "RECIPE_LTX8_I2V_v2",
                   "render_canvas": "832x480", "vram_peak_mb": 16086}},
        "trace": [],
    }
    mft = rd.build_clip_manifest(result, episode_id="ep")
    payload = _build_render_engines_payload(mft, 16086)
    row = payload["by_engine"]["ltx_8gb"]
    assert row["recipe"] == "RECIPE_LTX8_I2V_v2"    # agreed -> still reported
    assert row["render_canvas"] is None             # disagreed -> named
    assert row["varied"] == ["render_canvas"]
    assert row["vram_peak_mb"] == 16086             # the worst, not the first
    assert cr._recipe_suffix(payload, "ltx_8gb") == (
        "RECIPE_LTX8_I2V_v2 · mixed render_canvas")
