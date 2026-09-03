"""S2 -- the writer's four-dropdown router surface + migration.

Pins the 2026-06-01 go-forward plan S2 contract (indices shifted -2 by
the 2026-07-05 style-engine consolidation, which deleted the style /
style_custom widgets that used to sit at [8, 9], then shifted a further
-1 by the 2026-08-14 removal of the `target_words` widget, formerly
slot 1):
  * openrouter_slot_a_model / openrouter_slot_b_model are pinned at
    indices 16/17; the existing [0..15] widget
    order is byte-for-byte unchanged (saved workflows bind by index --
    the BUG-LOCAL-258/253 index-drift trap).
  * creative_writing_model's default is CONDITIONAL: openrouter:slot-a when
    remote is enabled, else local DEFAULT_LLM. technical_model never flips.
  * _resolve_inputs threads the two slot values through; an old workflow
    (no slot kwargs) resolves them to "" with no other value shifted.
"""
from __future__ import annotations

import pytest

from nodes import _otr_model_catalog as cat
from nodes.OTR_LedgerScriptWriter import OTR_LedgerScriptWriter as W
from nodes.OTR_LedgerScriptWriter import _resolve_inputs


# The frozen widget order for indices [0..15] -- must never shift.
# style / style_custom retired 2026-07-05 (style-engine consolidation);
# target_words retired 2026-08-14 (episode length is an observation now,
# never a word-count instruction).
_EXPECTED_0_15 = [
    "episode_title", "num_characters",
    "creative_writing_model", "technical_model", "custom_premise",
    "include_act_breaks", "act_count",
    "creativity", "perfect_run_spacesaver", "min_p",
    "repetition_penalty", "max_new_tokens_cap", "lemmy_cameo",
    "use_exchange", "enable_production_stage3_validators",
    "news_briefs_required",
]


@pytest.fixture
def remote_off(monkeypatch):
    for k in ("OPENROUTER_API_KEY", "OTR_ENABLE_OPENROUTER"):
        monkeypatch.delenv(k, raising=False)


@pytest.fixture
def remote_on(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setenv("OTR_ENABLE_OPENROUTER", "1")
    monkeypatch.setenv("OPENROUTER_MODEL_A", "anthropic/claude-opus-4.8")
    monkeypatch.setenv("OPENROUTER_MODEL_B", "deepseek/deepseek-v4-pro")


# --- append-not-insert: index order ----------------------------------------


def test_widget_order_appends_slots_at_end():
    """The widget KEYS are env-independent: [0..15] frozen, OpenRouter slots
    at 16/17, Comfy Credits slots at 18/19 (2026-06-01), and the Google
    API slots appended at 24/25 (2026-07-08), then source_ref at 26
    (Source Banks v2, 2026-07-08). Indices shifted -2 by the 2026-07-05
    style-engine consolidation (style / style_custom widgets deleted),
    then a further -1 by the 2026-08-14 target_words removal."""
    spec = W.INPUT_TYPES()
    order = list(spec["required"].keys()) + list(spec["optional"].keys())
    assert order[:16] == _EXPECTED_0_15, f"index drift in [0..15]: {order[:16]}"
    assert order[16] == "openrouter_slot_a_model"
    assert order[17] == "openrouter_slot_b_model"
    assert order[18] == "comfy_slot_a_model"
    assert order[19] == "comfy_slot_b_model"
    # slot 20 was refine_target_grade until 2026-08-28. It was an inert
    # widget promising a revision loop deleted a month earlier, so it was
    # removed WITH its migration and every later slot moved down by one.
    assert order[20] == "story_scaffold"          # scaffold toggle (2026-06-24)
    assert order[21] == "source_bank"             # Stage 2C (2026-07-05)
    assert order[22] == "visual_style"            # Stage 3C (2026-07-06)
    assert order[23] == "google_api_slot_a_model" # Google API (2026-07-08)
    assert order[24] == "google_api_slot_b_model" # Google API (2026-07-08)
    assert order[25] == "source_ref"              # Source Banks v2 (2026-07-08)
    # S5 platform-portability (2026-07-10): the six explicit LLM
    # runtime-policy widgets, appended at 27-32 (append-only).
    assert order[26] == "llm_device"
    assert order[27] == "llm_attn_impl"
    assert order[28] == "llm_quant_policy"
    assert order[29] == "llm_vram_ceiling_gb"
    assert order[30] == "gguf_n_ctx"
    assert order[31] == "gguf_quant"
    # gate_in (S5 validation-order fix) is a forceInput SOCKET -- present
    # in the INPUT_TYPES key order but consumes NO widgets_values slot
    # (the serialized widget vector stays 33).
    assert order[32] == "gate_in"
    # 33 since 2026-08-28: refine_target_grade (was slot 20) was removed
    # as an inert widget, with all 62 saved graphs re-indexed.
    # 34 since 2026-09-02: replay_from (CANONICAL REPLAY, campaign item 0)
    # appended as the trailing widget after the gate_in socket -- the
    # append-only rule this test exists to enforce.
    assert order[33] == "replay_from"
    assert len(order) == 34


# --- conditional creative default; technical never flips --------------------


def test_creative_default_local_when_remote_off(remote_off):
    spec = W.INPUT_TYPES()
    _, meta = spec["optional"]["creative_writing_model"]
    assert meta["default"] == cat.DEFAULT_LLM


def test_creative_default_slot_a_when_remote_on(remote_on):
    spec = W.INPUT_TYPES()
    _, meta = spec["optional"]["creative_writing_model"]
    assert meta["default"] == "openrouter:slot-a"
    # The flipped default must be a valid choice (in the dropdown).
    choices, _ = spec["optional"]["creative_writing_model"]
    assert "openrouter:slot-a" in choices


def test_technical_default_never_flips(remote_on):
    spec = W.INPUT_TYPES()
    _, meta = spec["optional"]["technical_model"]
    assert meta["default"] == cat.DEFAULT_LLM


def test_slot_picker_defaults_are_selectable(remote_off):
    """Each slot picker's default must be present in its own choice list
    (a COMBO whose default is out-of-list is a load-time hazard)."""
    spec = W.INPUT_TYPES()
    for key in ("openrouter_slot_a_model", "openrouter_slot_b_model"):
        choices, meta = spec["optional"][key]
        assert meta["default"] in choices
    # remote off -> the sole choice is the enable sentinel.
    a_choices, _ = spec["optional"]["openrouter_slot_a_model"]
    assert a_choices == [cat.OPENROUTER_ENABLE_SENTINEL]


# --- _resolve_inputs migration + threading ----------------------------------


def test_resolve_inputs_old_workflow_supplies_slot_defaults():
    """Old workflow shape: _resolve_inputs called with NO slot kwargs ->
    both slots default to "" (unset); creative/technical unchanged."""
    out = _resolve_inputs(
        num_characters=2,
        episode_title="",
        creative_writing_model=cat.DEFAULT_LLM,
        technical_model=cat.DEFAULT_LLM,
        custom_premise="a quiet town wakes to a strange signal",
        # no openrouter_slot_* kwargs -- the pre-S2 call shape.
    )
    assert out["creative_writing_model"] == cat.DEFAULT_LLM
    assert out["technical_model"] == cat.DEFAULT_LLM
    assert out["openrouter_slot_a_model"] == ""
    assert out["openrouter_slot_b_model"] == ""


def test_resolve_inputs_threads_slot_values():
    out = _resolve_inputs(
        num_characters=2,
        creative_writing_model=cat.DEFAULT_LLM,
        technical_model=cat.DEFAULT_LLM,
        custom_premise="seed",
        openrouter_slot_a_model="anthropic/claude-opus-4.8",
        openrouter_slot_b_model="deepseek/deepseek-v4-pro",
    )
    assert out["openrouter_slot_a_model"] == "anthropic/claude-opus-4.8"
    assert out["openrouter_slot_b_model"] == "deepseek/deepseek-v4-pro"


# --- BUG-LOCAL-400: saved sentinels must validate with lanes ENABLED ---------


def test_saved_slot_sentinels_validate_with_lanes_enabled_bug400(monkeypatch):
    """BUG-LOCAL-400 (the live GUI failure): the shipped workflow stores the
    '(enable ...)' sentinel in all four writer slots (pinned by
    test_workflow_json_guardrails). With the OpenRouter + Comfy Credits lanes
    ENABLED, those saved values MUST remain members of the node's INPUT_TYPES
    choices -- otherwise ComfyUI's COMBO validator rejects the prompt and every
    output is dropped (server log: "Value not in list ... Output will be
    ignored"). Before the fix the enabled dropdowns omitted the sentinel."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setenv("OTR_ENABLE_OPENROUTER", "1")
    monkeypatch.setenv("OTR_ENABLE_COMFY_CREDITS", "1")
    spec = W.INPUT_TYPES()
    saved = {
        "openrouter_slot_a_model": cat.OPENROUTER_ENABLE_SENTINEL,
        "openrouter_slot_b_model": cat.OPENROUTER_ENABLE_SENTINEL,
        "comfy_slot_a_model": cat.COMFY_ENABLE_SENTINEL,
        "comfy_slot_b_model": cat.COMFY_ENABLE_SENTINEL,
    }
    for key, saved_val in saved.items():
        choices, meta = spec["optional"][key]
        assert saved_val in choices, (
            f"{key}: saved {saved_val!r} not in INPUT_TYPES choices "
            f"(first 3: {choices[:3]}) -- COMBO validation would reject it"
        )
        assert meta["default"] in choices, (
            f"{key}: default {meta['default']!r} not in choices"
        )
