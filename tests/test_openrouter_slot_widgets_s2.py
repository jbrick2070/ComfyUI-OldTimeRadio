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


# THE WRITER'S DECLARED INPUT ORDER -- the single source of truth for it.
#
# `widgets_values` binds by POSITION (BUG-LOCAL-097), so this list is a
# contract with every saved graph in existence, not a style preference.
# Changing it means migrating all 17 shipped graphs in the same commit with
# scripts/otr_widget_surgery.py.
#
# `gate_in` is in this list because it is a declared INPUT, but it is a
# forceInput SOCKET: it consumes no widgets_values slot, which is why the
# saved widget vector is 37 while this list is 38 long.
#
# Departures, so a reader knows why the numbering here will not match older
# comments elsewhere in the repo:
#   style / style_custom  retired 2026-07-05 (style-engine consolidation)
#   refine_target_grade   retired 2026-08-28 (inert revision-loop promise)
#   target_words          retired 2026-08-14 (length is an observation now)
#   perfect_run_spacesaver retired 2026-09-13 (inert since 2026-08-08)
#
# REORDERED 2026-09-14 after three independent readers (Claude Fable, GPT-5.6
# Sol, DeepSeek v4 Pro) proposed an order blind to each other and a fourth pass
# converged them. Reasoning, the contested placements and one driver overrule
# are recorded in docs/GO_FORWARD_PLAN.md row A4.
#
# `source_bank` is the ONLY `required` entry, and that is load-bearing rather
# than cosmetic: ComfyUI iterates input.required BEFORE input.optional into one
# ordered map, so `required` always renders on top. While `episode_title` and
# `num_characters` were the required pair, no order could open with the control
# that decides what kind of episode you get.
_EXPECTED_INPUT_ORDER = [
    # What are we making?
    "source_bank", "source_ref", "visual_style", "episode_title",
    "custom_premise",
    "story_characters", "story_plot", "story_setting", "story_author",
    # How big, and who is in it?
    "act_count", "include_act_breaks", "num_characters", "lemmy_cameo",
    # How does it read?
    "story_scaffold", "creativity",
    "min_p", "repetition_penalty", "max_new_tokens_cap",
    # Which brain writes it?
    "creative_writing_model", "technical_model",
    "openrouter_slot_a_model", "openrouter_slot_b_model",
    "comfy_slot_a_model", "comfy_slot_b_model",
    "google_api_slot_a_model", "google_api_slot_b_model",
    # Set once for this machine.
    "llm_device", "llm_attn_impl", "llm_quant_policy",
    "llm_vram_ceiling_gb", "gguf_n_ctx", "gguf_quant",
    # Lab equipment.
    "use_exchange", "enable_production_stage3_validators",
    "news_briefs_required",
    "replay_from",
    # THE MULTILINGUAL ONE-SWITCH (2026-09-18). Appended after `replay_from`
    # and before the `gate_in` socket, which consumes no widgets_values slot --
    # so this is the TRAILING saved value and every earlier index is untouched
    # (BUG-LOCAL-097). A trailing widget is nearly free; a mid-list one would
    # have cost the re-index everywhere.
    "episode_language",
    "gate_in",                      # SOCKET -- no widgets_values slot
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
    """THE WRITER'S DECLARED INPUT ORDER, stated once, in full.

    This used to be twenty-odd assertions against hardcoded indexes
    (`order[16] == "openrouter_slot_a_model"`, and so on). Those literals had
    been renumbered by hand four separate times -- the style-engine
    consolidation took two slots out at 8/9, `refine_target_grade` took one out
    at 20, `target_words` took one out at 1, and `perfect_run_spacesaver` took
    one out at 8 on 2026-09-13 -- and every renumber was arithmetic performed
    on a layout someone had to reconstruct from comments. Three of those four
    comment blocks were stale by the time the fourth arrived.

    A single list says the same thing, reads as the thing it is pinning, and
    produces a diff a human can check at a glance. It is also the ONE place the
    approved writer reorder has to change: pin the order here first, watch this
    test go red, then move `INPUT_TYPES` to match it.

    `gate_in` appears in this list because it is a declared INPUT, but it is a
    forceInput SOCKET and consumes no `widgets_values` slot -- which is why the
    saved widget vector is 37 while this list is 38.
    """
    spec = W.INPUT_TYPES()
    order = list(spec["required"].keys()) + list(spec["optional"].keys())
    assert order == _EXPECTED_INPUT_ORDER, (
        "the writer's declared input order drifted.\n"
        "  declared: %r\n  expected: %r\n"
        "If this change is intended, update _EXPECTED_INPUT_ORDER in the same "
        "commit AND migrate all 17 shipped graphs with "
        "scripts/otr_widget_surgery.py -- the saved values bind by POSITION, "
        "so a reorder that touches only the class silently re-attaches every "
        "value after the moved index." % (order, _EXPECTED_INPUT_ORDER))

    # The socket, called out separately because it is the one entry here that
    # is NOT a widget and does NOT consume a saved value slot.
    widgets = [n for n in order if n != "gate_in"]
    assert len(widgets) == 37, (
        "the writer should declare 37 widgets plus the gate_in socket; got %d"
        % len(widgets))


# --- conditional creative default; technical never flips --------------------


def test_creative_default_local_when_remote_off(remote_off):
    spec = W.INPUT_TYPES()
    _, meta = spec["optional"]["creative_writing_model"]
    assert meta["default"] == cat.fresh_llm_option()


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
    assert meta["default"] == cat.fresh_llm_option()


def test_slot_picker_defaults_are_selectable(remote_off):
    """Each slot picker's default must be present in its own choice list
    (a COMBO whose default is out-of-list is a load-time hazard)."""
    spec = W.INPUT_TYPES()
    for key in ("openrouter_slot_a_model", "openrouter_slot_b_model"):
        choices, meta = spec["optional"][key]
        assert meta["default"] in choices
    # remote off -> sentinel leads; curated aliases stay listed so a
    # saved deluxe graph storing ~openai/gpt-latest still validates.
    a_choices, _ = spec["optional"]["openrouter_slot_a_model"]
    assert a_choices[0] == cat.OPENROUTER_ENABLE_SENTINEL
    assert "~openai/gpt-latest" in a_choices


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
