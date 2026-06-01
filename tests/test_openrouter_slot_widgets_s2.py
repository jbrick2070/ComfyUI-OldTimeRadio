"""S2 -- the writer's four-dropdown router surface + migration.

Pins the 2026-06-01 go-forward plan S2 contract:
  * openrouter_slot_a_model / openrouter_slot_b_model are APPENDED at the
    END of the optional block (indices 19/20); the existing [0..18] widget
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


# The frozen widget order for indices [0..18] -- must never shift.
_EXPECTED_0_18 = [
    "episode_title", "target_words", "num_characters",
    "creative_writing_model", "technical_model", "custom_premise",
    "include_act_breaks", "act_count", "style", "style_custom",
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
    """The widget KEYS are env-independent: [0..18] frozen, OpenRouter slots
    at 19/20, Comfy Credits slots appended at 21/22 (2026-06-01)."""
    spec = W.INPUT_TYPES()
    order = list(spec["required"].keys()) + list(spec["optional"].keys())
    assert order[:19] == _EXPECTED_0_18, f"index drift in [0..18]: {order[:19]}"
    assert order[19] == "openrouter_slot_a_model"
    assert order[20] == "openrouter_slot_b_model"
    assert order[21] == "comfy_slot_a_model"
    assert order[22] == "comfy_slot_b_model"
    assert len(order) == 23


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
        target_words=350,
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
        target_words=350,
        num_characters=2,
        creative_writing_model=cat.DEFAULT_LLM,
        technical_model=cat.DEFAULT_LLM,
        custom_premise="seed",
        openrouter_slot_a_model="anthropic/claude-opus-4.8",
        openrouter_slot_b_model="deepseek/deepseek-v4-pro",
    )
    assert out["openrouter_slot_a_model"] == "anthropic/claude-opus-4.8"
    assert out["openrouter_slot_b_model"] == "deepseek/deepseek-v4-pro"
