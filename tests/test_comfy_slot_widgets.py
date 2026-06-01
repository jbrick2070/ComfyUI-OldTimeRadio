"""Comfy Credits lane -- the sibling of the OpenRouter four-dropdown router.

Pins the 2026-06-01 contract:
  * comfy_slot_a_model / comfy_slot_b_model are APPENDED at indices 21/22
    of the writer's optional block (the OpenRouter pair stays at 19/20, and
    [0..18] is byte-for-byte unchanged).
  * The lane is opt-in / default-off via OTR_ENABLE_COMFY_CREDITS=1 -- when
    disabled the virtual rows + slug catalog never reach the dropdowns, so
    the offline baseline is untouched (mirrors the OpenRouter gate).
  * comfy:slot-a|b resolve to a real catalog slug via the bind -> env ->
    recommended chain; the backend tags provider="comfy_credits" and posts
    behind the cost guard with the ComfyUI-injected auth.
"""
from __future__ import annotations

import types

import pytest

from nodes import _otr_comfy_backend as occ
from nodes import _otr_model_catalog as cat
from nodes.OTR_LedgerScriptWriter import OTR_LedgerScriptWriter as W
from nodes.OTR_LedgerScriptWriter import _resolve_inputs


@pytest.fixture(autouse=True)
def _clean_lane_state():
    """Each test starts with no slot bindings / auth / accrued budget."""
    occ.clear_slot_bindings()
    occ.clear_auth()
    occ.reset_run_budget()
    yield
    occ.clear_slot_bindings()
    occ.clear_auth()
    occ.reset_run_budget()


@pytest.fixture
def comfy_off(monkeypatch):
    monkeypatch.delenv("OTR_ENABLE_COMFY_CREDITS", raising=False)


@pytest.fixture
def comfy_on(monkeypatch):
    monkeypatch.setenv("OTR_ENABLE_COMFY_CREDITS", "1")


# --- enable gate ------------------------------------------------------------


def test_lane_disabled_by_default(comfy_off):
    assert occ.comfy_credits_enabled() is False


def test_lane_enabled_with_flag(comfy_on):
    assert occ.comfy_credits_enabled() is True


# --- slot picker choices ----------------------------------------------------


def test_slot_choices_disabled_show_enable_sentinel(comfy_off):
    for slot in ("a", "b"):
        assert cat.comfy_catalog_dropdown_choices(slot) == [cat.COMFY_ENABLE_SENTINEL]


def test_slot_choices_enabled_lead_with_recommended(comfy_on):
    a = cat.comfy_catalog_dropdown_choices("a")
    b = cat.comfy_catalog_dropdown_choices("b")
    assert a[0] == occ.COMFY_RECOMMENDED_CREATIVE_DEFAULT
    assert b[0] == occ.COMFY_RECOMMENDED_TECHNICAL_DEFAULT
    # "the more options the better" -- the full pinned catalog is offered.
    for slug in occ.COMFY_LLM_MODELS:
        assert slug in a
    assert len(a) == len(set(a))  # deduped


def test_slot_choices_reject_bad_slot():
    with pytest.raises(ValueError):
        cat.comfy_catalog_dropdown_choices("c")


# --- virtual rows: present only when enabled --------------------------------


def test_virtual_rows_absent_when_disabled(comfy_off):
    labels = cat.dropdown_choices()
    assert occ.SLOT_A_ID not in labels
    assert occ.SLOT_B_ID not in labels


def test_virtual_rows_present_when_enabled(comfy_on):
    labels = [e.label for e in cat.build_dropdown_choices()]
    assert occ.SLOT_A_ID in labels
    assert occ.SLOT_B_ID in labels


# --- slug resolution chain --------------------------------------------------


def test_resolve_slug_recommended_when_unbound():
    assert occ.resolve_slug(occ.SLOT_A_ID) == occ.COMFY_RECOMMENDED_CREATIVE_DEFAULT
    assert occ.resolve_slug(occ.SLOT_B_ID) == occ.COMFY_RECOMMENDED_TECHNICAL_DEFAULT


def test_resolve_slug_prefers_binding():
    occ.set_slot_bindings(slot_a="z-ai/glm-5", slot_b="openai/gpt-5.5")
    assert occ.resolve_slug(occ.SLOT_A_ID) == "z-ai/glm-5"
    assert occ.resolve_slug(occ.SLOT_B_ID) == "openai/gpt-5.5"


def test_resolve_slug_env_override(monkeypatch):
    monkeypatch.setenv("OTR_COMFY_SLOT_A_DEFAULT", "x-ai/grok-4.3")
    assert occ.resolve_slug(occ.SLOT_A_ID) == "x-ai/grok-4.3"


def test_set_slot_bindings_ignores_sentinel():
    occ.set_slot_bindings(slot_a=cat.COMFY_ENABLE_SENTINEL)
    # The sentinel is not a real slug -> resolution falls through to recommended.
    assert occ.resolve_slug(occ.SLOT_A_ID) == occ.COMFY_RECOMMENDED_CREATIVE_DEFAULT


# --- backend load + generate ------------------------------------------------


def test_backend_load_tags_provider(comfy_on):
    row = types.SimpleNamespace(context_window=8192)
    entry = occ.ComfyCreditsBackend().load(occ.SLOT_A_ID, row)
    assert entry["provider"] == "comfy_credits"
    assert entry["slug"] == occ.COMFY_RECOMMENDED_CREATIVE_DEFAULT
    assert entry["slot_letter"] == "A"
    # No local handles -- zero VRAM.
    assert "model" not in entry and "tokenizer" not in entry


def test_backend_load_rejects_when_disabled(comfy_off):
    row = types.SimpleNamespace(context_window=8192)
    with pytest.raises(occ.ComfyCreditsConfigError):
        occ.ComfyCreditsBackend().load(occ.SLOT_A_ID, row)


def test_backend_generate_posts_and_extracts(comfy_on, monkeypatch):
    captured = {}

    def _fake_post(*, url, bearer, payload, timeout_s):
        captured["url"] = url
        captured["bearer"] = bearer
        captured["payload"] = payload
        return {
            "status_code": 200,
            "json": {"choices": [{"message": {"content": "a quiet signal"}}]},
            "text": "",
        }

    monkeypatch.setattr(occ, "_post_comfy_chat_completion", _fake_post)
    occ.set_auth(auth_token="tok-abc")
    row = types.SimpleNamespace(context_window=8192)
    backend = occ.ComfyCreditsBackend()
    entry = backend.load(occ.SLOT_A_ID, row)
    out = backend.generate(
        entry, [{"role": "user", "content": "hi"}], max_new_tokens=64,
    )
    assert out == "a quiet signal"
    assert captured["bearer"] == "tok-abc"
    assert captured["payload"]["model"] == occ.COMFY_RECOMMENDED_CREATIVE_DEFAULT


def test_backend_generate_requires_auth(comfy_on, monkeypatch):
    monkeypatch.setattr(
        occ, "_post_comfy_chat_completion",
        lambda **k: {"status_code": 200, "json": {}, "text": ""},
    )
    row = types.SimpleNamespace(context_window=8192)
    backend = occ.ComfyCreditsBackend()
    entry = backend.load(occ.SLOT_A_ID, row)
    # No set_auth() -> no credential -> fail closed before any network call.
    with pytest.raises(occ.ComfyCreditsConfigError):
        backend.generate(entry, [{"role": "user", "content": "hi"}], max_new_tokens=8)


def test_generate_fn_factory_marks_remote(comfy_on):
    row = types.SimpleNamespace(context_window=8192)
    entry = occ.ComfyCreditsBackend().load(occ.SLOT_B_ID, row)
    fn = occ.make_comfy_credits_generate_fn(entry)
    assert callable(fn)
    assert getattr(fn, "_otr_comfy_credits", False) is True


# --- request_slot routing (BUG-LOCAL-299) -----------------------------------


def test_request_slot_routes_comfy_handle_to_backend(comfy_on):
    """BUG-LOCAL-299: request_slot must route a comfy_credits_http row to the
    REMOTE backend (provider-tagged, zero local VRAM) -- not fall through to
    the local HF loader, which tried to download the literal 'comfy:slot-a'
    and raised HFValidationError. Pins the parity gate that shipped openrouter-
    only. No network/auth: load() only resolves the slug."""
    from nodes import _otr_model_loader as ml
    entry = ml.request_slot("creative", occ.SLOT_A_ID)
    assert entry["provider"] == "comfy_credits"
    assert entry["slug"] == occ.resolve_slug(occ.SLOT_A_ID)
    # Remote => zero local handles (never touched the local download/load path).
    assert "model" not in entry and "tokenizer" not in entry


# --- writer surface + _resolve_inputs threading -----------------------------


def test_writer_appends_comfy_slots_after_openrouter():
    spec = W.INPUT_TYPES()
    order = list(spec["required"].keys()) + list(spec["optional"].keys())
    assert order[21] == "comfy_slot_a_model"
    assert order[22] == "comfy_slot_b_model"
    # Hidden auth inputs are declared but are NOT widgets (absent from order).
    assert "auth_token_comfy_org" in spec.get("hidden", {})
    assert "api_key_comfy_org" in spec.get("hidden", {})
    assert "auth_token_comfy_org" not in order


def test_comfy_slot_defaults_selectable_when_enabled(comfy_on):
    spec = W.INPUT_TYPES()
    for key in ("comfy_slot_a_model", "comfy_slot_b_model"):
        choices, meta = spec["optional"][key]
        assert meta["default"] in choices


def test_resolve_inputs_threads_comfy_slots():
    out = _resolve_inputs(
        target_words=350,
        num_characters=2,
        creative_writing_model=cat.DEFAULT_LLM,
        technical_model=cat.DEFAULT_LLM,
        custom_premise="seed",
        comfy_slot_a_model="anthropic/claude-opus-4.7",
        comfy_slot_b_model="deepseek/deepseek-v4-pro",
    )
    assert out["comfy_slot_a_model"] == "anthropic/claude-opus-4.7"
    assert out["comfy_slot_b_model"] == "deepseek/deepseek-v4-pro"


def test_resolve_inputs_old_workflow_defaults_comfy_slots_empty():
    out = _resolve_inputs(
        target_words=350,
        num_characters=2,
        creative_writing_model=cat.DEFAULT_LLM,
        technical_model=cat.DEFAULT_LLM,
        custom_premise="a town wakes to a strange signal",
    )
    assert out["comfy_slot_a_model"] == ""
    assert out["comfy_slot_b_model"] == ""
