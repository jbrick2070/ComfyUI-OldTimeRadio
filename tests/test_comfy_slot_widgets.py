"""Comfy Credits lane -- the sibling of the OpenRouter four-dropdown router.

Pins the 2026-06-01 contract (indices shifted -2 by the 2026-07-05
style-engine consolidation, which deleted the style/style_custom
widgets that used to sit earlier in the optional block, then a further
-1 by the 2026-08-14 removal of the `target_words` widget):
  * comfy_slot_a_model / comfy_slot_b_model are APPENDED at indices 18/19
    of the writer's combined required+optional order (the OpenRouter
    pair stays at 16/17).
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
    # BUG-LOCAL-400: the enable-sentinel leads in every state; the recommended
    # default is the first REAL slug.
    assert a[0] == cat.COMFY_ENABLE_SENTINEL
    assert b[0] == cat.COMFY_ENABLE_SENTINEL
    assert a[1] == occ.COMFY_RECOMMENDED_CREATIVE_DEFAULT
    assert b[1] == occ.COMFY_RECOMMENDED_TECHNICAL_DEFAULT
    # "the more options the better" -- the full pinned catalog is offered.
    for slug in occ.COMFY_LLM_MODELS:
        assert slug in a
    assert len(a) == len(set(a))  # deduped


def test_enable_sentinel_leads_when_enabled_bug400(comfy_on):
    """BUG-LOCAL-400: the enable-sentinel must remain choices[0] when the lane
    is ENABLED so a saved workflow storing the sentinel validates (otherwise
    ComfyUI COMBO validation rejects it and drops every output)."""
    for slot in ("a", "b"):
        choices = cat.comfy_catalog_dropdown_choices(slot)
        assert choices[0] == cat.COMFY_ENABLE_SENTINEL
        assert choices.count(cat.COMFY_ENABLE_SENTINEL) == 1


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


def test_set_slot_bindings_sentinel_clears_prior_binding():
    """A persistent ComfyUI server must not leak a prior run's cloud slug into
    a later run whose slot picker is back on the sentinel."""
    occ.set_slot_bindings(slot_a="openai/gpt-5.5")
    assert occ.resolve_slug(occ.SLOT_A_ID) == "openai/gpt-5.5"
    occ.set_slot_bindings(slot_a=cat.COMFY_ENABLE_SENTINEL)
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
    occ.set_auth(api_key="key-abc")
    row = types.SimpleNamespace(context_window=8192)
    backend = occ.ComfyCreditsBackend()
    entry = backend.load(occ.SLOT_A_ID, row)
    out = backend.generate(
        entry, [{"role": "user", "content": "hi"}], max_new_tokens=64,
    )
    assert out == "a quiet signal"
    assert captured["bearer"] == "key-abc"
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


def test_min_output_tokens_floored_bug301(comfy_on, monkeypatch):
    """BUG-LOCAL-301: a small per-call max_new_tokens must be floored to at
    least DEFAULT_MIN_OUTPUT_TOKENS (1024, parity with the OpenRouter lane) so
    a verbose / reasoning technical model (deepseek-v4-pro) is not truncated
    mid-JSON (finish_reason=length -> JSONDecodeError)."""
    captured = {}

    def _fake_post(*, url, bearer, payload, timeout_s):
        captured["payload"] = payload
        return {"status_code": 200,
                "json": {"choices": [{"message": {"content": "{}"}}]},
                "text": ""}

    monkeypatch.setattr(occ, "_post_comfy_chat_completion", _fake_post)
    occ.set_auth(api_key="key-abc")
    backend = occ.ComfyCreditsBackend()
    entry = backend.load(occ.SLOT_B_ID, types.SimpleNamespace(context_window=8192))
    backend.generate(entry, [{"role": "user", "content": "hi"}], max_new_tokens=64)
    assert occ.DEFAULT_MIN_OUTPUT_TOKENS >= 1024
    assert captured["payload"]["max_tokens"] >= 1024


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
    assert order[18] == "comfy_slot_a_model"
    assert order[19] == "comfy_slot_b_model"
    # The hidden auth input is declared but is NOT a widget (absent from order).
    assert "api_key_comfy_org" in spec.get("hidden", {})
    assert "api_key_comfy_org" not in order
    # PBUG-20260902-04: the session-bearer hidden input is a Comfy Registry
    # prohibited string (critical, credential-access). It must never return.
    assert "auth_token_comfy_org" not in spec.get("hidden", {})


def test_comfy_slot_defaults_selectable_when_enabled(comfy_on):
    spec = W.INPUT_TYPES()
    for key in ("comfy_slot_a_model", "comfy_slot_b_model"):
        choices, meta = spec["optional"][key]
        assert meta["default"] in choices


def test_resolve_inputs_threads_comfy_slots():
    out = _resolve_inputs(
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
        num_characters=2,
        creative_writing_model=cat.DEFAULT_LLM,
        technical_model=cat.DEFAULT_LLM,
        custom_premise="a town wakes to a strange signal",
    )
    assert out["comfy_slot_a_model"] == ""
    assert out["comfy_slot_b_model"] == ""
