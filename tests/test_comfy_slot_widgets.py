"""Comfy Credits lane -- the sibling of the OpenRouter four-dropdown router.

Pins the 2026-06-01 contract:
  * comfy_slot_a_model / comfy_slot_b_model are APPENDED immediately after
    the OpenRouter pair in the writer's combined required+optional order,
    never inserted among the widgets that precede them. The position is
    asserted by NAME -- the absolute offset has moved four times now
    (style/style_custom 2026-07-05, target_words 2026-08-14,
    refine_target_grade 2026-08-28, perfect_run_spacesaver 2026-09-13) and
    every numbered comment written about it went stale within weeks. The
    writer's declared order is stated ONCE, in
    tests/test_openrouter_slot_widgets_s2.py::_EXPECTED_INPUT_ORDER.
  * The Comfy Credits catalog is always listed after the enable-sentinel
    so a saved cloud graph that stores anthropic/claude-sonnet-5 /
    openai/gpt-5.6-luna / openai/gpt-5.6-sol loads on Comfy Cloud.
    generate() still fails closed without a Comfy API key. The env flag
    OTR_ENABLE_COMFY_CREDITS=1 remains a headless opt-in.
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
from tests.fixtures.writer_slots import assert_relative_order


@pytest.fixture(autouse=True)
def _clean_lane_state(monkeypatch):
    """Each test starts with no slot bindings / auth / accrued budget.

    OTR_COMFY_API_KEY is a User-env credential on the operator box, so a
    test that means "no credential" must unpin it or `_bearer()` will
    silently succeed.
    """
    occ.clear_slot_bindings()
    occ.clear_auth()
    occ.reset_run_budget()
    monkeypatch.delenv("OTR_COMFY_API_KEY", raising=False)
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


def test_slot_choices_always_include_pinned_catalog(comfy_off):
    """Every graph, including canonical, uses this same combo. Sonnet 5,
    Luna, GPT-5.5 and Sol must stay pickable. The saved DEFAULT on
    canonical is still the enable-sentinel, not these slugs."""
    must = (
        occ.COMFY_CLAUDE_SONNET_5,
        occ.COMFY_GPT_LUNA,
        "openai/gpt-5.5",
        occ.COMFY_GPT_SOL,
    )
    for slot in ("a", "b"):
        choices = cat.comfy_catalog_dropdown_choices(slot)
        assert choices[0] == cat.COMFY_ENABLE_SENTINEL
        for slug in must:
            assert slug in choices
        assert occ.COMFY_GPT_TERRA in choices
        assert "x-ai/grok-4.20" in choices
        for slug in occ.COMFY_LLM_MODELS:
            assert slug in choices


def test_canonical_comfy_slots_keep_sentinel_not_catalog_defaults():
    """Canonical and local graphs share the Comfy combo, but they must not
    save Sonnet / Luna / GPT-5.5 as the default pick."""
    import json
    from pathlib import Path

    graph = json.loads(
        (Path(__file__).resolve().parents[1] / "workflows" / "otr_canonical.json"
         ).read_text(encoding="utf-8")
    )
    writer = next(n for n in graph["nodes"] if n.get("type") == "OTR_LedgerScriptWriter")
    names = [i["name"] for i in writer["inputs"] if i.get("widget")]
    values = writer["widgets_values"]
    for name in ("comfy_slot_a_model", "comfy_slot_b_model"):
        assert values[names.index(name)] == cat.COMFY_ENABLE_SENTINEL
    spec = W.INPUT_TYPES()
    for key in ("comfy_slot_a_model", "comfy_slot_b_model"):
        choices = list(spec["optional"][key][0])
        assert cat.COMFY_ENABLE_SENTINEL == choices[0]
        assert occ.COMFY_CLAUDE_SONNET_5 in choices
        assert occ.COMFY_GPT_LUNA in choices
        assert "openai/gpt-5.5" in choices


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


# --- virtual rows: always present (pick is the enable) ----------------------


def test_virtual_rows_present_when_disabled(comfy_off):
    labels = cat.dropdown_choices()
    assert occ.SLOT_A_ID in labels
    assert occ.SLOT_B_ID in labels


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


def test_credits_context_reads_openrouter_advertised_window(monkeypatch):
    """Credits is OpenRouter through Comfy's proxy -- same catalog cache."""
    from nodes import _otr_openrouter_backend as orb

    monkeypatch.setattr(
        orb, "_advertised_context_window",
        lambda slug: 400000 if slug == occ.COMFY_GPT_SOL else 0,
    )
    assert occ.resolve_context_window(
        occ.COMFY_GPT_SOL, row_default=8192) == 400000


def test_credits_context_never_falls_back_to_local_8192(monkeypatch):
    from nodes import _otr_openrouter_backend as orb

    monkeypatch.setattr(orb, "_advertised_context_window", lambda slug: 0)
    monkeypatch.setattr(orb, "openrouter_enabled", lambda: False)
    assert occ.resolve_context_window(
        occ.COMFY_GPT_SOL, row_default=8192
    ) == occ.DEFAULT_REMOTE_CONTEXT_WINDOW


def test_credits_load_ignores_virtual_row_8192(comfy_on, monkeypatch):
    monkeypatch.setattr(
        occ, "resolve_context_window",
        lambda slug, row_default=None: 131072,
    )
    occ.set_slot_bindings(slot_a=occ.COMFY_GPT_SOL)
    entry = occ.ComfyCreditsBackend().load(
        occ.SLOT_A_ID, types.SimpleNamespace(context_window=8192))
    assert entry["slug"] == occ.COMFY_GPT_SOL
    assert entry["context_cap"] == 131072
    assert entry["context_window"] == 131072


def test_credits_script_pass_does_not_clamp_against_virtual_8192(
        comfy_on, monkeypatch):
    """The 2026-09-15 deluxe 1-act abort: Sol requested 8192 output against
    an 8192 context_cap, fit_output_tokens left 5487, finish_reason=length."""
    from nodes._otr_generation_budget import ProviderCapacityMessages

    captured = {}

    def _fake_post(*, url, bearer, payload, timeout_s):
        captured["payload"] = payload
        return {
            "status_code": 200,
            "json": {"choices": [{
                "finish_reason": "stop",
                "message": {"content": "full script"},
            }]},
            "text": "",
        }

    monkeypatch.setattr(occ, "_post_comfy_chat_completion", _fake_post)
    monkeypatch.setattr(
        occ, "resolve_context_window",
        lambda slug, row_default=None: 131072,
    )
    occ.set_auth(api_key="key-abc")
    occ.set_slot_bindings(slot_a=occ.COMFY_GPT_SOL)
    backend = occ.ComfyCreditsBackend()
    entry = backend.load(
        occ.SLOT_A_ID, types.SimpleNamespace(context_window=8192))
    msgs = ProviderCapacityMessages([{"role": "user", "content": "write the act"}])
    out = backend.generate(entry, msgs, max_new_tokens=None)
    assert out == "full script"
    assert captured["payload"]["max_tokens"] == occ.DEFAULT_OUTPUT_TOKENS_CAP
    assert captured["payload"]["max_tokens"] == 16384


def test_backend_load_rejects_when_disabled_and_no_auth(comfy_off):
    row = types.SimpleNamespace(context_window=8192)
    with pytest.raises(occ.ComfyCreditsConfigError):
        occ.ComfyCreditsBackend().load(occ.SLOT_A_ID, row)


def test_backend_load_accepts_signed_in_auth_without_env_flag(comfy_off):
    occ.set_auth(api_key="key-abc")
    row = types.SimpleNamespace(context_window=8192)
    entry = occ.ComfyCreditsBackend().load(occ.SLOT_A_ID, row)
    assert entry["provider"] == "comfy_credits"
    assert entry["slug"] == occ.COMFY_RECOMMENDED_CREATIVE_DEFAULT


def test_backend_load_accepts_env_key_when_flag_off(comfy_off, monkeypatch):
    """Headless --cpu with sqlite:///:memory: never injects api_key_comfy_org.
    The first cheap-cloud 1-act died on ComfyCreditsConfigError for that
    reason. OTR_COMFY_API_KEY is the same credential the media lane already
    uses."""
    monkeypatch.setenv("OTR_COMFY_API_KEY", "env-headless-key")
    row = types.SimpleNamespace(context_window=8192)
    entry = occ.ComfyCreditsBackend().load(occ.SLOT_A_ID, row)
    assert entry["provider"] == "comfy_credits"
    assert occ._bearer() == "env-headless-key"


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
    assert captured["payload"]["reasoning_effort"] == "low"
    assert "temperature" not in captured["payload"]


def test_generate_sonnet5_sends_low_reasoning_and_omits_temperature(
        comfy_on, monkeypatch):
    captured = {}

    def _fake_post(*, url, bearer, payload, timeout_s):
        captured["payload"] = payload
        return {
            "status_code": 200,
            "json": {"choices": [{"message": {"content": "{}"}}]},
            "text": "",
        }

    monkeypatch.setattr(occ, "_post_comfy_chat_completion", _fake_post)
    occ.set_auth(api_key="key-abc")
    occ.set_slot_bindings(slot_b=occ.COMFY_CLAUDE_SONNET_5)
    backend = occ.ComfyCreditsBackend()
    entry = backend.load(occ.SLOT_B_ID, types.SimpleNamespace(context_window=8192))
    backend.generate(
        entry, [{"role": "user", "content": "hi"}],
        temperature=0.7, max_new_tokens=64,
    )
    assert captured["payload"]["model"] == occ.COMFY_CLAUDE_SONNET_5
    assert captured["payload"]["reasoning_effort"] == "low"
    assert "temperature" not in captured["payload"]


def test_generate_luna_sends_none_reasoning(comfy_on, monkeypatch):
    captured = {}

    def _fake_post(*, url, bearer, payload, timeout_s):
        captured["payload"] = payload
        return {
            "status_code": 200,
            "json": {"choices": [{"message": {"content": "{}"}}]},
            "text": "",
        }

    monkeypatch.setattr(occ, "_post_comfy_chat_completion", _fake_post)
    occ.set_auth(api_key="key-abc")
    occ.set_slot_bindings(slot_b=occ.COMFY_GPT_LUNA)
    backend = occ.ComfyCreditsBackend()
    entry = backend.load(occ.SLOT_B_ID, types.SimpleNamespace(context_window=8192))
    backend.generate(
        entry, [{"role": "user", "content": "hi"}],
        temperature=0.7, max_new_tokens=64,
    )
    assert captured["payload"]["model"] == occ.COMFY_GPT_LUNA
    assert captured["payload"]["reasoning_effort"] == "none"
    assert captured["payload"]["temperature"] == 0.7


def test_generate_sol_omits_reasoning_effort_and_lifts_output_floor(
        comfy_on, monkeypatch):
    captured = {}

    def _fake_post(*, url, bearer, payload, timeout_s):
        captured["payload"] = payload
        return {
            "status_code": 200,
            "json": {"choices": [{"message": {"content": "ok"}}]},
            "text": "",
        }

    monkeypatch.setattr(occ, "_post_comfy_chat_completion", _fake_post)
    occ.set_auth(api_key="key-abc")
    occ.set_slot_bindings(slot_a=occ.COMFY_GPT_SOL)
    backend = occ.ComfyCreditsBackend()
    entry = backend.load(occ.SLOT_A_ID, types.SimpleNamespace(context_window=8192))
    backend.generate(
        entry, [{"role": "user", "content": "hi"}],
        temperature=0.7, max_new_tokens=64,
    )
    assert captured["payload"]["model"] == occ.COMFY_GPT_SOL
    assert "reasoning_effort" not in captured["payload"]
    assert captured["payload"]["max_tokens"] >= occ.DEFAULT_MIN_OUTPUT_TOKENS_REASONING
    assert captured["payload"]["temperature"] == 0.7


def test_reasoning_effort_for_slug_matches_credits_defaults():
    assert occ.reasoning_effort_for_slug(occ.COMFY_GPT_TERRA) == "none"
    assert occ.reasoning_effort_for_slug(occ.COMFY_GPT_TERRA_PRO) == "none"
    assert occ.reasoning_effort_for_slug(occ.COMFY_GPT_LUNA) == "none"
    assert occ.reasoning_effort_for_slug(occ.COMFY_GPT_LUNA_PRO) == "none"
    assert occ.reasoning_effort_for_slug(occ.COMFY_CLAUDE_SONNET_5) == "low"
    assert occ.reasoning_effort_for_slug(occ.COMFY_GPT_SOL) is None
    assert occ.reasoning_effort_for_slug(occ.COMFY_GPT_SOL_PRO) is None
    assert occ.reasoning_effort_for_slug("x-ai/grok-4.20") is None
    assert occ.reasoning_effort_for_slug("openai/gpt-5.5") is None


def test_backend_generate_requires_auth(comfy_on, monkeypatch):
    monkeypatch.setattr(
        occ, "_post_comfy_chat_completion",
        lambda **k: {"status_code": 200, "json": {}, "text": ""},
    )
    row = types.SimpleNamespace(context_window=8192)
    backend = occ.ComfyCreditsBackend()
    entry = backend.load(occ.SLOT_A_ID, row)
    # No set_auth() and no OTR_COMFY_API_KEY -> fail closed before any network call.
    with pytest.raises(occ.ComfyCreditsConfigError):
        backend.generate(entry, [{"role": "user", "content": "hi"}], max_new_tokens=8)


def test_min_output_tokens_floored_bug301(comfy_on, monkeypatch):
    """BUG-LOCAL-301: a small per-call max_new_tokens must be floored to at
    least DEFAULT_MIN_OUTPUT_TOKENS (1024) on a non-reasoning Credits slug."""
    captured = {}

    def _fake_post(*, url, bearer, payload, timeout_s):
        captured["payload"] = payload
        return {"status_code": 200,
                "json": {"choices": [{"message": {"content": "{}"}}]},
                "text": ""}

    monkeypatch.setattr(occ, "_post_comfy_chat_completion", _fake_post)
    occ.set_auth(api_key="key-abc")
    occ.set_slot_bindings(slot_b="x-ai/grok-4.20")
    backend = occ.ComfyCreditsBackend()
    entry = backend.load(occ.SLOT_B_ID, types.SimpleNamespace(context_window=8192))
    backend.generate(entry, [{"role": "user", "content": "hi"}], max_new_tokens=64)
    assert occ.DEFAULT_MIN_OUTPUT_TOKENS >= 1024
    assert captured["payload"]["max_tokens"] >= 1024
    assert "reasoning_effort" not in captured["payload"]


def test_sonnet5_technical_uses_reasoning_output_floor(comfy_on, monkeypatch):
    """Sonnet 5 cannot turn reasoning off; JSON passes need the 4096 floor
    the OpenRouter lane already uses when effort is not none."""
    captured = {}

    def _fake_post(*, url, bearer, payload, timeout_s):
        captured["payload"] = payload
        return {"status_code": 200,
                "json": {"choices": [{"message": {"content": "{}"}}]},
                "text": ""}

    monkeypatch.setattr(occ, "_post_comfy_chat_completion", _fake_post)
    occ.set_auth(api_key="key-abc")
    occ.set_slot_bindings(slot_b=occ.COMFY_CLAUDE_SONNET_5)
    backend = occ.ComfyCreditsBackend()
    entry = backend.load(occ.SLOT_B_ID, types.SimpleNamespace(context_window=8192))
    backend.generate(entry, [{"role": "user", "content": "hi"}], max_new_tokens=64)
    assert captured["payload"]["reasoning_effort"] == "low"
    assert captured["payload"]["max_tokens"] >= occ.DEFAULT_MIN_OUTPUT_TOKENS_REASONING


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
    # The lane's whole contract is that it was APPENDED behind the OpenRouter
    # pair rather than inserted among the older widgets -- so assert the group,
    # which is what an absolute offset was standing in for.
    assert_relative_order(order, [
        "openrouter_slot_a_model", "openrouter_slot_b_model",
        "comfy_slot_a_model", "comfy_slot_b_model",
    ])
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


def test_credits_run_cap_is_one_million():
    assert occ.DEFAULT_MAX_TOKENS_PER_RUN == 1_000_000


def test_usage_tokens_prefers_provider_total():
    assert occ._usage_tokens({"usage": {"total_tokens": 412}}) == 412
    assert occ._usage_tokens({
        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
    }) == 120
    assert occ._usage_tokens({}) == 0


def test_generate_accounts_provider_usage_not_the_output_cap(monkeypatch):
    """Live 1-act abort: estimate added max_tokens (16384) per call."""
    occ.reset_run_budget()
    monkeypatch.setattr(occ, "_bearer", lambda: "test-token")
    backend = occ.ComfyCreditsBackend()

    def post(*, bearer, payload, slug, fail_on_output_limit=False):
        backend._last_usage_tokens = 350
        return "ok"

    monkeypatch.setattr(backend, "_post_with_retries", post)
    backend.generate(
        {
            "slug": occ.COMFY_GPT_LUNA,
            "model_id": occ.SLOT_B_ID,
            "context_cap": 131072,
            "max_tokens_cap": 16384,
        },
        [{"role": "user", "content": "hello"}],
        max_new_tokens=256,
    )
    assert occ._run_token_total == 350


def _credits_choice(text="ok"):
    return {
        "status_code": 200,
        "json": {
            "choices": [{"message": {"content": text}}],
            "usage": {"total_tokens": 12},
        },
        "text": "",
    }


def test_generate_retries_invalid_comfy_api_key_401(comfy_on, monkeypatch):
    """Live 2026-09-16: deluxe died on one 401 after billed Sol calls.

    The same key then probed 200. Treat 401 as a proxy flake, not a dead
    credential, and wait longer than the 5xx 2s cap.
    """
    sleeps = []
    calls = {"n": 0}

    def fake_post(*, url, bearer, payload, timeout_s):
        calls["n"] += 1
        if calls["n"] == 1:
            return {
                "status_code": 401,
                "json": {"message": "Invalid Comfy API key"},
                "text": '{"message":"Invalid Comfy API key"}',
            }
        return _credits_choice()

    monkeypatch.setattr(occ, "_post_comfy_chat_completion", fake_post)
    monkeypatch.setattr(occ.time, "sleep", lambda s: sleeps.append(s))
    occ.set_auth(api_key="key-abc")
    backend = occ.ComfyCreditsBackend()
    entry = backend.load(occ.SLOT_A_ID, types.SimpleNamespace(context_window=8192))
    text = backend.generate(
        entry, [{"role": "user", "content": "hi"}], max_new_tokens=64,
    )
    assert text == "ok"
    assert calls["n"] == 2
    assert sleeps and sleeps[0] >= 2.0


def test_generate_401_exhausted_reports_actual_attempts(comfy_on, monkeypatch):
    monkeypatch.setattr(
        occ, "_post_comfy_chat_completion",
        lambda **k: {
            "status_code": 401,
            "json": {"message": "Invalid Comfy API key"},
            "text": "",
        },
    )
    monkeypatch.setattr(occ.time, "sleep", lambda s: None)
    occ.set_auth(api_key="key-abc")
    backend = occ.ComfyCreditsBackend()
    entry = backend.load(occ.SLOT_A_ID, types.SimpleNamespace(context_window=8192))
    with pytest.raises(occ.ComfyCreditsCallFailedError, match="failed after 5 attempt"):
        backend.generate(
            entry, [{"role": "user", "content": "hi"}], max_new_tokens=64,
        )


def test_generate_402_does_not_retry_and_names_top_up(comfy_on, monkeypatch):
    calls = {"n": 0}

    def fake_post(*, url, bearer, payload, timeout_s):
        calls["n"] += 1
        return {
            "status_code": 402,
            "json": {"error": "Payment Required"},
            "text": "Payment Required",
        }

    monkeypatch.setattr(occ, "_post_comfy_chat_completion", fake_post)
    monkeypatch.setattr(occ.time, "sleep", lambda s: None)
    occ.set_auth(api_key="key-abc")
    backend = occ.ComfyCreditsBackend()
    entry = backend.load(occ.SLOT_A_ID, types.SimpleNamespace(context_window=8192))
    with pytest.raises(
            occ.ComfyCreditsCallFailedError,
            match="cloud.comfy.org") as ei:
        backend.generate(
            entry, [{"role": "user", "content": "hi"}], max_new_tokens=64,
        )
    assert calls["n"] == 1
    assert "failed after 1 attempt" in str(ei.value)
    assert "endpoint-config" in str(ei.value)


def test_generate_500_still_stops_at_default_retries(comfy_on, monkeypatch):
    calls = {"n": 0}

    def fake_post(*, url, bearer, payload, timeout_s):
        calls["n"] += 1
        return {"status_code": 500, "json": {"message": "upstream"}, "text": ""}

    monkeypatch.setattr(occ, "_post_comfy_chat_completion", fake_post)
    monkeypatch.setattr(occ.time, "sleep", lambda s: None)
    occ.set_auth(api_key="key-abc")
    backend = occ.ComfyCreditsBackend()
    entry = backend.load(occ.SLOT_A_ID, types.SimpleNamespace(context_window=8192))
    with pytest.raises(occ.ComfyCreditsCallFailedError, match="failed after 3 attempt"):
        backend.generate(
            entry, [{"role": "user", "content": "hi"}], max_new_tokens=64,
        )
    assert calls["n"] == 3


def test_error_snippet_reads_comfy_top_level_message():
    snippet = occ.ComfyCreditsBackend._error_snippet({
        "json": {"message": "Invalid Comfy API key"},
        "text": '{"message":"Invalid Comfy API key"}',
    })
    assert snippet == "Invalid Comfy API key"


def test_every_cloud_credits_json_raises_the_run_cap():
    from pathlib import Path
    import json

    root = Path(__file__).resolve().parents[1] / "config" / "profiles"
    rows = list(root.glob("otr_cloud*.json"))
    assert rows, "no cloud profiles"
    for path in rows:
        data = json.loads(path.read_text(encoding="utf-8"))
        env = ((data.get("launch") or {}).get("env") or {})
        cap = int(env.get("OTR_COMFY_MAX_TOKENS_PER_RUN") or 0)
        assert cap >= 1_000_000, "%s missing Credits run cap" % path.name
