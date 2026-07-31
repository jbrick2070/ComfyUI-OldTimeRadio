"""S1 -- OpenRouter backend, mocked HTTP (no network, no secrets).

Proves the FC1 surface (load/generate/unload), the C6 cost-ceiling
abort (with a mocked token counter, asserting NO request is sent), the
C5 bounded-retry-then-clean-abort ladder, and the C3 offline-first
gate. The only network seam (`_post_chat_completion`) is patched in
every call test, so CI never touches the network and no key is ever
sent or printed.
"""
from __future__ import annotations

import types

import pytest

from nodes import _otr_openrouter_backend as orb
from nodes import _otr_model_runtime as runtime


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_and_silence(monkeypatch):
    """Zero the per-run budget, clear S3 slot bindings (so env-fallback
    resolution is the default), and make backoff instant for every test."""
    orb.reset_run_budget()
    orb.clear_slot_bindings()
    orb._MANDATORY_REASONING_SLUGS.clear()
    orb._MANDATORY_REASONING_OVERRIDE_LOGGED.clear()
    orb._TEMPERATURE_UNSUPPORTED_ROUTES.clear()
    monkeypatch.setattr(orb.time, "sleep", lambda *_a, **_k: None)
    yield
    orb.reset_run_budget()
    orb.clear_slot_bindings()
    orb._MANDATORY_REASONING_SLUGS.clear()
    orb._MANDATORY_REASONING_OVERRIDE_LOGGED.clear()
    orb._TEMPERATURE_UNSUPPORTED_ROUTES.clear()


@pytest.fixture
def enabled_env(monkeypatch):
    """Minimal enabled + bound configuration."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-NEVER-LOGGED")
    monkeypatch.setenv("OTR_ENABLE_OPENROUTER", "1")
    monkeypatch.setenv("OPENROUTER_MODEL_A", "anthropic/claude-3.5-sonnet")
    monkeypatch.setenv("OPENROUTER_MODEL_B", "openai/gpt-4o")
    # Clear any inherited cost overrides so defaults apply.
    for k in (
        "OPENROUTER_MAX_TOKENS_PER_CALL",
        "OPENROUTER_MAX_TOKENS_PER_RUN",
        "OPENROUTER_A_TEMP",
        "OPENROUTER_A_MAXTOK",
        "OPENROUTER_TIMEOUT_S",
        "OPENROUTER_MAX_RETRIES",
        "OPENROUTER_REASONING_EFFORT",
    ):
        monkeypatch.delenv(k, raising=False)


def _row(context_window: int = 8192):
    return types.SimpleNamespace(
        repo_id=orb.SLOT_A_ID,
        loader_backend=orb.OPENROUTER_BACKEND_KEY,
        context_window=context_window,
    )


def _ok_result(content: str = "the decoded reply"):
    return {
        "status_code": 200,
        "json": {"choices": [{"message": {"content": content}}]},
        "text": "",
    }


@pytest.fixture
def catalog_cache(tmp_path, monkeypatch):
    """Point the catalog cache at an isolated dir and let a test seed it.

    Without this the backend reads the REAL repo cache, so a context-window
    assertion would depend on whatever OpenRouter last published.
    """
    monkeypatch.setenv("OTR_OPENROUTER_CACHE_DIR", str(tmp_path))

    def seed(models):
        import json

        (tmp_path / "openrouter_models.json").write_text(
            json.dumps(
                {
                    "schema_version": orb.CATALOG_SCHEMA_VERSION,
                    "fetched_at": "2026-07-13T12:36:57+00:00",
                    "source": "live",
                    "count": len(models),
                    "models": models,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    return seed


# ---------------------------------------------------------------------------
# Context window belongs to the SLUG, never to the static virtual row
# ---------------------------------------------------------------------------


def test_context_window_comes_from_the_resolved_slug_not_the_row(
    enabled_env, catalog_cache, monkeypatch,
):
    """The row is a stand-in for ANY bound slug; its 8192 cannot describe one.

    `aion-labs/aion-3.0-mini` advertises 131,072 tokens. Reading the window off
    the static virtual row handed it 8,192 -- a local, VRAM-shaped number that
    is simply false for a remote model.
    """
    catalog_cache([
        {"id": "aion-labs/aion-3.0-mini", "context_length": 131072},
    ])
    monkeypatch.setenv("OPENROUTER_MODEL_A", "aion-labs/aion-3.0-mini")

    entry = orb.OpenRouterBackend().load(orb.SLOT_A_ID, _row(context_window=8192))

    assert entry["context_cap"] == 131072
    assert entry["context_window"] == 131072


def test_context_window_falls_back_to_the_row_when_the_cache_is_cold(
    enabled_env, catalog_cache, monkeypatch,
):
    """An unknown window is unknown. Stay conservative and say so."""
    catalog_cache([])  # cold cache: the slug is absent
    monkeypatch.setenv("OPENROUTER_MODEL_A", "aion-labs/aion-3.0-mini")

    entry = orb.OpenRouterBackend().load(orb.SLOT_A_ID, _row(context_window=8192))

    assert entry["context_cap"] == 8192


def test_long_artifact_request_survives_intact_on_a_large_window(
    enabled_env, catalog_cache, monkeypatch,
):
    """THE 720-WORD REGRESSION.

    `original_codex56sol` P6 budgets `240 + 160*beats + 4*target_words`. At 720
    words the beat ceiling is 40, so it asks for 9,520 output tokens. Against
    the fictitious 8,192 window that request was silently reduced to whatever
    was left after the prompt, and the performance script came back cut off
    mid-JSON -- undecodable, three times, blaming the model instead of the
    budget. Against the model's REAL 131,072-token window it must reach the
    wire whole.
    """
    catalog_cache([
        {"id": "aion-labs/aion-3.0-mini", "context_length": 131072},
    ])
    monkeypatch.setenv("OPENROUTER_MODEL_A", "aion-labs/aion-3.0-mini")
    monkeypatch.setenv("OPENROUTER_A_MAXTOK", "16384")
    seen = {}
    monkeypatch.setattr(
        orb, "_post_chat_completion",
        lambda **kw: seen.update(kw) or _ok_result('{"ok": true}'),
    )

    backend = orb.OpenRouterBackend()
    entry = backend.load(orb.SLOT_A_ID, _row(context_window=8192))
    backend.generate(
        entry,
        [{"role": "user", "content": "score + manifest + truth map"}],
        temperature=0.72,
        max_new_tokens=9520,
    )

    assert seen["payload"]["max_tokens"] == 9520


def test_long_artifact_request_is_clamped_on_a_small_window(
    enabled_env, catalog_cache, monkeypatch,
):
    """The clamp is still correct when the window really IS small.

    This is the honest half of the fix: a genuinely 8k model cannot be handed a
    9,520-token artifact request, and the reduction must still happen. What was
    wrong was applying it to a model with 131k.
    """
    catalog_cache([
        {"id": "tiny/eight-k", "context_length": 8192},
    ])
    monkeypatch.setenv("OPENROUTER_MODEL_A", "tiny/eight-k")
    monkeypatch.setenv("OPENROUTER_A_MAXTOK", "16384")
    seen = {}
    monkeypatch.setattr(
        orb, "_post_chat_completion",
        lambda **kw: seen.update(kw) or _ok_result('{"ok": true}'),
    )

    backend = orb.OpenRouterBackend()
    entry = backend.load(orb.SLOT_A_ID, _row(context_window=8192))
    backend.generate(
        entry,
        [{"role": "user", "content": "score + manifest + truth map"}],
        temperature=0.72,
        max_new_tokens=9520,
    )

    assert seen["payload"]["max_tokens"] < 9520
    assert seen["payload"]["max_tokens"] <= 8192


# ---------------------------------------------------------------------------
# Registration + identity
# ---------------------------------------------------------------------------


def test_backend_registered_in_dispatch_table():
    assert orb.OPENROUTER_BACKEND_KEY in runtime.BACKENDS_BY_KEY
    assert isinstance(
        runtime.BACKENDS_BY_KEY[orb.OPENROUTER_BACKEND_KEY], orb.OpenRouterBackend
    )


def test_row_id_helpers():
    assert orb.is_openrouter_row_id("openrouter:slot-a")
    assert orb.is_openrouter_row_id("openrouter:slot-b")
    assert not orb.is_openrouter_row_id("mistralai/Mistral-Nemo-Instruct-2407")


# ---------------------------------------------------------------------------
# C3 offline-first gate
# ---------------------------------------------------------------------------


def test_disabled_when_key_missing(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OTR_ENABLE_OPENROUTER", "1")
    assert orb.openrouter_enabled() is False


def test_enabled_when_key_present_no_flag_needed(monkeypatch):
    # C6 (2026-06-29 -- "registry IS the menu"): the OTR_ENABLE_OPENROUTER opt-in
    # flag is GONE as a gate. With the API key present (creds), OpenRouter is
    # ENABLED -- no separate launch flag required.
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-x")
    monkeypatch.delenv("OTR_ENABLE_OPENROUTER", raising=False)
    assert orb.openrouter_enabled() is True


def test_enabled_when_both_set(enabled_env):
    assert orb.openrouter_enabled() is True


def test_load_raises_when_disabled(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OTR_ENABLE_OPENROUTER", raising=False)
    with pytest.raises(orb.OpenRouterConfigError):
        orb.OpenRouterBackend().load(orb.SLOT_A_ID, _row())


def test_resolve_slug_falls_back_to_recommended_when_unbound(monkeypatch):
    """S3: env is DEMOTED to a fallback. With no binding, no SLOT_DEFAULT and
    no OPENROUTER_MODEL_A, resolve_slug returns the recommended creative
    default (the plan §5 case-1 chain ends at the recommended constant before
    the config error) rather than raising."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-x")
    monkeypatch.setenv("OTR_ENABLE_OPENROUTER", "1")
    monkeypatch.delenv("OPENROUTER_MODEL_A", raising=False)
    monkeypatch.delenv("OTR_OPENROUTER_SLOT_A_DEFAULT", raising=False)
    assert orb.resolve_slug(orb.SLOT_A_ID) == orb.OPENROUTER_RECOMMENDED_CREATIVE_DEFAULT


def test_resolve_slug_chain_priority(monkeypatch):
    """A bound slug > OTR_OPENROUTER_SLOT_x_DEFAULT > OPENROUTER_MODEL_x env."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-x")
    monkeypatch.setenv("OTR_ENABLE_OPENROUTER", "1")
    monkeypatch.setenv("OPENROUTER_MODEL_A", "vendor/env-model")
    monkeypatch.delenv("OTR_OPENROUTER_SLOT_A_DEFAULT", raising=False)
    # env only
    assert orb.resolve_slug(orb.SLOT_A_ID) == "vendor/env-model"
    # SLOT_A_DEFAULT beats env
    monkeypatch.setenv("OTR_OPENROUTER_SLOT_A_DEFAULT", "vendor/slot-default")
    assert orb.resolve_slug(orb.SLOT_A_ID) == "vendor/slot-default"
    # a bound slug (the widget pick) beats everything
    orb.set_slot_bindings(slot_a="vendor/bound-model", slot_b=None)
    assert orb.resolve_slug(orb.SLOT_A_ID) == "vendor/bound-model"


def test_resolve_slug_raises_when_nothing_resolvable(monkeypatch):
    """The config error is still reachable: no binding, no env, no
    SLOT_DEFAULT, and the recommended constant forced empty -> raise."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-x")
    monkeypatch.setenv("OTR_ENABLE_OPENROUTER", "1")
    monkeypatch.delenv("OPENROUTER_MODEL_A", raising=False)
    monkeypatch.delenv("OTR_OPENROUTER_SLOT_A_DEFAULT", raising=False)
    monkeypatch.setattr(orb, "OPENROUTER_RECOMMENDED_CREATIVE_DEFAULT", "")
    with pytest.raises(orb.OpenRouterConfigError):
        orb.resolve_slug(orb.SLOT_A_ID)


# ---------------------------------------------------------------------------
# load() / unload() contract
# ---------------------------------------------------------------------------


def test_load_returns_provider_tagged_entry_no_weights(enabled_env):
    entry = orb.OpenRouterBackend().load(orb.SLOT_A_ID, _row(context_window=8192))
    assert entry["provider"] == "openrouter"
    assert entry["model_id"] == orb.SLOT_A_ID
    assert entry["slug"] == "anthropic/claude-3.5-sonnet"
    assert entry["slot_letter"] == "A"
    assert entry["context_cap"] == 8192
    # No local resources of any kind.
    assert "model" not in entry
    assert "tokenizer" not in entry


def test_unload_is_noop(enabled_env):
    entry = orb.OpenRouterBackend().load(orb.SLOT_A_ID, _row())
    assert orb.OpenRouterBackend().unload(entry) is None


def test_load_does_not_touch_network(enabled_env, monkeypatch):
    calls = []
    monkeypatch.setattr(
        orb, "_post_chat_completion", lambda **kw: calls.append(kw) or _ok_result()
    )
    orb.OpenRouterBackend().load(orb.SLOT_A_ID, _row())
    assert calls == [], "load() must not make any network call"


# ---------------------------------------------------------------------------
# generate() happy path
# ---------------------------------------------------------------------------


def test_generate_happy_path(enabled_env, monkeypatch):
    seen = {}

    def fake_post(*, base_url, api_key, payload, timeout_s):
        seen["base_url"] = base_url
        seen["payload"] = payload
        seen["api_key_present"] = bool(api_key)
        return _ok_result("hello from sonnet")

    monkeypatch.setattr(orb, "_post_chat_completion", fake_post)
    backend = orb.OpenRouterBackend()
    entry = backend.load(orb.SLOT_A_ID, _row())
    out = backend.generate(
        entry,
        [{"role": "user", "content": "hi"}],
        temperature=0.6,
        max_new_tokens=128,
    )
    assert out == "hello from sonnet"
    assert seen["payload"]["model"] == "anthropic/claude-3.5-sonnet"
    # 128 is below the remote min-output floor and is bumped up so a free-form
    # remote reply isn't truncated mid-JSON. max_tokens is a ceiling.
    #
    # This asserted 1024 until 2026-07-14, which quietly pinned the BUG:
    # DEFAULT_REASONING_EFFORT is "low", so reasoning is ON for this call, and
    # `max_tokens` bounds reasoning + content TOGETHER with the reasoning emitted
    # FIRST. At 1024 a reasoning model spends the whole budget thinking and is cut
    # before it writes a content token (finish_reason=length). So the floor is
    # reasoning-aware and the happy path gets the reasoning floor.
    assert seen["payload"]["reasoning_effort"] == "low"
    assert seen["payload"]["max_tokens"] == orb.DEFAULT_MIN_OUTPUT_TOKENS_REASONING
    assert seen["payload"]["temperature"] == 0.6
    assert seen["api_key_present"] is True


def test_generate_passes_response_format_when_given(enabled_env, monkeypatch):
    seen = {}
    monkeypatch.setattr(
        orb, "_post_chat_completion",
        lambda **kw: seen.update(kw) or _ok_result('{"ok": true}'),
    )
    backend = orb.OpenRouterBackend()
    entry = backend.load(orb.SLOT_B_ID, _row())
    rf = {"type": "json_schema", "json_schema": {"name": "X", "schema": {}}}
    backend.generate(
        entry, [{"role": "user", "content": "x"}],
        temperature=0.1, max_new_tokens=64, response_format=rf,
    )
    assert seen["payload"]["response_format"] == rf
    assert seen["payload"]["model"] == "openai/gpt-4o"  # slot B slug


def test_generate_passes_grammar_when_given(enabled_env, monkeypatch):
    # A (2026-06-04): a GBNF grammar threads into the payload (the local
    # llama-server lane), with require_parameters set so a backend that
    # ignores `grammar` is filtered out rather than left unconstrained.
    seen = {}
    monkeypatch.setattr(
        orb, "_post_chat_completion",
        lambda **kw: seen.update(kw) or _ok_result("a_b\nc_d\ne_f\ng_h\ni_j"),
    )
    backend = orb.OpenRouterBackend()
    entry = backend.load(orb.SLOT_B_ID, _row())
    gbnf = 'root ::= line\nline ::= word "_" word\nword ::= [a-z]+'
    backend.generate(
        entry, [{"role": "user", "content": "x"}],
        temperature=0.6, max_new_tokens=80, grammar=gbnf,
    )
    assert seen["payload"]["grammar"] == gbnf
    assert seen["payload"]["provider"]["require_parameters"] is True


def test_generate_omits_grammar_when_absent(enabled_env, monkeypatch):
    seen = {}
    monkeypatch.setattr(
        orb, "_post_chat_completion",
        lambda **kw: seen.update(kw) or _ok_result(),
    )
    backend = orb.OpenRouterBackend()
    entry = backend.load(orb.SLOT_A_ID, _row())
    backend.generate(entry, [{"role": "user", "content": "x"}],
                     temperature=0.5, max_new_tokens=50)
    assert "grammar" not in seen["payload"]


def test_make_generate_fn_threads_grammar_and_marker(enabled_env, monkeypatch):
    # The closure threads a per-call grammar to backend.generate and
    # advertises grammar support via the _otr_supports_grammar marker.
    captured = {}

    def fake_generate(self, model, messages, *, temperature=None,
                      max_new_tokens=None, stop=None, response_format=None,
                      grammar=None, **_):
        captured["grammar"] = grammar
        return "ok"

    monkeypatch.setattr(orb.OpenRouterBackend, "generate", fake_generate)
    entry = orb.OpenRouterBackend().load(orb.SLOT_A_ID, _row())
    fn = orb.make_openrouter_generate_fn(entry)
    assert getattr(fn, "_otr_supports_grammar", False) is True
    fn([{"role": "user", "content": "x"}], temperature=0.6,
       max_new_tokens=80, grammar="GBNF-HERE")
    assert captured["grammar"] == "GBNF-HERE"


def test_small_max_new_tokens_is_floored(enabled_env, monkeypatch):
    """The writer's local grammar-era per-call budget (~200) must be floored
    for the remote path so a free-form model isn't truncated mid-JSON
    (the cast-JSON truncation bug, 2026-05-31). max_tokens is a ceiling, so
    flooring it costs nothing on short replies."""
    seen = {}
    monkeypatch.setattr(
        orb, "_post_chat_completion",
        lambda **kw: seen.update(kw) or _ok_result(),
    )
    backend = orb.OpenRouterBackend()
    entry = backend.load(orb.SLOT_A_ID, _row())
    backend.generate(entry, [{"role": "user", "content": "x"}],
                     temperature=0.5, max_new_tokens=50)
    assert seen["payload"]["max_tokens"] >= 1024


def test_floor_overridable_via_env(enabled_env, monkeypatch):
    # Reasoning OFF isolates the BASE floor, which is what this test is about.
    monkeypatch.setenv("OPENROUTER_REASONING_EFFORT", "none")
    monkeypatch.setenv("OPENROUTER_MIN_OUTPUT_TOKENS", "1500")
    seen = {}
    monkeypatch.setattr(
        orb, "_post_chat_completion",
        lambda **kw: seen.update(kw) or _ok_result(),
    )
    backend = orb.OpenRouterBackend()
    entry = backend.load(orb.SLOT_A_ID, _row())
    backend.generate(entry, [{"role": "user", "content": "x"}],
                     temperature=0.5, max_new_tokens=50)
    assert seen["payload"]["max_tokens"] == 1500


# ---------------------------------------------------------------------------
# The reasoning floor -- `max_tokens` bounds reasoning + content TOGETHER
# ---------------------------------------------------------------------------


def _seen_max_tokens(monkeypatch, *, max_new_tokens=50):
    seen = {}
    monkeypatch.setattr(
        orb, "_post_chat_completion",
        lambda **kw: seen.update(kw) or _ok_result(),
    )
    backend = orb.OpenRouterBackend()
    entry = backend.load(orb.SLOT_A_ID, _row(context_window=131072))
    backend.generate(entry, [{"role": "user", "content": "x"}],
                     temperature=0.5, max_new_tokens=max_new_tokens)
    return seen["payload"]


def test_reasoning_model_gets_room_to_think_AND_answer(enabled_env, monkeypatch):
    """A reasoning model must not spend its whole budget on the preamble.

    THE BUG (live 2026-07-14, 420w public_domain_story): `max_tokens` bounds
    reasoning tokens and content tokens TOGETHER, and the hidden reasoning is
    emitted FIRST. The announcer's news-coda bridge asks for ~150 tokens (a budget
    sized for the LOCAL grammar-constrained path), the old floor lifted that to
    1024, and aion-3.0-mini -- a MANDATORY-reasoning model -- burned all 1024
    thinking and was cut before writing a single content token. Both attempts hit
    finish_reason=length, the bridge failed, and the no-fallback rip correctly
    aborted the episode rather than ship canned text.

    The OUTPUT CAP had already learned this (8192 -> 16384, R3 2026-06-22) and it
    protected the BIG calls. The FLOOR never did, so the SMALL calls kept dying.
    """
    payload = _seen_max_tokens(monkeypatch)
    assert payload["reasoning_effort"] == "low"          # on by default
    assert payload["max_tokens"] >= orb.DEFAULT_MIN_OUTPUT_TOKENS_REASONING
    # Room for the preamble AND the answer -- strictly more than a budget the
    # preamble alone can swallow.
    assert payload["max_tokens"] > orb.DEFAULT_MIN_OUTPUT_TOKENS


def test_reasoning_off_keeps_the_lean_base_floor(enabled_env, monkeypatch):
    """No preamble to pay for -> no reason to inflate the budget."""
    monkeypatch.setenv("OPENROUTER_REASONING_EFFORT", "none")
    payload = _seen_max_tokens(monkeypatch)
    assert "reasoning_effort" not in payload or payload["reasoning_effort"] == "none"
    assert payload["max_tokens"] == orb.DEFAULT_MIN_OUTPUT_TOKENS


def test_reasoning_floor_overridable_via_env(enabled_env, monkeypatch):
    monkeypatch.setenv("OPENROUTER_MIN_OUTPUT_TOKENS_REASONING", "6000")
    payload = _seen_max_tokens(monkeypatch)
    assert payload["max_tokens"] == 6000


def test_reasoning_floor_never_lowers_a_bigger_request(enabled_env, monkeypatch):
    """The floor is a FLOOR. A call that already asks for more keeps its budget."""
    payload = _seen_max_tokens(
        monkeypatch,
        max_new_tokens=orb.DEFAULT_MIN_OUTPUT_TOKENS_REASONING + 5000,
    )
    assert payload["max_tokens"] == orb.DEFAULT_MIN_OUTPUT_TOKENS_REASONING + 5000


def test_reasoning_floor_degrades_it_never_aborts_a_survivable_call(
        enabled_env, monkeypatch):
    """The reasoning floor is a DESIRED minimum, not a HARD one.

    A long prompt on a small-context model can leave less room than the reasoning
    floor (8192 cap minus a ~7000-token prompt = 1192 tokens). That call must still
    RUN with what is left -- degraded -- not refuse to start. The HARD minimum (the
    threshold below which fit_output_tokens raises context-overflow) stays the lean
    BASE floor.

    Caught by the known-fail guard on 2026-07-14: the first cut of the reasoning
    floor passed it as `min_output_tokens`, which turned this survivable call into
    an abort.
    """
    seen = {}
    monkeypatch.setattr(
        orb, "_post_chat_completion",
        lambda **kw: seen.update(kw) or _ok_result(),
    )
    backend = orb.OpenRouterBackend()
    entry = backend.load(orb.SLOT_A_ID, _row(context_window=8192))
    backend.generate(
        entry,
        [{"role": "user", "content": "x" * 28000}],   # ~7000 prompt tokens
        temperature=0.2,
        max_new_tokens=9520,
    )
    # It ran, and took the room that was actually left.
    fitted = seen["payload"]["max_tokens"]
    assert 0 < fitted < orb.DEFAULT_MIN_OUTPUT_TOKENS_REASONING


def test_strict_message_budget_bypasses_overridden_floor(enabled_env, monkeypatch):
    monkeypatch.setenv("OPENROUTER_MIN_OUTPUT_TOKENS", "1500")
    seen = {}
    monkeypatch.setattr(
        orb, "_post_chat_completion",
        lambda **kw: seen.update(kw) or _ok_result(),
    )

    class StrictMessages(list):
        _otr_strict_remote_output_budget = True

    backend = orb.OpenRouterBackend()
    entry = backend.load(orb.SLOT_A_ID, _row())
    backend.generate(
        entry, StrictMessages([{"role": "user", "content": "x"}]),
        temperature=0.5, max_new_tokens=1024,
    )
    assert seen["payload"]["max_tokens"] == 1024


def test_max_tokens_clamped_to_cap(enabled_env, monkeypatch):
    monkeypatch.setenv("OPENROUTER_A_MAXTOK", "256")
    seen = {}
    monkeypatch.setattr(
        orb, "_post_chat_completion",
        lambda **kw: seen.update(kw) or _ok_result(),
    )
    backend = orb.OpenRouterBackend()
    entry = backend.load(orb.SLOT_A_ID, _row())
    backend.generate(
        entry, [{"role": "user", "content": "x"}],
        temperature=0.5, max_new_tokens=9999,
    )
    assert seen["payload"]["max_tokens"] == 256


def test_generate_sends_reasoning_effort_when_env_set(enabled_env, monkeypatch):
    # gemma-4 lane (2026-06-04): OPENROUTER_REASONING_EFFORT=none tells Ollama's
    # /v1 to suppress the <think> preamble so a reasoning model emits the
    # structured answer directly instead of spending the output budget on
    # reasoning (-> finish_reason=length -> unparseable JSON).
    monkeypatch.setenv("OPENROUTER_REASONING_EFFORT", "none")
    seen = {}
    monkeypatch.setattr(
        orb, "_post_chat_completion",
        lambda **kw: seen.update(kw) or _ok_result(),
    )
    backend = orb.OpenRouterBackend()
    entry = backend.load(orb.SLOT_A_ID, _row())
    backend.generate(entry, [{"role": "user", "content": "x"}],
                     temperature=0.5, max_new_tokens=64)
    assert seen["payload"]["reasoning_effort"] == "none"


def test_generate_uses_low_reasoning_effort_when_env_unset(enabled_env, monkeypatch):
    # R3 (2026-06-22): the frontier-writer DEFAULT is reasoning_effort="low"
    # (the live re-measure HALVED critic flatness). UNSET env -> "low".
    monkeypatch.delenv("OPENROUTER_REASONING_EFFORT", raising=False)
    seen = {}
    monkeypatch.setattr(
        orb, "_post_chat_completion",
        lambda **kw: seen.update(kw) or _ok_result(),
    )
    backend = orb.OpenRouterBackend()
    entry = backend.load(orb.SLOT_A_ID, _row())
    backend.generate(entry, [{"role": "user", "content": "x"}],
                     temperature=0.5, max_new_tokens=64)
    assert seen["payload"]["reasoning_effort"] == orb.DEFAULT_REASONING_EFFORT
    assert seen["payload"]["reasoning_effort"] == "low"


def test_generate_uses_lowest_catalog_effort_when_reasoning_is_mandatory(
    enabled_env, monkeypatch,
):
    monkeypatch.setenv("OPENROUTER_REASONING_EFFORT", "none")
    monkeypatch.setattr(
        orb,
        "_cached_model",
        lambda _slug: {
            "reasoning": {
                "mandatory": True,
                "supported_efforts": ["high", "medium", "low", "minimal"],
            }
        },
    )
    seen = {}
    monkeypatch.setattr(
        orb, "_post_chat_completion", lambda **kw: seen.update(kw) or _ok_result()
    )

    backend = orb.OpenRouterBackend()
    entry = backend.load(orb.SLOT_A_ID, _row())
    backend.generate(
        entry, [{"role": "user", "content": "x"}],
        temperature=0.5, max_new_tokens=64,
    )

    assert seen["payload"]["reasoning_effort"] == "minimal"


# ---------------------------------------------------------------------------
# C6 cost-ceiling abort -- proven with a mocked token counter
# ---------------------------------------------------------------------------


def test_cost_ceiling_per_call_aborts_before_network(enabled_env, monkeypatch):
    sent = {"called": False}
    monkeypatch.setattr(
        orb, "_post_chat_completion",
        lambda **kw: sent.__setitem__("called", True) or _ok_result(),
    )
    # Mocked token counter forces a huge estimate over the per-call cap.
    monkeypatch.setattr(orb, "_estimate_request_tokens", lambda *_a, **_k: 10_000_000)
    backend = orb.OpenRouterBackend()
    entry = backend.load(orb.SLOT_A_ID, _row())
    with pytest.raises(orb.OpenRouterCostCeilingError):
        backend.generate(
            entry, [{"role": "user", "content": "x"}],
            temperature=0.5, max_new_tokens=64,
        )
    assert sent["called"] is False, "no request may be sent when the ceiling trips"


def test_cost_ceiling_per_run_accumulates(enabled_env, monkeypatch):
    monkeypatch.setenv("OPENROUTER_MAX_TOKENS_PER_CALL", "1000")
    monkeypatch.setenv("OPENROUTER_MAX_TOKENS_PER_RUN", "1500")
    monkeypatch.setattr(orb, "_estimate_request_tokens", lambda *_a, **_k: 800)
    monkeypatch.setattr(orb, "_post_chat_completion", lambda **kw: _ok_result())
    backend = orb.OpenRouterBackend()
    entry = backend.load(orb.SLOT_A_ID, _row())
    msgs = [{"role": "user", "content": "x"}]
    # 1st call (800) ok; run total -> 800.
    backend.generate(entry, msgs, temperature=0.5, max_new_tokens=10)
    # 2nd call (800) would push run total to 1600 > 1500 -> abort.
    with pytest.raises(orb.OpenRouterCostCeilingError):
        backend.generate(entry, msgs, temperature=0.5, max_new_tokens=10)


def test_reset_run_budget_clears_accumulator(enabled_env, monkeypatch):
    monkeypatch.setenv("OPENROUTER_MAX_TOKENS_PER_RUN", "1500")
    monkeypatch.setattr(orb, "_estimate_request_tokens", lambda *_a, **_k: 800)
    monkeypatch.setattr(orb, "_post_chat_completion", lambda **kw: _ok_result())
    backend = orb.OpenRouterBackend()
    entry = backend.load(orb.SLOT_A_ID, _row())
    msgs = [{"role": "user", "content": "x"}]
    backend.generate(entry, msgs, temperature=0.5, max_new_tokens=10)
    orb.reset_run_budget()
    # After reset the accumulator is 0, so another 800-token call is fine.
    backend.generate(entry, msgs, temperature=0.5, max_new_tokens=10)


# ---------------------------------------------------------------------------
# C5 retry ladder -> clean abort (no half-remote fall-back)
# ---------------------------------------------------------------------------


def test_transient_then_success(enabled_env, monkeypatch):
    seq = [
        {"status_code": 503, "json": None, "text": "busy"},
        _ok_result("recovered"),
    ]
    monkeypatch.setattr(orb, "_post_chat_completion", lambda **kw: seq.pop(0))
    backend = orb.OpenRouterBackend()
    entry = backend.load(orb.SLOT_A_ID, _row())
    out = backend.generate(entry, [{"role": "user", "content": "x"}],
                           temperature=0.5, max_new_tokens=16)
    assert out == "recovered"
    assert seq == []  # both responses consumed


def test_retry_exhaustion_aborts(enabled_env, monkeypatch):
    monkeypatch.setenv("OPENROUTER_MAX_RETRIES", "2")
    attempts = {"n": 0}

    def always_503(**kw):
        attempts["n"] += 1
        return {"status_code": 503, "json": None, "text": "still busy"}

    monkeypatch.setattr(orb, "_post_chat_completion", always_503)
    backend = orb.OpenRouterBackend()
    entry = backend.load(orb.SLOT_A_ID, _row())
    with pytest.raises(orb.OpenRouterCallFailedError):
        backend.generate(entry, [{"role": "user", "content": "x"}],
                         temperature=0.5, max_new_tokens=16)
    assert attempts["n"] == 3  # retries(2) + 1


def test_non_retryable_status_aborts_immediately(enabled_env, monkeypatch):
    attempts = {"n": 0}

    def unauthorized(**kw):
        attempts["n"] += 1
        return {"status_code": 401, "json": {"error": {"message": "bad key"}},
                "text": ""}

    monkeypatch.setattr(orb, "_post_chat_completion", unauthorized)
    backend = orb.OpenRouterBackend()
    entry = backend.load(orb.SLOT_A_ID, _row())
    with pytest.raises(orb.OpenRouterCallFailedError):
        backend.generate(entry, [{"role": "user", "content": "x"}],
                         temperature=0.5, max_new_tokens=16)
    assert attempts["n"] == 1  # no retries on a 401


def test_stale_cache_learns_mandatory_reasoning_from_exact_400(
    enabled_env, monkeypatch,
):
    monkeypatch.setenv("OPENROUTER_REASONING_EFFORT", "none")
    monkeypatch.setenv("OPENROUTER_MAX_RETRIES", "0")
    sent_efforts = []

    def mandatory_then_ok(**kw):
        effort = kw["payload"].get("reasoning_effort")
        sent_efforts.append(effort)
        if effort == "none":
            return {
                "status_code": 400,
                "json": {
                    "error": {
                        "message": (
                            "Reasoning is mandatory for this endpoint and "
                            "cannot be disabled."
                        )
                    }
                },
                "text": "",
            }
        return _ok_result("recovered")

    monkeypatch.setattr(orb, "_post_chat_completion", mandatory_then_ok)
    backend = orb.OpenRouterBackend()
    entry = backend.load(orb.SLOT_A_ID, _row())

    assert backend.generate(
        entry, [{"role": "user", "content": "x"}],
        temperature=0.5, max_new_tokens=16,
    ) == "recovered"
    assert sent_efforts == ["none", "low"]

    sent_efforts.clear()
    assert backend.generate(
        entry, [{"role": "user", "content": "x"}],
        temperature=0.5, max_new_tokens=16,
    ) == "recovered"
    assert sent_efforts == ["low"]


def test_provider_temperature_deprecation_retries_without_it_and_remembers(
    enabled_env, monkeypatch,
):
    """A structured-output provider can reject a catalog-advertised knob.

    The live Opus 4.8 JSON route returned a generic OpenRouter error whose
    nested Azure detail said ``temperature`` was deprecated.  Preserve the
    same model and response format, remove only that optional parameter, and
    learn the capability for later calls in this process.
    """
    monkeypatch.setenv("OPENROUTER_MAX_RETRIES", "0")
    sent_temperatures = []

    def deprecated_then_ok(**kw):
        temperature = kw["payload"].get("temperature")
        sent_temperatures.append(temperature)
        if temperature is not None:
            return {
                "status_code": 400,
                "json": {
                    "error": {
                        "message": "Provider returned error",
                        "metadata": {
                            "raw": (
                                '{"type":"error","error":{'
                                '"type":"invalid_request_error",'
                                '"message":"`temperature` is deprecated '
                                'for this model."}}'
                            ),
                        },
                    },
                },
                "text": "",
            }
        return _ok_result("recovered")

    monkeypatch.setattr(orb, "_post_chat_completion", deprecated_then_ok)
    backend = orb.OpenRouterBackend()
    entry = backend.load(orb.SLOT_A_ID, _row())

    assert backend.generate(
        entry, [{"role": "user", "content": "x"}],
        temperature=0.85, max_new_tokens=16,
    ) == "recovered"
    assert sent_temperatures == [0.85, None]

    sent_temperatures.clear()
    assert backend.generate(
        entry, [{"role": "user", "content": "x"}],
        temperature=0.4, max_new_tokens=16,
    ) == "recovered"
    assert sent_temperatures == [None]


def test_structured_temperature_retry_preserves_route_and_learns_only_it(
    enabled_env, monkeypatch,
):
    """Retry only the rejected JSON route and preserve its other controls."""
    monkeypatch.setenv("OPENROUTER_MAX_RETRIES", "0")
    monkeypatch.setenv("OPENROUTER_REASONING_EFFORT", "none")
    sent_payloads = []

    def deprecated_then_ok(**kw):
        payload = dict(kw["payload"])
        sent_payloads.append(payload)
        if payload.get("temperature") is not None:
            return {
                "status_code": 400,
                "json": {
                    "error": {
                        "message": "Provider returned error",
                        "metadata": {
                            "raw": (
                                '{"error":{"message":"temperature is '
                                'deprecated for this model"}}'
                            ),
                        },
                    },
                },
                "text": "",
            }
        return _ok_result("recovered")

    monkeypatch.setattr(orb, "_post_chat_completion", deprecated_then_ok)
    backend = orb.OpenRouterBackend()
    entry = backend.load(orb.SLOT_A_ID, _row())
    response_format = {"type": "json_object"}

    assert backend.generate(
        entry,
        [{"role": "user", "content": "x"}],
        temperature=0.85,
        max_new_tokens=16,
        response_format=response_format,
    ) == "recovered"
    assert [payload.get("temperature") for payload in sent_payloads] == [0.85, None]
    for payload in sent_payloads:
        assert payload["model"] == "anthropic/claude-3.5-sonnet"
        assert payload["response_format"] == response_format
        assert payload["provider"] == {"require_parameters": True}
        assert payload["reasoning_effort"] == "none"

    sent_payloads.clear()
    assert backend.generate(
        entry,
        [{"role": "user", "content": "x"}],
        temperature=0.4,
        max_new_tokens=16,
        response_format=response_format,
    ) == "recovered"
    assert [payload.get("temperature") for payload in sent_payloads] == [None]

    # The learning is restricted to the structured route that actually failed.
    sent_payloads.clear()
    assert backend.generate(
        entry,
        [{"role": "user", "content": "x"}],
        temperature=0.4,
        max_new_tokens=16,
    ) == "recovered"
    assert [payload.get("temperature") for payload in sent_payloads] == [0.4, None]


def test_failed_temperature_retry_does_not_learn_a_route_capability(
    enabled_env, monkeypatch,
):
    monkeypatch.setenv("OPENROUTER_MAX_RETRIES", "0")
    calls = []

    def deprecated_then_unrelated_400(**kw):
        payload = dict(kw["payload"])
        calls.append(payload)
        if len(calls) == 1:
            return {
                "status_code": 400,
                "json": {
                    "error": {
                        "message": "Provider returned error",
                        "metadata": {
                            "raw": (
                                '{"error":{"message":"temperature is '
                                'not supported for this model"}}'
                            ),
                        },
                    },
                },
                "text": "",
            }
        return {
            "status_code": 400,
            "json": {"error": {"message": "The JSON schema is invalid."}},
            "text": "",
        }

    monkeypatch.setattr(
        orb, "_post_chat_completion", deprecated_then_unrelated_400,
    )
    backend = orb.OpenRouterBackend()
    entry = backend.load(orb.SLOT_A_ID, _row())

    with pytest.raises(orb.OpenRouterCallFailedError, match="JSON schema"):
        backend.generate(
            entry,
            [{"role": "user", "content": "x"}],
            temperature=0.85,
            max_new_tokens=16,
            response_format={"type": "json_object"},
        )

    assert [payload.get("temperature") for payload in calls] == [0.85, None]
    assert not orb._TEMPERATURE_UNSUPPORTED_ROUTES

def test_generic_nested_provider_400_does_not_drop_temperature(
    enabled_env, monkeypatch,
):
    monkeypatch.setenv("OPENROUTER_MAX_RETRIES", "0")
    attempts = {"n": 0}

    def invalid_schema(**kw):
        attempts["n"] += 1
        return {
            "status_code": 400,
            "json": {
                "error": {
                    "message": "Provider returned error",
                    "metadata": {
                        "raw": (
                            '{"type":"error","error":{'
                            '"type":"invalid_request_error",'
                            '"message":"The JSON schema is invalid."}}'
                        ),
                    },
                },
            },
            "text": "",
        }

    monkeypatch.setattr(orb, "_post_chat_completion", invalid_schema)
    backend = orb.OpenRouterBackend()
    entry = backend.load(orb.SLOT_A_ID, _row())

    with pytest.raises(
        orb.OpenRouterCallFailedError,
        match="The JSON schema is invalid",
    ):
        backend.generate(
            entry, [{"role": "user", "content": "x"}],
            temperature=0.85, max_new_tokens=16,
        )
    assert attempts["n"] == 1
    assert not orb._TEMPERATURE_UNSUPPORTED_ROUTES


def test_transport_exception_is_retried_then_aborts(enabled_env, monkeypatch):
    monkeypatch.setenv("OPENROUTER_MAX_RETRIES", "1")
    attempts = {"n": 0}

    def boom(**kw):
        attempts["n"] += 1
        raise TimeoutError("simulated request timeout")

    monkeypatch.setattr(orb, "_post_chat_completion", boom)
    backend = orb.OpenRouterBackend()
    entry = backend.load(orb.SLOT_A_ID, _row())
    with pytest.raises(orb.OpenRouterCallFailedError):
        backend.generate(entry, [{"role": "user", "content": "x"}],
                         temperature=0.5, max_new_tokens=16)
    assert attempts["n"] == 2  # retries(1) + 1


def test_empty_choices_aborts(enabled_env, monkeypatch):
    monkeypatch.setattr(
        orb, "_post_chat_completion",
        lambda **kw: {"status_code": 200, "json": {"choices": []}, "text": ""},
    )
    backend = orb.OpenRouterBackend()
    entry = backend.load(orb.SLOT_A_ID, _row())
    with pytest.raises(orb.OpenRouterCallFailedError):
        backend.generate(entry, [{"role": "user", "content": "x"}],
                         temperature=0.5, max_new_tokens=16)


# ---------------------------------------------------------------------------
# C9 -- the API key never leaks into errors
# ---------------------------------------------------------------------------


def test_key_not_in_call_failed_error(enabled_env, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-SECRET-DO-NOT-LEAK")
    monkeypatch.setattr(
        orb, "_post_chat_completion",
        lambda **kw: {"status_code": 401, "json": {"error": {"message": "no"}},
                      "text": ""},
    )
    backend = orb.OpenRouterBackend()
    entry = backend.load(orb.SLOT_A_ID, _row())
    try:
        backend.generate(entry, [{"role": "user", "content": "x"}],
                         temperature=0.5, max_new_tokens=16)
    except orb.OpenRouterCallFailedError as exc:
        assert "sk-SECRET-DO-NOT-LEAK" not in str(exc)
    else:
        pytest.fail("expected OpenRouterCallFailedError")


# ---------------------------------------------------------------------------
# Thinking-mode reasoning strip (BUG-306 / BUG-LOCAL-308 family)
# ---------------------------------------------------------------------------


def test_strip_reasoning_removes_balanced_think_block():
    raw = '<think>plan the cast then the arc</think>\n{"ok": true}'
    assert orb._strip_reasoning_tags(raw) == '{"ok": true}'


def test_strip_reasoning_handles_dangling_close_from_ollama_prefill():
    # Ollama pre-fills the opening <think> in the chat template, so the
    # completion carries only the closing tag.
    raw = 'reasoning about the news beat...</think>\n\n{"cast": []}'
    assert orb._strip_reasoning_tags(raw) == '{"cast": []}'


def test_strip_reasoning_removes_multiple_blocks():
    raw = '<think>a</think>FOO<think>b</think>BAR'
    assert orb._strip_reasoning_tags(raw) == 'FOOBAR'


def test_strip_reasoning_is_case_insensitive_and_multiline():
    raw = '<THINK>\nline1\nline2\n</Think>\nanswer'
    assert orb._strip_reasoning_tags(raw) == 'answer'


def test_strip_reasoning_noop_for_plain_text():
    raw = '{"title": "The 3:10 to Yuma", "scene": "a dusty depot"}'
    assert orb._strip_reasoning_tags(raw) == raw


def test_strip_reasoning_preserves_unrelated_angle_brackets():
    # A '<' that is not a think/channel marker must survive untouched.
    raw = 'if x < 3 and y > 1 then go'
    assert orb._strip_reasoning_tags(raw) == raw


def test_strip_reasoning_keeps_final_harmony_channel():
    raw = ('<|channel|>analysis<|message|>I should think first<|end|>'
           '<|channel|>final<|message|>{"line": "Action!"}')
    assert orb._strip_reasoning_tags(raw) == '{"line": "Action!"}'


def test_strip_reasoning_handles_empty_and_none():
    assert orb._strip_reasoning_tags("") == ""
    assert orb._strip_reasoning_tags(None) is None


def test_generate_strips_think_block_end_to_end(enabled_env, monkeypatch):
    # The strip must apply on the live generate() path, not just the helper.
    dirty = '<think>weigh the angles</think>{"verdict": "go"}'
    monkeypatch.setattr(orb, "_post_chat_completion",
                        lambda **kw: _ok_result(dirty))
    backend = orb.OpenRouterBackend()
    entry = backend.load(orb.SLOT_A_ID, _row())
    out = backend.generate(entry, [{"role": "user", "content": "x"}],
                           temperature=0.5, max_new_tokens=16)
    assert out == '{"verdict": "go"}'


def test_generate_aborts_when_only_reasoning_remains(enabled_env, monkeypatch):
    # If the model emits ONLY a <think> block (no answer), stripping yields
    # empty -> clean abort, not an empty string handed to the JSON parser.
    monkeypatch.setattr(orb, "_post_chat_completion",
                        lambda **kw: _ok_result("<think>just musing</think>"))
    backend = orb.OpenRouterBackend()
    entry = backend.load(orb.SLOT_A_ID, _row())
    with pytest.raises(orb.OpenRouterCallFailedError):
        backend.generate(entry, [{"role": "user", "content": "x"}],
                         temperature=0.5, max_new_tokens=16)
