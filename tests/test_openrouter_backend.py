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
    monkeypatch.setattr(orb.time, "sleep", lambda *_a, **_k: None)
    yield
    orb.reset_run_budget()
    orb.clear_slot_bindings()


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


def test_disabled_when_flag_unset(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-x")
    monkeypatch.delenv("OTR_ENABLE_OPENROUTER", raising=False)
    assert orb.openrouter_enabled() is False


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
    # 128 is below the remote min-output floor (1024) and is bumped up so a
    # free-form remote reply isn't truncated mid-JSON. max_tokens is a ceiling.
    assert seen["payload"]["max_tokens"] == 1024
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


def test_generate_omits_reasoning_effort_when_env_unset(enabled_env, monkeypatch):
    # Default: no reasoning-control field, so non-thinking models and
    # OpenRouter-proper payloads stay byte-identical.
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
    assert "reasoning_effort" not in seen["payload"]


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
