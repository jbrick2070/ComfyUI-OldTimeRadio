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
    """Zero the per-run budget and make backoff instant for every test."""
    orb.reset_run_budget()
    monkeypatch.setattr(orb.time, "sleep", lambda *_a, **_k: None)
    yield
    orb.reset_run_budget()


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


def test_resolve_slug_raises_when_unbound(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-x")
    monkeypatch.setenv("OTR_ENABLE_OPENROUTER", "1")
    monkeypatch.delenv("OPENROUTER_MODEL_A", raising=False)
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
