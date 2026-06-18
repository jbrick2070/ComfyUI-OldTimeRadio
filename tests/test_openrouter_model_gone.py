"""Deleted-model resilience (2026-06-18, ChatGPT-reviewed).

A model the operator picked can be DELETED from OpenRouter -> a deletion-class
404 ("No endpoints found"). That is a *renderability* failure, not the quality
swap C5 forbids, so the dispatch layer falls over to the slot's REMOTE fallback
exactly ONCE, loudly. Every other failure (401/403/422/429/5xx/timeout, and a
bodyless 404) still aborts. Remote->remote only -- never local.

The only network seam (`_post_chat_completion`) is mocked, so no network/secret.
"""
from __future__ import annotations

import types

import pytest

from nodes import _otr_openrouter_backend as orb


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    orb.reset_run_budget()
    orb.clear_slot_bindings()
    monkeypatch.setattr(orb.time, "sleep", lambda *_a, **_k: None)
    yield
    orb.reset_run_budget()
    orb.clear_slot_bindings()


@pytest.fixture
def enabled_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-NEVER-LOGGED")
    monkeypatch.setenv("OTR_ENABLE_OPENROUTER", "1")
    monkeypatch.setenv("OPENROUTER_MODEL_A", "anthropic/claude-opus-4.8")
    monkeypatch.setenv("OPENROUTER_MODEL_B", "deepseek/deepseek-v4-pro")
    for k in ("OPENROUTER_MAX_TOKENS_PER_CALL", "OPENROUTER_MAX_TOKENS_PER_RUN",
              "OPENROUTER_TIMEOUT_S", "OPENROUTER_MAX_RETRIES",
              "OTR_OPENROUTER_SLOT_A_DEFAULT", "OTR_OPENROUTER_SLOT_B_DEFAULT"):
        monkeypatch.delenv(k, raising=False)


def _row(cw: int = 8192):
    return types.SimpleNamespace(
        repo_id=orb.SLOT_A_ID, loader_backend=orb.OPENROUTER_BACKEND_KEY,
        context_window=cw,
    )


def _ok(content="ok reply"):
    return {"status_code": 200, "json": {"choices": [{"message": {"content": content}}]}, "text": ""}


def _gone(slug):
    msg = f"No endpoints found for {slug}."
    return {"status_code": 404, "json": {"error": {"message": msg}}, "text": msg}


# --- unit: deletion-class classifier ----------------------------------------

def test_is_model_gone_true_on_404_with_marker():
    assert orb.is_model_gone_error(404, "No endpoints found for vendor/x")
    assert orb.is_model_gone_error(404, "model does not exist")


def test_is_model_gone_false_on_bodyless_or_generic_404():
    assert not orb.is_model_gone_error(404, "")
    assert not orb.is_model_gone_error(404, "rate limited, slow down")


def test_is_model_gone_false_on_non_404():
    # 401/403/422/429/5xx never count, even if the body says "not found".
    for status in (401, 403, 422, 429, 500, 503):
        assert not orb.is_model_gone_error(status, "not found")


# --- unit: fallback chain (skips the dead slug, remote-only) -----------------

def test_fallback_skips_dead_and_returns_env(enabled_env):
    fb = orb.fallback_slug_for_slot("A", exclude="vendor/dead")
    assert fb == "anthropic/claude-opus-4.8"


def test_fallback_falls_through_when_env_is_the_dead_one(enabled_env, monkeypatch):
    monkeypatch.setenv("OPENROUTER_MODEL_A", "vendor/dead")
    # env == dead -> skip to the recommended creative default.
    fb = orb.fallback_slug_for_slot("A", exclude="vendor/dead")
    assert fb == orb.OPENROUTER_RECOMMENDED_CREATIVE_DEFAULT


def test_fallback_none_when_every_candidate_is_dead(enabled_env, monkeypatch):
    monkeypatch.setenv("OPENROUTER_MODEL_A", "vendor/dead")
    monkeypatch.setattr(orb, "OPENROUTER_RECOMMENDED_CREATIVE_DEFAULT", "vendor/dead")
    assert orb.fallback_slug_for_slot("A", exclude="vendor/dead") is None


def test_fallback_ignores_route_suffix_when_comparing(enabled_env, monkeypatch):
    monkeypatch.setenv("OPENROUTER_MODEL_A", "vendor/dead:nitro")
    monkeypatch.setattr(orb, "OPENROUTER_RECOMMENDED_CREATIVE_DEFAULT", "vendor/live")
    # the env candidate is the dead slug wearing a :nitro suffix -> still skipped.
    assert orb.fallback_slug_for_slot("A", exclude="vendor/dead") == "vendor/live"


# --- dispatch: the live fallover -------------------------------------------

def test_deleted_slot_model_falls_over_to_remote_fallback_once(enabled_env, monkeypatch):
    orb.set_slot_bindings(slot_a="vendor/dead-model", slot_b=None)
    calls = []

    def fake_post(*, base_url, api_key, payload, timeout_s):
        calls.append(payload["model"])
        if payload["model"] == "vendor/dead-model":
            return _gone("vendor/dead-model")
        return _ok("from the fallback")

    monkeypatch.setattr(orb, "_post_chat_completion", fake_post)
    backend = orb.OpenRouterBackend()
    entry = backend.load(orb.SLOT_A_ID, _row())
    out = backend.generate(entry, [{"role": "user", "content": "x"}],
                           temperature=0.5, max_new_tokens=16)
    assert out == "from the fallback"
    # dead first, then the env fallback -- exactly two calls.
    assert calls == ["vendor/dead-model", "anthropic/claude-opus-4.8"]


def test_fallover_happens_only_once_then_aborts(enabled_env, monkeypatch):
    # Both the chosen model AND the fallback are gone -> abort after one
    # fallover (no infinite retry into 404s).
    orb.set_slot_bindings(slot_a="vendor/dead-1", slot_b=None)
    monkeypatch.setenv("OPENROUTER_MODEL_A", "vendor/dead-2")
    calls = []

    def fake_post(*, base_url, api_key, payload, timeout_s):
        calls.append(payload["model"])
        return _gone(payload["model"])  # everything is gone

    monkeypatch.setattr(orb, "_post_chat_completion", fake_post)
    backend = orb.OpenRouterBackend()
    entry = backend.load(orb.SLOT_A_ID, _row())
    with pytest.raises(orb.OpenRouterModelGoneError):
        backend.generate(entry, [{"role": "user", "content": "x"}],
                         temperature=0.5, max_new_tokens=16)
    assert calls == ["vendor/dead-1", "vendor/dead-2"]


def test_no_fallback_candidate_aborts_cleanly(enabled_env, monkeypatch):
    orb.set_slot_bindings(slot_a="vendor/dead", slot_b=None)
    monkeypatch.setenv("OPENROUTER_MODEL_A", "vendor/dead")
    monkeypatch.setattr(orb, "OPENROUTER_RECOMMENDED_CREATIVE_DEFAULT", "vendor/dead")
    calls = []

    def fake_post(*, base_url, api_key, payload, timeout_s):
        calls.append(payload["model"])
        return _gone(payload["model"])

    monkeypatch.setattr(orb, "_post_chat_completion", fake_post)
    backend = orb.OpenRouterBackend()
    entry = backend.load(orb.SLOT_A_ID, _row())
    with pytest.raises(orb.OpenRouterModelGoneError):
        backend.generate(entry, [{"role": "user", "content": "x"}],
                         temperature=0.5, max_new_tokens=16)
    assert calls == ["vendor/dead"]  # no distinct fallback -> no retry


def test_generic_404_aborts_without_fallover(enabled_env, monkeypatch):
    # A 404 with no gone-marker (e.g. a transient gateway 404) is NOT deletion:
    # plain abort, no fallover.
    orb.set_slot_bindings(slot_a="vendor/maybe", slot_b=None)
    calls = []

    def fake_post(*, base_url, api_key, payload, timeout_s):
        calls.append(payload["model"])
        return {"status_code": 404, "json": {"error": {"message": "gateway hiccup"}},
                "text": "gateway hiccup"}

    monkeypatch.setattr(orb, "_post_chat_completion", fake_post)
    backend = orb.OpenRouterBackend()
    entry = backend.load(orb.SLOT_A_ID, _row())
    with pytest.raises(orb.OpenRouterCallFailedError) as ei:
        backend.generate(entry, [{"role": "user", "content": "x"}],
                         temperature=0.5, max_new_tokens=16)
    assert not isinstance(ei.value, orb.OpenRouterModelGoneError)
    assert calls == ["vendor/maybe"]


def test_directly_pinned_deleted_model_aborts_no_fallover(enabled_env, monkeypatch):
    # No slot handle (a directly-pinned slug) -> there is no slot to fall back
    # through, so a deleted direct pin aborts.
    calls = []

    def fake_post(*, base_url, api_key, payload, timeout_s):
        calls.append(payload["model"])
        return _gone(payload["model"])

    monkeypatch.setattr(orb, "_post_chat_completion", fake_post)
    entry = {"model_id": "vendor/dead-direct", "slug": "vendor/dead-direct",
             "provider": "openrouter"}
    backend = orb.OpenRouterBackend()
    with pytest.raises(orb.OpenRouterModelGoneError):
        backend.generate(entry, [{"role": "user", "content": "x"}],
                         temperature=0.5, max_new_tokens=16)
    assert calls == ["vendor/dead-direct"]
