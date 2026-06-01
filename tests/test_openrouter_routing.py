"""Provider routing -- ":nitro" fastest / ":floor" cheapest (speed/cost).

A single OpenRouter slug is served by several upstream providers; `provider.sort`
biases which one serves a call. This proves the writer can route its many
internal LLM calls to the fastest provider (":nitro" == throughput) or the
cheapest (":floor" == price) via either a slug suffix or an env knob, that the
sort merges cleanly with the require_parameters guard, that the default sends
no `provider` object at all, and that the resolved route is recorded in run
meta. No network and no secrets -- the only network seam is patched.
"""
from __future__ import annotations

import types

import pytest

from nodes import _otr_openrouter_backend as orb


@pytest.fixture(autouse=True)
def _reset_and_silence(monkeypatch):
    orb.reset_run_budget()
    monkeypatch.setattr(orb.time, "sleep", lambda *_a, **_k: None)
    yield
    orb.reset_run_budget()


@pytest.fixture
def enabled_env(monkeypatch):
    """Enabled + bound, with every routing/cost override cleared so defaults
    apply unless a test sets one explicitly."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-NEVER-LOGGED")
    monkeypatch.setenv("OTR_ENABLE_OPENROUTER", "1")
    monkeypatch.setenv("OPENROUTER_MODEL_A", "anthropic/claude-3.5-sonnet")
    monkeypatch.setenv("OPENROUTER_MODEL_B", "openai/gpt-4o")
    for k in (
        "OPENROUTER_SORT",
        "OPENROUTER_A_ROUTE",
        "OPENROUTER_B_ROUTE",
        "OPENROUTER_MAX_TOKENS_PER_CALL",
        "OPENROUTER_MAX_TOKENS_PER_RUN",
        "OPENROUTER_A_TEMP",
        "OPENROUTER_A_MAXTOK",
    ):
        monkeypatch.delenv(k, raising=False)


def _row(context_window: int = 8192):
    return types.SimpleNamespace(
        repo_id=orb.SLOT_A_ID,
        loader_backend=orb.OPENROUTER_BACKEND_KEY,
        context_window=context_window,
    )


def _ok_result(content: str = "ok"):
    return {"status_code": 200,
            "json": {"choices": [{"message": {"content": content}}]},
            "text": ""}


def _capture_payload(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        orb, "_post_chat_completion",
        lambda **kw: seen.update(kw) or _ok_result(),
    )
    return seen


# ---------------------------------------------------------------------------
# Token normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("token,expected", [
    ("nitro", "throughput"), ("NITRO", "throughput"), (" fast ", "throughput"),
    ("throughput", "throughput"), ("speed", "throughput"),
    ("floor", "price"), ("Floor", "price"), ("cheap", "price"), ("price", "price"),
    ("latency", "latency"),
    (None, None), ("", None), ("turbo", None), ("nonsense", None),
])
def test_normalize_sort(token, expected):
    assert orb._normalize_sort(token) == expected


# ---------------------------------------------------------------------------
# Slug-suffix shortcuts (OpenRouter's native ":nitro" / ":floor")
# ---------------------------------------------------------------------------


def test_split_slug_suffix_nitro():
    assert orb._split_slug_suffix("anthropic/claude-3.5-sonnet:nitro") == (
        "anthropic/claude-3.5-sonnet", "throughput")


def test_split_slug_suffix_floor():
    assert orb._split_slug_suffix("openai/gpt-4o:floor") == (
        "openai/gpt-4o", "price")


def test_split_slug_suffix_none_when_plain():
    # A bare slug -- and a slug whose author tag uses a colon -- is unchanged.
    assert orb._split_slug_suffix("openai/gpt-4o") == ("openai/gpt-4o", None)
    assert orb._split_slug_suffix("vendor/model:free") == ("vendor/model:free", None)


# ---------------------------------------------------------------------------
# resolve_route precedence
# ---------------------------------------------------------------------------


def test_route_default_is_none(enabled_env):
    assert orb.resolve_route("A", "anthropic/claude-3.5-sonnet") == (
        "anthropic/claude-3.5-sonnet", None)


def test_route_slug_suffix_wins_over_env(enabled_env, monkeypatch):
    monkeypatch.setenv("OPENROUTER_A_ROUTE", "floor")
    # The suffix is the most explicit signal: it wins, and is stripped.
    assert orb.resolve_route("A", "x/y:nitro") == ("x/y", "throughput")


def test_route_per_slot_env(enabled_env, monkeypatch):
    monkeypatch.setenv("OPENROUTER_A_ROUTE", "nitro")
    assert orb.resolve_route("A", "x/y") == ("x/y", "throughput")


def test_route_global_env_fallback(enabled_env, monkeypatch):
    monkeypatch.setenv("OPENROUTER_SORT", "price")
    assert orb.resolve_route("B", "x/y") == ("x/y", "price")


def test_route_per_slot_overrides_global(enabled_env, monkeypatch):
    monkeypatch.setenv("OPENROUTER_SORT", "price")
    monkeypatch.setenv("OPENROUTER_A_ROUTE", "nitro")
    assert orb.resolve_route("A", "x/y") == ("x/y", "throughput")


def test_route_unrecognized_env_warns_and_defaults(enabled_env, monkeypatch, caplog):
    monkeypatch.setenv("OPENROUTER_A_ROUTE", "supersonic")
    with caplog.at_level("WARNING"):
        slug, sort = orb.resolve_route("A", "x/y")
    assert (slug, sort) == ("x/y", None)
    assert any("supersonic" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# load() carries the resolved sort; the suffix is stripped from the slug
# ---------------------------------------------------------------------------


def test_load_strips_suffix_and_stores_sort(enabled_env, monkeypatch):
    monkeypatch.setenv("OPENROUTER_MODEL_A", "anthropic/claude-3.5-sonnet:nitro")
    entry = orb.OpenRouterBackend().load(orb.SLOT_A_ID, _row())
    assert entry["slug"] == "anthropic/claude-3.5-sonnet"  # suffix stripped
    assert entry["provider_sort"] == "throughput"


def test_load_default_sort_is_none(enabled_env):
    entry = orb.OpenRouterBackend().load(orb.SLOT_A_ID, _row())
    assert entry["provider_sort"] is None


# ---------------------------------------------------------------------------
# generate() wire payload
# ---------------------------------------------------------------------------


def test_generate_default_sends_no_provider_object(enabled_env, monkeypatch):
    seen = _capture_payload(monkeypatch)
    backend = orb.OpenRouterBackend()
    entry = backend.load(orb.SLOT_A_ID, _row())
    backend.generate(entry, [{"role": "user", "content": "x"}],
                     temperature=0.5, max_new_tokens=64)
    assert "provider" not in seen["payload"]


def test_generate_sends_sort_from_suffix(enabled_env, monkeypatch):
    monkeypatch.setenv("OPENROUTER_MODEL_A", "anthropic/claude-3.5-sonnet:nitro")
    seen = _capture_payload(monkeypatch)
    backend = orb.OpenRouterBackend()
    entry = backend.load(orb.SLOT_A_ID, _row())
    backend.generate(entry, [{"role": "user", "content": "x"}],
                     temperature=0.5, max_new_tokens=64)
    assert seen["payload"]["provider"] == {"sort": "throughput"}
    assert seen["payload"]["model"] == "anthropic/claude-3.5-sonnet"  # clean


def test_generate_sends_sort_from_env(enabled_env, monkeypatch):
    monkeypatch.setenv("OPENROUTER_A_ROUTE", "floor")
    seen = _capture_payload(monkeypatch)
    backend = orb.OpenRouterBackend()
    entry = backend.load(orb.SLOT_A_ID, _row())
    backend.generate(entry, [{"role": "user", "content": "x"}],
                     temperature=0.5, max_new_tokens=64)
    assert seen["payload"]["provider"] == {"sort": "price"}


def test_sort_and_require_parameters_coexist(enabled_env, monkeypatch):
    """The technical-JSON guard (require_parameters) and the speed/cost sort
    must merge into ONE provider object, not clobber each other."""
    monkeypatch.setenv("OPENROUTER_A_ROUTE", "nitro")
    seen = _capture_payload(monkeypatch)
    backend = orb.OpenRouterBackend()
    entry = backend.load(orb.SLOT_A_ID, _row())
    rf = {"type": "json_schema", "json_schema": {"name": "X", "schema": {}}}
    backend.generate(entry, [{"role": "user", "content": "x"}],
                     temperature=0.1, max_new_tokens=64, response_format=rf)
    assert seen["payload"]["provider"] == {
        "sort": "throughput", "require_parameters": True}
    assert seen["payload"]["response_format"] == rf


def test_require_parameters_only_when_no_sort(enabled_env, monkeypatch):
    seen = _capture_payload(monkeypatch)
    backend = orb.OpenRouterBackend()
    entry = backend.load(orb.SLOT_A_ID, _row())
    rf = {"type": "json_schema", "json_schema": {"name": "X", "schema": {}}}
    backend.generate(entry, [{"role": "user", "content": "x"}],
                     temperature=0.1, max_new_tokens=64, response_format=rf)
    assert seen["payload"]["provider"] == {"require_parameters": True}


# ---------------------------------------------------------------------------
# meta stamp records the route (S5 provenance)
# ---------------------------------------------------------------------------


def test_meta_records_default_route(enabled_env):
    meta = orb.openrouter_meta_for(orb.SLOT_A_ID, "local-model")
    assert meta["llm_creative_route"] == "default"
    assert meta["llm_creative_slug"] == "anthropic/claude-3.5-sonnet"


def test_meta_records_resolved_route_and_clean_slug(enabled_env, monkeypatch):
    monkeypatch.setenv("OPENROUTER_MODEL_A", "anthropic/claude-3.5-sonnet:floor")
    meta = orb.openrouter_meta_for(orb.SLOT_A_ID, "local-model")
    assert meta["llm_creative_route"] == "price"
    assert meta["llm_creative_slug"] == "anthropic/claude-3.5-sonnet"  # no suffix


def test_meta_empty_for_local_stays_empty(enabled_env):
    assert orb.openrouter_meta_for("local-a", "local-b") == {}
