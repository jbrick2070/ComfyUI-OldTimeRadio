"""S4 -- remote technical JSON is controlled + fail-closed (C4).

The remote constrained generate-fn maps the call's Pydantic schema to an
OpenRouter json_schema response_format, then rides the EXISTING
structured_call validate + bounded-repair ladder (zero new validation
logic). Malformed remote output that survives the ladder raises
StructuredCallFailedError -> the call aborts and nothing reaches the
ledger. A model with no schema support returns a 4xx -> the backend
raises OpenRouterCallFailedError (also fail-closed). No network.
"""
from __future__ import annotations

import types

import pytest
from pydantic import BaseModel

from nodes import _otr_openrouter_backend as orb
from nodes import _otr_constrained_generate as cg
from nodes._otr_structured_call import structured_call, StructuredCallFailedError


class Tiny(BaseModel):
    name: str
    score: int


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    orb.reset_run_budget()
    monkeypatch.setattr(orb.time, "sleep", lambda *_a, **_k: None)
    yield
    orb.reset_run_budget()


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setenv("OTR_ENABLE_OPENROUTER", "1")
    monkeypatch.setenv("OPENROUTER_MODEL_A", "anthropic/claude-3.5-sonnet")
    monkeypatch.setenv("OPENROUTER_MODEL_B", "openai/gpt-4o")


def _row():
    return types.SimpleNamespace(
        repo_id=orb.SLOT_A_ID, loader_backend="openrouter_http", context_window=8192
    )


def _entry(enabled):  # noqa: ARG001 -- fixture ordering only
    return orb.OpenRouterBackend().load(orb.SLOT_A_ID, _row())


def _post_returning(content):
    return {"status_code": 200, "json": {"choices": [{"message": {"content": content}}]}, "text": ""}


# ---------------------------------------------------------------------------
# response_format is set from the Pydantic schema
# ---------------------------------------------------------------------------


def test_remote_constrained_fn_sets_response_format(enabled, monkeypatch):
    seen = {}
    monkeypatch.setattr(
        orb, "_post_chat_completion",
        lambda **kw: seen.update(kw) or _post_returning('{"name":"a","score":1}'),
    )
    entry = _entry(enabled)
    fn = cg.make_constrained_generate_fn(entry, Tiny)
    fn([{"role": "user", "content": "x"}], temperature=0.2, max_new_tokens=64)
    rf = seen["payload"]["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == "Tiny"
    assert rf["json_schema"]["strict"] is True
    assert rf["json_schema"]["schema"] == Tiny.model_json_schema()


# ---------------------------------------------------------------------------
# rides the existing structured_call ladder (valid + fail-closed)
# ---------------------------------------------------------------------------


def test_remote_valid_output_passes_structured_call(enabled, monkeypatch):
    monkeypatch.setattr(
        orb, "_post_chat_completion",
        lambda **kw: _post_returning('{"name":"Ada","score":42}'),
    )
    entry = _entry(enabled)
    slot_fn = cg.make_constrained_generate_fn(entry, Tiny)
    result = structured_call(
        prompt="give me a record",
        schema=Tiny,
        slot_fn=slot_fn,
        base_temperature=0.3,
        structural_retry_temperature=0.1,
        helper_name="s4-test",
    )
    assert isinstance(result, Tiny)
    assert result.name == "Ada"
    assert result.score == 42


def test_remote_malformed_output_fails_closed(enabled, monkeypatch):
    """Garbage on every attempt -> the ladder exhausts and raises
    StructuredCallFailedError. The caller gets NO instance, so nothing
    is written to the ledger (C4 fail-closed)."""
    monkeypatch.setattr(
        orb, "_post_chat_completion",
        lambda **kw: _post_returning("this is not json at all"),
    )
    entry = _entry(enabled)
    slot_fn = cg.make_constrained_generate_fn(entry, Tiny)
    with pytest.raises(StructuredCallFailedError):
        structured_call(
            prompt="give me a record",
            schema=Tiny,
            slot_fn=slot_fn,
            base_temperature=0.3,
            structural_retry_temperature=0.1,
            helper_name="s4-test",
        )


def test_remote_schema_invalid_output_fails_closed(enabled, monkeypatch):
    """Well-formed JSON that violates the schema (missing 'score')
    still fails closed after the ladder."""
    monkeypatch.setattr(
        orb, "_post_chat_completion",
        lambda **kw: _post_returning('{"name":"only-name"}'),
    )
    entry = _entry(enabled)
    slot_fn = cg.make_constrained_generate_fn(entry, Tiny)
    with pytest.raises(StructuredCallFailedError):
        structured_call(
            prompt="x", schema=Tiny, slot_fn=slot_fn,
            base_temperature=0.3, structural_retry_temperature=0.1,
        )


# ---------------------------------------------------------------------------
# no-schema-support model -> fail-closed with an actionable error
# ---------------------------------------------------------------------------


def test_remote_no_schema_support_fails_closed(enabled, monkeypatch):
    """A model that rejects response_format returns a 4xx; the backend
    raises OpenRouterCallFailedError (the run aborts -- nothing reaches
    the ledger)."""
    monkeypatch.setattr(
        orb, "_post_chat_completion",
        lambda **kw: {
            "status_code": 400,
            "json": {"error": {"message": "response_format not supported"}},
            "text": "",
        },
    )
    entry = _entry(enabled)
    fn = cg.make_constrained_generate_fn(entry, Tiny)
    with pytest.raises(orb.OpenRouterCallFailedError):
        fn([{"role": "user", "content": "x"}], temperature=0.2, max_new_tokens=64)


# ---------------------------------------------------------------------------
# local path unaffected (regression guard)
# ---------------------------------------------------------------------------


def test_local_entry_without_tokenizer_still_errors():
    from nodes._otr_model_loader import ModelLoaderError
    with pytest.raises(ModelLoaderError):
        cg.make_constrained_generate_fn({"provider": "local"}, Tiny)
