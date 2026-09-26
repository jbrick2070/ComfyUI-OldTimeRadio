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


# ---------------------------------------------------------------------------
# Lean hardening: remote creative structured_call gets json_object mode
# ---------------------------------------------------------------------------


def test_remote_creative_structured_call_forces_json_object(enabled, monkeypatch):
    """A remote fn with NO bound schema, driven through structured_call,
    must request response_format=json_object (+ provider.require_parameters)
    so a free-form frontier model returns parseable JSON."""
    seen = {}
    monkeypatch.setattr(
        orb, "_post_chat_completion",
        lambda **kw: seen.update(kw) or _post_returning('{"name":"Ada","score":1}'),
    )
    slot_fn = orb.make_openrouter_generate_fn(_entry(enabled))  # no schema bound
    assert getattr(slot_fn, "_otr_openrouter", False) is True
    assert getattr(slot_fn, "_otr_response_format", "x") is None
    structured_call(prompt="x", schema=Tiny, slot_fn=slot_fn,
                    base_temperature=0.3, structural_retry_temperature=0.1)
    assert seen["payload"]["response_format"] == {"type": "json_object"}
    assert seen["payload"]["provider"] == {"require_parameters": True}


def test_remote_bound_schema_not_overridden_by_json_object(enabled, monkeypatch):
    """A remote fn that already carries a json_schema (S4 grammar path) keeps
    it -- _invoke_slot must not clobber it with json_object."""
    seen = {}
    monkeypatch.setattr(
        orb, "_post_chat_completion",
        lambda **kw: seen.update(kw) or _post_returning('{"name":"A","score":2}'),
    )
    rf = orb.schema_to_response_format(Tiny)
    slot_fn = orb.make_openrouter_generate_fn(_entry(enabled), response_format=rf)
    structured_call(prompt="x", schema=Tiny, slot_fn=slot_fn,
                    base_temperature=0.3, structural_retry_temperature=0.1)
    assert seen["payload"]["response_format"]["type"] == "json_schema"

# ---------------------------------------------------------------------------
# THE NEGATIVE HALF OF THE SAME CONTRACT, recovered 2026-09-24.
#
# `test_remote_creative_structured_call_forces_json_object` above proves the
# REMOTE lane gets `response_format={"type": "json_object"}` forced onto it for
# a schema-less structured pass. The proof that the LOCAL transformers lane
# never does lived in a registry test file deleted wholesale with a retired
# writer backend -- but that particular test was not about that backend at all,
# and a QA pass caught it going out with the bathwater.
#
# It matters because `invoke_structured_slot` is live shared code: it is reached
# from several call sites in `_otr_structured_call`, which a long list of
# production modules import. The local lane has no json_object mode, so forcing
# the kwarg onto it is a runtime error on an ordinary writer pass -- and the
# only thing standing between that and a release is this assertion.
# ---------------------------------------------------------------------------

def test_json_object_is_never_forced_on_the_local_transformers_lane(monkeypatch):
    """A local slot fn must receive response_format=None, not json_object."""
    from nodes import _otr_structured_call as SC
    from nodes import _otr_model_loader as loader
    from nodes.OTR_LedgerScriptWriter import _SlotScheduler

    captured = {}

    def fake_local_fn(messages, *, temperature, max_new_tokens, stop=None,
                      response_format=None):
        captured["response_format"] = response_format
        return "ok"

    # A local (transformers) slot has no json_object mode -> marker False.
    monkeypatch.setattr(
        loader, "request_slot",
        lambda slot, model_id, policy=None: {
            "model": object(), "tokenizer": object(),
        },
    )
    monkeypatch.setattr(
        "nodes.OTR_LedgerScriptWriter._build_truncating_generate_fn",
        lambda cache_entry, **_kw: fake_local_fn,
    )
    sched = _SlotScheduler(
        creative_id="mistralai/Mistral-Nemo-Instruct-2407",
        technical_id="mistralai/Mistral-Nemo-Instruct-2407", min_p=0.0, repetition_penalty=1.0,
    )
    fn = sched.for_slot("technical")
    assert getattr(fn, "_otr_supports_json_object", False) is False, (
        "the local lane must not advertise json_object support")

    SC.invoke_structured_slot(
        fn, [{"role": "user", "content": "x"}], temperature=0.2,
        max_new_tokens=16,
    )
    assert captured["response_format"] is None, (
        "response_format was forced onto the local transformers lane (%r); it "
        "has no json_object mode and this is a runtime error on an ordinary "
        "writer pass" % (captured["response_format"],))

