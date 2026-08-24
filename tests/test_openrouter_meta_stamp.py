"""S5 -- remote-LLM provenance is stamped into run meta.

Every remote run records provider + virtual handle + resolved slug +
basic params + schema-mode for any remote slot. Local runs get NO extra
keys (byte-identical baseline, C1). No secret material is ever stamped
(C9). The writer wires the helper right after the creative_model stamp.
"""
from __future__ import annotations

import pathlib

import pytest

from nodes import _otr_openrouter_backend as orb


LOCAL = "mistralai/Mistral-Nemo-Instruct-2407"
A = "openrouter:slot-a"
B = "openrouter:slot-b"


@pytest.fixture(autouse=True)
def _clear_slot_bindings():
    """S3: clear the process-global slot bindings around every test so meta
    resolution always exercises the env-fallback path here (a binding set by
    another module must never contaminate these env-driven assertions)."""
    orb.clear_slot_bindings()
    yield
    orb.clear_slot_bindings()


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-SECRET-DO-NOT-LEAK")
    monkeypatch.setenv("OTR_ENABLE_OPENROUTER", "1")
    monkeypatch.setenv("OPENROUTER_MODEL_A", "anthropic/claude-3.5-sonnet")
    monkeypatch.setenv("OPENROUTER_MODEL_B", "openai/gpt-4o")


def test_both_local_yields_empty(enabled):
    # Even with remote enabled, two LOCAL ids stamp nothing.
    assert orb.openrouter_meta_for(LOCAL, LOCAL) == {}


def test_creative_remote_technical_local(enabled):
    meta = orb.openrouter_meta_for(A, LOCAL)
    assert meta["llm_creative_provider"] == "openrouter"
    assert meta["llm_creative_handle"] == A
    assert meta["llm_creative_slug"] == "anthropic/claude-3.5-sonnet"
    assert "llm_technical_provider" not in meta
    assert meta["llm_remote_provider"] == "openrouter"
    # technical is local -> schema mode off
    assert meta["llm_remote_schema_mode"] is False


def test_both_remote_records_both_and_schema_mode(enabled):
    meta = orb.openrouter_meta_for(A, B)
    assert meta["llm_creative_slug"] == "anthropic/claude-3.5-sonnet"
    assert meta["llm_technical_slug"] == "openai/gpt-4o"
    assert meta["llm_technical_handle"] == B
    # technical is remote -> json_schema response_format path is in play
    assert meta["llm_remote_schema_mode"] is True


def test_basic_params_present(enabled, monkeypatch):
    monkeypatch.setenv("OPENROUTER_A_MAXTOK", "512")
    monkeypatch.setenv("OPENROUTER_A_TEMP", "0.4")
    meta = orb.openrouter_meta_for(A, LOCAL)
    assert meta["llm_creative_max_tokens_cap"] == 512
    assert meta["llm_creative_temperature_override"] == 0.4


def test_slug_is_resolved_not_handle_and_no_suffix(enabled):
    meta = orb.openrouter_meta_for(A, B)
    for key in ("llm_creative_slug", "llm_technical_slug"):
        assert "openrouter:slot-" not in meta[key]  # not the handle
        assert "[NOT DOWNLOADED]" not in meta[key]


def test_no_api_key_in_any_meta_value(enabled):
    meta = orb.openrouter_meta_for(A, B)
    for v in meta.values():
        assert "sk-SECRET-DO-NOT-LEAK" not in str(v)


def test_unbound_slug_falls_back_to_recommended(monkeypatch):
    """S3: the env is now a FALLBACK, not the primary source. An unbound slot
    with no OPENROUTER_MODEL_A no longer stamps '<unresolved>' -- it resolves
    to the recommended creative default. The stamp still never raises (PD1)."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-x")
    monkeypatch.setenv("OTR_ENABLE_OPENROUTER", "1")
    monkeypatch.delenv("OPENROUTER_MODEL_A", raising=False)
    monkeypatch.delenv("OTR_OPENROUTER_SLOT_A_DEFAULT", raising=False)
    meta = orb.openrouter_meta_for(A, LOCAL)
    assert meta["llm_creative_slug"] == orb.OPENROUTER_RECOMMENDED_CREATIVE_DEFAULT


def test_writer_wires_the_stamp():
    """Source-pin: the writer module must call openrouter_meta_for so a
    future refactor cannot silently drop the remote provenance stamp."""
    from tests.fixtures.writer_family import family_source

    src = family_source()
    assert "openrouter_meta_for(" in src
    assert 'resolved["technical_model"]' in src
