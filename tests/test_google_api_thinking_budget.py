"""Thought tokens are billed against max_output_tokens on the Gemini
Interactions API (measured 2026-09-19 on the first live google_veo_low_1act
leg: gemini-flash-latest spent 242 of a 256-token budget thinking, returned 7
visible tokens with status "incomplete", and the writer's 250-token
description call died as "no decodable JSON").

Pins: the per-model thinking level and its env override, that no
max_output_tokens is sent at all since 2026-09-28 (operator: "no caps") -- so
thinking can no longer starve the answer -- and that `status: incomplete` is
read as an output limit (raised for bounded patches, warned-and-returned
otherwise).
"""
from __future__ import annotations

import logging

import pytest

from nodes._otr_google_api import llm as gllm


def _entry(model):
    return {"provider": "google_api", "model_id": "google_api:slot-a",
            "google_model": model, "context_cap": 32768, "context_window": 32768}


def _capture(monkeypatch, body):
    captured = {}

    def fake_create(payload):
        captured["payload"] = payload
        return body

    monkeypatch.setattr(gllm, "create_interaction", fake_create)
    return captured


@pytest.fixture(autouse=True)
def _no_env_override(monkeypatch):
    monkeypatch.delenv("OTR_GOOGLE_THINKING_LEVEL", raising=False)


def test_thinking_level_is_per_model_family():
    assert gllm.thinking_level_for("gemini-flash-latest") == "low"
    assert gllm.thinking_level_for("gemini-2.5-flash") == "low"
    # flash-lite is silent by default and a level would switch thinking ON
    assert gllm.thinking_level_for("gemini-flash-lite-latest") is None
    # pro cannot switch thinking off; leave the provider default
    assert gllm.thinking_level_for("gemini-pro-latest") is None


def test_env_override_wins_and_default_means_send_nothing(monkeypatch):
    monkeypatch.setenv("OTR_GOOGLE_THINKING_LEVEL", "high")
    assert gllm.thinking_level_for("gemini-flash-lite-latest") == "high"
    monkeypatch.setenv("OTR_GOOGLE_THINKING_LEVEL", "default")
    assert gllm.thinking_level_for("gemini-flash-latest") is None


def test_flash_payload_carries_low_and_no_output_cap(monkeypatch):
    captured = _capture(monkeypatch, {"status": "completed", "steps": [
        {"type": "model_output", "content": [{"type": "text", "text": "{}"}]}]})
    fn = gllm.make_google_api_generate_fn(_entry("gemini-flash-latest"))
    assert fn([{"role": "user", "content": "x"}], max_new_tokens=250) == "{}"
    gc = captured["payload"]["generation_config"]
    assert gc["thinking_level"] == "low"
    assert "max_output_tokens" not in gc


def test_flash_lite_payload_sends_no_level(monkeypatch):
    captured = _capture(monkeypatch, {"status": "completed", "steps": [
        {"type": "model_output", "content": [{"type": "text", "text": "ok"}]}]})
    fn = gllm.make_google_api_generate_fn(_entry("gemini-flash-lite-latest"))
    fn([{"role": "user", "content": "x"}], max_new_tokens=100)
    assert "thinking_level" not in (captured["payload"].get("generation_config") or {})


def test_incomplete_status_is_an_output_limit():
    body = {"status": "incomplete", "usage": {"total_thought_tokens": 242,
                                              "total_output_tokens": 7},
            "steps": [{"type": "model_output",
                       "content": [{"type": "text", "text": "{\n  \"character"}]}]}
    assert gllm._output_limit_reason(body) == "incomplete"
    with pytest.raises(gllm.GoogleAPIRequestShapeError, match="incomplete.*thought_tokens=242"):
        gllm._extract_text(body, fail_on_output_limit=True)


def test_incomplete_without_fail_flag_returns_partial_and_warns(caplog):
    body = {"status": "incomplete", "usage": {"total_thought_tokens": 242,
                                              "total_output_tokens": 7},
            "steps": [{"type": "model_output",
                       "content": [{"type": "text", "text": "partial"}]}]}
    with caplog.at_level(logging.WARNING, logger="OTR.google_api"):
        assert gllm._extract_text(body) == "partial"
    assert any("truncated (incomplete)" in r.getMessage() and "242" in r.getMessage()
               for r in caplog.records)


def test_nested_status_never_counts_as_a_limit():
    """Sonnet QA 2026-09-19: only the TOP-LEVEL status is the budget signal.
    A step or tool-call carrying its own status: incomplete must not refuse
    a complete answer."""
    body = {"status": "completed", "steps": [{
        "type": "model_output",
        "content": [{"type": "text", "text": "the real, complete answer"}],
        "tool_call": {"status": "incomplete", "note": "unrelated"}}]}
    assert gllm._output_limit_reason(body) == ""
    assert gllm._extract_text(body, fail_on_output_limit=True) == "the real, complete answer"
    # a nested finish_reason still counts, as before
    body["steps"][0]["finish_reason"] = "max_tokens"
    assert gllm._output_limit_reason(body) == "max_tokens"


def test_truncation_warning_blames_thinking_only_with_thought_tokens(caplog):
    body = {"status": "incomplete", "usage": {"total_thought_tokens": 0,
                                              "total_output_tokens": 256},
            "steps": [{"type": "model_output",
                       "content": [{"type": "text", "text": "partial"}]}]}
    with caplog.at_level(logging.WARNING, logger="OTR.google_api"):
        gllm._extract_text(body)
    msg = "\n".join(r.getMessage() for r in caplog.records)
    assert "reached the model's own output limit" in msg
    assert "thinking model" not in msg


def test_completed_status_is_not_a_limit():
    body = {"status": "completed", "steps": [
        {"type": "model_output", "content": [{"type": "text", "text": "ok"}]}]}
    assert gllm._output_limit_reason(body) == ""
    assert gllm._extract_text(body, fail_on_output_limit=True) == "ok"
