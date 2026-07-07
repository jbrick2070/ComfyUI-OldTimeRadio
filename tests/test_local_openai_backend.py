"""Tests for the external local OpenAI-compatible Gemma 4 12B lane."""
from __future__ import annotations

import pathlib
import types

import pytest

from nodes import _otr_local_openai_backend as LOB


def _row(ctx=8192):
    return types.SimpleNamespace(context_window=ctx)


def _ok(content="hello"):
    return {
        "status_code": 200,
        "json": {"choices": [{"message": {"content": content},
                              "finish_reason": "stop"}]},
        "text": "",
    }


def test_constants_are_ollama_free():
    assert LOB.LOCAL_OPENAI_BACKEND_KEY == "local_openai_http"
    assert LOB.PROVIDER == "local_openai"
    assert LOB.ROW_ID == "local_gemma4_12b"
    assert LOB.DEFAULT_BASE_URL == "http://127.0.0.1:8080/v1"
    src = pathlib.Path(LOB.__file__).read_text(encoding="utf-8")
    assert "subprocess" not in src
    assert "ollama serve" not in src
    assert "GEMMA4_12B_ENABLED" not in src


def test_load_does_not_require_enable_flag(monkeypatch):
    monkeypatch.delenv("GEMMA4_12B_ENABLED", raising=False)
    ce = LOB.LocalOpenAIBackend().load(LOB.ROW_ID, _row())
    assert ce["provider"] == "local_openai"
    assert ce["model_id"] == LOB.ROW_ID


def test_load_builds_zero_vram_cache_entry(monkeypatch):
    monkeypatch.setenv("GEMMA4_12B_BASE_URL", "http://127.0.0.1:8080/v1")
    monkeypatch.setenv("GEMMA4_12B_MODEL_ID", "gemma-4-12B-it-GGUF:Q4_K_M")
    ce = LOB.LocalOpenAIBackend().load(LOB.ROW_ID, _row())
    assert ce["provider"] == "local_openai"
    assert ce["model_id"] == LOB.ROW_ID
    assert ce["slug"] == "gemma-4-12B-it-GGUF:Q4_K_M"
    assert ce["base_url"] == "http://127.0.0.1:8080/v1"
    assert "model" not in ce and "tokenizer" not in ce


def test_generate_posts_to_configured_local_endpoint(monkeypatch):
    captured = {}

    def fake_post(*, base_url, payload, timeout_s):
        captured["base_url"] = base_url
        captured["payload"] = payload
        captured["timeout_s"] = timeout_s
        return _ok("the answer")

    monkeypatch.setenv("GEMMA4_12B_MAX_RETRIES", "0")
    monkeypatch.setenv("GEMMA4_12B_MAX_NEW_TOKENS", "512")
    monkeypatch.setattr(LOB, "_post_chat_completion", fake_post)
    ce = LOB.LocalOpenAIBackend().load(LOB.ROW_ID, _row())
    out = LOB.LocalOpenAIBackend().generate(
        ce, [{"role": "user", "content": "hi"}],
        temperature=0.2, max_new_tokens=999,
    )
    assert out == "the answer"
    assert captured["base_url"] == LOB.DEFAULT_BASE_URL
    assert captured["payload"]["model"] == LOB.DEFAULT_MODEL_ID
    assert captured["payload"]["max_tokens"] == 512
    assert captured["payload"]["reasoning_effort"] == "none"


def test_generate_carries_response_format(monkeypatch):
    captured = {}

    def fake_post(*, base_url, payload, timeout_s):
        captured["payload"] = payload
        return _ok("{}")

    monkeypatch.setattr(LOB, "_post_chat_completion", fake_post)
    ce = LOB.LocalOpenAIBackend().load(LOB.ROW_ID, _row())
    fn = LOB.make_local_openai_generate_fn(
        ce, response_format={"type": "json_schema", "json_schema": {"name": "x"}},
    )
    assert getattr(fn, "_otr_local_openai", False) is True
    assert fn([{"role": "user", "content": "x"}], max_new_tokens=32) == "{}"
    assert captured["payload"]["response_format"]["type"] == "json_schema"


def test_model_loader_generate_factory_routes_local_openai(monkeypatch):
    from nodes import _otr_model_loader as loader

    monkeypatch.setattr(LOB, "_post_chat_completion", lambda **_kwargs: _ok("ok"))
    ce = LOB.LocalOpenAIBackend().load(LOB.ROW_ID, _row())
    fn = loader.make_generate_fn(ce)
    assert getattr(fn, "_otr_local_openai", False) is True
    assert (
        fn([{"role": "user", "content": "x"}],
           temperature=0.1, max_new_tokens=8) == "ok"
    )


def test_constrained_generate_routes_response_format(monkeypatch):
    from pydantic import BaseModel

    from nodes import _otr_constrained_generate as cg

    class ProbeSchema(BaseModel):
        value: str

    captured = {}

    def fake_post(*, base_url, payload, timeout_s):
        captured["payload"] = payload
        return _ok('{"value":"ok"}')

    monkeypatch.setattr(LOB, "_post_chat_completion", fake_post)
    ce = LOB.LocalOpenAIBackend().load(LOB.ROW_ID, _row())
    fn = cg.make_constrained_generate_fn(ce, ProbeSchema)
    assert (
        fn([{"role": "user", "content": "x"}],
           temperature=0.1, max_new_tokens=32) == '{"value":"ok"}'
    )
    assert captured["payload"]["response_format"]["type"] == "json_schema"


def test_generate_rejects_raw_grammar(monkeypatch):
    posted = {"called": False}

    def fake_post(**_kwargs):
        posted["called"] = True
        return _ok()

    monkeypatch.setattr(LOB, "_post_chat_completion", fake_post)
    ce = LOB.LocalOpenAIBackend().load(LOB.ROW_ID, _row())
    with pytest.raises(LOB.LocalOpenAIConfigError):
        LOB.LocalOpenAIBackend().generate(
            ce, [{"role": "user", "content": "x"}], grammar='root ::= "a"',
        )
    assert posted["called"] is False


def test_generate_fail_closed_on_transport_error(monkeypatch):
    def boom(**_kwargs):
        raise ConnectionError("connection refused")

    monkeypatch.setenv("GEMMA4_12B_MAX_RETRIES", "0")
    monkeypatch.setattr(LOB, "_post_chat_completion", boom)
    ce = LOB.LocalOpenAIBackend().load(LOB.ROW_ID, _row())
    with pytest.raises(LOB.LocalOpenAICallFailedError):
        LOB.LocalOpenAIBackend().generate(ce, [{"role": "user", "content": "x"}])


def test_validate_local_openai_server_uses_tiny_completion(monkeypatch):
    monkeypatch.setattr(
        LOB, "_get_models",
        lambda **_kwargs: {"status_code": 200, "json": {"data": []}, "text": ""},
    )
    monkeypatch.setattr(
        LOB, "_post_chat_completion",
        lambda **_kwargs: _ok("OK"),
    )
    status = LOB.validate_local_openai_server()
    assert status["ok"] is True
    assert status["models_status"] == 200
    assert status["completion_status"] == 200


def test_dispatch_table_registers_local_openai():
    from nodes import _otr_model_runtime as RT
    assert LOB.LOCAL_OPENAI_BACKEND_KEY in RT.BACKENDS_BY_KEY
    assert isinstance(
        RT.BACKENDS_BY_KEY[LOB.LOCAL_OPENAI_BACKEND_KEY],
        LOB.LocalOpenAIBackend,
    )


def test_source_is_ascii_no_em_dash():
    src = pathlib.Path(LOB.__file__).read_text(encoding="utf-8")
    src.encode("ascii")
    assert "\u2014" not in src
