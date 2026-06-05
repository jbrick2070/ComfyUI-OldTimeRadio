"""Tests for the dedicated LOCAL llama.cpp / Ollama writer lane (2026-06-04).

Headless + network-free: every test that exercises generate() patches the
module's _post_chat_completion, so no socket is opened. Verifies the lane is
LOCAL (127.0.0.1 / localhost), fail-closed (never a cloud endpoint),
self-contained (no import coupling to the cloud OpenRouter lane), and that the
gemma-4-12b catalog row + the runtime dispatch table both route to it.
"""
from __future__ import annotations

import pathlib
import types

import pytest

from nodes import _otr_ollama_backend as OLL


def _row(tag="hf.co/unsloth/gemma-4-12b-it-GGUF:Q4_K_M", ctx=8192):
    return types.SimpleNamespace(ollama_model_tag=tag, context_window=ctx)


def _ok(content="hello"):
    return {
        "status_code": 200,
        "json": {"choices": [{"message": {"content": content},
                              "finish_reason": "stop"}]},
        "text": "",
    }


# --------------------------------------------------------------------------- #
# Lane identity: LOCAL, and distinct from the cloud lanes
# --------------------------------------------------------------------------- #
def test_lane_constants_are_local_and_distinct():
    assert OLL.OLLAMA_BACKEND_KEY == "ollama_local_http"
    assert OLL.PROVIDER == "ollama"
    base = OLL.DEFAULT_OLLAMA_BASE_URL
    assert base.startswith("http://localhost") or base.startswith("http://127.0.0.1")
    # Distinct provider tag from BOTH cloud lanes.
    from nodes import _otr_comfy_backend as OCC
    from nodes import _otr_openrouter_backend as ORB
    assert OLL.PROVIDER not in (ORB.PROVIDER, OCC.PROVIDER)


def test_base_url_default_is_local_and_env_overridable(monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    assert OLL._base_url() == OLL.DEFAULT_OLLAMA_BASE_URL
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:8080/v1")
    assert OLL._base_url() == "http://localhost:8080/v1"


def test_enabled_always_true_no_flag_no_key():
    # Local infrastructure: no opt-in flag, no API key required.
    assert OLL.ollama_enabled() is True


def test_module_does_not_import_openrouter_at_module_scope():
    # Separation: the local lane must not be coupled to the cloud lane.
    src = pathlib.Path(OLL.__file__).read_text(encoding="utf-8")
    assert "import _otr_openrouter_backend" not in src
    assert "from ._otr_openrouter_backend" not in src


# --------------------------------------------------------------------------- #
# load(): provider-tagged local cache_entry; fail-closed on missing tag
# --------------------------------------------------------------------------- #
def test_load_builds_local_cache_entry(monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    ce = OLL.OllamaBackend().load("google/gemma-4-12b-it", _row())
    assert ce["provider"] == "ollama"
    assert ce["model_id"] == "google/gemma-4-12b-it"
    assert ce["slug"] == "hf.co/unsloth/gemma-4-12b-it-GGUF:Q4_K_M"
    assert ce["base_url"] == OLL.DEFAULT_OLLAMA_BASE_URL
    # Zero ComfyUI-process VRAM: no resident model / tokenizer.
    assert "model" not in ce and "tokenizer" not in ce


def test_load_without_tag_fails_closed():
    with pytest.raises(OLL.OllamaConfigError):
        OLL.OllamaBackend().load("x/y", _row(tag=""))


# --------------------------------------------------------------------------- #
# generate(): local POST, payload shape, reasoning_effort carried, raw grammar rejected
# --------------------------------------------------------------------------- #
def test_generate_posts_local_and_returns_content(monkeypatch):
    captured = {}

    def fake_post(*, base_url, payload, timeout_s):
        captured["base_url"] = base_url
        captured["payload"] = payload
        return _ok("the answer")

    monkeypatch.setattr(OLL, "_post_chat_completion", fake_post)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_REASONING_EFFORT", raising=False)
    ce = OLL.OllamaBackend().load("google/gemma-4-12b-it", _row())
    out = OLL.OllamaBackend().generate(
        ce, [{"role": "user", "content": "hi"}],
        temperature=0.6, max_new_tokens=128,
    )
    assert out == "the answer"
    # LOCAL endpoint, never cloud.
    assert "localhost" in captured["base_url"] or "127.0.0.1" in captured["base_url"]
    assert captured["payload"]["model"] == "hf.co/unsloth/gemma-4-12b-it-GGUF:Q4_K_M"
    # reasoning_effort omitted when env unset (byte-identical for non-thinking).
    assert "reasoning_effort" not in captured["payload"]


def test_generate_carries_reasoning_effort(monkeypatch):
    captured = {}

    def fake_post(*, base_url, payload, timeout_s):
        captured["payload"] = payload
        return _ok()

    monkeypatch.setattr(OLL, "_post_chat_completion", fake_post)
    monkeypatch.setenv("OLLAMA_REASONING_EFFORT", "none")
    ce = OLL.OllamaBackend().load("g", _row())
    OLL.OllamaBackend().generate(
        ce, [{"role": "user", "content": "x"}],
    )
    assert captured["payload"]["reasoning_effort"] == "none"


def test_generate_rejects_raw_grammar(monkeypatch):
    # Ollama /v1 does not accept a raw GBNF grammar; the lane fails loud rather
    # than POST an unsupported field or silently drop it.
    posted = {"called": False}

    def fake_post(*, base_url, payload, timeout_s):
        posted["called"] = True
        return _ok()

    monkeypatch.setattr(OLL, "_post_chat_completion", fake_post)
    ce = OLL.OllamaBackend().load("g", _row())
    with pytest.raises(OLL.OllamaConfigError):
        OLL.OllamaBackend().generate(
            ce, [{"role": "user", "content": "x"}], grammar='root ::= "a"',
        )
    assert posted["called"] is False


def test_generate_fail_closed_on_non_200(monkeypatch):
    monkeypatch.setattr(
        OLL, "_post_chat_completion",
        lambda **k: {"status_code": 404, "json": None, "text": "model not found"},
    )
    monkeypatch.setenv("OLLAMA_MAX_RETRIES", "0")
    ce = OLL.OllamaBackend().load("g", _row())
    with pytest.raises(OLL.OllamaCallFailedError):
        OLL.OllamaBackend().generate(ce, [{"role": "user", "content": "x"}])


def test_generate_fail_closed_on_transport_error(monkeypatch):
    def boom(**k):
        raise ConnectionError("connection refused")

    monkeypatch.setattr(OLL, "_post_chat_completion", boom)
    monkeypatch.setenv("OLLAMA_MAX_RETRIES", "0")
    ce = OLL.OllamaBackend().load("g", _row())
    with pytest.raises(OLL.OllamaCallFailedError):
        OLL.OllamaBackend().generate(ce, [{"role": "user", "content": "x"}])


# --------------------------------------------------------------------------- #
# generate-fn factory markers (so the inventor passes GBNF only to honouring
# backends; mirrors make_openrouter_generate_fn)
# --------------------------------------------------------------------------- #
def test_make_generate_fn_markers():
    ce = OLL.OllamaBackend().load("g", _row())
    fn = OLL.make_ollama_generate_fn(ce)
    assert getattr(fn, "_otr_ollama", False) is True
    # Ollama /v1 cannot honour raw GBNF, so the lane must NOT advertise grammar
    # support -- the style-picker inventor gates on this marker's absence.
    assert getattr(fn, "_otr_supports_grammar", False) is False


def test_inventor_flag_on_does_not_leak_grammar_to_ollama(monkeypatch):
    # Even with the inventor GBNF flag ON, the Ollama lane must not advertise
    # grammar support, so _run_inventor's gate keeps the raw grammar away from
    # Ollama; the ordinary (no-grammar) call still works.
    monkeypatch.setenv("OTR_ENABLE_INVENTOR_GBNF", "1")
    captured = {}

    def fake_post(*, base_url, payload, timeout_s):
        captured["payload"] = payload
        return _ok("ok")

    monkeypatch.setattr(OLL, "_post_chat_completion", fake_post)
    ce = OLL.OllamaBackend().load("g", _row())
    fn = OLL.make_ollama_generate_fn(ce)
    assert getattr(fn, "_otr_supports_grammar", False) is False
    out = fn([{"role": "user", "content": "x"}], temperature=0.6, max_new_tokens=64)
    assert out == "ok"
    assert "grammar" not in captured["payload"]


# --------------------------------------------------------------------------- #
# Catalog + runtime dispatch wiring
# --------------------------------------------------------------------------- #
def test_gemma_12b_row_routes_to_local_ollama():
    from nodes import _otr_model_catalog as CAT
    row = CAT._by_repo_id().get("google/gemma-4-12b-it")
    assert row is not None
    assert row.loader_backend == "ollama_local_http"
    assert row.provider == "ollama"
    assert row.ollama_model_tag == "hf.co/unsloth/gemma-4-12b-it-GGUF:Q4_K_M"
    assert row.requires_auth is False  # Ollama-managed, not an HF download


def test_dispatch_table_registers_ollama():
    from nodes import _otr_model_runtime as RT
    assert OLL.OLLAMA_BACKEND_KEY in RT.BACKENDS_BY_KEY
    assert isinstance(RT.BACKENDS_BY_KEY[OLL.OLLAMA_BACKEND_KEY], OLL.OllamaBackend)


def test_get_backend_for_gemma_row_is_ollama():
    from nodes import _otr_model_catalog as CAT
    from nodes import _otr_model_runtime as RT
    row = CAT._by_repo_id().get("google/gemma-4-12b-it")
    assert isinstance(RT.get_backend_for_row(row), OLL.OllamaBackend)


def test_source_is_ascii_no_em_dash():
    src = pathlib.Path(OLL.__file__).read_text(encoding="utf-8")
    assert "—" not in src  # em-dash forbidden (CLAUDE.md)
    src.encode("ascii")
