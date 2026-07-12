"""Tests for the native GGUF Gemma 4 12B writer lane."""
from __future__ import annotations

import pathlib
import types

import pytest

from nodes import _otr_gguf_backend as GGF


def _row(ctx=8192):
    return types.SimpleNamespace(context_window=ctx)


class FakeLlama:
    def __init__(self):
        self.calls = []
        self.closed = False

    def create_chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "choices": [
                {"message": {"content": "the answer"}, "finish_reason": "stop"}
            ]
        }

    def close(self):
        self.closed = True


def test_constants_are_sidecar_free():
    assert GGF.GGUF_BACKEND_KEY == "gguf_native"
    assert GGF.PROVIDER == "gguf_native"
    assert GGF.ROW_ID == "unsloth/gemma-4-12b-it-GGUF"
    assert GGF.DEFAULT_GGUF_FILENAME == "gemma-4-12b-it-Q8_0.gguf"
    src = pathlib.Path(GGF.__file__).read_text(encoding="utf-8")
    assert "subprocess" not in src
    assert "ollama serve" not in src
    assert "127.0.0.1:8080" not in src


def test_default_path_uses_operator_model_root(monkeypatch):
    monkeypatch.setenv("OTR_COMFYUI_MODELS_ROOT", r"D:\ModelsRoot")
    expected = pathlib.Path(
        r"D:\ModelsRoot\LLM\converted\gemma-4-12b-it\gemma-4-12b-it-Q8_0.gguf"
    )
    assert GGF.default_gguf_path() == expected


def test_load_requires_gguf_file(monkeypatch, tmp_path):
    missing = tmp_path / "missing.gguf"
    monkeypatch.setenv("GEMMA4_12B_GGUF_PATH", str(missing))
    with pytest.raises(GGF.GGUFNativeConfigError, match="Missing Gemma 4 12B"):
        GGF.GGUFNativeBackend().load(GGF.ROW_ID, _row())


def test_load_builds_native_cache_entry(monkeypatch, tmp_path):
    model_path = tmp_path / "custom.gguf"
    model_path.write_bytes(b"gguf")
    fake = FakeLlama()
    seen = {}

    def fake_load(**kwargs):
        seen.update(kwargs)
        return fake

    monkeypatch.setenv("GEMMA4_12B_GGUF_PATH", str(model_path))
    monkeypatch.setenv("GEMMA4_12B_N_CTX", "4096")
    monkeypatch.setattr(GGF, "_load_llama", fake_load)
    ce = GGF.GGUFNativeBackend().load(GGF.ROW_ID, _row())
    assert ce["provider"] == "gguf_native"
    assert ce["model_id"] == GGF.ROW_ID
    assert ce["model"] is fake
    assert ce["model_path"] == str(model_path)
    assert ce["context_cap"] == 4096
    assert seen["model_path"] == model_path
    assert seen["n_ctx"] == 4096
    assert seen["n_gpu_layers"] == -1


def test_production_vram_gate_uses_resolved_model_path(monkeypatch, tmp_path):
    model_path = tmp_path / "custom.gguf"
    model_path.write_bytes(b"gguf")
    fake = FakeLlama()
    monkeypatch.setenv("GEMMA4_12B_GGUF_PATH", str(model_path))
    monkeypatch.setenv("GEMMA4_12B_N_CTX", "1024")
    monkeypatch.setenv("OTR_TEST_MODE", "0")
    monkeypatch.setattr(GGF, "_load_llama", lambda **_kwargs: fake)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.mem_get_info", lambda: (32 << 30, 32 << 30))
    entry = GGF.GGUFNativeBackend().load(GGF.ROW_ID, _row())
    assert entry["model_path"] == str(model_path)


def test_generate_calls_llama_cpp_and_caps_tokens(monkeypatch, tmp_path):
    fake = FakeLlama()
    monkeypatch.setenv("GEMMA4_12B_MAX_NEW_TOKENS", "512")
    ce = {"provider": "gguf_native", "model": fake}
    out = GGF.GGUFNativeBackend().generate(
        ce,
        [{"role": "user", "content": "hi"}],
        temperature=0.2,
        max_new_tokens=999,
    )
    assert out == "the answer"
    call = fake.calls[-1]
    assert call["messages"][0]["content"] == "hi"
    assert call["temperature"] == 0.2
    assert call["max_tokens"] == 512


def test_generate_translates_openai_json_schema_response_format():
    fake = FakeLlama()
    ce = {"provider": "gguf_native", "model": fake}
    schema = {"type": "object", "properties": {"value": {"type": "string"}}}
    fn = GGF.make_gguf_generate_fn(
        ce,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "Probe", "schema": schema},
        },
    )
    assert getattr(fn, "_otr_gguf_native", False) is True
    assert fn([{"role": "user", "content": "x"}], max_new_tokens=32) == "the answer"
    assert fake.calls[-1]["response_format"] == {
        "type": "json_object",
        "schema": schema,
    }


def test_model_loader_generate_factory_routes_gguf_native():
    from nodes import _otr_model_loader as loader

    fake = FakeLlama()
    ce = {"provider": "gguf_native", "model": fake}
    fn = loader.make_generate_fn(ce)
    assert getattr(fn, "_otr_gguf_native", False) is True
    assert (
        fn([{"role": "user", "content": "x"}],
           temperature=0.1, max_new_tokens=8) == "the answer"
    )


def test_constrained_generate_routes_response_format():
    from pydantic import BaseModel

    from nodes import _otr_constrained_generate as cg

    class ProbeSchema(BaseModel):
        value: str

    fake = FakeLlama()
    ce = {"provider": "gguf_native", "model": fake}
    fn = cg.make_constrained_generate_fn(ce, ProbeSchema)
    assert (
        fn([{"role": "user", "content": "x"}],
           temperature=0.1, max_new_tokens=32) == "the answer"
    )
    assert fake.calls[-1]["response_format"]["type"] == "json_object"
    assert fake.calls[-1]["response_format"]["schema"]["title"] == "ProbeSchema"


def test_generate_rejects_raw_grammar():
    fake = FakeLlama()
    ce = {"provider": "gguf_native", "model": fake}
    with pytest.raises(GGF.GGUFNativeConfigError):
        GGF.GGUFNativeBackend().generate(
            ce, [{"role": "user", "content": "x"}], grammar='root ::= "a"',
        )
    assert fake.calls == []


def test_generate_fail_closed_on_call_error():
    class Broken:
        def create_chat_completion(self, **_kwargs):
            raise RuntimeError("boom")

    with pytest.raises(GGF.GGUFNativeCallFailedError):
        GGF.GGUFNativeBackend().generate(
            {"provider": "gguf_native", "model": Broken()},
            [{"role": "user", "content": "x"}],
        )


def test_validate_gemma_gguf_ready_checks_file_without_import(monkeypatch, tmp_path):
    model_path = tmp_path / "custom.gguf"
    model_path.write_bytes(b"gguf")
    monkeypatch.setenv("GEMMA4_12B_GGUF_PATH", str(model_path))
    status = GGF.validate_gemma_gguf_ready(require_binding=False)
    assert status["ok"] is True
    assert status["model_exists"] is True
    assert status["binding_available"] is None


def test_default_named_partial_file_is_rejected(monkeypatch, tmp_path):
    model_path = tmp_path / GGF.DEFAULT_GGUF_FILENAME
    model_path.write_bytes(b"partial")
    monkeypatch.setenv("GEMMA4_12B_GGUF_PATH", str(model_path))
    status = GGF.validate_gemma_gguf_ready(require_binding=False)
    assert status["ok"] is False
    assert "incomplete GGUF file" in status["error"]
    with pytest.raises(GGF.GGUFNativeConfigError, match="Incomplete Gemma"):
        GGF.GGUFNativeBackend().load(GGF.ROW_ID, _row())


def test_dispatch_table_registers_gguf_native():
    from nodes import _otr_model_runtime as RT

    assert GGF.GGUF_BACKEND_KEY in RT.BACKENDS_BY_KEY
    assert isinstance(
        RT.BACKENDS_BY_KEY[GGF.GGUF_BACKEND_KEY],
        GGF.GGUFNativeBackend,
    )


def test_unload_closes_llama_cpp_model():
    fake = FakeLlama()
    GGF.GGUFNativeBackend().unload({"model": fake})
    assert fake.closed is True


def test_source_is_ascii_no_em_dash():
    src = pathlib.Path(GGF.__file__).read_text(encoding="utf-8")
    src.encode("ascii")
    assert "\u2014" not in src
