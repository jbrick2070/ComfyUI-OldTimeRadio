"""tests/test_writer_llm_unload.py -- S3 post-script LLM VRAM reclaim.

Unit-tests the env gate + the never-raises guarantee of the writer's
after-script LLM unload. The actual VRAM reclaim needs an operator GPU smoke
(this path does nothing observable headless), so this pins the LOGIC only.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_NODES_DIR = _REPO_ROOT / "nodes"
for _p in (_REPO_ROOT, _NODES_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import _otr_writer_vram as _VRAM  # noqa: E402


def _set_llm_cache(entry):
    import _otr_model_loader as ML
    ML.LLM_CACHE.clear()
    ML.LLM_CACHE.update({
        "model_id": entry.get("model_id") if isinstance(entry, dict) else None,
        "slot": entry.get("slot") if isinstance(entry, dict) else None,
        "cache_entry": entry,
    })
    return ML


def test_unload_skips_when_env_off(monkeypatch):
    monkeypatch.setenv("OTR_WRITER_UNLOAD_AFTER_SCRIPT", "0")
    calls = []
    status = _VRAM.unload_writer_llm_after_script(lambda: calls.append(1))
    assert status == "skipped_env"
    assert calls == []  # the escape hatch must not unload


def test_unload_calls_when_env_unset(monkeypatch):
    monkeypatch.delenv("OTR_WRITER_UNLOAD_AFTER_SCRIPT", raising=False)
    calls = []
    status = _VRAM.unload_writer_llm_after_script(lambda: calls.append(1))
    assert status == "unloaded"
    assert calls == [1]


def test_unload_calls_when_env_on(monkeypatch):
    monkeypatch.setenv("OTR_WRITER_UNLOAD_AFTER_SCRIPT", "1")
    calls = []
    status = _VRAM.unload_writer_llm_after_script(lambda: calls.append(1))
    assert status == "unloaded"
    assert calls == [1]


def test_unload_never_raises_on_teardown_failure(monkeypatch):
    monkeypatch.delenv("OTR_WRITER_UNLOAD_AFTER_SCRIPT", raising=False)

    def boom():
        raise RuntimeError("cuda context gone")

    # PD1: a teardown failure must NOT propagate -- the writer's output stands.
    status = _VRAM.unload_writer_llm_after_script(boom)
    assert status == "skipped_error"


def test_unload_skips_when_no_local_writer_cache(monkeypatch):
    """All-cloud LLM runs leave the local singleton empty; the writer must not
    call the full local teardown, because that imports torch just to clear an
    already-empty allocator."""
    monkeypatch.delenv("OTR_WRITER_UNLOAD_AFTER_SCRIPT", raising=False)
    ML = _set_llm_cache(None)
    calls = []
    monkeypatch.setattr(ML, "unload_llm", lambda: calls.append(1))
    status = _VRAM.unload_writer_llm_after_script()
    assert status == "skipped_no_local"
    assert calls == []


def test_unload_skips_remote_provider_cache_entries(monkeypatch):
    monkeypatch.delenv("OTR_WRITER_UNLOAD_AFTER_SCRIPT", raising=False)
    for provider in ("openrouter", "comfy_credits"):
        ML = _set_llm_cache({
            "provider": provider,
            "model_id": f"{provider}:slot-a",
            "slot": "creative",
        })
        calls = []
        monkeypatch.setattr(ML, "unload_llm", lambda: calls.append(1))
        status = _VRAM.unload_writer_llm_after_script()
        assert status == "skipped_no_local"
        assert calls == []


def test_unload_calls_canonical_teardown_for_local_cache(monkeypatch):
    monkeypatch.delenv("OTR_WRITER_UNLOAD_AFTER_SCRIPT", raising=False)
    ML = _set_llm_cache({
        "provider": "local",
        "model_id": "mistralai/Mistral-Nemo-Instruct-2407",
        "slot": "creative",
    })
    calls = []
    monkeypatch.setattr(ML, "unload_llm", lambda: calls.append(1))
    status = _VRAM.unload_writer_llm_after_script()
    assert status == "unloaded"
    assert calls == [1]
