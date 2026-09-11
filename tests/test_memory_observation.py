"""Read-only process/Metal observations at actual native lifecycle boundaries."""
import builtins
import json
import runpy
import sys
from types import SimpleNamespace

from nodes import _vram_log as memory


def test_memory_observer_is_import_safe(monkeypatch):
    original = builtins.__import__
    imported = []
    def observe(name, *args, **kwargs):
        if name in ("torch", "psutil"):
            imported.append(name)
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", observe)
    runpy.run_path(memory.__file__)
    assert not imported


def test_missing_memory_counters_are_unknown_and_logging_failure_is_harmless(monkeypatch):
    monkeypatch.setitem(sys.modules, "psutil", None)
    monkeypatch.setitem(sys.modules, "torch", None)
    def unavailable(*args, **kwargs):
        raise OSError("log not writable")
    monkeypatch.setattr(memory, "_write_runtime_log", unavailable)
    row = memory.memory_snapshot("generation_returned", model_id="local/test")
    assert row["process_rss_bytes"] is None
    assert row["mps_current_bytes"] is None and row["mps_driver_bytes"] is None
    assert row["model_id"] == "local/test" and row["timestamp_utc"].endswith("+00:00")
    json.dumps(row)


def test_counters_are_independent_zero_is_measured_and_no_allocator_mutation(monkeypatch):
    actions, lines = [], []
    def failed_counter():
        raise RuntimeError("counter unavailable")
    mps = SimpleNamespace(current_allocated_memory=failed_counter,
                          driver_allocated_memory=lambda: 0,
                          empty_cache=lambda: actions.append("empty"),
                          synchronize=lambda: actions.append("sync"))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        mps=mps, backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True))))
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(
        Process=lambda: SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=123456))))
    monkeypatch.setattr(memory, "_write_runtime_log", lines.append)
    row = memory.memory_snapshot("boundary")
    assert row["process_rss_bytes"] == 123456
    assert row["mps_current_bytes"] is None and row["mps_driver_bytes"] == 0
    assert not actions
    assert json.loads(lines[0].removeprefix("MEMORY_SNAPSHOT ")) == row


def test_retirement_observes_before_move_after_move_and_existing_flush(monkeypatch):
    import gc
    import torch
    from nodes import _otr_model_loader as loader
    events = []
    monkeypatch.setattr(memory, "memory_snapshot",
                        lambda phase, **kwargs: events.append(phase))
    monkeypatch.setattr(gc, "collect", lambda: events.append("collect"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: events.append("empty"))
    monkeypatch.setattr(torch.mps, "synchronize", lambda: events.append("sync"))
    model = SimpleNamespace(to=lambda device: events.append("move:" + device))
    entry = {"model": model, "model_id": "native/test"}
    loader._teardown_gpu_for_entry(entry)
    assert events == ["llm_retirement_before", "move:cpu", "llm_retirement_after_cpu_move",
                      "collect", "empty", "sync", "llm_retirement_after_allocator_flush"]
    assert entry["model"] is model  # The caller still owns a reference; no release claim.
