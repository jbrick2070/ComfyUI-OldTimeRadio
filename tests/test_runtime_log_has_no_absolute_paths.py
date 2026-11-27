"""otr_runtime.log carries no absolute paths.

The runtime log is a plain file the pack appends to itself, under ComfyUI's
output folder (``otr/episodes/_shared/state/otr_runtime.log``). ``/view``
serves that folder with no login, and because the file is written directly --
not through ``logging`` -- the log-record redaction in ``_otr_error_text``
never saw its lines.

THE DEFECT THIS PINS (shipped through 2.3.64): the writer LLM's loader wrote
``HF_HOME resolved -> <absolute path>`` and ``Loading from canonical snapshot:
<absolute path>`` into it, naming the install folder and the account.

Both writers of the file are exercised against a temporary path. CPU only.
UTF-8 no BOM.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from nodes import _vram_log
from nodes import story_orchestrator

_LINES = (
    "HF_HOME resolved -> /home/someone/ComfyUI-Models/huggingface",
    r"Loading from canonical snapshot: C:\Users\someone\models\LLM\Qwen--Qwen3.5-4B\snapshots\abc123",
    "LLM loaded: Qwen/Qwen3.5-4B",
    "VRAM_SNAPSHOT phase=writer peak_gb=6.41",
)


def _assert_clean(text: str) -> None:
    assert "someone" not in text
    assert "/home/" not in text and "C:\\Users" not in text
    # What the video engine's status line reads back out of this file survives.
    assert "LLM loaded: Qwen/Qwen3.5-4B" in text
    assert "peak_gb=6.41" in text
    assert "huggingface" in text and "abc123" in text


def test_the_orchestrator_heartbeat_redacts_paths(tmp_path, monkeypatch):
    target = tmp_path / "state" / "otr_runtime.log"
    monkeypatch.setattr(story_orchestrator._OTR_PATHS, "otr_runtime_log_path",
                        lambda: Path(target))
    for line in _LINES:
        story_orchestrator._runtime_log(line)
    _assert_clean(target.read_text(encoding="utf-8"))


def test_the_vram_telemetry_writer_redacts_paths(tmp_path, monkeypatch):
    target = tmp_path / "state" / "otr_runtime.log"
    monkeypatch.setattr(_vram_log, "_runtime_log_path", lambda: str(target))
    for line in _LINES:
        _vram_log._write_runtime_log(line)
    _assert_clean(target.read_text(encoding="utf-8"))


@pytest.mark.parametrize("writer", [story_orchestrator._runtime_log,
                                    _vram_log._write_runtime_log])
def test_a_writer_never_raises(writer, monkeypatch):
    """A monitoring write must never fail an episode, redaction included."""
    monkeypatch.setattr(story_orchestrator._OTR_PATHS, "otr_runtime_log_path",
                        lambda: (_ for _ in ()).throw(OSError("no output tree")))
    monkeypatch.setattr(_vram_log, "_runtime_log_path",
                        lambda: (_ for _ in ()).throw(OSError("no output tree")))
    writer(object())
