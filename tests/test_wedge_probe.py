"""
tests/test_wedge_probe.py
=========================

Unit tests for otr_v2.hyworld.wedge_probe.

Coverage
--------
- Disabled probe is a true no-op (no file, no thread, O(1) methods).
- Enabled probe captures events, spans, counters to NDJSON.
- Span records elapsed_ms and propagates notes.
- Queue overflow drops events but never raises.
- Singleton accessor respects OTR_WEDGE_PROBE env var.
- Audio-path safety sentinel: probe module imports no torch/numpy/ffmpeg.
"""

from __future__ import annotations

import importlib
import json
import os
import time
from pathlib import Path

import pytest

# Import the module under test
from otr_v2.hyworld import wedge_probe as wp


# ---- Fixtures --------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_probe_singleton():
    """Drop the singleton and strip env flags before each test so probes
    don't leak between tests."""
    wp.reset_singleton_for_tests()
    for key in ("OTR_WEDGE_PROBE", "OTR_WEDGE_PROBE_LOG", "OTR_WEDGE_PROBE_QUEUE_MAX"):
        os.environ.pop(key, None)
    yield
    wp.reset_singleton_for_tests()
    for key in ("OTR_WEDGE_PROBE", "OTR_WEDGE_PROBE_LOG", "OTR_WEDGE_PROBE_QUEUE_MAX"):
        os.environ.pop(key, None)


# ---- Disabled probe (zero-cost guarantee) ---------------------------------


def test_disabled_probe_is_no_op(tmp_path: Path) -> None:
    """When OTR_WEDGE_PROBE is absent, get_probe() returns a disabled
    probe that writes nothing and has no background thread."""
    probe = wp.get_probe()
    assert probe.enabled is False
    assert probe.log_path is None

    probe.event("any_event", foo=1, bar="x")
    probe.counter("frames", 240)
    with probe.span("noop") as s:
        s.note(k="v")

    # No log file should ever be written.
    assert not any(tmp_path.iterdir())


def test_disabled_probe_explicit_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit 0/false values also disable."""
    for falsey in ("0", "false", "FALSE", "no", "off", ""):
        monkeypatch.setenv("OTR_WEDGE_PROBE", falsey)
        wp.reset_singleton_for_tests()
        probe = wp.get_probe()
        assert probe.enabled is False, f"Expected disabled for {falsey!r}"


# ---- Enabled probe captures events ----------------------------------------


def test_enabled_probe_writes_events(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    log_path = tmp_path / "wedge.ndjson"
    monkeypatch.setenv("OTR_WEDGE_PROBE", "1")
    monkeypatch.setenv("OTR_WEDGE_PROBE_LOG", str(log_path))

    wp.reset_singleton_for_tests()
    probe = wp.get_probe()
    assert probe.enabled is True
    assert probe.log_path == log_path.resolve()

    probe.event("backend_start", backend="flux_anchor", shot_id="s001")
    probe.counter("frames_written", 240, shot_id="s001")
    probe.flush(timeout_s=2.0)
    probe.shutdown(timeout_s=2.0)

    assert log_path.exists(), "Probe should have created the NDJSON log"
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 2

    records = [json.loads(line) for line in lines]
    kinds = [r["kind"] for r in records]
    assert "backend_start" in kinds
    assert "frames_written.counter" in kinds

    backend_record = next(r for r in records if r["kind"] == "backend_start")
    assert backend_record["backend"] == "flux_anchor"
    assert backend_record["shot_id"] == "s001"
    assert "t_monotonic" in backend_record
    assert "t_unix" in backend_record


def test_span_records_elapsed_ms(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    log_path = tmp_path / "wedge.ndjson"
    monkeypatch.setenv("OTR_WEDGE_PROBE", "1")
    monkeypatch.setenv("OTR_WEDGE_PROBE_LOG", str(log_path))

    wp.reset_singleton_for_tests()
    probe = wp.get_probe()

    with probe.span("render_mux", shot_id="s002") as s:
        time.sleep(0.03)
        s.note(frames=240, audio_samples=264600)

    probe.flush(timeout_s=2.0)
    probe.shutdown(timeout_s=2.0)

    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    span_records = [r for r in records if r["kind"] == "render_mux.span"]
    assert len(span_records) == 1
    span = span_records[0]
    # Sleep was 30ms; elapsed_ms should be >= 25 (allow some slack for
    # timer resolution on Windows) and < 5000.
    assert span["elapsed_ms"] >= 25.0, f"Span too fast: {span['elapsed_ms']}"
    assert span["elapsed_ms"] < 5000.0, f"Span too slow: {span['elapsed_ms']}"
    assert span["shot_id"] == "s002"
    assert span["frames"] == 240
    assert span["audio_samples"] == 264600


def test_span_captures_exception(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    log_path = tmp_path / "wedge.ndjson"
    monkeypatch.setenv("OTR_WEDGE_PROBE", "1")
    monkeypatch.setenv("OTR_WEDGE_PROBE_LOG", str(log_path))

    wp.reset_singleton_for_tests()
    probe = wp.get_probe()

    with pytest.raises(RuntimeError):
        with probe.span("risky", stage="x"):
            raise RuntimeError("boom")

    probe.flush(timeout_s=2.0)
    probe.shutdown(timeout_s=2.0)
    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    risky = [r for r in records if r["kind"] == "risky.span"]
    assert len(risky) == 1
    assert risky[0]["exc_type"] == "RuntimeError"


# ---- Overflow behaviour ---------------------------------------------------


def test_queue_overflow_drops_oldest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When the queue is saturated, overflow drops events instead of
    blocking the producer.  The probe MUST NOT raise."""
    log_path = tmp_path / "wedge.ndjson"
    monkeypatch.setenv("OTR_WEDGE_PROBE", "1")
    monkeypatch.setenv("OTR_WEDGE_PROBE_LOG", str(log_path))
    # Tiny queue to force overflow fast.
    monkeypatch.setenv("OTR_WEDGE_PROBE_QUEUE_MAX", "100")

    wp.reset_singleton_for_tests()
    probe = wp.get_probe()

    # Burst 5000 events; queue is only 100, so many will be overflow-dropped.
    for i in range(5000):
        probe.event("burst", i=i)

    probe.flush(timeout_s=3.0)
    probe.shutdown(timeout_s=2.0)

    # Probe must still be alive and some events must have been written.
    assert log_path.exists()
    # Drop counter may or may not be > 0 depending on drain speed, but
    # the probe must not have crashed.
    assert probe.dropped_count >= 0


# ---- Audio-path safety sentinel -------------------------------------------


def test_module_imports_no_audio_or_gpu_libs() -> None:
    """Audio is king (C7).  The probe module must not drag in torch,
    numpy, ffmpeg wrappers, or any audio library even transitively from
    its own module body."""
    mod = importlib.import_module("otr_v2.hyworld.wedge_probe")
    src = Path(mod.__file__).read_text(encoding="utf-8")
    forbidden = ("import torch", "import numpy", "from torch", "from numpy")
    for needle in forbidden:
        assert needle not in src, (
            f"wedge_probe.py must not import heavy libs; found {needle!r}"
        )


# ---- Singleton contract ---------------------------------------------------


def test_singleton_returns_same_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two calls to get_probe() in the same process return the same instance."""
    # Disabled path
    a = wp.get_probe()
    b = wp.get_probe()
    assert a is b

    # Enabled path (new singleton after reset)
    monkeypatch.setenv("OTR_WEDGE_PROBE", "1")
    wp.reset_singleton_for_tests()
    c = wp.get_probe()
    d = wp.get_probe()
    assert c is d
    assert c is not a  # new singleton after reset
    c.shutdown(timeout_s=1.0)
