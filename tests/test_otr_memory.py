"""
tests/test_otr_memory.py -- BUG-LOCAL-030 long-form hardening regression
============================================================================

Covers ``nodes/_otr_memory.py`` -- the shared DRAM/VRAM hygiene
helpers used at composite phase boundaries. There is one live
``phase_gc`` call site today (Composite -> PostUpscaleProcgenBlend);
the BatchHumo and RTXUpscale barriers went away with their nodes.

Acceptance gates:
  * ``phase_gc`` never raises (best-effort even when torch is missing
    or empty_cache fails)
  * ``dram_canary`` returns (True, reason) when psutil is unavailable
    or the syscall fails -- degrades OPEN, never blocks a render
  * ``dram_canary`` returns (False, reason) only when psutil
    successfully reports below the threshold
  * Default threshold is 6.0 GB per the round-robin recommendation
"""
from __future__ import annotations

from unittest import mock

from nodes import _otr_memory as M


class TestPhaseGc:
    def test_never_raises(self):
        # Just call it -- should not raise.
        M.phase_gc("test-label")

    def test_handles_missing_torch(self):
        # Force the inner torch import to fail; helper should still return.
        with mock.patch.dict("sys.modules", {"torch": None}):
            try:
                M.phase_gc("torch-missing")
            except Exception as exc:
                raise AssertionError(
                    f"phase_gc must not raise when torch is missing: {exc}"
                )


class TestDramCanary:
    def test_default_threshold_is_six_gb(self):
        assert M.DEFAULT_MIN_FREE_DRAM_GB == 6.0

    def test_degrades_open_when_psutil_missing(self):
        # Patch the psutil import inside the helper to raise ImportError.
        with mock.patch.dict("sys.modules", {"psutil": None}):
            ok, reason = M.dram_canary(min_free_gb=1.0, label="no-psutil")
        assert ok is True
        assert "psutil" in reason.lower()

    def test_returns_false_when_below_threshold(self):
        # Mock psutil.virtual_memory() to report 2 GB available.
        fake_psutil = mock.MagicMock()
        fake_vm = mock.MagicMock()
        fake_vm.available = int(2 * 1024 ** 3)  # 2 GB in bytes
        fake_psutil.virtual_memory.return_value = fake_vm
        with mock.patch.dict("sys.modules", {"psutil": fake_psutil}):
            ok, reason = M.dram_canary(min_free_gb=6.0, label="low-mem")
        assert ok is False
        assert "2.00" in reason or "2.0" in reason
        assert "6.00" in reason

    def test_returns_true_when_above_threshold(self):
        fake_psutil = mock.MagicMock()
        fake_vm = mock.MagicMock()
        fake_vm.available = int(20 * 1024 ** 3)  # 20 GB in bytes
        fake_psutil.virtual_memory.return_value = fake_vm
        with mock.patch.dict("sys.modules", {"psutil": fake_psutil}):
            ok, reason = M.dram_canary(min_free_gb=6.0, label="plenty")
        assert ok is True
        assert "20.00" in reason or "20.0" in reason

    def test_degrades_open_when_syscall_fails(self):
        fake_psutil = mock.MagicMock()
        fake_psutil.virtual_memory.side_effect = OSError("simulated syscall failure")
        with mock.patch.dict("sys.modules", {"psutil": fake_psutil}):
            ok, reason = M.dram_canary(min_free_gb=6.0, label="syscall-fail")
        assert ok is True
        assert "failed" in reason.lower()
