"""Truthful sampled VRAM telemetry, without pytest, NVML, Comfy or a GPU.

Import the real pure leaves under a private namespace; only NVML responses and
sampler scheduling are faked. No source extraction or replacement node package.
"""
import importlib
from pathlib import Path
import sys
import threading
import types
import unittest
from unittest import mock


REPO = Path(__file__).resolve().parents[1]
MB = 1024 * 1024


class ParkedThread:
    """Explicitly scheduled test thread: start never executes background work."""
    def __init__(self, target, daemon):
        self.target = target
        self.daemon = daemon
        self.started = False
        self.join_timeout = None

    def start(self):
        self.started = True

    def join(self, timeout=None):
        self.join_timeout = timeout


class VramTelemetryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.namespace = "_vram_telemetry_stdlib"
        if cls.namespace in sys.modules:
            raise AssertionError("Telemetry test namespace already in use")
        cls.addClassCleanup(cls.cleanup_namespace)
        for name, path in ((cls.namespace, REPO / "nodes"),
                           (cls.namespace + "._otr_video_engines",
                            REPO / "nodes/_otr_video_engines")):
            package = types.ModuleType(name)
            package.__path__ = [str(path)]
            sys.modules[name] = package

        class StdlibBoundary:
            blocked = []

            def find_spec(self, fullname, path=None, target=None):
                root = fullname.partition(".")[0]
                if root != cls.namespace and root not in sys.stdlib_module_names:
                    self.blocked.append(fullname)
                    raise AssertionError("Non-stdlib telemetry import: " + fullname)
                return None

        boundary = StdlibBoundary()
        sys.meta_path.insert(0, boundary)
        try:
            cls.gr = importlib.import_module(cls.namespace + "._otr_shared.gpu_residency")
            cls.mc = importlib.import_module(cls.namespace + "._otr_video_engines.motion_common")
        finally:
            sys.meta_path.remove(boundary)
        if boundary.blocked:
            raise AssertionError("Blocked import attempts: " + repr(boundary.blocked))

    @classmethod
    def cleanup_namespace(cls):
        for name in list(sys.modules):
            if name == cls.namespace or name.startswith(cls.namespace + "."):
                del sys.modules[name]

    def binding(self, used_mb=1234):
        binding = types.ModuleType("pynvml")
        binding.nvmlInit = mock.Mock()
        binding.nvmlDeviceGetHandleByIndex = mock.Mock(return_value="fake-device")
        binding.nvmlDeviceGetMemoryInfo = mock.Mock(
            return_value=types.SimpleNamespace(used=used_mb * MB))
        binding.nvmlShutdown = mock.Mock()
        return binding

    def sample(self, binding, device_index=0):
        with mock.patch.dict(sys.modules, {"pynvml": binding}):
            return self.gr.sample_used_mb(device_index)

    def test_optional_helper_is_public(self):
        self.assertIn("sample_used_mb", self.gr.__all__)

    def test_missing_binding_is_unknown(self):
        self.assertIsNone(self.sample(None))

    def test_failed_init_does_not_shutdown(self):
        binding = self.binding()
        binding.nvmlInit.side_effect = RuntimeError("init failed")
        self.assertIsNone(self.sample(binding))
        binding.nvmlShutdown.assert_not_called()
        binding.nvmlDeviceGetHandleByIndex.assert_not_called()

    def test_failed_handle_shuts_down(self):
        binding = self.binding()
        binding.nvmlDeviceGetHandleByIndex.side_effect = RuntimeError("handle failed")
        self.assertIsNone(self.sample(binding))
        binding.nvmlShutdown.assert_called_once_with()
        binding.nvmlDeviceGetMemoryInfo.assert_not_called()

    def test_failed_query_shuts_down_and_is_unknown(self):
        binding = self.binding()
        binding.nvmlDeviceGetMemoryInfo.side_effect = RuntimeError("query failed")
        self.assertIsNone(self.sample(binding))
        binding.nvmlShutdown.assert_called_once_with()

    def test_successful_zero_is_a_measurement(self):
        binding = self.binding(0)
        self.assertEqual(self.sample(binding), 0)
        binding.nvmlShutdown.assert_called_once_with()

    def test_positive_measurement_and_device_forwarding(self):
        binding = self.binding(14500)
        self.assertEqual(self.sample(binding, 2), 14500)
        binding.nvmlDeviceGetHandleByIndex.assert_called_once_with(2)
        binding.nvmlDeviceGetMemoryInfo.assert_called_once_with("fake-device")
        binding.nvmlShutdown.assert_called_once_with()

    def test_shutdown_failure_does_not_discard_valid_measurement(self):
        binding = self.binding(3072)
        binding.nvmlShutdown.side_effect = RuntimeError("shutdown failed")
        self.assertEqual(self.sample(binding), 3072)

    def test_integer_mebibyte_conversion_is_preserved(self):
        binding = self.binding()
        binding.nvmlDeviceGetMemoryInfo.return_value.used = 5 * MB + 123
        self.assertEqual(self.sample(binding), 5)

    def test_legacy_probe_contract_for_missing_zero_and_positive(self):
        for binding, expected in ((None, 0), (self.binding(0), 0),
                                  (self.binding(14500), 14500)):
            with self.subTest(expected=expected), mock.patch.dict(sys.modules, {"pynvml": binding}):
                self.assertEqual(self.gr.probe_used_mb(0), expected)

    def test_legacy_query_failure_still_returns_integer_zero(self):
        binding = self.binding()
        binding.nvmlDeviceGetMemoryInfo.side_effect = RuntimeError("query failed")
        with mock.patch.dict(sys.modules, {"pynvml": binding}):
            self.assertEqual(self.gr.probe_used_mb(), 0)

    def test_existing_floor_interfaces_fail_closed_when_unavailable(self):
        with mock.patch.dict(sys.modules, {"pynvml": None}):
            self.assertFalse(self.gr.nvml_available())
            self.assertFalse(self.gr.wait_until_below_mb(100, sleep_s=0))
            self.assertFalse(self.gr.wait_until_stable(sleep_s=0))

    def test_existing_numeric_floor_and_stability_behavior(self):
        with mock.patch.object(self.gr, "nvml_available", return_value=True):
            with mock.patch.object(self.gr, "probe_used_mb", side_effect=[200, 50]):
                self.assertTrue(self.gr.wait_until_below_mb(100, attempts=2, sleep_s=0))
            with mock.patch.object(self.gr, "probe_used_mb", side_effect=[200, 200]):
                self.assertFalse(self.gr.wait_until_below_mb(100, attempts=2, sleep_s=0))
            with mock.patch.object(self.gr, "probe_used_mb", side_effect=[200, 190]):
                self.assertTrue(self.gr.wait_until_stable(attempts=1, sleep_s=0, delta_mb=20))

    def test_motion_helper_does_not_use_legacy_zero_fallback(self):
        binding = self.binding()
        binding.nvmlDeviceGetMemoryInfo.side_effect = RuntimeError("query failed")
        with mock.patch.dict(sys.modules, {"pynvml": binding}):
            self.assertIsNone(self.mc.vram_used_mb())

    def test_motion_helper_preserves_numeric_samples(self):
        for value in (0, 14500):
            with self.subTest(value=value), mock.patch.dict(sys.modules, {"pynvml": self.binding(value)}):
                self.assertEqual(self.mc.vram_used_mb(), value)

    def peak_sequence(self, samples):
        # Drive the real loop deterministically: one initial read, then one read
        # per successful interval wait. No wall-clock sleeps or hardware access.
        class ScriptedEvent:
            def __init__(self):
                self.closed = False
                self.remaining = len(samples) - 1

            def is_set(self):
                return self.closed

            def set(self):
                self.closed = True

            def wait(self, timeout):
                if self.remaining and not self.closed:
                    self.remaining -= 1
                    return False
                self.closed = True
                return True

        class InlineThread(ParkedThread):
            def start(self):
                self.started = True
                self.target()

        with mock.patch.object(threading, "Event", ScriptedEvent), \
                mock.patch.object(threading, "Thread", InlineThread), \
                mock.patch.object(self.mc, "vram_used_mb", side_effect=samples) as sample:
            probe = self.mc.VramPeakProbe().start()
            value = probe.stop()
        self.assertEqual(sample.call_count, len(samples))
        return value

    def test_all_failed_samples_are_unknown(self):
        self.assertIsNone(self.peak_sequence([None, None, None]))

    def test_sampled_zero_stays_zero(self):
        self.assertEqual(self.peak_sequence([0]), 0)

    def test_positive_sample_maximum_is_preserved(self):
        self.assertEqual(self.peak_sequence([100, 300, 200]), 300)

    def test_initial_failure_can_recover(self):
        self.assertEqual(self.peak_sequence([None, 200, None, 300]), 300)

    def test_later_failure_does_not_erase_observed_maximum(self):
        self.assertEqual(self.peak_sequence([300, None]), 300)

    def test_first_synchronous_sample_is_retained(self):
        with mock.patch.object(threading, "Thread", ParkedThread), \
                mock.patch.object(self.mc, "vram_used_mb", return_value=7123) as sample:
            probe = self.mc.VramPeakProbe().start()
            self.assertEqual(probe.stop(), 7123)
            self.assertEqual(probe.stop(), 7123)
            sample.assert_called_once_with()
            self.assertEqual(probe._thread.join_timeout, 2.0)

    def test_stop_before_start_is_unknown_and_terminal(self):
        probe = self.mc.VramPeakProbe()
        self.assertIsNone(probe.stop())
        with mock.patch.object(self.mc, "vram_used_mb") as sample:
            self.assertIs(probe.start(), probe)
            sample.assert_not_called()

    def test_repeated_start_does_not_spawn_duplicate_sampler(self):
        with mock.patch.object(threading, "Thread", side_effect=ParkedThread) as factory, \
                mock.patch.object(self.mc, "vram_used_mb", return_value=500) as sample:
            probe = self.mc.VramPeakProbe().start()
            self.assertIs(probe.start(), probe)
            self.assertEqual(probe.stop(), 500)
            factory.assert_called_once()
            sample.assert_called_once_with()

    def test_late_inflight_read_cannot_mutate_stopped_peak(self):
        entered, release = threading.Event(), threading.Event()
        calls = []

        def sample():
            calls.append(True)
            if len(calls) == 1:
                return 100
            entered.set()
            if not release.wait(2):
                raise AssertionError("Test did not release fake memory query")
            return 999

        with mock.patch.object(self.mc, "vram_used_mb", side_effect=sample):
            probe = self.mc.VramPeakProbe(interval_s=0).start()
            try:
                self.assertTrue(entered.wait(2), "Background sampler did not start")
                # Model a query outlasting the bounded join without sleeping 2s.
                with mock.patch.object(probe._thread, "join"):
                    self.assertEqual(probe.stop(), 100)
                release.set()
                probe._thread.join(timeout=2)
                self.assertFalse(probe._thread.is_alive())
                self.assertEqual(probe.peak_mb, 100)
                self.assertEqual(probe.stop(), 100)
            finally:
                release.set()
                probe.stop()


if __name__ == "__main__":
    unittest.main()
