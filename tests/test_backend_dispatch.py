"""
tests/test_backend_dispatch.py  --  Day 1 harness regression
===========================================================

Torch-free unit tests for the video stack Day 1 harness:

* ``backends`` package registry (resolve / list / KeyError on unknown).
* ``PlaceholderTestBackend.run`` writes STATUS.json READY + per-shot PNGs.
* ``cooldown_gate`` gracefully ships-and-warns when LHM is unreachable.
* Bridge ``backend=`` arg plumbs through to the sidecar env dict.

Mirrors the pattern in ``tests/test_anchor_gen.py`` -- no GPU required.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


class BackendRegistryTests(unittest.TestCase):
    def test_resolve_placeholder_test(self):
        from otr_v2.visual import backends as _backends
        backend = _backends.resolve("placeholder_test")
        self.assertEqual(backend.name, "placeholder_test")
        self.assertTrue(hasattr(backend, "run"))

    def test_resolve_case_insensitive(self):
        from otr_v2.visual import backends as _backends
        backend = _backends.resolve("  PlaceHolder_Test  ")
        self.assertEqual(backend.name, "placeholder_test")

    def test_resolve_unknown_raises(self):
        from otr_v2.visual import backends as _backends
        with self.assertRaises(KeyError):
            _backends.resolve("flux_anchor_not_registered_yet")

    def test_list_backends_sorted(self):
        from otr_v2.visual import backends as _backends
        names = _backends.list_backends()
        self.assertEqual(names, sorted(names))
        self.assertIn("placeholder_test", names)


class PlaceholderBackendTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmp = Path(tempfile.mkdtemp(prefix="otr_placeholder_"))
        # Emulate repo layout: <root>/io/visual_in/<job>/shotlist.json
        # out_dir_for walks parent.parent.parent, so job_dir must be
        # 3 levels deep under a fake root.
        self.job_id = "hw_testjob_001"
        self.fake_root = self._tmp / "repo"
        self.in_dir = self.fake_root / "io" / "visual_in" / self.job_id
        self.out_dir = self.fake_root / "io" / "visual_out" / self.job_id
        self.in_dir.mkdir(parents=True)

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _write_shotlist(self, shots):
        (self.in_dir / "shotlist.json").write_text(
            json.dumps({"shots": shots}), encoding="utf-8",
        )

    def test_run_writes_ready_status_and_pngs(self):
        from otr_v2.visual.backends.placeholder_test import PlaceholderTestBackend
        self._write_shotlist([
            {"shot_id": "shot_000", "env_prompt": "a", "camera": "push in",
             "duration_sec": 5.0},
            {"shot_id": "shot_001", "env_prompt": "b", "camera": "static",
             "duration_sec": 7.0},
        ])
        PlaceholderTestBackend().run(self.in_dir)
        status_path = self.out_dir / "STATUS.json"
        self.assertTrue(status_path.exists(), "STATUS.json not written")
        status = json.loads(status_path.read_text(encoding="utf-8"))
        self.assertEqual(status["status"], "READY")
        self.assertEqual(status.get("backend"), "placeholder_test")

        # Per-shot PNGs and meta.json
        for shot_id in ("shot_000", "shot_001"):
            png = self.out_dir / shot_id / "render.png"
            meta = self.out_dir / shot_id / "meta.json"
            self.assertTrue(png.exists(), f"{shot_id} render.png missing")
            self.assertGreater(png.stat().st_size, 0)
            # PNG magic bytes
            self.assertEqual(png.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertTrue(meta.exists())
            meta_data = json.loads(meta.read_text(encoding="utf-8"))
            self.assertEqual(meta_data["backend"], "placeholder_test")
            self.assertEqual(meta_data["shot_id"], shot_id)

    def test_run_empty_shotlist_writes_error(self):
        from otr_v2.visual.backends.placeholder_test import PlaceholderTestBackend
        self._write_shotlist([])
        PlaceholderTestBackend().run(self.in_dir)
        status = json.loads(
            (self.out_dir / "STATUS.json").read_text(encoding="utf-8"),
        )
        self.assertEqual(status["status"], "ERROR")
        self.assertIn("zero shots", status["detail"])

    def test_run_missing_shotlist_writes_error(self):
        from otr_v2.visual.backends.placeholder_test import PlaceholderTestBackend
        # no shotlist.json written
        PlaceholderTestBackend().run(self.in_dir)
        status = json.loads(
            (self.out_dir / "STATUS.json").read_text(encoding="utf-8"),
        )
        self.assertEqual(status["status"], "ERROR")
        self.assertIn("FileNotFoundError", status["detail"])


class CooldownGateTests(unittest.TestCase):
    def test_unreachable_lhm_proceeds(self):
        """LHM down -> ship-and-warn, not block."""
        from otr_v2.visual.backends import _base
        with mock.patch.object(_base, "_read_lhm_gpu_temp", return_value=None):
            ok, reason = _base.cooldown_gate(max_wait_s=0.1)
        self.assertTrue(ok)
        self.assertEqual(reason, "lhm_unreachable")

    def test_cool_gpu_proceeds_immediately(self):
        from otr_v2.visual.backends import _base
        with mock.patch.object(_base, "_read_lhm_gpu_temp", return_value=55.0):
            ok, reason = _base.cooldown_gate(max_wait_s=0.1, temp_threshold_c=82.0)
        self.assertTrue(ok)
        self.assertTrue(reason.startswith("cool:"))

    def test_hot_gpu_times_out_and_proceeds(self):
        """Hot card -> returns False after max_wait, never blocks forever."""
        from otr_v2.visual.backends import _base
        with mock.patch.object(_base, "_read_lhm_gpu_temp", return_value=95.0):
            ok, reason = _base.cooldown_gate(
                max_wait_s=0.2, temp_threshold_c=82.0, poll_interval_s=0.05,
            )
        self.assertFalse(ok)
        self.assertTrue(reason.startswith("hot_after_wait:"))


class BridgeBackendEnvTests(unittest.TestCase):
    """Verify bridge._spawn_sidecar plumbs OTR_VISUAL_BACKEND correctly."""

    def _fake_python(self, tmp: Path) -> Path:
        """Create a placeholder python.exe file so the candidates loop picks it up."""
        p = tmp / "python.exe"
        p.write_bytes(b"")  # empty file is enough; we mock Popen before exec
        return p

    def test_auto_does_not_set_env_var(self):
        from otr_v2.visual import bridge
        import tempfile
        tmp = Path(tempfile.mkdtemp(prefix="otr_bridge_env_"))
        try:
            # Pre-set the env var so we can verify the auto-path clears it.
            with mock.patch.dict(os.environ, {"OTR_VISUAL_BACKEND": "leaked"}):
                with mock.patch("subprocess.Popen") as mock_popen:
                    mock_popen.return_value = mock.Mock(pid=4242)
                    with mock.patch.object(
                        bridge.Path, "home", return_value=tmp,
                    ):
                        job_dir = tmp / "io" / "visual_in" / "hw_testjob"
                        job_dir.mkdir(parents=True)
                        # Stub out cooldown so the test is fast
                        with mock.patch.object(
                            bridge.VisualBridge, "_cooldown_gate",
                            lambda self, job_id: None,
                        ):
                            b = bridge.VisualBridge()
                            status = b._spawn_sidecar(
                                "hw_testjob", job_dir, backend="auto",
                            )
            self.assertEqual(status, "SPAWNED")
            self.assertTrue(mock_popen.called)
            kwargs = mock_popen.call_args.kwargs
            env = kwargs["env"]
            self.assertNotIn("OTR_VISUAL_BACKEND", env)
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

    def test_named_backend_sets_env_var(self):
        from otr_v2.visual import bridge
        import tempfile
        tmp = Path(tempfile.mkdtemp(prefix="otr_bridge_env_"))
        try:
            with mock.patch("subprocess.Popen") as mock_popen:
                mock_popen.return_value = mock.Mock(pid=4243)
                with mock.patch.object(
                    bridge.Path, "home", return_value=tmp,
                ):
                    job_dir = tmp / "io" / "visual_in" / "hw_testjob2"
                    job_dir.mkdir(parents=True)
                    with mock.patch.object(
                        bridge.VisualBridge, "_cooldown_gate",
                        lambda self, job_id: None,
                    ):
                        b = bridge.VisualBridge()
                        status = b._spawn_sidecar(
                            "hw_testjob2", job_dir, backend="placeholder_test",
                        )
            self.assertEqual(status, "SPAWNED")
            kwargs = mock_popen.call_args.kwargs
            env = kwargs["env"]
            self.assertEqual(env.get("OTR_VISUAL_BACKEND"), "placeholder_test")
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

    def test_spawn_uses_file_not_pipe_for_stdout(self):
        """Regression guard: stdout/stderr MUST be file handles, not PIPE."""
        from otr_v2.visual import bridge
        import tempfile
        tmp = Path(tempfile.mkdtemp(prefix="otr_bridge_pipes_"))
        try:
            with mock.patch("subprocess.Popen") as mock_popen:
                mock_popen.return_value = mock.Mock(pid=4244)
                with mock.patch.object(
                    bridge.Path, "home", return_value=tmp,
                ):
                    job_dir = tmp / "io" / "visual_in" / "hw_pipeguard"
                    job_dir.mkdir(parents=True)
                    with mock.patch.object(
                        bridge.VisualBridge, "_cooldown_gate",
                        lambda self, job_id: None,
                    ):
                        bridge.VisualBridge()._spawn_sidecar(
                            "hw_pipeguard", job_dir, backend="auto",
                        )
            kwargs = mock_popen.call_args.kwargs
            self.assertNotEqual(kwargs.get("stdout"), subprocess.PIPE)
            self.assertNotEqual(kwargs.get("stderr"), subprocess.PIPE)
            # And they must be writable file-like objects
            self.assertTrue(hasattr(kwargs["stdout"], "write"))
            self.assertTrue(hasattr(kwargs["stderr"], "write"))
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)


class BridgeInputTypesTests(unittest.TestCase):
    """Guardrail: backend arg must be a closed choice list with 'auto' default."""

    def test_backend_arg_in_input_types(self):
        from otr_v2.visual.bridge import VisualBridge
        it = VisualBridge.INPUT_TYPES()
        self.assertIn("backend", it["optional"])
        choices, meta = it["optional"]["backend"]
        self.assertIsInstance(choices, list)
        self.assertIn("auto", choices)
        self.assertIn("placeholder_test", choices)
        self.assertEqual(meta.get("default"), "auto")


if __name__ == "__main__":
    unittest.main()
