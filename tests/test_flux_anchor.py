"""
tests/test_flux_anchor.py  --  Day 2 FLUX anchor backend (stub-mode)
====================================================================

Torch-free unit tests for ``otr_v2.visual.backends.flux_anchor``:

* Registry wiring (resolve + list).
* Stub-mode run via ``OTR_FLUX_STUB=1`` -- verifies PNG dimensions,
  per-shot meta.json schema, STATUS.json READY contract.
* Deterministic seed derivation.
* Prompt builder shape.
* Empty / missing shotlist graceful ERROR.

No CUDA, no diffusers, no model weights required.  The real-mode path
is exercised by the Day 2 smoke test (out-of-band, manual) once
FLUX.1-dev lands on disk.
"""

from __future__ import annotations

import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


class FluxAnchorRegistryTests(unittest.TestCase):
    def test_flux_anchor_registered(self):
        from otr_v2.visual import backends as _backends
        names = _backends.list_backends()
        self.assertIn("flux_anchor", names)

    def test_resolve_flux_anchor(self):
        from otr_v2.visual import backends as _backends
        backend = _backends.resolve("flux_anchor")
        self.assertEqual(backend.name, "flux_anchor")
        self.assertTrue(hasattr(backend, "run"))


class FluxAnchorStubModeTests(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp(prefix="otr_flux_anchor_"))
        self.job_id = "vs_flux_001"
        self.fake_root = self._tmp / "repo"
        self.in_dir = self.fake_root / "io" / "visual_in" / self.job_id
        self.out_dir = self.fake_root / "io" / "visual_out" / self.job_id
        self.in_dir.mkdir(parents=True)
        # Force stub mode regardless of whether real weights exist.
        self._env_patch = mock.patch.dict(
            os.environ, {"OTR_FLUX_STUB": "1"}, clear=False,
        )
        self._env_patch.start()

    def tearDown(self):
        self._env_patch.stop()
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _write_shotlist(self, shots):
        (self.in_dir / "shotlist.json").write_text(
            json.dumps({"shots": shots}), encoding="utf-8",
        )

    @staticmethod
    def _read_png_dims(path: Path) -> tuple[int, int]:
        """Parse the IHDR chunk of a PNG to get (width, height)."""
        data = path.read_bytes()
        # PNG signature (8 bytes) + IHDR length (4) + "IHDR" (4) = 16
        width, height = struct.unpack(">II", data[16:24])
        return width, height

    def test_stub_renders_ready_with_1024_pngs(self):
        from otr_v2.visual.backends.flux_anchor import FluxAnchorBackend
        self._write_shotlist([
            {"shot_id": "shot_000", "env_prompt": "a neon-lit Tokyo alley",
             "camera": "push in", "duration_sec": 9.0},
            {"shot_id": "shot_001", "env_prompt": "a quiet diner at dawn",
             "camera": "static", "duration_sec": 7.0},
        ])
        FluxAnchorBackend().run(self.in_dir)

        status_path = self.out_dir / "STATUS.json"
        self.assertTrue(status_path.exists(), "STATUS.json not written")
        status = json.loads(status_path.read_text(encoding="utf-8"))
        self.assertEqual(status["status"], "READY")
        self.assertEqual(status.get("backend"), "flux_anchor")
        self.assertEqual(status.get("mode"), "stub")

        for shot_id in ("shot_000", "shot_001"):
            png = self.out_dir / shot_id / "render.png"
            meta = self.out_dir / shot_id / "meta.json"
            self.assertTrue(png.exists(), f"{shot_id}/render.png missing")
            self.assertEqual(png.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            w, h = self._read_png_dims(png)
            self.assertEqual((w, h), (1024, 1024),
                             f"{shot_id} not 1024x1024 (got {w}x{h})")
            self.assertTrue(meta.exists())
            meta_data = json.loads(meta.read_text(encoding="utf-8"))
            self.assertEqual(meta_data["backend"], "flux_anchor")
            self.assertEqual(meta_data["mode"], "stub")
            self.assertEqual(meta_data["shot_id"], shot_id)
            self.assertEqual(meta_data["width"], 1024)
            self.assertEqual(meta_data["height"], 1024)
            self.assertIn("prompt", meta_data)
            self.assertIn("seed", meta_data)
            self.assertIsInstance(meta_data["seed"], int)

    def test_empty_shotlist_writes_error(self):
        from otr_v2.visual.backends.flux_anchor import FluxAnchorBackend
        self._write_shotlist([])
        FluxAnchorBackend().run(self.in_dir)
        status = json.loads(
            (self.out_dir / "STATUS.json").read_text(encoding="utf-8"),
        )
        self.assertEqual(status["status"], "ERROR")
        self.assertIn("zero shots", status["detail"])

    def test_missing_shotlist_writes_error(self):
        from otr_v2.visual.backends.flux_anchor import FluxAnchorBackend
        # no shotlist.json written
        FluxAnchorBackend().run(self.in_dir)
        status = json.loads(
            (self.out_dir / "STATUS.json").read_text(encoding="utf-8"),
        )
        self.assertEqual(status["status"], "ERROR")
        self.assertIn("FileNotFoundError", status["detail"])


class FluxAnchorHelperTests(unittest.TestCase):
    def test_build_prompt_joins_env_camera_style(self):
        from otr_v2.visual.backends.flux_anchor import _build_prompt
        prompt = _build_prompt({
            "env_prompt": "a smoky jazz club",
            "camera": "slow dolly",
        })
        self.assertIn("a smoky jazz club", prompt)
        self.assertIn("slow dolly", prompt)
        self.assertIn("cinematic", prompt.lower())

    def test_build_prompt_no_env_no_camera(self):
        from otr_v2.visual.backends.flux_anchor import _build_prompt
        prompt = _build_prompt({})
        # Style suffix alone is always present.
        self.assertIn("cinematic", prompt.lower())

    def test_derive_seed_is_deterministic(self):
        from otr_v2.visual.backends.flux_anchor import _derive_seed
        shot = {"shot_id": "shot_042"}
        a = _derive_seed(shot, 0)
        b = _derive_seed(shot, 0)
        self.assertEqual(a, b)
        # And non-negative 32-bit (fits torch generator).
        self.assertGreaterEqual(a, 0)
        self.assertLess(a, 2**31)

    def test_derive_seed_varies_by_shot(self):
        from otr_v2.visual.backends.flux_anchor import _derive_seed
        a = _derive_seed({"shot_id": "shot_000"}, 0)
        b = _derive_seed({"shot_id": "shot_001"}, 1)
        self.assertNotEqual(a, b)

    def test_should_stub_respects_env(self):
        from otr_v2.visual.backends import flux_anchor as _fa
        with mock.patch.dict(os.environ, {"OTR_FLUX_STUB": "1"}, clear=False):
            stub, reason = _fa._should_stub()
        self.assertTrue(stub)
        self.assertIn("OTR_FLUX_STUB", reason)


if __name__ == "__main__":
    unittest.main()
