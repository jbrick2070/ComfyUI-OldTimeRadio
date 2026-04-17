"""
tests/test_ltx_motion.py  --  Day 5 LTX-2.3 motion backend (stub mode)
======================================================================

Torch-free unit tests for ``otr_v2.hyworld.backends.ltx_motion``:

* Registry wiring + cross-day backend roster.
* Stub-mode run via ``OTR_LTX_STUB=1`` -- verifies motion.mp4 is emitted
  with a valid MP4 ftyp signature, per-shot meta.json schema,
  STATUS.json READY contract.
* Handoff priority (Day 5 gate): keyframe.png > render.png > error.
  The backend must prefer the Day 4 keyframe when present, fall back
  to the Day 2 anchor otherwise, and record ``input_still_source`` in
  meta.json so downstream audit knows which upstream stage fed each
  clip.
* Handoff invariant: same input still path -> same stub motion bytes
  (so the deterministic FLUX -> LTX chain is unit-testable).
* Empty / missing shotlist graceful ERROR.
* Helper determinism: _still_hash, _derive_seed stable and distinct
  from flux_anchor / pulid_portrait / flux_keyframe.
* C4 gate: stub records duration_s <= 10.0 and 24 fps.

No CUDA, no diffusers, no model weights, no ffmpeg required.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


class LtxMotionRegistryTests(unittest.TestCase):
    def test_ltx_motion_registered(self):
        from otr_v2.hyworld import backends as _backends
        self.assertIn("ltx_motion", _backends.list_backends())

    def test_resolve_ltx_motion(self):
        from otr_v2.hyworld import backends as _backends
        backend = _backends.resolve("ltx_motion")
        self.assertEqual(backend.name, "ltx_motion")
        self.assertTrue(hasattr(backend, "run"))

    def test_all_day_1_through_5_backends_registered(self):
        from otr_v2.hyworld import backends as _backends
        names = set(_backends.list_backends())
        self.assertTrue({
            "placeholder_test", "flux_anchor", "pulid_portrait",
            "flux_keyframe", "ltx_motion",
        }.issubset(names))


class LtxMotionStubModeTests(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp(prefix="otr_ltx_"))
        self.job_id = "hw_ltx_001"
        self.fake_root = self._tmp / "repo"
        self.in_dir = self.fake_root / "io" / "hyworld_in" / self.job_id
        self.out_dir = self.fake_root / "io" / "hyworld_out" / self.job_id
        self.in_dir.mkdir(parents=True)
        self._env_patch = mock.patch.dict(
            os.environ, {"OTR_LTX_STUB": "1"}, clear=False,
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

    def test_stub_renders_ready_with_valid_mp4(self):
        from otr_v2.hyworld.backends.ltx_motion import (
            LtxMotionBackend, _is_mp4,
        )
        self._write_shotlist([
            {"shot_id": "shot_000", "env_prompt": "cockpit",
             "camera": "medium", "motion_prompt": "slow push-in",
             "duration_sec": 9.0},
            {"shot_id": "shot_001", "env_prompt": "alleyway",
             "camera": "static wide", "duration_sec": 7.0},
        ])
        LtxMotionBackend().run(self.in_dir)

        status = json.loads(
            (self.out_dir / "STATUS.json").read_text(encoding="utf-8"),
        )
        self.assertEqual(status["status"], "READY")
        self.assertEqual(status.get("backend"), "ltx_motion")
        self.assertEqual(status.get("mode"), "stub")

        for shot_id in ("shot_000", "shot_001"):
            mp4 = self.out_dir / shot_id / "motion.mp4"
            meta = self.out_dir / shot_id / "meta.json"
            self.assertTrue(mp4.exists(), f"motion.mp4 missing for {shot_id}")
            self.assertTrue(
                _is_mp4(mp4),
                f"motion.mp4 for {shot_id} is not a valid MP4 "
                f"(ftyp atom not found in first 12 bytes)",
            )
            meta_data = json.loads(meta.read_text(encoding="utf-8"))
            self.assertEqual(meta_data["backend"], "ltx_motion")
            self.assertEqual(meta_data["mode"], "stub")
            self.assertEqual(meta_data["fps"], 24)
            self.assertLessEqual(meta_data["duration_s"], 10.0,
                                 "C4: clip must be <=10s")
            self.assertIn("input_still_source", meta_data)
            self.assertIn("input_still_hash", meta_data)

    def test_empty_shotlist_writes_error(self):
        from otr_v2.hyworld.backends.ltx_motion import LtxMotionBackend
        self._write_shotlist([])
        LtxMotionBackend().run(self.in_dir)
        status = json.loads(
            (self.out_dir / "STATUS.json").read_text(encoding="utf-8"),
        )
        self.assertEqual(status["status"], "ERROR")
        self.assertIn("zero shots", status["detail"])

    def test_missing_shotlist_writes_error(self):
        from otr_v2.hyworld.backends.ltx_motion import LtxMotionBackend
        LtxMotionBackend().run(self.in_dir)
        status = json.loads(
            (self.out_dir / "STATUS.json").read_text(encoding="utf-8"),
        )
        self.assertEqual(status["status"], "ERROR")
        self.assertIn("FileNotFoundError", status["detail"])


class LtxMotionHandoffTests(unittest.TestCase):
    """Day 5 gate, stubified: FLUX still -> LTX handoff priority +
    byte-level determinism on input-still identity."""

    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp(prefix="otr_ltx_hand_"))
        self.job_id = "hw_ltx_hand"
        self.fake_root = self._tmp / "repo"
        self.in_dir = self.fake_root / "io" / "hyworld_in" / self.job_id
        self.out_dir = self.fake_root / "io" / "hyworld_out" / self.job_id
        self.in_dir.mkdir(parents=True)
        self._env_patch = mock.patch.dict(
            os.environ, {"OTR_LTX_STUB": "1"}, clear=False,
        )
        self._env_patch.start()

    def tearDown(self):
        self._env_patch.stop()
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _place_upstream(self, shot_id: str, kind: str) -> Path:
        """Create a placeholder upstream asset on disk.  ``kind`` is
        'keyframe' or 'anchor'."""
        shot_dir = self.out_dir / shot_id
        shot_dir.mkdir(parents=True, exist_ok=True)
        filename = "keyframe.png" if kind == "keyframe" else "render.png"
        path = shot_dir / filename
        path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
        return path

    def _run(self, shots):
        from otr_v2.hyworld.backends.ltx_motion import LtxMotionBackend
        (self.in_dir / "shotlist.json").write_text(
            json.dumps({"shots": shots}), encoding="utf-8",
        )
        LtxMotionBackend().run(self.in_dir)

    def test_keyframe_preferred_over_anchor(self):
        # Both keyframe.png and render.png present -- keyframe must win.
        self._place_upstream("shot_000", "keyframe")
        self._place_upstream("shot_000", "anchor")
        self._run([{"shot_id": "shot_000", "env_prompt": "x",
                    "duration_sec": 5.0}])
        meta = json.loads(
            (self.out_dir / "shot_000" / "meta.json").read_text(encoding="utf-8"),
        )
        self.assertEqual(meta["input_still_source"], "keyframe")
        self.assertTrue(meta["input_still"].endswith("keyframe.png"))

    def test_anchor_used_when_no_keyframe(self):
        # Only render.png present -- anchor is the fallback.
        self._place_upstream("shot_000", "anchor")
        self._run([{"shot_id": "shot_000", "env_prompt": "x",
                    "duration_sec": 5.0}])
        meta = json.loads(
            (self.out_dir / "shot_000" / "meta.json").read_text(encoding="utf-8"),
        )
        self.assertEqual(meta["input_still_source"], "anchor")
        self.assertTrue(meta["input_still"].endswith("render.png"))

    def test_missing_still_recorded_in_stub(self):
        # No upstream asset on disk -- stub still proceeds (fabricates
        # a key from the would-be keyframe path) and records absence.
        self._run([{"shot_id": "shot_000", "env_prompt": "x",
                    "duration_sec": 5.0}])
        meta = json.loads(
            (self.out_dir / "shot_000" / "meta.json").read_text(encoding="utf-8"),
        )
        self.assertEqual(meta["input_still_source"], "missing")
        self.assertFalse(meta["still_present"])
        # motion.mp4 must still exist so downstream renderer doesn't
        # choke on a missing file.
        mp4 = self.out_dir / "shot_000" / "motion.mp4"
        self.assertTrue(mp4.exists())

    def test_same_still_yields_same_stub_bytes(self):
        """Deterministic handoff: running the same shot twice with the
        same upstream still produces byte-identical motion.mp4 stubs."""
        self._place_upstream("shot_000", "keyframe")
        self._run([{"shot_id": "shot_000", "env_prompt": "cockpit",
                    "duration_sec": 5.0}])
        mp4_1 = (self.out_dir / "shot_000" / "motion.mp4").read_bytes()

        # Re-run with different prompt, same still.
        self._run([{"shot_id": "shot_000", "env_prompt": "diner",
                    "motion_prompt": "wild different", "duration_sec": 8.0}])
        mp4_2 = (self.out_dir / "shot_000" / "motion.mp4").read_bytes()

        self.assertEqual(
            mp4_1, mp4_2,
            "Handoff determinism broken: same input still produced "
            "different stub motion bytes across prompt variations",
        )

    def test_different_stills_yield_different_stub_bytes(self):
        self._place_upstream("shot_a", "keyframe")
        self._place_upstream("shot_b", "keyframe")
        self._run([
            {"shot_id": "shot_a", "env_prompt": "cockpit",
             "duration_sec": 5.0},
            {"shot_id": "shot_b", "env_prompt": "cockpit",
             "duration_sec": 5.0},
        ])
        mp4_a = (self.out_dir / "shot_a" / "motion.mp4").read_bytes()
        mp4_b = (self.out_dir / "shot_b" / "motion.mp4").read_bytes()
        self.assertNotEqual(
            mp4_a, mp4_b,
            "Handoff collision: different input stills produced the "
            "same stub motion bytes",
        )


class LtxMotionStubEnvvarTests(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp(prefix="otr_ltx_env_"))
        self.job_id = "hw_ltx_env"
        self.fake_root = self._tmp / "repo"
        self.in_dir = self.fake_root / "io" / "hyworld_in" / self.job_id
        self.out_dir = self.fake_root / "io" / "hyworld_out" / self.job_id
        self.in_dir.mkdir(parents=True)
        (self.in_dir / "shotlist.json").write_text(
            json.dumps({"shots": [
                {"shot_id": "shot_000", "env_prompt": "x",
                 "duration_sec": 5.0},
            ]}),
            encoding="utf-8",
        )

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_otr_ltx_stub_triggers_stub(self):
        with mock.patch.dict(os.environ, {"OTR_LTX_STUB": "1"}, clear=False):
            from otr_v2.hyworld.backends.ltx_motion import LtxMotionBackend
            LtxMotionBackend().run(self.in_dir)
        status = json.loads(
            (self.out_dir / "STATUS.json").read_text(encoding="utf-8"),
        )
        self.assertEqual(status["status"], "READY")
        self.assertEqual(status["mode"], "stub")

    def test_stub_reason_names_trigger(self):
        with mock.patch.dict(os.environ, {"OTR_LTX_STUB": "1"}, clear=False):
            from otr_v2.hyworld.backends.ltx_motion import LtxMotionBackend
            LtxMotionBackend().run(self.in_dir)
        meta = json.loads(
            (self.out_dir / "shot_000" / "meta.json").read_text(encoding="utf-8"),
        )
        self.assertIn("OTR_LTX_STUB", meta.get("reason", ""))


class LtxMotionHelperTests(unittest.TestCase):
    def test_still_hash_stable(self):
        from otr_v2.hyworld.backends.ltx_motion import _still_hash
        a = _still_hash("/io/hyworld_out/job/shot_000/keyframe.png")
        b = _still_hash("/io/hyworld_out/job/shot_000/keyframe.png")
        self.assertEqual(a, b)

    def test_still_hash_distinct(self):
        from otr_v2.hyworld.backends.ltx_motion import _still_hash
        a = _still_hash("/io/hyworld_out/job/shot_000/keyframe.png")
        b = _still_hash("/io/hyworld_out/job/shot_001/keyframe.png")
        self.assertNotEqual(a, b)

    def test_still_hash_empty(self):
        from otr_v2.hyworld.backends.ltx_motion import _still_hash
        self.assertEqual(_still_hash(""), "no_still")

    def test_derive_seed_deterministic(self):
        from otr_v2.hyworld.backends.ltx_motion import _derive_seed
        shot = {"shot_id": "shot_042"}
        self.assertEqual(_derive_seed(shot, 0), _derive_seed(shot, 0))

    def test_derive_seed_distinct_from_flux_anchor(self):
        from otr_v2.hyworld.backends.ltx_motion import _derive_seed as ltx_seed
        from otr_v2.hyworld.backends.flux_anchor import _derive_seed as fa_seed
        shot = {"shot_id": "shot_042"}
        self.assertNotEqual(ltx_seed(shot, 0), fa_seed(shot, 0))

    def test_derive_seed_distinct_from_pulid(self):
        from otr_v2.hyworld.backends.ltx_motion import _derive_seed as ltx_seed
        from otr_v2.hyworld.backends.pulid_portrait import _derive_seed as pulid_seed
        shot = {"shot_id": "shot_042"}
        self.assertNotEqual(ltx_seed(shot, 0), pulid_seed(shot, 0))

    def test_derive_seed_distinct_from_flux_keyframe(self):
        from otr_v2.hyworld.backends.ltx_motion import _derive_seed as ltx_seed
        from otr_v2.hyworld.backends.flux_keyframe import _derive_seed as kf_seed
        shot = {"shot_id": "shot_042"}
        self.assertNotEqual(
            ltx_seed(shot, 0), kf_seed(shot, 0),
            "ltx_motion and flux_keyframe must use different seed bases",
        )

    def test_build_prompt_uses_motion_prompt_when_set(self):
        from otr_v2.hyworld.backends.ltx_motion import _build_prompt
        p = _build_prompt({
            "motion_prompt": "slow push-in toward the console",
            "env_prompt": "cockpit",
            "camera": "medium",
        })
        self.assertIn("slow push-in toward the console", p)
        self.assertIn("medium", p)

    def test_build_prompt_falls_back_to_env_when_no_motion(self):
        from otr_v2.hyworld.backends.ltx_motion import _build_prompt
        p = _build_prompt({"env_prompt": "warehouse at night",
                           "camera": "wide"})
        self.assertIn("warehouse at night", p)
        self.assertIn("24fps", p)

    def test_is_mp4_recognises_ftyp(self):
        from otr_v2.hyworld.backends.ltx_motion import _is_mp4, _stub_mp4
        tmp = Path(tempfile.mkdtemp(prefix="otr_ltx_mp4_"))
        try:
            p = tmp / "m.mp4"
            _stub_mp4(p, "some_hash")
            self.assertTrue(_is_mp4(p))
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

    def test_is_mp4_rejects_non_mp4(self):
        from otr_v2.hyworld.backends.ltx_motion import _is_mp4
        tmp = Path(tempfile.mkdtemp(prefix="otr_ltx_notmp4_"))
        try:
            p = tmp / "x.png"
            p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
            self.assertFalse(_is_mp4(p))
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

    def test_stub_mp4_same_hash_yields_same_bytes(self):
        from otr_v2.hyworld.backends.ltx_motion import _stub_mp4
        tmp = Path(tempfile.mkdtemp(prefix="otr_ltx_det_"))
        try:
            a = tmp / "a.mp4"
            b = tmp / "b.mp4"
            _stub_mp4(a, "hash_x")
            _stub_mp4(b, "hash_x")
            self.assertEqual(a.read_bytes(), b.read_bytes())
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

    def test_resolve_input_still_prefers_keyframe(self):
        from otr_v2.hyworld.backends.ltx_motion import _resolve_input_still
        tmp = Path(tempfile.mkdtemp(prefix="otr_ltx_res_"))
        try:
            job_dir = tmp / "in"
            out_dir = tmp / "out"
            job_dir.mkdir(parents=True)
            shot_dir = out_dir / "shot_000"
            shot_dir.mkdir(parents=True)
            (shot_dir / "render.png").write_bytes(b"x")
            (shot_dir / "keyframe.png").write_bytes(b"y")
            path, source = _resolve_input_still(
                {"shot_id": "shot_000"}, job_dir, out_dir,
            )
            self.assertEqual(source, "keyframe")
            self.assertEqual(path.name, "keyframe.png")
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

    def test_resolve_input_still_falls_back_to_anchor(self):
        from otr_v2.hyworld.backends.ltx_motion import _resolve_input_still
        tmp = Path(tempfile.mkdtemp(prefix="otr_ltx_res2_"))
        try:
            job_dir = tmp / "in"
            out_dir = tmp / "out"
            job_dir.mkdir(parents=True)
            shot_dir = out_dir / "shot_000"
            shot_dir.mkdir(parents=True)
            (shot_dir / "render.png").write_bytes(b"x")
            # No keyframe.png on disk.
            path, source = _resolve_input_still(
                {"shot_id": "shot_000"}, job_dir, out_dir,
            )
            self.assertEqual(source, "anchor")
            self.assertEqual(path.name, "render.png")
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

    def test_resolve_input_still_missing(self):
        from otr_v2.hyworld.backends.ltx_motion import _resolve_input_still
        tmp = Path(tempfile.mkdtemp(prefix="otr_ltx_res3_"))
        try:
            job_dir = tmp / "in"
            out_dir = tmp / "out"
            job_dir.mkdir(parents=True)
            path, source = _resolve_input_still(
                {"shot_id": "shot_000"}, job_dir, out_dir,
            )
            self.assertIsNone(path)
            self.assertEqual(source, "missing")
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

    def test_should_stub_honours_env_flag(self):
        from otr_v2.hyworld.backends.ltx_motion import _should_stub
        with mock.patch.dict(os.environ, {"OTR_LTX_STUB": "1"}, clear=False):
            stub, reason = _should_stub()
            self.assertTrue(stub)
            self.assertIn("OTR_LTX_STUB", reason)


if __name__ == "__main__":
    unittest.main()
