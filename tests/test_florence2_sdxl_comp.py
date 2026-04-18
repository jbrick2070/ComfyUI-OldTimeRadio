"""
tests/test_florence2_sdxl_comp.py  --  Day 7 Florence-2 + SDXL inpaint (stub)
==============================================================================

Torch-free unit tests for ``otr_v2.visual.backends.florence2_sdxl_comp``:

* Registry wiring + cross-day backend roster (Days 1-7).
* Stub-mode run via ``OTR_FLORENCE_STUB=1`` -- verifies composite.png
  and mask.png are emitted with valid PNG magic, per-shot meta.json
  schema, STATUS.json READY contract.
* Handoff priority (inherits from Days 5-6): keyframe.png > render.png
  > error.  The backend must prefer the Day 4 keyframe when present,
  fall back to the Day 2 anchor otherwise, and record
  ``input_still_source`` in meta.json.
* Day 7 three-way composite invariant:
    - same (still, mask_prompt, insert_prompt) -> same composite bytes
    - change any of the three -> composite bytes shift
    - mask.png depends on mask_prompt ALONE (not insert_prompt)
* mask_prompt missing in real mode is a per-shot error; in stub mode
  it produces a "no_mask" mask but still emits files.
* Empty / missing shotlist graceful ERROR.
* Filename gate: output is ``composite.png`` (distinct from Day 4's
  ``keyframe.png``) so the planner can mux both per-shot.
* Helper determinism: stable + distinct from all 5 prior backends.
* PNG magic: composite is RGB (colour type 2), mask is grayscale
  (colour type 0).  Ensures downstream renderers can treat mask as
  a true 8-bit mask.

No CUDA, no transformers, no diffusers, no Florence-2 weights, no
SDXL inpaint weights, no PIL required.
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


def _read_png_header(path: Path) -> tuple[bytes, int]:
    """Return (magic, colour_type).  Colour type 2 = RGB, 0 = grayscale."""
    data = path.read_bytes()
    magic = data[:8]
    # IHDR colour type is at byte 25 (after PNG magic + IHDR size + "IHDR"
    # + 4-byte width + 4-byte height + 1-byte depth).
    colour_type = data[25] if len(data) > 25 else -1
    return (magic, colour_type)


class Florence2SdxlRegistryTests(unittest.TestCase):
    def test_florence2_sdxl_comp_registered(self):
        from otr_v2.visual import backends as _backends
        self.assertIn("florence2_sdxl_comp", _backends.list_backends())

    def test_resolve_florence2_sdxl_comp(self):
        from otr_v2.visual import backends as _backends
        backend = _backends.resolve("florence2_sdxl_comp")
        self.assertEqual(backend.name, "florence2_sdxl_comp")
        self.assertTrue(hasattr(backend, "run"))

    def test_all_day_1_through_7_backends_registered(self):
        from otr_v2.visual import backends as _backends
        names = set(_backends.list_backends())
        self.assertTrue({
            "placeholder_test", "flux_anchor", "pulid_portrait",
            "flux_keyframe", "ltx_motion", "wan21_loop",
            "florence2_sdxl_comp",
        }.issubset(names))


class Florence2SdxlStubModeTests(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp(prefix="otr_f2sd_"))
        self.job_id = "hw_f2sd_001"
        self.fake_root = self._tmp / "repo"
        self.in_dir = self.fake_root / "io" / "visual_in" / self.job_id
        self.out_dir = self.fake_root / "io" / "visual_out" / self.job_id
        self.in_dir.mkdir(parents=True)
        self._env_patch = mock.patch.dict(
            os.environ, {"OTR_FLORENCE_STUB": "1"}, clear=False,
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

    def test_stub_renders_ready_with_valid_pngs(self):
        from otr_v2.visual.backends.florence2_sdxl_comp import (
            Florence2SdxlCompBackend,
        )
        self._write_shotlist([
            {"shot_id": "shot_000", "env_prompt": "cockpit",
             "mask_prompt": "cockpit window",
             "insert_prompt": "CRT overlay showing pirate signal"},
            {"shot_id": "shot_001", "env_prompt": "alleyway",
             "mask_prompt": "billboard",
             "insert_prompt": "faded noir poster"},
        ])
        Florence2SdxlCompBackend().run(self.in_dir)

        status = json.loads(
            (self.out_dir / "STATUS.json").read_text(encoding="utf-8"),
        )
        self.assertEqual(status["status"], "READY")
        self.assertEqual(status.get("backend"), "florence2_sdxl_comp")
        self.assertEqual(status.get("mode"), "stub")

        for shot_id in ("shot_000", "shot_001"):
            comp = self.out_dir / shot_id / "composite.png"
            mask = self.out_dir / shot_id / "mask.png"
            meta = self.out_dir / shot_id / "meta.json"
            self.assertTrue(comp.exists(),
                            f"composite.png missing for {shot_id}")
            self.assertTrue(mask.exists(),
                            f"mask.png missing for {shot_id}")

            # PNG magic and colour-type sanity.
            comp_magic, comp_ct = _read_png_header(comp)
            self.assertEqual(comp_magic, b"\x89PNG\r\n\x1a\n",
                             f"composite.png for {shot_id} not a valid PNG")
            self.assertEqual(comp_ct, 2,
                             "composite.png must be colour type 2 (RGB)")

            mask_magic, mask_ct = _read_png_header(mask)
            self.assertEqual(mask_magic, b"\x89PNG\r\n\x1a\n",
                             f"mask.png for {shot_id} not a valid PNG")
            self.assertEqual(mask_ct, 0,
                             "mask.png must be colour type 0 (grayscale)")

            meta_data = json.loads(meta.read_text(encoding="utf-8"))
            self.assertEqual(meta_data["backend"], "florence2_sdxl_comp")
            self.assertEqual(meta_data["mode"], "stub")
            self.assertIn("input_still_source", meta_data)
            self.assertIn("composite_hash", meta_data)
            self.assertIn("mask_hash", meta_data)
            self.assertIn("mask_prompt", meta_data)
            self.assertIn("insert_prompt", meta_data)

    def test_stub_output_filename_is_composite_png(self):
        """Gate G7: output filename MUST be composite.png so planner
        can distinguish Day 7 composites from Day 4 keyframes."""
        from otr_v2.visual.backends.florence2_sdxl_comp import (
            Florence2SdxlCompBackend,
        )
        self._write_shotlist([
            {"shot_id": "shot_000", "env_prompt": "x",
             "mask_prompt": "window",
             "insert_prompt": "poster"},
        ])
        Florence2SdxlCompBackend().run(self.in_dir)
        comp = self.out_dir / "shot_000" / "composite.png"
        # Day 4's output name -- Day 7 must NOT overwrite it.
        kf = self.out_dir / "shot_000" / "keyframe.png"
        self.assertTrue(comp.exists(),
                        "composite.png must be the Day 7 output")
        self.assertFalse(kf.exists(),
                         "keyframe.png is reserved for Day 4 flux_keyframe; "
                         "Day 7 must not write it")

    def test_empty_shotlist_writes_error(self):
        from otr_v2.visual.backends.florence2_sdxl_comp import (
            Florence2SdxlCompBackend,
        )
        self._write_shotlist([])
        Florence2SdxlCompBackend().run(self.in_dir)
        status = json.loads(
            (self.out_dir / "STATUS.json").read_text(encoding="utf-8"),
        )
        self.assertEqual(status["status"], "ERROR")
        self.assertIn("zero shots", status["detail"])

    def test_missing_shotlist_writes_error(self):
        from otr_v2.visual.backends.florence2_sdxl_comp import (
            Florence2SdxlCompBackend,
        )
        Florence2SdxlCompBackend().run(self.in_dir)
        status = json.loads(
            (self.out_dir / "STATUS.json").read_text(encoding="utf-8"),
        )
        self.assertEqual(status["status"], "ERROR")
        self.assertIn("FileNotFoundError", status["detail"])


class Florence2SdxlThreeWayInvariantTests(unittest.TestCase):
    """Day 7 stub invariant: composite.png is a pure function of the
    tuple (input_still, mask_prompt, insert_prompt); mask.png is a
    pure function of mask_prompt alone.  Both directions (same -> same
    and different -> different) are exercised."""

    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp(prefix="otr_f2sd_inv_"))
        self.job_id = "hw_f2sd_inv"
        self.fake_root = self._tmp / "repo"
        self.in_dir = self.fake_root / "io" / "visual_in" / self.job_id
        self.out_dir = self.fake_root / "io" / "visual_out" / self.job_id
        self.in_dir.mkdir(parents=True)
        self._env_patch = mock.patch.dict(
            os.environ, {"OTR_FLORENCE_STUB": "1"}, clear=False,
        )
        self._env_patch.start()

    def tearDown(self):
        self._env_patch.stop()
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _place_keyframe(self, shot_id: str) -> Path:
        shot_dir = self.out_dir / shot_id
        shot_dir.mkdir(parents=True, exist_ok=True)
        path = shot_dir / "keyframe.png"
        path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
        return path

    def _run(self, shots):
        from otr_v2.visual.backends.florence2_sdxl_comp import (
            Florence2SdxlCompBackend,
        )
        (self.in_dir / "shotlist.json").write_text(
            json.dumps({"shots": shots}), encoding="utf-8",
        )
        Florence2SdxlCompBackend().run(self.in_dir)

    def test_same_triple_yields_same_composite(self):
        self._place_keyframe("shot_000")
        self._run([{"shot_id": "shot_000", "env_prompt": "cockpit",
                    "mask_prompt": "cockpit window",
                    "insert_prompt": "CRT overlay"}])
        comp1 = (self.out_dir / "shot_000" / "composite.png").read_bytes()
        self._run([{"shot_id": "shot_000", "env_prompt": "different env",
                    "mask_prompt": "cockpit window",
                    "insert_prompt": "CRT overlay",
                    "camera": "different camera"}])
        comp2 = (self.out_dir / "shot_000" / "composite.png").read_bytes()
        self.assertEqual(
            comp1, comp2,
            "Composite invariant broken: same (still, mask, insert) "
            "produced different bytes across env/camera changes",
        )

    def test_different_mask_prompt_shifts_composite(self):
        self._place_keyframe("shot_000")
        self._run([{"shot_id": "shot_000", "env_prompt": "x",
                    "mask_prompt": "cockpit window",
                    "insert_prompt": "CRT overlay"}])
        comp1 = (self.out_dir / "shot_000" / "composite.png").read_bytes()
        self._run([{"shot_id": "shot_000", "env_prompt": "x",
                    "mask_prompt": "dashboard",
                    "insert_prompt": "CRT overlay"}])
        comp2 = (self.out_dir / "shot_000" / "composite.png").read_bytes()
        self.assertNotEqual(
            comp1, comp2,
            "mask_prompt change must shift composite bytes",
        )

    def test_different_insert_prompt_shifts_composite(self):
        self._place_keyframe("shot_000")
        self._run([{"shot_id": "shot_000", "env_prompt": "x",
                    "mask_prompt": "cockpit window",
                    "insert_prompt": "CRT overlay"}])
        comp1 = (self.out_dir / "shot_000" / "composite.png").read_bytes()
        self._run([{"shot_id": "shot_000", "env_prompt": "x",
                    "mask_prompt": "cockpit window",
                    "insert_prompt": "starfield"}])
        comp2 = (self.out_dir / "shot_000" / "composite.png").read_bytes()
        self.assertNotEqual(
            comp1, comp2,
            "insert_prompt change must shift composite bytes",
        )

    def test_mask_png_depends_on_mask_prompt_alone(self):
        """Two runs with identical mask_prompt but different
        insert_prompt must produce identical mask.png bytes -- the mask
        is a function of the mask_prompt alone so downstream filters
        can cache it by prompt without re-running segmentation."""
        self._place_keyframe("shot_000")
        self._run([{"shot_id": "shot_000",
                    "mask_prompt": "cockpit window",
                    "insert_prompt": "A"}])
        mask1 = (self.out_dir / "shot_000" / "mask.png").read_bytes()
        self._run([{"shot_id": "shot_000",
                    "mask_prompt": "cockpit window",
                    "insert_prompt": "B"}])
        mask2 = (self.out_dir / "shot_000" / "mask.png").read_bytes()
        self.assertEqual(
            mask1, mask2,
            "mask.png must be a function of mask_prompt alone",
        )

    def test_mask_png_shifts_with_mask_prompt(self):
        self._place_keyframe("shot_000")
        self._run([{"shot_id": "shot_000",
                    "mask_prompt": "cockpit window",
                    "insert_prompt": "x"}])
        mask1 = (self.out_dir / "shot_000" / "mask.png").read_bytes()
        self._run([{"shot_id": "shot_000",
                    "mask_prompt": "dashboard",
                    "insert_prompt": "x"}])
        mask2 = (self.out_dir / "shot_000" / "mask.png").read_bytes()
        self.assertNotEqual(
            mask1, mask2,
            "mask_prompt change must produce a different mask.png",
        )


class Florence2SdxlHandoffTests(unittest.TestCase):
    """Inherited Day 5/6 handoff priority: keyframe > anchor > missing."""

    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp(prefix="otr_f2sd_hand_"))
        self.job_id = "hw_f2sd_hand"
        self.fake_root = self._tmp / "repo"
        self.in_dir = self.fake_root / "io" / "visual_in" / self.job_id
        self.out_dir = self.fake_root / "io" / "visual_out" / self.job_id
        self.in_dir.mkdir(parents=True)
        self._env_patch = mock.patch.dict(
            os.environ, {"OTR_FLORENCE_STUB": "1"}, clear=False,
        )
        self._env_patch.start()

    def tearDown(self):
        self._env_patch.stop()
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _place_upstream(self, shot_id: str, kind: str) -> Path:
        shot_dir = self.out_dir / shot_id
        shot_dir.mkdir(parents=True, exist_ok=True)
        filename = "keyframe.png" if kind == "keyframe" else "render.png"
        path = shot_dir / filename
        path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
        return path

    def _run(self, shots):
        from otr_v2.visual.backends.florence2_sdxl_comp import (
            Florence2SdxlCompBackend,
        )
        (self.in_dir / "shotlist.json").write_text(
            json.dumps({"shots": shots}), encoding="utf-8",
        )
        Florence2SdxlCompBackend().run(self.in_dir)

    def test_keyframe_preferred_over_anchor(self):
        self._place_upstream("shot_000", "keyframe")
        self._place_upstream("shot_000", "anchor")
        self._run([{"shot_id": "shot_000",
                    "mask_prompt": "x", "insert_prompt": "y"}])
        meta = json.loads(
            (self.out_dir / "shot_000" / "meta.json").read_text(encoding="utf-8"),
        )
        self.assertEqual(meta["input_still_source"], "keyframe")
        self.assertTrue(meta["input_still"].endswith("keyframe.png"))

    def test_anchor_used_when_no_keyframe(self):
        self._place_upstream("shot_000", "anchor")
        self._run([{"shot_id": "shot_000",
                    "mask_prompt": "x", "insert_prompt": "y"}])
        meta = json.loads(
            (self.out_dir / "shot_000" / "meta.json").read_text(encoding="utf-8"),
        )
        self.assertEqual(meta["input_still_source"], "anchor")
        self.assertTrue(meta["input_still"].endswith("render.png"))

    def test_missing_still_recorded_in_stub(self):
        self._run([{"shot_id": "shot_000",
                    "mask_prompt": "x", "insert_prompt": "y"}])
        meta = json.loads(
            (self.out_dir / "shot_000" / "meta.json").read_text(encoding="utf-8"),
        )
        self.assertEqual(meta["input_still_source"], "missing")
        self.assertFalse(meta["still_present"])
        self.assertTrue((self.out_dir / "shot_000" / "composite.png").exists())
        self.assertTrue((self.out_dir / "shot_000" / "mask.png").exists())


class Florence2SdxlStubEnvvarTests(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp(prefix="otr_f2sd_env_"))
        self.job_id = "hw_f2sd_env"
        self.fake_root = self._tmp / "repo"
        self.in_dir = self.fake_root / "io" / "visual_in" / self.job_id
        self.out_dir = self.fake_root / "io" / "visual_out" / self.job_id
        self.in_dir.mkdir(parents=True)
        (self.in_dir / "shotlist.json").write_text(
            json.dumps({"shots": [
                {"shot_id": "shot_000",
                 "mask_prompt": "x", "insert_prompt": "y"},
            ]}),
            encoding="utf-8",
        )

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_otr_florence_stub_triggers_stub(self):
        with mock.patch.dict(os.environ, {"OTR_FLORENCE_STUB": "1"},
                             clear=False):
            from otr_v2.visual.backends.florence2_sdxl_comp import (
                Florence2SdxlCompBackend,
            )
            Florence2SdxlCompBackend().run(self.in_dir)
        status = json.loads(
            (self.out_dir / "STATUS.json").read_text(encoding="utf-8"),
        )
        self.assertEqual(status["status"], "READY")
        self.assertEqual(status["mode"], "stub")

    def test_stub_reason_names_trigger(self):
        with mock.patch.dict(os.environ, {"OTR_FLORENCE_STUB": "1"},
                             clear=False):
            from otr_v2.visual.backends.florence2_sdxl_comp import (
                Florence2SdxlCompBackend,
            )
            Florence2SdxlCompBackend().run(self.in_dir)
        meta = json.loads(
            (self.out_dir / "shot_000" / "meta.json").read_text(encoding="utf-8"),
        )
        self.assertIn("OTR_FLORENCE_STUB", meta.get("reason", ""))


class Florence2SdxlHelperTests(unittest.TestCase):
    def test_composite_hash_stable(self):
        from otr_v2.visual.backends.florence2_sdxl_comp import (
            _composite_hash,
        )
        a = _composite_hash("/kf.png", "cockpit window", "CRT overlay")
        b = _composite_hash("/kf.png", "cockpit window", "CRT overlay")
        self.assertEqual(a, b)

    def test_composite_hash_shifts_with_still(self):
        from otr_v2.visual.backends.florence2_sdxl_comp import (
            _composite_hash,
        )
        a = _composite_hash("/kf_a.png", "x", "y")
        b = _composite_hash("/kf_b.png", "x", "y")
        self.assertNotEqual(a, b)

    def test_composite_hash_shifts_with_mask(self):
        from otr_v2.visual.backends.florence2_sdxl_comp import (
            _composite_hash,
        )
        a = _composite_hash("/kf.png", "window", "y")
        b = _composite_hash("/kf.png", "door", "y")
        self.assertNotEqual(a, b)

    def test_composite_hash_shifts_with_insert(self):
        from otr_v2.visual.backends.florence2_sdxl_comp import (
            _composite_hash,
        )
        a = _composite_hash("/kf.png", "x", "poster")
        b = _composite_hash("/kf.png", "x", "billboard")
        self.assertNotEqual(a, b)

    def test_mask_hash_stable(self):
        from otr_v2.visual.backends.florence2_sdxl_comp import _mask_hash
        self.assertEqual(_mask_hash("window"), _mask_hash("window"))

    def test_mask_hash_distinct(self):
        from otr_v2.visual.backends.florence2_sdxl_comp import _mask_hash
        self.assertNotEqual(_mask_hash("window"), _mask_hash("door"))

    def test_mask_hash_empty(self):
        from otr_v2.visual.backends.florence2_sdxl_comp import _mask_hash
        self.assertEqual(_mask_hash(""), "no_mask")

    def test_derive_seed_deterministic(self):
        from otr_v2.visual.backends.florence2_sdxl_comp import _derive_seed
        shot = {"shot_id": "shot_042"}
        self.assertEqual(_derive_seed(shot, 0), _derive_seed(shot, 0))

    def test_derive_seed_distinct_from_flux_anchor(self):
        from otr_v2.visual.backends.florence2_sdxl_comp import (
            _derive_seed as f2_seed,
        )
        from otr_v2.visual.backends.flux_anchor import (
            _derive_seed as fa_seed,
        )
        shot = {"shot_id": "shot_042"}
        self.assertNotEqual(f2_seed(shot, 0), fa_seed(shot, 0))

    def test_derive_seed_distinct_from_pulid(self):
        from otr_v2.visual.backends.florence2_sdxl_comp import (
            _derive_seed as f2_seed,
        )
        from otr_v2.visual.backends.pulid_portrait import (
            _derive_seed as pulid_seed,
        )
        shot = {"shot_id": "shot_042"}
        self.assertNotEqual(f2_seed(shot, 0), pulid_seed(shot, 0))

    def test_derive_seed_distinct_from_flux_keyframe(self):
        from otr_v2.visual.backends.florence2_sdxl_comp import (
            _derive_seed as f2_seed,
        )
        from otr_v2.visual.backends.flux_keyframe import (
            _derive_seed as kf_seed,
        )
        shot = {"shot_id": "shot_042"}
        self.assertNotEqual(f2_seed(shot, 0), kf_seed(shot, 0))

    def test_derive_seed_distinct_from_ltx_motion(self):
        from otr_v2.visual.backends.florence2_sdxl_comp import (
            _derive_seed as f2_seed,
        )
        from otr_v2.visual.backends.ltx_motion import (
            _derive_seed as ltx_seed,
        )
        shot = {"shot_id": "shot_042"}
        self.assertNotEqual(f2_seed(shot, 0), ltx_seed(shot, 0))

    def test_derive_seed_distinct_from_wan21_loop(self):
        from otr_v2.visual.backends.florence2_sdxl_comp import (
            _derive_seed as f2_seed,
        )
        from otr_v2.visual.backends.wan21_loop import (
            _derive_seed as wan_seed,
        )
        shot = {"shot_id": "shot_042"}
        self.assertNotEqual(
            f2_seed(shot, 0), wan_seed(shot, 0),
            "florence2_sdxl_comp and wan21_loop must use different seed bases",
        )

    def test_build_mask_prompt_passthrough(self):
        from otr_v2.visual.backends.florence2_sdxl_comp import (
            _build_mask_prompt,
        )
        self.assertEqual(
            _build_mask_prompt({"mask_prompt": "cockpit window"}),
            "cockpit window",
        )

    def test_build_mask_prompt_empty(self):
        from otr_v2.visual.backends.florence2_sdxl_comp import (
            _build_mask_prompt,
        )
        self.assertEqual(_build_mask_prompt({}), "")

    def test_build_insert_prompt_uses_insert(self):
        from otr_v2.visual.backends.florence2_sdxl_comp import (
            _build_insert_prompt,
        )
        p = _build_insert_prompt({"insert_prompt": "CRT overlay",
                                   "env_prompt": "ignored"})
        self.assertIn("CRT overlay", p)
        self.assertIn("photorealistic", p)

    def test_build_insert_prompt_falls_back_to_env(self):
        from otr_v2.visual.backends.florence2_sdxl_comp import (
            _build_insert_prompt,
        )
        p = _build_insert_prompt({"env_prompt": "warehouse at night"})
        self.assertIn("warehouse at night", p)
        self.assertIn("photorealistic", p)

    def test_build_insert_prompt_defaults_when_all_empty(self):
        from otr_v2.visual.backends.florence2_sdxl_comp import (
            _build_insert_prompt,
        )
        p = _build_insert_prompt({})
        self.assertIn("cinematic interior", p)

    def test_resolve_input_still_prefers_keyframe(self):
        from otr_v2.visual.backends.florence2_sdxl_comp import (
            _resolve_input_still,
        )
        tmp = Path(tempfile.mkdtemp(prefix="otr_f2sd_res_"))
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
        from otr_v2.visual.backends.florence2_sdxl_comp import (
            _resolve_input_still,
        )
        tmp = Path(tempfile.mkdtemp(prefix="otr_f2sd_res2_"))
        try:
            job_dir = tmp / "in"
            out_dir = tmp / "out"
            job_dir.mkdir(parents=True)
            shot_dir = out_dir / "shot_000"
            shot_dir.mkdir(parents=True)
            (shot_dir / "render.png").write_bytes(b"x")
            path, source = _resolve_input_still(
                {"shot_id": "shot_000"}, job_dir, out_dir,
            )
            self.assertEqual(source, "anchor")
            self.assertEqual(path.name, "render.png")
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

    def test_resolve_input_still_missing(self):
        from otr_v2.visual.backends.florence2_sdxl_comp import (
            _resolve_input_still,
        )
        tmp = Path(tempfile.mkdtemp(prefix="otr_f2sd_res3_"))
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
        from otr_v2.visual.backends.florence2_sdxl_comp import _should_stub
        with mock.patch.dict(os.environ, {"OTR_FLORENCE_STUB": "1"},
                             clear=False):
            stub, reason = _should_stub()
            self.assertTrue(stub)
            self.assertIn("OTR_FLORENCE_STUB", reason)

    def test_mask_value_bounds(self):
        """Stub mask grayscale value must be in (0, 255) exclusive so
        no test will ever see a degenerate all-black or all-white mask."""
        from otr_v2.visual.backends.florence2_sdxl_comp import (
            _mask_value_from_hash,
        )
        self.assertEqual(_mask_value_from_hash("00abcdef123456"), 1)
        self.assertEqual(_mask_value_from_hash("ffabcdef123456"), 254)
        self.assertGreaterEqual(_mask_value_from_hash("80abcd"), 1)
        self.assertLessEqual(_mask_value_from_hash("80abcd"), 254)


if __name__ == "__main__":
    unittest.main()
