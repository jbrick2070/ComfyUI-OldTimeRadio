"""
tests/test_flux_keyframe.py  --  Day 4 FLUX + ControlNet keyframe backend
===========================================================================

Torch-free unit tests for ``otr_v2.visual.backends.flux_keyframe``:

* Registry wiring.
* Stub-mode run via ``OTR_FLUX_KEYFRAME_STUB=1`` -- verifies PNG dimensions,
  per-shot meta.json schema, STATUS.json READY contract, keyframe.png +
  depth.png both emitted.
* Layout-lock invariant (Day 4 gate): same anchor path produces the same
  keyframe color across different prompts / seeds / shot indices.  This
  is the Day 4 ControlNet contract stubified so we can validate the
  harness before real weights land.
* Different anchors yield different colors (no spurious collision).
* Row 3 enforcement: ``shot["control_image"]`` is ignored -- control is
  always derived from the Day 2 anchor at ``out_dir/<shot_id>/render.png``.
* Row 8 stub-mode envvars: OTR_FLUX_KEYFRAME_STUB=1 and OTR_FLUX_STUB=1
  both trigger stub path; missing weights trigger stub path.
* Helper determinism: _control_image_hash, _color_from_hash salt split,
  _derive_seed stable and distinct from flux_anchor / pulid_portrait.
* Empty / missing shotlist graceful ERROR.

No CUDA, no diffusers, no model weights required.
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


class FluxKeyframeRegistryTests(unittest.TestCase):
    def test_flux_keyframe_registered(self):
        from otr_v2.visual import backends as _backends
        self.assertIn("flux_keyframe", _backends.list_backends())

    def test_resolve_flux_keyframe(self):
        from otr_v2.visual import backends as _backends
        backend = _backends.resolve("flux_keyframe")
        self.assertEqual(backend.name, "flux_keyframe")
        self.assertTrue(hasattr(backend, "run"))

    def test_all_day_1_through_4_backends_registered(self):
        from otr_v2.visual import backends as _backends
        names = set(_backends.list_backends())
        self.assertTrue({
            "placeholder_test", "flux_anchor",
            "pulid_portrait", "flux_keyframe",
        }.issubset(names))


class FluxKeyframeStubModeTests(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp(prefix="otr_flux_kf_"))
        self.job_id = "hw_kf_001"
        self.fake_root = self._tmp / "repo"
        self.in_dir = self.fake_root / "io" / "visual_in" / self.job_id
        self.out_dir = self.fake_root / "io" / "visual_out" / self.job_id
        self.in_dir.mkdir(parents=True)
        self._env_patch = mock.patch.dict(
            os.environ, {"OTR_FLUX_KEYFRAME_STUB": "1"}, clear=False,
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
        data = path.read_bytes()
        width, height = struct.unpack(">II", data[16:24])
        return width, height

    def test_stub_renders_ready_with_1024_pngs(self):
        from otr_v2.visual.backends.flux_keyframe import FluxKeyframeBackend
        # Characters and shot_ids whatever the LLM produced for this
        # episode -- no fixed roster.
        self._write_shotlist([
            {"shot_id": "shot_000", "env_prompt": "cockpit at dusk",
             "camera": "medium", "character": "actor_a",
             "duration_sec": 9.0},
            {"shot_id": "shot_001", "env_prompt": "alleyway",
             "camera": "static wide", "character": "actor_b",
             "duration_sec": 7.0},
        ])
        FluxKeyframeBackend().run(self.in_dir)

        status = json.loads(
            (self.out_dir / "STATUS.json").read_text(encoding="utf-8"),
        )
        self.assertEqual(status["status"], "READY")
        self.assertEqual(status.get("backend"), "flux_keyframe")
        self.assertEqual(status.get("mode"), "stub")

        for shot_id in ("shot_000", "shot_001"):
            keyframe = self.out_dir / shot_id / "keyframe.png"
            depth = self.out_dir / shot_id / "depth.png"
            meta = self.out_dir / shot_id / "meta.json"
            self.assertTrue(keyframe.exists(), f"keyframe.png missing for {shot_id}")
            self.assertTrue(depth.exists(), f"depth.png missing for {shot_id}")
            self.assertEqual(keyframe.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(depth.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            w, h = self._read_png_dims(keyframe)
            self.assertEqual((w, h), (1024, 1024))
            w, h = self._read_png_dims(depth)
            self.assertEqual((w, h), (1024, 1024))
            meta_data = json.loads(meta.read_text(encoding="utf-8"))
            self.assertEqual(meta_data["backend"], "flux_keyframe")
            self.assertEqual(meta_data["mode"], "stub")
            self.assertEqual(meta_data["control_mode"], "depth")
            self.assertIn("control_image_hash", meta_data)
            self.assertEqual(meta_data["width"], 1024)
            self.assertEqual(meta_data["height"], 1024)

    def test_stub_mode_depth_and_keyframe_have_distinct_colors(self):
        """Salt separation: depth.png and keyframe.png must differ even
        though both derive from the same control_image_hash."""
        from otr_v2.visual.backends.flux_keyframe import FluxKeyframeBackend
        self._write_shotlist([
            {"shot_id": "shot_000", "env_prompt": "cockpit",
             "camera": "medium", "character": "actor_a",
             "duration_sec": 9.0},
        ])
        FluxKeyframeBackend().run(self.in_dir)
        kf_rgb = _read_rgb(self.out_dir / "shot_000" / "keyframe.png")
        dp_rgb = _read_rgb(self.out_dir / "shot_000" / "depth.png")
        self.assertNotEqual(
            kf_rgb, dp_rgb,
            "depth.png and keyframe.png must use different salt "
            "so downstream tests can distinguish them",
        )

    def test_stub_meta_records_anchor_absence(self):
        """In stub with no pre-created anchor, meta.json must record
        ``anchor_present: false`` so run analysis can flag it."""
        from otr_v2.visual.backends.flux_keyframe import FluxKeyframeBackend
        self._write_shotlist([
            {"shot_id": "shot_000", "env_prompt": "cockpit",
             "character": "actor_a", "duration_sec": 9.0},
        ])
        FluxKeyframeBackend().run(self.in_dir)
        meta = json.loads(
            (self.out_dir / "shot_000" / "meta.json").read_text(encoding="utf-8"),
        )
        self.assertFalse(meta["anchor_present"])

    def test_stub_meta_records_anchor_present_when_day2_output_exists(self):
        """When a Day 2 anchor exists on disk, meta must record it."""
        from otr_v2.visual.backends.flux_keyframe import FluxKeyframeBackend
        shot_dir = self.out_dir / "shot_000"
        shot_dir.mkdir(parents=True)
        # Simulate a Day 2 anchor on disk.
        (shot_dir / "render.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
        self._write_shotlist([
            {"shot_id": "shot_000", "env_prompt": "cockpit",
             "character": "actor_a", "duration_sec": 9.0},
        ])
        FluxKeyframeBackend().run(self.in_dir)
        meta = json.loads(
            (shot_dir / "meta.json").read_text(encoding="utf-8"),
        )
        self.assertTrue(meta["anchor_present"])
        # control_image path must point at the anchor we pre-created.
        self.assertTrue(meta["control_image"].endswith("render.png"))

    def test_empty_shotlist_writes_error(self):
        from otr_v2.visual.backends.flux_keyframe import FluxKeyframeBackend
        self._write_shotlist([])
        FluxKeyframeBackend().run(self.in_dir)
        status = json.loads(
            (self.out_dir / "STATUS.json").read_text(encoding="utf-8"),
        )
        self.assertEqual(status["status"], "ERROR")
        self.assertIn("zero shots", status["detail"])

    def test_missing_shotlist_writes_error(self):
        from otr_v2.visual.backends.flux_keyframe import FluxKeyframeBackend
        FluxKeyframeBackend().run(self.in_dir)
        status = json.loads(
            (self.out_dir / "STATUS.json").read_text(encoding="utf-8"),
        )
        self.assertEqual(status["status"], "ERROR")
        self.assertIn("FileNotFoundError", status["detail"])


class FluxKeyframeLayoutLockTests(unittest.TestCase):
    """The Day 4 gate, stubified: same anchor -> same keyframe color.

    The real-mode contract is "3 prompt variations on the same anchor
    preserve layout".  We can't render real pixels in CI, so we enforce
    the path-keyed analogue: same shot_id (therefore same anchor path)
    produces the same keyframe color regardless of prompt / seed /
    camera / character.  When real weights land, this test is
    superseded by a depth-map correlation / layout-overlap metric.
    """

    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp(prefix="otr_flux_kf_lock_"))
        self.job_id = "hw_kf_lock"
        self.fake_root = self._tmp / "repo"
        self.in_dir = self.fake_root / "io" / "visual_in" / self.job_id
        self.out_dir = self.fake_root / "io" / "visual_out" / self.job_id
        self.in_dir.mkdir(parents=True)
        self._env_patch = mock.patch.dict(
            os.environ, {"OTR_FLUX_KEYFRAME_STUB": "1"}, clear=False,
        )
        self._env_patch.start()

    def tearDown(self):
        self._env_patch.stop()
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _run_with(self, shots):
        from otr_v2.visual.backends.flux_keyframe import FluxKeyframeBackend
        (self.in_dir / "shotlist.json").write_text(
            json.dumps({"shots": shots}), encoding="utf-8",
        )
        FluxKeyframeBackend().run(self.in_dir)

    def test_same_anchor_path_yields_same_keyframe_across_prompts(self):
        # Run 1: shot_000 with "cockpit" prompt
        self._run_with([
            {"shot_id": "shot_000", "env_prompt": "cockpit at dusk",
             "camera": "medium", "character": "actor_a",
             "duration_sec": 9.0},
        ])
        rgb_1 = _read_rgb(self.out_dir / "shot_000" / "keyframe.png")

        # Run 2: same shot_id, different prompt + camera + character
        self._run_with([
            {"shot_id": "shot_000", "env_prompt": "diner neon",
             "camera": "wide dolly", "character": "actor_b",
             "duration_sec": 5.0},
        ])
        rgb_2 = _read_rgb(self.out_dir / "shot_000" / "keyframe.png")

        # Run 3: same shot_id, yet another variation
        self._run_with([
            {"shot_id": "shot_000", "env_prompt": "warehouse at night",
             "camera": "handheld", "character": "actor_c",
             "duration_sec": 12.0},
        ])
        rgb_3 = _read_rgb(self.out_dir / "shot_000" / "keyframe.png")

        self.assertEqual(
            rgb_1, rgb_2,
            "Layout lock broken: same anchor path produced different "
            "keyframe colors under different prompts (run 1 vs 2)",
        )
        self.assertEqual(
            rgb_1, rgb_3,
            "Layout lock broken: same anchor path produced different "
            "keyframe colors under different prompts (run 1 vs 3)",
        )

    def test_different_anchors_yield_different_keyframes(self):
        self._run_with([
            {"shot_id": "shot_a", "env_prompt": "cockpit",
             "character": "actor_a", "duration_sec": 5.0},
            {"shot_id": "shot_b", "env_prompt": "cockpit",
             "character": "actor_a", "duration_sec": 5.0},
        ])
        rgb_a = _read_rgb(self.out_dir / "shot_a" / "keyframe.png")
        rgb_b = _read_rgb(self.out_dir / "shot_b" / "keyframe.png")
        self.assertNotEqual(
            rgb_a, rgb_b,
            "Layout lock spurious: different anchor paths produced "
            "the same keyframe color",
        )

    def test_shotlist_control_image_is_ignored_row3(self):
        """Row 3: the shotlist's ``control_image`` entry must not
        change the resolved control source -- the anchor path under
        out_dir is the only legal origin."""
        # Run 1: no control_image specified
        self._run_with([
            {"shot_id": "shot_000", "env_prompt": "cockpit",
             "character": "actor_a", "duration_sec": 5.0},
        ])
        rgb_without = _read_rgb(self.out_dir / "shot_000" / "keyframe.png")

        # Run 2: same shot_id, but now shotlist tries to inject a
        # storyboard sketch path.  Row 3 says: ignore it.
        self._run_with([
            {"shot_id": "shot_000", "env_prompt": "cockpit",
             "character": "actor_a", "duration_sec": 5.0,
             "control_image": "/fake/path/storyboard.png"},
        ])
        rgb_with = _read_rgb(self.out_dir / "shot_000" / "keyframe.png")

        self.assertEqual(
            rgb_without, rgb_with,
            "Row 3 violation: shot['control_image'] changed the keyframe. "
            "The anchor path must be the only legal control source.",
        )


class FluxKeyframeStubEnvvarTests(unittest.TestCase):
    """Row 8: multiple envvars must trigger stub path."""

    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp(prefix="otr_flux_kf_env_"))
        self.job_id = "hw_kf_env"
        self.fake_root = self._tmp / "repo"
        self.in_dir = self.fake_root / "io" / "visual_in" / self.job_id
        self.out_dir = self.fake_root / "io" / "visual_out" / self.job_id
        self.in_dir.mkdir(parents=True)
        (self.in_dir / "shotlist.json").write_text(
            json.dumps({"shots": [
                {"shot_id": "shot_000", "env_prompt": "x",
                 "character": "actor_a", "duration_sec": 5.0},
            ]}),
            encoding="utf-8",
        )

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_otr_flux_keyframe_stub_triggers_stub(self):
        with mock.patch.dict(os.environ, {
            "OTR_FLUX_KEYFRAME_STUB": "1",
        }, clear=False):
            from otr_v2.visual.backends.flux_keyframe import FluxKeyframeBackend
            FluxKeyframeBackend().run(self.in_dir)
        status = json.loads(
            (self.out_dir / "STATUS.json").read_text(encoding="utf-8"),
        )
        self.assertEqual(status["status"], "READY")
        self.assertEqual(status["mode"], "stub")

    def test_otr_flux_stub_shared_flag_triggers_stub(self):
        # Shared FLUX stub flag (inherited by anchor + keyframe) should
        # be honoured so one env flag stubs out the whole FLUX family.
        env = {k: v for k, v in os.environ.items()
               if k not in ("OTR_FLUX_KEYFRAME_STUB",)}
        env["OTR_FLUX_STUB"] = "1"
        # Explicitly drop the per-backend flag if present.
        env.pop("OTR_FLUX_KEYFRAME_STUB", None)
        with mock.patch.dict(os.environ, env, clear=True):
            from otr_v2.visual.backends.flux_keyframe import FluxKeyframeBackend
            FluxKeyframeBackend().run(self.in_dir)
        status = json.loads(
            (self.out_dir / "STATUS.json").read_text(encoding="utf-8"),
        )
        self.assertEqual(status["status"], "READY")
        self.assertEqual(status["mode"], "stub")

    def test_stub_reason_names_trigger(self):
        with mock.patch.dict(os.environ, {
            "OTR_FLUX_KEYFRAME_STUB": "1",
        }, clear=False):
            from otr_v2.visual.backends.flux_keyframe import FluxKeyframeBackend
            FluxKeyframeBackend().run(self.in_dir)
        meta = json.loads(
            (self.out_dir / "shot_000" / "meta.json").read_text(encoding="utf-8"),
        )
        self.assertIn("OTR_FLUX_KEYFRAME_STUB", meta.get("reason", ""))


class FluxKeyframeHelperTests(unittest.TestCase):
    def test_control_image_hash_stable(self):
        from otr_v2.visual.backends.flux_keyframe import _control_image_hash
        a = _control_image_hash("/io/visual_out/job_x/shot_000/render.png")
        b = _control_image_hash("/io/visual_out/job_x/shot_000/render.png")
        self.assertEqual(a, b)

    def test_control_image_hash_distinct_paths(self):
        from otr_v2.visual.backends.flux_keyframe import _control_image_hash
        a = _control_image_hash("/io/visual_out/job_x/shot_000/render.png")
        b = _control_image_hash("/io/visual_out/job_x/shot_001/render.png")
        self.assertNotEqual(a, b)

    def test_control_image_hash_empty(self):
        from otr_v2.visual.backends.flux_keyframe import _control_image_hash
        self.assertEqual(_control_image_hash(""), "no_control")

    def test_color_from_hash_salt_separation(self):
        from otr_v2.visual.backends.flux_keyframe import _color_from_hash
        h = "abcdef123456"
        kf = _color_from_hash(h, salt="keyframe")
        dp = _color_from_hash(h, salt="depth")
        self.assertNotEqual(
            kf, dp,
            "Same hash + different salt must yield different colors so "
            "keyframe.png and depth.png don't collide",
        )

    def test_color_from_hash_floor(self):
        from otr_v2.visual.backends.flux_keyframe import _color_from_hash
        # Force a near-zero hash to confirm the floor lifts it to >= 40.
        rgb = _color_from_hash("000000000000", salt="keyframe")
        self.assertGreaterEqual(min(rgb), 40)

    def test_derive_seed_deterministic(self):
        from otr_v2.visual.backends.flux_keyframe import _derive_seed
        shot = {"shot_id": "shot_042"}
        self.assertEqual(_derive_seed(shot, 0), _derive_seed(shot, 0))

    def test_derive_seed_distinct_from_flux_anchor(self):
        from otr_v2.visual.backends.flux_keyframe import _derive_seed as kf_seed
        from otr_v2.visual.backends.flux_anchor import _derive_seed as fa_seed
        shot = {"shot_id": "shot_042"}
        self.assertNotEqual(
            kf_seed(shot, 0), fa_seed(shot, 0),
            "flux_keyframe and flux_anchor must use different seed bases "
            "so the same shot doesn't land on the same seed in both",
        )

    def test_derive_seed_distinct_from_pulid(self):
        from otr_v2.visual.backends.flux_keyframe import _derive_seed as kf_seed
        from otr_v2.visual.backends.pulid_portrait import _derive_seed as pulid_seed
        shot = {"shot_id": "shot_042"}
        self.assertNotEqual(
            kf_seed(shot, 0), pulid_seed(shot, 0),
            "flux_keyframe and pulid_portrait must use different seed bases",
        )

    def test_build_prompt_includes_character_when_set(self):
        from otr_v2.visual.backends.flux_keyframe import _build_prompt
        p = _build_prompt({
            "env_prompt": "cockpit at dusk",
            "camera": "medium close-up",
            "character": "actor_a",
        })
        self.assertIn("featuring actor_a", p)
        self.assertIn("cockpit at dusk", p)
        self.assertIn("medium close-up", p)

    def test_build_prompt_omits_character_when_missing(self):
        from otr_v2.visual.backends.flux_keyframe import _build_prompt
        p = _build_prompt({"env_prompt": "warehouse", "camera": "wide"})
        self.assertIn("warehouse", p)
        self.assertIn("wide", p)
        self.assertNotIn("featuring", p)

    def test_resolve_control_image_only_reads_anchor_path(self):
        """Row 3: _resolve_control_image must NOT look at
        shot['control_image']."""
        from otr_v2.visual.backends.flux_keyframe import _resolve_control_image
        tmp = Path(tempfile.mkdtemp(prefix="otr_kf_resolve_"))
        try:
            job_dir = tmp / "io" / "visual_in" / "job_x"
            out_dir = tmp / "io" / "visual_out" / "job_x"
            job_dir.mkdir(parents=True)
            (out_dir / "shot_000").mkdir(parents=True)
            (out_dir / "shot_000" / "render.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)

            shot = {"shot_id": "shot_000",
                    "control_image": "/fake/storyboard.png"}
            got = _resolve_control_image(shot, job_dir, out_dir)
            self.assertIsNotNone(got)
            self.assertEqual(got.name, "render.png")
            self.assertTrue(str(got).endswith(
                str(Path("shot_000") / "render.png")
            ))
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

    def test_resolve_control_image_returns_none_when_anchor_missing(self):
        from otr_v2.visual.backends.flux_keyframe import _resolve_control_image
        tmp = Path(tempfile.mkdtemp(prefix="otr_kf_resolve2_"))
        try:
            job_dir = tmp / "io" / "visual_in" / "job_y"
            out_dir = tmp / "io" / "visual_out" / "job_y"
            job_dir.mkdir(parents=True)
            # No anchor on disk.
            shot = {"shot_id": "shot_000"}
            got = _resolve_control_image(shot, job_dir, out_dir)
            self.assertIsNone(got)
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

    def test_should_stub_honours_env_flags(self):
        from otr_v2.visual.backends.flux_keyframe import _should_stub
        with mock.patch.dict(os.environ, {
            "OTR_FLUX_KEYFRAME_STUB": "1",
        }, clear=False):
            stub, reason = _should_stub()
            self.assertTrue(stub)
            self.assertIn("OTR_FLUX_KEYFRAME_STUB", reason)


# ---- shared helpers ------------------------------------------------------

def _read_rgb(path: Path) -> tuple[int, int, int]:
    """Extract the R, G, B of the top-left pixel of a solid-color PNG."""
    import zlib
    data = path.read_bytes()
    idat_start = data.index(b"IDAT") + 4
    length = int.from_bytes(data[idat_start - 8:idat_start - 4], "big")
    compressed = data[idat_start:idat_start + length]
    raw = zlib.decompress(compressed)
    return (raw[1], raw[2], raw[3])


if __name__ == "__main__":
    unittest.main()
