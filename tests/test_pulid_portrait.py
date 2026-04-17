"""
tests/test_pulid_portrait.py  --  Day 3 PuLID portrait backend (stub mode)
===========================================================================

Torch-free unit tests for ``otr_v2.hyworld.backends.pulid_portrait``:

* Registry wiring.
* Stub-mode run via ``OTR_PULID_STUB=1`` -- verifies PNG dimensions,
  per-shot meta.json schema, STATUS.json READY contract.
* Identity-lock invariant: shots with identical ``refs`` produce
  identical color signatures regardless of seed / prompt / shot index.
  This is the Day 3 gate stubified so we can validate the harness
  before real PuLID weights land.
* Ref extraction handles refs / reference_images / id_refs / ref keys.
* Empty / missing shotlist graceful ERROR.

Character names and ref identifiers in these tests are deliberately
generic ("actor_a", "actor_b", "ref_a_01", etc.).  Real episodes emit
character names and ref filenames dynamically from the LLM script
pipeline -- there are no fixed/named characters in OTR v2.  The
backend only cares about the per-shot ``refs`` list and ``character``
string, whatever they happen to be for that episode.

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


class PulidRegistryTests(unittest.TestCase):
    def test_pulid_portrait_registered(self):
        from otr_v2.hyworld import backends as _backends
        self.assertIn("pulid_portrait", _backends.list_backends())

    def test_resolve_pulid_portrait(self):
        from otr_v2.hyworld import backends as _backends
        backend = _backends.resolve("pulid_portrait")
        self.assertEqual(backend.name, "pulid_portrait")
        self.assertTrue(hasattr(backend, "run"))


class PulidStubModeTests(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp(prefix="otr_pulid_"))
        self.job_id = "hw_pulid_001"
        self.fake_root = self._tmp / "repo"
        self.in_dir = self.fake_root / "io" / "hyworld_in" / self.job_id
        self.out_dir = self.fake_root / "io" / "hyworld_out" / self.job_id
        self.in_dir.mkdir(parents=True)
        self._env_patch = mock.patch.dict(
            os.environ, {"OTR_PULID_STUB": "1"}, clear=False,
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
        from otr_v2.hyworld.backends.pulid_portrait import PulidPortraitBackend
        # Character names and refs are whatever the per-episode LLM
        # script produced.  The backend must not assume any specific
        # naming scheme -- only that ``refs`` is a list of strings.
        self._write_shotlist([
            {"shot_id": "shot_000", "env_prompt": "cockpit at dusk",
             "camera": "medium close-up", "character": "actor_a",
             "refs": ["ref_a_01.jpg", "ref_a_02.jpg"],
             "duration_sec": 9.0},
            {"shot_id": "shot_001", "env_prompt": "alleyway",
             "camera": "static", "character": "actor_b",
             "refs": ["ref_b_01.jpg"],
             "duration_sec": 7.0},
        ])
        PulidPortraitBackend().run(self.in_dir)

        status = json.loads(
            (self.out_dir / "STATUS.json").read_text(encoding="utf-8"),
        )
        self.assertEqual(status["status"], "READY")
        self.assertEqual(status.get("backend"), "pulid_portrait")
        self.assertEqual(status.get("mode"), "stub")

        for shot_id in ("shot_000", "shot_001"):
            png = self.out_dir / shot_id / "render.png"
            meta = self.out_dir / shot_id / "meta.json"
            self.assertTrue(png.exists())
            self.assertEqual(png.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            w, h = self._read_png_dims(png)
            self.assertEqual((w, h), (1024, 1024))
            meta_data = json.loads(meta.read_text(encoding="utf-8"))
            self.assertEqual(meta_data["backend"], "pulid_portrait")
            self.assertEqual(meta_data["mode"], "stub")
            self.assertIn("refs", meta_data)
            self.assertIn("refs_hash", meta_data)
            self.assertEqual(meta_data["width"], 1024)
            self.assertEqual(meta_data["height"], 1024)

    def test_empty_shotlist_writes_error(self):
        from otr_v2.hyworld.backends.pulid_portrait import PulidPortraitBackend
        self._write_shotlist([])
        PulidPortraitBackend().run(self.in_dir)
        status = json.loads(
            (self.out_dir / "STATUS.json").read_text(encoding="utf-8"),
        )
        self.assertEqual(status["status"], "ERROR")
        self.assertIn("zero shots", status["detail"])

    def test_missing_shotlist_writes_error(self):
        from otr_v2.hyworld.backends.pulid_portrait import PulidPortraitBackend
        PulidPortraitBackend().run(self.in_dir)
        status = json.loads(
            (self.out_dir / "STATUS.json").read_text(encoding="utf-8"),
        )
        self.assertEqual(status["status"], "ERROR")
        self.assertIn("FileNotFoundError", status["detail"])


class PulidIdentityLockTests(unittest.TestCase):
    """The Day 3 gate, stubified: same refs -> same color signature.

    OTR v2 episodes generate characters freshly per-episode from the
    LLM script pipeline -- there are no recurring named characters.
    What matters is that *within* one episode, every shot that points
    at the same reference set renders the same identity.  The stub
    enforces that via a color-keyed render; when real PuLID weights
    land, this test is superseded by a face-embedding SSIM check.
    """

    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp(prefix="otr_pulid_id_"))
        self.job_id = "hw_pulid_idlock"
        self.fake_root = self._tmp / "repo"
        self.in_dir = self.fake_root / "io" / "hyworld_in" / self.job_id
        self.out_dir = self.fake_root / "io" / "hyworld_out" / self.job_id
        self.in_dir.mkdir(parents=True)
        self._env_patch = mock.patch.dict(
            os.environ, {"OTR_PULID_STUB": "1"}, clear=False,
        )
        self._env_patch.start()

    def tearDown(self):
        self._env_patch.stop()
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    @staticmethod
    def _read_rgb(path: Path) -> tuple[int, int, int]:
        """Extract the R, G, B of the top-left pixel from a solid PNG."""
        data = path.read_bytes()
        # IHDR ends at byte 33; IDAT length at 33..36, "IDAT" at 37..40,
        # compressed data starts at 41.  We decompress and read the
        # first filter byte (0) then 3 bytes of RGB.
        import zlib
        idat_start = data.index(b"IDAT") + 4
        # Length is stored in the 4 bytes immediately before the
        # "IDAT" chunk type.
        length = int.from_bytes(data[idat_start - 8:idat_start - 4], "big")
        compressed = data[idat_start:idat_start + length]
        raw = zlib.decompress(compressed)
        # raw[0] is the filter byte for row 0; raw[1:4] is the first RGB.
        return (raw[1], raw[2], raw[3])

    def test_same_refs_yield_same_color_different_shot_ids(self):
        from otr_v2.hyworld.backends.pulid_portrait import PulidPortraitBackend
        # Same three refs used across two shots.  The per-episode
        # "actor_a" character name and ref filenames are placeholders
        # for whatever the LLM produced for this run.
        shots = [
            {"shot_id": "shot_000", "env_prompt": "cockpit",
             "character": "actor_a",
             "refs": ["ref_a_01.jpg", "ref_a_02.jpg", "ref_a_03.jpg"],
             "duration_sec": 5.0},
            {"shot_id": "shot_005", "env_prompt": "diner",
             "character": "actor_a",
             "refs": ["ref_a_01.jpg", "ref_a_02.jpg", "ref_a_03.jpg"],
             "duration_sec": 7.0},
        ]
        (self.in_dir / "shotlist.json").write_text(
            json.dumps({"shots": shots}), encoding="utf-8",
        )
        PulidPortraitBackend().run(self.in_dir)

        rgb_a = self._read_rgb(self.out_dir / "shot_000" / "render.png")
        rgb_b = self._read_rgb(self.out_dir / "shot_005" / "render.png")
        self.assertEqual(rgb_a, rgb_b,
                         "Identity lock broken: same refs produced different colors")

    def test_different_refs_yield_different_color(self):
        from otr_v2.hyworld.backends.pulid_portrait import PulidPortraitBackend
        shots = [
            {"shot_id": "shot_a", "env_prompt": "cockpit",
             "character": "actor_a",
             "refs": ["ref_a_01.jpg"], "duration_sec": 5.0},
            {"shot_id": "shot_b", "env_prompt": "cockpit",
             "character": "actor_b",
             "refs": ["ref_b_01.jpg"], "duration_sec": 5.0},
        ]
        (self.in_dir / "shotlist.json").write_text(
            json.dumps({"shots": shots}), encoding="utf-8",
        )
        PulidPortraitBackend().run(self.in_dir)

        rgb_a = self._read_rgb(self.out_dir / "shot_a" / "render.png")
        rgb_b = self._read_rgb(self.out_dir / "shot_b" / "render.png")
        self.assertNotEqual(rgb_a, rgb_b,
                            "Identity lock spurious: different refs collided")


class PulidHelperTests(unittest.TestCase):
    def test_extract_refs_from_refs_list(self):
        from otr_v2.hyworld.backends.pulid_portrait import _extract_refs
        refs = _extract_refs({"refs": ["a.jpg", "b.jpg", "a.jpg"]})
        self.assertEqual(refs, ["a.jpg", "b.jpg"])  # deduped, order preserved

    def test_extract_refs_from_legacy_ref_key(self):
        from otr_v2.hyworld.backends.pulid_portrait import _extract_refs
        refs = _extract_refs({"ref": "single.jpg"})
        self.assertEqual(refs, ["single.jpg"])

    def test_extract_refs_merges_multiple_key_styles(self):
        from otr_v2.hyworld.backends.pulid_portrait import _extract_refs
        refs = _extract_refs({
            "refs": ["a.jpg"],
            "reference_images": ["b.jpg"],
            "portrait_ref": "c.jpg",
        })
        self.assertEqual(refs, ["a.jpg", "b.jpg", "c.jpg"])

    def test_extract_refs_empty(self):
        from otr_v2.hyworld.backends.pulid_portrait import _extract_refs
        self.assertEqual(_extract_refs({}), [])

    def test_refs_hash_stable(self):
        from otr_v2.hyworld.backends.pulid_portrait import _refs_hash
        a = _refs_hash(["ref_a_01.jpg", "ref_a_02.jpg"])
        b = _refs_hash(["ref_a_01.jpg", "ref_a_02.jpg"])
        self.assertEqual(a, b)

    def test_refs_hash_order_sensitive(self):
        from otr_v2.hyworld.backends.pulid_portrait import _refs_hash
        a = _refs_hash(["x.jpg", "y.jpg"])
        b = _refs_hash(["y.jpg", "x.jpg"])
        self.assertNotEqual(a, b)

    def test_refs_hash_empty(self):
        from otr_v2.hyworld.backends.pulid_portrait import _refs_hash
        self.assertEqual(_refs_hash([]), "no_refs")

    def test_derive_seed_deterministic_and_distinct_from_flux(self):
        from otr_v2.hyworld.backends.pulid_portrait import _derive_seed as pulid_seed
        from otr_v2.hyworld.backends.flux_anchor import _derive_seed as flux_seed
        shot = {"shot_id": "shot_042"}
        a = pulid_seed(shot, 0)
        b = pulid_seed(shot, 0)
        self.assertEqual(a, b)
        # PuLID + FLUX must use different seed bases so the same shot
        # doesn't land on the same seed in both pipelines.
        self.assertNotEqual(pulid_seed(shot, 0), flux_seed(shot, 0))

    def test_build_prompt_includes_character_when_set(self):
        from otr_v2.hyworld.backends.pulid_portrait import _build_prompt
        # Character name is whatever the LLM emitted for this shot --
        # the backend just splices it into the portrait prompt.
        p = _build_prompt({
            "env_prompt": "cockpit",
            "camera": "close-up",
            "character": "actor_a",
        })
        self.assertIn("portrait of actor_a", p)
        self.assertIn("cockpit", p)
        self.assertIn("close-up", p)


if __name__ == "__main__":
    unittest.main()
