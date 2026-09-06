"""LTX-open identity regression: actual pure driver, no node-pack/GPU imports.

The private namespace bypasses the package and engine-registration initializers,
not the production function or its leaf dependencies. A temporary import guard
refuses any non-stdlib dependency before it can load. This is a focused unittest
route for environments without pytest, not a replacement for the full suite.
"""
import copy
import importlib
import os
from pathlib import Path
import sys
import types
import unittest
from unittest import mock


REPO = Path(__file__).resolve().parents[1]
PREVIOUS_LTX_ENGINES = frozenset({
    "ltx_video", "ltx25_video", "ltx_audio_in", "ltx25_foley_plus", "ltx25_mime",
})


def row(beat_id="b001", role="announcer_visual", engine="ltx_8gb", exists=True):
    return {"order": 0, "shot_id": "shot_" + beat_id, "beat_id": beat_id,
            "role": role, "engine_id": engine, "exists": exists}


def manifest(*rows):
    return {"episode_id": "health-regression", "clips": list(rows)}


class LtxOpenHealthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.namespace = "_ltx_open_health_stdlib"
        if cls.namespace in sys.modules:
            raise AssertionError("Private health-test namespace is already in use")
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
                    raise AssertionError("Non-stdlib import in pure health test: " + fullname)
                return None

        boundary = StdlibBoundary()
        sys.meta_path.insert(0, boundary)
        try:
            cls.rd = importlib.import_module(
                cls.namespace + "._otr_video_engines.render_driver")
        finally:
            sys.meta_path.remove(boundary)
        if boundary.blocked:
            raise AssertionError("Blocked import attempt(s): " + repr(boundary.blocked))

    @classmethod
    def cleanup_namespace(cls):
        for name in list(sys.modules):
            if name == cls.namespace or name.startswith(cls.namespace + "."):
                del sys.modules[name]

    def setUp(self):
        strict_env = mock.patch.dict(os.environ, {"OTR_LTX_OPEN_STRICT": "0"})
        strict_env.start()
        self.addCleanup(strict_env.stop)

    def test_ltx098_internal_identity_is_healthy_for_observed_open_roles(self):
        # The 4060 manifest used this exact ID for all four warned open rows.
        clips = manifest(row("music_opening_001", "music_visual"),
                         row("b001"), row("b006"),
                         row("music_closing_001", "music_visual"))
        with self.assertNoLogs(self.rd._LOG, level="WARNING"):
            self.assertEqual(self.rd.check_ltx_open_health(clips, strict=True), [])

    def test_allowlist_adds_only_the_exact_ltx098_internal_id(self):
        self.assertEqual(self.rd._LTX_OPEN_ENGINES,
                         PREVIOUS_LTX_ENGINES | {"ltx_8gb"})

    def test_previous_five_ltx_lanes_remain_healthy(self):
        for engine in sorted(PREVIOUS_LTX_ENGINES):
            with self.subTest(engine=engine), self.assertNoLogs(self.rd._LOG, level="WARNING"):
                clips = manifest(row(engine=engine), row("music_opening_001", "music_visual", engine))
                self.assertEqual(self.rd.check_ltx_open_health(clips, strict=True), [])

    def test_missing_clip_is_still_bad_for_every_accepted_engine(self):
        for engine in sorted(PREVIOUS_LTX_ENGINES | {"ltx_8gb"}):
            with self.subTest(engine=engine), self.assertLogs(self.rd._LOG, level="WARNING"):
                bad = self.rd.check_ltx_open_health(manifest(row(engine=engine, exists=False)))
                self.assertEqual(len(bad), 1)
                self.assertEqual(bad[0]["engine_id"], engine)
                self.assertIs(bad[0]["exists"], False)

    def test_missing_ltx098_clip_still_raises_when_strict(self):
        with self.assertLogs(self.rd._LOG, level="WARNING"), self.assertRaises(self.rd.RenderFloorError):
            self.rd.check_ltx_open_health(manifest(row(exists=False)), strict=True)

    def test_non_ltx_and_arbitrary_ltx_prefixes_still_rejected(self):
        for engine in ("still_motion", "abstract", "humo", "", "ltx_unknown", "ltx_8gb_other"):
            with self.subTest(engine=engine), self.assertLogs(self.rd._LOG, level="WARNING"):
                bad = self.rd.check_ltx_open_health(manifest(row(engine=engine)), strict=False)
                self.assertEqual(len(bad), 1)
                self.assertEqual(bad[0]["engine_id"], engine)

    def test_strict_environment_still_raises_for_missing_or_wrong_engine(self):
        with mock.patch.dict(os.environ, {"OTR_LTX_OPEN_STRICT": "1"}):
            for clip in (row(exists=False), row(engine="still_motion")):
                with self.subTest(clip=clip), self.assertLogs(self.rd._LOG, level="WARNING"):
                    with self.assertRaises(self.rd.RenderFloorError):
                        self.rd.check_ltx_open_health(manifest(clip))

    def test_explicit_nonstrict_overrides_strict_environment(self):
        with mock.patch.dict(os.environ, {"OTR_LTX_OPEN_STRICT": "1"}):
            with self.assertLogs(self.rd._LOG, level="WARNING"):
                self.assertEqual(len(self.rd.check_ltx_open_health(
                    manifest(row(engine="still_motion")), strict=False)), 1)

    def test_sanctioned_gap_stays_nonfatal_even_when_strict(self):
        clip = row(exists=False)
        clip["status"] = self.rd._receipt.STATUS_SANCTIONED_GAP
        with self.assertLogs(self.rd._LOG, level="WARNING") as logged:
            self.assertEqual(self.rd.check_ltx_open_health(manifest(clip), strict=True), [])
        self.assertTrue(any("SANCTIONED" in message for message in logged.output))
        self.assertFalse(any("LTX-OPEN HEALTH" in message for message in logged.output))

    def test_absent_or_unknown_status_does_not_sanction_a_missing_clip(self):
        for status in (None, "not_a_sanctioned_gap"):
            clip = row(exists=False)
            if status is not None:
                clip["status"] = status
            with self.subTest(status=status), self.assertLogs(self.rd._LOG, level="WARNING"):
                with self.assertRaises(self.rd.RenderFloorError):
                    self.rd.check_ltx_open_health(manifest(clip), strict=True)

    def test_non_open_character_beat_stays_out_of_scope(self):
        with self.assertNoLogs(self.rd._LOG, level="WARNING"):
            self.assertEqual(self.rd.check_ltx_open_health(
                manifest(row("b003", "character_video", "humo", False)), strict=True), [])

    def test_synthetic_music_open_suffix_still_identifies_an_open(self):
        clip = row("prefix_b000_music_open", "character_video", "still_motion")
        with self.assertLogs(self.rd._LOG, level="WARNING"):
            self.assertEqual(len(self.rd.check_ltx_open_health(manifest(clip))), 1)

    def test_health_check_does_not_mutate_the_manifest(self):
        clips = manifest(row(), row("b002", engine="still_motion", exists=False))
        before = copy.deepcopy(clips)
        with self.assertLogs(self.rd._LOG, level="WARNING"):
            self.rd.check_ltx_open_health(clips, strict=False)
        self.assertEqual(clips, before)

    def test_empty_manifests_are_unchanged(self):
        for clips in (None, {}, {"clips": []}):
            with self.subTest(clips=clips):
                self.assertEqual(self.rd.check_ltx_open_health(clips, strict=True), [])


if __name__ == "__main__":
    unittest.main()
