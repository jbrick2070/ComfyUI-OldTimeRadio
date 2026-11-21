"""A minimal ffmpeg must be caught BEFORE the render, not at the caption burn.

`otr_caption_burn` composes an `ass={name}` filter and re-encodes with libx264.
A Homebrew ffmpeg carries both by default, so a Mac that ran `brew install
ffmpeg` is fine -- but a MINIMAL build (notably the binary bundled inside
imageio-ffmpeg, the obvious candidate for an automatic fallback) typically
ships neither. Without a probe, such a box renders the writer, the voices, the
music and every video clip, and only then dies at the caption stage, twenty-odd
minutes after the answer was knowable.

These tests use fake binaries and monkeypatched probes. Nothing here spawns
ffmpeg, renders, or touches a model.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nodes"))

from _otr_shared import ffmpeg as FF  # noqa: E402

_REAL_PROBE = FF.probe_ffmpeg_capabilities


class CapabilityProbeTests(unittest.TestCase):
    def setUp(self):
        FF._CAPABILITY_CACHE.clear()

    def tearDown(self):
        FF._CAPABILITY_CACHE.clear()
        FF.probe_ffmpeg_capabilities = _REAL_PROBE

    def test_a_dead_path_reports_no_ffmpeg_rather_than_silently_passing(self):
        """The bug this test exists for: an explicit path that does not exist
        probed to all-None, and None means 'could not ask', so the gap check
        reported no problem. A missing binary answered 'fine'."""
        caps = FF.probe_ffmpeg_capabilities("/nonexistent/ffmpeg")
        self.assertIsNone(caps["path"])
        gap = FF.caption_support_gap("/nonexistent/ffmpeg")
        self.assertIsNotNone(gap, "a dead path must not report captions OK")
        self.assertIn("no ffmpeg", gap)

    def _fake_probe(self, ass, libx264, path="/fake/ffmpeg"):
        """Patch the probe, not the cache: _usable() rejects a nonexistent
        path before the cache is ever read, which is correct production
        behaviour and exactly what test_a_dead_path... pins."""
        FF.probe_ffmpeg_capabilities = lambda p=None: {
            "path": path, "ass": ass, "libx264": libx264}

    def test_a_full_build_reports_no_gap(self):
        self._fake_probe(True, True)
        self.assertIsNone(FF.caption_support_gap("/fake/full"))

    def test_a_minimal_build_names_exactly_what_is_missing(self):
        self._fake_probe(False, True)
        gap = FF.caption_support_gap("/fake/min")
        self.assertIsNotNone(gap)
        self.assertIn("ass", gap)
        self.assertIn("imageio-ffmpeg", gap,
                      "the message must name the fix, not just the fault")
        self.assertNotIn("`brew install ffmpeg` ships", gap,
                         "Homebrew's formula has no libass (2026-11-20)")

    def test_both_missing_are_reported_together(self):
        self._fake_probe(False, False)
        gap = FF.caption_support_gap("/fake/none")
        self.assertIn("ass", gap)
        self.assertIn("libx264", gap)

    def test_an_unrunnable_probe_does_NOT_refuse(self):
        """None is 'could not ask', not 'answered no'. Refusing a render
        because a subprocess failed would be the guess this module avoids."""
        self._fake_probe(None, None)
        self.assertIsNone(FF.caption_support_gap("/fake/unknown"))

    def test_the_result_is_cached_and_callers_cannot_corrupt_it(self):
        """Probed against the REAL binary on this host, so the cache path is
        exercised rather than simulated."""
        real = FF.resolve_ffmpeg()
        if not real:
            self.skipTest("no ffmpeg on this host to probe")
        first = FF.probe_ffmpeg_capabilities(real)
        self.assertIn(real, FF._CAPABILITY_CACHE)
        first["ass"] = "mutated"
        self.assertNotEqual(
            FF.probe_ffmpeg_capabilities(real)["ass"], "mutated",
            "callers must not corrupt the cache by mutating a returned dict")

    def test_filter_name_is_matched_as_a_token_not_a_substring(self):
        """`ass` appears inside `subtitles`, `pass`, `compass`. Matching the
        blob rather than the column would report libass on a build without it."""
        source = (ROOT / "nodes" / "_otr_shared" / "ffmpeg.py").read_text(
            encoding="utf-8")
        self.assertIn("line.split()", source)


class MacOSResolutionTests(unittest.TestCase):
    def test_homebrew_paths_are_candidates(self):
        """Apple Silicon first. A GUI-launched ComfyUI does not inherit the
        PATH a login shell would give it, so brew's ffmpeg is invisible to the
        PATH probe even when plainly installed."""
        self.assertIn("/opt/homebrew/bin/ffmpeg", FF._MACOS_INSTALL_CANDIDATES)
        self.assertIn("/usr/local/bin/ffmpeg", FF._MACOS_INSTALL_CANDIDATES)
        self.assertEqual(FF._MACOS_INSTALL_CANDIDATES[0],
                         "/opt/homebrew/bin/ffmpeg")

    def test_windows_candidates_are_untouched(self):
        self.assertTrue(any("WinGet" in c for c in FF._WINDOWS_INSTALL_CANDIDATES))
        self.assertTrue(any("ffmpeg.exe" in c for c in FF._WINDOWS_INSTALL_CANDIDATES))



class CaptionResolutionTests(unittest.TestCase):
    """resolve_caption_ffmpeg: Homebrew's ffmpeg (9.0.2) has no libass and a Mac
    finds it first, while imageio-ffmpeg 0.8.1's bundled 9.0 has libass and
    libx264 (measured on an M6, 2026-11-20). A Mac episode published with no
    captions."""

    FULL = {"ass": True, "libx264": True}

    def setUp(self):
        self._saved = (FF.resolve_ffmpeg, FF._imageio_ffmpeg, FF.probe_ffmpeg_capabilities)

    def tearDown(self):
        FF.resolve_ffmpeg, FF._imageio_ffmpeg, FF.probe_ffmpeg_capabilities = self._saved

    def _host(self, primary, bundled, caps_by_path):
        FF.resolve_ffmpeg = lambda preferred=None: primary
        FF._imageio_ffmpeg = lambda: bundled
        FF.probe_ffmpeg_capabilities = lambda p=None: dict(caps_by_path[p], path=p)

    def test_a_brew_build_without_libass_hands_captions_to_the_bundled_build(self):
        self._host("/opt/homebrew/bin/ffmpeg", "/venv/imageio/ffmpeg", {
            "/opt/homebrew/bin/ffmpeg": {"ass": False, "libx264": True},
            "/venv/imageio/ffmpeg": self.FULL,
        })
        self.assertEqual(FF.resolve_caption_ffmpeg(), "/venv/imageio/ffmpeg")

    def test_the_switch_is_logged_once(self):
        FF._CAPTION_SWITCH_LOGGED.clear()
        self._host("/opt/homebrew/bin/ffmpeg", "/venv/imageio/ffmpeg", {
            "/opt/homebrew/bin/ffmpeg": {"ass": False, "libx264": True},
            "/venv/imageio/ffmpeg": self.FULL,
        })
        with self.assertLogs("OTR.ffmpeg", level="INFO") as seen:
            FF.resolve_caption_ffmpeg()
            FF.resolve_caption_ffmpeg()
        self.assertEqual(len(seen.output), 1)
        self.assertIn("has no ass", seen.output[0])
        FF._CAPTION_SWITCH_LOGGED.clear()

    def test_a_usual_build_that_burns_captions_is_kept(self):
        self._host("C:/winget/ffmpeg.exe", "/venv/imageio/ffmpeg", {
            "C:/winget/ffmpeg.exe": self.FULL,
            "/venv/imageio/ffmpeg": self.FULL,
        })
        self.assertEqual(FF.resolve_caption_ffmpeg(), "C:/winget/ffmpeg.exe")

    def test_an_unrunnable_probe_keeps_the_usual_build(self):
        self._host("/usr/bin/ffmpeg", "/venv/imageio/ffmpeg", {
            "/usr/bin/ffmpeg": {"ass": None, "libx264": None},
            "/venv/imageio/ffmpeg": self.FULL,
        })
        self.assertEqual(FF.resolve_caption_ffmpeg(), "/usr/bin/ffmpeg")

    def test_a_bundled_build_that_also_lacks_libass_is_not_used(self):
        self._host("/opt/homebrew/bin/ffmpeg", "/venv/imageio/ffmpeg", {
            "/opt/homebrew/bin/ffmpeg": {"ass": False, "libx264": True},
            "/venv/imageio/ffmpeg": {"ass": False, "libx264": True},
        })
        self.assertEqual(FF.resolve_caption_ffmpeg(), "/opt/homebrew/bin/ffmpeg")

    def test_no_ffmpeg_at_all_is_none(self):
        self._host(None, None, {})
        self.assertIsNone(FF.resolve_caption_ffmpeg())

    def test_the_caption_burn_asks_the_caption_resolver(self):
        src = (ROOT / "nodes" / "otr_caption_burn.py").read_text(encoding="utf-8")
        self.assertIn("return resolve_caption_ffmpeg(ffmpeg)", src)
        self.assertIn("fb = _ffmpeg_bin(ffmpeg)", src)


if __name__ == "__main__":
    unittest.main()
