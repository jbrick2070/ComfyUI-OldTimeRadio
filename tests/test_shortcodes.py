"""Four-character codes for every dimension a generated filename carries.

Operator ruling 2026-09-07: four characters, five for the writer LLM. The name reached
249 of its 250-unit budget spelling components in full, and three more
dimensions (writer LLM, music engine, upscaler) are wanted in it.

THE TEST THAT MATTERS IS `test_every_live_dropdown_value_has_a_code`. A code
table is only useful while it is COMPLETE: the moment an engine ships without
one, its episodes silently fall back to "unk" and two different engines produce
the same filename. That test reads the LIVE dropdowns when a server is up and
the catalog otherwise, so a new engine cannot be added without being named here.

Offline and instant: no server required for anything but the completeness check,
which skips when none is running.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nodes._otr_shared import shortcodes as SC  # noqa: E402


class TableShapeTests(unittest.TestCase):
    def test_no_code_exceeds_its_dimension_cap(self):
        """Four characters, except the writer LLM which is allowed five so a row
        can carry family AND parameter count (`q354b` = Qwen 3.5 4B). The
        exception is per dimension and deliberate, not a general relaxation."""
        for name, table in SC.DIMENSIONS.items():
            cap = SC.max_code_len(name)
            for value, code in table.items():
                self.assertLessEqual(
                    len(code), cap,
                    "%s: %r -> %r is %d chars; the cap for %s is %d"
                    % (name, value, code, len(code), name, cap))
                self.assertTrue(code, "%s: %r has an empty code" % (name, value))

    def test_only_the_llm_dimension_has_a_raised_cap(self):
        """Every extra character is spent on every episode forever, so the
        exception list is asserted here rather than left to drift."""
        self.assertEqual(SC.MAX_CODE_LEN, 4)
        self.assertEqual(SC.MAX_CODE_LEN_BY_DIMENSION, {"llm": 5})

    def test_codes_are_unique_within_each_dimension(self):
        """Global uniqueness is NOT required -- the name is positional, so the
        same code may mean different things in the video and image slots. Within
        one slot a collision makes two engines indistinguishable."""
        for name, table in SC.DIMENSIONS.items():
            codes = list(table.values())
            dupes = {c for c in codes if codes.count(c) > 1}
            self.assertFalse(
                dupes, "%s has colliding codes: %s" % (name, sorted(dupes)))

    def test_codes_are_filename_and_filtergraph_safe(self):
        """These land in filenames and, downstream, in an ffmpeg filtergraph."""
        for name, table in SC.DIMENSIONS.items():
            for value, code in table.items():
                self.assertRegex(
                    code, r"^[a-z0-9]+$",
                    "%s: %r -> %r must be lowercase alphanumeric" % (name, value, code))

    def test_the_label_decorations_are_stripped(self):
        """Live labels carry a size badge, an aspect tag or a trailing note; the
        table is written against the bare value."""
        self.assertEqual(SC.code_for("llm", "Qwen/Qwen3.5-4B (4.3 GB)"), "q354b")
        self.assertEqual(SC.code_for("video_lane", "ltx098_low_video (16:9)"), "l098")
        self.assertEqual(
            SC.code_for("video_lane",
                        "viz_mxc_cpu (16:9) (audio-reactive, no scene image)"),
            "vmcp")

    def test_an_unknown_value_degrades_instead_of_raising(self):
        """A custom model is allowed by the dropdown. A finished render must not
        fail because its filename cannot be spelled."""
        self.assertEqual(SC.code_for("image_gen", "someone/custom-thing"), "unk")
        self.assertEqual(SC.code_for("nonexistent_dimension", "whatever"), "unk")


class CompletenessTests(unittest.TestCase):
    """A table with a hole is worse than no table: two engines collide on 'unk'."""

    LIVE = {
        "llm": ("OTR_LedgerScriptWriter", "creative_writing_model"),
        "source_bank": ("OTR_LedgerScriptWriter", "source_bank"),
        "visual_style": ("OTR_LedgerScriptWriter", "visual_style"),
        "video_lane": ("OTR_VideoDirector", "announcer_video_model"),
        "image_gen": ("OTR_VideoDirector", "announcer_image_model"),
        "tts": ("OTR_AnnouncerVoice", "engine"),
        "music_gen": ("OTR_StableAudioTheme", "engine"),
        "upscaler": ("OTR_SilentComposite", "upscale_engine"),
    }

    def _object_info(self):
        import json
        import urllib.error
        import urllib.request

        try:
            with urllib.request.urlopen(
                    "http://127.0.0.1:8188/object_info", timeout=90) as fh:
                return json.load(fh)
        except (urllib.error.URLError, OSError, ValueError):
            return None

    def test_every_live_dropdown_value_has_a_code(self):
        info = self._object_info()
        if not info:
            self.skipTest("no ComfyUI on 127.0.0.1:8188 to read dropdowns from")
        missing = []
        for dimension, (node, slot) in self.LIVE.items():
            spec = (info.get(node) or {}).get("input") or {}
            slots = {}
            for group in ("required", "optional"):
                slots.update(spec.get(group) or {})
            entry = slots.get(slot)
            if not entry or not isinstance(entry[0], list):
                continue
            for value in entry[0]:
                text = str(value)
                if text in SC.SENTINELS:
                    continue
                if SC.code_for(dimension, text) == "unk":
                    missing.append("%s: %r" % (dimension, text))
        self.assertFalse(
            missing,
            "these shipped dropdown values have no four-character code, so "
            "their episodes would all be named 'unk':\n  " + "\n  ".join(missing))


if __name__ == "__main__":
    unittest.main()
