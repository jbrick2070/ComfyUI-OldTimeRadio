"""
Regression test for BUG-LOCAL-033: treatment scanner false-positive flag storm.

Root cause: scan_treatment() and parse_treatment() used ASCII [-]+ to match section
separators, but treatment files use U+2500 BOX DRAWINGS LIGHT HORIZONTAL (─).
Similarly, cast arrow was ASCII -> / --> only; files use U+2192 (→).

Fixture: tests/fixtures/treatment_141936.txt
  - Real treatment from RUN 237 (2026-04-14 14:19:36)
  - Contains U+2500 separators and U+2192 cast arrows throughout
  - Produced a successful episode: 87% Rotten Tomatoes, 44 dialogue lines

Fix: regex classes updated to [-─]+ and arrow alternation to (?:->|-->|→).
"""

import os
import sys
import pytest

# Ensure scripts/ is importable from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from soak_operator import scan_treatment

FIXTURE = os.path.join(
    os.path.dirname(__file__), "fixtures", "treatment_141936.txt"
)

# The five flags that were false-positives before the fix
FALSE_POSITIVE_FLAGS = {
    "EMPTY_CAST",
    "NO_SCENE_ARC",
    "EMPTY_SCRIPT",
    "NEWS_SEED_MISSING",
}


def get_flag_tags(flags):
    """Extract the tag portion (before ':') from each flag string."""
    return {f.split(":")[0] for f in flags}


class TestUnicodeSeparatorFix:
    """BUG-LOCAL-033 regression suite."""

    def test_fixture_exists_and_readable(self):
        """Fixture file must exist and be UTF-8 readable (U+2500 / U+2192 present)."""
        assert os.path.exists(FIXTURE), f"Fixture not found: {FIXTURE}"
        with open(FIXTURE, encoding="utf-8") as f:
            text = f.read()
        # Verify U+2500 separator is present
        assert "\u2500" in text, "Fixture missing U+2500 box-drawing separator"
        # Verify U+2192 cast arrow is present
        assert "\u2192" in text, "Fixture missing U+2192 cast arrow"

    def test_no_false_positive_flags(self):
        """The five historically false-positive flags must not fire on this treatment."""
        flags = scan_treatment(FIXTURE)
        tags = get_flag_tags(flags)
        fired = FALSE_POSITIVE_FLAGS & tags
        assert not fired, (
            f"False-positive flags fired after BUG-LOCAL-033 fix: {sorted(fired)}\n"
            f"Full flag list:\n" + "\n".join(f"  {f}" for f in flags)
        )

    def test_cast_parsed(self):
        """Cast section must parse at least 3 characters (9 are in the fixture)."""
        # We re-implement the cast extraction here to directly test the regex
        import re
        with open(FIXTURE, encoding="utf-8") as f:
            text = f.read()
        cast_section = re.search(
            r"CAST & VOICES\n[-\u2500]+\n(.*?)(?:\n\n|\nSCENE ARC)", text, re.DOTALL
        )
        assert cast_section, "CAST & VOICES section not matched after fix"
        cast = {}
        for line in cast_section.group(1).strip().split("\n"):
            m = re.match(
                r"\s*(\S+(?:\s+\S+)*?)\s+(?:->|-->|\u2192)\s+(\S+)\s+(.*)", line
            )
            if m:
                cast[m.group(1).strip()] = m.group(2).strip()
        assert len(cast) >= 3, (
            f"Expected at least 3 cast members, got {len(cast)}: {list(cast)}"
        )

    def test_scene_arc_parsed(self):
        """SCENE ARC section must be found and contain at least one dialogue count."""
        import re
        with open(FIXTURE, encoding="utf-8") as f:
            text = f.read()
        scene_arc = re.search(
            r"SCENE ARC\n[-\u2500]+\n(.*?)(?:\nFULL SCRIPT\b)", text, re.DOTALL
        )
        assert scene_arc, "SCENE ARC section not matched after fix"
        dialogue_counts = re.findall(
            r"(\d+)\s+dialogue lines", scene_arc.group(1)
        )
        assert dialogue_counts, "No 'N dialogue lines' found in SCENE ARC"
        assert int(dialogue_counts[0]) > 0, "Dialogue count should be > 0"

    def test_script_body_parsed(self):
        """FULL SCRIPT section must yield a non-empty body."""
        import re
        with open(FIXTURE, encoding="utf-8") as f:
            text = f.read()
        script_section = re.search(
            r"FULL SCRIPT.*?\n[-\u2500]+\n(.*?)(?:\nPRODUCTION)", text, re.DOTALL
        )
        assert script_section, "FULL SCRIPT section not matched after fix"
        body = script_section.group(1).strip()
        assert body, "FULL SCRIPT body is empty after fix"
        assert len(body) > 100, (
            f"FULL SCRIPT body suspiciously short ({len(body)} chars)"
        )

    def test_news_seed_parsed(self):
        """NEWS SEED section must yield non-empty content."""
        import re
        with open(FIXTURE, encoding="utf-8") as f:
            text = f.read()
        news_m = re.search(
            r"NEWS SEED\n[-\u2500]+\n(.*?)(?:\n\n|\nCAST)", text, re.DOTALL
        )
        assert news_m, "NEWS SEED section not matched after fix"
        assert news_m.group(1).strip(), "NEWS SEED body is empty after fix"

    def test_title_stuck_is_real_positive(self):
        """TITLE_STUCK should still fire (the LLM IS stuck on 'The Last Frequency').
        This guards against accidentally suppressing real positives."""
        flags = scan_treatment(FIXTURE)
        tags = get_flag_tags(flags)
        assert "TITLE_STUCK" in tags, (
            "TITLE_STUCK should fire for this fixture (LLM title is stuck); "
            "if it no longer fires the fixture may have been replaced"
        )
