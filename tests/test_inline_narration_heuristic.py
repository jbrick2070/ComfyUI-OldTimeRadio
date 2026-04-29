"""Regression test for BUG-LOCAL-100: inline-narration heuristic.

The script parser's v1 + v2 inline matches used to greedily capture
everything after a [VOICE: NAME, traits] tag as the dialogue line,
even when the LLM emitted a third-person stage direction inline and
put the actual dialogue on the next line:

    [VOICE: LEV SHAW, male, 50s, urgent] Lev bursts onto deck...
    Got the newest readings yet?

The fix adds ``_is_inline_narration(speaker_name, text)`` -- a heuristic
that returns True when the captured text starts with the speaker's
first word in third-person form (a strong narration signal). When True,
the caller looks ahead to the next non-empty line for the actual
dialogue.

This test exercises the helper directly without loading transformers /
torch / comfy. Pattern matches ``test_cast_fuzzy_consolidate.py``:
extract the helper definition via regex from the source, exec into a
clean namespace, run assertions.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))


def _import_helper():
    """Extract _is_inline_narration from story_orchestrator.py source.

    The full module imports ComfyUI / transformers / torch which we
    don't want pulled in by a unit test. Pulling just the helper's
    source block sidesteps that.
    """
    import importlib.util  # noqa: F401  (kept for parity with sister test)
    import re

    src = (_REPO / "nodes" / "story_orchestrator.py").read_text(
        encoding="utf-8",
    )
    block = re.search(
        r"^def _is_inline_narration.*?(?=^def |\Z)",
        src,
        re.MULTILINE | re.DOTALL,
    )
    if not block:
        raise RuntimeError("could not extract _is_inline_narration source")
    ns: dict = {"__name__": "_inline_narration_test", "re": re}
    exec(compile(block.group(0), "story_orchestrator.py", "exec"), ns)
    return ns["_is_inline_narration"]


_is_inline_narration = _import_helper()


# ---------------------------------------------------------------------------
# Stellar Shadows actual failure cases (2026-04-28)
# ---------------------------------------------------------------------------

class TestStellarShadowsCases:
    """The exact text strings that produced the lip-sync regression."""

    def test_lev_burst_narration_detected(self):
        text = (
            "Lev bursts onto catwalk deck connecting habitat pods, "
            "hurrying toward central operations center."
        )
        assert _is_inline_narration("LEV SHAW", text) is True

    def test_stanley_follows_narration_detected(self):
        text = (
            "Stanley follows closely, hunched against the swirling "
            "red grit. His voice carries thinly over the roar."
        )
        assert _is_inline_narration("STANLEY CRANSTON", text) is True


# ---------------------------------------------------------------------------
# Real dialogue (must NOT be flagged as narration)
# ---------------------------------------------------------------------------

class TestRealDialoguePassesThrough:
    """A character does not normally say their own name in third person.
    These cases must return False so the parser keeps the original
    inline-dialogue capture.
    """

    @pytest.mark.parametrize("speaker, dialogue", [
        ("LEV SHAW",
         "We've lost too much time already, Stanley! We can't afford another delay!"),
        ("LEV SHAW", "(panting) We have lost too much time"),
        ("STANLEY CRANSTON",
         "I'm well aware of the stakes, Lev! I've been calculating these shadows for decades!"),
        ("STANLEY CRANSTON", "A hunch?"),
        ("ANNOUNCER", "Welcome to Signal Lost. Tonight's broadcast takes us into the unknown."),
        ("ANNOUNCER", "Stay safe."),
        ("EDNA WALLACE", "What was that noise?"),  # speaker first word != dialogue first word
        ("DR PARK", "Hold the line."),
    ])
    def test_real_dialogue_returns_false(self, speaker, dialogue):
        assert _is_inline_narration(speaker, dialogue) is False


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Defensive behavior on malformed or borderline inputs."""

    def test_empty_text_returns_false(self):
        assert _is_inline_narration("LEV SHAW", "") is False

    def test_empty_speaker_returns_false(self):
        assert _is_inline_narration("", "Anything") is False

    def test_none_text_returns_false(self):
        assert _is_inline_narration("LEV SHAW", None) is False

    def test_none_speaker_returns_false(self):
        assert _is_inline_narration(None, "Anything") is False

    def test_speaker_first_word_with_punctuation_still_matches(self):
        # "Lev." or "Lev," should still trigger the heuristic
        assert _is_inline_narration("LEV SHAW", "Lev. He turned to the console.") is True
        assert _is_inline_narration("LEV SHAW", "Lev, breathing heavily, ran to the airlock.") is True

    def test_leading_quotes_stripped_before_check(self):
        # Smart quotes around dialogue must not hide a third-person prefix
        smart_left = chr(0x201C)
        smart_right = chr(0x201D)
        assert _is_inline_narration(
            "LEV SHAW", smart_left + "Lev burst through the door." + smart_right,
        ) is True
        assert _is_inline_narration(
            "LEV SHAW", '"Got the readings yet?"',
        ) is False

    def test_leading_asterisks_stripped(self):
        # The script parser strips outer asterisks (lines ~7428-7430 of
        # story_orchestrator.py) BEFORE this helper is invoked, so the
        # helper itself only needs to handle leading-asterisk *prefix*
        # noise (single side of a partial markdown italic), not full
        # `**Lev**` paired markdown. The latter is killed upstream.
        assert _is_inline_narration("LEV SHAW", "*Lev bursts through the door") is True
        assert _is_inline_narration("LEV SHAW", "_Lev bursts through the door") is True

    def test_case_insensitive_match(self):
        # Speaker name and dialogue prefix in different cases
        assert _is_inline_narration("Lev Shaw", "lev bursts onto deck.") is True
        assert _is_inline_narration("LEV SHAW", "LEV stares at the screen.") is True

    def test_partial_name_match_does_not_falsely_trigger(self):
        # "Levitation" starts with "Lev" but that's a substring, not the
        # speaker's first word. The heuristic does whole-word comparison
        # via split()[0] so it should NOT match -- the dialogue text
        # gets through.
        assert _is_inline_narration("LEV SHAW", "Levitation drills tomorrow.") is False

    def test_single_name_speaker(self):
        # Speaker is a single word (e.g. ANNOUNCER, RADIO)
        assert _is_inline_narration("ANNOUNCER", "Announcer reads the bulletin.") is True
        assert _is_inline_narration("ANNOUNCER", "Tonight's broadcast.") is False

    def test_whitespace_only_text(self):
        assert _is_inline_narration("LEV SHAW", "   \t\n  ") is False

    def test_text_starts_with_action_verb_not_name(self):
        # Stage directions that don't lead with the speaker's name
        # are NOT flagged -- false negative is acceptable; parser
        # falls through to current behavior.
        assert _is_inline_narration(
            "LEV SHAW",
            "He bursts through the door, panting.",
        ) is False
