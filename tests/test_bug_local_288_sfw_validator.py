"""BUG-LOCAL-288's SFW validator is RETIRED. This file proves it stays retired.

The original production failure was a profane first character line, and the
whole-word policy that answered it grew to cover explicit guns/knives/weapons
and explicit sexual/nudity language. ``validate_sfw`` was gutted 2026-08-05 by
operator directive -- no content guardrails on generated episodes -- because on
the adaptation lanes it rejected the source's own words.

This file previously asserted that each category DID block a line. It is
inverted rather than deleted, for the same reason the G9 file was: a deleted
test is silence, and silence is how a policy creeps back. The shared vocabulary
constants still exist and are still exported, so a future NON-blocking use (a
report, a tag, an advisory) stays possible. What must not return is the blocking
finding.

The deliberate non-gates -- smoking, visual vocabulary, short/long text, blocked
tokens inside ordinary words -- are kept as they were. They still pass, and they
never depended on the policy.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from nodes._otr_content_safety import (
    EXPLICIT_NUDITY_TERMS,
    EXPLICIT_WEAPON_TERMS,
    PROFANITY_TERMS,
    find_text_hits,
    profanity_terms,
)
from nodes._otr_stage3_validators import (
    CODE_SFW_VIOLATION,
    validate_line,
    validate_sfw,
)


def _beat(speaker: str = "REN BLACK", beat_id: str = "b002"):
    return SimpleNamespace(
        beat_id=beat_id,
        speaker=speaker,
        speaker_role="character",
    )


class TestRetiredPolicy:
    """The words that used to block a line now pass it."""

    def test_live_damn_it_line_passes(self):
        text = "Damn it, the relay is dark."
        assert validate_sfw(_beat(), text) is None
        assert validate_line(_beat(), text).ok

    @pytest.mark.parametrize("text", [
        "DAMN the transmitter.",
        "damn.",
        "Well, damn!",
    ])
    def test_profanity_no_longer_blocks(self, text):
        assert validate_sfw(_beat(), text) is None

    @pytest.mark.parametrize("term", [
        *PROFANITY_TERMS, *EXPLICIT_WEAPON_TERMS, *EXPLICIT_NUDITY_TERMS,
    ])
    def test_no_term_in_the_shared_vocabulary_blocks_a_line(self, term):
        """The WHOLE retired list must pass, not a sampled few of it."""
        assert validate_sfw(_beat(), f"and then {term} happened") is None

    def test_the_sources_own_language_survives(self):
        """The line this directive exists for."""
        macbeth = "Is this a dagger which I see before me?"
        assert validate_sfw(_beat(), macbeth) is None
        assert validate_line(_beat(), macbeth).ok

    def test_no_safety_finding_reaches_a_validate_line_result(self):
        result = validate_line(_beat(), "The gun was on the table, damn it.")
        assert result.ok
        assert not [f for f in result.findings if f.code == CODE_SFW_VIOLATION]


class TestDeliberateNonGates:
    """Unchanged by the rip -- these were always supposed to pass."""

    def test_smoking_terms_pass(self):
        assert validate_sfw(_beat(), "A smoking chimney over the roofline.") is None

    def test_visual_vocabulary_passes(self):
        assert validate_sfw(_beat(), "The silver machine hummed in the dark.") is None

    @pytest.mark.parametrize("text", ["Hi.", "word " * 400])
    def test_short_and_long_text_are_not_safety_findings(self, text):
        assert validate_sfw(_beat(), text) is None

    @pytest.mark.parametrize("text", [
        "The class had begun.",
        "She passed the parcel.",
    ])
    def test_blocked_tokens_inside_ordinary_words_pass(self, text):
        assert validate_sfw(_beat(), text) is None


class TestSharedVocabularySurvivesThePolicy:
    """The constants and the matcher outlive the gate that used them.

    ``find_text_hits`` is still correct and still importable; it simply has no
    blocking caller any more. Keeping these green means a future advisory or
    reporting use does not have to re-derive the vocabulary.
    """

    def test_public_profanity_accessor_is_the_shared_tuple(self):
        assert profanity_terms() == PROFANITY_TERMS
        assert isinstance(PROFANITY_TERMS, tuple)

    def test_explicit_categories_expose_representative_terms(self):
        assert "gun" in EXPLICIT_WEAPON_TERMS
        assert "dagger" in EXPLICIT_WEAPON_TERMS
        assert EXPLICIT_NUDITY_TERMS

    def test_the_matcher_itself_still_works(self):
        assert find_text_hits("and then damn happened")
        assert not find_text_hits("The class had begun.")


def test_non_spoken_roles_are_still_ignored():
    beat = SimpleNamespace(beat_id="b001", speaker="", speaker_role="music_open")
    assert validate_sfw(beat, "damn") is None
