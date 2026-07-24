"""BUG-LOCAL-288 and shared narrow spoken-safety regressions.

The original production failure was a profane first character line. The same
whole-word policy now covers explicit guns/knives/weapons and explicit
sexual/nudity language. It deliberately does not classify smoking, style,
visual vocabulary, or harmless words that merely contain a blocked token.
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


class TestLiveProfanityRegression:
    def test_live_damn_it_line_fails(self):
        finding = validate_sfw(
            _beat(),
            "Damn it, we've got thirty-seven minutes before the window closes.",
        )

        assert finding is not None
        assert finding.code == CODE_SFW_VIOLATION
        assert finding.severity == "error"
        assert finding.got == "damn"

    @pytest.mark.parametrize(
        "text",
        [
            "What the hell is going on?",
            "DAMN, this is bad.",
            "Screw you, I am publishing it anyway.",
            "...damn!",
        ],
    )
    def test_profanity_is_case_insensitive_and_boundary_aware(self, text):
        finding = validate_sfw(_beat(), text)

        assert finding is not None
        assert finding.code == CODE_SFW_VIOLATION


class TestWholeWordSafetyContract:
    @pytest.mark.parametrize(
        "text",
        [
            "Hello, Maeve. The team is ready.",
            "It was a hellish week, but we shipped.",
            "The shellfish reports came back clean.",
            "The transmission has begun.",
        ],
    )
    def test_blocked_tokens_inside_ordinary_words_pass(self, text):
        assert find_text_hits(text) == ()
        assert validate_sfw(_beat(), text) is None

    @pytest.mark.parametrize(
        ("text", "category", "term"),
        [
            ("A gun lay on the table.", "weapon", "gun"),
            ("The guns were removed.", "weapon", "guns"),
            ("She counted two knives.", "weapon", "knives"),
            ("No weapon should appear.", "weapon", "weapon"),
            ("No weapons should appear.", "weapon", "weapons"),
            (
                "The report described explicit sex.",
                "sexual_nudity",
                "explicit sex",
            ),
            (
                "The report described sexual intercourse.",
                "sexual_nudity",
                "sexual intercourse",
            ),
            ("The gallery displayed nudity.", "sexual_nudity", "nudity"),
            ("The figure was naked.", "sexual_nudity", "naked"),
        ],
    )
    def test_explicit_weapon_and_sexual_nudity_terms_fail(
        self, text, category, term
    ):
        assert (category, term) in find_text_hits(text)

        finding = validate_sfw(_beat(), text)
        assert finding is not None
        assert finding.code == CODE_SFW_VIOLATION
        assert finding.got == term

    def test_multiple_categories_are_reported_by_the_shared_matcher(self):
        assert find_text_hits("Damn, the gun was beside the nudity exhibit.") == (
            ("profanity", "damn"),
            ("weapon", "gun"),
            ("sexual_nudity", "nudity"),
        )


class TestDeliberateNonGates:
    def test_smoking_terms_pass(self):
        text = "He lit a cigarette, packed tobacco, and watched the pipe smoke."

        assert find_text_hits(text) == ()
        assert validate_sfw(_beat(), text) is None

    def test_visual_vocabulary_passes(self):
        text = (
            "The violet lighthouse framed a brass telescope, a red bicycle, "
            "and seven silver umbrellas."
        )

        assert validate_sfw(_beat(), text) is None

    @pytest.mark.parametrize(
        "text",
        [
            "Yes.",
            " ".join(["signal"] * 500),
        ],
    )
    def test_short_and_long_text_are_not_safety_findings(self, text):
        assert validate_sfw(_beat(), text) is None


class TestSharedVocabulary:
    def test_public_profanity_accessor_is_the_shared_tuple(self):
        assert profanity_terms() == PROFANITY_TERMS
        for term in ("damn", "hell", "shit", "fuck"):
            assert term in profanity_terms()

    def test_explicit_categories_expose_representative_terms(self):
        for term in ("gun", "guns", "knife", "knives", "weapon", "weapons"):
            assert term in EXPLICIT_WEAPON_TERMS
        for term in ("nude", "nudity", "naked", "explicit sex"):
            assert term in EXPLICIT_NUDITY_TERMS


class TestValidateLineWiring:
    def test_shared_safety_violation_reaches_the_result(self):
        result = validate_line(
            _beat(),
            "The gun is beside the console.",
        )

        assert not result.ok
        assert [row.code for row in result.errors] == [CODE_SFW_VIOLATION]

    def test_clean_line_has_no_safety_finding(self):
        result = validate_line(
            _beat(),
            "The first transmission has begun beside the tobacco tin.",
        )

        assert result.ok
        assert CODE_SFW_VIOLATION not in [row.code for row in result.findings]
