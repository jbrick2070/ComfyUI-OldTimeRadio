"""tests/test_bug_local_288_sfw_validator.py

BUG-LOCAL-288 (2026-05-27): SFW post-validator in compose_line.

Live evidence: pending_20260527_153942 ("Space Force / SpaceX
targeting") shipped first character line "Damn it, we've got
thirty-seven minutes..." -- profanity that PD4 explicitly bans.
The whole-episode SFW critic axis catches it post-assembly but
compose_line had no per-line gate. validate_sfw + a new
CODE_SFW_VIOLATION code close the loop so the line-level repair
branch fires before the line ever reaches the cascade.
"""
from __future__ import annotations

import pytest

from nodes._otr_stage1_plan import Stage1Beat
from nodes._otr_stage3_validators import (
    CODE_SFW_VIOLATION,
    DEFAULT_PROFANITY_TERMS,
    validate_line,
    validate_sfw,
)


def _beat(speaker: str = "REN BLACK", beat_id: str = "b002") -> Stage1Beat:
    return Stage1Beat(
        beat_id=beat_id,
        speaker=speaker,
        intent="A short procedural beat that names the premise.",
        length_target_words=20,
        emotional_register="measured",
    )


class TestValidateSFW:
    """Direct validator tests."""

    def test_clean_line_returns_none(self):
        finding = validate_sfw(
            _beat(),
            "We need to evacuate the building before the storm.",
            "REN BLACK",
        )
        assert finding is None

    def test_damn_it_fires_violation(self):
        # The live-run "Damn it, we've got thirty-seven minutes..."
        finding = validate_sfw(
            _beat(),
            "Damn it, we've got thirty-seven minutes before the next "
            "signal window closes.",
            "REN BLACK",
        )
        assert finding is not None
        assert finding.code == CODE_SFW_VIOLATION
        assert finding.severity == "error"
        assert "damn" in finding.got.lower()

    def test_hell_fires_violation(self):
        finding = validate_sfw(
            _beat(),
            "What the hell is going on here?",
            "REN BLACK",
        )
        assert finding is not None
        assert finding.code == CODE_SFW_VIOLATION

    def test_hello_does_NOT_fire(self):
        """Word-boundary check: 'hell' must NOT match inside 'hello'."""
        finding = validate_sfw(
            _beat(),
            "Hello, Maeve. The team is ready.",
            "REN BLACK",
        )
        assert finding is None

    def test_hellish_does_NOT_fire(self):
        """'hell' inside 'hellish' is OK."""
        finding = validate_sfw(
            _beat(),
            "It was a hellish week but we shipped.",
            "REN BLACK",
        )
        assert finding is None

    def test_shellfish_does_NOT_fire(self):
        """'hell' inside 'shellfish' is OK."""
        finding = validate_sfw(
            _beat(),
            "The shellfish reports came back clean.",
            "REN BLACK",
        )
        assert finding is None

    def test_case_insensitive(self):
        finding = validate_sfw(
            _beat(),
            "DAMN this is bad.",
            "REN BLACK",
        )
        assert finding is not None
        assert finding.code == CODE_SFW_VIOLATION

    def test_multiword_phrase_fires(self):
        # "screw you" is in DEFAULT_PROFANITY_TERMS.
        finding = validate_sfw(
            _beat(),
            "Screw you, I'm publishing it anyway.",
            "REN BLACK",
        )
        assert finding is not None
        assert finding.code == CODE_SFW_VIOLATION

    def test_punctuation_does_not_block_match(self):
        # The regex uses \b which respects punctuation as a
        # word boundary.
        finding = validate_sfw(
            _beat(),
            "...damn!",
            "REN BLACK",
        )
        assert finding is not None
        assert finding.code == CODE_SFW_VIOLATION

    def test_custom_terms_override_default(self):
        finding = validate_sfw(
            _beat(),
            "She tossed the wrench across the lab.",
            "REN BLACK",
            terms=["wrench"],
        )
        assert finding is not None
        assert finding.got == "wrench"

    def test_empty_term_list_yields_no_finding(self):
        finding = validate_sfw(
            _beat(),
            "Damn it.",
            "REN BLACK",
            terms=[],
        )
        assert finding is None

    def test_non_string_input_returns_none(self):
        finding = validate_sfw(_beat(), None, "REN BLACK")
        assert finding is None

    def test_default_term_list_includes_common_profanity(self):
        # Spot pin: the list ships with the everyday terms PD4 cares
        # about. Don't enumerate the full list (operator may extend).
        assert "damn" in DEFAULT_PROFANITY_TERMS
        assert "hell" in DEFAULT_PROFANITY_TERMS
        assert "shit" in DEFAULT_PROFANITY_TERMS
        assert "fuck" in DEFAULT_PROFANITY_TERMS


class TestValidateLineWiring:
    """validate_line dispatches to validate_sfw."""

    def _plan(self, speaker_name: str = "REN BLACK"):
        from nodes._otr_stage1_plan import (
            Stage1Arc,
            Stage1CastMember,
            Stage1Plan,
        )
        return Stage1Plan(
            premise="A two-hander about a publication decision.",
            arc=Stage1Arc(
                setup="The recording matches a living pod.",
                complication="Publication will draw boats.",
                resolution="Suri opts for sound-only release.",
            ),
            cast=[
                Stage1CastMember(
                    name=speaker_name,
                    gender="male",
                    pronouns="he/him",
                    voice_id="v2/en_speaker_6",
                    persona="Station operator. Clipped, wary.",
                    arc_role="reluctant insider",
                ),
            ],
            beats=[
                Stage1Beat(
                    beat_id="b001",
                    speaker=speaker_name,
                    intent="Open the scene at the station console.",
                    length_target_words=18,
                    emotional_register="grounded",
                ),
                Stage1Beat(
                    beat_id="b002",
                    speaker=speaker_name,
                    intent="Refuse the publication offer concretely.",
                    length_target_words=20,
                    emotional_register="guarded",
                ),
                Stage1Beat(
                    beat_id="b003",
                    speaker=speaker_name,
                    intent="Land the refusal and walk out.",
                    length_target_words=15,
                    emotional_register="resigned",
                ),
            ],
            running_facts=[],
        )

    def test_sfw_violation_lands_in_findings(self):
        plan = self._plan()
        result = validate_line(
            plan,
            plan.beats[1],  # the b002 refusal beat
            "Damn it, I'm not signing that paper.",
        )
        codes = [f.code for f in result.findings]
        assert CODE_SFW_VIOLATION in codes

    def test_clean_line_has_no_sfw_finding(self):
        plan = self._plan()
        result = validate_line(
            plan,
            plan.beats[0],
            "I'm not signing that paper. The pod stays off the maps.",
        )
        codes = [f.code for f in result.findings]
        assert CODE_SFW_VIOLATION not in codes
