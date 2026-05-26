"""Sprint 10A step 5 regression -- Stage 3 per-line validators.

Covers:
  * Each individual validator's happy path (returns None for a clean
    line) and at least one failure mode (returns a ValidationFinding
    with the expected code).
  * validate_line orchestrator: runs all six validators, returns
    aggregated ValidationResult.
  * validate_episode orchestrator: runs across multiple lines.
  * Reserved speakers (ANNOUNCER, MUSIC): pronoun + continuity
    validators are skipped (no cast row); length + speaker-leak +
    banned-phrase + on-beat still run.
  * Historical-fixture regressions: lines drawn from past failed
    soak runs trigger the expected validator findings. These are
    the operator-gate guarantees per the plan: 'every historical
    flat line, length drift, and pronoun inversion is caught.'

Tests do NOT load LLMs; validators are pure-Python.
"""

from __future__ import annotations

import pytest

from nodes._otr_stage1_plan import (
    Stage1Arc,
    Stage1Beat,
    Stage1CastMember,
    Stage1Plan,
)
from nodes._otr_stage3_validators import (
    CODE_BANNED_PHRASE,
    CODE_CONTINUITY_BREAK,
    CODE_LENGTH_DRIFT,
    CODE_OFF_BEAT,
    CODE_PRONOUN_MISMATCH,
    CODE_SPEAKER_LEAK,
    DEFAULT_BANNED_PHRASES,
    ValidationFinding,
    ValidationResult,
    validate_banned_phrases,
    validate_continuity,
    validate_episode,
    validate_length,
    validate_line,
    validate_on_beat,
    validate_pronoun_consistency,
    validate_speaker_leak,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _cast_member(
    name: str = "DALE PORTER",
    gender: str = "male",
    pronouns: str = "he/him",
    voice_id: str = "v2/en_speaker_5",
) -> Stage1CastMember:
    return Stage1CastMember(
        name=name,
        gender=gender,
        pronouns=pronouns,
        voice_id=voice_id,
        persona="Experienced relay technician. Measured cadences; trusts procedure.",
        arc_role="anchor of competence",
    )


def _beat(
    beat_id: str = "b002",
    speaker: str = "DALE PORTER",
    intent: str = "Notice the off-band carrier and call it out.",
    length_target: int = 24,
) -> Stage1Beat:
    return Stage1Beat(
        beat_id=beat_id,
        speaker=speaker,
        intent=intent,
        length_target_words=length_target,
        emotional_register="alert curious",
        callback_to=None,
    )


def _plan(
    cast: list[Stage1CastMember] | None = None,
    beats: list[Stage1Beat] | None = None,
    running_facts: list[str] | None = None,
) -> Stage1Plan:
    if cast is None:
        cast = [
            _cast_member("DALE PORTER", "male", "he/him", "v2/en_speaker_5"),
            _cast_member("MEG SAARI", "female", "she/her", "v2/en_speaker_2"),
        ]
    if beats is None:
        beats = [
            _beat("b001", "ANNOUNCER", "Open with date stamp.", 18),
            _beat("b002", "DALE PORTER", "Notice the off-band carrier.", 24),
            _beat("b003", "MEG SAARI", "Ask Dale about it.", 22),
        ]
    if running_facts is None:
        running_facts = ["The relay is on Mars.", "Dale is senior to Meg."]
    return Stage1Plan(
        premise="Two technicians at a Mars relay catch a signal that should not exist.",
        arc=Stage1Arc(
            setup="Routine night shift, calm comms traffic.",
            complication="An off-band carrier locks in and refuses to drop.",
            resolution="They identify the source and decide whether to acknowledge it.",
        ),
        cast=cast,
        beats=beats,
        running_facts=running_facts,
    )


# ---------------------------------------------------------------------------
# Length validator
# ---------------------------------------------------------------------------


class TestLengthValidator:
    def test_in_band_returns_none(self):
        # target=24 -> band [12..40]
        beat = _beat(length_target=24)
        text = " ".join(["word"] * 24)
        assert validate_length(beat, text, "DALE PORTER") is None

    def test_too_short_flagged(self):
        beat = _beat(length_target=24)
        finding = validate_length(beat, "too short", "DALE PORTER")
        assert finding is not None
        assert finding.code == CODE_LENGTH_DRIFT
        assert finding.severity == "error"
        assert finding.beat_id == "b002"

    def test_too_long_flagged(self):
        beat = _beat(length_target=24)
        text = " ".join(["word"] * 60)
        finding = validate_length(beat, text, "DALE PORTER")
        assert finding is not None
        assert finding.code == CODE_LENGTH_DRIFT

    def test_zero_words_flagged(self):
        beat = _beat(length_target=24)
        finding = validate_length(beat, "", "DALE PORTER")
        assert finding is not None
        assert finding.code == CODE_LENGTH_DRIFT
        assert finding.got == "0"


# ---------------------------------------------------------------------------
# Pronoun-consistency validator
# ---------------------------------------------------------------------------


class TestPronounValidator:
    def test_matching_pronoun_clean(self):
        cm = _cast_member("DALE", "male", "he/him")
        beat = _beat()
        # Speaker uses his own pronoun.
        assert validate_pronoun_consistency(
            beat, "I told him to wait by the console.", cm,
        ) is None

    def test_no_pronouns_clean(self):
        cm = _cast_member()
        beat = _beat()
        assert validate_pronoun_consistency(
            beat, "The signal locked in at oh-three-twenty.", cm,
        ) is None

    def test_male_speaker_using_she_flagged(self):
        cm = _cast_member("DALE", "male", "he/him")
        beat = _beat()
        finding = validate_pronoun_consistency(
            beat, "She walked across the deck and called it in.", cm,
        )
        assert finding is not None
        assert finding.code == CODE_PRONOUN_MISMATCH
        assert finding.expected == "he/him"
        assert finding.got == "she/her"

    def test_female_speaker_using_he_flagged(self):
        cm = _cast_member("MEG", "female", "she/her")
        beat = _beat()
        finding = validate_pronoun_consistency(
            beat, "He braced against the bulkhead.", cm,
        )
        assert finding is not None
        assert finding.code == CODE_PRONOUN_MISMATCH

    def test_mixed_pronouns_with_own_present_clean(self):
        """When the speaker's OWN pronoun is present, mixing in
        other pronouns is fine -- the line may reference another
        character. We only flag when own is absent + wrong is
        present."""
        cm = _cast_member("DALE", "male", "he/him")
        beat = _beat()
        # "He" (own) + "she" (other) both present.
        assert validate_pronoun_consistency(
            beat, "He told her the carrier was off-band.", cm,
        ) is None


# ---------------------------------------------------------------------------
# Speaker-leak validator
# ---------------------------------------------------------------------------


class TestSpeakerLeakValidator:
    def test_clean_line(self):
        beat = _beat()
        assert validate_speaker_leak(
            beat, "The carrier is off-band, locked into channel four.", "DALE",
        ) is None

    def test_bracketed_direction_flagged(self):
        beat = _beat()
        finding = validate_speaker_leak(
            beat, "I see it now [walks over to the console] -- there.", "DALE",
        )
        assert finding is not None
        assert finding.code == CODE_SPEAKER_LEAK

    def test_parenthesized_cue_flagged(self):
        beat = _beat()
        finding = validate_speaker_leak(
            beat, "I see it (sighs deeply) -- there.", "DALE",
        )
        assert finding is not None
        assert finding.code == CODE_SPEAKER_LEAK

    def test_asterisk_action_flagged(self):
        beat = _beat()
        finding = validate_speaker_leak(
            beat, "I see it *leans forward* -- there.", "DALE",
        )
        assert finding is not None
        assert finding.code == CODE_SPEAKER_LEAK

    def test_narrator_attribution_flagged(self):
        beat = _beat()
        finding = validate_speaker_leak(
            beat, "He said the carrier was off-band.", "DALE",
        )
        assert finding is not None
        assert finding.code == CODE_SPEAKER_LEAK

    def test_self_naming_flagged(self):
        beat = _beat()
        finding = validate_speaker_leak(
            beat, "As DALE PORTER, I would not say this aloud.", "DALE PORTER",
        )
        assert finding is not None
        assert finding.code == CODE_SPEAKER_LEAK


# ---------------------------------------------------------------------------
# Banned-phrase validator
# ---------------------------------------------------------------------------


class TestBannedPhraseValidator:
    def test_clean_line(self):
        beat = _beat()
        assert validate_banned_phrases(
            beat, "The carrier is off-band, locked.", "DALE",
        ) is None

    def test_purple_prose_flagged(self):
        beat = _beat()
        finding = validate_banned_phrases(
            beat,
            "She knew the descent was a forfeit to the void of the deep ocean.",
            "MEG",
        )
        assert finding is not None
        assert finding.code == CODE_BANNED_PHRASE
        assert finding.got == "forfeit to the void"

    def test_custom_banned_list(self):
        beat = _beat()
        custom = ["pomegranate"]
        finding = validate_banned_phrases(
            beat, "I detected a pomegranate signal in the carrier.", "DALE",
            banned=custom,
        )
        assert finding is not None
        assert finding.got == "pomegranate"

    def test_case_insensitive_match(self):
        beat = _beat()
        # Capitalized in line, lowercase in list -- still matches.
        finding = validate_banned_phrases(
            beat, "It was DESTINY CALLS yet again.", "DALE",
        )
        assert finding is not None
        assert finding.code == CODE_BANNED_PHRASE


# ---------------------------------------------------------------------------
# Continuity validator
# ---------------------------------------------------------------------------


class TestContinuityValidator:
    def test_clean_line(self):
        beat = _beat()
        assert validate_continuity(
            beat,
            "The relay is locked on the carrier signal.",
            "DALE",
            ["The relay is on Mars.", "Dale is senior to Meg."],
        ) is None

    def test_negation_of_established_fact_flagged(self):
        """Fact: 'The relay is on Mars.' Line says: 'It is not Mars.'
        Conservative negation detector should flag this."""
        beat = _beat()
        finding = validate_continuity(
            beat,
            "The signal source is not Mars -- it came from further.",
            "DALE",
            ["The relay is on Mars."],
        )
        assert finding is not None
        assert finding.code == CODE_CONTINUITY_BREAK
        assert finding.severity == "warn"

    def test_no_facts_returns_none(self):
        beat = _beat()
        assert validate_continuity(
            beat, "Any text here.", "DALE", [],
        ) is None


# ---------------------------------------------------------------------------
# On-beat validator
# ---------------------------------------------------------------------------


class TestOnBeatValidator:
    def test_on_topic_clean(self):
        beat = _beat(intent="Notice the off-band carrier and call it out.")
        finding = validate_on_beat(
            beat, "The carrier is off-band, channel four.", "DALE",
        )
        assert finding is None

    def test_off_topic_flagged(self):
        beat = _beat(intent="Notice the off-band carrier and call it out.")
        finding = validate_on_beat(
            beat,
            "Cocktails by sunset under starlit palms shimmer endlessly.",
            "DALE",
        )
        assert finding is not None
        assert finding.code == CODE_OFF_BEAT
        assert finding.severity == "warn"

    def test_empty_inputs_safe(self):
        beat = _beat()
        assert validate_on_beat(beat, "", "DALE") is None


# ---------------------------------------------------------------------------
# validate_line orchestrator
# ---------------------------------------------------------------------------


class TestValidateLine:
    def test_clean_line_no_findings(self):
        plan = _plan()
        beat = plan.beats[1]  # DALE PORTER beat
        text = "The carrier locked in at oh-three-twenty on channel four, eastern edge of the band."
        result = validate_line(plan, beat, text)
        assert result.ok
        assert len(result.errors) == 0

    def test_aggregates_multiple_errors(self):
        plan = _plan()
        beat = plan.beats[1]
        # Too short + banned phrase + asterisk action all in one
        text = "Forfeit to the void *sighs*"
        result = validate_line(plan, beat, text)
        codes = {f.code for f in result.errors}
        assert CODE_LENGTH_DRIFT in codes
        assert CODE_BANNED_PHRASE in codes
        assert CODE_SPEAKER_LEAK in codes
        assert not result.ok

    def test_announcer_speaker_skips_pronoun_check(self):
        """ANNOUNCER and MUSIC are reserved speakers; they have no
        cast row so pronoun + continuity validators must skip
        cleanly without raising."""
        plan = _plan()
        announcer_beat = plan.beats[0]   # ANNOUNCER
        text = "Welcome to Signal Lost, episode three. We open at the Mars relay."
        result = validate_line(plan, announcer_beat, text)
        # Length + speaker-leak + banned-phrase + on-beat all ran.
        # pronoun + continuity were skipped silently.
        codes = {f.code for f in result.findings}
        assert CODE_PRONOUN_MISMATCH not in codes
        assert CODE_CONTINUITY_BREAK not in codes


# ---------------------------------------------------------------------------
# validate_episode orchestrator
# ---------------------------------------------------------------------------


class TestValidateEpisode:
    def test_multiline_episode_per_beat_results(self):
        plan = _plan()
        rendered = [
            {"beat_id": "b001", "text": "Welcome to the show, episode three."},
            {"beat_id": "b002", "text": " ".join(["carrier"] * 24)},  # may be off-beat but on-length
            {"beat_id": "b003", "text": "Hi"},  # too short
        ]
        results = validate_episode(plan, rendered)
        assert set(results.keys()) == {"b001", "b002", "b003"}
        # b003 is too short -> length_drift error
        b3 = results["b003"]
        assert any(f.code == CODE_LENGTH_DRIFT for f in b3.errors)

    def test_skips_unknown_beat_ids(self):
        plan = _plan()
        rendered = [
            {"beat_id": "bZZZ", "text": "Nope."},
        ]
        results = validate_episode(plan, rendered)
        assert results == {}


# ---------------------------------------------------------------------------
# Historical-fixture regressions
# ---------------------------------------------------------------------------


class TestHistoricalFixtures:
    """Named regressions from real soak runs. Each test pins a known
    bad line from a past episode and asserts the matching validator
    catches it. New lines added here as soak surfaces new failure
    modes.

    Per the operator gate from the final plan:
        'every historical flat line, length drift, and pronoun
        inversion is caught.'
    """

    def test_run1_forfeit_to_the_void_banned(self):
        """Episode signal_lost_bioluminescent_trench_descent (run 1
        of prior session, 2026-05-25). Operator flagged 'forfeit to
        the void' as purple prose. Banned-phrase validator must
        catch it."""
        plan = _plan()
        beat = plan.beats[1]  # length target 24 -> band [12..40]
        text = (
            "She knew the descent was a forfeit to the void of the "
            "deep ocean and there was no turning back."
        )
        result = validate_line(plan, beat, text)
        assert any(
            f.code == CODE_BANNED_PHRASE
            and "forfeit to the void" in (f.got or "")
            for f in result.errors
        )

    def test_run1_expected_toll_banned(self):
        """Same run, second flagged purple phrase 'expected toll'."""
        plan = _plan()
        beat = plan.beats[1]
        text = (
            "She paid the expected toll the Indian Ocean would claim "
            "from any vessel that dared the trench at this depth."
        )
        result = validate_line(plan, beat, text)
        assert any(
            f.code == CODE_BANNED_PHRASE
            and "expected toll" in (f.got or "")
            for f in result.errors
        )

    def test_pronoun_inversion_male_speaker_she(self):
        """Operator-flagged class: a male-coded character (e.g. the
        previous 'Reginald is female' run) referring to themselves
        with female pronouns. Pronoun-consistency must catch this."""
        plan = _plan()
        beat = plan.beats[1]   # DALE PORTER, he/him
        text = "She braced against the bulkhead and ran the diagnostic."
        result = validate_line(plan, beat, text)
        assert any(
            f.code == CODE_PRONOUN_MISMATCH for f in result.errors
        )

    def test_zero_word_dialogue_caught(self):
        """The 'length_target_words=0' failure family (3-D root cause)
        in terms of RENDERED line text -- a rendered line of zero
        words must be flagged by length_drift."""
        plan = _plan()
        beat = plan.beats[1]
        result = validate_line(plan, beat, "")
        assert any(
            f.code == CODE_LENGTH_DRIFT for f in result.errors
        )

    def test_stage_direction_leak_caught(self):
        """The OTR_LineComposer 'narration-leak detected' family
        (visible in many soak logs as 'polish_line firing' messages)
        -- speaker-leak validator catches it post-render."""
        plan = _plan()
        beat = plan.beats[1]
        text = (
            "I see it now -- [Dale walks across the deck and squints "
            "at the console] -- the carrier is here."
        )
        result = validate_line(plan, beat, text)
        assert any(
            f.code == CODE_SPEAKER_LEAK for f in result.errors
        )


# ---------------------------------------------------------------------------
# Default banned-phrase list shape
# ---------------------------------------------------------------------------


class TestDefaultBannedPhrases:
    def test_list_is_non_empty(self):
        assert len(DEFAULT_BANNED_PHRASES) > 0

    def test_operator_flagged_present(self):
        # Operator-flagged purple-prose markers must be in the
        # default list.
        for phrase in ("forfeit to the void", "expected toll"):
            assert phrase in DEFAULT_BANNED_PHRASES

    def test_all_entries_lowercase_or_consistent(self):
        # The validator lower-cases at compare time, so the list
        # values don't strictly need to be lowercase -- but for
        # readability + grep we keep them lowercase by convention.
        for phrase in DEFAULT_BANNED_PHRASES:
            assert isinstance(phrase, str)
            assert phrase.strip() == phrase
            assert len(phrase) > 0


# ---------------------------------------------------------------------------
# ValidationResult shape
# ---------------------------------------------------------------------------


class TestValidationResultShape:
    def test_empty_result_ok(self):
        r = ValidationResult()
        assert r.ok is True
        assert r.errors == []
        assert r.warns == []

    def test_to_dict_serializes(self):
        r = ValidationResult(findings=[
            ValidationFinding(
                severity="error", code="length_drift",
                beat_id="b001", speaker="DALE",
                message="too short",
            ),
        ])
        d = r.to_dict()
        assert d["ok"] is False
        assert d["error_count"] == 1
        assert d["warn_count"] == 0
        assert len(d["findings"]) == 1
        assert d["findings"][0]["code"] == "length_drift"
