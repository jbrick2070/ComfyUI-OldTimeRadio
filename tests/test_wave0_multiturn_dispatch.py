"""Sprint 10B Wave 0 -- in-loop multiturn dispatch regression.

Covers the wire-up between the production writer and the dormant
Stage 2 multi-turn composer:

  * nodes/_otr_wave0_multiturn.line_request_to_stage1_beat
      adapter from legacy outline Beat shape to Stage1Beat.

  * nodes/_otr_wave0_multiturn.build_inloop_stage1_plan
      builds a Stage1Plan from in-loop writer state (outline, cast_rows,
      meta) via the existing legacy_ledger_to_stage1_plan path.

  * nodes/_otr_wave0_multiturn.dispatch_compose_line
      branches on use_multiturn_dialogue widget. Tests cover:
        - widget OFF -> path='legacy'
        - widget ON + reserved speaker (ANNOUNCER) -> 'legacy_fallback'
        - widget ON + plan None -> 'legacy_fallback'
        - widget ON + multiturn raises -> 'legacy_fallback'
        - widget ON + roleplay_multiturn mode -> 'multiturn'
        - widget ON + roleplay_single_turn mode -> 'multiturn'
        - known-bad fixture beat (speaker-leak-prone) does not crash
          the multiturn path; either ships a clean line or falls back.

Tests do NOT load a real LLM; generate_fn is mocked with canned
strings. The 30 green Stage 2 internals tests in
tests/test_stage2_multiturn.py cover compose_line itself; this file
covers the dispatch glue.

# LLM slot: technical
# Reason: this is a regression test module. The slot sweep accepts
# the tag here even though no LLM is called.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import pytest

from nodes._otr_stage1_plan import (
    Stage1Arc,
    Stage1Beat,
    Stage1CastMember,
    Stage1Plan,
)
from nodes._otr_wave0_multiturn import (
    DEFAULT_MULTITURN_MODE,
    RESERVED_SPEAKERS,
    MultiturnDispatchResult,
    build_inloop_stage1_plan,
    dispatch_compose_line,
    line_request_to_stage1_beat,
)


# ---------------------------------------------------------------------------
# Fixtures -- mocks for legacy Beat + cast_rows + outline
# ---------------------------------------------------------------------------


@dataclass
class _LegacyBeat:
    """Duck-typed legacy outline Beat for adapter tests."""
    beat_id: str = "b002"
    speaker: str = "DALE PORTER"
    speaker_role: str = "character"
    intent: str = "Notice the off-band carrier and call it out."
    mood: str = "alert calm"
    target_words: int = 24
    arc_phase: str = ""
    sfx_cue: str = ""


@dataclass
class _LegacyOutline:
    """Duck-typed legacy Outline for plan-builder tests."""
    premise: str = (
        "Two technicians at a Mars relay catch a signal that should "
        "not exist."
    )
    beats: List[_LegacyBeat] = field(default_factory=list)


def _cast_rows() -> List[dict]:
    return [
        {
            "char_id": "c1",
            "name": "DALE PORTER",
            "gender": "male",
            "voice_preset": "v2/en_speaker_5",
            "character_description": (
                "Experienced relay technician. Measured cadences; "
                "trusts procedure."
            ),
            "role": "anchor of competence",
        },
        {
            "char_id": "c2",
            "name": "MEG SAARI",
            "gender": "female",
            "voice_preset": "v2/en_speaker_2",
            "character_description": (
                "Junior tech, six months in, quick to question."
            ),
            "role": "pressure source",
        },
    ]


def _outline() -> _LegacyOutline:
    return _LegacyOutline(
        beats=[
            _LegacyBeat(
                beat_id="b001", speaker="ANNOUNCER",
                speaker_role="announcer",
                intent="Open the episode.", mood="neutral",
                target_words=18,
            ),
            _LegacyBeat(
                beat_id="b002", speaker="DALE PORTER",
                speaker_role="character",
                intent="Notice the off-band carrier and call it out.",
                mood="alert calm", target_words=24,
            ),
            _LegacyBeat(
                beat_id="b003", speaker="MEG SAARI",
                speaker_role="character",
                intent="Ask Dale what the carrier means.",
                mood="alert curious", target_words=22,
            ),
        ],
    )


def _plan_with_dale_meg() -> Stage1Plan:
    """A complete Stage1Plan for dispatch tests that need one without
    going through build_inloop_stage1_plan."""
    return Stage1Plan(
        premise="Two technicians at a Mars relay catch a signal that should not exist.",
        arc=Stage1Arc(
            setup="Routine night shift, calm comms traffic.",
            complication="An off-band carrier locks in and refuses to drop.",
            resolution="They identify the source and decide whether to acknowledge it.",
        ),
        cast=[
            Stage1CastMember(
                name="DALE PORTER", gender="male", pronouns="he/him",
                voice_id="v2/en_speaker_5",
                persona="Experienced relay technician. Measured cadences; trusts procedure.",
                arc_role="anchor of competence",
            ),
            Stage1CastMember(
                name="MEG SAARI", gender="female", pronouns="she/her",
                voice_id="v2/en_speaker_2",
                persona="Junior tech, six months in, quick to question.",
                arc_role="pressure source",
            ),
        ],
        beats=[
            Stage1Beat(
                beat_id="b001", speaker="ANNOUNCER",
                intent="Open the episode.", length_target_words=18,
                emotional_register="neutral", callback_to=None,
            ),
            Stage1Beat(
                beat_id="b002", speaker="DALE PORTER",
                intent="Notice the off-band carrier and call it out.",
                length_target_words=24,
                emotional_register="alert calm", callback_to=None,
            ),
            Stage1Beat(
                beat_id="b003", speaker="MEG SAARI",
                intent="Ask Dale what the carrier means.",
                length_target_words=22,
                emotional_register="alert curious", callback_to="b002",
            ),
        ],
        running_facts=["The relay is on Mars.", "Dale is senior to Meg."],
    )


# ---------------------------------------------------------------------------
# Adapter -- line_request_to_stage1_beat
# ---------------------------------------------------------------------------


class TestLineRequestToStage1Beat:
    def test_basic_character_beat(self):
        legacy = _LegacyBeat()
        s1 = line_request_to_stage1_beat(legacy, fallback_index=1)
        assert isinstance(s1, Stage1Beat)
        assert s1.beat_id == "b002"
        assert s1.speaker == "DALE PORTER"
        assert s1.intent == legacy.intent
        assert s1.length_target_words == 24
        assert s1.emotional_register == "alert calm"

    def test_coerces_missing_beat_id_with_fallback_index(self):
        legacy = _LegacyBeat(beat_id="")
        s1 = line_request_to_stage1_beat(legacy, fallback_index=5)
        assert s1.beat_id == "b005"

    def test_pads_short_intent(self):
        legacy = _LegacyBeat(intent="hm")
        s1 = line_request_to_stage1_beat(legacy)
        assert len(s1.intent) >= 5
        assert "Beat intent not preserved" in s1.intent

    def test_clamps_target_words_below_min(self):
        legacy = _LegacyBeat(target_words=1)
        s1 = line_request_to_stage1_beat(legacy)
        assert s1.length_target_words == 5

    def test_clamps_target_words_above_max(self):
        legacy = _LegacyBeat(target_words=10_000)
        s1 = line_request_to_stage1_beat(legacy)
        assert s1.length_target_words == 200

    def test_normalizes_empty_mood_to_neutral(self):
        legacy = _LegacyBeat(mood="")
        s1 = line_request_to_stage1_beat(legacy)
        assert s1.emotional_register == "neutral"

    def test_accepts_announcer_speaker(self):
        legacy = _LegacyBeat(speaker="ANNOUNCER")
        s1 = line_request_to_stage1_beat(legacy)
        # Adapter is content-neutral; the dispatch layer routes
        # reserved speakers to legacy.
        assert s1.speaker == "ANNOUNCER"

    def test_coerces_non_bnnn_beat_id(self):
        legacy = _LegacyBeat(beat_id="beat_42")
        s1 = line_request_to_stage1_beat(legacy)
        assert s1.beat_id == "b042"


# ---------------------------------------------------------------------------
# Plan builder -- build_inloop_stage1_plan
# ---------------------------------------------------------------------------


class TestBuildInloopStage1Plan:
    def test_basic_plan_built(self):
        plan = build_inloop_stage1_plan(
            outline=_outline(),
            cast_rows=_cast_rows(),
            meta={},
        )
        assert plan is not None
        assert plan.premise.startswith("Two technicians")
        cast_names = {c.name for c in plan.cast}
        assert "DALE PORTER" in cast_names
        assert "MEG SAARI" in cast_names

    def test_returns_none_when_no_cast(self):
        plan = build_inloop_stage1_plan(
            outline=_outline(),
            cast_rows=[],
            meta={},
        )
        assert plan is None

    def test_returns_none_when_cast_only_reserved(self):
        plan = build_inloop_stage1_plan(
            outline=_outline(),
            cast_rows=[
                {"char_id": "c0", "name": "ANNOUNCER"},
            ],
            meta={},
        )
        assert plan is None

    def test_pulls_running_facts_from_continuity_ledger(self):
        plan = build_inloop_stage1_plan(
            outline=_outline(),
            cast_rows=_cast_rows(),
            meta={
                "continuity_ledger": {
                    "facts": [
                        {"text": "The relay is on Mars."},
                        "Dale is senior to Meg.",
                    ],
                },
            },
        )
        assert plan is not None
        assert "The relay is on Mars." in plan.running_facts
        assert "Dale is senior to Meg." in plan.running_facts


# ---------------------------------------------------------------------------
# Dispatch -- dispatch_compose_line
# ---------------------------------------------------------------------------


def _mock_generate_fn_text(text: str):
    """Return a generate_fn that always returns the given text."""
    def fn(messages, *, temperature=0.85, max_new_tokens=200):
        return text
    return fn


def _mock_generate_fn_raises(exc_cls=RuntimeError, message="boom"):
    def fn(messages, *, temperature=0.85, max_new_tokens=200):
        raise exc_cls(message)
    return fn


class TestDispatchRouting:
    def test_widget_off_returns_path_legacy(self):
        plan = _plan_with_dale_meg()
        beat = plan.beats[1]   # DALE PORTER
        result = dispatch_compose_line(
            use_multiturn_dialogue=False,
            plan=plan,
            beat=beat,
            creative_generate_fn=_mock_generate_fn_text(""),
        )
        assert isinstance(result, MultiturnDispatchResult)
        assert result.path == "legacy"
        assert result.text == ""
        assert result.record is None

    def test_widget_on_plan_none_falls_back(self):
        plan = _plan_with_dale_meg()
        beat = plan.beats[1]
        result = dispatch_compose_line(
            use_multiturn_dialogue=True,
            plan=None,
            beat=beat,
            creative_generate_fn=_mock_generate_fn_text("anything"),
        )
        assert result.path == "legacy_fallback"
        assert result.text == ""
        assert "returned None" in (result.fallback_reason or "")

    def test_widget_on_reserved_speaker_falls_back(self):
        plan = _plan_with_dale_meg()
        announcer_beat = plan.beats[0]
        assert announcer_beat.speaker in RESERVED_SPEAKERS
        result = dispatch_compose_line(
            use_multiturn_dialogue=True,
            plan=plan,
            beat=announcer_beat,
            creative_generate_fn=_mock_generate_fn_text("anything"),
        )
        assert result.path == "legacy_fallback"
        assert "reserved" in (result.fallback_reason or "").lower()

    def test_widget_on_speaker_not_in_cast_falls_back(self):
        plan = _plan_with_dale_meg()
        beat = Stage1Beat(
            beat_id="b099",
            speaker="UNKNOWN PERSON",
            intent="Try to speak from outside the cast roster.",
            length_target_words=20,
            emotional_register="confused",
            callback_to=None,
        )
        result = dispatch_compose_line(
            use_multiturn_dialogue=True,
            plan=plan,
            beat=beat,
            creative_generate_fn=_mock_generate_fn_text("hi"),
        )
        assert result.path == "legacy_fallback"
        assert "not in plan.cast" in (result.fallback_reason or "")

    def test_widget_on_generate_fn_raises_falls_back(self):
        plan = _plan_with_dale_meg()
        beat = plan.beats[1]
        # All N candidates raise inside compose_line; record still
        # comes back (Stage 2 catches per-candidate exceptions). The
        # record's chosen_text is empty -> dispatch returns
        # legacy_fallback with the record attached.
        result = dispatch_compose_line(
            use_multiturn_dialogue=True,
            plan=plan,
            beat=beat,
            creative_generate_fn=_mock_generate_fn_raises(),
            multiturn_n=2,
        )
        assert result.path == "legacy_fallback"
        assert result.text == ""

    def test_widget_on_roleplay_multiturn_produces_line(self):
        plan = _plan_with_dale_meg()
        beat = plan.beats[1]
        text = "The carrier is locked. Off-band, not ours."
        result = dispatch_compose_line(
            use_multiturn_dialogue=True,
            plan=plan,
            beat=beat,
            creative_generate_fn=_mock_generate_fn_text(text),
            multiturn_mode="roleplay_multiturn",
            multiturn_n=2,
        )
        assert result.path == "multiturn"
        assert result.text == text
        assert result.record is not None
        assert result.record.mode == "roleplay_multiturn"

    def test_widget_on_roleplay_single_turn_produces_line(self):
        plan = _plan_with_dale_meg()
        beat = plan.beats[1]
        text = "Off-band carrier. Holding lock."
        result = dispatch_compose_line(
            use_multiturn_dialogue=True,
            plan=plan,
            beat=beat,
            creative_generate_fn=_mock_generate_fn_text(text),
            multiturn_mode="roleplay_single_turn",
            multiturn_n=2,
        )
        assert result.path == "multiturn"
        assert result.text == text
        assert result.record is not None
        assert result.record.mode == "roleplay_single_turn"

    def test_default_mode_is_roleplay_multiturn(self):
        plan = _plan_with_dale_meg()
        beat = plan.beats[1]
        result = dispatch_compose_line(
            use_multiturn_dialogue=True,
            plan=plan,
            beat=beat,
            creative_generate_fn=_mock_generate_fn_text("ok."),
            multiturn_n=2,
        )
        assert DEFAULT_MULTITURN_MODE == "roleplay_multiturn"
        assert result.path == "multiturn"
        assert result.record.mode == "roleplay_multiturn"


# ---------------------------------------------------------------------------
# Known-bad fixture -- speaker-leak-prone beat does not crash multiturn
# ---------------------------------------------------------------------------


class TestKnownBadFixture:
    """The Spray of Hope production failure (`__01_problem-statement.md`
    Section 1.1) shipped an announcer narrating embedded character
    dialogue: `Breathless, Ren Black mutters, "..."`. The Wave 0
    dispatch path must NOT crash when the multiturn composer is asked
    to render a beat whose generated candidate text trips a Stage 3
    validator (speaker_leak). The composer's scorer penalizes such
    candidates and ships the highest non-zero scorer; if every
    candidate is broken, the dispatch falls back to legacy without
    raising. Either path keeps PD1 alive.
    """

    def test_speaker_leak_prone_text_does_not_crash(self):
        plan = _plan_with_dale_meg()
        beat = plan.beats[1]   # DALE PORTER
        # Canned text that simulates the Spray of Hope failure shape:
        # the candidate quotes the speaker's name + says "mutters".
        # Stage 3 validators (speaker_leak / banned_phrase) will
        # penalize this; the scorer still ships SOMETHING.
        leaky_text = (
            'Breathless, DALE PORTER mutters, '
            '"the carrier is locked, off-band, not ours."'
        )
        result = dispatch_compose_line(
            use_multiturn_dialogue=True,
            plan=plan,
            beat=beat,
            creative_generate_fn=_mock_generate_fn_text(leaky_text),
            multiturn_n=3,
        )
        # Either path is acceptable -- the test asserts NO CRASH and
        # that a result object comes back with a path string set.
        assert isinstance(result, MultiturnDispatchResult)
        assert result.path in {"multiturn", "legacy_fallback"}
        # If multiturn shipped the line, the record must be populated.
        if result.path == "multiturn":
            assert result.record is not None
            assert result.text != ""

    def test_empty_candidate_text_falls_back_cleanly(self):
        plan = _plan_with_dale_meg()
        beat = plan.beats[1]
        result = dispatch_compose_line(
            use_multiturn_dialogue=True,
            plan=plan,
            beat=beat,
            creative_generate_fn=_mock_generate_fn_text(""),
            multiturn_n=3,
        )
        assert result.path == "legacy_fallback"
        assert result.text == ""
