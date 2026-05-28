"""Sprint 2 (2026-05-28) -- DramaticState + structural beat validators.

Pinned invariants:
  - DramaticState schema accepts well-formed input and rejects
    obvious mistakes (echoed wants, missing fields).
  - Stage1Beat carries the Sprint 2 typed-state fields optionally;
    legacy plans still parse.
  - validate_beat_sheet returns [] on clean fixtures; non-empty
    defect list on hand-crafted broken fixtures. The defect string
    starts with one of the DefectKind constants.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from nodes._otr_dramatic_state import DramaticState
from nodes._otr_beat_validators import (
    DefectKind,
    is_dead_beat,
    validate_beat_sheet,
)
from nodes._otr_stage1_plan import (
    Stage1Arc,
    Stage1Beat,
    Stage1CastMember,
    Stage1Plan,
    stamp_dialogue_slot_ids,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ds_clean() -> DramaticState:
    return DramaticState(
        dramatic_question="Will Maeve sign the confession before the audit closes?",
        character_a_wants="hide the falsified ledger before the audit",
        character_b_wants="get Maeve's signature on the record tonight",
        costly_choice_beat="d003",
        ending_change="the falsified ledger is now part of the public record",
    )


def _plan_factory(*, with_state: bool, beats: list[Stage1Beat]) -> Stage1Plan:
    plan = Stage1Plan(
        premise="A short test premise long enough to pass schema.",
        arc=Stage1Arc(
            setup="setup statement long enough to validate.",
            complication="complication statement long enough to validate.",
            resolution="resolution statement long enough to validate.",
        ),
        cast=[
            Stage1CastMember(
                name="ALICE", gender="female", pronouns="she/her",
                voice_id="v2/en_speaker_3",
                persona="weary forensic engineer with dry humor and "
                        "twenty years on the night shift.",
                arc_role="reluctant insider",
            ),
            Stage1CastMember(
                name="BOB", gender="male", pronouns="he/him",
                voice_id="v2/en_speaker_5",
                persona="ambitious grant officer evasive about funding "
                        "and prone to deflection.",
                arc_role="pressure source",
            ),
        ],
        beats=beats,
        running_facts=["the audit closes at midnight"],
        dramatic_state=_ds_clean() if with_state else None,
    )
    stamp_dialogue_slot_ids(plan)
    return plan


def _clean_beats() -> list[Stage1Beat]:
    return [
        Stage1Beat(
            beat_id="b001", speaker="ANNOUNCER",
            intent="open the episode",
            length_target_words=15,
            emotional_register="welcoming",
            state_before="the audit has not yet begun",
            state_after="the audit window is now open",
            turn="the clock starts",
            tension=2,
            next_turn="alice realizes the ledger is missing",
        ),
        Stage1Beat(
            beat_id="b002", speaker="ALICE",
            intent="discover the missing record",
            length_target_words=20,
            emotional_register="tight unease",
            state_before="ledger is somewhere in the office",
            state_after="ledger is confirmed missing",
            turn="alice finds the empty drawer",
            tension=3,
            next_turn="bob arrives and presses for the signature",
        ),
        Stage1Beat(
            beat_id="b003", speaker="ALICE",
            intent="make the costly choice",
            length_target_words=30,
            emotional_register="grim resolve",
            state_before="alice has not yet decided",
            state_after="alice agrees to sign the record",
            turn="alice makes the irreversible commitment",
            tension=5,
            next_turn="the signed ledger becomes public record",
        ),
        Stage1Beat(
            beat_id="b004", speaker="ANNOUNCER",
            intent="close the episode",
            length_target_words=15,
            emotional_register="quiet aftermath",
            state_before="the signature is now real",
            state_after="the record is now public",
            turn="the audit closes with the record",
            tension=3,
            next_turn="end of episode",
        ),
    ]


# ---------------------------------------------------------------------------
# DramaticState schema
# ---------------------------------------------------------------------------


def test_dramatic_state_happy_path():
    ds = _ds_clean()
    assert ds.costly_choice_beat == "d003"
    assert ds.dramatic_question.endswith("?")


def test_dramatic_state_rejects_echoed_wants():
    """Round-robin floor: the LLM cannot satisfy opposed-desires by
    emitting the same string twice."""
    with pytest.raises(ValidationError) as ctx:
        DramaticState(
            dramatic_question="Will the bridge hold under the train?",
            character_a_wants="cross the bridge with the cargo",
            character_b_wants="Cross the Bridge with the Cargo",
            costly_choice_beat="d004",
            ending_change="the train reaches the other side",
        )
    assert "differ" in str(ctx.value)


def test_dramatic_state_rejects_bad_slot_id_pattern():
    with pytest.raises(ValidationError):
        DramaticState(
            dramatic_question="Will the audit close in time?",
            character_a_wants="hide the falsified ledger",
            character_b_wants="get the signature tonight",
            costly_choice_beat="b003",   # b-pattern not d-pattern
            ending_change="the record is now public",
        )


# ---------------------------------------------------------------------------
# Stage1Beat Sprint 2 fields
# ---------------------------------------------------------------------------


def test_stage1beat_sprint2_fields_optional():
    """A beat with no Sprint 2 fields still parses (Sprint 2 is
    optional during this sprint -- back-compat for legacy producers)."""
    b = Stage1Beat(
        beat_id="b001", speaker="ALICE",
        intent="set up the question",
        length_target_words=20,
        emotional_register="curious dread",
    )
    assert b.objective is None
    assert b.state_before is None
    assert b.tension is None


def test_stage1beat_sprint2_fields_round_trip():
    b = Stage1Beat(
        beat_id="b002", speaker="ALICE",
        intent="set up the question",
        length_target_words=20,
        emotional_register="curious dread",
        objective="press maeve to admit the omission",
        obstacle="maeve has rehearsed her denials",
        turn="maeve names her own forged signature",
        tactics_used="open-ended question with long silence",
        state_before="maeve is hiding the omission",
        state_after="maeve has named her own forgery aloud",
        subtext="alice already knows the answer",
        tension=4,
        next_turn="bob enters with the audit clock",
    )
    assert b.tension == 4
    assert b.next_turn.startswith("bob enters")


def test_stage1beat_tension_range_enforced():
    with pytest.raises(ValidationError):
        Stage1Beat(
            beat_id="b002", speaker="ALICE",
            intent="set up the question",
            length_target_words=20,
            emotional_register="curious dread",
            tension=6,
        )


# ---------------------------------------------------------------------------
# Beat validators
# ---------------------------------------------------------------------------


def test_validate_beat_sheet_clean_returns_empty():
    plan = _plan_factory(with_state=True, beats=_clean_beats())
    assert validate_beat_sheet(plan) == []


def test_validate_dead_beat_flagged():
    beats = _clean_beats()
    # Force a dead beat on b002.
    beats[1] = Stage1Beat(
        beat_id="b002", speaker="ALICE",
        intent="recover the ledger",
        length_target_words=20,
        emotional_register="quiet alarm",
        state_before="alice notices the missing ledger",
        state_after="alice notices the missing ledger",  # dead
        tension=2,
    )
    plan = _plan_factory(with_state=True, beats=beats)
    defects = validate_beat_sheet(plan)
    assert any(d.startswith(DefectKind.DEAD_BEAT) for d in defects)


def test_validate_unresolved_costly_choice_flagged():
    """costly_choice_beat exists but its state does not change."""
    beats = _clean_beats()
    # Force b003 to be a dead beat.
    beats[2] = Stage1Beat(
        beat_id="b003", speaker="ALICE",
        intent="make the costly choice",
        length_target_words=30,
        emotional_register="grim resolve",
        state_before="alice is undecided about signing",
        state_after="alice is undecided about signing",  # same
        tension=5,
    )
    plan = _plan_factory(with_state=True, beats=beats)
    defects = validate_beat_sheet(plan)
    assert any(
        d.startswith(DefectKind.UNRESOLVED_COSTLY_CHOICE)
        or d.startswith(DefectKind.DEAD_BEAT)
        for d in defects
    )


def test_validate_missing_costly_choice_target_flagged():
    """costly_choice_beat names a slot that does not exist."""
    plan = _plan_factory(with_state=True, beats=_clean_beats())
    plan.dramatic_state.costly_choice_beat = "d099"
    defects = validate_beat_sheet(plan)
    assert any(d.startswith(DefectKind.NO_COSTLY_CHOICE) for d in defects)


def test_validate_unchanged_ending_flagged():
    plan = _plan_factory(with_state=True, beats=_clean_beats())
    # Make ending_change identical to the opening state.
    plan.dramatic_state.ending_change = (
        plan.beats[0].state_before  # opening state of first voiced beat
    )
    defects = validate_beat_sheet(plan)
    assert any(d.startswith(DefectKind.UNCHANGED_ENDING) for d in defects)


def test_validate_legacy_plan_without_dramatic_state_skips_quietly():
    """A plan without dramatic_state should not throw, and only
    dead-beat validator can fire."""
    plan = _plan_factory(with_state=False, beats=_clean_beats())
    assert plan.dramatic_state is None
    assert validate_beat_sheet(plan) == []


def test_is_dead_beat_helper_untyped_beats_not_dead():
    """A beat with no state fields is NOT dead by definition."""
    b = Stage1Beat(
        beat_id="b001", speaker="ALICE",
        intent="set up the question",
        length_target_words=20,
        emotional_register="curious dread",
    )
    assert is_dead_beat(b) is False
