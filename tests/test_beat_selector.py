"""Sprint 4 (2026-05-28) -- best-of-N beat sheet selector.

Pinned invariants:
  - score_beat_sheet returns 0/1 on each of the five axes; total
    in [0..5].
  - select_winning_beat_sheet picks the highest-total eligible
    candidate; ties break on lowest index (deterministic).
  - All-invalid input raises NoValidBeatSheetError; the audit is
    attached to the exception.
  - Eligibility = validate_beat_sheet returns empty list.
  - A candidate carrying the alarm/countdown/rescue pattern fails
    the no_alarm_countdown_rescue penalty axis.
"""
from __future__ import annotations

import pytest

from nodes._otr_beat_selector import (
    BeatSelectorScores,
    NoValidBeatSheetError,
    score_beat_sheet,
    select_winning_beat_sheet,
)
from nodes._otr_dramatic_state import DramaticState
from nodes._otr_stage1_plan import (
    Stage1Arc,
    Stage1Beat,
    Stage1CastMember,
    Stage1Plan,
    stamp_dialogue_slot_ids,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _ds_clean(costly_choice: str = "d003") -> DramaticState:
    return DramaticState(
        dramatic_question="Will Maeve sign the confession before the audit closes?",
        character_a_wants="hide the falsified ledger before the audit",
        character_b_wants="get Maeve's signature on the record tonight",
        costly_choice_beat=costly_choice,
        ending_change="the falsified ledger is now part of the public record",
    )


def _clean_beats(*, override_b003_state_after: str | None = None) -> list[Stage1Beat]:
    """Default 4-voiced-beat shape; b003 is the costly choice."""
    return [
        Stage1Beat(
            beat_id="b001", speaker="ANNOUNCER",
            intent="open the episode",
            length_target_words=15,
            emotional_register="welcoming",
            state_before="audit has not begun",
            state_after="audit window is open",
            tension=2,
        ),
        Stage1Beat(
            beat_id="b002", speaker="ALICE",
            intent="discover the missing record",
            length_target_words=20,
            emotional_register="tight unease",
            state_before="ledger somewhere",
            state_after="ledger missing",
            tension=3,
        ),
        Stage1Beat(
            beat_id="b003", speaker="ALICE",
            intent="make the costly choice",
            length_target_words=30,
            emotional_register="grim resolve",
            state_before="alice undecided",
            state_after=(
                override_b003_state_after
                if override_b003_state_after is not None
                else "alice has signed"
            ),
            tension=5,
        ),
        Stage1Beat(
            beat_id="b004", speaker="ANNOUNCER",
            intent="close the episode",
            length_target_words=15,
            emotional_register="quiet aftermath",
            state_before="signature is real",
            state_after="record is public",
            tension=3,
        ),
    ]


def _plan(*, beats, with_state=True, costly="d003") -> Stage1Plan:
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
                persona="weary forensic engineer with dry humor.",
                arc_role="reluctant insider",
            ),
            Stage1CastMember(
                name="BOB", gender="male", pronouns="he/him",
                voice_id="v2/en_speaker_5",
                persona="ambitious grant officer evasive about funding.",
                arc_role="pressure source",
            ),
        ],
        beats=beats,
        running_facts=["the audit closes at midnight"],
        dramatic_state=_ds_clean(costly_choice=costly) if with_state else None,
    )
    stamp_dialogue_slot_ids(plan)
    return plan


# ---------------------------------------------------------------------------
# score_beat_sheet
# ---------------------------------------------------------------------------


def test_score_clean_plan_is_five():
    """A clean plan with DramaticState, populated wants, resolved
    costly choice, no dead beats, ending change, and no alarm
    pattern scores 5/5."""
    plan = _plan(beats=_clean_beats())
    s = score_beat_sheet(plan)
    assert s == BeatSelectorScores(1, 1, 1, 1, 1)
    assert s.total == 5


def test_score_no_dramatic_state_loses_three_axes():
    """No DramaticState: clear_opposed_desires +
    costly_choice_present + ending_changed_from_beginning all fail.
    Each-beat-changes-situation may still pass, alarm axis still
    passes."""
    plan = _plan(beats=_clean_beats(), with_state=False)
    s = score_beat_sheet(plan)
    assert s.clear_opposed_desires == 0
    assert s.costly_choice_present == 0
    assert s.ending_changed_from_beginning == 0
    assert s.each_beat_changes_situation == 1
    assert s.no_alarm_countdown_rescue == 1
    assert s.total == 2


def test_score_dead_costly_choice_loses_that_axis():
    plan = _plan(beats=_clean_beats(override_b003_state_after="alice undecided"))
    s = score_beat_sheet(plan)
    assert s.costly_choice_present == 0
    # The dead beat also flips each_beat_changes_situation.
    assert s.each_beat_changes_situation == 0


def test_score_alarm_pattern_loses_penalty_axis():
    beats = _clean_beats()
    beats[1] = Stage1Beat(
        beat_id="b002", speaker="ALICE",
        intent="alarm sounds and the countdown begins",
        length_target_words=20,
        emotional_register="tight panic",
        state_before="audit room is calm",
        state_after="audit room is in chaos",
        tension=4,
    )
    plan = _plan(beats=beats)
    s = score_beat_sheet(plan)
    assert s.no_alarm_countdown_rescue == 0


def test_score_unchanged_ending_loses_that_axis():
    """ending_change normalize-equal to opening state."""
    plan = _plan(beats=_clean_beats())
    plan.dramatic_state.ending_change = plan.beats[0].state_before
    s = score_beat_sheet(plan)
    assert s.ending_changed_from_beginning == 0


# ---------------------------------------------------------------------------
# select_winning_beat_sheet
# ---------------------------------------------------------------------------


def test_selector_picks_higher_total():
    """Two valid candidates: the one with higher structural total wins."""
    candidate_a = _plan(beats=_clean_beats())  # 5/5
    # candidate_b sacrifices the alarm-clean axis (still passes
    # validate_beat_sheet because alarm pattern is a SCORE axis, not
    # a validator).
    beats_b = _clean_beats()
    beats_b[1] = Stage1Beat(
        beat_id="b002", speaker="ALICE",
        intent="alarm sounds; countdown begins; tight",
        length_target_words=20,
        emotional_register="tight panic",
        state_before="audit room is calm",
        state_after="audit room is in chaos",
        tension=4,
    )
    candidate_b = _plan(beats=beats_b)
    winner, audit = select_winning_beat_sheet([candidate_b, candidate_a])
    assert winner is candidate_a
    assert audit.winner_index == 1
    assert audit.eligible_indices == [0, 1]
    assert audit.scores_per_candidate[1].total == 5
    assert audit.scores_per_candidate[0].total == 4


def test_selector_tie_break_lowest_index():
    """Two equally-strong valid candidates -- lowest index wins
    (deterministic, reproducible)."""
    c1 = _plan(beats=_clean_beats())
    c2 = _plan(beats=_clean_beats())
    winner, audit = select_winning_beat_sheet([c1, c2])
    assert winner is c1
    assert audit.winner_index == 0


def test_selector_rejects_invalid_even_if_score_is_high():
    """An invalid candidate (validate_beat_sheet returns defects)
    is INELIGIBLE regardless of structural score."""
    c_invalid = _plan(beats=_clean_beats())
    # Break the costly choice: name a slot that does not exist.
    c_invalid.dramatic_state.costly_choice_beat = "d099"
    c_valid = _plan(beats=_clean_beats())
    winner, audit = select_winning_beat_sheet([c_invalid, c_valid])
    assert winner is c_valid
    assert audit.winner_index == 1
    assert audit.eligible_indices == [1]
    assert audit.defects_per_candidate[0]


def test_selector_raises_when_all_invalid():
    """Plan principle: no silent shipping of a dead sheet."""
    bad_beats = _clean_beats(override_b003_state_after="alice undecided")
    cs = [_plan(beats=bad_beats) for _ in range(3)]
    for c in cs:
        c.dramatic_state.costly_choice_beat = "d099"
    with pytest.raises(NoValidBeatSheetError) as ctx:
        select_winning_beat_sheet(cs)
    # Audit attached to the exception for caller stamping.
    audit = getattr(ctx.value, "audit", None)
    assert audit is not None
    assert audit.eligible_indices == []
    assert len(audit.scores_per_candidate) == 3
    assert all(audit.defects_per_candidate)


def test_selector_raises_on_empty_input():
    with pytest.raises(NoValidBeatSheetError) as ctx:
        select_winning_beat_sheet([])
    audit = getattr(ctx.value, "audit", None)
    assert audit is not None
    assert audit.winner_index == -1


def test_audit_to_dict_serializes_cleanly():
    """The audit dict must be JSON-friendly for stamping on meta."""
    import json
    winner, audit = select_winning_beat_sheet([_plan(beats=_clean_beats())])
    blob = json.dumps(audit.to_dict())
    assert "winner_index" in blob
    assert "scores_per_candidate" in blob
