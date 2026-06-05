"""Sprint 5 (2026-05-28) -- editor downgrade to constraint checker.

Pinned invariants:
  * EDITOR_CONSTRAINTS_SYSTEM_PROMPT carries no taste verbs.
  * EditorConstraint.ALL lists the five canonical codes.
  * DEFAULT_MAX_EDITOR_CONSTRAINT_CYCLES == 1 (Sprint 5 cap).
  * check_constraints returns pass_decision=True with empty
    failing_constraints on a clean plan.
  * Each of the five constraints fires on its hand-crafted defect
    fixture.
  * check_constraints does NOT trigger on a structurally-valid-but-
    plain plan (no taste pings).
"""
from __future__ import annotations

from nodes._otr_dramatic_state import DramaticState
from nodes._otr_editor_constraints import (
    DEFAULT_MAX_EDITOR_CONSTRAINT_CYCLES,
    EDITOR_CONSTRAINTS_SYSTEM_PROMPT,
    EditorConstraint,
    EditorConstraintVerdict,
    _cast_first_last_aliases,
    check_constraints,
)
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


def _ds() -> DramaticState:
    return DramaticState(
        dramatic_question="Will Maeve sign the confession before the audit closes?",
        character_a_wants="hide the falsified ledger before the audit",
        character_b_wants="get Maeve's signature on the record tonight",
        costly_choice_beat="d003",
        ending_change="the falsified ledger is now part of the public record",
    )


def _clean_beats() -> list[Stage1Beat]:
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
            intent="discover the missing ledger",
            length_target_words=20,
            emotional_register="tight unease",
            state_before="ledger somewhere",
            state_after="ledger missing",
            tension=3,
        ),
        Stage1Beat(
            beat_id="b003", speaker="ALICE",
            intent="make the costly choice to sign",
            length_target_words=30,
            emotional_register="grim resolve",
            state_before="alice undecided",
            state_after="alice has signed",
            tension=5,
        ),
        Stage1Beat(
            beat_id="b004", speaker="ANNOUNCER",
            intent="close the episode and tag the broadcast",
            length_target_words=15,
            emotional_register="quiet aftermath",
            state_before="signature is real",
            state_after="record is public",
            tension=3,
        ),
    ]


def _plan(beats=None, with_ds: bool = True) -> Stage1Plan:
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
        beats=beats if beats is not None else _clean_beats(),
        running_facts=["the audit closes at midnight"],
        dramatic_state=_ds() if with_ds else None,
    )
    stamp_dialogue_slot_ids(plan)
    return plan


# ---------------------------------------------------------------------------
# Surface invariants
# ---------------------------------------------------------------------------


def test_editor_constraints_prompt_carries_no_taste_verbs():
    """The Sprint 5 prompt must strip 'make it better' / 'improve
    pacing' / 'more drama' verbs."""
    p = EDITOR_CONSTRAINTS_SYSTEM_PROMPT.lower()
    assert "make it better" not in p
    assert "improve pacing" not in p
    assert "more drama" not in p
    assert "rewrite" not in p


def test_constraint_codes_listed_in_all():
    assert EditorConstraint.WRONG_SPEAKER in EditorConstraint.ALL
    assert EditorConstraint.PHANTOM_CHARACTER in EditorConstraint.ALL
    assert EditorConstraint.MISSING_COSTLY_CHOICE in EditorConstraint.ALL
    assert EditorConstraint.NO_FINAL_THIRD_TURN in EditorConstraint.ALL
    assert EditorConstraint.FORMAT_FAILURE in EditorConstraint.ALL
    assert len(EditorConstraint.ALL) == 5


def test_revision_cap_is_one():
    """Sprint 5: hard cap on serial revision turns."""
    assert DEFAULT_MAX_EDITOR_CONSTRAINT_CYCLES == 1


# ---------------------------------------------------------------------------
# Clean plan passes
# ---------------------------------------------------------------------------


def test_clean_plan_passes():
    verdict = check_constraints(_plan())
    assert verdict.pass_decision is True
    assert verdict.failing_constraints == []
    assert verdict.repair_note == ""


def test_structurally_valid_but_plain_plan_passes():
    """A plan that is structurally fine but boring should NOT
    trigger any constraint -- taste is the Sprint 4 selector's job,
    not the editor's."""
    beats = _clean_beats()
    # Strip color but keep state changes structurally intact.
    for b in beats:
        # Reach into model_fields to bypass copy; just rebuild.
        pass
    boring = [
        Stage1Beat(
            beat_id=b.beat_id, speaker=b.speaker,
            intent=f"plain beat {b.beat_id}",
            length_target_words=b.length_target_words,
            emotional_register="neutral",
            state_before=b.state_before,
            state_after=b.state_after,
            tension=b.tension,
        )
        for b in beats
    ]
    plan = _plan(beats=boring)
    verdict = check_constraints(plan)
    assert verdict.pass_decision is True
    assert verdict.failing_constraints == []


# ---------------------------------------------------------------------------
# Each constraint fires on its defect
# ---------------------------------------------------------------------------


def test_wrong_speaker_fires():
    beats = _clean_beats()
    beats[1] = Stage1Beat(
        beat_id="b002",
        speaker="MAEVE",  # not in cast
        intent="discover the missing ledger",
        length_target_words=20,
        emotional_register="tight",
        state_before="ledger somewhere",
        state_after="ledger missing",
        tension=3,
    )
    # Bypass Stage1Plan.validate (which rejects non-cast speakers via
    # validate_plan_semantics) by constructing the plan directly with
    # cast extended.
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
                persona="ambitious grant officer evasive.",
                arc_role="pressure source",
            ),
            # Add MAEVE to cast just so Stage1 parses; then check_constraints
            # is told MAEVE is NOT a valid speaker via a stripped-down
            # cast view.
            Stage1CastMember(
                name="MAEVE", gender="female", pronouns="she/her",
                voice_id="v2/en_speaker_7",
                persona="auditor with rehearsed denials and a thin smile.",
                arc_role="opposition",
            ),
        ],
        beats=beats,
        dramatic_state=_ds(),
    )
    stamp_dialogue_slot_ids(plan)
    # Drop MAEVE from the cast NAME pool for the constraint check by
    # mutating the cast list to ALICE+BOB only.
    plan.cast = plan.cast[:2]
    verdict = check_constraints(plan)
    assert verdict.pass_decision is False
    assert EditorConstraint.WRONG_SPEAKER in verdict.failing_constraints


def test_phantom_character_fires():
    beats = _clean_beats()
    beats[1] = Stage1Beat(
        beat_id="b002", speaker="ALICE",
        intent="discover the missing ledger ROBERT left behind",
        length_target_words=20,
        emotional_register="tight",
        state_before="ledger somewhere",
        state_after="ledger missing",
        tension=3,
    )
    plan = _plan(beats=beats)
    verdict = check_constraints(plan)
    assert EditorConstraint.PHANTOM_CHARACTER in verdict.failing_constraints


def test_missing_costly_choice_fires_on_unresolved_slot():
    plan = _plan()
    plan.dramatic_state.costly_choice_beat = "d099"
    verdict = check_constraints(plan)
    assert EditorConstraint.MISSING_COSTLY_CHOICE in verdict.failing_constraints


def test_missing_costly_choice_fires_on_dead_pivot():
    beats = _clean_beats()
    beats[2] = Stage1Beat(
        beat_id="b003", speaker="ALICE",
        intent="make the costly choice to sign",
        length_target_words=30,
        emotional_register="grim resolve",
        state_before="alice undecided",
        state_after="alice undecided",   # dead pivot
        tension=5,
    )
    plan = _plan(beats=beats)
    verdict = check_constraints(plan)
    assert EditorConstraint.MISSING_COSTLY_CHOICE in verdict.failing_constraints


def test_no_final_third_turn_fires():
    """Every typed beat in the last third is dead."""
    beats = _clean_beats()
    # 4 voiced -> last third = last 1 beat. Make b004 dead.
    beats[3] = Stage1Beat(
        beat_id="b004", speaker="ANNOUNCER",
        intent="close the episode",
        length_target_words=15,
        emotional_register="quiet aftermath",
        state_before="record is public",
        state_after="record is public",  # dead
        tension=3,
    )
    plan = _plan(beats=beats)
    verdict = check_constraints(plan)
    assert EditorConstraint.NO_FINAL_THIRD_TURN in verdict.failing_constraints


def test_format_failure_fires_on_bracket_stage_direction():
    beats = _clean_beats()
    beats[1] = Stage1Beat(
        beat_id="b002", speaker="ALICE",
        intent="discover the missing ledger [SFX: drawer slams]",
        length_target_words=20,
        emotional_register="tight",
        state_before="ledger somewhere",
        state_after="ledger missing",
        tension=3,
    )
    plan = _plan(beats=beats)
    verdict = check_constraints(plan)
    assert EditorConstraint.FORMAT_FAILURE in verdict.failing_constraints


def test_verdict_to_dict_serializes():
    """Verdict must serialize cleanly for stamping on
    meta.editor_constraints."""
    import json
    verdict = check_constraints(_plan())
    blob = json.dumps(verdict.to_dict())
    assert "pass_decision" in blob
    assert "failing_constraints" in blob
    assert "repair_note" in blob
    assert "cycle" in blob


# ---------------------------------------------------------------------------
# Weak-model name-abbreviation resolution (first/last-name aliases)
# ---------------------------------------------------------------------------


def test_cast_first_last_aliases_unambiguous():
    aliases = _cast_first_last_aliases({"NIA HIBBERT", "DJINN SIMPSON"})
    assert aliases == {"NIA", "HIBBERT", "DJINN", "SIMPSON"}


def test_cast_first_last_aliases_excludes_ambiguous_first_name():
    # Two characters share the first name NIA -> 'NIA' must NOT alias (ambiguous);
    # the distinct surnames still do.
    aliases = _cast_first_last_aliases({"NIA HIBBERT", "NIA SIMPSON"})
    assert "NIA" not in aliases
    assert {"HIBBERT", "SIMPSON"} <= aliases


def test_cast_first_last_aliases_ignores_single_token_names():
    assert _cast_first_last_aliases({"ALICE", "BOB"}) == set()


def _plan_with_multiword_cast(speaker: str) -> Stage1Plan:
    """A plan whose cast is two multi-word names; one voiced beat uses `speaker`."""
    plan = Stage1Plan(
        premise="A short test premise long enough to pass schema.",
        arc=Stage1Arc(
            setup="setup statement long enough to validate.",
            complication="complication statement long enough to validate.",
            resolution="resolution statement long enough to validate.",
        ),
        cast=[
            Stage1CastMember(
                name="NIA HIBBERT", gender="female", pronouns="she/her",
                voice_id="v2/en_speaker_3",
                persona="weary forensic engineer with dry humor.",
                arc_role="reluctant insider",
            ),
            Stage1CastMember(
                name="DJINN SIMPSON", gender="male", pronouns="he/him",
                voice_id="v2/en_speaker_5",
                persona="ambitious grant officer evasive about funding.",
                arc_role="pressure source",
            ),
        ],
        beats=[
            Stage1Beat(
                beat_id="b001", speaker="ANNOUNCER",
                intent="open the episode",
                length_target_words=15, emotional_register="welcoming",
                state_before="audit has not begun", state_after="audit window open",
                tension=2,
            ),
            Stage1Beat(
                beat_id="b002", speaker=speaker,
                intent="discover the missing ledger",
                length_target_words=20, emotional_register="tight unease",
                state_before="ledger somewhere", state_after="ledger missing",
                tension=3,
            ),
            Stage1Beat(
                beat_id="b003", speaker="DJINN SIMPSON",
                intent="press for the signature tonight",
                length_target_words=20, emotional_register="cold pressure",
                state_before="signature unsigned", state_after="signature demanded",
                tension=4,
            ),
        ],
        running_facts=["the audit closes at midnight"],
        dramatic_state=_ds(),
    )
    stamp_dialogue_slot_ids(plan)
    return plan


def test_abbreviated_speaker_is_not_wrong_speaker():
    # A weak model writes 'NIA' for the locked 'NIA HIBBERT'; the first-name
    # alias resolves it, so WRONG_SPEAKER must NOT fire.
    verdict = check_constraints(_plan_with_multiword_cast("NIA"))
    assert EditorConstraint.WRONG_SPEAKER not in verdict.failing_constraints


def test_genuine_offcast_speaker_still_wrong_speaker():
    # A name that is neither a cast member nor an unambiguous first/last name
    # must still trip WRONG_SPEAKER (the resolver is conservative, not a bypass).
    verdict = check_constraints(_plan_with_multiword_cast("ZORAK"))
    assert EditorConstraint.WRONG_SPEAKER in verdict.failing_constraints
