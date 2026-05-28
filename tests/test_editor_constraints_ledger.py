"""Sprint 5.1 (2026-05-28) -- ledger-shaped wire for check_constraints.

OTR_LedgerScriptWriter stamps `meta.editor_constraints` after
DramaticState is built. The wire reads ledger.cast / ledger.lines /
ledger.meta.dramatic_state, reconstructs a Stage1Plan-shaped view,
and runs the same check_constraints the Sprint 5 surface ships.

Pinned invariants:
  * Clean ledger -> pass_decision=True.
  * Ledger with a costly_choice_beat that names a non-existent slot
    on the ledger lines -> MISSING_COSTLY_CHOICE fires.
  * Empty ledger -> the helper does not raise.
"""
from __future__ import annotations

from nodes._otr_editor_constraints import (
    EditorConstraint,
    check_constraints_from_ledger,
)


def _ledger(*, with_state: bool = True, costly_slot: str = "d003",
            lines=None, cast=None) -> dict:
    return {
        "cast": cast if cast is not None else [
            {"char_id": "alice", "name": "ALICE"},
            {"char_id": "bob", "name": "BOB"},
        ],
        "lines": lines if lines is not None else [
            {"line_id": "l001", "beat_id": "b001", "char_id": "announcer",
             "speaker_role": "announcer", "text": "Open.",
             "dialogue_slot_id": "d001"},
            {"line_id": "l002", "beat_id": "b002", "char_id": "alice",
             "speaker_role": "character", "text": "Find it.",
             "dialogue_slot_id": "d002"},
            {"line_id": "l003", "beat_id": "b003", "char_id": "alice",
             "speaker_role": "character", "text": "Sign it.",
             "dialogue_slot_id": "d003"},
            {"line_id": "l004", "beat_id": "b004", "char_id": "announcer",
             "speaker_role": "announcer", "text": "Close.",
             "dialogue_slot_id": "d004"},
        ],
        "meta": {
            "dramatic_state": (
                {
                    "dramatic_question": "Will Maeve sign in time?",
                    "character_a_wants": "hide the falsified ledger",
                    "character_b_wants": "get the signature tonight",
                    "costly_choice_beat": costly_slot,
                    "ending_change": "the record becomes public",
                }
                if with_state else None
            ),
        },
    }


def test_clean_ledger_passes():
    v = check_constraints_from_ledger(_ledger())
    assert v.pass_decision is True
    assert v.failing_constraints == []


def test_missing_costly_choice_target_fires():
    """costly_choice_beat names a slot the ledger does not carry."""
    v = check_constraints_from_ledger(_ledger(costly_slot="d099"))
    assert EditorConstraint.MISSING_COSTLY_CHOICE in v.failing_constraints
    assert v.pass_decision is False


def test_ledger_without_dramatic_state_skips_state_axes():
    """No dramatic_state -> MISSING_COSTLY_CHOICE does not fire (the
    Sprint 5 checker only claims when ds is present)."""
    v = check_constraints_from_ledger(_ledger(with_state=False))
    assert EditorConstraint.MISSING_COSTLY_CHOICE not in v.failing_constraints


def test_empty_ledger_does_not_raise():
    v = check_constraints_from_ledger({})
    # No cast, no lines, no state -> no constraints to claim against.
    assert v.pass_decision is True


def test_wrong_speaker_fires_on_off_cast_line():
    """A line attributed to a name not in cast (Sprint 1 + 5).

    The default fixture lines all use char_id 'alice' so cutting
    cast to just ALICE wouldn't trip the check; this test replaces
    b003's char_id with 'bob' (not in the cut cast) so the speaker
    resolver falls back to 'BOB' which fails WRONG_SPEAKER.
    """
    led = _ledger()
    led["cast"] = [{"char_id": "alice", "name": "ALICE"}]
    led["lines"] = [
        {"line_id": "l001", "beat_id": "b001", "char_id": "announcer",
         "speaker_role": "announcer", "text": "Open.",
         "dialogue_slot_id": "d001"},
        {"line_id": "l002", "beat_id": "b002", "char_id": "alice",
         "speaker_role": "character", "text": "Find it.",
         "dialogue_slot_id": "d002"},
        {"line_id": "l003", "beat_id": "b003", "char_id": "bob",
         "speaker_role": "character", "text": "Sign it.",
         "dialogue_slot_id": "d003"},
        {"line_id": "l004", "beat_id": "b004", "char_id": "announcer",
         "speaker_role": "announcer", "text": "Close.",
         "dialogue_slot_id": "d004"},
    ]
    v = check_constraints_from_ledger(led)
    assert EditorConstraint.WRONG_SPEAKER in v.failing_constraints
