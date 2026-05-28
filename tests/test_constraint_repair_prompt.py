"""Sprint 5.2 (2026-05-28) -- derive_repair_prompt helper.

Bridges an EditorConstraintVerdict into a Writer-side revision
prompt block. The live constraint-editor swap (a later sprint that
touches run_story_room) splices the result into one targeted
Writer revision turn before shipping or failing loud.

Pinned invariants:
  * MAX_REPAIR_TURNS == 1 (the Sprint 5 cap).
  * Passing verdict -> empty string (no repair block to splice).
  * Failing verdict -> a single REVISE block listing each
    constraint with its repair header + the verdict's repair_note.
  * The block names each failing constraint by its enum code so
    the operator can grep ledger logs for specific defect types.
"""
from __future__ import annotations

from nodes._otr_editor_constraints import (
    EditorConstraint,
    EditorConstraintVerdict,
    MAX_REPAIR_TURNS,
    derive_repair_prompt,
)


def test_max_repair_turns_is_one():
    """Sprint 5 plan: one targeted repair turn, no open-ended loop."""
    assert MAX_REPAIR_TURNS == 1


def test_passing_verdict_returns_empty_string():
    """Splice-into-prompt safety: passing verdicts produce no
    repair block, so callers can unconditionally splice."""
    v = EditorConstraintVerdict(
        pass_decision=True, failing_constraints=[],
        repair_note="",
    )
    assert derive_repair_prompt(v) == ""


def test_failing_verdict_renders_revise_block():
    v = EditorConstraintVerdict(
        pass_decision=False,
        failing_constraints=[EditorConstraint.WRONG_SPEAKER],
        repair_note="beat b003 attributed to MAEVE; cast is ALICE/BOB.",
    )
    out = derive_repair_prompt(v)
    assert out.startswith("REVISE")
    assert "WRONG_SPEAKER" in out
    assert "beat b003 attributed to MAEVE" in out


def test_failing_verdict_lists_each_constraint():
    v = EditorConstraintVerdict(
        pass_decision=False,
        failing_constraints=[
            EditorConstraint.WRONG_SPEAKER,
            EditorConstraint.MISSING_COSTLY_CHOICE,
            EditorConstraint.NO_FINAL_THIRD_TURN,
        ],
        repair_note="placeholder",
    )
    out = derive_repair_prompt(v)
    assert "WRONG_SPEAKER" in out
    assert "MISSING_COSTLY_CHOICE" in out
    assert "NO_FINAL_THIRD_TURN" in out


def test_failing_verdict_includes_repair_note_when_present():
    v = EditorConstraintVerdict(
        pass_decision=False,
        failing_constraints=[EditorConstraint.FORMAT_FAILURE],
        repair_note="bracket [SFX:] inside intent on beat b002",
    )
    out = derive_repair_prompt(v)
    assert "Editor's note:" in out
    assert "bracket [SFX:]" in out


def test_failing_verdict_no_repair_note_skips_note_block():
    v = EditorConstraintVerdict(
        pass_decision=False,
        failing_constraints=[EditorConstraint.PHANTOM_CHARACTER],
        repair_note="",
    )
    out = derive_repair_prompt(v)
    assert "PHANTOM_CHARACTER" in out
    assert "Editor's note:" not in out


def test_unknown_constraint_string_renders_as_bare_code():
    """An unknown constraint string (e.g. an experimental code from
    a future variant) renders as the bare code without a header --
    forensic-friendly, no crash."""
    v = EditorConstraintVerdict(
        pass_decision=False,
        failing_constraints=["EXPERIMENTAL_NEW_CONSTRAINT"],
        repair_note="",
    )
    out = derive_repair_prompt(v)
    assert "EXPERIMENTAL_NEW_CONSTRAINT" in out
