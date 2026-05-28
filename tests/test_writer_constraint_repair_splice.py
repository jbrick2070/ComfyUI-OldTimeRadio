"""Sprint 5.5 (2026-05-28) -- Writer revision turn renders the
derive_repair_prompt block when use_constraint_editor=True.

Pinned invariants:
  * Legacy path (use_constraint_editor=False) renders the
    EDITOR NOTES block via _format_editor_notes_block exactly as
    before -- byte-identical for back-compat.
  * Constraint-editor path renders a REVISE block with per-code
    repair headers via derive_repair_prompt (Sprint 5.2 helper).
  * Constraint-editor path uses the Sprint 5.5 revision directive
    ("Do NOT rewrite passing passages").
  * _format_constraint_repair_block reconstructs an
    EditorConstraintVerdict from the EditorVerdict-shape adapter
    that _call_constraint_editor produces and re-emits the
    derive_repair_prompt output.
"""
from __future__ import annotations

import inspect

from nodes._otr_director_brief import DirectorBrief
from nodes._otr_editor_pass import EditorVerdict
from nodes._otr_editor_constraints import EditorConstraint
from nodes._otr_story_room import (
    _format_constraint_repair_block,
    _format_editor_notes_block,
    build_writer_user_prompt,
)


def _brief() -> DirectorBrief:
    return DirectorBrief(
        news_premise="An audit team races to verify a falsified ledger.",
        dramatic_question="Will Maeve sign in time?",
        opposed_desires="hide the ledger vs get the signature",
        arc_shape="discover -> press -> sign or refuse",
        audience_feel="tight forensic tension",
        forbidden_drift=[],
        raw_brief="raw brief text",
    )


# ---------------------------------------------------------------------------
# build_writer_user_prompt signature
# ---------------------------------------------------------------------------


def test_build_writer_user_prompt_signature_carries_kwarg():
    sig = inspect.signature(build_writer_user_prompt)
    assert "use_constraint_editor" in sig.parameters
    assert sig.parameters["use_constraint_editor"].default is False


# ---------------------------------------------------------------------------
# Legacy path: byte-identical EDITOR NOTES block + revision directive
# ---------------------------------------------------------------------------


def test_legacy_path_uses_editor_notes_block():
    """use_constraint_editor=False MUST render the taste-rubric notes
    block and the original revision directive verbatim."""
    legacy_verdict = EditorVerdict(
        pass_decision=False,
        failing_axes=["premise_clarity", "resolution"],
        per_axis_notes={
            "premise_clarity": "the central question is not posed clearly",
            "resolution": "the ending does not pay off the setup",
        },
        overall_note="tighten the close to land the question",
        cycle=1,
    )
    prompt = build_writer_user_prompt(
        brief=_brief(),
        cast=[{"name": "ALICE"}, {"name": "BOB"}],
        news_seed="seed",
        last_editor_verdict=legacy_verdict,
        last_draft="ALICE: open. BOB: close.",
        cycle=1,
        use_constraint_editor=False,
    )
    assert "EDITOR NOTES:" in prompt
    assert "premise_clarity" in prompt
    assert "resolution" in prompt
    assert "every failing axis" in prompt   # legacy directive
    assert "REVISE" not in prompt           # constraint block absent


# ---------------------------------------------------------------------------
# Constraint-editor path: derive_repair_prompt + Sprint 5.5 directive
# ---------------------------------------------------------------------------


def test_constraint_editor_path_uses_repair_block():
    """use_constraint_editor=True MUST render the derive_repair_prompt
    block (REVISE header + per-code lines) and the constraint
    revision directive."""
    # Mirrors the EditorVerdict-shape _call_constraint_editor emits.
    constraint_shaped = EditorVerdict(
        pass_decision=False,
        failing_axes=[
            EditorConstraint.WRONG_SPEAKER,
            EditorConstraint.MISSING_COSTLY_CHOICE,
        ],
        per_axis_notes={},   # constraint editor has no taste rubric
        overall_note="beat b003 attributed to MAEVE; cast is ALICE/BOB.",
        cycle=0,
    )
    prompt = build_writer_user_prompt(
        brief=_brief(),
        cast=[{"name": "ALICE"}, {"name": "BOB"}],
        news_seed="seed",
        last_editor_verdict=constraint_shaped,
        last_draft="ALICE: open. MAEVE: close.",
        cycle=1,
        use_constraint_editor=True,
    )
    # Sprint 5.2 REVISE block + the named constraint codes.
    assert "REVISE:" in prompt
    assert "WRONG_SPEAKER" in prompt
    assert "MISSING_COSTLY_CHOICE" in prompt
    assert "Editor's note:" in prompt
    assert "MAEVE" in prompt
    # Sprint 5.5 revision directive (no taste verb).
    assert "Do NOT rewrite passing passages" in prompt
    assert "every failing axis" not in prompt   # legacy directive absent


def test_constraint_editor_path_empty_verdict_falls_back():
    """A pass-decision verdict in the revision call (shouldn't happen
    in production but defensive) renders a minimal EDITOR NOTES
    fallback so the Writer still has signal."""
    passing_shaped = EditorVerdict(
        pass_decision=True,
        failing_axes=[],
        per_axis_notes={},
        overall_note="",
        cycle=0,
    )
    prompt = build_writer_user_prompt(
        brief=_brief(),
        cast=[{"name": "ALICE"}],
        news_seed="seed",
        last_editor_verdict=passing_shaped,
        last_draft="placeholder",
        cycle=1,
        use_constraint_editor=True,
    )
    assert "EDITOR NOTES:" in prompt
    assert "constraint editor passed" in prompt


# ---------------------------------------------------------------------------
# _format_constraint_repair_block conversion fidelity
# ---------------------------------------------------------------------------


def test_format_constraint_repair_block_round_trips_codes_and_note():
    v = EditorVerdict(
        pass_decision=False,
        failing_axes=[EditorConstraint.FORMAT_FAILURE],
        per_axis_notes={},
        overall_note="bracket [SFX:] inside intent on beat b002",
        cycle=2,
    )
    block = _format_constraint_repair_block(v)
    assert "REVISE:" in block
    assert "FORMAT_FAILURE" in block
    assert "bracket [SFX:]" in block


def test_format_constraint_repair_block_passing_verdict_returns_empty():
    v = EditorVerdict(
        pass_decision=True,
        failing_axes=[],
        per_axis_notes={},
        overall_note="",
        cycle=0,
    )
    assert _format_constraint_repair_block(v) == ""


# ---------------------------------------------------------------------------
# First-draft path (no verdict) unchanged regardless of flag
# ---------------------------------------------------------------------------


def test_first_draft_no_verdict_unchanged_under_either_flag():
    prompt_legacy = build_writer_user_prompt(
        brief=_brief(),
        cast=[{"name": "ALICE"}],
        news_seed="seed",
        last_editor_verdict=None,
        last_draft=None,
        cycle=0,
        use_constraint_editor=False,
    )
    prompt_constraint = build_writer_user_prompt(
        brief=_brief(),
        cast=[{"name": "ALICE"}],
        news_seed="seed",
        last_editor_verdict=None,
        last_draft=None,
        cycle=0,
        use_constraint_editor=True,
    )
    # No revision context -> identical first-draft prompt either way.
    assert prompt_legacy == prompt_constraint
    assert "REVISE" not in prompt_legacy
    assert "EDITOR NOTES" not in prompt_legacy
