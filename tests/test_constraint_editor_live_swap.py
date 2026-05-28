"""Sprint 5.4 (2026-05-28) -- live constraint-editor swap in
run_story_room + OTR_StoryRoom node.

Pinned invariants:
  * OTR_StoryRoom.INPUT_TYPES exposes use_constraint_editor with
    default False (back-compat).
  * run_story_room takes use_constraint_editor kwarg, default False.
  * When True, run_story_room clamps max_editor_cycles -> 1
    (Sprint 5 cap).
  * The loop swaps _call_editor for _call_constraint_editor when
    the flag is True.
  * _call_constraint_editor adapts an EditorConstraintVerdict into
    an EditorVerdict-shape so the transcript stays back-compat.
  * On ConstraintEditorCallFailedError the loop soft-passes (matches
    the taste editor's failure contract).
"""
from __future__ import annotations

import inspect
from unittest.mock import patch

import pytest

from nodes.OTR_StoryRoom import OTR_StoryRoom
from nodes._otr_director_brief import DirectorBrief
from nodes._otr_editor_constraints import (
    ConstraintEditorCallFailedError,
    EditorConstraint,
    EditorConstraintVerdict,
)
from nodes._otr_editor_pass import EditorVerdict
from nodes._otr_story_room import (
    DEFAULT_MAX_EDITOR_CYCLES,
    _call_constraint_editor,
    run_story_room,
)


# ---------------------------------------------------------------------------
# Node-side widget + run signature
# ---------------------------------------------------------------------------


def test_story_room_node_widget_exists_default_false():
    schema = OTR_StoryRoom.INPUT_TYPES()
    req = schema.get("required", {}) or {}
    assert "use_constraint_editor" in req
    widget = req["use_constraint_editor"]
    assert widget[0] == "BOOLEAN"
    assert widget[1]["default"] is False


def test_story_room_node_run_signature_carries_kwarg():
    sig = inspect.signature(OTR_StoryRoom.run)
    assert "use_constraint_editor" in sig.parameters
    assert sig.parameters["use_constraint_editor"].default is False


def test_run_story_room_signature_carries_kwarg():
    sig = inspect.signature(run_story_room)
    assert "use_constraint_editor" in sig.parameters
    assert sig.parameters["use_constraint_editor"].default is False


# ---------------------------------------------------------------------------
# _call_constraint_editor adapter
# ---------------------------------------------------------------------------


def _brief() -> DirectorBrief:
    return DirectorBrief(
        news_premise="An audit team races to verify a falsified ledger.",
        dramatic_question="Will Maeve sign in time?",
        opposed_desires="hide the falsified ledger vs get the signature",
        arc_shape="discover -> press -> sign or refuse",
        audience_feel="tight forensic tension",
        forbidden_drift=[],
        raw_brief="raw brief text",
    )


def test_call_constraint_editor_returns_editorverdict_shape():
    """Adapter must convert the constraint verdict to an
    EditorVerdict so the transcript + Writer revision turn stay
    back-compat."""
    fake_verdict = EditorConstraintVerdict(
        pass_decision=False,
        failing_constraints=[EditorConstraint.WRONG_SPEAKER],
        repair_note="beat b003 attributed to MAEVE; cast is ALICE/BOB.",
        cycle=0,
    )

    with patch(
        "nodes._otr_editor_constraints.run_constraint_editor",
        return_value=fake_verdict,
    ):
        verdict = _call_constraint_editor(
            brief=_brief(),
            draft="placeholder draft",
            editor_generate_fn=lambda *a, **kw: "",
            cast=[{"name": "ALICE"}, {"name": "BOB"}],
            cycle=0,
        )

    # Adapter mapping
    assert isinstance(verdict, EditorVerdict)
    assert verdict.pass_decision is False
    assert verdict.failing_axes == [EditorConstraint.WRONG_SPEAKER]
    # repair_note -> overall_note so the Writer's _format_editor_notes
    # block picks it up.
    assert "MAEVE" in verdict.overall_note
    assert verdict.per_axis_notes == {}  # no taste rubric
    assert verdict.cycle == 0


def test_call_constraint_editor_softpasses_on_failure():
    """On ConstraintEditorCallFailedError the loop soft-passes the
    Writer's last draft so the episode ships, matching the taste
    editor's failure contract."""

    def fake_run_ce(*args, **kwargs):
        raise ConstraintEditorCallFailedError(
            attempts=2,
            last_raw_output="garbage",
            last_error=RuntimeError("LLM exhausted"),
        )

    with patch(
        "nodes._otr_editor_constraints.run_constraint_editor",
        side_effect=fake_run_ce,
    ):
        verdict = _call_constraint_editor(
            brief=_brief(),
            draft="placeholder draft",
            editor_generate_fn=lambda *a, **kw: "",
            cast=[{"name": "ALICE"}],
            cycle=2,
        )

    assert isinstance(verdict, EditorVerdict)
    assert verdict.pass_decision is True  # soft pass
    assert "ConstraintEditorCallFailedError" in verdict.overall_note
    assert verdict.cycle == 2


def test_call_constraint_editor_pass_path_clean():
    fake_verdict = EditorConstraintVerdict(
        pass_decision=True,
        failing_constraints=[],
        repair_note="",
        cycle=0,
    )

    with patch(
        "nodes._otr_editor_constraints.run_constraint_editor",
        return_value=fake_verdict,
    ):
        verdict = _call_constraint_editor(
            brief=_brief(),
            draft="placeholder draft",
            editor_generate_fn=lambda *a, **kw: "",
            cast=[{"name": "ALICE"}],
            cycle=0,
        )

    assert verdict.pass_decision is True
    assert verdict.failing_axes == []
    assert verdict.overall_note == ""


# ---------------------------------------------------------------------------
# run_story_room: cycle cap clamp + dispatch
# ---------------------------------------------------------------------------


def test_run_story_room_clamps_max_editor_cycles_when_constraint_on(caplog):
    """When use_constraint_editor=True, max_editor_cycles > 1 is
    silently clamped to 1 (Sprint 5 cap). The dispatch is hard to
    exercise without a writer LLM stub; this test focuses on the
    clamp path by asserting the log message at INFO level."""
    import logging

    writer_calls = {"n": 0}

    def fake_writer(messages, *, temperature, max_new_tokens):
        writer_calls["n"] += 1
        if writer_calls["n"] == 1:
            return "ALICE: One spoken line so the writer draft is non-empty."
        return ""  # subsequent calls return empty so the loop has no spare draft

    # Editor is mocked at the import site so the actual LLM call
    # never fires. The constraint editor always passes.
    fake_verdict = EditorConstraintVerdict(
        pass_decision=True, failing_constraints=[], repair_note="",
        cycle=0,
    )

    with patch(
        "nodes._otr_editor_constraints.run_constraint_editor",
        return_value=fake_verdict,
    ), caplog.at_level(logging.INFO, logger="OTR"):
        run_story_room(
            director_brief=_brief(),
            news_seed="seed",
            cast=[{"name": "ALICE"}, {"name": "BOB"}],
            writer_generate_fn=fake_writer,
            editor_generate_fn=lambda *a, **kw: "",
            max_writer_turns=1,
            max_editor_cycles=3,   # would be 3 under taste editor
            max_total_turns=8,
            use_constraint_editor=True,
        )
    # Look for the clamp log.
    msgs = [r.getMessage() for r in caplog.records]
    assert any("clamping max_editor_cycles" in m for m in msgs), (
        f"Expected clamp log in {msgs}"
    )


def test_run_story_room_legacy_path_does_not_clamp(caplog):
    """When use_constraint_editor=False (default), the clamp log
    must not fire and max_editor_cycles=3 is honored."""
    import logging
    from unittest.mock import patch as _patch

    def fake_writer(messages, *, temperature, max_new_tokens):
        return "ALICE: One spoken line."

    fake_verdict = EditorVerdict(
        pass_decision=True, failing_axes=[], per_axis_notes={},
        overall_note="", cycle=0,
    )

    with _patch(
        "nodes._otr_story_room.run_editor",
        return_value=fake_verdict,
    ), caplog.at_level(logging.INFO, logger="OTR"):
        run_story_room(
            director_brief=_brief(),
            news_seed="seed",
            cast=[{"name": "ALICE"}],
            writer_generate_fn=fake_writer,
            editor_generate_fn=lambda *a, **kw: "",
            max_writer_turns=1,
            max_editor_cycles=3,
            max_total_turns=8,
            # use_constraint_editor defaults False.
        )
    msgs = [r.getMessage() for r in caplog.records]
    assert not any("clamping max_editor_cycles" in m for m in msgs)


# ---------------------------------------------------------------------------
# Module-level sanity: default max cycles unchanged
# ---------------------------------------------------------------------------


def test_default_max_editor_cycles_unchanged():
    """Sprint 5.4 must NOT lower the legacy taste editor's default
    cycle count (back-compat for the OFF path)."""
    assert DEFAULT_MAX_EDITOR_CYCLES == 3
