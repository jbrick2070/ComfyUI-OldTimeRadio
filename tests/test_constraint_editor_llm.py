"""Sprint 5.3 (2026-05-28) -- LLM-side constraint editor surface.

The Sprint 5 EDITOR_CONSTRAINTS_SYSTEM_PROMPT + pure-Python
check_constraints land in Sprint 5; the Sprint 5.1 wire stamped
diagnostic verdicts on every ledger. This sprint ships the LLM-
side surface mirroring _otr_editor_pass.run_editor so the
forthcoming Sprint 5.4 swap in run_story_room is a thin wrap:
  - EditorConstraintVerdictSchema (pydantic, Literal-enum
    failing_constraints) -- bindable into the existing constrained-
    decode pipeline.
  - build_constraint_editor_prompt(brief, draft, *, cast_names,
    cycle) -> messages.
  - run_constraint_editor(...) -> EditorConstraintVerdict; mirrors
    run_editor's contract (retry-twice ladder, raise on
    exhaustion).
"""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from nodes._otr_editor_constraints import (
    ConstraintEditorCallFailedError,
    EditorConstraint,
    EditorConstraintVerdictSchema,
    build_constraint_editor_prompt,
    run_constraint_editor,
)


# ---------------------------------------------------------------------------
# EditorConstraintVerdictSchema -- the Literal-enum failing_constraints
# ---------------------------------------------------------------------------


def test_schema_accepts_well_formed_pass_verdict():
    v = EditorConstraintVerdictSchema(
        pass_decision=True,
        failing_constraints=[],
        repair_note="",
    )
    assert v.pass_decision is True
    assert v.failing_constraints == []


def test_schema_accepts_full_failing_constraint_set():
    v = EditorConstraintVerdictSchema(
        pass_decision=False,
        failing_constraints=list(EditorConstraint.ALL),
        repair_note="placeholder note",
    )
    assert len(v.failing_constraints) == 5


def test_schema_rejects_off_enum_constraint():
    """Literal union must reject codes outside the canonical 5."""
    with pytest.raises(ValidationError):
        EditorConstraintVerdictSchema(
            pass_decision=False,
            failing_constraints=["EXPERIMENTAL_NEW"],
            repair_note="x",
        )


def test_schema_rejects_overlong_repair_note():
    with pytest.raises(ValidationError):
        EditorConstraintVerdictSchema(
            pass_decision=False,
            failing_constraints=[EditorConstraint.FORMAT_FAILURE],
            repair_note="x" * 1000,
        )


# ---------------------------------------------------------------------------
# build_constraint_editor_prompt
# ---------------------------------------------------------------------------


def _brief_dict() -> dict:
    return {
        "news_premise": "An audit team races to verify a falsified ledger.",
        "dramatic_question": "Will Maeve sign in time?",
        "opposed_desires": "hide the falsified ledger vs get the signature",
        "arc_shape": "discover -> press -> sign or refuse",
        "audience_feel": "tight forensic tension",
        "forbidden_drift": [],
        "raw_brief": "raw brief text here",
    }


def test_prompt_carries_system_and_user_messages():
    msgs = build_constraint_editor_prompt(
        _brief_dict(),
        writer_draft="ALICE: Sign it.\nBOB: I won't.",
        cycle=0,
    )
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"


def test_prompt_renders_cast_block_when_supplied():
    msgs = build_constraint_editor_prompt(
        _brief_dict(),
        writer_draft="placeholder draft",
        cast_names=["ALICE", "BOB"],
    )
    user_text = msgs[1]["content"]
    assert "LOCKED CAST" in user_text
    assert "ALICE" in user_text
    assert "BOB" in user_text


def test_prompt_omits_cast_block_when_none():
    msgs = build_constraint_editor_prompt(
        _brief_dict(),
        writer_draft="placeholder draft",
    )
    assert "LOCKED CAST" not in msgs[1]["content"]


def test_prompt_carries_cycle_number():
    msgs = build_constraint_editor_prompt(
        _brief_dict(),
        writer_draft="placeholder draft",
        cycle=2,
    )
    assert "cycle 2" in msgs[1]["content"]


def test_prompt_renders_director_brief_fields():
    msgs = build_constraint_editor_prompt(
        _brief_dict(),
        writer_draft="placeholder draft",
    )
    user_text = msgs[1]["content"]
    assert "audit team races to verify" in user_text
    assert "Will Maeve sign" in user_text


# ---------------------------------------------------------------------------
# run_constraint_editor -- happy + retry + exhaustion paths
# ---------------------------------------------------------------------------


def _ok_payload(
    pass_decision: bool = True,
    failing: list | None = None,
    note: str = "",
) -> str:
    return json.dumps({
        "pass_decision": pass_decision,
        "failing_constraints": failing or [],
        "repair_note": note,
    })


def test_run_constraint_editor_happy_pass():
    calls = {"n": 0}

    def gen_fn(messages, *, temperature, max_new_tokens):
        calls["n"] += 1
        return _ok_payload(pass_decision=True)

    verdict = run_constraint_editor(
        _brief_dict(),
        "ALICE: Sign it.\nBOB: I won't.",
        generate_fn=gen_fn,
        cast_names=["ALICE", "BOB"],
        cycle=0,
    )
    assert verdict.pass_decision is True
    assert verdict.failing_constraints == []
    assert verdict.cycle == 0
    assert calls["n"] == 1  # passed on the first attempt


def test_run_constraint_editor_happy_fail():
    def gen_fn(messages, *, temperature, max_new_tokens):
        return _ok_payload(
            pass_decision=False,
            failing=[EditorConstraint.WRONG_SPEAKER],
            note="beat b003 says MAEVE but cast is ALICE/BOB.",
        )

    verdict = run_constraint_editor(
        _brief_dict(),
        "ALICE: Sign it.\nMAEVE: I won't.",
        generate_fn=gen_fn,
        cast_names=["ALICE", "BOB"],
        cycle=1,
    )
    assert verdict.pass_decision is False
    assert verdict.failing_constraints == [EditorConstraint.WRONG_SPEAKER]
    assert "MAEVE" in verdict.repair_note
    assert verdict.cycle == 1


def test_run_constraint_editor_retries_on_invalid_json():
    """First attempt returns garbage, second returns valid JSON."""
    raw_outputs = ["not json at all {", _ok_payload(pass_decision=True)]
    calls = {"n": 0}

    def gen_fn(messages, *, temperature, max_new_tokens):
        idx = calls["n"]
        calls["n"] += 1
        return raw_outputs[idx]

    verdict = run_constraint_editor(
        _brief_dict(),
        "placeholder draft",
        generate_fn=gen_fn,
        cast_names=["ALICE"],
    )
    assert verdict.pass_decision is True
    assert calls["n"] == 2


def test_run_constraint_editor_raises_on_retry_exhaustion():
    """Both attempts return garbage -> ConstraintEditorCallFailedError."""
    def gen_fn(messages, *, temperature, max_new_tokens):
        return "still not json"

    with pytest.raises(ConstraintEditorCallFailedError) as ctx:
        run_constraint_editor(
            _brief_dict(),
            "placeholder draft",
            generate_fn=gen_fn,
            cast_names=["ALICE"],
        )
    assert ctx.value.attempts == 2


def test_run_constraint_editor_handles_generate_fn_raise():
    """generate_fn itself raises (e.g. LLM timeout) on attempt 1;
    attempt 2 succeeds."""
    raw_outputs = [RuntimeError("LLM timeout"), _ok_payload(True)]
    calls = {"n": 0}

    def gen_fn(messages, *, temperature, max_new_tokens):
        idx = calls["n"]
        calls["n"] += 1
        v = raw_outputs[idx]
        if isinstance(v, Exception):
            raise v
        return v

    verdict = run_constraint_editor(
        _brief_dict(),
        "placeholder draft",
        generate_fn=gen_fn,
        cast_names=["ALICE"],
    )
    assert verdict.pass_decision is True
    assert calls["n"] == 2


def test_run_constraint_editor_uses_lower_temperature_on_retry():
    temperatures: list[float] = []

    def gen_fn(messages, *, temperature, max_new_tokens):
        temperatures.append(temperature)
        if len(temperatures) == 1:
            return "garbage"
        return _ok_payload(True)

    run_constraint_editor(
        _brief_dict(),
        "placeholder draft",
        generate_fn=gen_fn,
    )
    assert len(temperatures) == 2
    # Structural retry uses a strictly lower temperature than base.
    assert temperatures[1] < temperatures[0]
