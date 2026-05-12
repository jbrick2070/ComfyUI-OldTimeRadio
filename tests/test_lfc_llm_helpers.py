"""tests/test_lfc_llm_helpers.py

LFC sprint Commit 6 -- regression coverage for the two-step gen +
JSON repair helpers used by Phase 4 / 5 / 6.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import _otr_lfc_llm_helpers as _LFC_LLM  # noqa: E402


# ---------------------------------------------------------------------------
# parse_with_repair -- regex fast-path
# ---------------------------------------------------------------------------


class _SimpleSchema(BaseModel):
    a: int
    b: str = ""


class TestParseWithRepairRegexFastPath:
    def test_clean_json_passes_through(self):
        result = _LFC_LLM.parse_with_repair(
            '{"a": 1, "b": "hello"}', _SimpleSchema,
        )
        assert result is not None
        assert result.a == 1
        assert result.b == "hello"

    def test_markdown_fence_stripped(self):
        wrapped = '```json\n{"a": 7, "b": "fenced"}\n```'
        result = _LFC_LLM.parse_with_repair(wrapped, _SimpleSchema)
        assert result is not None
        assert result.a == 7
        assert result.b == "fenced"

    def test_trailing_prose_stripped(self):
        wrapped = '{"a": 3, "b": "x"} Hope that helps!'
        result = _LFC_LLM.parse_with_repair(wrapped, _SimpleSchema)
        assert result is not None
        assert result.a == 3

    def test_trailing_comma_in_list_fixed(self):
        # Pydantic model with a list field.
        class _Listed(BaseModel):
            items: list[int]

        result = _LFC_LLM.parse_with_repair(
            '{"items": [1, 2, 3,]}', _Listed,
        )
        assert result is not None
        assert result.items == [1, 2, 3]

    def test_trailing_comma_in_object_fixed(self):
        result = _LFC_LLM.parse_with_repair(
            '{"a": 1, "b": "x",}', _SimpleSchema,
        )
        assert result is not None

    def test_smart_quotes_replaced(self):
        # Smart-quoted JSON is invalid; the repair regex normalises.
        smart = '{“a”: 1, “b”: “quoted”}'
        result = _LFC_LLM.parse_with_repair(smart, _SimpleSchema)
        assert result is not None
        assert result.a == 1
        assert result.b == "quoted"

    def test_empty_input_returns_none(self):
        assert _LFC_LLM.parse_with_repair("", _SimpleSchema) is None
        assert _LFC_LLM.parse_with_repair("   ", _SimpleSchema) is None

    def test_unsalvageable_input_returns_none(self):
        # No JSON at all -- regex fast path can't help.
        assert _LFC_LLM.parse_with_repair(
            "this is not JSON",
            _SimpleSchema,
            generate_fn=None,  # disable LLM repair
        ) is None

    def test_combined_repair_path(self):
        # Markdown fence + trailing comma + smart quotes + trailing prose.
        nasty = (
            "```json\n"
            '{“a”: 5, “b”: “combo”,}\n'
            "```\n"
            "Let me know if you need more!"
        )
        result = _LFC_LLM.parse_with_repair(nasty, _SimpleSchema)
        assert result is not None
        assert result.a == 5
        assert result.b == "combo"


# ---------------------------------------------------------------------------
# parse_with_repair -- LLM repair loop
# ---------------------------------------------------------------------------


class TestParseWithRepairLLMLoop:
    def test_llm_repair_recovers_after_validation_error(self):
        # First raw output is unparseable; the LLM repair returns
        # valid JSON.
        repaired_once = {"i": 0}

        def repair_fn(messages, *, temperature, max_new_tokens):
            repaired_once["i"] += 1
            return '{"a": 99, "b": "after-repair"}'

        result = _LFC_LLM.parse_with_repair(
            'not even close to JSON',
            _SimpleSchema,
            generate_fn=repair_fn,
            max_attempts=2,
        )
        assert result is not None
        assert result.a == 99
        assert repaired_once["i"] == 1

    def test_llm_repair_exhausted_returns_none(self):
        # Repair fn keeps returning garbage.
        def keep_breaking(messages, *, temperature, max_new_tokens):
            return "still not JSON"

        result = _LFC_LLM.parse_with_repair(
            "garbage in",
            _SimpleSchema,
            generate_fn=keep_breaking,
            max_attempts=2,
        )
        assert result is None

    def test_llm_repair_disabled_when_fn_is_none(self):
        # No generate_fn provided -- repair loop short-circuits to
        # None on validation failure.
        result = _LFC_LLM.parse_with_repair(
            '{"a": "not_an_int"}',
            _SimpleSchema,
            generate_fn=None,
        )
        assert result is None

    def test_llm_repair_exception_returns_none(self):
        def explode(messages, *, temperature, max_new_tokens):
            raise RuntimeError("LLM crashed")

        result = _LFC_LLM.parse_with_repair(
            "garbage",
            _SimpleSchema,
            generate_fn=explode,
            max_attempts=3,
        )
        assert result is None


# ---------------------------------------------------------------------------
# two_step_generate
# ---------------------------------------------------------------------------


class TestTwoStepGenerate:
    def test_happy_path_uses_both_fns(self):
        calls = {"reasoning": 0, "formatter": 0}

        def reasoning(messages, *, temperature, max_new_tokens):
            calls["reasoning"] += 1
            return (
                "The scene needs tightening. Line l003 should be "
                "rewritten to remove the visual stage direction."
            )

        def formatter(messages, *, temperature, max_new_tokens):
            calls["formatter"] += 1
            return (
                '{"edits": [{"line_id": "l003", "original": "x", '
                '"revised": "y", "reason": "remove visuals"}], '
                '"rationale": "tightening"}'
            )

        result = _LFC_LLM.two_step_generate(
            reasoning, formatter,
            reasoning_prompt="Evaluate this scene...",
            formatter_prompt_template=(
                "Translate to JSON.\n\nReasoning:\n{{REASONING}}"
            ),
            schema=_LFC_LLM.ReviewerOutput,
        )
        assert result is not None
        assert len(result.edits) == 1
        assert result.edits[0].line_id == "l003"
        assert calls["reasoning"] == 1
        assert calls["formatter"] == 1

    def test_formatter_none_reuses_reasoning_fn(self):
        replies = [
            "scene needs tightening; edit l003",
            '{"edits": [], "rationale": "ok"}',
        ]
        state = {"i": 0}

        def both(messages, *, temperature, max_new_tokens):
            i = state["i"]
            state["i"] += 1
            return replies[i] if i < len(replies) else replies[-1]

        result = _LFC_LLM.two_step_generate(
            both, None,
            reasoning_prompt="...",
            formatter_prompt_template="JSON:\n{{REASONING}}",
            schema=_LFC_LLM.ReviewerOutput,
        )
        assert result is not None
        assert state["i"] == 2

    def test_missing_marker_returns_none(self):
        def fn(messages, *, temperature, max_new_tokens):
            return "x"

        result = _LFC_LLM.two_step_generate(
            fn, fn,
            reasoning_prompt="...",
            formatter_prompt_template="no marker here",
            schema=_LFC_LLM.ReviewerOutput,
        )
        assert result is None

    def test_reasoning_exception_returns_none(self):
        def reasoning(messages, *, temperature, max_new_tokens):
            raise RuntimeError("OOM")

        def formatter(messages, *, temperature, max_new_tokens):
            raise AssertionError("formatter must not fire if reasoning crashed")

        result = _LFC_LLM.two_step_generate(
            reasoning, formatter,
            reasoning_prompt="...",
            formatter_prompt_template="x {{REASONING}}",
            schema=_LFC_LLM.ReviewerOutput,
        )
        assert result is None

    def test_empty_reasoning_returns_none(self):
        def reasoning(messages, *, temperature, max_new_tokens):
            return ""

        def formatter(messages, *, temperature, max_new_tokens):
            raise AssertionError("formatter must not fire on empty reasoning")

        result = _LFC_LLM.two_step_generate(
            reasoning, formatter,
            reasoning_prompt="...",
            formatter_prompt_template="x {{REASONING}}",
            schema=_LFC_LLM.ReviewerOutput,
        )
        assert result is None

    def test_formatter_exception_returns_none(self):
        def reasoning(messages, *, temperature, max_new_tokens):
            return "long prose reasoning"

        def formatter(messages, *, temperature, max_new_tokens):
            raise RuntimeError("formatter died")

        result = _LFC_LLM.two_step_generate(
            reasoning, formatter,
            reasoning_prompt="...",
            formatter_prompt_template="x {{REASONING}}",
            schema=_LFC_LLM.ReviewerOutput,
        )
        assert result is None


# ---------------------------------------------------------------------------
# Schema shapes
# ---------------------------------------------------------------------------


class TestSchemaShapes:
    def test_reviewer_edit_required_fields(self):
        edit = _LFC_LLM.ReviewerEdit(
            line_id="l001", original="x", revised="y", reason="why",
        )
        assert edit.line_id == "l001"

    def test_reviewer_output_default_empty(self):
        out = _LFC_LLM.ReviewerOutput()
        assert out.edits == []
        assert out.rationale == ""

    def test_editor_note_scope_default(self):
        note = _LFC_LLM.EditorNote(note="trim scene 3")
        assert note.scope == "episode"
        assert note.note == "trim scene 3"
        assert note.suggested_edits == []

    def test_editor_notes_output_default(self):
        out = _LFC_LLM.EditorNotesOutput()
        assert out.notes == []
