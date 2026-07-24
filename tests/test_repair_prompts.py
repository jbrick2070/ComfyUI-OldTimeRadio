"""Behavioral tests for structural structured-output repair prompts."""

from __future__ import annotations

import json

from pydantic import BaseModel, Field, ValidationError

from nodes import _otr_repair_prompts as rp
from nodes._otr_structured_call import PostValidationError


class _SampleSchema(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    score: int = Field(ge=0, le=100)


_ORIGINAL_PROMPT = "produce a title and a score for the episode"
_FAILED_OUTPUT = '{"title": "Out Of Range", "score": 9999}'


def _json_error() -> json.JSONDecodeError:
    try:
        json.loads("this is not json {{{")
    except json.JSONDecodeError as exc:
        return exc
    raise AssertionError("json.loads should have raised")


def _validation_error() -> ValidationError:
    try:
        _SampleSchema.model_validate({"title": "ok", "score": 9999})
    except ValidationError as exc:
        return exc
    raise AssertionError("model_validate should have raised")


def _payload_null_validation_error() -> ValidationError:
    class _Edit(BaseModel):
        line_id: str
        payload: str

    class _Report(BaseModel):
        edits: list[_Edit] = Field(default_factory=list)

    try:
        _Report.model_validate({
            "edits": [{"line_id": "b003", "payload": None}],
        })
    except ValidationError as exc:
        return exc
    raise AssertionError("model_validate should have raised")


def _content(messages) -> str:
    assert isinstance(messages, list)
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    return messages[0]["content"]


def _dispatch(error):
    return rp.make_dispatching_repair_factory()(
        original_prompt=_ORIGINAL_PROMPT,
        failed_output=_FAILED_OUTPUT,
        error=error,
    )


def test_json_syntax_repair_requests_one_valid_object():
    text = _content(rp.json_syntax_repair(
        original_prompt=_ORIGINAL_PROMPT,
        failed_output=_FAILED_OUTPUT,
        error=_json_error(),
    ))
    assert text.startswith("CRITICAL:")
    assert "not valid JSON" in text
    assert "Out Of Range" in text
    assert _ORIGINAL_PROMPT in text


def test_schema_field_repair_preserves_other_fields():
    text = _content(rp.schema_field_repair(
        original_prompt=_ORIGINAL_PROMPT,
        failed_output=_FAILED_OUTPUT,
        error=_validation_error(),
    ))
    assert "valid JSON but failed schema validation" in text
    assert "Fix ONLY the" in text


def test_locked_roster_repair_uses_authoritative_names():
    error = PostValidationError(
        "speaker 'BANDITO' is not in locked cast ['ALICE', 'BEN']"
    )
    text = _content(rp.cast_membership_repair(
        original_prompt=_ORIGINAL_PROMPT,
        failed_output=_FAILED_OUTPUT,
        error=error,
    ))
    assert "cast is FIXED" in text
    assert "ALICE" in text and "BEN" in text


def test_payload_null_repair_can_omit_an_empty_edit_row():
    text = _content(rp.payload_null_repair(
        original_prompt=_ORIGINAL_PROMPT,
        failed_output=_FAILED_OUTPUT,
        error=_payload_null_validation_error(),
    ))
    assert "non-null string" in text
    assert "OMIT" in text


def test_structural_builders_return_one_user_message():
    error = PostValidationError("structural failure")
    for builder in (
        rp.json_syntax_repair,
        rp.schema_field_repair,
        rp.cast_membership_repair,
        rp.payload_null_repair,
    ):
        message = builder(
            original_prompt=_ORIGINAL_PROMPT,
            failed_output=_FAILED_OUTPUT,
            error=error,
        )
        assert len(message) == 1
        assert set(message[0]) == {"role", "content"}


def test_dispatches_json_schema_payload_and_roster_failures():
    assert "not valid JSON" in _content(_dispatch(_json_error()))
    assert "failed schema validation" in _content(
        _dispatch(_validation_error())
    )
    assert "OMIT" in _content(_dispatch(_payload_null_validation_error()))
    roster = PostValidationError(
        "speaker 'BANDITO' is not in locked cast ['ALICE', 'BEN']"
    )
    assert "cast is FIXED" in _content(_dispatch(roster))


def test_other_post_validation_errors_use_generic_structural_repair():
    text = _content(_dispatch(PostValidationError(
        "authored prose was returned with a provider-side receipt"
    )))
    assert "failed schema validation because:" in text


def test_deterministic_roster_repair_short_circuits_model_prompt():
    resolved = _SampleSchema(title="Repaired", score=10)
    calls = []

    def deterministic_repair(failed_output, error):
        calls.append((failed_output, error))
        return resolved

    factory = rp.make_dispatching_repair_factory(
        deterministic_repair=deterministic_repair,
    )
    result = factory(
        original_prompt=_ORIGINAL_PROMPT,
        failed_output=_FAILED_OUTPUT,
        error=PostValidationError(
            "speaker 'ALCIE' is not in locked cast ['ALICE']"
        ),
    )
    assert result is resolved
    assert len(calls) == 1


def test_unresolved_roster_membership_gets_typed_prompt():
    factory = rp.make_dispatching_repair_factory(
        deterministic_repair=lambda _output, _error: None,
    )
    result = factory(
        original_prompt=_ORIGINAL_PROMPT,
        failed_output=_FAILED_OUTPUT,
        error=PostValidationError(
            "speaker 'XQ' is not in locked cast ['ALICE', 'BEN']"
        ),
    )
    assert "cast is FIXED" in _content(result)
