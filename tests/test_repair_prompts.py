"""Sprint 2C: typed repair-prompt factory tests.

The module under test (`nodes/_otr_repair_prompts.py`) is pure -- no
I/O, no GPU, no ComfyUI imports. Every test here runs against plain
exception objects and a tiny pydantic schema.

Coverage map:

  The seven typed builders:
    1. json_syntax_repair    -- CRITICAL "not valid JSON" directive,
       echoes failed output, restates the original instruction.
    2. schema_field_repair   -- CRITICAL "valid JSON but failed schema
       validation" directive.
    3. cast_membership_repair-- CRITICAL "not in the locked cast" /
       "cast is FIXED" directive.
    4. too_many_words_repair -- CRITICAL "over the length budget".
    5. narration_leak_repair -- CRITICAL "leaked narration".
    6. forbidden_name_repair -- CRITICAL "must not name anyone".
    7. payload_null_repair   -- CRITICAL "payload field set to null,
       OMIT the entire edit row" (Sprint 7C / BUG-LOCAL-275).
    8. every builder returns a one-element `user` messages list.

  make_dispatching_repair_factory (error -> builder routing):
    9.  json.JSONDecodeError      -> json_syntax_repair.
    10. pydantic.ValidationError  -> schema_field_repair (default).
    11. pydantic.ValidationError on null `payload` field
                                  -> payload_null_repair (Sprint 7C).
    12. PostValidationError "...locked cast..." -> cast_membership LLM
        prompt when no deterministic callback is supplied.
    13. PostValidationError "named_character" -> forbidden_name_repair.
    14. PostValidationError "dialogue_verb" -> narration_leak_repair.
    15. PostValidationError "plot_verb"     -> narration_leak_repair.
    16. PostValidationError "too_long"      -> too_many_words_repair.
    17. an unrecognised content failure (news V1) -> the generic
        default_repair_prompt_factory.

  Deterministic cast-membership short-circuit:
    18. a deterministic_repair callback that resolves the phantom ->
        the dispatcher returns the callback's pydantic instance
        directly (no messages list built).
    19. a deterministic_repair callback that returns None -> the
        dispatcher falls through to the cast_membership LLM prompt.
    20. the deterministic callback is consulted ONLY for the locked-
        cast branch, never for a JSON / schema / other-content error.
"""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel, Field, ValidationError

from nodes import _otr_repair_prompts as rp
from nodes._otr_structured_call import PostValidationError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _SampleSchema(BaseModel):
    """Minimal pydantic schema for the deterministic-repair tests."""

    title: str = Field(min_length=1, max_length=120)
    score: int = Field(ge=0, le=100)


_ORIGINAL_PROMPT = "produce a title and a score for the episode"
_FAILED_OUTPUT = '{"title": "Out Of Range", "score": 9999}'


def _json_error() -> json.JSONDecodeError:
    """A real json.JSONDecodeError, captured the way the ladder sees it."""
    try:
        json.loads("this is not json {{{")
    except json.JSONDecodeError as exc:
        return exc
    raise AssertionError("json.loads should have raised")  # pragma: no cover


def _validation_error() -> ValidationError:
    """A real pydantic ValidationError for _SampleSchema."""
    try:
        _SampleSchema.model_validate({"title": "ok", "score": 9999})
    except ValidationError as exc:
        return exc
    raise AssertionError("model_validate should have raised")  # pragma: no cover


def _payload_null_validation_error() -> ValidationError:
    """A real pydantic ValidationError mirroring the BUG-LOCAL-275 trace.

    Reproduces the exact failure the Script Doctor edits pass surfaced
    in the 2026-05-25 live run: a list of `ReviewerEdit`-shaped rows
    where one row's `payload` arrived as `None`. Validating against the
    real `ScriptDoctorReport` model would couple this test file to the
    reviewer module; a local schema that uses the same `payload: str`
    field shape produces a structurally identical error.
    """

    class _Edit(BaseModel):
        line_id: str
        action: str
        payload: str
        rationale: str = ""

    class _Report(BaseModel):
        edits: list[_Edit] = Field(default_factory=list)

    try:
        _Report.model_validate({
            "edits": [
                {"line_id": "b001", "action": "annotate",
                 "payload": "ok", "rationale": ""},
                {"line_id": "b002", "action": "skip",
                 "payload": "ok", "rationale": ""},
                # The pathological row -- payload is None.
                {"line_id": "b003", "action": "rewrite",
                 "payload": None, "rationale": "no replacement text"},
            ],
        })
    except ValidationError as exc:
        return exc
    raise AssertionError("model_validate should have raised")  # pragma: no cover


def _content(messages) -> str:
    """Pull the single user-message content string out of a builder result."""
    assert isinstance(messages, list), messages
    assert len(messages) == 1, messages
    assert messages[0]["role"] == "user"
    return messages[0]["content"]


# ---------------------------------------------------------------------------
# 1-7. The six typed builders
# ---------------------------------------------------------------------------


def test_json_syntax_repair_directive_and_shape():
    messages = rp.json_syntax_repair(
        original_prompt=_ORIGINAL_PROMPT,
        failed_output=_FAILED_OUTPUT,
        error=_json_error(),
    )
    text = _content(messages)
    assert text.startswith("CRITICAL:")
    assert "not valid JSON" in text
    # The failed output is echoed and the original instruction restated.
    assert "Out Of Range" in text
    assert _ORIGINAL_PROMPT in text
    assert "Original instruction follows." in text


def test_schema_field_repair_directive():
    text = _content(rp.schema_field_repair(
        original_prompt=_ORIGINAL_PROMPT,
        failed_output=_FAILED_OUTPUT,
        error=_validation_error(),
    ))
    assert text.startswith("CRITICAL:")
    assert "valid JSON but failed schema validation" in text
    # It tells the model to touch only the named field(s).
    assert "Fix ONLY the" in text


def test_cast_membership_repair_directive():
    err = PostValidationError(
        "phase 'rising' beat speaker 'BANDITO' is not in locked cast "
        "['ALICE', 'BEN']"
    )
    text = _content(rp.cast_membership_repair(
        original_prompt=_ORIGINAL_PROMPT,
        failed_output=_FAILED_OUTPUT,
        error=err,
    ))
    assert text.startswith("CRITICAL:")
    assert "locked cast" in text
    assert "cast is FIXED" in text
    # The rejecting error -- which quotes the legal cast -- is echoed.
    assert "ALICE" in text and "BEN" in text


def test_too_many_words_repair_directive():
    text = _content(rp.too_many_words_repair(
        original_prompt=_ORIGINAL_PROMPT,
        failed_output=_FAILED_OUTPUT,
        error=PostValidationError("too_long"),
    ))
    assert text.startswith("CRITICAL:")
    assert "over the length budget" in text
    assert "SHORTER" in text


def test_narration_leak_repair_directive():
    text = _content(rp.narration_leak_repair(
        original_prompt=_ORIGINAL_PROMPT,
        failed_output=_FAILED_OUTPUT,
        error=PostValidationError("dialogue_verb"),
    ))
    assert text.startswith("CRITICAL:")
    assert "leaked narration" in text
    assert "visually present" in text


def test_forbidden_name_repair_directive():
    text = _content(rp.forbidden_name_repair(
        original_prompt=_ORIGINAL_PROMPT,
        failed_output=_FAILED_OUTPUT,
        error=PostValidationError("named_character"),
    ))
    assert text.startswith("CRITICAL:")
    assert "must not name anyone" in text
    assert "generic noun" in text


def test_payload_null_repair_directive():
    # Sprint 7C / BUG-LOCAL-275. The directive must (a) tell the model
    # the `payload` field cannot be null, (b) tell it to OMIT the edit
    # row entirely rather than emitting a null / empty placeholder.
    text = _content(rp.payload_null_repair(
        original_prompt=_ORIGINAL_PROMPT,
        failed_output=_FAILED_OUTPUT,
        error=_payload_null_validation_error(),
    ))
    assert text.startswith("CRITICAL:")
    assert "payload" in text.lower()
    assert "null" in text.lower()
    # The "omit the row" instruction is the load-bearing escape hatch
    # the generic schema_field_repair was missing.
    assert "OMIT" in text
    assert "non-null string" in text


def test_every_builder_returns_single_user_message():
    err = PostValidationError("named_character")
    builders = [
        rp.json_syntax_repair,
        rp.schema_field_repair,
        rp.cast_membership_repair,
        rp.too_many_words_repair,
        rp.narration_leak_repair,
        rp.forbidden_name_repair,
        rp.payload_null_repair,
    ]
    for builder in builders:
        messages = builder(
            original_prompt=_ORIGINAL_PROMPT,
            failed_output=_FAILED_OUTPUT,
            error=err,
        )
        assert isinstance(messages, list)
        assert len(messages) == 1
        assert set(messages[0]) == {"role", "content"}
        assert messages[0]["role"] == "user"


# ---------------------------------------------------------------------------
# 8-15. Dispatcher routing
# ---------------------------------------------------------------------------


def _dispatch(error):
    """Run the dispatching factory (no deterministic callback) on `error`."""
    factory = rp.make_dispatching_repair_factory()
    return factory(
        original_prompt=_ORIGINAL_PROMPT,
        failed_output=_FAILED_OUTPUT,
        error=error,
    )


def test_dispatch_json_decode_error_routes_to_json_syntax():
    text = _content(_dispatch(_json_error()))
    assert "not valid JSON" in text


def test_dispatch_validation_error_routes_to_schema_field():
    # The _validation_error() fixture is a `score=9999` out-of-range
    # failure -- not a null-payload failure, so the dispatcher must
    # fall through to the generic schema_field_repair directive.
    text = _content(_dispatch(_validation_error()))
    assert "valid JSON but failed schema validation" in text
    # The Sprint 7C typed directive must NOT fire for this error class.
    assert "OMIT the entire edit row" not in text


def test_dispatch_payload_null_routes_to_payload_null_repair():
    # Sprint 7C / BUG-LOCAL-275. A pydantic ValidationError whose repr
    # names a null `payload` field must take the dedicated route, not
    # the generic schema_field_repair fallback.
    text = _content(_dispatch(_payload_null_validation_error()))
    # The Sprint 7C directive's distinctive escape-hatch wording.
    assert "OMIT" in text
    assert "non-null string" in text
    # The generic schema_field_repair directive must NOT fire.
    assert "valid JSON but failed schema validation" not in text


def test_dispatch_payload_null_signal_is_specific():
    # A ValidationError that mentions some OTHER field being null must
    # NOT trigger the payload_null route -- the signal is `payload` +
    # `input_value=None` in the same error repr.

    class _UnrelatedSchema(BaseModel):
        rationale: str

    try:
        _UnrelatedSchema.model_validate({"rationale": None})
    except ValidationError as exc:
        err = exc
    else:
        raise AssertionError("model_validate should have raised")  # pragma: no cover

    # Sanity: this error DOES carry `input_value=None` but the field
    # path does NOT contain "payload" -- the dispatcher should not be
    # tricked into the Sprint 7C route by the null annotation alone.
    assert "input_value=none" in str(err).lower()
    assert "payload" not in str(err).lower()

    text = _content(_dispatch(err))
    assert "valid JSON but failed schema validation" in text
    assert "OMIT" not in text


def test_dispatch_locked_cast_routes_to_cast_membership():
    err = PostValidationError(
        "phase 'rising' beat speaker 'BANDITO' is not in locked cast "
        "['ALICE', 'BEN']"
    )
    text = _content(_dispatch(err))
    assert "cast is FIXED" in text


def test_dispatch_named_character_routes_to_forbidden_name():
    text = _content(_dispatch(PostValidationError("named_character")))
    assert "must not name anyone" in text


def test_dispatch_dialogue_verb_routes_to_narration_leak():
    text = _content(_dispatch(PostValidationError("dialogue_verb")))
    assert "leaked narration" in text


def test_dispatch_plot_verb_routes_to_narration_leak():
    text = _content(_dispatch(PostValidationError("plot_verb")))
    assert "leaked narration" in text


def test_dispatch_too_long_routes_to_too_many_words():
    text = _content(_dispatch(PostValidationError("too_long")))
    assert "over the length budget" in text


def test_dispatch_anchor_a2_routes_to_key_term_grounding():
    # Live-smoke hardening 2026-07-09: the original_radio brief gate's
    # "anchor A2" rejection must take the typed verbatim-copy route,
    # not the generic default fallback.
    err = PostValidationError(
        "key_term 'grieving mother' not grounded in the concept text "
        "(anchor A2)"
    )
    text = _content(_dispatch(err))
    assert "COPIED verbatim" in text
    assert "cast is FIXED" not in text


def test_dispatch_unrecognised_content_failure_routes_to_default():
    # A news V1 rejection matches no typed marker -> the generic
    # default_repair_prompt_factory handles it. The default factory's
    # distinctive wording is "failed schema validation because:".
    err = PostValidationError("V1: key_term 'zeppelin' not in source")
    text = _content(_dispatch(err))
    assert "because:" in text
    # It is NOT one of the typed directives.
    assert "leaked narration" not in text
    assert "over the length budget" not in text


# ---------------------------------------------------------------------------
# 16-18. Deterministic cast-membership short-circuit
# ---------------------------------------------------------------------------


def test_deterministic_callback_result_is_returned_directly():
    # When the deterministic callback resolves the phantom it returns a
    # pydantic instance; the dispatcher must hand that straight back
    # (structured_call then accepts it with no LLM call).
    resolved = _SampleSchema(title="Repaired", score=10)
    calls: list = []

    def deterministic_repair(failed_output, error):
        calls.append((failed_output, error))
        return resolved

    factory = rp.make_dispatching_repair_factory(
        deterministic_repair=deterministic_repair,
    )
    err = PostValidationError(
        "phase 'x' beat speaker 'ALCIE' is not in locked cast ['ALICE']"
    )
    result = factory(
        original_prompt=_ORIGINAL_PROMPT,
        failed_output=_FAILED_OUTPUT,
        error=err,
    )
    assert result is resolved
    assert len(calls) == 1, "deterministic callback should run exactly once"


def test_deterministic_callback_none_falls_through_to_llm_prompt():
    # An ambiguous phantom -> callback returns None -> the dispatcher
    # builds the cast_membership LLM repair prompt instead.
    def deterministic_repair(failed_output, error):
        return None

    factory = rp.make_dispatching_repair_factory(
        deterministic_repair=deterministic_repair,
    )
    err = PostValidationError(
        "phase 'x' beat speaker 'XQ' is not in locked cast ['ALICE', 'BEN']"
    )
    result = factory(
        original_prompt=_ORIGINAL_PROMPT,
        failed_output=_FAILED_OUTPUT,
        error=err,
    )
    text = _content(result)
    assert "cast is FIXED" in text


def test_deterministic_callback_consulted_for_anchor_a2():
    # The A2 branch mirrors the locked-cast contract: a schema instance
    # from the deterministic prune is handed straight back, no LLM turn.
    resolved = _SampleSchema(title="Pruned", score=5)
    calls: list = []

    def deterministic_repair(failed_output, error):
        calls.append(error)
        return resolved

    factory = rp.make_dispatching_repair_factory(
        deterministic_repair=deterministic_repair,
    )
    err = PostValidationError(
        "key_term 'grieving mother' not grounded in the concept text "
        "(anchor A2)"
    )
    result = factory(
        original_prompt=_ORIGINAL_PROMPT,
        failed_output=_FAILED_OUTPUT,
        error=err,
    )
    assert result is resolved
    assert len(calls) == 1, "deterministic callback should run exactly once"


def test_deterministic_callback_none_falls_to_a2_llm_prompt():
    # Prune declined (too few grounded survivors) -> the typed A2 LLM
    # repair prompt is built instead.
    def deterministic_repair(failed_output, error):
        return None

    factory = rp.make_dispatching_repair_factory(
        deterministic_repair=deterministic_repair,
    )
    err = PostValidationError(
        "key_term 'zeppelin post' not grounded in the concept text "
        "(anchor A2)"
    )
    result = factory(
        original_prompt=_ORIGINAL_PROMPT,
        failed_output=_FAILED_OUTPUT,
        error=err,
    )
    text = _content(result)
    assert "COPIED verbatim" in text


def test_deterministic_callback_not_consulted_for_non_cast_errors():
    # The callback must only ever run for the locked-cast and anchor-A2
    # branches. A JSON error, a schema error, and other content errors
    # must all leave it untouched.
    calls: list = []

    def deterministic_repair(failed_output, error):
        calls.append(error)
        return _SampleSchema(title="should not happen", score=0)

    factory = rp.make_dispatching_repair_factory(
        deterministic_repair=deterministic_repair,
    )
    for error in (
        _json_error(),
        _validation_error(),
        PostValidationError("too_long"),
        PostValidationError("named_character"),
    ):
        factory(
            original_prompt=_ORIGINAL_PROMPT,
            failed_output=_FAILED_OUTPUT,
            error=error,
        )
    assert calls == [], "deterministic callback ran for a non-cast error"
