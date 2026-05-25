"""Sprint 2A step 1: shared structured-LLM-call helper tests.

The module under test (`nodes/_otr_structured_call.py`) is pure -- no
I/O, no GPU, no ComfyUI imports. Every test here runs against a small
pydantic schema and a call-
counting stub `slot_fn` closure. No existing call site is exercised;
converting the structured passes onto `structured_call` is later work.

Coverage map:

  Retry ladder (`structured_call`):
    1. Attempt 1 succeeds -> one slot call, valid result returned.
    2. Attempt 1 fails (bad JSON), Attempt 2 succeeds -> two calls;
       Attempt 2 ran at `structural_retry_temperature`.
    3. Attempt 1 fails (schema), Attempt 2 succeeds -> same, schema
       rejection path.
    4. Attempts 1+2 fail, Attempt 3 (repair) succeeds -> three calls,
       `repair_prompt_factory` was invoked.
    5. Attempts 1+2 fail, default repair factory used when caller
       passes None -> Attempt 3 still runs (CRITICAL prefix present).
    6. All attempts fail -> StructuredCallFailedError, message carries
       helper_name + last error.
    7. max_attempts cap is honoured (max_attempts=1 -> one call only).

  2B invariant:
    8. structural_retry_temperature == base_temperature -> ValueError.
    9. structural_retry_temperature > base_temperature -> ValueError.

  post_validator (content validation beyond the pydantic schema):
   10. post_validator returns None -> the schema-valid instance is
       accepted; one slot call.
   11. post_validator rejection advances the ladder past a schema-
       valid Attempt 1 to Attempt 2.
   12. a content rejection feeds the typed-repair factory as a
       PostValidationError carrying the content message.
   13. all attempts content-rejected -> StructuredCallFailedError
       whose last_error is a PostValidationError.
   14. PostValidationError subclasses ValueError (so the ladder's
       existing except arms catch it unchanged).

  max_new_tokens (per-caller token budget):
   15. a caller-supplied max_new_tokens reaches slot_fn on every
       attempt.
   16. max_new_tokens defaults to _STRUCTURED_MAX_NEW_TOKENS (512).
"""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel, Field

from nodes import _otr_structured_call as sc


# ---------------------------------------------------------------------------
# Fixtures -- schema + call-counting stub slot fn
# ---------------------------------------------------------------------------


class SampleSchema(BaseModel):
    """Small pydantic schema exercised by the ladder tests."""

    title: str = Field(min_length=1, max_length=120)
    score: int = Field(ge=0, le=100)


def _valid_json() -> str:
    """A response string that parses + validates against SampleSchema."""
    return '{"title": "The Bulkhead Closes", "score": 42}'


def _bad_json() -> str:
    """A response string that is not decodable JSON."""
    return "this is not JSON at all {{{"


def _schema_violating_json() -> str:
    """Valid JSON, but `score` is out of the 0-100 range -> schema reject."""
    return '{"title": "Out Of Range", "score": 9999}'


def _make_counting_slot_fn(*, responses: list[str]):
    """Return a slot fn that records every call and pops queued responses.

    Records `(messages, temperature, max_new_tokens)` on each call.
    When `responses` is empty it returns an empty string,
    which fails JSON parsing -- so an over-run ladder still fails
    cleanly rather than raising IndexError.

    The closure name is deliberately descriptive; the project forbids
    the word "dummy" in any stub / fixture name.
    """
    calls: list[dict] = []

    def slot_fn(messages, *, temperature, max_new_tokens):
        calls.append({
            "messages": messages,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
        })
        if not responses:
            return ""
        return responses.pop(0)

    slot_fn.calls = calls  # type: ignore[attr-defined]
    return slot_fn


def _make_recording_repair_factory():
    """Return a repair_prompt_factory that records its invocations.

    Builds a minimal `user` message and stamps `.invocations` so a
    test can assert the typed-repair attempt ran.
    """
    invocations: list[dict] = []

    def factory(*, original_prompt, failed_output, error):
        invocations.append({
            "original_prompt": original_prompt,
            "failed_output": failed_output,
            "error": error,
        })
        return [{"role": "user", "content": "REPAIR: please return valid JSON"}]

    factory.invocations = invocations  # type: ignore[attr-defined]
    return factory


# ---------------------------------------------------------------------------
# 1. Attempt 1 succeeds
# ---------------------------------------------------------------------------


def test_attempt_one_success_returns_validated_instance():
    slot = _make_counting_slot_fn(responses=[_valid_json()])
    result = sc.structured_call(
        prompt="produce a title and score",
        schema=SampleSchema,
        slot_fn=slot,
        base_temperature=0.40,
        structural_retry_temperature=0.20,
        helper_name="sample_pass",
    )
    assert isinstance(result, SampleSchema)
    assert result.title == "The Bulkhead Closes"
    assert result.score == 42
    assert len(slot.calls) == 1, "Attempt 1 success should make exactly one call"
    assert slot.calls[0]["temperature"] == pytest.approx(0.40)


# ---------------------------------------------------------------------------
# 2. Attempt 1 fails (bad JSON), Attempt 2 succeeds at the lower temperature
# ---------------------------------------------------------------------------


def test_attempt_two_runs_at_structural_retry_temperature():
    slot = _make_counting_slot_fn(responses=[_bad_json(), _valid_json()])
    result = sc.structured_call(
        prompt="produce a title and score",
        schema=SampleSchema,
        slot_fn=slot,
        base_temperature=0.45,
        structural_retry_temperature=0.15,
        helper_name="sample_pass",
    )
    assert isinstance(result, SampleSchema)
    assert len(slot.calls) == 2, "bad JSON on Attempt 1 should trigger Attempt 2"
    # Attempt 1 ran at base temperature.
    assert slot.calls[0]["temperature"] == pytest.approx(0.45)
    # Attempt 2 ran at the (lower) structural retry temperature.
    assert slot.calls[1]["temperature"] == pytest.approx(0.15)


# ---------------------------------------------------------------------------
# 3. Attempt 1 fails (schema violation), Attempt 2 succeeds
# ---------------------------------------------------------------------------


def test_schema_violation_on_attempt_one_advances_to_attempt_two():
    slot = _make_counting_slot_fn(
        responses=[_schema_violating_json(), _valid_json()],
    )
    result = sc.structured_call(
        prompt="produce a title and score",
        schema=SampleSchema,
        slot_fn=slot,
        base_temperature=0.50,
        structural_retry_temperature=0.25,
        helper_name="sample_pass",
    )
    assert result.score == 42
    assert len(slot.calls) == 2
    assert slot.calls[1]["temperature"] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# 4. Attempts 1+2 fail, Attempt 3 (repair) succeeds; factory was invoked
# ---------------------------------------------------------------------------


def test_repair_attempt_runs_and_invokes_factory():
    slot = _make_counting_slot_fn(
        responses=[_bad_json(), _bad_json(), _valid_json()],
    )
    repair_factory = _make_recording_repair_factory()
    result = sc.structured_call(
        prompt="produce a title and score",
        schema=SampleSchema,
        slot_fn=slot,
        base_temperature=0.40,
        structural_retry_temperature=0.20,
        repair_prompt_factory=repair_factory,
        helper_name="sample_pass",
    )
    assert isinstance(result, SampleSchema)
    assert len(slot.calls) == 3, "Attempt 3 (repair) should be the third call"
    assert len(repair_factory.invocations) == 1, (
        "repair_prompt_factory should be invoked exactly once on Attempt 3"
    )
    # The factory received the failed output + the prior error.
    invocation = repair_factory.invocations[0]
    assert invocation["failed_output"] == _bad_json()
    assert isinstance(invocation["error"], BaseException)
    # Attempt 3 used the static low repair temperature, not a raised one.
    assert slot.calls[2]["temperature"] == pytest.approx(sc._REPAIR_TEMPERATURE)
    assert slot.calls[2]["temperature"] < 0.40


# ---------------------------------------------------------------------------
# 5. Default repair factory is used when the caller passes None
# ---------------------------------------------------------------------------


def test_default_repair_factory_used_when_none_passed():
    slot = _make_counting_slot_fn(
        responses=[_bad_json(), _bad_json(), _valid_json()],
    )
    result = sc.structured_call(
        prompt="produce a title and score",
        schema=SampleSchema,
        slot_fn=slot,
        base_temperature=0.40,
        structural_retry_temperature=0.20,
        repair_prompt_factory=None,
        helper_name="sample_pass",
    )
    assert isinstance(result, SampleSchema)
    assert len(slot.calls) == 3
    # The default factory prepends a CRITICAL directive to the repair
    # message, so Attempt 3's prompt should carry that marker.
    attempt_three_messages = slot.calls[2]["messages"]
    repair_text = attempt_three_messages[0]["content"]
    assert repair_text.startswith("CRITICAL:"), repair_text[:80]


# ---------------------------------------------------------------------------
# 6. All attempts fail -> StructuredCallFailedError, message detail
# ---------------------------------------------------------------------------


def test_all_attempts_fail_raises_structured_call_failed_error():
    # Three bad responses -> Attempt 1, 2, 3 all fail; the ladder ends
    # after Attempt 3.
    slot = _make_counting_slot_fn(
        responses=[_bad_json(), _bad_json(), _bad_json()],
    )
    with pytest.raises(sc.StructuredCallFailedError) as exc_info:
        sc.structured_call(
            prompt="produce a title and score",
            schema=SampleSchema,
            slot_fn=slot,
            base_temperature=0.40,
            structural_retry_temperature=0.20,
            helper_name="critic_verdict_pass",
        )
    err = exc_info.value
    # helper_name is carried on the exception and in its message.
    assert err.helper_name == "critic_verdict_pass"
    assert "critic_verdict_pass" in str(err)
    # The last validation / parse error is carried and named.
    assert err.last_error is not None
    assert "last error" in str(err)
    # attempts attribute reflects the ladder length actually run.
    assert err.attempts == 3


def test_failed_error_message_names_last_error_type():
    slot = _make_counting_slot_fn(
        responses=[_bad_json(), _bad_json(), _bad_json()],
    )
    with pytest.raises(sc.StructuredCallFailedError) as exc_info:
        sc.structured_call(
            prompt="produce a title and score",
            schema=SampleSchema,
            slot_fn=slot,
            base_temperature=0.40,
            structural_retry_temperature=0.20,
            helper_name="sample_pass",
        )
    # Bad JSON on every attempt -> last error is a JSONDecodeError.
    assert isinstance(exc_info.value.last_error, json.JSONDecodeError)
    assert "JSONDecodeError" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 7. max_attempts cap is honoured
# ---------------------------------------------------------------------------


def test_max_attempts_one_caps_ladder_at_single_call():
    slot = _make_counting_slot_fn(responses=[_bad_json(), _valid_json()])
    with pytest.raises(sc.StructuredCallFailedError) as exc_info:
        sc.structured_call(
            prompt="produce a title and score",
            schema=SampleSchema,
            slot_fn=slot,
            base_temperature=0.40,
            structural_retry_temperature=0.20,
            max_attempts=1,
            helper_name="sample_pass",
        )
    # max_attempts=1 -> only Attempt 1 runs; the valid second response
    # is never reached.
    assert len(slot.calls) == 1
    assert exc_info.value.attempts == 1


# ---------------------------------------------------------------------------
# 8-9. The 2B invariant -- structural_retry_temperature < base_temperature
# ---------------------------------------------------------------------------


def test_equal_temperatures_fail_loud():
    slot = _make_counting_slot_fn(responses=[_valid_json()])
    with pytest.raises(ValueError) as exc_info:
        sc.structured_call(
            prompt="produce a title and score",
            schema=SampleSchema,
            slot_fn=slot,
            base_temperature=0.30,
            structural_retry_temperature=0.30,
            helper_name="sample_pass",
        )
    assert "structural_retry_temperature" in str(exc_info.value)
    # The invariant is checked at entry -- the slot fn never ran.
    assert len(slot.calls) == 0


def test_higher_structural_retry_temperature_fails_loud():
    slot = _make_counting_slot_fn(responses=[_valid_json()])
    with pytest.raises(ValueError) as exc_info:
        sc.structured_call(
            prompt="produce a title and score",
            schema=SampleSchema,
            slot_fn=slot,
            base_temperature=0.30,
            structural_retry_temperature=0.55,
            helper_name="sample_pass",
        )
    assert "STRICTLY LOWER" in str(exc_info.value)
    assert len(slot.calls) == 0


# ---------------------------------------------------------------------------
# 10-14. post_validator -- content validation beyond the pydantic schema
# ---------------------------------------------------------------------------


def _make_counting_post_validator(*, verdicts: list):
    """Return a post_validator that records calls and pops queued verdicts.

    Each queued verdict is either None (content accepted) or an error
    string (content rejected). When `verdicts` is empty the validator
    accepts -- so an over-run ladder never raises IndexError.
    """
    seen: list = []

    def post_validator(instance):
        seen.append(instance)
        if not verdicts:
            return None
        return verdicts.pop(0)

    post_validator.seen = seen  # type: ignore[attr-defined]
    return post_validator


def test_post_validator_none_return_accepts_instance():
    # Schema-valid response + a post_validator that accepts -> one call.
    slot = _make_counting_slot_fn(responses=[_valid_json()])
    post_validator = _make_counting_post_validator(verdicts=[None])
    result = sc.structured_call(
        prompt="produce a title and score",
        schema=SampleSchema,
        slot_fn=slot,
        base_temperature=0.40,
        structural_retry_temperature=0.20,
        post_validator=post_validator,
        helper_name="sample_pass",
    )
    assert isinstance(result, SampleSchema)
    assert len(slot.calls) == 1
    # post_validator saw the schema-valid instance exactly once.
    assert len(post_validator.seen) == 1
    assert isinstance(post_validator.seen[0], SampleSchema)


def test_post_validator_rejection_advances_the_ladder():
    # Both responses are schema-valid; the post_validator rejects the
    # first on content grounds and accepts the second. The ladder must
    # advance to Attempt 2 even though Attempt 1 was schema-valid.
    slot = _make_counting_slot_fn(responses=[_valid_json(), _valid_json()])
    post_validator = _make_counting_post_validator(
        verdicts=["score is content-invalid for this pass", None],
    )
    result = sc.structured_call(
        prompt="produce a title and score",
        schema=SampleSchema,
        slot_fn=slot,
        base_temperature=0.45,
        structural_retry_temperature=0.15,
        post_validator=post_validator,
        helper_name="sample_pass",
    )
    assert isinstance(result, SampleSchema)
    assert len(slot.calls) == 2, (
        "a content rejection on Attempt 1 should advance to Attempt 2"
    )
    # Attempt 2 ran at the lowered structural retry temperature.
    assert slot.calls[1]["temperature"] == pytest.approx(0.15)
    assert len(post_validator.seen) == 2


def test_post_validator_failure_feeds_typed_repair_factory():
    # Attempts 1+2 are schema-valid but content-rejected; Attempt 3
    # (repair) is accepted. The repair factory must receive a
    # PostValidationError carrying the content message.
    slot = _make_counting_slot_fn(
        responses=[_valid_json(), _valid_json(), _valid_json()],
    )
    repair_factory = _make_recording_repair_factory()
    post_validator = _make_counting_post_validator(
        verdicts=["too few key terms", "too few key terms", None],
    )
    result = sc.structured_call(
        prompt="produce a title and score",
        schema=SampleSchema,
        slot_fn=slot,
        base_temperature=0.40,
        structural_retry_temperature=0.20,
        repair_prompt_factory=repair_factory,
        post_validator=post_validator,
        helper_name="sample_pass",
    )
    assert isinstance(result, SampleSchema)
    assert len(slot.calls) == 3
    assert len(repair_factory.invocations) == 1
    repair_error = repair_factory.invocations[0]["error"]
    assert isinstance(repair_error, sc.PostValidationError)
    assert "too few key terms" in str(repair_error)


def test_post_validator_exhaustion_raises_with_post_validation_error():
    # Every attempt is schema-valid but the post_validator rejects all
    # of them -> StructuredCallFailedError whose last_error is a
    # PostValidationError.
    slot = _make_counting_slot_fn(
        responses=[_valid_json(), _valid_json(), _valid_json()],
    )
    post_validator = _make_counting_post_validator(
        verdicts=["content reject", "content reject", "content reject"],
    )
    with pytest.raises(sc.StructuredCallFailedError) as exc_info:
        sc.structured_call(
            prompt="produce a title and score",
            schema=SampleSchema,
            slot_fn=slot,
            base_temperature=0.40,
            structural_retry_temperature=0.20,
            post_validator=post_validator,
            helper_name="casting_pass",
        )
    assert len(slot.calls) == 3
    assert isinstance(exc_info.value.last_error, sc.PostValidationError)
    assert "content reject" in str(exc_info.value.last_error)
    assert exc_info.value.attempts == 3


def test_post_validation_error_is_a_value_error():
    # PostValidationError subclasses ValueError so the ladder's existing
    # except (json.JSONDecodeError, ValidationError, ValueError) arms
    # catch a content rejection with no change.
    assert issubclass(sc.PostValidationError, ValueError)


# ---------------------------------------------------------------------------
# 15-16. max_new_tokens -- per-caller token budget
# ---------------------------------------------------------------------------


def test_max_new_tokens_passthrough_reaches_slot_fn():
    # A caller-supplied token budget must reach slot_fn unchanged on
    # every attempt. The structured passes vary widely (story brief
    # ~160, cast audit ~2000, script doctor ~3500); the helper must not
    # clamp them to its 512 default.
    slot = _make_counting_slot_fn(responses=[_bad_json(), _valid_json()])
    result = sc.structured_call(
        prompt="produce a title and score",
        schema=SampleSchema,
        slot_fn=slot,
        base_temperature=0.40,
        structural_retry_temperature=0.20,
        max_new_tokens=3500,
        helper_name="script_doctor_pass",
    )
    assert isinstance(result, SampleSchema)
    assert len(slot.calls) == 2
    # Both the base attempt and the structural retry used the budget.
    assert slot.calls[0]["max_new_tokens"] == 3500
    assert slot.calls[1]["max_new_tokens"] == 3500


def test_max_new_tokens_defaults_to_structured_constant():
    # When the caller does not set max_new_tokens it falls back to the
    # module default (512) -- a small JSON object fits comfortably.
    slot = _make_counting_slot_fn(responses=[_valid_json()])
    sc.structured_call(
        prompt="produce a title and score",
        schema=SampleSchema,
        slot_fn=slot,
        base_temperature=0.40,
        structural_retry_temperature=0.20,
        helper_name="sample_pass",
    )
    assert len(slot.calls) == 1
    assert slot.calls[0]["max_new_tokens"] == sc._STRUCTURED_MAX_NEW_TOKENS
    assert sc._STRUCTURED_MAX_NEW_TOKENS == 512
