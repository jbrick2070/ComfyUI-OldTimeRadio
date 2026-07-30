from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from nodes import _otr_scifi_codex as lane
from nodes._otr_generation_budget import (
    CAPACITY_PHASE_OUTPUT_LIMIT,
    CAPACITY_PHASE_PROMPT_NO_ROOM,
    PromptContextOverflowError,
)
from nodes._otr_structured_call import (
    PostValidationError,
    StructuredCallFailedError,
)


def _question(text: str = "Who answers?") -> lane.DramaticQuestionV4:
    return lane.DramaticQuestionV4(
        question=text,
        consequence="A choice changes the station.",
        ending_direction="The listeners choose for themselves.",
    )


def _invoke(slot_fn, *, journal=None, retry=True):
    return lane.invoke_codex_structured(
        pass_id="P1",
        slot="creative",
        slot_fn=slot_fn,
        pack=SimpleNamespace(prompt_stages={
            "codex_question_system": "question seam",
        }),
        seam_refs=("codex_question_system",),
        artifact_inputs={"stable": "input"},
        result_type=lane.DramaticQuestionV4,
        post_validator=lambda value: (
            "candidate rejected"
            if value.question.startswith("REJECTED_RAW_") else None
        ),
        base_temperature=.72,
        structural_retry_temperature=.32,
        max_new_tokens=None,
        call_journal=journal if journal is not None else {},
        retry_until_valid=retry,
    )


def test_thirteen_recoverable_candidate_exhaustions_then_success(monkeypatch):
    monkeypatch.setattr(lane, "_poll_processing_interrupt", lambda: None)
    prompts = []
    calls = 0

    def slot(messages, **_kwargs):
        nonlocal calls
        prompts.append(messages)
        calls += 1
        if calls <= 26:
            return _question(f"REJECTED_RAW_{calls:02d}").model_dump_json()
        return _question("Entirely fresh accepted fiction.").model_dump_json()

    journal = {}
    result = _invoke(slot, journal=journal)

    assert result.question == "Entirely fresh accepted fiction."
    assert calls == 27
    entries = journal["calls"]
    assert [entry["candidate_cycle"] for entry in entries] == list(range(1, 15))
    assert [entry["status"] for entry in entries[:-1]] == ["failed"] * 13
    assert all("accepted" not in entry for entry in entries[:-1])
    assert entries[-1]["status"] == "accepted"
    nonces = [entry["candidate_nonce"] for entry in entries[1:]]
    assert len(nonces) == len(set(nonces)) == 13
    assert all(len(nonce) == 32 for nonce in nonces)

    first_body = json.loads(prompts[0][1]["content"])
    assert first_body["artifact_inputs"] == {"stable": "input"}
    assert "writer_retry" not in first_body
    for prompt in prompts[2::2]:
        body = json.loads(prompt[1]["content"])
        retry = body["writer_retry"]
        assert retry["cycle"] >= 2
        assert len(retry["previous_rejection"]) <= 600
        assert "prior candidate is abandoned" in retry["instruction"]
        assert "REJECTED_RAW_" not in prompt[1]["content"]
    assert "REJECTED_RAW_" not in json.dumps(journal, sort_keys=True)


def test_pydantic_rejection_does_not_echo_abandoned_prose(monkeypatch):
    monkeypatch.setattr(lane, "_poll_processing_interrupt", lambda: None)
    sentinel = "REJECTED_PROSE_LEAK_SENTINEL"
    invalid = json.dumps({
        "question": [sentinel],
        "consequence": "A choice changes the station.",
        "ending_direction": "The listeners choose for themselves.",
    })
    accepted = _question("Fresh fiction with no inherited draft.").model_dump_json()
    responses = [invalid, invalid, accepted]
    prompts = []

    def slot(messages, **_kwargs):
        prompts.append(messages)
        return responses[len(prompts) - 1]

    journal = {}
    result = _invoke(slot, journal=journal)

    assert result.question == "Fresh fiction with no inherited draft."
    assert len(prompts) == 3
    assert sentinel in str(prompts[1])
    assert sentinel not in str(prompts[2])
    assert sentinel not in json.dumps(journal, sort_keys=True)
    assert journal["calls"][0]["terminal_error"].startswith(
        "ValidationError: schema validation failed"
    )


def _validation_error() -> ValidationError:
    try:
        lane.DramaticQuestionV4.model_validate({})
    except ValidationError as exc:
        return exc
    raise AssertionError("fixture did not raise")


@pytest.mark.parametrize(
    "terminal",
    (
        json.JSONDecodeError("bad json", "{", 0),
        _validation_error(),
        PostValidationError("bad content"),
        PromptContextOverflowError(
            "ran out of output",
            phase=CAPACITY_PHASE_OUTPUT_LIMIT,
        ),
    ),
)
def test_only_typed_candidate_failures_start_a_fresh_cycle(
        monkeypatch, terminal):
    monkeypatch.setattr(lane, "_poll_processing_interrupt", lambda: None)
    expected = _question()
    calls = 0

    def once(**_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise StructuredCallFailedError(
                helper_name="fixture",
                attempts=2,
                last_error=terminal,
            )
        return expected

    monkeypatch.setattr(lane, "_invoke_codex_structured_once", once)
    assert _invoke(lambda *_args, **_kwargs: "") == expected
    assert calls == 2


def test_prompt_no_room_and_provider_failures_do_not_reroll(monkeypatch):
    monkeypatch.setattr(lane, "_poll_processing_interrupt", lambda: None)
    calls = 0

    def provider(_messages, **_kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("provider unavailable")

    with pytest.raises(lane.CodexPassError, match="provider unavailable"):
        _invoke(provider)
    assert calls == 1

    once_calls = 0

    def prompt_no_room(**_kwargs):
        nonlocal once_calls
        once_calls += 1
        raise StructuredCallFailedError(
            helper_name="fixture",
            attempts=1,
            last_error=PromptContextOverflowError(
                "no prompt room",
                phase=CAPACITY_PHASE_PROMPT_NO_ROOM,
            ),
        )

    monkeypatch.setattr(lane, "_invoke_codex_structured_once", prompt_no_room)
    with pytest.raises(lane.CodexPassError, match="no prompt room"):
        _invoke(lambda *_args, **_kwargs: "")
    assert once_calls == 1


def test_retry_disabled_preserves_one_ladder_fail_loud_contract(monkeypatch):
    monkeypatch.setattr(lane, "_poll_processing_interrupt", lambda: None)
    calls = 0

    def once(**_kwargs):
        nonlocal calls
        calls += 1
        raise StructuredCallFailedError(
            helper_name="fixture",
            attempts=2,
            last_error=PostValidationError("still invalid"),
        )

    monkeypatch.setattr(lane, "_invoke_codex_structured_once", once)
    with pytest.raises(lane.CodexPassError, match="still invalid"):
        _invoke(lambda *_args, **_kwargs: "", retry=False)
    assert calls == 1


def test_comfy_cancel_escapes_by_identity_before_next_candidate(monkeypatch):
    class Cancelled(BaseException):
        pass

    sentinel = Cancelled("cancel")
    polls = 0
    calls = 0

    def poll():
        nonlocal polls
        polls += 1
        if polls == 2:
            raise sentinel

    def once(**_kwargs):
        nonlocal calls
        calls += 1
        raise StructuredCallFailedError(
            helper_name="fixture",
            attempts=2,
            last_error=PostValidationError("retryable"),
        )

    monkeypatch.setattr(lane, "_poll_processing_interrupt", poll)
    monkeypatch.setattr(lane, "_invoke_codex_structured_once", once)
    with pytest.raises(Cancelled) as caught:
        _invoke(lambda *_args, **_kwargs: "")
    assert caught.value is sentinel
    assert calls == 1
