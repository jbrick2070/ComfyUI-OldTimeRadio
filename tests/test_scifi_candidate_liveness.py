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


def test_the_candidate_loop_is_BOUNDED_and_kills_the_pass(monkeypatch):
    """OPERATOR RULING 2026-08-13: three cycles, then kill the workflow.

    THIS TEST USED TO ASSERT THE OPPOSITE. It was
    `test_thirteen_recoverable_candidate_exhaustions_then_success` and it pinned
    that a pass may burn THIRTEEN candidate ladders and succeed on the
    fourteenth -- the unbounded liveness `016ad146` introduced so the
    `scifi_news` bank could reroll its way out of cast-coverage failures.

    That generosity has a cost the operator saw on a live leg: a pass that
    cannot terminate grinds, and the only outer bound is a six-hour client
    timeout that does not even cancel the server job. The ruling, made with the
    trade-off stated explicitly -- it DOES cap the announcer-coverage recovery
    above -- was: "if it's running away three times and it can't get a good
    ledger, that means we need to actually fix the workflow."

    So the contract is now: three ladders, then a LOUD refusal.
    """
    monkeypatch.setattr(lane, "_poll_processing_interrupt", lambda: None)
    calls = 0

    def slot(messages, **_kwargs):
        nonlocal calls
        calls += 1
        return _question(f"REJECTED_RAW_{calls:02d}").model_dump_json()

    journal = {}
    with pytest.raises(lane.CodexPassError) as excinfo:
        _invoke(slot, journal=journal)

    assert "abandoned after 3 candidate cycles" in str(excinfo.value)

    entries = journal["calls"]
    assert [e["candidate_cycle"] for e in entries] == [1, 2, 3]
    assert all(e["status"] == "failed" for e in entries)

    # CYCLES are the bound, not attempts. A ladder may end early -- a
    # deterministic repair can close it at two rungs instead of three -- so the
    # attempt count is a RANGE, and asserting an exact product is how the first
    # version of this test (and the refusal message it mirrored) claimed nine
    # attempts on a run that made six.
    assert lane.MAX_CANDIDATE_CYCLES <= calls <= lane.MAX_CANDIDATE_CYCLES * 4, (
        f"{calls} attempts across 3 bounded cycles is outside any sane range")


def test_two_cycles_still_recover(monkeypatch):
    """The bound must not break the reroll it exists alongside.

    A cast-coverage failure that clears on the second ladder is the ordinary,
    healthy path -- exactly what `scifi_news` relies on. Bounding at three must
    leave it untouched; only the third exhaustion is fatal.
    """
    monkeypatch.setattr(lane, "_poll_processing_interrupt", lambda: None)
    prompts = []
    calls = 0

    def slot(messages, **_kwargs):
        nonlocal calls
        prompts.append(messages)
        calls += 1
        if calls <= 3:                       # the whole first ladder fails
            return _question(f"REJECTED_RAW_{calls:02d}").model_dump_json()
        return _question("Entirely fresh accepted fiction.").model_dump_json()

    journal = {}
    result = _invoke(slot, journal=journal)

    assert result.question == "Entirely fresh accepted fiction."
    assert calls == 4
    entries = journal["calls"]
    assert [e["candidate_cycle"] for e in entries] == [1, 2]
    assert entries[-1]["status"] == "accepted"
    nonces = [e["candidate_nonce"] for e in entries[1:]]
    assert len(nonces) == len(set(nonces)) and all(len(n) == 32 for n in nonces)

    first_body = json.loads(prompts[0][1]["content"])
    assert first_body["artifact_inputs"] == {"stable": "input"}

    # The reroll's CARRY-FORWARD contract, kept from the retired 13-cycle test:
    # the first prompt has no retry block, and the cycle-2 prompt carries the
    # cycle, a BOUNDED rejection summary, and never echoes the abandoned prose
    # back to the model.
    assert "writer_retry" not in first_body

    # Select by SHAPE, not by index. A ladder interleaves base prompts
    # ([system, user]) with repair prompts, which carry a single message -- the
    # retired test used a prompts[2::2] stride for exactly that reason, and
    # indexing blind into the last prompt lands on a repair and raises.
    retries = []
    for prompt in prompts:
        if len(prompt) < 2:
            continue
        body = json.loads(prompt[1]["content"])
        if "writer_retry" in body:
            retries.append((prompt, body["writer_retry"]))
    assert retries, "no reroll prompt carried a writer_retry block"

    prompt, retry = retries[0]
    assert retry["cycle"] >= 2
    assert len(retry["previous_rejection"]) <= 600
    assert "prior candidate is abandoned" in retry["instruction"]
    # The abandoned draft must never be echoed back to the model.
    assert "REJECTED_RAW_" not in prompt[1]["content"]
    assert "REJECTED_RAW_" not in json.dumps(journal, sort_keys=True)


def test_the_refusal_claims_no_number_it_cannot_defend(monkeypatch, caplog):
    """The kill message must not invent arithmetic.

    Two drafts of that line carried an ATTEMPT count and both were wrong. The
    first multiplied cycles by a mirrored rung constant and said nine on a run
    that made six, because a ladder can close early on a deterministic repair.
    The second read the journal and said three, because the journal records one
    entry PER CYCLE -- the attempts happen inside the ladder and never surface
    at this level at all.

    This refusal justifies killing an operator's render, so it now states only
    the cycle count, which is the one number this scope actually owns.
    """
    import inspect
    import re

    # Asserted against the SOURCE, not a captured log line. Two attempts to
    # capture the record fought this pack's logging configuration and told me
    # nothing about the product; the property worth guarding is simply that the
    # refusal never formats an attempt count, and that is visible statically.
    src = inspect.getsource(lane.invoke_codex_structured)
    kill = src[src.index("ABANDONED"):]

    assert "candidate cycle(s)" in kill, \
        "the refusal no longer states the one count it owns"
    assert not re.search(r"attempt(ed|s)?\s+%[ds]", kill), (
        "the refusal formats an attempt count again -- that number lives "
        "inside the ladder and is not knowable here; two earlier drafts got "
        "it wrong in two different ways")
    assert "_LADDER_RUNGS" not in src, (
        "the mirrored rung constant is back; it is what produced the first "
        "wrong number")


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
