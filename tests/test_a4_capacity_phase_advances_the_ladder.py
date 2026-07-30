"""A-4: a capacity failure has a phase, and the phase decides the retry.

PBUG-20260729-02, from the live 45-word campaign: `PromptContextOverflowError`
is a plain `RuntimeError`, the ladder's attempt handlers caught only JSON,
schema and content failures, so the capacity raise escaped through
`except Exception: raise` on attempt 1 of 3. The ladder advertised a
three-call budget and spent one -- 24 minutes for one dead leg -- and the
operator's ruling is that a runaway advances the ladder instead of ending the
episode.

The phase is what makes that safe to do:

  output_limit   -- the call RAN and used its whole output allowance without
                    stopping. Sampling is stochastic, so the same prompt at a
                    lower temperature is a real second chance. RETRIES.
  prompt_no_room -- refused BEFORE the call by deterministic arithmetic. A
                    re-roll re-derives the identical refusal. NEVER retries.

What this file pins:

  FIRES          -- an output_limit failure on attempt 1 lets attempt 2 run,
                    and a good attempt 2 returns a result.
  FIRES          -- an unlucky pass spends its whole advertised budget and
                    dies as a StructuredCallFailedError carrying the capacity
                    error, not as a bare RuntimeError on attempt 1.
  COUNTERFACTUAL -- a prompt_no_room failure still propagates untouched after
                    exactly ONE call. Being catchable is not being retryable.
  CONTROL        -- the phase vocabulary is closed, the predicate is the only
                    opinion, and the writer's re-export is the SAME object the
                    ladder decides about.

No GPU, no network. UTF-8, no BOM, ASCII-only source.
"""
from __future__ import annotations

import pytest
from pydantic import BaseModel

from nodes import OTR_LedgerScriptWriter as writer
from nodes import _otr_generation_budget as budget
from nodes import _otr_structured_call as ladder


class _Tiny(BaseModel):
    ok: bool


GOOD = '{"ok": true}'
BASE_TEMPERATURE = 0.4
# NOT 0.1. The first draft used 0.1 and the mutation round caught it: that is
# also `_REPAIR_TEMPERATURE`, so "the second call ran at 0.1" was true whether
# the structural rung or the typed-repair rung made it, and reverting the
# structural gate went undetected. The rungs must be distinguishable by the
# only thing the fake slot can see, and `test_the_two_rungs_stay_
# distinguishable` below fails if a future constant change collides again.
STRUCTURAL_TEMPERATURE = 0.05


def _run(slot_fn, **overrides):
    kwargs = {
        "prompt": "produce the artifact",
        "schema": _Tiny,
        "slot_fn": slot_fn,
        "base_temperature": BASE_TEMPERATURE,
        "structural_retry_temperature": STRUCTURAL_TEMPERATURE,
        "helper_name": "a4_probe",
    }
    kwargs.update(overrides)
    return ladder.structured_call(**kwargs)


def _capacity(phase):
    return budget.PromptContextOverflowError(
        "prose generation exhausted the full remaining provider/context "
        "capacity (14697 output tokens after a 1687-token prompt); the partial "
        "artifact is discarded, never repaired as prose",
        phase=phase,
    )


def test_an_output_limit_failure_lets_the_next_rung_run():
    """FIRES: before A-4 this raised on attempt 1 and never reached attempt 2."""
    calls = []

    def slot(messages, **kwargs):
        calls.append(kwargs.get("temperature"))
        if len(calls) == 1:
            raise _capacity(budget.CAPACITY_PHASE_OUTPUT_LIMIT)
        return GOOD

    result = _run(slot)

    assert result.ok is True
    assert len(calls) == 2
    # The second call is the STRUCTURAL rung -- same prompt, lower temperature.
    # Asserting the temperature, not just the count, is what tells the
    # structural rung apart from the typed-repair rung that would otherwise
    # have made attempt 2.
    assert calls == [BASE_TEMPERATURE, STRUCTURAL_TEMPERATURE]


def test_the_two_rungs_stay_distinguishable():
    """The guard on the test above: if these collide, it goes blind."""
    assert STRUCTURAL_TEMPERATURE != ladder._REPAIR_TEMPERATURE
    assert STRUCTURAL_TEMPERATURE != BASE_TEMPERATURE


def test_an_unlucky_pass_spends_the_whole_advertised_budget():
    """FIRES: three attempts, then a typed death that carries the cause."""
    calls = []

    def slot(messages, **kwargs):
        calls.append(kwargs.get("temperature"))
        raise _capacity(budget.CAPACITY_PHASE_OUTPUT_LIMIT)

    with pytest.raises(ladder.StructuredCallFailedError) as caught:
        _run(slot)

    exc = caught.value
    assert len(calls) == 3
    assert exc.attempts == 3
    assert exc.terminal_disposition == "primary_ladder_exhausted"
    assert isinstance(exc.last_error, budget.PromptContextOverflowError)
    assert exc.last_error.phase == budget.CAPACITY_PHASE_OUTPUT_LIMIT


def test_a_prompt_that_cannot_fit_still_dies_on_the_spot():
    """COUNTERFACTUAL: catchable is not retryable."""
    calls = []

    def slot(messages, **kwargs):
        calls.append(kwargs.get("temperature"))
        raise _capacity(budget.CAPACITY_PHASE_PROMPT_NO_ROOM)

    with pytest.raises(budget.PromptContextOverflowError) as caught:
        _run(slot)

    assert len(calls) == 1
    assert caught.value.phase == budget.CAPACITY_PHASE_PROMPT_NO_ROOM


def test_a_pre_call_budget_refusal_also_dies_on_the_spot():
    """COUNTERFACTUAL: the same rule for the error `fit_output_tokens` raises."""
    calls = []

    def slot(messages, **kwargs):
        calls.append(kwargs.get("temperature"))
        raise budget.GenerationContextOverflowError("prompt cannot fit")

    with pytest.raises(budget.GenerationContextOverflowError):
        _run(slot)

    assert len(calls) == 1


def test_an_ordinary_json_failure_still_advances_the_ladder():
    """CONTROL: A-4 changed nothing for the failures that always retried."""
    calls = []

    def slot(messages, **kwargs):
        calls.append(kwargs.get("temperature"))
        if len(calls) == 1:
            return "not json at all"
        return GOOD

    result = _run(slot)

    assert result.ok is True
    assert len(calls) == 2


def test_an_unrelated_runtime_error_is_still_terminal():
    """CONTROL: the widened catch is the capacity types, not `Exception`."""
    calls = []

    def slot(messages, **kwargs):
        calls.append(kwargs.get("temperature"))
        raise RuntimeError("the GPU fell over")

    with pytest.raises(RuntimeError, match="GPU fell over"):
        _run(slot)

    assert len(calls) == 1


def test_both_budget_refusals_report_the_pre_call_phase():
    with pytest.raises(budget.GenerationContextOverflowError) as no_room:
        budget.fit_output_tokens(512, context_cap=8192, prompt_tokens=8150)
    with pytest.raises(budget.GenerationContextOverflowError) as no_complete:
        budget.fit_output_tokens(
            2000, context_cap=8192, prompt_tokens=7000, require_full=True,
        )

    assert no_room.value.phase == budget.CAPACITY_PHASE_PROMPT_NO_ROOM
    assert no_complete.value.phase == budget.CAPACITY_PHASE_PROMPT_NO_ROOM


def test_the_phase_vocabulary_is_closed():
    with pytest.raises(ValueError, match="unknown capacity phase"):
        budget.PromptContextOverflowError("x", phase="output-limit")
    with pytest.raises(ValueError, match="unknown capacity phase"):
        budget.GenerationContextOverflowError("x", phase="")


@pytest.mark.parametrize(
    "error, expected",
    [
        pytest.param(None, False, id="none"),
        pytest.param(RuntimeError("x"), False, id="plain_runtime_error"),
        pytest.param(
            budget.PromptContextOverflowError("x"), False, id="default_phase",
        ),
        pytest.param(
            budget.GenerationContextOverflowError("x"), False, id="pre_call",
        ),
    ],
)
def test_the_predicate_says_no_by_default(error, expected):
    assert budget.is_rerollable_capacity_error(error) is expected


def test_the_predicate_says_yes_only_for_output_limit():
    assert budget.is_rerollable_capacity_error(
        _capacity(budget.CAPACITY_PHASE_OUTPUT_LIMIT)
    ) is True


def test_the_writer_re_export_is_the_same_object():
    """CONTROL: two definitions of one error would be two policies."""
    assert writer.PromptContextOverflowError is budget.PromptContextOverflowError
    assert budget.PromptContextOverflowError in ladder._ATTEMPT_ERRORS
    assert budget.GenerationContextOverflowError in ladder._ATTEMPT_ERRORS


def test_a_repair_that_ran_out_of_room_gets_the_last_swing():
    """FIRES on the SECOND gate -- the plan's 'there are two, patch both'.

    A content failure on the base call skips the structural rung, so the typed
    repair is attempt 2. When THAT call runs out of output room it produced no
    shape rather than a wrong one, and the repair-syntax rung re-sends the same
    repair prompt as attempt 3. With only the first gate patched this pass dies
    at the ladder instead.
    """
    calls = []
    verdicts = []

    def slot(messages, **kwargs):
        calls.append(kwargs.get("temperature"))
        if len(calls) == 2:
            raise _capacity(budget.CAPACITY_PHASE_OUTPUT_LIMIT)
        return GOOD

    def post_validator(_result):
        verdicts.append(len(verdicts) + 1)
        # Reject the base call on CONTENT so the structural rung is skipped.
        return "the first artifact is content-invalid" if len(verdicts) == 1 else None

    result = _run(slot, post_validator=post_validator)

    assert result.ok is True
    assert len(calls) == 3
