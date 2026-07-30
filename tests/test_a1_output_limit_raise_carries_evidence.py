"""A-1: the output-limit raise carries the completion and the arithmetic.

The live 45-word campaign (2026-07-29) killed three legs with
`PromptContextOverflowError: prose generation exhausted the full remaining
provider/context capacity (14697 output tokens after a 1687-token prompt)`.
Nobody could say what the model had actually written, because the raise fired
ABOVE the `tokenizer.decode` -- the completion existed only as token ids that
went out of scope with the frame. The OUTPUT_TRUNCATED / OUTPUT_CAP arithmetic
sat below the raise too, so the one leg that most needed it logged nothing.

This file pins the wiring, not the arithmetic (`test_generation_budget.py`
owns that):

  FIRES         -- the raise carries raw_completion plus every token count.
  FIRES         -- the decode really runs BEFORE the raise (call counter).
  FIRES         -- the truncation arithmetic is logged on the dying leg.
  COUNTERFACTUAL-- the completion is NOT in the message: str(exc) and
                   exc.args stay exactly what they were, because a
                   14,000-token artifact in an exception string floods every
                   log that formats it.
  CONTROL       -- a leg that ends with EOS at the cap still returns its text.
  CONTROL       -- a caller that did not opt in to fail-on-output-limit is
                   untouched.
  CONTROL       -- the prompt-side re-wrap reports None, never a guess.

No GPU, no network. UTF-8, no BOM, ASCII-only source.
"""
from __future__ import annotations

import logging

import pytest

from nodes import OTR_LedgerScriptWriter as writer
from nodes import _otr_loader_backends as loader_backends


CONTEXT_CAP = 8192
PROMPT_TOKENS = 7000
REQUESTED_OUTPUT = 2000
# fit_output_tokens(2000, context_cap=8192, prompt_tokens=7000) == 1192
FITTED_OUTPUT = 1192
COMPLETION = '{"lines": [{"line_id": "l001", "text": "' + ("x" * 4000)


class _FailOnOutputLimitMessages(list):
    _otr_fail_on_output_limit = True


class _PromptTensor:
    shape = (1, PROMPT_TOKENS)


class _GeneratedIds:
    """The generated tail, as the real slice behaves for our purposes.

    ``__len__`` is not decoration: the production line reads
    ``getattr(generated_ids, "shape", [len(generated_ids)])``, and Python
    builds that default EAGERLY, so a stand-in without ``__len__`` raises
    inside the surrounding ``except Exception`` and silently reports
    ``generated_tokens = None`` -- which looks exactly like the raise not
    being wired. A real tensor has both; so does this.
    """

    def __init__(self, count: int, last_token: int) -> None:
        self.shape = (count,)
        self._last_token = last_token

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, key):
        if key == -1:
            return self._last_token
        raise AssertionError("only the final token is read here")


class _Output:
    def __init__(self, generated: _GeneratedIds) -> None:
        self._generated = generated

    def __getitem__(self, key):
        if isinstance(key, slice):
            assert key.start == PROMPT_TOKENS
            return self._generated
        raise AssertionError("the batch row is taken whole")


class _Inputs(dict):
    def __init__(self) -> None:
        super().__init__(input_ids=_PromptTensor())

    def to(self, _device):
        return self


class _Tokenizer:
    eos_token_id = 0

    def __init__(self) -> None:
        self.decode_calls = 0

    def apply_chat_template(self, _messages, **_kwargs):
        return "serialized prompt"

    def __call__(self, _prompt, *, return_tensors):
        assert return_tensors == "pt"
        return _Inputs()

    def decode(self, _tokens, *, skip_special_tokens):
        assert skip_special_tokens is True
        self.decode_calls += 1
        return COMPLETION


class _Model:
    device = "cpu"

    def __init__(self, generated: _GeneratedIds) -> None:
        self._generated = generated

    def generate(self, **_kwargs):
        return [_Output(self._generated)]


def _generate_fn(monkeypatch, *, count=FITTED_OUTPUT, last_token=5):
    monkeypatch.setattr(
        loader_backends, "tokenizer_supports_system_role", lambda _tokenizer: True,
    )
    tokenizer = _Tokenizer()
    generated = _GeneratedIds(count, last_token)
    return writer._build_truncating_generate_fn({
        "model": _Model(generated),
        "tokenizer": tokenizer,
        "context_cap": CONTEXT_CAP,
    }), tokenizer


def test_output_limit_raise_carries_the_completion_and_the_arithmetic(monkeypatch):
    generate, tokenizer = _generate_fn(monkeypatch)

    with pytest.raises(writer.PromptContextOverflowError) as caught:
        generate(
            _FailOnOutputLimitMessages([{"role": "user", "content": "45 words"}]),
            temperature=.2,
            max_new_tokens=REQUESTED_OUTPUT,
        )

    exc = caught.value
    # FIRES: the decode ran, and it ran BEFORE the raise. With the decode back
    # below the raise this counter is 0 and every field below is None.
    assert tokenizer.decode_calls == 1
    assert exc.raw_completion == COMPLETION
    assert exc.prompt_tokens == PROMPT_TOKENS
    assert exc.generated_tokens == FITTED_OUTPUT
    assert exc.requested_output_tokens == REQUESTED_OUTPUT
    assert exc.effective_output_tokens == FITTED_OUTPUT
    assert exc.context_cap == CONTEXT_CAP
    assert exc.ended_with_eos is False


def test_the_completion_never_reaches_the_message(monkeypatch):
    """COUNTERFACTUAL for the flood guard: evidence by field, not by string."""
    generate, _tokenizer = _generate_fn(monkeypatch)

    with pytest.raises(writer.PromptContextOverflowError) as caught:
        generate(
            _FailOnOutputLimitMessages([{"role": "user", "content": "45 words"}]),
            temperature=.2,
            max_new_tokens=REQUESTED_OUTPUT,
        )

    exc = caught.value
    message = str(exc)
    assert COMPLETION not in message
    assert "xxxx" not in message
    assert exc.args == (message,)
    assert "prose generation exhausted the full remaining provider/context" in message
    assert f"({FITTED_OUTPUT} output tokens after a {PROMPT_TOKENS}-token prompt)" in message


def test_the_dying_leg_logs_its_own_truncation_arithmetic(monkeypatch, caplog):
    """FIRES: the raise used to jump over this log, on the only leg that needed it."""
    generate, _tokenizer = _generate_fn(monkeypatch)

    with caplog.at_level(logging.ERROR, logger=writer.log.name):
        with pytest.raises(writer.PromptContextOverflowError):
            generate(
                _FailOnOutputLimitMessages([{"role": "user", "content": "45 words"}]),
                temperature=.2,
                max_new_tokens=REQUESTED_OUTPUT,
            )

    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert "OUTPUT_TRUNCATED" in logged
    assert str(PROMPT_TOKENS) in logged
    assert str(FITTED_OUTPUT) in logged


def test_a_leg_that_ends_with_eos_at_the_cap_still_returns_its_text(monkeypatch):
    """CONTROL: the raise condition is unchanged -- EOS at the cap is a finish."""
    generate, tokenizer = _generate_fn(monkeypatch, last_token=_Tokenizer.eos_token_id)

    result = generate(
        _FailOnOutputLimitMessages([{"role": "user", "content": "45 words"}]),
        temperature=.2,
        max_new_tokens=REQUESTED_OUTPUT,
    )

    assert result == COMPLETION
    assert tokenizer.decode_calls == 1


def test_a_caller_that_did_not_opt_in_is_untouched(monkeypatch):
    """CONTROL: no fail-on-output-limit marker, no raise, one decode."""
    generate, tokenizer = _generate_fn(monkeypatch)

    result = generate(
        [{"role": "user", "content": "45 words"}],
        temperature=.2,
        max_new_tokens=REQUESTED_OUTPUT,
    )

    assert result == COMPLETION
    assert tokenizer.decode_calls == 1


def test_the_prompt_side_rewrap_reports_none_rather_than_a_guess():
    """CONTROL: the other raise site knows no counts, so it claims none."""
    exc = writer.PromptContextOverflowError("cannot fit")

    assert str(exc) == "cannot fit"
    assert exc.raw_completion is None
    assert exc.prompt_tokens is None
    assert exc.generated_tokens is None
    assert exc.requested_output_tokens is None
    assert exc.effective_output_tokens is None
    assert exc.context_cap is None
    assert exc.ended_with_eos is None
