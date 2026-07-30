"""Shared context-window arithmetic for OTR text-generation transports.

The caller's ``max_new_tokens`` is an upper bound, not permission to spend
the whole context window after the prompt has already been serialized.  Every
transport uses this helper after it has measured the prompt it will send.
"""
from __future__ import annotations

from typing import Any


MIN_OUTPUT_TOKENS = 64
class ProviderCapacityMessages(list):
    """Message payload for prose whose size must never be pre-judged.

    Transports reserve their full remaining provider/context capacity. If the
    provider consumes that capacity before reaching its own stop condition, the
    call fails as a capacity defect and the caller must not reinterpret the
    partial artifact as bad prose or feed it into a retry ladder.
    """

    _otr_prompt_must_fit = True
    _otr_output_budget_mode = "provider_capacity"
    _otr_reserve_remaining_output_capacity = True
    _otr_fail_on_output_limit = True
    _otr_strict_remote_output_budget = True


# A-4 (2026-07-30, writer repair): a capacity failure has a PHASE, and the
# phase decides whether a retry is honest. Both spellings live here because
# this module owns capacity arithmetic; the retry ladder asks this module
# rather than keeping its own opinion.
#
#   prompt_no_room -- refused BEFORE the call: the measured prompt leaves less
#                     room than the artifact needs. Deterministic. Re-rolling
#                     re-derives the identical refusal, so it never retries.
#   output_limit   -- the call RAN and consumed its whole output allowance
#                     without reaching a stop condition. Sampling is
#                     stochastic (the live 45-word campaign had nine engines
#                     produce both a pass and a fail on byte-identical code),
#                     so a re-roll at a lower temperature is a real second
#                     chance, not a wasted call.
CAPACITY_PHASE_PROMPT_NO_ROOM = "prompt_no_room"
CAPACITY_PHASE_OUTPUT_LIMIT = "output_limit"
CAPACITY_PHASES = (CAPACITY_PHASE_PROMPT_NO_ROOM, CAPACITY_PHASE_OUTPUT_LIMIT)


class _CapacityError(RuntimeError):
    """Shared phase contract for the two capacity failures.

    ``args`` stays ``(message,)`` so ``str(exc)`` and every existing raise
    site are unchanged. An unknown phase is a coding error and fails loudly
    here rather than silently becoming un-retryable at the ladder.
    """

    def __init__(
        self,
        message: str,
        *,
        phase: str = CAPACITY_PHASE_PROMPT_NO_ROOM,
    ) -> None:
        super().__init__(message)
        if phase not in CAPACITY_PHASES:
            raise ValueError(
                f"unknown capacity phase {phase!r}; expected one of "
                f"{CAPACITY_PHASES}"
            )
        self.phase = phase


class GenerationContextOverflowError(_CapacityError):
    """The prompt leaves no honest room for a usable artifact."""


class PromptContextOverflowError(_CapacityError):
    """A transport could not honour its caller's output contract.

    Moved here from ``OTR_LedgerScriptWriter`` by A-4 so the retry ladder --
    which is documented pure and may not import the writer -- can name the
    type it is deciding about. The writer re-exports it, so
    ``writer.PromptContextOverflowError`` is the same object it always was.

    A-1 (2026-07-30): the output-limit raise carries the completion the model
    actually produced plus the token arithmetic, as FIELDS. Never in the
    message -- a ~14,000-token artifact inside an exception string floods
    every log and receipt that formats it, and the ladder's disposition lines
    are read by humans. A caller that wants the evidence asks for it by name.

    Every field is optional: the prompt-side re-wrap knows none of them, and
    reports ``None`` rather than a guess.
    """

    def __init__(
        self,
        message: str,
        *,
        phase: str = CAPACITY_PHASE_PROMPT_NO_ROOM,
        raw_completion: str | None = None,
        prompt_tokens: int | None = None,
        generated_tokens: int | None = None,
        requested_output_tokens: int | None = None,
        effective_output_tokens: int | None = None,
        context_cap: int | None = None,
        ended_with_eos: bool | None = None,
    ) -> None:
        super().__init__(message, phase=phase)
        self.raw_completion = raw_completion
        self.prompt_tokens = prompt_tokens
        self.generated_tokens = generated_tokens
        self.requested_output_tokens = requested_output_tokens
        self.effective_output_tokens = effective_output_tokens
        self.context_cap = context_cap
        self.ended_with_eos = ended_with_eos


CAPACITY_ERRORS = (GenerationContextOverflowError, PromptContextOverflowError)


def is_rerollable_capacity_error(error: Any) -> bool:
    """Return True only for a capacity failure a re-roll could actually fix.

    ONE predicate, one owner: the transports raise the phase and the ladder
    asks this. A `prompt_no_room` failure answers False forever -- the
    arithmetic that refused it is deterministic.
    """
    return (
        isinstance(error, CAPACITY_ERRORS)
        and getattr(error, "phase", None) == CAPACITY_PHASE_OUTPUT_LIMIT
    )


def estimate_prompt_tokens(messages: Any) -> int:
    """Return a conservative provider-independent prompt-token estimate.

    Local Transformers callers use their real tokenizer instead.  Remote
    providers do not expose a tokenizer here, so the existing four-characters
    per token estimate is retained as a conservative preflight estimate.
    """
    if not isinstance(messages, (list, tuple)):
        text = str(messages or "")
        return max(1, (len(text) + 3) // 4)
    chars = 0
    for message in messages:
        if isinstance(message, dict):
            chars += len(str(message.get("content", "")))
        else:
            chars += len(str(message))
    return max(1, (chars + 3) // 4)


def fit_output_tokens(
    requested: int,
    *,
    context_cap: int,
    prompt_tokens: int,
    min_output_tokens: int = MIN_OUTPUT_TOKENS,
    label: str = "generation",
    require_full: bool = False,
) -> int:
    """Clamp output to the room left by the measured prompt.

    A prompt that cannot leave the minimum output room fails before the
    provider call.  A larger requested ceiling is normally reduced to the
    remaining room; this prevents the old ``context_cap - requested`` math
    from deleting prompt context.  ``require_full`` is the opt-in contract for
    bounded patches whose complete requested output must fit or make no call.
    Unmarked callers preserve the historical clamping behavior.
    """
    cap = int(context_cap)
    prompt = int(prompt_tokens)
    minimum = max(1, int(min_output_tokens))
    requested_tokens = max(1, int(requested))
    available = cap - prompt
    if cap <= 0 or prompt >= cap or available < minimum:
        raise GenerationContextOverflowError(
            f"{label} cannot fit: prompt requires {prompt} input tokens, "
            f"context_cap={cap} leaves {max(0, available)} output tokens "
            f"but at least {minimum} are required",
            phase=CAPACITY_PHASE_PROMPT_NO_ROOM,
        )
    if require_full and requested_tokens > available:
        raise GenerationContextOverflowError(
            f"{label} cannot fit the complete requested output: prompt requires "
            f"{prompt} input tokens, requested_output={requested_tokens}, "
            f"context_cap={cap} leaves only {available} output tokens",
            # Still PRE-call and still deterministic: the prompt is the reason
            # the complete artifact has no room. A re-roll cannot change it.
            phase=CAPACITY_PHASE_PROMPT_NO_ROOM,
        )
    return min(requested_tokens, available)
