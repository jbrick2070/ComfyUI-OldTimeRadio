"""Shared context-window arithmetic for OTR text-generation transports.

The caller's ``max_new_tokens`` is an upper bound, not permission to spend
the whole context window after the prompt has already been serialized.  Every
transport uses this helper after it has measured the prompt it will send.
"""
from __future__ import annotations

from typing import Any


MIN_OUTPUT_TOKENS = 64


class GenerationContextOverflowError(RuntimeError):
    """The prompt leaves no honest room for a usable artifact."""


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
            f"but at least {minimum} are required"
        )
    if require_full and requested_tokens > available:
        raise GenerationContextOverflowError(
            f"{label} cannot fit the complete requested output: prompt requires "
            f"{prompt} input tokens, requested_output={requested_tokens}, "
            f"context_cap={cap} leaves only {available} output tokens"
        )
    return min(requested_tokens, available)
