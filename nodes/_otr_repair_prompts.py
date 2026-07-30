"""Typed repair prompts for machine-readable structured output.

Repairs are limited to JSON syntax, schema shape/type, null payloads, and
membership in an authoritative locked roster. Authored prose is never repaired
because of word count, vocabulary, names, style, visual content, or quality.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Optional

from pydantic import BaseModel, ValidationError

# Shared structured-call surface. Package import in production; flat
# import when this module is loaded standalone / under test. Mirrors
# the import guard used by every other structured-pass module.
try:
    from ._otr_structured_call import (
        PostValidationError,
        RepairPromptFactory,
        _prompt_to_text,
        default_repair_prompt_factory,
    )
except ImportError:  # pragma: no cover - standalone / test load
    from _otr_structured_call import (  # type: ignore
        PostValidationError,
        RepairPromptFactory,
        _prompt_to_text,
        default_repair_prompt_factory,
    )


__all__ = [
    "json_syntax_repair",
    "schema_field_repair",
    "cast_membership_repair",
    "payload_null_repair",
    "make_dispatching_repair_factory",
]


# How many characters of the failed model output to echo back into the
# repair prompt. Matches `default_repair_prompt_factory` so every
# repair turn -- generic or typed -- shows the model the same slice of
# what it got wrong.
_FAILED_OUTPUT_ECHO_CHARS: int = 400


# ---------------------------------------------------------------------------
# Internal helpers -- shared message assembly
# ---------------------------------------------------------------------------


def _error_text(error: BaseException) -> str:
    """Render an exception as a single `TypeName: message` string."""
    return f"{type(error).__name__}: {error}"


def _compose_repair(
    directive: str,
    *,
    failed_output: str,
    original_prompt: Any,
) -> list[dict[str, str]]:
    """Assemble a typed-repair message from a class-specific directive.

    Every typed factory shares the same shape: a `CRITICAL:` directive
    that names the failure class and tells the model exactly how to
    fix it, then the truncated failed output, then the original
    instruction restated so the repair turn is self-contained. Only the
    `directive` differs between classes. This mirrors the structure of
    `default_repair_prompt_factory` so the typed factories stay
    drop-in compatible with it.
    """
    original_text = _prompt_to_text(original_prompt)
    body = (
        directive.rstrip()
        + "\n\n"
        + f"Failed response: {failed_output[:_FAILED_OUTPUT_ECHO_CHARS]}\n\n"
        + "Original instruction follows.\n\n"
        + original_text
    )
    return [{"role": "user", "content": body}]


# ---------------------------------------------------------------------------
# Typed structural repair-prompt factories
# ---------------------------------------------------------------------------
#
# Each matches the `RepairPromptFactory` Protocol exactly:
#   factory(*, original_prompt, failed_output, error) -> messages
# and returns a single-element `user` messages list.


def json_syntax_repair(
    *,
    original_prompt: Any,
    failed_output: str,
    error: BaseException,
) -> list[dict[str, str]]:
    """Repair prompt for a `json.JSONDecodeError` -- unparseable output."""
    directive = (
        "CRITICAL: Your previous response was not valid JSON "
        f"({_error_text(error)}). Return ONE valid JSON object and "
        "nothing else: no Markdown code fences, no commentary, no text "
        "before or after the object. Begin your response with '{' and "
        "end it with '}'. Every string must use double quotes and every "
        "brace and bracket must be balanced."
    )
    return _compose_repair(
        directive, failed_output=failed_output, original_prompt=original_prompt,
    )


def schema_field_repair(
    *,
    original_prompt: Any,
    failed_output: str,
    error: BaseException,
) -> list[dict[str, str]]:
    """Repair prompt for a `pydantic.ValidationError` -- a bad field."""
    directive = (
        "CRITICAL: Your previous response was valid JSON but failed "
        f"schema validation: {_error_text(error)}. Fix ONLY the "
        "field(s) named in that error -- correct the type, the value "
        "range, or supply the missing field. Keep every other field "
        "exactly as it was. Return ONE valid JSON object, no Markdown, "
        "no prose."
    )
    return _compose_repair(
        directive, failed_output=failed_output, original_prompt=original_prompt,
    )


def cast_membership_repair(
    *,
    original_prompt: Any,
    failed_output: str,
    error: BaseException,
) -> list[dict[str, str]]:
    """Repair prompt for a speaker / name outside the locked cast.

    This is the LLM fallback used only when the deterministic
    Levenshtein remap could not resolve the phantom unambiguously (or
    when the call site supplied no deterministic callback). The
    rejection error already quotes the full locked-cast list, so the
    directive points the model straight at it.
    """
    directive = (
        "CRITICAL: Your previous response named a speaker or character "
        f"that is not in the locked cast: {_error_text(error)}. The "
        "cast is FIXED -- you may not invent, rename, or add anyone. "
        "Use ONLY the exact names listed in that error. Replace the "
        "invalid name with the locked-cast member you intended. Return "
        "ONE valid JSON object, no Markdown, no prose."
    )
    return _compose_repair(
        directive, failed_output=failed_output, original_prompt=original_prompt,
    )


def payload_null_repair(
    *,
    original_prompt: Any,
    failed_output: str,
    error: BaseException,
) -> list[dict[str, str]]:
    """Repair prompt for a `payload: null` pydantic rejection.

    Sprint 7C / BUG-LOCAL-275. The Script Doctor edits-pass schema
    requires `payload: str` (`ReviewerEdit.payload`), but the technical-
    slot model keeps emitting `payload: null` on no-op / annotation-
    only edit rows. The generic `schema_field_repair` directive ("fix
    the field named in the error") was too vague to recover -- the
    model kept re-emitting the same null. This directive is explicit:
    every `payload` MUST be a non-null replacement string; if the
    model has no replacement text, it must DROP the entire edit row.
    """
    directive = (
        "CRITICAL: Your previous response had a `payload` field set to "
        f"null, which is not allowed: {_error_text(error)}. Every edit "
        "row's `payload` field MUST be a non-null string containing the "
        "actual replacement text. If you have no replacement text for a "
        "line, OMIT the entire edit row -- do not emit it with "
        "`payload: null`, `payload: \"\"`, or any other placeholder. "
        "Return ONE valid JSON object, no Markdown, no prose."
    )
    return _compose_repair(
        directive, failed_output=failed_output, original_prompt=original_prompt,
    )


# ---------------------------------------------------------------------------
# Dispatcher -- routes an Attempt-3 failure to the matching typed factory
# ---------------------------------------------------------------------------


# Sprint 7C / BUG-LOCAL-275: pydantic v2 ValidationError repr for a
# null-where-string-required failure includes BOTH the offending field
# path and the `input_value=None` annotation, e.g.:
#
#   edits.2.payload
#     Input should be a valid string [type=string_type,
#     input_value=None, input_type=NoneType]
#
# The dispatcher matches on the lowered string. `payload` + the
# `input_value=none` annotation is specific enough that no other
# ValidationError in the project's schemas collides with it (no other
# pydantic field in the structured-call call graph is named `payload`,
# and `input_value=none` only appears when the model literally sent
# `null`). If pydantic ever changes the repr we lose the typed route
# and fall through to `schema_field_repair` -- behaviour-degraded but
# never broken.
_PAYLOAD_NULL_FIELD_TOKEN: str = "payload"
_PAYLOAD_NULL_VALUE_TOKEN: str = "input_value=none"


def _is_payload_null_validation_error(error: BaseException) -> bool:
    """True iff `error` is a pydantic ValidationError whose repr names
    a null `payload` field. See `_PAYLOAD_NULL_*` constants above for
    the signal we match on.
    """
    if not isinstance(error, ValidationError):
        return False
    text = str(error).lower()
    return (
        _PAYLOAD_NULL_FIELD_TOKEN in text
        and _PAYLOAD_NULL_VALUE_TOKEN in text
    )


# A deterministic-repair callback: given the raw failed output and the
# rejecting error, either return a finished pydantic instance (the
# failure was resolved with no LLM call) or None (fall through to an
# LLM repair turn).
DeterministicRepair = Callable[[str, BaseException], Optional[BaseModel]]


def make_dispatching_repair_factory(
    *,
    deterministic_repair: Optional[DeterministicRepair] = None,
) -> RepairPromptFactory:
    """Build a repair factory for structural machine-output failures.

    JSON syntax, pydantic schema, null payload, and locked-roster membership
    have typed repair prompts. A caller-supplied deterministic repair is tried
    first on ANY post-validation failure and may short-circuit the model
    entirely. Every other post-validation error uses the generic structural
    repair prompt; prose semantics never select a repair path.
    """

    def factory(
        *,
        original_prompt: Any,
        failed_output: str,
        error: BaseException,
    ) -> Any:
        # json.JSONDecodeError is a subclass of ValueError, so it must
        # be tested before the PostValidationError (also a ValueError)
        # branch.
        if isinstance(error, json.JSONDecodeError):
            return json_syntax_repair(
                original_prompt=original_prompt,
                failed_output=failed_output,
                error=error,
            )
        if isinstance(error, ValidationError):
            # Sprint 7C / BUG-LOCAL-275: a null `payload` rejection
            # gets a dedicated directive; the generic field-repair
            # prompt did not recover the Script Doctor edits pass.
            if _is_payload_null_validation_error(error):
                return payload_null_repair(
                    original_prompt=original_prompt,
                    failed_output=failed_output,
                    error=error,
                )
            return schema_field_repair(
                original_prompt=original_prompt,
                failed_output=failed_output,
                error=error,
            )
        if isinstance(error, PostValidationError):
            # THE DETERMINISTIC RUNG IS NOT A P2 FEATURE, though it shipped as
            # one. Until 2026-07-29 this attempt lived INSIDE the "locked
            # cast" prose test below, so a caller-supplied deterministic
            # repair was undispatchable for every other pass. P0's literal-
            # source repairer was therefore unreachable even once wired --
            # its errors say "non-literal source span", never "locked cast".
            #
            # Selecting a repair MECHANISM by matching the wording of an error
            # message also contradicts this factory's own docstring ("prose
            # semantics never select a repair path"). The prose test below now
            # chooses only the fallback PROMPT, which is all it was ever fit
            # for.
            #
            # Trying the deterministic repair for ANY PostValidationError is
            # safe by construction: it is fail-closed (returns None when it
            # cannot help), and structured_call still puts whatever it returns
            # through the pass's real post_validator before accepting it. A
            # pass that supplied no deterministic repair is unaffected.
            if deterministic_repair is not None:
                resolved = deterministic_repair(failed_output, error)
                if resolved is not None:
                    # structured_call accepts a schema instance as
                    # a finished result -- no LLM repair call.
                    return resolved
            message = str(error).lower()
            if "locked cast" in message:
                return cast_membership_repair(
                    original_prompt=original_prompt,
                    failed_output=failed_output,
                    error=error,
                )
        # Unrecognised content failure, or a non-ladder error type:
        # the generic CRITICAL-prefix repair turn is the right
        # fallback.
        return default_repair_prompt_factory(
            original_prompt=original_prompt,
            failed_output=failed_output,
            error=error,
        )

    return factory
