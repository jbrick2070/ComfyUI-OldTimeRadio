"""nodes/_otr_repair_prompts.py -- typed repair-prompt factories (Sprint 2C).

`structured_call`'s Attempt 3 (typed repair) builds its repair prompt
through a `RepairPromptFactory`. Sprint 2A shipped exactly one generic
factory, `default_repair_prompt_factory`, which every converted call
site used. This module ships the SIX bespoke per-failure-class
factories the v4 plan (section 2C) calls for, plus a dispatcher that
routes an Attempt-3 failure to the right one by inspecting the error.

The six failure classes:

  json_syntax_repair     -- json.JSONDecodeError: output was not
                            parseable JSON at all.
  schema_field_repair    -- pydantic.ValidationError: JSON parsed, but
                            a field is missing / wrong type / out of
                            range.
  cast_membership_repair -- a speaker / character named outside the
                            locked cast.
  too_many_words_repair  -- output ran over a length budget.
  narration_leak_repair  -- action / dialogue / plot prose leaked into
                            a description meant to be purely visual.
  forbidden_name_repair  -- a character name appears where the pass is
                            forbidden to name characters.

`cast_membership_repair` is special. The v4 plan requires it to NEVER
call the LLM when the project's existing Levenshtein matcher resolves
the phantom name deterministically. A `RepairPromptFactory` cannot do
that on its own -- it only builds a prompt -- so the work is split:

  * This module owns the LLM-prompt half (`cast_membership_repair`) and
    the dispatch.
  * The call site that has a locked cast (the outline phase stage)
    supplies a `deterministic_repair` callback to
    `make_dispatching_repair_factory`. The callback reuses the
    project's one Levenshtein matcher -- `auto_remap_phantom` in
    `_otr_ledger_reviewer.py`, threshold 3 -- and, when every phantom
    resolves unambiguously, returns the corrected pydantic instance.

`structured_call` accepts a schema instance handed back by a repair
factory as a finished result and returns it with NO LLM repair call.
That is the "never calls the LLM if Levenshtein resolves" path.

This module is PURE: no I/O, no GPU, no ComfyUI imports. It depends
only on stdlib, pydantic, and the sibling `_otr_structured_call`
helper. It does NOT import `_otr_ledger_reviewer` -- the Levenshtein
reuse lives in the call-site callback, which keeps this module free of
the node-side import graph.

UTF-8 no BOM. No em-dashes (Windows cp1252 subprocess decode trap).
4-space indentation.
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
    "too_many_words_repair",
    "narration_leak_repair",
    "forbidden_name_repair",
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
# The six typed repair-prompt factories
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


def too_many_words_repair(
    *,
    original_prompt: Any,
    failed_output: str,
    error: BaseException,
) -> list[dict[str, str]]:
    """Repair prompt for output that ran over a length budget."""
    directive = (
        "CRITICAL: Your previous response was over the length budget: "
        f"{_error_text(error)}. Rewrite it SHORTER -- cut words, drop "
        "every clause that is not essential, and keep the core meaning. "
        "Do not add new content. Return ONE valid JSON object, no "
        "Markdown, no prose."
    )
    return _compose_repair(
        directive, failed_output=failed_output, original_prompt=original_prompt,
    )


def narration_leak_repair(
    *,
    original_prompt: Any,
    failed_output: str,
    error: BaseException,
) -> list[dict[str, str]]:
    """Repair prompt for action / dialogue / plot prose in a visual brief."""
    directive = (
        "CRITICAL: Your previous response leaked narration into a brief "
        f"that must be purely visual: {_error_text(error)}. Describe "
        "ONLY what is visually present -- light, color, texture, space, "
        "material, weather. Remove every action, every line of "
        "dialogue, and every plot verb (no speaking, arguing, "
        "discovering, escaping, and the like). Return ONE valid JSON "
        "object, no Markdown, no prose."
    )
    return _compose_repair(
        directive, failed_output=failed_output, original_prompt=original_prompt,
    )


def forbidden_name_repair(
    *,
    original_prompt: Any,
    failed_output: str,
    error: BaseException,
) -> list[dict[str, str]]:
    """Repair prompt for a character name where naming is forbidden."""
    directive = (
        "CRITICAL: Your previous response named a character, but this "
        f"pass must not name anyone: {_error_text(error)}. Remove the "
        "name entirely. Refer to any person only with a generic noun -- "
        "a figure, a man, a woman, a child, a passer-by. Return ONE "
        "valid JSON object, no Markdown, no prose."
    )
    return _compose_repair(
        directive, failed_output=failed_output, original_prompt=original_prompt,
    )


# ---------------------------------------------------------------------------
# Dispatcher -- routes an Attempt-3 failure to the matching typed factory
# ---------------------------------------------------------------------------


# A deterministic-repair callback: given the raw failed output and the
# rejecting error, either return a finished pydantic instance (the
# failure was resolved with no LLM call) or None (fall through to an
# LLM repair turn).
DeterministicRepair = Callable[[str, BaseException], Optional[BaseModel]]


def make_dispatching_repair_factory(
    *,
    deterministic_repair: Optional[DeterministicRepair] = None,
) -> RepairPromptFactory:
    """Build a `RepairPromptFactory` that dispatches by failure class.

    The returned callable inspects the Attempt-3 error and routes:

      * `json.JSONDecodeError`      -> `json_syntax_repair`
      * `pydantic.ValidationError`  -> `schema_field_repair`
      * `PostValidationError`       -> classified by its message:
            "locked cast"                  -> cast membership
            "named_character"              -> forbidden name
            "dialogue_verb" / "plot_verb"  -> narration leak
            "too_long"                     -> too many words
      * anything else / unrecognised content failure
                                    -> `default_repair_prompt_factory`

    `deterministic_repair`, when supplied, is consulted for the cast-
    membership case BEFORE the LLM prompt is built. If it returns a
    pydantic instance, that instance is returned straight to
    `structured_call`, which accepts it as a finished result and makes
    no LLM repair call. If it returns None, the LLM
    `cast_membership_repair` prompt is used instead. Call sites with no
    locked cast omit the argument entirely.

    The message-substring dispatch is deliberate: the project's
    `post_validator` functions return short, stable rejection codes
    (`named_character`, `dialogue_verb`, `too_long`) or quoted phrases
    (`is not in locked cast [...]`), and a content failure that matches
    none of them is correctly handled by the generic default factory.
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
            return schema_field_repair(
                original_prompt=original_prompt,
                failed_output=failed_output,
                error=error,
            )
        if isinstance(error, PostValidationError):
            message = str(error).lower()
            if "locked cast" in message:
                if deterministic_repair is not None:
                    resolved = deterministic_repair(failed_output, error)
                    if resolved is not None:
                        # structured_call accepts a schema instance as
                        # a finished result -- no LLM repair call.
                        return resolved
                return cast_membership_repair(
                    original_prompt=original_prompt,
                    failed_output=failed_output,
                    error=error,
                )
            if "named_character" in message:
                return forbidden_name_repair(
                    original_prompt=original_prompt,
                    failed_output=failed_output,
                    error=error,
                )
            if "dialogue_verb" in message or "plot_verb" in message:
                return narration_leak_repair(
                    original_prompt=original_prompt,
                    failed_output=failed_output,
                    error=error,
                )
            if "too_long" in message:
                return too_many_words_repair(
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
