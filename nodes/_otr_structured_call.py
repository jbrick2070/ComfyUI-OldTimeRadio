"""nodes/_otr_structured_call.py -- shared structured-JSON LLM call helper.

Sprint 2A step 1. Centralises the retry ladder that every structured-
JSON LLM pass in the project should use. Today each structured pass
(story brief reflection, cast contract, critic, news interpreter)
hand-rolls its own "call -> parse -> validate -> repair once" sequence
with subtly different temperature math. This module is the single
home for that ladder.

Step 1 ships the MODULE + TESTS only. No existing call site is
converted here -- migrating `run_story_brief_reflection` and the other
structured passes onto `structured_call` is later, separate work
(Sprint 2B onward) and is explicitly out of scope.

The "2B principle" on temperature: raising entropy during a JSON-
schema repair encourages MORE structural hallucination, not less. So
the structural retry (Attempt 2) LOWERS temperature relative to the
base attempt, and the typed repair (Attempt 3) runs at a static low
temperature -- the ladder never raises temperature to "shake loose"
a valid object.

Slot-fn calling convention is read from `nodes/_otr_story_brief.py`
(`run_story_brief_reflection`, `_repair_pass`): a slot fn has the
signature `slot_fn(messages, *, temperature, max_new_tokens[, stop])`
and returns a raw string. JSON extraction reuses the shared
`_otr_json.parse_first_json_object` -- this module does NOT hand-roll
a second JSON parser.

This module is PURE: no I/O beyond an optional grammar-path existence
probe, no GPU, no ComfyUI imports, no writer imports.

UTF-8 no BOM. No em-dashes (Windows cp1252 subprocess decode trap).
4-space indentation.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable, Optional, Protocol, TypeVar

from pydantic import BaseModel, ValidationError


log = logging.getLogger("OTR")


# Shared JSON extraction. Package import in production; flat import
# when the module is loaded standalone / under test. Mirrors the
# import guard in `_otr_story_brief.py`.
try:
    from . import _otr_json
except ImportError:  # pragma: no cover - standalone / test load
    import _otr_json  # type: ignore


# `T` is the validated-instance type the caller gets back: whatever
# pydantic model class they pass as `schema`.
T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------------------
# Constants -- ladder settings shared by every structured pass
# ---------------------------------------------------------------------------

# Default cap on the retry ladder. Attempt 4 (grammar-enforced) only
# runs when a `grammar_path` is supplied AND `max_attempts >= 4`.
_DEFAULT_MAX_ATTEMPTS: int = 4

# Token budget for a structured-JSON pass. A JSON object plus a short
# payload fits comfortably here; structured passes are not narrative
# generation. Matches the order of magnitude used by the story-brief
# reflection pass.
_STRUCTURED_MAX_NEW_TOKENS: int = 512

# Attempt 3 (typed repair) runs at this STATIC low temperature. Per the
# 2B principle the repair pass does not raise temperature -- a repair
# is a "follow the schema exactly" instruction, and low entropy is what
# makes a local model comply. Deliberately below any sane base
# temperature so the repair attempt is the calmest attempt in the
# ladder.
_REPAIR_TEMPERATURE: float = 0.10

# Attempt 4 (grammar-enforced) runs at this temperature. With a GBNF
# grammar constraining the token stream the structural shape is
# guaranteed by the decoder, so temperature only affects content; a
# low value keeps content honest.
_GRAMMAR_TEMPERATURE: float = 0.10


# ---------------------------------------------------------------------------
# Public exception -- fail-loud terminal state of the ladder
# ---------------------------------------------------------------------------


class StructuredCallFailedError(RuntimeError):
    """Raised when the retry ladder is exhausted with no schema-valid result.

    Carries the `helper_name` (so a log reader can attribute the
    failure to a specific structured pass), the number of attempts
    actually run, and the last validation / parse error observed. The
    helper NEVER returns a silent sentinel -- an exhausted ladder is a
    hard failure the caller must handle.
    """

    def __init__(
        self,
        *,
        helper_name: str,
        attempts: int,
        last_error: Optional[BaseException],
    ) -> None:
        self.helper_name = helper_name
        self.attempts = attempts
        self.last_error = last_error
        last_error_text = (
            f"{type(last_error).__name__}: {last_error}"
            if last_error is not None
            else "no error captured"
        )
        super().__init__(
            f"[OTR_StructuredCall] '{helper_name}' failed after "
            f"{attempts} attempt(s); last error -> {last_error_text}"
        )


class PostValidationError(ValueError):
    """Raised inside the ladder when `post_validator` rejects a result.

    A response can be JSON-parseable AND schema-valid yet still wrong:
    a casting pick may name a voice preset outside the runtime pool, a
    news brief may carry too few key terms, a story brief may mention a
    character name the brief is forbidden to name, an outline stage may
    route a line to a speaker outside the locked cast. Those are
    CONTENT failures, not structural ones -- pydantic cannot see them
    because they depend on runtime state the schema does not carry.

    `post_validator` reports such a failure by returning an error
    string (see `structured_call`); the ladder wraps that string in
    this exception. It subclasses `ValueError` so the ladder's existing
    `except (json.JSONDecodeError, ValidationError, ValueError)` arms
    catch it with no change -- a content failure advances the ladder
    exactly like a schema failure, and is fed to the typed-repair
    factory as the Attempt 3 error.
    """


# ---------------------------------------------------------------------------
# repair_prompt_factory -- typed-repair prompt builder contract
# ---------------------------------------------------------------------------


class RepairPromptFactory(Protocol):
    """Builds the Attempt 3 (typed repair) prompt.

    Given the raw failed model output plus the validation / parse error
    that rejected it, return the messages payload for the repair call.
    Concrete typed factories (one per structured pass) are a later
    sprint (2C); for step 1 the parameter is just this Protocol and the
    ladder supplies `default_repair_prompt_factory` when the caller
    passes `None`.

    The return value is whatever shape the project's slot fns accept as
    their first positional `messages` argument -- in this codebase a
    list of `{"role": ..., "content": ...}` dicts.
    """

    def __call__(
        self,
        *,
        original_prompt: Any,
        failed_output: str,
        error: BaseException,
    ) -> Any:
        ...


def default_repair_prompt_factory(
    *,
    original_prompt: Any,
    failed_output: str,
    error: BaseException,
) -> list[dict[str, str]]:
    """Generic typed-repair prompt used when the caller passes no factory.

    Prepends an explicit `CRITICAL:` directive naming the validation
    error, echoes the failed output (truncated), and restates the
    original instruction. Mirrors the CRITICAL-prefix repair style of
    `_otr_story_brief._build_repair_messages` so a structured pass that
    has not yet shipped a typed factory still gets a sane repair turn.
    """
    error_text = f"{type(error).__name__}: {error}"
    original_text = _prompt_to_text(original_prompt)
    critical = (
        "CRITICAL: Your previous response failed schema validation "
        f"because: {error_text}.\n\n"
        "Return ONE valid JSON object that satisfies the schema. No "
        "Markdown, no prose, no preamble -- only the JSON object.\n\n"
        f"Failed response: {failed_output[:400]}\n\n"
        "Original instruction follows.\n\n"
    )
    return [{"role": "user", "content": critical + original_text}]


# ---------------------------------------------------------------------------
# Helpers -- prompt normalisation, slot-fn invocation, parse + validate
# ---------------------------------------------------------------------------


def _prompt_to_text(prompt: Any) -> str:
    """Render an arbitrary prompt payload to a plain string.

    `prompt` may already be a string, or a messages list of
    `{"role", "content"}` dicts. The default repair factory needs a
    flat string to embed; this normalises both shapes.
    """
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, (list, tuple)):
        parts: list[str] = []
        for item in prompt:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(prompt)


def _prompt_to_messages(prompt: Any) -> Any:
    """Coerce a prompt payload into the messages shape a slot fn expects.

    A plain string is wrapped into a single `user` message; an existing
    messages list (or any other already-structured payload) is passed
    through untouched.
    """
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    return prompt


def _invoke_slot(
    slot_fn: Callable[..., str],
    messages: Any,
    *,
    temperature: float,
    max_new_tokens: int,
    grammar_path: Optional[str],
) -> str:
    """Call the slot fn, passing `grammar_path` best-effort when present.

    Grammar enforcement is not fully wired until a later sprint (2E).
    For step 1 the call tries to pass `grammar_path` through as a
    keyword; if the slot fn does not accept it (TypeError on the
    keyword), the call is retried without it. Either way the slot fn
    runs -- a slot fn that ignores grammar simply behaves like a normal
    structured attempt.
    """
    if grammar_path is None:
        return slot_fn(
            messages,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
    try:
        return slot_fn(
            messages,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            grammar_path=grammar_path,
        )
    except TypeError:
        # Slot fn does not yet accept a grammar_path keyword. Grammar
        # wiring lands in sprint 2E; for now degrade to a plain call.
        log.info(
            "[OTR_StructuredCall] slot fn does not accept grammar_path; "
            "running attempt without grammar enforcement"
        )
        return slot_fn(
            messages,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )


def _parse_and_validate(
    raw: str,
    schema: type[T],
    post_validator: Optional[Callable[[T], Optional[str]]] = None,
) -> T:
    """Extract the first JSON object from `raw`, validate, content-check.

    Uses the shared `_otr_json.parse_first_json_object` -- no second
    JSON parser. Raises `json.JSONDecodeError` on unparseable output or
    `pydantic.ValidationError` on a schema mismatch.

    When `post_validator` is supplied it runs on the schema-valid
    instance: a non-None return value is a content rejection and is
    raised as `PostValidationError`. All three exception types are
    caught by the ladder, which advances to the next attempt.

    `post_validator` is expected to RETURN an error string (or None),
    never to raise -- a raising post_validator is a programming error
    and is allowed to propagate.
    """
    data = _otr_json.parse_first_json_object(raw or "")
    instance = schema.model_validate(data)
    if post_validator is not None:
        content_error = post_validator(instance)
        if content_error is not None:
            raise PostValidationError(content_error)
    return instance


# ---------------------------------------------------------------------------
# Public entrypoint -- the 4-attempt retry ladder
# ---------------------------------------------------------------------------


def structured_call(
    *,
    prompt: Any,
    schema: type[T],
    slot_fn: Callable[..., str],
    base_temperature: float,
    structural_retry_temperature: float,
    repair_prompt_factory: Optional[RepairPromptFactory] = None,
    post_validator: Optional[Callable[[T], Optional[str]]] = None,
    max_new_tokens: int = _STRUCTURED_MAX_NEW_TOKENS,
    grammar_path: Optional[str] = None,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    helper_name: str = "structured_call",
) -> T:
    """Run a structured-JSON LLM call through the shared retry ladder.

    Returns the validated `schema` instance from the first attempt that
    yields a schema-valid result. Raises `StructuredCallFailedError`
    when the ladder is exhausted -- never returns a silent sentinel.

    Parameters (all keyword-only):
      prompt
        The user/system message payload for the LLM call. A plain
        string is wrapped into a single `user` message; an existing
        messages list is passed through.
      schema
        A pydantic model class. The LLM response is parsed and
        validated against it; the validated instance is returned.
      slot_fn
        The LLM generate callable. Signature in this codebase:
        `slot_fn(messages, *, temperature, max_new_tokens[, stop])
        -> str`.
      base_temperature
        Attempt 1 temperature.
      structural_retry_temperature
        Attempt 2 temperature. MUST be strictly lower than
        `base_temperature` (the 2B principle -- the structural retry
        lowers entropy, it does not raise it). The invariant is
        asserted at entry and fails loud if violated.
      repair_prompt_factory
        Callable that builds the Attempt 3 typed-repair prompt from the
        failed output + the validation error. `None` selects
        `default_repair_prompt_factory`.
      post_validator
        Optional content check run on every schema-valid instance,
        `post_validator(instance) -> str | None`. It returns an error
        string to REJECT the instance (a content failure pydantic
        cannot see -- e.g. a voice preset outside the runtime pool, a
        news brief below the key-term floor, a speaker outside the
        locked cast) or `None` to ACCEPT it. A rejection is raised as
        `PostValidationError` and advances the ladder exactly like a
        schema failure. Mirrors the `extra_check` idiom already used by
        `_otr_outline._run_call_with_retry`. `None` disables the check.
      max_new_tokens
        Token budget passed to `slot_fn` on every attempt. Structured
        passes vary widely -- a story-brief reflection needs ~160, a
        cast-contract audit ~2000, a script-doctor pass ~3500 -- so the
        caller sets its own budget. Defaults to
        `_STRUCTURED_MAX_NEW_TOKENS` (512) for a small JSON object.
      grammar_path
        Optional path to a GBNF grammar file. Used only by Attempt 4.
        Real grammar enforcement lands in a later sprint; for step 1
        the path is passed through to `slot_fn` best-effort. When
        `None`, Attempt 4 is unavailable and the ladder ends at
        Attempt 3.
      max_attempts
        Caps the ladder (default 4).
      helper_name
        Short string for logging / slot attribution.

    The ladder (stops at the first schema-valid result that also
    clears `post_validator`):
      Attempt 1: `slot_fn` at `base_temperature`.
      Attempt 2: SAME prompt at `structural_retry_temperature` (lower).
      Attempt 3: typed repair -- prompt built via
                 `repair_prompt_factory`, run at a static low
                 temperature (`_REPAIR_TEMPERATURE`).
      Attempt 4: grammar-enforced via `grammar_path` when available;
                 otherwise the ladder ends here.
    """
    # --- Invariant: structural retry must LOWER temperature (2B). ---
    # Fail loud at entry. A structural retry at >= base temperature is
    # a configuration bug, not a recoverable runtime state.
    if not structural_retry_temperature < base_temperature:
        raise ValueError(
            f"[OTR_StructuredCall] '{helper_name}': "
            f"structural_retry_temperature ({structural_retry_temperature}) "
            f"must be STRICTLY LOWER than base_temperature "
            f"({base_temperature}). Raising entropy during JSON-schema "
            "repair encourages more structural hallucination, not less; "
            "the structural retry lowers temperature."
        )

    if max_attempts < 1:
        raise ValueError(
            f"[OTR_StructuredCall] '{helper_name}': max_attempts must be "
            f">= 1, got {max_attempts}"
        )

    factory: RepairPromptFactory = (
        repair_prompt_factory
        if repair_prompt_factory is not None
        else default_repair_prompt_factory
    )

    base_messages = _prompt_to_messages(prompt)
    last_error: Optional[BaseException] = None
    last_raw: str = ""
    attempts_run = 0

    # --- Attempt 1: base temperature. ---
    if attempts_run < max_attempts:
        attempts_run += 1
        log.info(
            "[OTR_StructuredCall] '%s' attempt 1/%d: base call at "
            "temperature=%.3f",
            helper_name, max_attempts, base_temperature,
        )
        try:
            last_raw = _invoke_slot(
                slot_fn, base_messages,
                temperature=base_temperature,
                max_new_tokens=max_new_tokens, grammar_path=None,
            )
            return _parse_and_validate(last_raw, schema, post_validator)
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            last_error = exc
            log.warning(
                "[OTR_StructuredCall] '%s' attempt 1 failed: %s",
                helper_name, exc,
            )

    # --- Attempt 2: SAME prompt, lower temperature (structural retry). ---
    if attempts_run < max_attempts:
        attempts_run += 1
        log.info(
            "[OTR_StructuredCall] '%s' attempt 2/%d: structural retry at "
            "temperature=%.3f (lowered from %.3f)",
            helper_name, max_attempts,
            structural_retry_temperature, base_temperature,
        )
        try:
            last_raw = _invoke_slot(
                slot_fn, base_messages,
                temperature=structural_retry_temperature,
                max_new_tokens=max_new_tokens, grammar_path=None,
            )
            return _parse_and_validate(last_raw, schema, post_validator)
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            last_error = exc
            log.warning(
                "[OTR_StructuredCall] '%s' attempt 2 failed: %s",
                helper_name, exc,
            )

    # --- Attempt 3: typed repair at a static low temperature. ---
    if attempts_run < max_attempts:
        attempts_run += 1
        log.info(
            "[OTR_StructuredCall] '%s' attempt 3/%d: typed repair at "
            "temperature=%.3f",
            helper_name, max_attempts, _REPAIR_TEMPERATURE,
        )
        try:
            repair_error: BaseException = (
                last_error
                if last_error is not None
                else ValueError("no prior error captured")
            )
            repair_prompt = factory(
                original_prompt=prompt,
                failed_output=last_raw,
                error=repair_error,
            )
            repair_messages = _prompt_to_messages(repair_prompt)
            last_raw = _invoke_slot(
                slot_fn, repair_messages,
                temperature=_REPAIR_TEMPERATURE,
                max_new_tokens=max_new_tokens, grammar_path=None,
            )
            return _parse_and_validate(last_raw, schema, post_validator)
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            last_error = exc
            log.warning(
                "[OTR_StructuredCall] '%s' attempt 3 (repair) failed: %s",
                helper_name, exc,
            )

    # --- Attempt 4: grammar-enforced (only when a grammar_path exists). ---
    if attempts_run < max_attempts:
        if grammar_path is None:
            log.info(
                "[OTR_StructuredCall] '%s': no grammar_path supplied; "
                "Attempt 4 (grammar-enforced) unavailable -- ladder ends",
                helper_name,
            )
        elif not os.path.isfile(grammar_path):
            log.warning(
                "[OTR_StructuredCall] '%s': grammar_path %r does not "
                "exist; Attempt 4 unavailable -- ladder ends",
                helper_name, grammar_path,
            )
        else:
            attempts_run += 1
            log.info(
                "[OTR_StructuredCall] '%s' attempt 4/%d: grammar-enforced "
                "call (grammar_path=%r) at temperature=%.3f",
                helper_name, max_attempts, grammar_path, _GRAMMAR_TEMPERATURE,
            )
            try:
                last_raw = _invoke_slot(
                    slot_fn, base_messages,
                    temperature=_GRAMMAR_TEMPERATURE,
                    max_new_tokens=max_new_tokens,
                    grammar_path=grammar_path,
                )
                return _parse_and_validate(last_raw, schema, post_validator)
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                last_error = exc
                log.warning(
                    "[OTR_StructuredCall] '%s' attempt 4 (grammar) "
                    "failed: %s",
                    helper_name, exc,
                )

    # --- Ladder exhausted: fail loud. ---
    log.error(
        "[OTR_StructuredCall] '%s' exhausted the retry ladder after %d "
        "attempt(s); raising StructuredCallFailedError",
        helper_name, attempts_run,
    )
    raise StructuredCallFailedError(
        helper_name=helper_name,
        attempts=attempts_run,
        last_error=last_error,
    )
