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
the structural retry LOWERS temperature relative to the base attempt,
and the typed repair runs at a static low temperature. One bounded
exception exists: if that typed repair itself ends in incomplete JSON
syntax and call budget remains, its exact prompt may be retried once
above the static repair temperature to avoid repeating an early stop.

Slot-fn calling convention is read from `nodes/_otr_story_brief.py`
(`run_story_brief_reflection`, `_repair_pass`): a slot fn has the
signature `slot_fn(messages, *, temperature, max_new_tokens[, stop])`
and returns a raw string. JSON extraction reuses the shared
`_otr_json.parse_first_json_object` -- this module does NOT hand-roll
a second JSON parser.

This module is PURE: no I/O, no GPU, no ComfyUI imports, no writer
imports.

UTF-8 no BOM. No em-dashes (Windows cp1252 subprocess decode trap).
4-space indentation.
"""
from __future__ import annotations

import json
import logging
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

# Default cap on the retry ladder: base -> structural retry -> typed
# repair. When a schema/content failure skips the structural rung, the
# remaining third call may recover JSON syntax from the typed repair.
_DEFAULT_MAX_ATTEMPTS: int = 3

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

# A typed repair that starts but does not finish decodable JSON gets one
# syntax-only retry when call budget remains. The retry temperature is
# capped by the caller's base temperature, so callers at or below this
# value keep their own lower ceiling.
_REPAIR_SYNTAX_RETRY_FLOOR: float = 0.25


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
    this exception. It subclasses `ValueError`, but the ladder's
    `except (json.JSONDecodeError, ValidationError, PostValidationError)`
    arms catch it EXPLICITLY (a plain `ValueError` is NOT recoverable and
    propagates) -- a content failure advances the ladder exactly like a
    schema failure, and is fed to the typed-repair factory as the
    typed-repair error.
    """


def schema_required_paths(schema: type[BaseModel]) -> tuple[str, ...]:
    """Return every required schema path, resolving nested model refs."""
    raw = schema.model_json_schema()
    definitions = raw.get("$defs", {})
    legacy_definitions = raw.get("definitions", {})
    paths: list[str] = []
    seen: set[tuple[str, str]] = set()

    def walk(node: object, path: str) -> None:
        if not isinstance(node, dict):
            return
        ref = node.get("$ref")
        if isinstance(ref, str):
            prefix, _, name = ref.rpartition("/")
            if prefix in {"#/$defs", "#/definitions"}:
                marker = (ref, path)
                if marker in seen:
                    return
                seen.add(marker)
                target = definitions.get(name, legacy_definitions.get(name, {}))
                walk(target, path)
                return
        for branch_key in ("anyOf", "oneOf", "allOf"):
            branches = node.get(branch_key)
            if isinstance(branches, list):
                for branch in branches:
                    walk(branch, path)
        if node.get("type") == "array":
            walk(node.get("items", {}), f"{path}[*]")
            return
        required = node.get("required", [])
        properties = node.get("properties", {})
        if not isinstance(required, list) or not isinstance(properties, dict):
            return
        for name in required:
            if not isinstance(name, str):
                continue
            child_path = f"{path}.{name}" if path else name
            paths.append(child_path)
            walk(properties.get(name, {}), child_path)

    walk(raw, "")
    return tuple(dict.fromkeys(paths))


def schema_shape_instruction(schema: type[BaseModel]) -> str:
    """Describe required nested schema paths for weak local JSON writers.

    The full JSON schema is already supplied as typed input, but compact path
    inventory is more reliably followed by small local models. This describes
    structure only; it never supplies or rewrites story content.
    """
    compact = ", ".join(schema_required_paths(schema))
    return (
        "\nReturn exactly one JSON object, with no Markdown, headings, or prose. "
        f"Its exact top-level keys are: {', '.join(schema.model_fields)}. "
        "Every required nested path must be present, including repeated graph "
        f"references. Required paths: {compact}"
    )


# ---------------------------------------------------------------------------
# repair_prompt_factory -- typed-repair prompt builder contract
# ---------------------------------------------------------------------------


class RepairPromptFactory(Protocol):
    """Builds the Attempt 3 (typed repair) prompt.

    Given the raw failed model output plus the validation / parse error
    that rejected it, a factory normally returns the messages payload
    for the repair LLM call -- in this codebase a list of
    `{"role": ..., "content": ...}` dicts (a plain string is also
    accepted and wrapped). The concrete typed factories live in
    `_otr_repair_prompts.py` (Sprint 2C); the ladder supplies
    `default_repair_prompt_factory` when the caller passes `None`.

    A factory MAY instead resolve the failure itself and return a
    finished, schema-valid pydantic instance of the call's `schema`
    type. `structured_call` detects that, runs it through
    `post_validator`, and returns it WITHOUT making an LLM repair call.
    This is how `cast_membership_repair` short-circuits the LLM when the
    project's Levenshtein matcher resolves a phantom name
    deterministically.
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
) -> str:
    """Call the slot fn with the standard structured-pass signature.

    Every slot fn in this codebase has the signature
    `slot_fn(messages, *, temperature, max_new_tokens) -> str`.

    [OpenRouter] A remote OpenRouter slot fn with NO schema-bound
    response_format is asked for json_object mode, so a free-form frontier
    model (e.g. Opus, whose default is prose) returns a parseable JSON
    object for the parse+validate ladder below. A remote fn that already
    carries a json_schema keeps it; local fns take neither kwarg. This is a
    wiring change only -- the prompt, schema, and fail-closed ladder are
    unchanged, and the local path is byte-identical.
    """
    if (
        getattr(slot_fn, "_otr_openrouter", False)
        and getattr(slot_fn, "_otr_response_format", None) is None
    ):
        return slot_fn(
            messages,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            response_format={"type": "json_object"},
        )
    return slot_fn(
        messages,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )


def apply_field_aliases(
    aliases: dict[str, tuple[str, ...]],
    data: Any,
) -> Any:
    """Deterministic, whitelist-exact key normalization for ONE model's own
    top-level keys, for use inside a pydantic ``mode="before"`` validator.

    Maps a declared SYNONYM key to its CANONICAL field name ONLY when the
    canonical key is absent and EXACTLY ONE synonym is present. An explicit
    canonical key always wins; 0 or >= 2 synonyms (a collision) leave the field
    untouched, so a genuinely missing required field still fails LOUD. No-op on
    canonical input (byte-identical) and on a non-dict ``data`` -- a
    ``mode="before"`` validator may receive a model instance or another type
    during internal pydantic operations (e.g. ``model_copy``). Copies the dict
    once on the first move; never mutates the input. Whitelist-exact and
    deterministic: no fuzzy or positional matching, so a given input yields the
    same normalization every run.

    ``aliases`` maps ``canonical_field -> tuple(synonym_keys)`` (top-level keys
    of the annotated model). This is the single home for the alias rule; the
    proven nested case (a ``BeatEdit`` whose action arrived under ``lever``) and
    any future annotated schema route through it via that schema's own
    ``mode="before"`` validator. pydantic runs that validator on nested models
    during recursion, so the nested fix needs no path-walking in the core.
    """
    if not isinstance(data, dict) or not aliases:
        return data
    out: Optional[dict] = None
    for canonical, synonyms in aliases.items():
        src = out if out is not None else data
        if canonical in src:
            continue  # explicit canonical always wins
        present = [s for s in synonyms if s in src]
        if len(present) != 1:
            continue  # 0 -> nothing to map; >= 2 -> ambiguous, leave fail-loud
        if out is None:
            out = dict(data)
        out[canonical] = out.pop(present[0])
        log.debug(
            "[OTR_StructuredCall] field alias: %r -> %r", present[0], canonical,
        )
    return out if out is not None else data


def validate_tolerant_data(
    data: object,
    schema: type[T],
    *,
    post_validator: Optional[Callable[[T], Optional[str]]] = None,
) -> T:
    """Strict-first validate, then bounded tolerance, then the content check.

    The shared tolerant core reused by ``structured_call`` (through
    ``_parse_and_validate`` / ``parse_validate_tolerant``) AND by the binary
    decision lane. The ladder:

      1. ``schema.model_validate(data)`` -- the EXACT current strict parse. A
         schema with no tolerance hooks is byte-identical to today.
      2. On ``ValidationError`` ONLY: clamp over-long top-level string fields to
         their declared ``max_length`` and re-validate. A verbose/weak model can
         overflow a short capped tag field (e.g. the outline's ``time_of_day``,
         max 40 chars: an 8B model wrote a whole sentence -> the whole episode
         aborted, 2026-06-18). This fires solely on a would-fail output -- a good
         model's result is never touched (byte-identical) -- and any OTHER
         validation error still propagates.
      3. ``post_validator`` content check (a CONTENT failure pydantic cannot
         see -- a voice preset outside the pool, a speaker outside the locked
         cast) -> ``PostValidationError`` on rejection.

    DELIBERATELY NO alias key-normalization here: alias drift is handled DURING
    step 1 by each annotated schema's own ``mode="before"`` validator
    (``apply_field_aliases``), so it also covers NESTED models (pydantic
    recursion) and stays byte-identical on canonical input. Keeping the except
    arm to CLAMPING only avoids a second alias code path in the shared core.
    """
    try:
        instance = schema.model_validate(data)
    except ValidationError as ve:
        repaired = _clamp_overlong_strings(data, ve)
        if repaired is None:
            raise
        repaired_data, clamped = repaired
        instance = schema.model_validate(repaired_data)  # other errors propagate
        log.warning(
            "[OTR_StructuredCall] coerced %d over-long field(s) to the schema "
            "max_length to avoid an abort: %s",
            len(clamped), ", ".join(clamped),
        )
    if post_validator is not None:
        content_error = post_validator(instance)
        if content_error is not None:
            raise PostValidationError(content_error)
    return instance


def parse_validate_tolerant(
    raw: str,
    schema: type[T],
    *,
    post_validator: Optional[Callable[[T], Optional[str]]] = None,
) -> T:
    """Extract the first JSON object from ``raw`` then ``validate_tolerant_data``.

    Uses the shared ``_otr_json.parse_first_json_object`` -- no second JSON
    parser. Raises ``json.JSONDecodeError`` on unparseable output. Raw-string
    structured sites call this; already-parsed-dict sites call
    ``validate_tolerant_data`` directly (e.g. the binary decision lane).
    """
    data = _otr_json.parse_first_json_object(raw or "")
    return validate_tolerant_data(data, schema, post_validator=post_validator)


def _raw_head(raw: "str | None", cap: int = 400) -> str:
    """Sanitized head of a failed model output for the ladder WARNING
    logs (live-smoke hardening 2026-07-10: three production stage
    failures in one night were undiagnosable because the raw output
    was never logged). Whitespace runs collapse to single spaces so
    the log line stays one line and greppable."""
    if not raw:
        return "<empty>"
    head = " ".join(str(raw).split())
    if len(head) > cap:
        return head[:cap] + "..."
    return head


def _parse_and_validate(
    raw: str,
    schema: type[T],
    post_validator: Optional[Callable[[T], Optional[str]]] = None,
) -> T:
    """Back-compat thin wrapper over ``parse_validate_tolerant``.

    Preserved so the retry ladder's three call sites stay unchanged. Behavior is
    byte-identical to the pre-refactor function: parse the first JSON object ->
    strict validate with the over-long-string clamp fallback -> ``post_validator``
    content check (raised as ``PostValidationError``). All three recoverable
    exception types are caught by the ladder, which advances to the next attempt.
    """
    return parse_validate_tolerant(raw, schema, post_validator=post_validator)


def _clamp_overlong_strings(
    data: object, ve: "ValidationError"
) -> Optional[tuple[dict, list[str]]]:
    """Given a ValidationError, clamp every ``string_too_long`` field in
    ``data`` -- at ANY depth the error's ``loc`` path reaches (nested
    models / list items included; the scifi_fable2 S1b live smoke
    2026-07-10 overflowed ``pitches.0.hook`` and the old top-level-only
    clamp skipped it) -- to the ``max_length`` carried in that error's
    ``ctx`` (pydantic supplies it), trimming at a word boundary where
    possible. Returns ``(repaired_dict, clamped_field_paths)`` or
    ``None`` when nothing is clampable (so the caller re-raises the
    original error untouched)."""
    if not isinstance(data, dict):
        return None
    import copy
    out = copy.deepcopy(data)
    clamped: list[str] = []
    for err in ve.errors():
        if err.get("type") != "string_too_long":
            continue
        loc = err.get("loc") or ()
        max_len = (err.get("ctx") or {}).get("max_length")
        if not loc or not isinstance(max_len, int):
            continue
        # Walk the loc path to the leaf's parent container.
        node: object = out
        reachable = True
        for key in loc[:-1]:
            if isinstance(node, dict) and key in node:
                node = node[key]
            elif (isinstance(node, list) and isinstance(key, int)
                    and 0 <= key < len(node)):
                node = node[key]
            else:
                reachable = False
                break
        if not reachable:
            continue
        leaf = loc[-1]
        if isinstance(node, dict) and leaf in node:
            val = node[leaf]
        elif (isinstance(node, list) and isinstance(leaf, int)
                and 0 <= leaf < len(node)):
            val = node[leaf]
        else:
            continue
        if not isinstance(val, str) or len(val) <= max_len:
            continue
        cut = val[:max_len].rstrip()
        if " " in cut:  # prefer a clean word boundary over a mid-word chop
            cut = cut.rsplit(" ", 1)[0].rstrip()
        new_val = cut or val[:max_len]
        node[leaf] = new_val  # type: ignore[index]
        clamped.append(".".join(str(k) for k in loc))
    return (out, clamped) if clamped else None


# ---------------------------------------------------------------------------
# Public entrypoint -- the 3-attempt retry ladder
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
        `default_repair_prompt_factory`. A factory may instead return a
        finished `schema` instance, which is accepted directly with no
        LLM repair call (see `RepairPromptFactory`). The built repair
        messages are cached for one syntax-only retry if budget remains.
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
      max_attempts
        Caps the ladder (default 3).
      helper_name
        Short string for logging / slot attribution.

    The ladder (stops at the first schema-valid result that also
    clears `post_validator`; only `json.JSONDecodeError`,
    `pydantic.ValidationError`, and `PostValidationError` are recoverable
    -- a plain `ValueError` propagates):
      Attempt 1: `slot_fn` at `base_temperature`.
      Structural retry: SAME prompt at `structural_retry_temperature`
                 (lower) -- run ONLY when the prior failure was a
                 `json.JSONDecodeError` (malformed JSON a calmer re-prompt
                 may fix). A `ValidationError` / `PostValidationError`
                 SKIPS this rung (a re-prompt re-emits the same bad shape
                 and only burns tokens) and goes straight to typed repair.
      Typed repair: `repair_prompt_factory` either builds a repair prompt
                 (run at the static low temperature `_REPAIR_TEMPERATURE`)
                 or hands back a finished `schema` instance, returned
                 directly with no LLM call.
      Repair syntax retry: if the typed repair returned non-decodable JSON
                 and budget remains, retry its exact cached prompt once at
                 a bounded temperature above the static repair floor. This
                 does not run for schema or content rejection.
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
    repair_messages: Any = None

    # --- Attempt 1: base temperature. ---
    if attempts_run < max_attempts:
        attempts_run += 1
        log.info(
            "[OTR_StructuredCall] '%s' attempt %d/%d: base call at "
            "temperature=%.3f",
            helper_name, attempts_run, max_attempts, base_temperature,
        )
        try:
            last_raw = _invoke_slot(
                slot_fn, base_messages,
                temperature=base_temperature,
                max_new_tokens=max_new_tokens,
            )
            return _parse_and_validate(last_raw, schema, post_validator)
        except (json.JSONDecodeError, ValidationError, PostValidationError) as exc:
            last_error = exc
            log.warning(
                "[OTR_StructuredCall] '%s' attempt %d failed: %s | raw "
                "head: %s", helper_name, attempts_run, exc,
                _raw_head(last_raw),
            )

    # --- Structural retry: SAME prompt, lower temperature -- ONLY for a JSON
    # SYNTAX failure. A re-prompt at lower temperature can shake loose a
    # parseable object when the model emitted malformed JSON. It does NOT help a
    # ValidationError / PostValidationError (the JSON parsed; the SHAPE or
    # CONTENT is wrong) -- a re-prompt just re-emits the same shape, burning a
    # credit-billed call (the 2026-06-25 Opus normalize_length exhaustion: the
    # structural rung never helped, it only spent tokens). So on a non-syntax
    # failure skip straight to the typed repair; spend the structural retry only
    # on json.JSONDecodeError. attempts_run advances ONLY when this branch runs.
    if attempts_run < max_attempts and isinstance(last_error, json.JSONDecodeError):
        attempts_run += 1
        log.info(
            "[OTR_StructuredCall] '%s' attempt %d/%d: structural retry at "
            "temperature=%.3f (lowered from %.3f)",
            helper_name, attempts_run, max_attempts,
            structural_retry_temperature, base_temperature,
        )
        try:
            last_raw = _invoke_slot(
                slot_fn, base_messages,
                temperature=structural_retry_temperature,
                max_new_tokens=max_new_tokens,
            )
            return _parse_and_validate(last_raw, schema, post_validator)
        except (json.JSONDecodeError, ValidationError, PostValidationError) as exc:
            last_error = exc
            log.warning(
                "[OTR_StructuredCall] '%s' attempt %d failed: %s | raw "
                "head: %s", helper_name, attempts_run, exc,
                _raw_head(last_raw),
            )

    # --- Typed repair at a static low temperature (the final rung). ---
    if attempts_run < max_attempts:
        attempts_run += 1
        log.info(
            "[OTR_StructuredCall] '%s' attempt %d/%d: typed repair at "
            "temperature=%.3f",
            helper_name, attempts_run, max_attempts, _REPAIR_TEMPERATURE,
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
            # A typed repair factory MAY resolve the failure itself --
            # e.g. cast_membership_repair remapping a phantom speaker to
            # a locked-cast member via Levenshtein -- and hand back a
            # finished `schema` instance instead of a repair prompt.
            # Accept it directly: no LLM repair call is made. The
            # instance still passes through post_validator so a
            # deterministic "fix" that is itself content-invalid fails
            # the ladder loudly rather than slipping through.
            if isinstance(repair_prompt, schema):
                if post_validator is not None:
                    content_error = post_validator(repair_prompt)
                    if content_error is not None:
                        raise PostValidationError(content_error)
                log.info(
                    "[OTR_StructuredCall] '%s' attempt %d: repair factory "
                    "resolved the failure deterministically; no LLM "
                    "repair call made",
                    helper_name, attempts_run,
                )
                return repair_prompt
            repair_messages = _prompt_to_messages(repair_prompt)
            last_raw = _invoke_slot(
                slot_fn, repair_messages,
                temperature=_REPAIR_TEMPERATURE,
                max_new_tokens=max_new_tokens,
            )
            return _parse_and_validate(last_raw, schema, post_validator)
        except (json.JSONDecodeError, ValidationError, PostValidationError) as exc:
            last_error = exc
            log.warning(
                "[OTR_StructuredCall] '%s' attempt %d (repair) failed: "
                "%s | raw head: %s",
                helper_name, attempts_run, exc, _raw_head(last_raw),
            )

    # A semantic/schema failure on the base call skips the structural rung,
    # so typed repair can be attempt 2 of a three-call budget. If that repair
    # starts but does not finish decodable JSON, spend the one remaining call
    # on the exact same repair prompt. Do not rebuild it from the incomplete
    # response (which would enlarge the prompt), and do not retry a repair that
    # was itself schema-valid but content-invalid.
    if (
        attempts_run < max_attempts
        and repair_messages is not None
        and isinstance(last_error, json.JSONDecodeError)
    ):
        attempts_run += 1
        retry_temperature = min(
            base_temperature,
            max(structural_retry_temperature, _REPAIR_SYNTAX_RETRY_FLOOR),
        )
        log.info(
            "[OTR_StructuredCall] '%s' attempt %d/%d: typed repair syntax "
            "retry at temperature=%.3f",
            helper_name, attempts_run, max_attempts, retry_temperature,
        )
        try:
            last_raw = _invoke_slot(
                slot_fn, repair_messages,
                temperature=retry_temperature,
                max_new_tokens=max_new_tokens,
            )
            return _parse_and_validate(last_raw, schema, post_validator)
        except (json.JSONDecodeError, ValidationError, PostValidationError) as exc:
            last_error = exc
            log.warning(
                "[OTR_StructuredCall] '%s' attempt %d (repair syntax retry) "
                "failed: %s | raw head: %s",
                helper_name, attempts_run, exc, _raw_head(last_raw),
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
