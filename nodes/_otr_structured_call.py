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

import copy
import json
import logging
import uuid
from typing import Any, Callable, Mapping, Optional, Protocol, TypeVar

from pydantic import BaseModel, ValidationError


log = logging.getLogger("OTR")


# Shared JSON extraction. Package import in production; flat import
# when the module is loaded standalone / under test. Mirrors the
# import guard in `_otr_story_brief.py`.
try:
    from . import _otr_json
except ImportError:  # pragma: no cover - standalone / test load
    import _otr_json  # type: ignore


# A-4 (2026-07-30, writer repair): the ladder must know WHICH capacity
# failures a retry could fix. It asks the module that owns capacity
# arithmetic; it does not keep its own opinion, and it still imports no
# writer -- `_otr_generation_budget` is pure (stdlib typing only). Same dual
# guard as the `_otr_json` import above: without the flat fallback this module
# fails at COLLECTION under a standalone/test load.
try:
    from ._otr_generation_budget import (
        CAPACITY_ERRORS,
        is_rerollable_capacity_error,
    )
except ImportError:  # pragma: no cover - standalone / test load
    from _otr_generation_budget import (  # type: ignore
        CAPACITY_ERRORS,
        is_rerollable_capacity_error,
    )


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

# The alternate-owner handoff is deliberately a single direct branch. A
# caller can disable it with zero, but cannot accidentally turn this helper
# into an unbounded owner loop.
# Existing callers have no alternate owner yet, so the safe compatibility
# default is disabled. A caller that supplies an owner opts into one attempt.
_DEFAULT_MAX_REPAIR_ATTEMPTS: int = 0
_DEFAULT_REPAIR_CONTEXT_MAX_BYTES: int = 16_384

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
# ladder.  P3's bounded authored-text patch uses this same value rather than
# inventing a second repair temperature. The underscore name is the only
# spelling: the unused public alias was ripped on 2026-09-04 (dead-code audit
# row F), and every call site and test already used this one.
_REPAIR_TEMPERATURE: float = 0.10

# A typed repair that starts but does not finish decodable JSON gets one
# syntax-only retry when call budget remains. The retry temperature is
# capped by the caller's base temperature, so callers at or below this
# value keep their own lower ceiling.
_REPAIR_SYNTAX_RETRY_FLOOR: float = 0.25

# A compact, model-visible rendering of the complete output contract.  The
# marker makes injection idempotent: a number of established lane wrappers
# already append ``schema_shape_instruction`` themselves, while the shared
# ladder now guarantees coverage for every other caller and every repair turn.
_SCHEMA_CONTRACT_MARKER = "[OTR_SCHEMA_CONTRACT_V1]"
_SCHEMA_CONTRACT_END_MARKER = "[/OTR_SCHEMA_CONTRACT_V1]"
_SCHEMA_CONSTRAINT_KEYS = (
    "minLength",
    "maxLength",
    "pattern",
    "minItems",
    "maxItems",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
)


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
        repair_attempted: bool = False,
        terminal_disposition: str = "primary_ladder_exhausted",
    ) -> None:
        self.helper_name = helper_name
        self.attempts = attempts
        self.last_error = last_error
        self.repair_attempted = repair_attempted
        self.terminal_disposition = terminal_disposition
        last_error_text = (
            f"{type(last_error).__name__}: {last_error}"
            if last_error is not None
            else "no error captured"
        )
        super().__init__(
            f"[OTR_StructuredCall] '{helper_name}' failed after "
            f"{attempts} attempt(s); disposition={terminal_disposition}; "
            f"last error -> {last_error_text}"
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


# A-4 (2026-07-30, writer repair): the failures ONE attempt of the ladder may
# absorb rather than let escape. A capacity failure is in this tuple so it can
# be CAUGHT instead of escaping as an unhandled Exception on attempt 1 of 3 --
# which is what killed three 45-word legs (PBUG-20260729-02: "the ladder
# advertised a budget it never spent", 24 minutes for one leg).
#
# Being CAUGHT is not the same as being RETRIED. `_attempt_is_retryable` below
# refuses a `prompt_no_room` phase and the ladder re-raises it untouched, so a
# prompt that cannot fit still fails on the spot instead of burning two more
# calls on arithmetic that cannot change.
_ATTEMPT_ERRORS = (
    json.JSONDecodeError,
    ValidationError,
    PostValidationError,
) + CAPACITY_ERRORS


def _attempt_is_retryable(error: Any) -> bool:
    """Return False for a caught failure the ladder must re-raise instead.

    Only capacity failures can answer False: a JSON, schema or content
    failure has always advanced the ladder and still does.
    """
    if isinstance(error, CAPACITY_ERRORS):
        return is_rerollable_capacity_error(error)
    return True


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


def schema_property_paths(schema: type[BaseModel]) -> tuple[str, ...]:
    """Return every declared property path, resolving refs and array items.

    ``schema_required_paths`` intentionally answers the narrower question that
    legacy seams used to need.  Audit contracts need a stronger, schema-grounded
    inventory: a blocking claim may only point at a property the repair owner can
    actually author.  This walker is shared by the prompt renderer and those
    static checks, so the two cannot drift.
    """
    raw = schema.model_json_schema()
    definitions = raw.get("$defs", {})
    legacy_definitions = raw.get("definitions", {})
    paths: list[str] = []
    seen: set[tuple[str, str]] = set()

    def resolve_ref(node: Mapping[str, Any]) -> Mapping[str, Any] | None:
        ref = node.get("$ref")
        if not isinstance(ref, str):
            return None
        prefix, _, name = ref.rpartition("/")
        if prefix not in {"#/$defs", "#/definitions"}:
            return None
        target = definitions.get(name, legacy_definitions.get(name))
        if not isinstance(target, dict):
            return None
        merged = dict(target)
        merged.update({key: value for key, value in node.items() if key != "$ref"})
        return merged

    def walk(node: object, path: str) -> None:
        if not isinstance(node, dict):
            return
        ref = node.get("$ref")
        if isinstance(ref, str):
            marker = (ref, path)
            if marker in seen:
                return
            seen.add(marker)
            resolved = resolve_ref(node)
            if resolved is not None:
                walk(resolved, path)
            return

        for branch_key in ("anyOf", "oneOf", "allOf"):
            branches = node.get(branch_key)
            if isinstance(branches, list):
                for branch in branches:
                    walk(branch, path)

        if node.get("type") == "array":
            walk(node.get("items", {}), f"{path}[*]")
            return

        properties = node.get("properties", {})
        if not isinstance(properties, dict):
            return
        for name, child in properties.items():
            if not isinstance(name, str):
                continue
            child_path = f"{path}.{name}" if path else name
            paths.append(child_path)
            walk(child, child_path)

    walk(raw, "")
    return tuple(dict.fromkeys(paths))


def _schema_constraint_description(node: Mapping[str, Any]) -> str:
    """Render the finite, enforceable facts carried by one JSON-schema node."""
    parts: list[str] = []
    kind = node.get("type")
    if isinstance(kind, list):
        parts.append("type=" + "|".join(str(value) for value in kind))
    elif isinstance(kind, str):
        parts.append(f"type={kind}")
    for branch_key in ("anyOf", "oneOf", "allOf"):
        branches = node.get(branch_key)
        if isinstance(branches, list):
            parts.append(f"{branch_key}={len(branches)}")
    if "const" in node:
        parts.append("const=" + json.dumps(node["const"], ensure_ascii=False))
    enum = node.get("enum")
    if isinstance(enum, list):
        parts.append("enum=" + json.dumps(enum, ensure_ascii=False))
    for key in _SCHEMA_CONSTRAINT_KEYS:
        if key in node:
            parts.append(f"{key}=" + json.dumps(node[key], ensure_ascii=False))
    if "format" in node:
        parts.append(f"format={node['format']}")
    return "; ".join(parts) or "schema-defined value"


def _schema_contract_lines(schema: type[BaseModel]) -> tuple[str, ...]:
    """Return a compact recursive contract for every output schema field."""
    raw = schema.model_json_schema()
    definitions = raw.get("$defs", {})
    legacy_definitions = raw.get("definitions", {})
    lines: list[str] = []
    seen_lines: set[str] = set()
    seen_refs: set[tuple[str, str]] = set()

    def add(line: str) -> None:
        if line not in seen_lines:
            seen_lines.add(line)
            lines.append(line)

    def resolve_ref(node: Mapping[str, Any]) -> Mapping[str, Any] | None:
        ref = node.get("$ref")
        if not isinstance(ref, str):
            return None
        prefix, _, name = ref.rpartition("/")
        if prefix not in {"#/$defs", "#/definitions"}:
            return None
        target = definitions.get(name, legacy_definitions.get(name))
        if not isinstance(target, dict):
            return None
        merged = dict(target)
        merged.update({key: value for key, value in node.items() if key != "$ref"})
        return merged

    def walk(node: object, path: str, required: bool) -> None:
        if not isinstance(node, dict):
            return
        ref = node.get("$ref")
        if isinstance(ref, str):
            marker = (ref, path)
            if marker in seen_refs:
                return
            seen_refs.add(marker)
            resolved = resolve_ref(node)
            if resolved is not None:
                walk(resolved, path, required)
            return

        label = path or "<root>"
        if path:
            requirement = "required" if required else "optional"
            add(f"- {label}: {requirement}; {_schema_constraint_description(node)}")

        properties = node.get("properties", {})
        # An "exact keys=" line is only meaningful for a node that actually
        # declares properties. Scalars carry none, so emitting the line for them
        # produced an empty, meaningless key list on every string/int/bool field
        # -- pure token cost and a confusion risk for the small technical model.
        names = (
            [name for name in properties if isinstance(name, str)]
            if isinstance(properties, dict) else []
        )
        if names:
            closed = node.get("additionalProperties") is False
            closure = "closed; " if closed else ""
            add(
                f"- object {label}: {closure}exact keys="
                + ", ".join(names)
            )
            required_names = node.get("required", [])
            required_set = {
                name for name in required_names if isinstance(name, str)
            } if isinstance(required_names, list) else set()
            for name in names:
                child_path = f"{path}.{name}" if path else name
                walk(properties[name], child_path, name in required_set)

        if node.get("type") == "array":
            walk(node.get("items", {}), f"{path}[*]", True)

        for branch_key in ("anyOf", "oneOf", "allOf"):
            branches = node.get(branch_key)
            if isinstance(branches, list):
                for branch in branches:
                    walk(branch, path, required)

    walk(raw, "", True)
    return tuple(lines)


def schema_shape_instruction(schema: type[BaseModel]) -> str:
    """Render the complete model-visible JSON contract for ``schema``.

    Required paths alone were not enough: local writers could still miss a
    literal, a bound, or a closed nested object that Python enforced later.
    This text is deliberately compact enough for prompt-fit callers while
    carrying every JSON-schema constraint that can be stated mechanically.
    """
    compact = ", ".join(schema_required_paths(schema))
    details = "\n".join(_schema_contract_lines(schema))
    return (
        f"\n{_SCHEMA_CONTRACT_MARKER}\n"
        "Return exactly one JSON object, with no Markdown, headings, or prose. "
        f"Its exact top-level keys are: {', '.join(schema.model_fields)}. "
        "Do not add undeclared fields. Every required nested path must be present, "
        "including repeated graph references. "
        f"Required paths: {compact}\n"
        "Complete output schema contract:\n"
        f"{details}\n{_SCHEMA_CONTRACT_END_MARKER}"
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


class RepairLedgerBuilder(Protocol):
    """Build the bounded, data-only prompt for an alternate repair owner."""

    def __call__(
        self,
        *,
        failed_output: str,
        error: BaseException,
        repair_nonce: str,
        max_bytes: int,
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
    through unchanged.
    """
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    return prompt


def _prompt_has_schema_contract(prompt: Any) -> bool:
    """Return whether a prompt already carries the idempotency marker."""
    if isinstance(prompt, str):
        return _SCHEMA_CONTRACT_MARKER in prompt
    if isinstance(prompt, (list, tuple)):
        return any(
            isinstance(row, Mapping)
            and _SCHEMA_CONTRACT_MARKER in str(row.get("content", ""))
            for row in prompt
        )
    return False


def _copy_message_prompt(prompt: list[Any] | tuple[Any, ...]) -> Any:
    """Shallow-copy rows without erasing a message-container contract."""
    rows = [dict(row) if isinstance(row, Mapping) else row for row in prompt]
    if isinstance(prompt, list):
        copied = copy.copy(prompt)
        copied[:] = rows
        return copied
    return tuple(rows)


def _inherit_generation_contract(source: Any, target: Any) -> Any:
    """Rewrap repair rows in the original list subtype and all its markers."""
    if not isinstance(source, list) or type(source) is list:
        return target
    if not isinstance(target, (list, tuple)):
        return target
    copied = copy.copy(source)
    copied[:] = [
        dict(row) if isinstance(row, Mapping) else row for row in target
    ]
    return copied


def _prompt_with_schema_contract(prompt: Any, schema: type[BaseModel]) -> Any:
    """Attach ``schema``'s contract without mutating a caller-owned prompt.

    Structured callers pass either a plain string or OpenAI-style messages.  A
    messages prompt receives the contract in its first system row. If it has no
    system row, the first textual message carries the contract instead so
    established one-message callers keep their message order and testable
    user-payload position. A string remains a string so custom repair factories
    that deliberately rely on that shape keep their contract.
    The marker prevents a wrapper and the shared ladder from adding duplicate
    schema text, including on syntax retries and custom typed repairs.
    """
    if isinstance(prompt, str):
        if _SCHEMA_CONTRACT_MARKER in prompt:
            return prompt
        return prompt + schema_shape_instruction(schema)
    if not isinstance(prompt, (list, tuple)):
        return prompt

    messages = _copy_message_prompt(prompt)
    if _prompt_has_schema_contract(messages):
        return messages
    contract = schema_shape_instruction(schema)
    for row in messages:
        if (
            isinstance(row, dict)
            and row.get("role") == "system"
            and isinstance(row.get("content"), str)
        ):
            row["content"] = str(row["content"]) + contract
            return messages
    for row in messages:
        if isinstance(row, dict) and isinstance(row.get("content"), str):
            row["content"] = str(row["content"]) + contract
            return messages
    return [{"role": "system", "content": contract}, *messages]


def invoke_structured_slot(
    slot_fn: Callable[..., str],
    messages: Any,
    *,
    temperature: float,
    max_new_tokens: int | None,
    force_json_object: bool = True,
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
    # force_json_object=False is how a TEXT pass opts out (2026-08-25). A
    # `gguf_native` or OpenRouter row carries `_otr_supports_json_object`, and
    # on the GGUF backend that kwarg installs a real JSON grammar on the
    # decode -- so a labelled-section prompt would be sampled under a
    # constraint that forbids the very format it asks for. The transformers
    # rows in tonight's failure never carried the marker, which is why the
    # live bug is fixed either way; this closes the same hole for the model
    # classes the fix was NOT tested against.
    if (
        force_json_object
        and (
            getattr(slot_fn, "_otr_supports_json_object", False)
            # Compat fallback: older/direct OpenRouter fns predate the
            # _otr_supports_json_object marker but still force json_object.
            or getattr(slot_fn, "_otr_openrouter", False)
        )
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


# Compatibility alias retained while callers migrate to the explicit shared
# invocation name.  It preserves OpenRouter json_object behavior at every
# structured boundary.
_invoke_slot = invoke_structured_slot


def validate_tolerant_data(
    data: object,
    schema: type[T],
    *,
    post_validator: Optional[Callable[[T], Optional[str]]] = None,
) -> T:
    """Validate exact structured data, then run an optional structural check.

    The core never truncates or rewrites model-authored strings. Schemas retain
    only genuine machine, provenance, graph, cardinality, and nonempty
    constraints; validation errors advance the bounded structural repair ladder.
    """
    instance = schema.model_validate(data)
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
    """Extract the first JSON object and validate it without prose mutation."""
    data = _otr_json.parse_first_json_object(raw or "")
    return validate_tolerant_data(
        data,
        schema,
        post_validator=post_validator,
    )


def _raw_head(
    raw: "str | None", cap: int = 400, error: "BaseException | None" = None,
) -> str:
    """Sanitized head of a failed model output for the ladder WARNING
    logs (live-smoke hardening 2026-07-10: three production stage
    failures in one night were undiagnosable because the raw output
    was never logged). Whitespace runs collapse to single spaces so
    the log line stays one line and greppable.

    FALLS BACK TO THE ERROR'S OWN COMPLETION (2026-08-13). A transport that
    raises from INSIDE the generate call -- the in-decode liveness guard does
    exactly this -- never assigns the caller's `last_raw`, so the ladder logged
    `raw head: <empty>` while the halted text sat on the exception all along.
    That was observed live on a `wan_ti2v` leg: the writer printed the runaway
    evidence and the ladder line beside it said empty.
    """
    if not raw and error is not None:
        raw = getattr(error, "raw_completion", None)
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
    text_parser: Optional[Callable[[str], Any]] = None,
) -> T:
    """Parse and validate one structured response without rewriting prose.

    ``text_parser`` (2026-08-25) lets a caller supply its own raw-text ->
    dict step in place of JSON extraction, for a pass whose reply is a
    LABELLED-SECTION document rather than a JSON object. Everything after the
    parse is unchanged -- the same schema, the same tolerant validation, the
    same post_validator, the same ladder around it -- so a text pass keeps
    every guarantee a JSON pass has EXCEPT the requirement that a small model
    balance braces. Default ``None`` keeps every existing caller byte-identical.
    """
    if text_parser is not None:
        return validate_tolerant_data(
            text_parser(raw or ""),
            schema,
            post_validator=post_validator,
        )
    return parse_validate_tolerant(
        raw,
        schema,
        post_validator=post_validator,
    )


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
    max_new_tokens: int | None = _STRUCTURED_MAX_NEW_TOKENS,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    helper_name: str = "structured_call",
    on_attempt_complete: Optional[
        Callable[[int, str, Optional[BaseException]], None]
    ] = None,
    repair_slot_fn: Optional[Callable[..., str]] = None,
    repair_ledger_builder: Optional[RepairLedgerBuilder] = None,
    repair_context_max_bytes: int = _DEFAULT_REPAIR_CONTEXT_MAX_BYTES,
    max_repair_attempts: int = _DEFAULT_MAX_REPAIR_ATTEMPTS,
    text_parser: Optional[Callable[[str], Any]] = None,
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
      on_attempt_complete
        Optional observability hook invoked once for each actual slot attempt
        after validation succeeds or fails. It receives the one-based attempt
        number, the slot output passed to validation, and ``None`` on success
        or the raised validation/parse/provider exception on failure. The hook
        is diagnostic only: a hook exception is logged and never changes the
        structured-call result or retry ladder.
      repair_slot_fn
        Optional alternate owner callable using the same slot-fn convention.
        It runs only after the primary ladder exhausts with a recoverable
        parse/schema/post-validation failure.
      repair_ledger_builder
        Optional builder for the alternate owner's bounded, data-only prompt.
        It receives the failed output, last recoverable error, a fresh nonce,
        and the byte ceiling. A builder without a repair slot is invalid.
      repair_context_max_bytes
        Hard UTF-8 byte ceiling for the alternate repair prompt. Oversized
        context fails explicitly; it is never silently clipped.
      max_repair_attempts
        Alternate-owner budget. This helper permits zero or one direct repair
        attempt; recursive re-entry is not supported.

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

    if max_repair_attempts not in (0, 1):
        raise ValueError(
            f"[OTR_StructuredCall] '{helper_name}': max_repair_attempts "
            f"must be 0 or 1, got {max_repair_attempts}"
        )
    if repair_context_max_bytes < 1:
        raise ValueError(
            f"[OTR_StructuredCall] '{helper_name}': "
            f"repair_context_max_bytes must be >= 1, got "
            f"{repair_context_max_bytes}"
        )
    if repair_slot_fn is None and repair_ledger_builder is not None:
        raise ValueError(
            f"[OTR_StructuredCall] '{helper_name}': a repair ledger builder "
            "requires repair_slot_fn"
        )
    if repair_slot_fn is not None and repair_ledger_builder is None:
        raise ValueError(
            f"[OTR_StructuredCall] '{helper_name}': repair_slot_fn requires "
            "repair_ledger_builder"
        )
    if max_repair_attempts and repair_slot_fn is None:
        raise ValueError(
            f"[OTR_StructuredCall] '{helper_name}': max_repair_attempts "
            "requires repair_slot_fn and repair_ledger_builder"
        )

    factory: RepairPromptFactory = (
        repair_prompt_factory
        if repair_prompt_factory is not None
        else default_repair_prompt_factory
    )

    # A text pass owns its own format instruction in the pack seam; injecting
    # the JSON schema shape on top of it would tell the model to do BOTH and
    # is exactly how a labelled-section reply turns back into broken JSON.
    contract_prompt = (
        prompt if text_parser is not None
        else _prompt_with_schema_contract(prompt, schema))
    base_messages = _prompt_to_messages(contract_prompt)
    last_error: Optional[BaseException] = None
    last_raw: str = ""
    attempts_run = 0
    repair_messages: Any = None
    repair_attempted = False

    def notify_attempt(error: Optional[BaseException]) -> None:
        if on_attempt_complete is None:
            return
        try:
            on_attempt_complete(attempts_run, last_raw, error)
        except Exception:
            log.exception(
                "[OTR_StructuredCall] '%s' attempt completion hook failed",
                helper_name,
            )

    # --- Attempt 1: base temperature. ---
    if attempts_run < max_attempts:
        attempts_run += 1
        log.info(
            "[OTR_StructuredCall] '%s' attempt %d/%d: base call at "
            "temperature=%.3f",
            helper_name, attempts_run, max_attempts, base_temperature,
        )
        try:
            last_raw = ""
            last_raw = _invoke_slot(
                slot_fn, base_messages,
                temperature=base_temperature,
                max_new_tokens=max_new_tokens,
                force_json_object=text_parser is None,
            )
            result = _parse_and_validate(
                last_raw,
                schema,
                post_validator,
                text_parser,
            )
            notify_attempt(None)
            return result
        except _ATTEMPT_ERRORS as exc:
            if not _attempt_is_retryable(exc):
                notify_attempt(exc)
                raise
            last_error = exc
            notify_attempt(exc)
            log.warning(
                "[OTR_StructuredCall] '%s' attempt %d failed: %s | raw "
                "head: %s", helper_name, attempts_run, exc,
                _raw_head(last_raw, error=exc),
            )
        except Exception as exc:
            notify_attempt(exc)
            raise

    # --- Structural retry: SAME prompt, lower temperature -- ONLY for a JSON
    # SYNTAX failure. A re-prompt at lower temperature can shake loose a
    # parseable object when the model emitted malformed JSON. It does NOT help a
    # ValidationError / PostValidationError (the JSON parsed; the SHAPE or
    # CONTENT is wrong) -- a re-prompt just re-emits the same shape, burning a
    # credit-billed call (the 2026-06-25 Opus normalize_length exhaustion: the
    # structural rung never helped, it only spent tokens). So on a non-syntax
    # failure skip straight to the typed repair; spend the structural retry only
    # on json.JSONDecodeError. attempts_run advances ONLY when this branch runs.
    # A-4: an `output_limit` capacity failure joins the syntax failure here.
    # It is the SAME remedy for the same shape of problem -- the model did not
    # finish, and the same prompt at a lower temperature is a real second
    # chance rather than a re-emission of an identical wrong shape. It is NOT
    # given to the typed repair for the reason the rung's own comment gives:
    # `last_raw` is bound to "" before every call and only rebound when the
    # call RETURNS, so a capacity raise leaves no artifact to repair, and the
    # completion A-1 attached to the exception is deliberately not fed into a
    # repair prompt (bounding that is a separate, unratified change).
    if attempts_run < max_attempts and (
        isinstance(last_error, json.JSONDecodeError)
        or is_rerollable_capacity_error(last_error)
    ):
        attempts_run += 1
        log.info(
            "[OTR_StructuredCall] '%s' attempt %d/%d: structural retry at "
            "temperature=%.3f (lowered from %.3f)",
            helper_name, attempts_run, max_attempts,
            structural_retry_temperature, base_temperature,
        )
        try:
            last_raw = ""
            last_raw = _invoke_slot(
                slot_fn, base_messages,
                temperature=structural_retry_temperature,
                max_new_tokens=max_new_tokens,
                force_json_object=text_parser is None,
            )
            result = _parse_and_validate(
                last_raw,
                schema,
                post_validator,
                text_parser,
            )
            notify_attempt(None)
            return result
        except _ATTEMPT_ERRORS as exc:
            if not _attempt_is_retryable(exc):
                notify_attempt(exc)
                raise
            last_error = exc
            notify_attempt(exc)
            log.warning(
                "[OTR_StructuredCall] '%s' attempt %d failed: %s | raw "
                "head: %s", helper_name, attempts_run, exc,
                _raw_head(last_raw, error=exc),
            )
        except Exception as exc:
            notify_attempt(exc)
            raise

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
                original_prompt=contract_prompt,
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
                # The attempt COMPLETED here, so report it like every other
                # successful return does. This was the one exit of six that
                # skipped the hook: `attempts_run` had already been
                # incremented for this rung, so a caller counting attempts
                # under-reported by one whenever a typed repair factory
                # resolved the failure itself. Unreachable from callers that
                # pass no `deterministic_repair` (the story-brief reflection),
                # but a lane MAY pass one.
                # Mutation-checked 2026-08-09: deleting this call turns
                # test_deterministic_repair_return_still_reports_its_attempt
                # red, so the guard is real rather than decorative.
                notify_attempt(None)
                return repair_prompt
            repair_prompt = _inherit_generation_contract(
                contract_prompt, repair_prompt,
            )
            repair_messages = _prompt_to_messages(
                repair_prompt if text_parser is not None
                else _prompt_with_schema_contract(repair_prompt, schema)
            )
            last_raw = ""
            last_raw = _invoke_slot(
                slot_fn, repair_messages,
                temperature=_REPAIR_TEMPERATURE,
                max_new_tokens=max_new_tokens,
                force_json_object=text_parser is None,
            )
            result = _parse_and_validate(
                last_raw,
                schema,
                post_validator,
                text_parser,
            )
            notify_attempt(None)
            return result
        except _ATTEMPT_ERRORS as exc:
            if not _attempt_is_retryable(exc):
                notify_attempt(exc)
                raise
            last_error = exc
            notify_attempt(exc)
            log.warning(
                "[OTR_StructuredCall] '%s' attempt %d (repair) failed: "
                "%s | raw head: %s",
                helper_name, attempts_run, exc, _raw_head(last_raw, error=exc),
            )
        except Exception as exc:
            notify_attempt(exc)
            raise

    # A semantic/schema failure on the base call skips the structural rung,
    # so typed repair can be attempt 2 of a three-call budget. If that repair
    # starts but does not finish decodable JSON, spend the one remaining call
    # on the exact same repair prompt. Do not rebuild it from the incomplete
    # response (which would enlarge the prompt), and do not retry a repair that
    # was itself schema-valid but content-invalid.
    # A-4: and here too, for the same reason one rung up -- a repair call that
    # ran out of output room did not produce a wrong shape, it produced no
    # shape, so re-sending the SAME repair prompt is the honest last swing.
    # The rung's existing discipline is unchanged: the prompt is never rebuilt
    # from the incomplete response, and a repair that was schema-valid but
    # content-invalid still does not come back here.
    if (
        attempts_run < max_attempts
        and repair_messages is not None
        and (
            isinstance(last_error, json.JSONDecodeError)
            or is_rerollable_capacity_error(last_error)
        )
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
            last_raw = ""
            last_raw = _invoke_slot(
                slot_fn, repair_messages,
                temperature=retry_temperature,
                max_new_tokens=max_new_tokens,
                force_json_object=text_parser is None,
            )
            result = _parse_and_validate(
                last_raw,
                schema,
                post_validator,
                text_parser,
            )
            notify_attempt(None)
            return result
        except _ATTEMPT_ERRORS as exc:
            if not _attempt_is_retryable(exc):
                notify_attempt(exc)
                raise
            last_error = exc
            notify_attempt(exc)
            log.warning(
                "[OTR_StructuredCall] '%s' attempt %d (repair syntax retry) "
                "failed: %s | raw head: %s",
                helper_name, attempts_run, exc, _raw_head(last_raw, error=exc),
            )
        except Exception as exc:
            notify_attempt(exc)
            raise

    # --- Alternate owner: one direct bounded handoff after recoverable
    # primary-ladder exhaustion. This branch intentionally does not call
    # `structured_call` again, so an alternate owner cannot recurse forever.
    if (
        repair_slot_fn is not None
        and repair_ledger_builder is not None
        and max_repair_attempts == 1
        and isinstance(
            last_error,
            (json.JSONDecodeError, ValidationError, PostValidationError),
        )
    ):
        repair_attempted = True
        repair_nonce = uuid.uuid4().hex
        repair_prompt = repair_ledger_builder(
            failed_output=last_raw,
            error=last_error,
            repair_nonce=repair_nonce,
            max_bytes=repair_context_max_bytes,
        )
        repair_prompt = _inherit_generation_contract(
            contract_prompt, repair_prompt,
        )
        repair_messages = _prompt_to_messages(
            repair_prompt if text_parser is not None
            else _prompt_with_schema_contract(repair_prompt, schema)
        )
        rendered_bytes = len(
            _prompt_to_text(repair_messages).encode("utf-8")
        )
        if rendered_bytes > repair_context_max_bytes:
            raise ValueError(
                f"[OTR_StructuredCall] '{helper_name}': alternate repair "
                f"context is {rendered_bytes} bytes, over the hard limit "
                f"{repair_context_max_bytes}"
            )
        attempts_run += 1
        log.info(
            "[OTR_StructuredCall] '%s' alternate repair attempt %d "
            "nonce=%s at temperature=%.3f",
            helper_name, attempts_run, repair_nonce, _REPAIR_TEMPERATURE,
        )
        try:
            last_raw = ""
            last_raw = _invoke_slot(
                repair_slot_fn,
                repair_messages,
                temperature=_REPAIR_TEMPERATURE,
                max_new_tokens=max_new_tokens,
                force_json_object=text_parser is None,
            )
            result = _parse_and_validate(
                last_raw,
                schema,
                post_validator,
                text_parser,
            )
            notify_attempt(None)
            return result
        except _ATTEMPT_ERRORS as exc:
            if not _attempt_is_retryable(exc):
                notify_attempt(exc)
                raise
            last_error = exc
            notify_attempt(exc)
            log.warning(
                "[OTR_StructuredCall] '%s' alternate repair failed: %s | "
                "raw head: %s",
                helper_name, exc, _raw_head(last_raw, error=exc),
            )
        except Exception as exc:
            notify_attempt(exc)
            raise

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
        repair_attempted=repair_attempted,
        terminal_disposition=(
            "repair_owner_exhausted"
            if repair_attempted
            else "primary_ladder_exhausted"
        ),
    )
