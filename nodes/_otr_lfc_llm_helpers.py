"""nodes/_otr_lfc_llm_helpers.py — two-step gen + JSON repair (ADR 6.11/6.12).

Helpers used by LFC LLM phases (Phase 4 per-scene coherence, Phase 5
voice drift, Phase 6 episode arc). Splits the LLM work into two
stages so creative reasoning is not constrained by JSON formatting:

  Step 1 (reasoning, big model)
    Prompt: open-ended -- "Evaluate the scene. Explain edits in
    Markdown / chain-of-thought. No JSON constraints."
    Goal: maximum creative bandwidth. Output is loose prose.

  Step 2 (formatter, small model)
    Prompt: schema-only -- "Translate this prose to the
    ReviewerOutput JSON schema. No new content, no creative work."
    Goal: deterministic JSON shape. Cheap call, sub-second on a
    Llama-3.2-3B-class formatter.

Then the JSON-repair loop catches small format slips (unescaped
quotes, trailing commas, smart quotes, markdown fences) without
needing a third LLM round-trip. ADR sections 6.11 + 6.12.

Both helpers are LLM-agnostic over Mistral-Nemo / Gemma / Qwen on
the reasoning side; the formatter is the small-LLM class
(Llama-3.2-3B Q4_K_M, etc.) loaded once into spare VRAM.

Token budget per ADR section 11:
  reasoning  big-model call    (large context, ~700 tok prompt)
  formatter  small-model call  (~300 tok prompt, sub-1-sec wall clock)

When the formatter generate_fn is not supplied (e.g. v2.0-alpha
deployments running only Mistral-Nemo), the reasoning fn is reused
for the formatter pass -- still 2 calls per phase, but on the same
model. The two-step contract holds (free reasoning -> structured
formatting); only the per-call cost differs.

Status: LFC sprint commit 6 of 14 (2026-05-11).
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

log = logging.getLogger("OTR.lfc_llm")


__all__ = [
    "ReviewerEdit",
    "ReviewerOutput",
    "EditorNote",
    "EditorNotesOutput",
    "two_step_generate",
    "parse_with_repair",
    "FORMATTER_REASONING_REUSE_LOG",
]


T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------------------
# Canonical structured output schemas (ADR section 6.10)
# ---------------------------------------------------------------------------


class ReviewerEdit(BaseModel):
    """One edit proposed by a Phase 4 / 5 / 6 LLM call.

    Distinct from _otr_ledger_reviewer.ReviewerEdit (which is the
    Phase 2 Script Doctor's edit shape). LFC edits are line-text-only
    rewrites; the reviewer's rewrite/skip/annotate action tagging is
    not needed here.
    """

    line_id: str
    original: str
    revised: str
    reason: str = ""


class ReviewerOutput(BaseModel):
    """Wrapper used by Phase 4 / 5 / 6 LLM formatter pass output."""

    edits: list[ReviewerEdit] = []
    rationale: str = ""


class EditorNote(BaseModel):
    """One Editor-role note used by Phase 6 arc audit (ADR 6.14)."""

    scope: str = "episode"   # "scene_<n>" / "speaker_<id>" / "episode"
    note: str
    suggested_edits: list[ReviewerEdit] = []


class EditorNotesOutput(BaseModel):
    """Wrapper used by the Editor-note scaffold formatter."""

    notes: list[EditorNote] = []


# ---------------------------------------------------------------------------
# Tag used for telemetry when the formatter fn falls back to the
# reasoning fn (mostly so soak metrics can distinguish "ran on the
# fast path with a dedicated 3B formatter" vs "reused Mistral-Nemo
# for both stages").
# ---------------------------------------------------------------------------


FORMATTER_REASONING_REUSE_LOG = (
    "[LFC:two_step] formatter_generate_fn not supplied; reusing "
    "reasoning_generate_fn for formatter pass"
)


# ---------------------------------------------------------------------------
# JSON repair fast-path regexes
# ---------------------------------------------------------------------------


_MD_FENCE_RE = re.compile(
    r"```(?:json)?\s*(.+?)\s*```",
    re.DOTALL | re.IGNORECASE,
)


def _strip_markdown_fences(text: str) -> str:
    if not text:
        return text
    m = _MD_FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    return text


def _strip_trailing_explanation(text: str) -> str:
    """Trim any prose AFTER the last closing brace.

    Models occasionally append a polite "Hope this helps!" sentence
    after a valid JSON object. The closing brace is the last
    semantically meaningful character; cut there.
    """
    if not text:
        return text
    last = text.rfind("}")
    if last == -1:
        return text
    return text[: last + 1]


def _fix_trailing_commas(text: str) -> str:
    """Remove trailing commas from JSON lists and objects.

    `[1, 2, 3,]` -> `[1, 2, 3]`. Cheap regex, no nesting tracking
    required for the common cases the formatter produces.
    """
    if not text:
        return text
    text = re.sub(r",\s*]", "]", text)
    text = re.sub(r",\s*}", "}", text)
    return text


def _fix_smart_quotes(text: str) -> str:
    """Replace curly smart quotes with ASCII quotes inside JSON.

    The replacement is unconditional -- json.loads cannot parse smart
    quotes anyway. Tests pin that LDQ / RDQ / smart apos are all
    handled.
    """
    if not text:
        return text
    return (
        text
        .replace("“", '"')   # LEFT DOUBLE QUOTATION MARK
        .replace("”", '"')   # RIGHT DOUBLE QUOTATION MARK
        .replace("‘", "'")   # LEFT SINGLE QUOTATION MARK
        .replace("’", "'")   # RIGHT SINGLE QUOTATION MARK
    )


def _extract_json_block(text: str) -> str:
    """Find the longest balanced {...} substring.

    Fallback for cases where _strip_markdown_fences /
    _strip_trailing_explanation leave junk before the first brace.
    """
    if not text:
        return text
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        return text[first : last + 1]
    return text


# ---------------------------------------------------------------------------
# parse_with_repair
# ---------------------------------------------------------------------------


def parse_with_repair(
    raw_output: str,
    schema: Type[T],
    generate_fn: Optional[Callable] = None,
    *,
    max_attempts: int = 3,
) -> Optional[T]:
    """Parse `raw_output` as `schema`. Repair on failure.

    Strategy (ADR section 6.12):
      Step 1  Deterministic regex repairs (free, in-process):
              strip markdown fences, trim trailing explanation,
              extract balanced { ... } substring, fix trailing
              commas, replace smart quotes.
      Step 2  Try schema.model_validate_json.
      Step 3  On ValidationError, ask `generate_fn` (if supplied) to
              repair the JSON. Up to `max_attempts` repair rounds.
              Each repair prompt includes the validation error so the
              model knows what to fix.
      Step 4  On exhaustion, return None and log. The phase advances
              with an empty edit list rather than crashing.

    `generate_fn` is optional -- callers may pass None to disable LLM
    repair entirely (regex-only fast path). In that case
    parse_with_repair returns None on any ValidationError that
    survives the regex pass.

    Returns the parsed schema instance, or None on exhaustion. Never
    raises.
    """
    if not isinstance(raw_output, str) or not raw_output.strip():
        return None

    # Fast-path deterministic repairs.
    cleaned = raw_output
    cleaned = _strip_markdown_fences(cleaned)
    cleaned = _strip_trailing_explanation(cleaned)
    cleaned = _extract_json_block(cleaned)
    cleaned = _fix_trailing_commas(cleaned)
    cleaned = _fix_smart_quotes(cleaned)

    attempt = 0
    while attempt < max_attempts:
        try:
            return schema.model_validate_json(cleaned)
        except (ValidationError, ValueError) as exc:
            attempt += 1
            if attempt >= max_attempts or generate_fn is None:
                log.warning(
                    "[LFC:parse_with_repair] exhausted "
                    "(attempt=%d/%d): %s",
                    attempt, max_attempts, exc,
                )
                return None
            # LLM repair pass.
            repair_prompt = (
                "You output invalid JSON. Fix it.\n"
                "Rules: Output ONLY valid JSON. Keep the same data "
                "and meaning. Do not add new keys.\n"
                f"Error: {exc}\n"
                f"Invalid output:\n{cleaned}"
            )
            try:
                next_raw = generate_fn(
                    [{"role": "user", "content": repair_prompt}],
                    temperature=0.1,
                    max_new_tokens=2000,
                )
            except Exception as gen_exc:  # noqa: BLE001
                log.warning(
                    "[LFC:parse_with_repair] LLM repair raised: %s",
                    gen_exc,
                )
                return None
            cleaned = _strip_markdown_fences(next_raw or "")
            cleaned = _strip_trailing_explanation(cleaned)
            cleaned = _extract_json_block(cleaned)
            cleaned = _fix_trailing_commas(cleaned)
            cleaned = _fix_smart_quotes(cleaned)
    return None


# ---------------------------------------------------------------------------
# two_step_generate
# ---------------------------------------------------------------------------


def two_step_generate(
    reasoning_generate_fn: Callable,
    formatter_generate_fn: Optional[Callable],
    *,
    reasoning_prompt: str,
    formatter_prompt_template: str,
    schema: Type[T],
    reasoning_temperature: float = 0.7,
    reasoning_max_new_tokens: int = 1500,
    formatter_temperature: float = 0.1,
    formatter_max_new_tokens: int = 2000,
    repair_max_attempts: int = 3,
) -> Optional[T]:
    """Two-step generate: free reasoning, then structured formatting.

    Per ADR section 6.11. The reasoning pass emits loose prose /
    Markdown; the formatter pass translates that into the schema.
    Returns the parsed schema instance, or None when both passes
    fail to produce valid JSON.

    `formatter_prompt_template` must contain the literal token
    `{{REASONING}}` -- it gets replaced with the reasoning pass
    output before the formatter call.

    `formatter_generate_fn` is optional. When None, the reasoning fn
    is reused for the formatter pass (still 2 calls per phase). The
    contract holds either way; only per-call cost differs (see ADR
    section 11 token-budget envelope).

    Never raises -- on any LLM call exception the function returns
    None with a logged warning so the calling phase advances.
    """
    if "{{REASONING}}" not in formatter_prompt_template:
        log.error(
            "[LFC:two_step] formatter_prompt_template missing "
            "{{REASONING}} marker; refusing call"
        )
        return None

    # Step 1: reasoning pass.
    try:
        reasoning_raw = reasoning_generate_fn(
            [{"role": "user", "content": reasoning_prompt}],
            temperature=reasoning_temperature,
            max_new_tokens=reasoning_max_new_tokens,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("[LFC:two_step] reasoning pass raised: %s", exc)
        return None
    if not (reasoning_raw or "").strip():
        log.warning("[LFC:two_step] reasoning pass returned empty")
        return None

    # Step 2: formatter pass.
    fmt_fn = formatter_generate_fn
    if fmt_fn is None:
        log.info(FORMATTER_REASONING_REUSE_LOG)
        fmt_fn = reasoning_generate_fn

    formatter_prompt = formatter_prompt_template.replace(
        "{{REASONING}}", reasoning_raw,
    )
    try:
        formatter_raw = fmt_fn(
            [{"role": "user", "content": formatter_prompt}],
            temperature=formatter_temperature,
            max_new_tokens=formatter_max_new_tokens,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("[LFC:two_step] formatter pass raised: %s", exc)
        return None

    # Step 3: parse + repair.
    return parse_with_repair(
        formatter_raw,
        schema,
        generate_fn=fmt_fn,
        max_attempts=repair_max_attempts,
    )
