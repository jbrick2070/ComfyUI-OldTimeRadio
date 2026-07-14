"""Shared capacity and repair contract for Sci-Fi source-evidence P0 passes.

The three Sci-Fi source banks emit a bounded evidence artifact before any
creative work.  Keeping the artifact surface bounded is what makes its
schema-bearing base and typed-repair prompts fit the local context window.
"""
from __future__ import annotations

import json
from typing import Any, Mapping, Sequence


MAX_FACT_ROWS = 6
MAX_ENTITY_ROWS = 4
MAX_NUMBER_ROWS = 4
MAX_SPANS_PER_EVIDENCE_ROW = 1
MAX_CLAIM_CHARS = 240
MAX_ENTITY_NAME_CHARS = 120
MAX_NUMERIC_TOKEN_CHARS = 96
MAX_QUOTE_CHARS = 240
MAX_TONE_CHARS = 80
MAX_HEADLINE_CLEAN_CHARS = 180
MAX_PROVENANCE_NOTE_CHARS = 240

_P0_BASE_OUTPUT_TOKENS = 2800
_P0_EXTRA_ROOT_FIELD_TOKENS = 100

# THE OTHER HALF OF THE CAPACITY CONTRACT.
#
# The ARTIFACT was bounded (the limits above) so the P0 prompt would fit. The
# SOURCE never was -- and the source is the only unbounded thing in that prompt.
# A long RSS article walked straight past the window: 5,424 input tokens into a
# 5,192-token opening, and `prompt_must_fit` correctly refused to left-truncate a
# provenance prompt it cannot slice (live: scifi_sonnet, prompt 415ca1fc). Fail
# loud is right; arriving there is not. Bound the evidence so it fits BY
# CONSTRUCTION.
#
# Measured against the live tokenizer (Mistral-Nemo-Instruct-2407, chat template
# applied, 2026-07-13):
#   * fixed overhead -- seam + P0 contract + schema contract + JSON scaffolding
#     + the chat template itself -- 2,544 tokens. Rounded up.
#   * source prose -- 0.177 tokens/char. Rounded UP to 0.20, so the estimate can
#     only ever over-count, never under-count.
_P0_LOCAL_CONTEXT_CAP = 8192
_P0_PROMPT_OVERHEAD_TOKENS = 2600
_P0_TOKENS_PER_CHAR = 0.20
_P0_FIT_MARGIN_TOKENS = 192

# The fields whose text the model must be able to quote from verbatim. `seed_text`
# and `headline`/`summary` are short and bounded by the fetcher; `full_text` is the
# article body, and it is the one that grows without limit.
_P0_TRIMMABLE_FIELD = "full_text"

_SENTENCE_END = (". ", ".\n", "! ", "?\n", "? ", "!\n")


def p0_source_char_budget(
    *,
    extra_root_fields: int = 0,
    context_cap: int = _P0_LOCAL_CONTEXT_CAP,
) -> int:
    """How many characters of source evidence provably fit the P0 prompt."""
    reserved = p0_output_token_budget(extra_root_fields=extra_root_fields)
    usable = int(context_cap) - reserved
    headroom = usable - _P0_PROMPT_OVERHEAD_TOKENS - _P0_FIT_MARGIN_TOKENS
    if headroom <= 0:
        return 0
    return int(headroom / _P0_TOKENS_PER_CHAR)


def p0_source_chunks(
    payload: Mapping[str, str],
    *,
    budget_chars: int,
) -> "list[tuple[int, dict[str, str]]]":
    """Split the article body into windows the P0 prompt can actually hold.

    NOT a trim. Trimming makes the tail of a long article UNCITABLE -- a fact in
    the last paragraph could never be quoted, and a 720-word episode is exactly the
    one that needs the evidence a short one can skip. So the article is read in
    windows, each of which fits, and each window's dossier is rebased back onto the
    full text afterwards.

    Every window carries the headline and summary (they are the framing, and they
    are small), so the budget for the BODY is what remains. Cuts land on sentence
    boundaries where one exists.

    Returns `(offset, payload)` pairs. `offset` is the character position of that
    window's body inside the original `full_text` -- the caller adds it back to
    every span it receives, which is what keeps the citations true.
    """
    windows: "list[tuple[int, dict[str, str]]]" = []
    body = str(payload.get(_P0_TRIMMABLE_FIELD) or "")
    frame_chars = sum(
        len(str(value or "")) for key, value in payload.items()
        if key != _P0_TRIMMABLE_FIELD
    )
    allowance = int(budget_chars) - frame_chars
    if allowance <= 0:
        # The frame alone exceeds the window; there is nothing honest to do but
        # hand back one window and let `prompt_must_fit` refuse it out loud.
        return [(0, dict(payload))]
    if len(body) <= allowance:
        return [(0, dict(payload))]

    offset = 0
    while offset < len(body):
        window = body[offset:offset + allowance]
        if offset + allowance < len(body):
            boundary = max(window.rfind(mark) for mark in _SENTENCE_END)
            # Honour a sentence boundary only if it keeps most of the window --
            # otherwise a period near the start would shred the article into
            # slivers and multiply the call count.
            if boundary > allowance // 2:
                window = window[: boundary + 1]
        fitted = dict(payload)
        fitted[_P0_TRIMMABLE_FIELD] = window
        windows.append((offset, fitted))
        offset += len(window)
    return windows


def p0_output_token_budget(*, extra_root_fields: int = 0) -> int:
    """Return the bounded Sci-Fi P0 output reservation.

    The fixed envelope is derived from the shared artifact limits above, not
    from an unbounded source article.  Codex/Gemini use the 2,800-token base;
    Sonnet's two additional required root strings reserve a small extra amount.
    """
    if (
        not isinstance(extra_root_fields, int)
        or isinstance(extra_root_fields, bool)
        or extra_root_fields < 0
    ):
        raise ValueError("extra_root_fields must be a non-negative integer")
    return _P0_BASE_OUTPUT_TOKENS + (
        _P0_EXTRA_ROOT_FIELD_TOKENS * extra_root_fields
    )


def p0_contract_receipt(*, extra_root_fields: int = 0) -> dict[str, int]:
    """Return the durable bounds receipt paired with a P0 call journal."""
    return {
        "max_new_tokens": p0_output_token_budget(
            extra_root_fields=extra_root_fields,
        ),
        "extra_root_fields": extra_root_fields,
        "max_fact_rows": MAX_FACT_ROWS,
        "max_entity_rows": MAX_ENTITY_ROWS,
        "max_number_rows": MAX_NUMBER_ROWS,
        "max_spans_per_evidence_row": MAX_SPANS_PER_EVIDENCE_ROW,
        "max_claim_chars": MAX_CLAIM_CHARS,
        "max_quote_chars": MAX_QUOTE_CHARS,
    }


def p0_contract_instruction(*, has_numeric_tokens: bool) -> str:
    """Return the model-visible compact-extraction contract for every P0 rung."""
    numeric_token_rule = (
        " Each fact's numeric_tokens array has at most four strings, each no "
        f"longer than {MAX_NUMERIC_TOKEN_CHARS} characters."
        if has_numeric_tokens else ""
    )
    return (
        "\nP0 COMPACT EXTRACTION CONTRACT: Select only story-usable literal "
        f"evidence: at most {MAX_FACT_ROWS} facts, {MAX_ENTITY_ROWS} named "
        f"entities, and {MAX_NUMBER_ROWS} numeric rows. Every fact and entity "
        f"has exactly one literal source span. Each claim is at most "
        f"{MAX_CLAIM_CHARS} characters, each entity name at most "
        f"{MAX_ENTITY_NAME_CHARS}, each numeric verbatim value at most "
        f"{MAX_NUMERIC_TOKEN_CHARS}, and every quoted source slice at most "
        f"{MAX_QUOTE_CHARS} characters.{numeric_token_rule} Tone is exactly one "
        f"nonempty source-derived string of at most {MAX_TONE_CHARS} characters, "
        "never an array or object. This is an evidence index, not an exhaustive "
        "transcription."
    )


def compact_p0_repair_context(
    *,
    failed_artifact: str,
    rejection: str,
    source_evidence: Mapping[str, Any],
    source_digest: str,
    allowed_source_fields: Sequence[str],
) -> str:
    """Render non-copyable P0 repair references without a JSON request wrapper."""
    return "\n".join((
        "INPUT REFERENCES ONLY -- they are not an output shape.",
        "<failed_fact_index>",
        str(failed_artifact),
        "</failed_fact_index>",
        "<rejection>",
        str(rejection),
        "</rejection>",
        "<source_evidence>",
        json.dumps(
            dict(source_evidence), sort_keys=True,
            separators=(",", ":"), ensure_ascii=False,
        ),
        "</source_evidence>",
        "<source_digest>",
        str(source_digest),
        "</source_digest>",
        "<allowed_source_fields>",
        json.dumps(
            sorted(str(field) for field in allowed_source_fields),
            separators=(",", ":"), ensure_ascii=False,
        ),
        "</allowed_source_fields>",
    ))
