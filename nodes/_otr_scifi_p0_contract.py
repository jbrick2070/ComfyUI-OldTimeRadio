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
