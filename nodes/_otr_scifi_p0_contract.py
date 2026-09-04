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
P0_REPAIR_CONTEXT_MAX_BYTES = 16_384
MAX_CLAIM_CHARS = 240
MAX_ENTITY_NAME_CHARS = 120
MAX_NUMERIC_TOKEN_CHARS = 96
MAX_QUOTE_CHARS = 240
MAX_TONE_CHARS = 80
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
    prompt_reserve_tokens: int = 0,
) -> int:
    """How many characters of source evidence provably fit the P0 prompt."""
    if (
        not isinstance(prompt_reserve_tokens, int)
        or isinstance(prompt_reserve_tokens, bool)
        or prompt_reserve_tokens < 0
    ):
        raise ValueError("prompt_reserve_tokens must be a non-negative integer")
    reserved = p0_output_token_budget(extra_root_fields=extra_root_fields)
    usable = int(context_cap) - reserved
    headroom = (
        usable
        - _P0_PROMPT_OVERHEAD_TOKENS
        - _P0_FIT_MARGIN_TOKENS
        - prompt_reserve_tokens
    )
    if headroom <= 0:
        return 0
    return int(headroom / _P0_TOKENS_PER_CHAR)


def p0_source_chunks(
    payload: Mapping[str, str],
    *,
    budget_chars: int,
    overlap_chars: int = 0,
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
    if (
        not isinstance(overlap_chars, int)
        or isinstance(overlap_chars, bool)
        or overlap_chars < 0
        or overlap_chars >= allowance
    ):
        raise ValueError(
            "overlap_chars must be a non-negative integer smaller than the "
            f"body allowance ({allowance})"
        )
    if len(body) <= allowance:
        return [(0, dict(payload))]

    offset = 0
    while offset < len(body):
        hard_end = min(len(body), offset + allowance)
        end = hard_end
        window = body[offset:hard_end]
        if hard_end < len(body):
            boundary = max(window.rfind(mark) for mark in _SENTENCE_END)
            # Honour a sentence boundary only if it keeps most of the window --
            # otherwise a period near the start would shred the article into
            # slivers and multiply the call count.
            if boundary > allowance // 2:
                candidate_end = offset + boundary + 1
                if candidate_end - offset > overlap_chars:
                    end = candidate_end
                    window = body[offset:end]
        fitted = dict(payload)
        fitted[_P0_TRIMMABLE_FIELD] = window
        windows.append((offset, fitted))
        if end == len(body):
            break
        next_offset = end - overlap_chars
        if next_offset <= offset:
            # Defensive invariant: a sentence rewind may never stall progress.
            end = hard_end
            fitted[_P0_TRIMMABLE_FIELD] = body[offset:end]
            next_offset = end - overlap_chars
        if next_offset <= offset:
            raise ValueError("P0 source windowing could not make progress")
        offset = next_offset
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
        "never an array or object. For every span, the literal identity MUST "
        "hold: payload[field][start:end] == quote. Do not paraphrase, "
        "normalize, or reconstruct quote text. This is an evidence index, "
        "not an exhaustive transcription."
    )


#: How much of the failed artifact the repair owner is shown. The generic
#: `default_repair_prompt_factory` in `_otr_structured_call.py` has always
#: bounded its echo to 400 characters; this path echoed the whole thing, which
#: is one of the two reasons a P0 repair context could outgrow its own budget.
#: The echo exists so the owner can SEE the shape it produced -- it is not the
#: artifact of record, so a head slice is enough.
MAX_FAILED_ARTIFACT_ECHO_CHARS = 400

#: Never trim an evidence field below this. A field cut to a stub is worse than
#: a field that is absent: the model cannot tell a short article from a
#: truncated one, and a quote span into a stub cannot be verified.
MIN_EVIDENCE_FIELD_CHARS = 200

#: Appended to a trimmed evidence field so the truncation is visible IN the
#: prompt, not only in the receipt. The model must never believe it is looking
#: at a whole article.
_TRIM_MARK = " [...TRIMMED]"


class P0RepairTrimReceipt(dict):
    """What the envelope had to remove to fit, field by field.

    A plain dict subclass so it serialises into the call journal with no
    special casing. Empty means nothing was trimmed -- which is the common
    case and must stay distinguishable from "trimming happened but was not
    recorded".
    """


def _trim_evidence_to_fit(
    source_evidence: Mapping[str, Any],
    *,
    overflow_bytes: int,
    receipt: "dict[str, Any]",
) -> "dict[str, Any]":
    """Shrink the LONGEST evidence fields until `overflow_bytes` is recovered.

    Longest-first, deterministically (ties broken by field name), because the
    field that blew the envelope is the one that should pay for it -- trimming
    every field equally would damage the short ones that were never the
    problem. Each cut lands on a sentence boundary where one exists inside the
    last 25% of the kept text, so the model is not handed half a sentence.

    Returns a NEW mapping; the caller's evidence is never mutated. Every cut is
    recorded in `receipt`, and a cut that cannot reach the target still records
    what it did -- the hard checks downstream stay armed, so an unreachable
    target surfaces as a loud refusal rather than a silent short-fall.
    """
    trimmed = {str(key): value for key, value in source_evidence.items()}
    if overflow_bytes <= 0:
        return trimmed

    def _by_length(item):
        key, value = item
        return (-len(str(value)), key)

    remaining = int(overflow_bytes)
    for key, value in sorted(trimmed.items(), key=_by_length):
        if remaining <= 0:
            break
        text = str(value)
        if not isinstance(value, str) or len(text) <= MIN_EVIDENCE_FIELD_CHARS:
            continue
        before_bytes = len(text.encode("utf-8"))
        # Aim to recover `remaining` bytes from THIS field, but never take it
        # below the floor. Bytes are not characters; work in characters and
        # re-measure, so a multibyte body cannot under-deliver the cut.
        #
        # THE MARK PAYS FOR ITSELF. `_TRIM_MARK` is appended after the slice,
        # so a cut sized to exactly `remaining` gives back `len(_TRIM_MARK)`
        # bytes and lands the envelope just over its budget -- measured at 10
        # bytes over on the first build of this, twice, because the sentence
        # boundary had already clawed back three of the thirteen. Size the cut
        # to include the mark.
        keep = max(
            MIN_EVIDENCE_FIELD_CHARS,
            len(text) - remaining - len(_TRIM_MARK),
        )
        candidate = text[:keep]
        # THE BOUNDARY REWIND MUST NOT BREACH THE FLOOR. Cutting back to the
        # last sentence end is a courtesy; keeping the field above the floor is
        # the contract. An earlier draft applied the rewind unconditionally and
        # a 200-char floor produced 188-char fields -- the rewind took more
        # than the slice had spare. If the rewind would go under, keep the
        # straight slice: a clean cut mid-sentence is honest, and the trim mark
        # says so.
        tail = candidate[int(len(candidate) * 0.75):]
        boundary = max((tail.rfind(mark) for mark in _SENTENCE_END), default=-1)
        if boundary > 0:
            rewound = candidate[: int(len(candidate) * 0.75) + boundary + 1]
            if len(rewound.rstrip()) >= MIN_EVIDENCE_FIELD_CHARS:
                candidate = rewound
        # NO SECOND FLOOR CHECK HERE, DELIBERATELY. `keep` is already floored,
        # and the rewind above is only taken when it stays above the floor, so
        # `candidate` cannot be under it by construction. A belt-and-braces
        # restore was written here first and then removed: it was unreachable,
        # and two implementations of one property mask each other under
        # mutation -- each mutant survived because the other guard covered it,
        # so the floor looked tested and was not. One guard, reachable and
        # provable, beats two that hide each other.
        candidate = candidate.rstrip() + _TRIM_MARK
        after_bytes = len(candidate.encode("utf-8"))
        if after_bytes >= before_bytes:
            continue
        trimmed[key] = candidate
        recovered = before_bytes - after_bytes
        remaining -= recovered
        receipt[key] = {
            "chars_before": len(text),
            "chars_after": len(candidate),
            "bytes_removed": recovered,
        }
    return trimmed


def compact_p0_repair_context(
    *,
    failed_artifact: str,
    rejection: str,
    source_evidence: Mapping[str, Any],
    source_digest: str,
    allowed_source_fields: Sequence[str],
    max_bytes: int = P0_REPAIR_CONTEXT_MAX_BYTES,
    trim_receipt: "dict[str, Any] | None" = None,
) -> str:
    """Render bounded, tagged P0 repair references as data-only input.

    TRIMS TO FIT, THEN CHECKS (2026-07-29). This used to assemble the whole
    context and refuse if it did not fit, with no trim anywhere on the path --
    and it refused live, twice, killing two campaign legs before the repair
    owner was ever called (`still_flat`, 16796 bytes; PBUG-20260729-03).
    A bound that refuses is not a budget, it is a veto: a limit on a repair
    context has to come with the trim that makes the context fit it.

    Two inputs are bounded here, in this order:
      1. `failed_artifact` -- head-sliced to `MAX_FAILED_ARTIFACT_ECHO_CHARS`,
         matching the discipline `default_repair_prompt_factory` has always
         applied to its own echo.
      2. `source_evidence` -- trimmed longest-field-first, only as far as the
         envelope requires, never below `MIN_EVIDENCE_FIELD_CHARS`.

    `rejection`, `source_digest` and `allowed_source_fields` are NEVER trimmed:
    they are the instruction the owner is being asked to act on, and the
    coordinate system its quotes must land in. Trimming them would corrupt the
    handoff rather than shrink it.

    The hard check below STAYS, and stays fail-loud. After this change it can
    only fire on an arithmetic regression -- which makes it a bug detector, not
    a veto.

    `trim_receipt`, when supplied, is populated with one entry per trimmed
    field. NO SILENT ANYTHING: a caller that wants the receipt gets it, and the
    trim marker is visible in the prompt itself either way.
    """
    if max_bytes < 1:
        raise ValueError(
            f"P0 repair context max_bytes must be positive, got {max_bytes}"
        )

    receipt: "dict[str, Any]" = (
        trim_receipt if trim_receipt is not None else {}
    )

    echo = str(failed_artifact)
    if len(echo) > MAX_FAILED_ARTIFACT_ECHO_CHARS:
        receipt["failed_artifact"] = {
            "chars_before": len(echo),
            "chars_after": MAX_FAILED_ARTIFACT_ECHO_CHARS,
        }
        echo = echo[:MAX_FAILED_ARTIFACT_ECHO_CHARS]

    evidence: Mapping[str, Any] = source_evidence
    rendered = _render_p0_repair_context(
        failed_artifact=echo, rejection=rejection,
        source_evidence=evidence, source_digest=source_digest,
        allowed_source_fields=allowed_source_fields,
    )

    # CONVERGE, DO NOT TRUST ONE PASS OF ARITHMETIC.
    #
    # A single sized cut is an estimate: JSON escaping, multibyte characters
    # and the sentence-boundary rewind all move the real byte count away from
    # the predicted one. So render, MEASURE, and cut again if it still does not
    # fit -- bounded, and stopping the moment a pass recovers nothing, because
    # a pass that cannot make progress will not make progress by repeating.
    # Whatever is left over then meets the hard check below, which is exactly
    # where an unfittable envelope should surface.
    evidence_receipt: "dict[str, Any]" = {}
    for _attempt in range(4):
        overflow = len(rendered.encode("utf-8")) - max_bytes
        if overflow <= 0:
            break
        before_keys = dict(evidence_receipt)
        evidence = _trim_evidence_to_fit(
            evidence, overflow_bytes=overflow, receipt=evidence_receipt,
        )
        if evidence_receipt == before_keys:
            break                      # nothing moved; stop and let it raise
        rendered = _render_p0_repair_context(
            failed_artifact=echo, rejection=rejection,
            source_evidence=evidence, source_digest=source_digest,
            allowed_source_fields=allowed_source_fields,
        )
    if evidence_receipt:
        # `chars_before` is restated against the ORIGINAL text so a field cut
        # across two passes reports what it actually lost, not what the second
        # pass alone took off an already-trimmed value.
        for field, row in evidence_receipt.items():
            original = source_evidence.get(field)
            if isinstance(original, str):
                row["chars_before"] = len(original)
        receipt["source_evidence"] = evidence_receipt

    rendered_bytes = len(rendered.encode("utf-8"))
    if rendered_bytes > max_bytes:
        raise ValueError(
            f"P0 repair context is {rendered_bytes} bytes, over the hard "
            f"limit {max_bytes}"
        )
    return rendered


def _render_p0_repair_context(
    *,
    failed_artifact: str,
    rejection: str,
    source_evidence: Mapping[str, Any],
    source_digest: str,
    allowed_source_fields: Sequence[str],
) -> str:
    """Assemble the tagged block. Pure -- no bounds, no checks, no receipts."""
    return "\n".join((
        "INPUT REFERENCES ONLY -- they are not an output shape.",
        "REPAIR RULE: payload[field][start:end] == quote; quote is a literal "
        "source slice, never a paraphrase.",
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
