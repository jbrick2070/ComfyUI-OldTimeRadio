"""Source windowing for the Sci-Fi source-evidence P0 pass.

The three Sci-Fi source banks emit a bounded evidence artifact before any
creative work, and a long article cannot be read through one prompt window.
``p0_source_chunks`` cuts the body into windows that fit and reports each
window's offset so the caller can rebase every quoted span onto the full text.
``MAX_QUOTE_CHARS`` is the longest literal slice a span may quote.
"""
from __future__ import annotations

from typing import Mapping


MAX_QUOTE_CHARS = 240

# The fields whose text the model must be able to quote from verbatim. `seed_text`
# and `headline`/`summary` are short and bounded by the fetcher; `full_text` is the
# article body, and it is the one that grows without limit.
_P0_TRIMMABLE_FIELD = "full_text"

_SENTENCE_END = (". ", ".\n", "! ", "?\n", "? ", "!\n")


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
