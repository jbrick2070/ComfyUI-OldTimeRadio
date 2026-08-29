"""``text_tails`` -- the ONE checked-in rule for where a shortened line may end.

Two surfaces in this repo shorten authored text to fit a budget, and both of
them are read by a viewer: the Ghost Signal prompt composer
(``ghost_signal_prompt._trim_to``) and the still_word title card
(``otr_meta_brief_image_prompt._still_word_fit_card``). Both had the same rule
half-written, in two places, with two vocabularies -- which is a table that
drifts once.

THE RULE, and it is a small one: a cut may land on a word boundary, but it may
never land ON A DANGLING FUNCTION WORD. ``"...two nested dial-eyes and a"``
reads to a sampler as an unfinished list and invites it to invent whatever came
next; on a title card, whose entire job is the words, it reads to a human as a
sentence someone forgot to finish.

This module is PURE and stdlib-only, so both consumers can import it without
dragging anything behind it.
"""
from __future__ import annotations

import re

#: Function words a trimmed line may not END on. Checked in and deliberately
#: small: articles, coordinators, prepositions, relatives, possessives and
#: copulas -- the words that promise another word after them.
#:
#: NOT a parts-of-speech judge. A trailing ADJECTIVE (the shipped card that read
#: ``"...of a subterranean"``) is not in here and cannot be, because telling an
#: adjective from a noun needs a tagger this repo does not ship and should not
#: start guessing at. That case is closed one level up, by cutting at a whole
#: comma clause before a word boundary is ever considered.
DANGLING_TAIL_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "nor", "of", "to", "in", "on",
    "at", "by", "for", "from", "with", "into", "onto", "over", "under",
    "that", "which", "who", "whose", "as", "its", "their", "his", "her",
    "this", "these", "those", "is", "are", "was", "were", "be", "been",
    "--", "-", "",
})

_EDGE_PUNCT = ",.;:-"


def is_dangling(word) -> bool:
    """True when ``word`` is a function word a line may not end on."""
    return str(word or "").strip(_EDGE_PUNCT).lower() in DANGLING_TAIL_WORDS


def drop_dangling_tail(text) -> str:
    """Drop trailing function words from ``text``.

    Returns the text unchanged when its last word is already a complete
    thought. Returns ``""`` only when EVERY word was a function word, which the
    caller must treat as its own failure rather than shipping the empty string.
    """
    words = str(text or "").split()
    while words and is_dangling(words[-1]):
        words.pop()
    return " ".join(words).rstrip(" ,;:-")


def longest_clause_prefix(text, budget) -> str:
    """The longest run of WHOLE comma clauses of ``text`` that fits ``budget``.

    Preferred over a word-boundary cut because a whole clause is a complete
    thought and half a clause is not. Returns ``""`` when even the first clause
    is too long, which tells the caller to fall through to its word cut.
    """
    text = str(text or "").strip()
    budget = int(budget)
    if len(text) <= budget:
        return text
    parts = [p.strip() for p in text.split(",") if p.strip()]
    while len(parts) > 1:
        parts.pop()
        candidate = ", ".join(parts)
        if len(candidate) <= budget:
            return candidate
    return ""


def word_boundary_cut(text, budget) -> str:
    """Cut ``text`` to ``budget`` on a word boundary, never mid-word.

    A single word longer than the whole budget is the only case with no
    boundary to cut on; it is returned whole so the caller's own ceiling
    assertion is what fails, rather than this function returning an empty
    string that looks like success.
    """
    text = str(text or "").strip()
    budget = int(budget)
    if len(text) <= budget:
        return text
    kept = []
    for word in text.split():
        candidate = " ".join(kept + [word])
        if len(candidate) > budget:
            break
        kept.append(word)
    if not kept:
        return text.split()[0]
    return " ".join(kept).rstrip(" ,;:-")


# `collapse_ws` / `_WS_RE` were removed 2026-08-28: exported, never called.
# The video lane keeps its own `_normalize_ws` (ghost_signal_prompt.py) with
# nine live call sites, and by the operator's 2026-08-23 ruling the engine
# lanes stay independent -- so this shared copy had no future consumer either.

__all__ = [
    "DANGLING_TAIL_WORDS", "is_dangling", "drop_dangling_tail",
    "longest_clause_prefix", "word_boundary_cut",
]
