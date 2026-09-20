"""Canonical derived text metrics for OTR ledger rows.

This module is deliberately stdlib-only so every ledger producer, repair pass,
readiness pass, and freeze auditor can share one word-boundary contract without
creating import cycles.  ASCII hyphens and straight/smart apostrophes stay
inside a word; en/em dashes are punctuation boundaries.
"""
from __future__ import annotations

import re
from collections.abc import MutableMapping
from typing import Any


#: A WORD STARTS WITH A LETTER OF ANY SCRIPT, not only A-Z. The first cut was
#: `[A-Za-z][A-Za-z0-9'-]*`, and on 2026-09-19 it counted Tsubouchi's whole
#: Japanese balcony scene -- 3,606 characters -- as TWO words, so the verbatim
#: planner refused the vendored translation ("a 2-word passage cannot fill 12
#: beats") and every Japanese Shakespeare episode fell back to a machine
#: translation wearing the translator's credit. The same regex split `n\u00e3o` at
#: the accent into two words, so every Portuguese, Spanish, Italian and French
#: ledger row was over-counted. ON ASCII THIS PATTERN IS THE OLD ONE, LETTER
#: FOR LETTER: `[^\W\d_]` is "a letter" and equals `[A-Za-z]` on ASCII;
#: `[^\W_]` is "a letter or digit" and equals `[A-Za-z0-9]` there; the
#: apostrophes and the hyphen are admitted anywhere in the continuation
#: exactly as before, so `Nay--I'll` is still one word. That is deliberate:
#: a stricter continuation was tried first and moved 15 of 27,597 pure-ASCII
#: rows -- every one a Folger double dash -- and frozen English ledgers are
#: re-derived by the freeze auditor, so the English count may not move at all.
#: Measured after this form: 0 of 27,597 ASCII rows differ.
WORD_RE = re.compile(r"[^\W\d_](?:[^\W_]|['\u2018\u2019-])*")

#: CJK HAS NO SPACES, SO ITS CHARACTERS CARRY THE COUNT. Han, hiragana,
#: katakana (including half-width) and hangul: each run of them is removed
#: before the word scan and counted at TWO CHARACTERS PER WORD, which is the
#: ratio the balcony scene measures at -- the eight English speeches the plan
#: selects run 259 words, and Tsubouchi's rendering of the same eight runs
#: about 560 characters. CJK punctuation (U+3000-303F) is not in the class and
#: is never counted. The ratio is a proxy for spoken length, the only thing a
#: word count is used for here; it is not a claim about Japanese morphology.
#: U+3005-3007 (the iteration mark in 人々, 〆, 〇) are letters of the same
#: script and sit just outside the kana block; left out, 々 counted as a
#: Latin word of its own (agy, on 96346118).
CJK_RUN_RE = re.compile(
    "[\u3005-\u3007\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
    "\uac00-\ud7af\uff66-\uff9f]+")
CJK_CHARS_PER_WORD = 2


def canonical_word_count(text: Any) -> int:
    """Return the canonical ledger word count for *text*."""
    if not isinstance(text, str) or not text:
        return 0
    cjk_chars = sum(len(run) for run in CJK_RUN_RE.findall(text))
    words = len(WORD_RE.findall(CJK_RUN_RE.sub(" ", text))) if cjk_chars else len(WORD_RE.findall(text))
    return words + (cjk_chars + CJK_CHARS_PER_WORD - 1) // CJK_CHARS_PER_WORD


def canonical_char_count(text: Any) -> int:
    """Return the canonical ledger character count for *text*."""
    if not isinstance(text, str) or not text:
        return 0
    return len(text)


def stamp_line_text_metrics(row: MutableMapping[str, Any]) -> bool:
    """Re-derive a row's counts from its existing canonical ``text``.

    Invalid/non-string text is left structurally untouched so the freeze audit
    can still report it as an error.  Its derived counts are reset to zero.
    """
    if not isinstance(row, MutableMapping):
        return False
    text = row.get("text")
    row["char_count"] = canonical_char_count(text)
    row["word_count"] = canonical_word_count(text)
    return isinstance(text, str)


def set_line_text_metrics(row: MutableMapping[str, Any], text: Any) -> bool:
    """Atomically set canonical text plus both derived row metrics."""
    if not isinstance(row, MutableMapping):
        return False
    safe_text = text if isinstance(text, str) else str(text or "")
    row["text"] = safe_text
    row["char_count"] = canonical_char_count(safe_text)
    row["word_count"] = canonical_word_count(safe_text)
    return True
