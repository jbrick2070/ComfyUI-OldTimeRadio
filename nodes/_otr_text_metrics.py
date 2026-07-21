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


WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'\u2018\u2019-]*")


def canonical_word_count(text: Any) -> int:
    """Return the canonical ledger word count for *text*."""
    if not isinstance(text, str) or not text:
        return 0
    return len(WORD_RE.findall(text))


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
