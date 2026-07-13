"""Shared strict RSS source-admission contract for the Sci-Fi v4 lanes.

The selector and all v4 payload envelopes must agree on this exact floor.
This is deliberately a stdlib-only leaf module: importing it cannot load a
model, touch ComfyUI state, or alter the fetched source string.
"""
from __future__ import annotations

import re


MIN_CHARS = 400
MIN_WORDS = 80
MIN_UNIQUE_TOKENS = 12


def meets_v4_rss_source_floor(
    text: object,
    *,
    min_chars: int = MIN_CHARS,
    min_words: int = MIN_WORDS,
    min_unique_tokens: int = MIN_UNIQUE_TOKENS,
) -> bool:
    """Return whether *text* meets the strict v4 RSS source floor.

    Words are alphanumeric runs, matching the production selector's historic
    semantics. Non-string payloads fail closed rather than raising while a
    fetch batch is being evaluated.
    """
    if not isinstance(text, str):
        return False
    tokens = re.findall(r"[A-Za-z0-9]+", text.lower())
    return (
        len(text) >= min_chars
        and len(tokens) >= min_words
        and len(set(tokens)) >= min_unique_tokens
    )
