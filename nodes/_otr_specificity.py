"""Prompt-only source-term context for line composition.

Optional key_terms are carried into the composer as authored context. This
module performs only whitespace cleanup, stable de-duplication, and a bounded
prompt-context cap. It does not classify nouns, entities, people, locations, or
objects, and it never participates in story acceptance or episode failure.
"""
from __future__ import annotations

import re
from typing import Any, List


MAX_ANCHORS = 5

_WS = re.compile(r"\s+")
_ANCHOR_BLOCK_HEAD = (
    "Optional source terms (use only when they fit naturally; do not force "
    "them into every line):"
)


def _as_str_list(items: Any) -> List[str]:
    """Coerce key-term-like input to clean nonempty strings."""
    if items is None:
        return []
    if isinstance(items, str):
        items = (items,)
    try:
        out: List[str] = []
        for item in items:
            value = _WS.sub(" ", str(item or "")).strip()
            if value:
                out.append(value)
        return out
    except TypeError:
        return []


def derive_specificity_anchors(key_terms: Any) -> List[str]:
    """Preserve up to five optional source terms without vocabulary judgment."""
    seen: set[str] = set()
    out: List[str] = []
    for term in _as_str_list(key_terms):
        folded = term.casefold()
        if folded in seen:
            continue
        seen.add(folded)
        out.append(term)
        if len(out) >= MAX_ANCHORS:
            break
    return out


def sanitize_for_prompt(value: Any) -> str:
    """Collapse controls and whitespace without inspecting vocabulary."""
    try:
        text = str(value or "")
        text = "".join(ch if ord(ch) >= 32 else " " for ch in text)
        return _WS.sub(" ", text).strip()
    except Exception:  # noqa: BLE001
        return ""


def inject_anchors_into_header(canon_header: Any, anchors: Any) -> str:
    """Append optional source terms to the model prompt, never as a gate."""
    header = str(canon_header or "")
    safe = [
        item
        for item in (sanitize_for_prompt(value) for value in _as_str_list(anchors))
        if item
    ]
    if not safe:
        return header
    block = _ANCHOR_BLOCK_HEAD + "\n- " + "\n- ".join(safe)
    return f"{header}\n\n{block}" if header else block
