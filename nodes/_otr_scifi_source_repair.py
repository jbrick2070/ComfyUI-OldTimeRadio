"""Fail-closed repair of model-owned sci-fi source metadata.

This helper repairs only deterministic metadata defects observed in a typed
P0 artifact: zero-padded identifiers and offsets for a quote that already
occurs verbatim in the supplied source payload. It never changes claims,
spoken text, or any other LLM-authored prose.
"""
from __future__ import annotations

import copy
import re
from typing import Any, Mapping

from pydantic import BaseModel


_PADDED_ID = re.compile(r"^(?P<prefix>[FEN])(?P<number>\d{1,2})$")


def _occurrences(text: str, needle: str) -> list[int]:
    if not needle:
        return []
    found: list[int] = []
    start = 0
    while True:
        index = text.find(needle, start)
        if index < 0:
            return found
        found.append(index)
        start = index + 1


def _normalize_id(value: Any, *, zero_padded: bool) -> Any:
    if not zero_padded or not isinstance(value, str):
        return value
    match = _PADDED_ID.fullmatch(value)
    if not match:
        return value
    number = int(match.group("number"))
    # The local model's bare F0/F1/F2 labels are zero-based positions, while
    # the v4 contract is one-based and zero-padded. Already two-digit IDs such
    # as F10 are canonical and must remain unchanged.
    if len(match.group("number")) == 1:
        number += 1
    if number > 12:
        return value
    return f"{match.group('prefix')}{number:02d}"


def repair_literal_source_metadata(
    failed_output: str,
    schema: type[BaseModel],
    payload: Mapping[str, str],
    *,
    zero_padded_ids: bool,
) -> BaseModel | None:
    """Return a validated metadata-only repair, or ``None`` to keep retry loud.

    A source span is repaired only when its existing quote is an exact
    substring of the declared payload field. If it is paraphrased or absent,
    this function refuses to guess and the normal typed-repair failure path
    remains active.
    """
    try:
        import json

        data = json.loads(failed_output)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    repaired = copy.deepcopy(data)
    changed = False

    def visit(node: Any) -> None:
        nonlocal changed
        if isinstance(node, dict):
            for key in ("fact_id", "entity_id", "number_id"):
                if key in node:
                    normalized = _normalize_id(node[key], zero_padded=zero_padded_ids)
                    if normalized != node[key]:
                        node[key] = normalized
                        changed = True
            if {"field", "start", "end", "quote"}.issubset(node):
                field = node.get("field")
                quote = node.get("quote")
                start = node.get("start")
                source = payload.get(field) if isinstance(field, str) else None
                if isinstance(source, str) and isinstance(quote, str) and isinstance(start, int):
                    if source[start:node.get("end", start)] != quote:
                        positions = _occurrences(source, quote)
                        if not positions:
                            raise ValueError(f"quote is not literal in payload[{field!r}]")
                        new_start = min(positions, key=lambda position: abs(position - start))
                        node["start"] = new_start
                        node["end"] = new_start + len(quote)
                        changed = True
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    try:
        visit(repaired)
        if not changed:
            return None
        return schema.model_validate(repaired)
    except Exception:
        return None
