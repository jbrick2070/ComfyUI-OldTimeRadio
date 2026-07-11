"""Fail-closed repair of model-owned sci-fi source metadata.

This helper repairs only deterministic metadata defects observed in a typed
P0 artifact: zero-padded identifiers and offsets for a quote that already
occurs verbatim in the supplied source payload. Unsupported evidence rows are
dropped; the helper never changes claims, spoken text, or other prose.
"""
from __future__ import annotations

import copy
import re
from typing import Any, Mapping

from pydantic import BaseModel


_PADDED_ID = re.compile(r"^(?P<prefix>[FEN])(?P<number>\d{1,2})$")
_SOURCE_FIELDS = ("headline", "summary", "full_text", "seed_text")


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
    substring of one unambiguous declared payload field. If it is paraphrased
    or absent, its evidence row is dropped; if no supported fact remains, the
    schema validation still fails loudly.
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
    invalid_span_ids: set[int] = set()

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
                if isinstance(quote, str) and isinstance(start, int):
                    same_field_positions = (
                        _occurrences(source, quote)
                        if isinstance(source, str) else []
                    )
                    if same_field_positions:
                        candidate_fields = [(field, same_field_positions)]
                    else:
                        candidate_fields = [
                            (candidate, positions)
                            for candidate in _SOURCE_FIELDS
                            if isinstance(payload.get(candidate), str)
                            and (positions := _occurrences(payload[candidate], quote))
                        ]
                        if len(candidate_fields) != 1:
                            invalid_span_ids.add(id(node))
                            changed = True
                            candidate_fields = []
                    if not candidate_fields:
                        for value in node.values():
                            visit(value)
                        return
                    corrected_field, positions = candidate_fields[0]
                    new_start = min(positions, key=lambda position: abs(position - start))
                    if node.get("field") != corrected_field or node.get("start") != new_start or node.get("end") != new_start + len(quote):
                        node["field"] = corrected_field
                        node["start"] = new_start
                        node["end"] = new_start + len(quote)
                        changed = True
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    def prune_unsupported_evidence() -> None:
        nonlocal changed
        for collection_name in ("facts", "verified_facts", "entities", "named_entities"):
            collection = repaired.get(collection_name)
            if not isinstance(collection, list):
                continue
            kept: list[Any] = []
            for row in collection:
                if not isinstance(row, dict) or not isinstance(row.get("source_spans"), list):
                    continue
                spans = [span for span in row["source_spans"] if id(span) not in invalid_span_ids]
                if spans:
                    row["source_spans"] = spans
                    kept.append(row)
                else:
                    changed = True
            if len(kept) != len(collection):
                repaired[collection_name] = kept
                changed = True
        facts = {
            row.get("fact_id")
            for row in repaired.get("facts", [])
            if isinstance(row, dict)
        }
        for collection_name in ("numbers", "key_numbers"):
            collection = repaired.get(collection_name)
            if not isinstance(collection, list):
                continue
            kept = []
            for row in collection:
                span = row.get("source_span") if isinstance(row, dict) else None
                if not isinstance(row, dict) or not isinstance(span, dict) or id(span) in invalid_span_ids:
                    changed = True
                    continue
                if row.get("fact_id") not in facts:
                    changed = True
                    continue
                kept.append(row)
            repaired[collection_name] = kept

    try:
        visit(repaired)
        prune_unsupported_evidence()
        if not changed:
            return None
        return schema.model_validate(repaired)
    except Exception:
        return None
