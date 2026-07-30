"""Fail-closed repair of model-owned sci-fi source metadata.

This helper repairs only deterministic metadata defects observed in a typed
P0 artifact: zero-padded identifiers and offsets for a quote that already
occurs verbatim in the supplied source payload. Unsupported evidence rows are
dropped; the helper never changes claims, spoken text, or other prose.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from typing import Any, Collection, Mapping, MutableMapping

from pydantic import BaseModel


_PADDED_ID = re.compile(r"^(?P<prefix>[FEN])(?P<number>\d{1,2})$")
_SOURCE_FIELDS = ("headline", "summary", "full_text", "seed_text")
_EVIDENCE_COLLECTIONS = (
    "facts", "verified_facts", "entities", "named_entities",
)
_DEPENDENT_COLLECTIONS = ("numbers", "key_numbers")
_RECEIPT_SCHEMA_VERSION = (
    "scifi_codex.literal_source_repair_receipt.v1"
)


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


def _collection_counts(data: Mapping[str, Any]) -> dict[str, int]:
    return {
        name: len(collection)
        for name in (*_EVIDENCE_COLLECTIONS, *_DEPENDENT_COLLECTIONS)
        if isinstance((collection := data.get(name)), list)
    }


def _row_id(row: Mapping[str, Any]) -> Any:
    for key in ("number_id", "entity_id", "fact_id"):
        if key in row:
            return row.get(key)
    return None


def _quote_sha256(quote: str) -> str:
    return hashlib.sha256(quote.encode("utf-8")).hexdigest()


def repair_literal_source_metadata(
    failed_output: str,
    schema: type[BaseModel],
    payload: Mapping[str, str],
    *,
    zero_padded_ids: bool,
    max_quote_chars: int | None = None,
    allowed_source_fields: Collection[str] | None = None,
    repair_receipt: MutableMapping[str, Any] | None = None,
) -> BaseModel | None:
    """Return a validated metadata-only repair, or ``None`` to keep retry loud.

    A source span is repaired only when its existing quote is an exact
    substring of one unambiguous allowed payload field. The optional allowlist
    constrains both same-field acceptance and cross-field rehoming. If a quote
    is paraphrased or absent, its evidence row is dropped; if no supported fact
    remains, schema validation still fails loudly. An exact literal quote that
    is wider than a finite schema cap may be narrowed only to the corresponding
    source prefix at its repaired coordinate. Claims and every other
    model-authored field remain byte-identical.
    """
    if (
        max_quote_chars is not None
        and (
            isinstance(max_quote_chars, bool)
            or not isinstance(max_quote_chars, int)
            or max_quote_chars < 1
        )
    ):
        raise ValueError("max_quote_chars must be a positive integer or None")
    try:
        data = json.loads(failed_output)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    repaired = copy.deepcopy(data)
    before_counts = _collection_counts(repaired)
    candidate_source_fields = tuple(
        field
        for field in _SOURCE_FIELDS
        if allowed_source_fields is None or field in allowed_source_fields
    )
    changed = False
    invalid_span_reasons: dict[int, str] = {}
    drops: list[dict[str, Any]] = []

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
                source = (
                    payload.get(field)
                    if field in candidate_source_fields else None
                )
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
                            for candidate in candidate_source_fields
                            if isinstance(payload.get(candidate), str)
                            and (positions := _occurrences(payload[candidate], quote))
                        ]
                        if len(candidate_fields) != 1:
                            invalid_span_reasons[id(node)] = (
                                "quote_not_literal"
                                if not candidate_fields
                                else "ambiguous_source_field"
                            )
                            changed = True
                            candidate_fields = []
                    if not candidate_fields:
                        for value in node.values():
                            visit(value)
                        return
                    corrected_field, positions = candidate_fields[0]
                    new_start = min(positions, key=lambda position: abs(position - start))
                    corrected_quote = quote
                    if (
                        max_quote_chars is not None
                        and len(corrected_quote) > max_quote_chars
                    ):
                        # The full model quote has already been proven literal by
                        # ``_occurrences`` above. Source-span text is deterministic
                        # metadata, so preserve its repaired coordinate and retain
                        # only the finite literal prefix the schema permits.
                        corrected_quote = payload[corrected_field][
                            new_start:new_start + max_quote_chars
                        ]
                    if not corrected_quote:
                        invalid_span_reasons[id(node)] = "quote_not_literal"
                        changed = True
                        for value in node.values():
                            visit(value)
                        return
                    new_end = new_start + len(corrected_quote)
                    if (
                        node.get("field") != corrected_field
                        or node.get("start") != new_start
                        or node.get("end") != new_end
                        or node.get("quote") != corrected_quote
                    ):
                        node["field"] = corrected_field
                        node["start"] = new_start
                        node["end"] = new_end
                        node["quote"] = corrected_quote
                        changed = True
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    def prune_unsupported_evidence() -> None:
        nonlocal changed
        for collection_name in _EVIDENCE_COLLECTIONS:
            collection = repaired.get(collection_name)
            if not isinstance(collection, list):
                continue
            kept: list[Any] = []
            for row in collection:
                if not isinstance(row, dict) or not isinstance(row.get("source_spans"), list):
                    drops.append({
                        "action": "dropped_evidence_row",
                        "collection": collection_name,
                        "row_id": _row_id(row) if isinstance(row, dict) else None,
                        "reason": "malformed_evidence_row",
                    })
                    changed = True
                    continue
                spans: list[Any] = []
                malformed = False
                for span_index, span in enumerate(row["source_spans"]):
                    if (
                        not isinstance(span, dict)
                        or not {"field", "start", "end", "quote"}.issubset(span)
                    ):
                        malformed = True
                        break
                    reason = invalid_span_reasons.get(id(span))
                    if reason is None:
                        spans.append(span)
                        continue
                    drops.append({
                        "action": "dropped_source_span",
                        "collection": collection_name,
                        "row_id": _row_id(row),
                        "span_index": span_index,
                        "field": span.get("field"),
                        "quote_sha256": _quote_sha256(span["quote"]),
                        "reason": reason,
                    })
                if malformed:
                    drops.append({
                        "action": "dropped_evidence_row",
                        "collection": collection_name,
                        "row_id": _row_id(row),
                        "reason": "malformed_evidence_row",
                    })
                    changed = True
                    continue
                if spans:
                    row["source_spans"] = spans
                    kept.append(row)
                else:
                    drops.append({
                        "action": "dropped_evidence_row",
                        "collection": collection_name,
                        "row_id": _row_id(row),
                        "reason": "no_literal_source_spans_remain",
                    })
                    changed = True
            if len(kept) != len(collection):
                repaired[collection_name] = kept
                changed = True
        facts = {
            row.get("fact_id")
            for row in repaired.get("facts", [])
            if isinstance(row, dict)
        }
        for collection_name in _DEPENDENT_COLLECTIONS:
            collection = repaired.get(collection_name)
            if not isinstance(collection, list):
                continue
            kept = []
            for row in collection:
                span = row.get("source_span") if isinstance(row, dict) else None
                if (
                    not isinstance(row, dict)
                    or not isinstance(span, dict)
                    or not {"field", "start", "end", "quote"}.issubset(span)
                ):
                    drops.append({
                        "action": "dropped_dependent_row",
                        "collection": collection_name,
                        "row_id": _row_id(row) if isinstance(row, dict) else None,
                        "fact_id": row.get("fact_id") if isinstance(row, dict) else None,
                        "reason": "malformed_evidence_row",
                    })
                    changed = True
                    continue
                reason = invalid_span_reasons.get(id(span))
                if reason is not None:
                    drops.append({
                        "action": "dropped_source_span",
                        "collection": collection_name,
                        "row_id": _row_id(row),
                        "span_index": 0,
                        "field": span.get("field"),
                        "quote_sha256": _quote_sha256(span["quote"]),
                        "reason": reason,
                    })
                    drops.append({
                        "action": "dropped_dependent_row",
                        "collection": collection_name,
                        "row_id": _row_id(row),
                        "fact_id": row.get("fact_id"),
                        "reason": "no_literal_source_spans_remain",
                    })
                    changed = True
                    continue
                if row.get("fact_id") not in facts:
                    drops.append({
                        "action": "dropped_dependent_row",
                        "collection": collection_name,
                        "row_id": _row_id(row),
                        "fact_id": row.get("fact_id"),
                        "reason": "fact_removed",
                    })
                    changed = True
                    continue
                kept.append(row)
            repaired[collection_name] = kept

    try:
        visit(repaired)
        prune_unsupported_evidence()
        if not changed:
            return None
        result = schema.model_validate(repaired)
        receipt = {
            "schema_version": _RECEIPT_SCHEMA_VERSION,
            "repairer": "repair_literal_source_metadata",
            "span_index_base": 0,
            "allowed_source_fields": [
                field
                for field in candidate_source_fields
                if isinstance(payload.get(field), str)
            ],
            "before_counts": before_counts,
            "after_counts": _collection_counts(repaired),
            "drops": drops,
        }
        if repair_receipt is not None:
            repair_receipt.clear()
            repair_receipt.update(receipt)
        return result
    except Exception:
        return None
