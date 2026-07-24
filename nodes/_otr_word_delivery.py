"""Shared actual-word telemetry for every OTR story producer.

Requested length may guide the first authoring prompt, but this module never
classifies the accepted story as near, far, under, over, passing, or failing.
It records the requested target as context, then stamps only canonical actual
counts and text hashes. There is no retry, candidate, mutation, range, ratio,
drift, or rejection API.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterator, Mapping, MutableMapping

try:
    from ._otr_text_metrics import canonical_word_count
except ImportError:  # pragma: no cover - direct custom-node import
    from _otr_text_metrics import canonical_word_count  # type: ignore


_VOICED_ROLES = frozenset({"character", "announcer"})
class WordDeliveryError(ValueError):
    """Invalid requested-length metadata supplied by a producer."""


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if number <= 0:
        return None
    try:
        if float(value) != float(number):
            return None
    except (TypeError, ValueError, OverflowError):
        return None
    return number



def stamp_contract(
    meta: MutableMapping[str, Any],
    *,
    target_words: int,
    owner: str,
    planned_voiced_words: int | None = None,
) -> dict[str, Any]:
    """Persist requested-length context without any acceptance judgment."""
    if not isinstance(meta, MutableMapping):
        raise WordDeliveryError("ledger meta must be mutable")
    target = _positive_int(target_words)
    if target is None:
        raise WordDeliveryError("target_words must be a positive integer")
    planned = target if planned_voiced_words is None else int(planned_voiced_words)
    receipt = meta.get("word_budget")
    if not isinstance(receipt, MutableMapping):
        receipt = {}
        meta["word_budget"] = receipt
    receipt.update({
        "schema_version": 4,
        "target_words": target,
        "planned_voiced_words": planned,
        "owner": str(owner or "").strip(),
        "policy": "requested_context_actual_count_only",
    })
    return dict(receipt)


def _iter_voiced_rows(
    ledger_data: Mapping[str, Any],
    *,
    roles: frozenset[str],
) -> Iterator[Mapping[str, Any]]:
    lines = ledger_data.get("lines") if isinstance(ledger_data, Mapping) else ()
    for row in lines or ():
        if (
            isinstance(row, Mapping)
            and not bool(row.get("skip"))
            and str(row.get("speaker_role") or "") in roles
        ):
            yield row


def _word_count_for_roles(
    ledger_data: Mapping[str, Any],
    *,
    roles: frozenset[str],
) -> int:
    return sum(
        canonical_word_count(row.get("text"))
        for row in _iter_voiced_rows(ledger_data, roles=roles)
    )


def character_word_count(ledger_data: Mapping[str, Any]) -> int:
    return _word_count_for_roles(ledger_data, roles=frozenset({"character"}))


def announcer_word_count(ledger_data: Mapping[str, Any]) -> int:
    return _word_count_for_roles(ledger_data, roles=frozenset({"announcer"}))


def total_voiced_word_count(ledger_data: Mapping[str, Any]) -> int:
    return _word_count_for_roles(ledger_data, roles=_VOICED_ROLES)


def _text_sha256_for_roles(
    ledger_data: Mapping[str, Any],
    *,
    roles: frozenset[str],
) -> str:
    payload = [
        {
            "line_id": str(row.get("line_id") or ""),
            "speaker_role": str(row.get("speaker_role") or ""),
            "text": str(row.get("text") or ""),
        }
        for row in _iter_voiced_rows(ledger_data, roles=roles)
    ]
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def character_text_sha256(ledger_data: Mapping[str, Any]) -> str:
    return _text_sha256_for_roles(
        ledger_data,
        roles=frozenset({"character"}),
    )


def announcer_text_sha256(ledger_data: Mapping[str, Any]) -> str:
    return _text_sha256_for_roles(
        ledger_data,
        roles=frozenset({"announcer"}),
    )


def total_voiced_text_sha256(ledger_data: Mapping[str, Any]) -> str:
    return _text_sha256_for_roles(ledger_data, roles=_VOICED_ROLES)


def stamp_actual(
    ledger_data: MutableMapping[str, Any],
    *,
    stage: str,
) -> dict[str, Any]:
    """Stamp canonical actual counts and hashes without evaluating length."""
    if not isinstance(ledger_data, MutableMapping):
        raise TypeError("ledger_data must be mutable")

    character_words = character_word_count(ledger_data)
    announcer_words = announcer_word_count(ledger_data)
    total_words = total_voiced_word_count(ledger_data)
    character_hash = character_text_sha256(ledger_data)

    meta = ledger_data.get("meta")
    if not isinstance(meta, MutableMapping):
        meta = {}
        ledger_data["meta"] = meta

    original_budget = meta.get("word_budget")
    malformed_budget = (
        original_budget is not None
        and not isinstance(original_budget, MutableMapping)
    )
    if isinstance(original_budget, MutableMapping):
        budget = original_budget
    else:
        budget = {}
        meta["word_budget"] = budget

    target = _positive_int(budget.get("target_words"))
    if malformed_budget:
        target_status = "invalid"
        target_error = "meta.word_budget is invalid"
    elif target is not None:
        target_status = "valid"
        target_error = ""
    elif budget:
        target_status = "invalid"
        target_error = "meta.word_budget.target_words is invalid"
    else:
        target_status = "missing"
        target_error = ""

    result: dict[str, Any] = {
        "schema_version": 4,
        "stage": str(stage or "final"),
        "target_status": target_status,
        "target_words": target,
        "actual_character_words": character_words,
        "actual_announcer_words": announcer_words,
        "actual_total_voiced_words": total_words,
        "actual_character_text_sha256": character_hash,
        "actual_announcer_text_sha256": announcer_text_sha256(ledger_data),
        "actual_total_voiced_text_sha256": total_voiced_text_sha256(ledger_data),
        "actual_voiced_words": character_words,
        "actual_text_sha256": character_hash,
    }
    if target_error:
        result["target_error"] = target_error

    budget.update(result)
    receipts = budget.get("actual_receipts")
    if not isinstance(receipts, MutableMapping):
        receipts = {}
        budget["actual_receipts"] = receipts
    receipts[result["stage"]] = dict(result)

    meta["character_word_count"] = character_words
    meta["announcer_word_count"] = announcer_words
    meta["total_word_count"] = total_words
    return result


__all__ = [
    "WordDeliveryError",
    "announcer_text_sha256",
    "announcer_word_count",
    "character_text_sha256",
    "character_word_count",
    "stamp_actual",
    "stamp_contract",
    "total_voiced_text_sha256",
    "total_voiced_word_count",
]
