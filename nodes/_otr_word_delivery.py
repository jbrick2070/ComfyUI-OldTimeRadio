"""Shared actual-word telemetry for every OTR story producer.

WORDS ARE AN OBSERVATION HERE, AND ONLY AN OBSERVATION. This module never
classifies the accepted story as near, far, under, over, passing or failing,
and since 2026-08-14 it does not record a requested target either -- there is
no longer one to record. It stamps canonical actual counts and text hashes,
attributed to the lane that produced them. There is no retry, candidate,
mutation, range, ratio, drift or rejection API.
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


# (lean-mean 2026-08-22) ``_positive_int`` was here and is DELETED. It validated
# a REQUESTED word length -- and since 2026-08-14 nothing is requested up front,
# so its only caller went with the word authority. The module docstring above
# already says there is no target to record; this removes the last helper that
# still existed to police one.


def stamp_contract(
    meta: MutableMapping[str, Any],
    *,
    owner: str,
) -> dict[str, Any]:
    """Record WHO owns this episode's word accounting. No target.

    2026-08-14: `target_words` and `planned_voiced_words` were removed with
    the word authority. Nothing is requested up front any more, so there is
    no target to persist and no planned per-beat allocation to record --
    `Beat.target_words` no longer exists either.

    The receipt survives because `owner` is genuinely read (the freeze
    cascade attributes the actual counts to a lane) and because
    `stamp_actual` fills the same dict with what the episode TURNED OUT to
    be. Requested length left; observed length stayed.
    """
    if not isinstance(meta, MutableMapping):
        raise WordDeliveryError("ledger meta must be mutable")
    receipt = meta.get("word_budget")
    if not isinstance(receipt, MutableMapping):
        receipt = {}
        meta["word_budget"] = receipt
    receipt.update({
        "schema_version": 5,
        "owner": str(owner or "").strip(),
        "policy": "actual_count_only",
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

    # 2026-08-14: there is no target to compare against. `target_status`
    # survives as a schema field so existing readers keep parsing, and it
    # reports the only honest value: nothing was requested. A malformed
    # receipt is still called out rather than silently normalised.
    if malformed_budget:
        target_status = "invalid"
        target_error = "meta.word_budget is invalid"
    else:
        target_status = "not_requested"
        target_error = ""

    result: dict[str, Any] = {
        # 5, not 4: `stamp_contract` writes 5 and this dict is merged OVER
        # the same receipt, so a stale 4 here silently demoted every
        # published ledger's word_budget schema back a version.
        "schema_version": 5,
        "stage": str(stage or "final"),
        "target_status": target_status,
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
