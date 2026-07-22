"""Shared explicit word-delivery contract for every OTR story producer.

Story taste is advisory, but a requested delivery length is measurable.  This
module owns only the common arithmetic, receipts, progress budget, and final
read-only verification.  Each producer still owns how its text is authored and
where its hashes/proof maps are rebuilt.

The operator-approved 180-word window is 163..200.  The same integer law scales
to every supported request: strictly above 90 percent at the low edge and at or
below the ceiling of 111 percent at the high edge (320 -> 289..356).
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping

try:
    from ._otr_text_metrics import canonical_word_count
except ImportError:  # pragma: no cover - direct custom-node import
    from _otr_text_metrics import canonical_word_count  # type: ignore


LOW_RATIO_EXCLUSIVE = 0.90
HIGH_RATIO_INCLUSIVE = 1.11
MAX_REPAIR_CYCLES = 18


class WordDeliveryError(RuntimeError):
    """An explicit requested-length contract was not met before delivery."""


@dataclass(frozen=True)
class WordDeliverySpec:
    target_words: int
    lower_words: int
    upper_words: int
    owner: str

    def contains(self, actual_words: int) -> bool:
        return self.lower_words <= int(actual_words) <= self.upper_words


def delivery_word_bounds(target_words: int) -> tuple[int, int]:
    """Return inclusive integer bounds for one positive word target."""
    if isinstance(target_words, bool):
        raise WordDeliveryError("target_words must be a positive integer")
    try:
        target = int(target_words)
    except (TypeError, ValueError, OverflowError) as exc:
        raise WordDeliveryError("target_words must be a positive integer") from exc
    if target <= 0:
        raise WordDeliveryError("target_words must be a positive integer")
    lower = int(math.floor(target * LOW_RATIO_EXCLUSIVE)) + 1
    upper = int(math.ceil(target * HIGH_RATIO_INCLUSIVE))
    return max(1, lower), max(lower, upper)


def word_band_distance(actual_words: int, lower_words: int,
                       upper_words: int) -> int:
    actual = int(actual_words)
    if actual < int(lower_words):
        return int(lower_words) - actual
    if actual > int(upper_words):
        return actual - int(upper_words)
    return 0


def delivery_step_words(target_words: int) -> int:
    """Small one-row correction size used to dimension dynamic loops."""
    return max(5, min(12, int(round(int(target_words) * 0.035))))


def delivery_repair_cycle_budget(*, actual_words: int, target_words: int,
                                 lower_words: int, upper_words: int) -> int:
    """Finite dynamic budget with room for failed/no-progress row attempts."""
    distance = word_band_distance(actual_words, lower_words, upper_words)
    if distance == 0:
        return 0
    progress_cycles = int(math.ceil(
        distance / max(1, delivery_step_words(target_words))
    ))
    return min(MAX_REPAIR_CYCLES, max(6, progress_cycles + 4))


def stamp_contract(meta: MutableMapping[str, Any], *, target_words: int,
                   owner: str, planned_voiced_words: int | None = None) -> dict:
    """Stamp the common bounds while retaining producer planning fields."""
    if not isinstance(meta, MutableMapping):
        raise WordDeliveryError("ledger meta must be mutable")
    target = int(target_words)
    lower, upper = delivery_word_bounds(target)
    receipt = meta.setdefault("word_budget", {})
    if not isinstance(receipt, MutableMapping):
        raise WordDeliveryError("meta.word_budget must be an object")
    planned = target if planned_voiced_words is None else int(planned_voiced_words)
    receipt.update({
        "target_words": target,
        "planned_voiced_words": planned,
        "planned_ratio": round(planned / max(1, target), 3),
        "band": [lower / float(target), upper / float(target)],
        "acceptance_words": [lower, upper],
        "planned_drift": not (lower <= planned <= upper),
        "owner": str(owner or "").strip(),
    })
    if not receipt["owner"]:
        raise WordDeliveryError("word-budget owner is required")
    return dict(receipt)


def _valid_positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number) or not number.is_integer() or number <= 0:
        return None
    return int(number)


def resolve_spec(ledger_data: Mapping[str, Any]) -> WordDeliverySpec:
    """Resolve the persisted producer declaration without changing it."""
    if not isinstance(ledger_data, Mapping):
        raise WordDeliveryError("ledger must be an object")
    meta = ledger_data.get("meta")
    receipt = meta.get("word_budget") if isinstance(meta, Mapping) else None
    if not isinstance(receipt, Mapping):
        raise WordDeliveryError("meta.word_budget is required")
    target = _valid_positive_int(receipt.get("target_words"))
    if target is None:
        raise WordDeliveryError("meta.word_budget.target_words is invalid")
    owner = str(receipt.get("owner") or "").strip()
    if not owner:
        raise WordDeliveryError("meta.word_budget.owner is required")

    acceptance = receipt.get("acceptance_words")
    lower = upper = None
    if isinstance(acceptance, (list, tuple)) and len(acceptance) == 2:
        lower = _valid_positive_int(acceptance[0])
        upper = _valid_positive_int(acceptance[1])
    if lower is None or upper is None or not lower <= target <= upper:
        # Compatibility for an older persisted producer receipt. New writers
        # always stamp acceptance_words; do not silently reinterpret a valid
        # historical ratio declaration when validating a loaded ledger.
        band = receipt.get("band")
        if not isinstance(band, (list, tuple)) or len(band) != 2:
            raise WordDeliveryError(
                "meta.word_budget.acceptance_words/band is invalid"
            )
        try:
            low_ratio, high_ratio = float(band[0]), float(band[1])
        except (TypeError, ValueError, OverflowError) as exc:
            raise WordDeliveryError("meta.word_budget.band is invalid") from exc
        if not (
            math.isfinite(low_ratio) and math.isfinite(high_ratio)
            and 0.0 < low_ratio <= 1.0 <= high_ratio
        ):
            raise WordDeliveryError("meta.word_budget.band is invalid")
        lower = max(1, int(math.ceil(target * low_ratio)))
        upper = max(lower, int(math.floor(target * high_ratio)))
    return WordDeliverySpec(target, int(lower), int(upper), owner)


def character_word_count(ledger_data: Mapping[str, Any]) -> int:
    total = 0
    for row in ledger_data.get("lines") or ():
        if (
            not isinstance(row, Mapping)
            or bool(row.get("skip"))
            or str(row.get("speaker_role") or "") != "character"
        ):
            continue
        total += canonical_word_count(row.get("text"))
    return total


def character_text_sha256(ledger_data: Mapping[str, Any]) -> str:
    payload = [
        {
            "line_id": str(row.get("line_id") or ""),
            "text": str(row.get("text") or ""),
        }
        for row in ledger_data.get("lines") or ()
        if isinstance(row, Mapping)
        and not bool(row.get("skip"))
        and str(row.get("speaker_role") or "") == "character"
    ]
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def stamp_actual(ledger_data: MutableMapping[str, Any], *, stage: str,
                 require_in_band: bool = False) -> dict:
    """Recount the exact current rows and stamp a hash-bound receipt."""
    spec = resolve_spec(ledger_data)
    actual = character_word_count(ledger_data)
    drift = not spec.contains(actual)
    meta = ledger_data.setdefault("meta", {})
    receipt = meta["word_budget"]
    if not isinstance(receipt, MutableMapping):
        raise WordDeliveryError("meta.word_budget must be mutable")
    result = {
        "stage": str(stage or "final"),
        "actual_voiced_words": actual,
        "actual_ratio": round(actual / max(1, spec.target_words), 3),
        "actual_drift": drift,
        "actual_text_sha256": character_text_sha256(ledger_data),
        "acceptance_words": [spec.lower_words, spec.upper_words],
    }
    receipt.update(result)
    receipts = receipt.setdefault("actual_receipts", {})
    if not isinstance(receipts, MutableMapping):
        raise WordDeliveryError(
            "meta.word_budget.actual_receipts must be mutable"
        )
    receipts[result["stage"]] = dict(result)
    if require_in_band and drift:
        raise WordDeliveryError(
            f"{spec.owner} word delivery missed {spec.lower_words}.."
            f"{spec.upper_words}: actual {actual} at {result['stage']}"
        )
    return result


__all__ = [
    "HIGH_RATIO_INCLUSIVE", "LOW_RATIO_EXCLUSIVE", "MAX_REPAIR_CYCLES",
    "WordDeliveryError", "WordDeliverySpec", "character_text_sha256",
    "character_word_count", "delivery_repair_cycle_budget",
    "delivery_step_words", "delivery_word_bounds", "resolve_spec",
    "stamp_actual", "stamp_contract", "word_band_distance",
]
