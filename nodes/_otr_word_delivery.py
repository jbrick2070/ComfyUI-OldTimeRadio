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
from dataclasses import dataclass, field
from typing import Any, Mapping, MutableMapping

try:
    from ._otr_text_metrics import canonical_word_count
except ImportError:  # pragma: no cover - direct custom-node import
    from _otr_text_metrics import canonical_word_count  # type: ignore


LOW_RATIO_EXCLUSIVE = 0.90
HIGH_RATIO_INCLUSIVE = 1.11
MAX_CONSECUTIVE_REPAIR_STALLS = 4


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


@dataclass
class WordFitLivenessController:
    """Finite progress/stall law for authored word-delivery repair.

    A fixed cycle estimate is unsound because a valid row repair may improve
    the integer distance by fewer words than requested.  This controller never
    charges strict progress against a guessed allowance.  Instead, accepted
    progress strictly decreases a non-negative integer and resets the stall
    counter; only consecutive invalid/duplicate/no-progress cycles exhaust.

    The loop is still provably finite.  For initial distance ``D`` and stall
    limit ``S``, at most ``D * S`` cycles can occur: before each possible
    one-word improvement there can be at most ``S - 1`` stalls, while ``S``
    consecutive stalls terminates the run.
    """

    lower_words: int
    upper_words: int
    initial_words: int
    initial_fingerprint: str = ""
    max_consecutive_stalls: int = MAX_CONSECUTIVE_REPAIR_STALLS
    current_words: int = field(init=False)
    initial_distance: int = field(init=False)
    current_distance: int = field(init=False)
    total_cycles: int = field(default=0, init=False)
    progress_cycles: int = field(default=0, init=False)
    stall_cycles: int = field(default=0, init=False)
    consecutive_stalls: int = field(default=0, init=False)
    fingerprints: set[str] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        self.lower_words = int(self.lower_words)
        self.upper_words = int(self.upper_words)
        self.initial_words = int(self.initial_words)
        self.max_consecutive_stalls = int(self.max_consecutive_stalls)
        if self.lower_words < 1 or self.upper_words < self.lower_words:
            raise WordDeliveryError("word-fit liveness bounds are invalid")
        if self.max_consecutive_stalls < 1:
            raise WordDeliveryError(
                "word-fit max_consecutive_stalls must be positive"
            )
        self.current_words = self.initial_words
        self.initial_distance = word_band_distance(
            self.initial_words, self.lower_words, self.upper_words,
        )
        self.current_distance = self.initial_distance
        fingerprint = str(self.initial_fingerprint or "").strip()
        if fingerprint:
            self.fingerprints.add(fingerprint)

    @property
    def cycle_ceiling(self) -> int:
        return self.initial_distance * self.max_consecutive_stalls

    @property
    def in_band(self) -> bool:
        return self.current_distance == 0

    @property
    def exhausted(self) -> bool:
        return (
            not self.in_band
            and self.consecutive_stalls >= self.max_consecutive_stalls
        )

    @property
    def can_continue(self) -> bool:
        return not self.in_band and not self.exhausted

    def observe(
        self,
        candidate_words: int | None,
        *,
        fingerprint: str = "",
    ) -> dict[str, Any]:
        """Judge one complete producer cycle and update liveness state.

        ``candidate_words=None`` records a cycle in which every producer slot
        was invalid.  The controller owns only arithmetic and duplicate-state
        detection; producer adapters remain responsible for validating prose,
        hashes, bindings, and spoken hygiene before calling this method.
        """
        if not self.can_continue:
            raise WordDeliveryError(
                "word-fit liveness observation arrived after termination"
            )
        before_distance = self.current_distance
        normalized_fingerprint = str(fingerprint or "").strip()
        duplicate = bool(
            normalized_fingerprint
            and normalized_fingerprint in self.fingerprints
        )
        candidate_distance = (
            before_distance
            if candidate_words is None
            else word_band_distance(
                int(candidate_words), self.lower_words, self.upper_words,
            )
        )
        strict_progress = (
            candidate_words is not None
            and candidate_distance < before_distance
            and not duplicate
        )
        self.total_cycles += 1
        if normalized_fingerprint:
            self.fingerprints.add(normalized_fingerprint)
        if strict_progress:
            self.current_words = int(candidate_words)
            self.current_distance = candidate_distance
            self.progress_cycles += 1
            self.consecutive_stalls = 0
        else:
            self.stall_cycles += 1
            self.consecutive_stalls += 1
        return {
            "strict_progress": strict_progress,
            "duplicate_fingerprint": duplicate,
            "before_distance": before_distance,
            "candidate_distance": candidate_distance,
            "after_distance": self.current_distance,
            "consecutive_stalls": self.consecutive_stalls,
            "exhausted": self.exhausted,
        }

    def receipt(self) -> dict[str, Any]:
        return {
            "initial_distance": self.initial_distance,
            "final_distance": self.current_distance,
            "total_cycles": self.total_cycles,
            "progress_cycles": self.progress_cycles,
            "stall_cycles": self.stall_cycles,
            "consecutive_stalls": self.consecutive_stalls,
            "max_consecutive_stalls": self.max_consecutive_stalls,
            "theoretical_cycle_ceiling": self.cycle_ceiling,
            "fingerprint_count": len(self.fingerprints),
        }


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
    """Compatibility view of the progress/stall proof's cycle ceiling.

    Producers must use :class:`WordFitLivenessController` for termination; this
    integer is receipt/test arithmetic, not a fixed ``range(...)`` allowance.
    """
    del target_words
    distance = word_band_distance(actual_words, lower_words, upper_words)
    return distance * MAX_CONSECUTIVE_REPAIR_STALLS


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


def _outer_liveness_state(
    meta: MutableMapping[str, Any], *, owner: str,
) -> MutableMapping[str, Any]:
    """Return the shared producer-candidate campaign receipt.

    Four consecutive stalls retire one authored candidate, never the episode.
    The campaign therefore has deliberately *no* outer model-output ceiling:
    producers keep authoring fresh candidates until the deterministic final
    ledger stamp accepts one (or the operator cancels the run).  Individual
    model calls and candidate fit loops remain bounded by their own contracts.
    """
    if not isinstance(meta, MutableMapping):
        raise WordDeliveryError("ledger meta must be mutable")
    normalized_owner = str(owner or "").strip()
    if not normalized_owner:
        raise WordDeliveryError("outer word-fit owner is required")
    state = meta.setdefault("outer_liveness_retry", {})
    if not isinstance(state, MutableMapping):
        raise WordDeliveryError("meta.outer_liveness_retry must be an object")
    prior_owner = str(state.get("owner") or "").strip()
    if prior_owner and prior_owner != normalized_owner:
        raise WordDeliveryError(
            "meta.outer_liveness_retry owner changed from "
            f"{prior_owner!r} to {normalized_owner!r}"
        )
    state.setdefault("schema_version", 1)
    state["owner"] = normalized_owner
    state.setdefault(
        "policy", "unbounded_model_output_retries_until_ledger_legal",
    )
    state.setdefault("active_candidate_index", 0)
    state.setdefault("discarded_candidates", [])
    if not isinstance(state["discarded_candidates"], list):
        raise WordDeliveryError(
            "meta.outer_liveness_retry.discarded_candidates must be a list"
        )
    return state


def retire_word_fit_candidate(
    meta: MutableMapping[str, Any], *, owner: str, boundary: str,
    reason: str, fingerprint: str = "",
    receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Archive one exhausted model candidate and open the next attempt.

    Only candidate-local evidence is retained.  No discarded text, hash, seal,
    or readiness verdict is allowed to become authoritative downstream.
    """
    state = _outer_liveness_state(meta, owner=owner)
    candidate_index = int(state.get("active_candidate_index") or 0)
    source = receipt if isinstance(receipt, Mapping) else {}
    row: dict[str, Any] = {
        "candidate_index": candidate_index,
        "boundary": str(boundary or "producer_candidate"),
        "discard_reason": str(reason or "model_output_exhausted")[:600],
        "fingerprint": str(fingerprint or ""),
    }
    for key in (
        "status", "target_words", "acceptance_words",
        "initial_character_words", "final_character_words",
        "repair_budget", "capacity", "liveness",
    ):
        if key in source:
            row[key] = source[key]
    cycles = source.get("cycles")
    if isinstance(cycles, (list, tuple)):
        row["cycles_attempted"] = len(cycles)
    state["discarded_candidates"].append(row)
    state["active_candidate_index"] = candidate_index + 1
    state["status"] = "authoring_candidate"
    state.pop("authoritative_candidate", None)

    # Candidate rerolls happen before delivery, but imported/re-entered ledgers
    # can carry stale consumer verdicts.  Remove them rather than mutating them
    # into a second source of truth; the real freeze/readiness owners will stamp
    # fresh values only after the legal candidate is accepted.
    for key in (
        "freeze_verdict", "freeze_report", "freeze_audit",
        "audio_ready", "video_ready", "media_ready", "obs_ready",
    ):
        meta.pop(key, None)
    return dict(row)


def accept_word_fit_candidate(
    meta: MutableMapping[str, Any], *, owner: str, boundary: str,
    fingerprint: str, actual_words: int,
    acceptance_words: tuple[int, int] | list[int],
) -> dict[str, Any]:
    """Mark the candidate proven by the caller's hard final ledger stamp."""
    state = _outer_liveness_state(meta, owner=owner)
    if len(acceptance_words) != 2:
        raise WordDeliveryError("acceptance_words must contain two bounds")
    lower, upper = int(acceptance_words[0]), int(acceptance_words[1])
    actual = int(actual_words)
    if not lower <= actual <= upper:
        raise WordDeliveryError(
            f"cannot accept out-of-band candidate {actual}; "
            f"required {lower}..{upper}"
        )
    row = {
        "candidate_index": int(state.get("active_candidate_index") or 0),
        "boundary": str(boundary or "producer_candidate"),
        "fingerprint": str(fingerprint or ""),
        "actual_character_words": actual,
        "acceptance_words": [lower, upper],
        "status": "pass",
    }
    state["status"] = "pass"
    state["authoritative_candidate"] = row
    return dict(row)


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
    "accept_word_fit_candidate",
    "HIGH_RATIO_INCLUSIVE", "LOW_RATIO_EXCLUSIVE",
    "MAX_CONSECUTIVE_REPAIR_STALLS", "WordDeliveryError",
    "WordDeliverySpec", "WordFitLivenessController", "character_text_sha256",
    "character_word_count", "delivery_repair_cycle_budget",
    "delivery_step_words", "delivery_word_bounds", "resolve_spec",
    "retire_word_fit_candidate",
    "stamp_actual", "stamp_contract", "word_band_distance",
]
