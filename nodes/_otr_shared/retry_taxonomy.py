"""Render-time retry taxonomy with an explicit block-class split (A-S7).

Every video render-time failure is classified into one of two BLOCK CLASSES,
and each class has a single deterministic action:

* ``BlockClass.HARD`` -- renderability / safety-integrity failures. The chosen
  engine cannot produce a usable clip for this shot. NO FALLBACKS (operator
  directive 2026-07-02: NO fallbacks / NO auto-defaults anywhere): a HARD
  failure is a LOUD STOP -- the render driver raises a named RenderError; there
  is NO engine swap, NO chain, NO floor. Per-kind retries on the SAME engine
  remain: ``crash_before_load`` retries once; ``corrupt_output`` retries once
  at the same seed then reseeds; ``transient_io`` does a small bounded retry;
  everything else fails on the first attempt.

* ``BlockClass.WARN`` -- subjective quality / coherence / NSFW gates (and the
  A/V-sync guard). These WARN only: the already-rendered clip is RETAINED and a
  warning is logged; a WARN gate NEVER discards rendered output, NEVER aborts
  an episode, and NEVER touches the frozen master audio. (The offline NSFW
  frame-QC sampler that used to be named here was REMOVED 2026-08-28; the
  FailureKind.NSFW value below is independent of it and stays -- an image
  model can still decline a card, and that is a WARN, not a stop.)
  The A/V-sync guard does a best-effort deterministic retime of the VIDEO
  frames (never the audio).

This mirrors the SHIPPED audio ``freeze_block_class`` structural/quality split
(``nodes/_otr_freeze_cascade.py``) on the render surface, but is a separate
module -- the frozen audio cascade is read-only and never imported here.

The module is PURE and dependency-free (stdlib only): it classifies a failure
and yields the deterministic :class:`RetryDecision`. The fallback-action API
(build_fallback_decision / restamp_shot_row / append_runtime_fallback_decision
/ format_swap_log) was DELETED in the Sprint A rip (2026-07-02); the ledger's
``runtime_fallback_decisions`` schema slot survives but is stamped never.
UTF-8, no BOM, ASCII-only source.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Dict


#: A bounded retry budget for a transient I/O blip before walking the fallback.
DEFAULT_TRANSIENT_IO_RETRIES = 3


class BlockClass(str, enum.Enum):
    """The two render-time block classes (A-S7).

    ``HARD`` == renderability / safety-integrity: the engine cannot yield a
    usable clip, so the render node walks the fallback chain to the radio floor
    (LOUD swap + ledger restamp). ``WARN`` == subjective quality / coherence /
    NSFW (plus the A/V-sync guard): warn only, keep the rendered output, never
    discard, never abort, never touch the frozen audio.
    """

    HARD = "hard"
    WARN = "warn"


class FailureKind(str, enum.Enum):
    """Every render-time failure / flag kind the taxonomy classifies.

    The first block is HARD (renderability / safety-integrity); the second is
    WARN (subjective gates + the A/V-sync guard). A new kind must be added to
    exactly one of :data:`HARD_KINDS` / :data:`WARN_KINDS`; a kind with no class
    is rejected fail-closed by :func:`classify`.
    """

    # --- HARD: renderability / safety-integrity ---
    DEPENDENCY_MISSING = "dependency_missing"
    ASSET_MISSING = "asset_missing"
    LICENSE_BLOCKED = "license_blocked"
    INVALID_DAG = "invalid_dag"
    OOM = "oom"
    TIMEOUT = "timeout"
    CRASH_BEFORE_LOAD = "crash_before_load"
    CORRUPT_OUTPUT = "corrupt_output"
    TRANSIENT_IO = "transient_io"
    # --- WARN: subjective quality / coherence / NSFW + A/V-sync ---
    QUALITY = "quality"
    COHERENCE = "coherence"
    NSFW = "nsfw"
    AV_SYNC = "av_sync"


HARD_KINDS = frozenset({
    FailureKind.DEPENDENCY_MISSING,
    FailureKind.ASSET_MISSING,
    FailureKind.LICENSE_BLOCKED,
    FailureKind.INVALID_DAG,
    FailureKind.OOM,
    FailureKind.TIMEOUT,
    FailureKind.CRASH_BEFORE_LOAD,
    FailureKind.CORRUPT_OUTPUT,
    FailureKind.TRANSIENT_IO,
})

WARN_KINDS = frozenset({
    FailureKind.QUALITY,
    FailureKind.COHERENCE,
    FailureKind.NSFW,
    FailureKind.AV_SYNC,
})


def block_class_of(kind) -> "BlockClass":
    """Return the :class:`BlockClass` for ``kind``; fail-closed on an unknown
    kind (an unclassified failure is a malformed taxonomy, never swallowed)."""
    k = FailureKind(kind)
    if k in HARD_KINDS:
        return BlockClass.HARD
    if k in WARN_KINDS:
        return BlockClass.WARN
    raise ValueError("FailureKind %r is in neither HARD nor WARN" % (k,))


@dataclass(frozen=True)
class RetryDecision:
    """The single deterministic action keyed to a failure's block class.

    HARD decisions retry (per the kind) on the SAME engine, then the render
    driver raises LOUD (NO FALLBACKS, 2026-07-02 -- the escalate_to_fallback
    flag was deleted with the chain machinery); WARN decisions ``keep_output``
    and ``warn_only``. The three guard flags ``discards_output`` /
    ``touches_audio`` / ``aborts_episode`` are FALSE for every decision the
    taxonomy emits -- they exist so a test (and
    :func:`assert_decision_invariants`) can prove no class can ever drop a
    beat, mutate the frozen audio, or abort the episode.
    """

    kind: FailureKind
    block_class: BlockClass
    same_seed_retries: int = 0
    reseed_retries: int = 0
    keep_output: bool = False
    retime: bool = False
    warn_only: bool = False
    discards_output: bool = False
    touches_audio: bool = False
    aborts_episode: bool = False

    @property
    def is_hard(self) -> bool:
        return self.block_class is BlockClass.HARD

    @property
    def is_warn(self) -> bool:
        return self.block_class is BlockClass.WARN

    @property
    def max_attempts(self) -> int:
        """Total render attempts on the CURRENT engine before escalating: the
        first attempt plus any same-seed and reseed retries."""
        return 1 + int(self.same_seed_retries) + int(self.reseed_retries)


# The frozen per-kind policy table (the deterministic action map). HARD kinds
# retry per-kind on the SAME engine then fail LOUD (keep_output False -- there
# is no usable clip; NO FALLBACKS); WARN kinds keep the rendered clip and warn.
_HARD = BlockClass.HARD
_WARN = BlockClass.WARN

_POLICY: Dict[FailureKind, RetryDecision] = {
    FailureKind.DEPENDENCY_MISSING: RetryDecision(
        FailureKind.DEPENDENCY_MISSING, _HARD),
    FailureKind.ASSET_MISSING: RetryDecision(
        FailureKind.ASSET_MISSING, _HARD),
    FailureKind.LICENSE_BLOCKED: RetryDecision(
        FailureKind.LICENSE_BLOCKED, _HARD),
    FailureKind.INVALID_DAG: RetryDecision(
        FailureKind.INVALID_DAG, _HARD),
    FailureKind.OOM: RetryDecision(
        FailureKind.OOM, _HARD, same_seed_retries=0),
    FailureKind.TIMEOUT: RetryDecision(
        FailureKind.TIMEOUT, _HARD, same_seed_retries=0),
    FailureKind.CRASH_BEFORE_LOAD: RetryDecision(
        FailureKind.CRASH_BEFORE_LOAD, _HARD, same_seed_retries=1),
    FailureKind.CORRUPT_OUTPUT: RetryDecision(
        FailureKind.CORRUPT_OUTPUT, _HARD, same_seed_retries=1,
        reseed_retries=1),
    FailureKind.TRANSIENT_IO: RetryDecision(
        FailureKind.TRANSIENT_IO, _HARD,
        same_seed_retries=DEFAULT_TRANSIENT_IO_RETRIES),
    FailureKind.QUALITY: RetryDecision(
        FailureKind.QUALITY, _WARN, keep_output=True, warn_only=True),
    FailureKind.COHERENCE: RetryDecision(
        FailureKind.COHERENCE, _WARN, keep_output=True, warn_only=True),
    FailureKind.NSFW: RetryDecision(
        FailureKind.NSFW, _WARN, keep_output=True, warn_only=True),
    FailureKind.AV_SYNC: RetryDecision(
        FailureKind.AV_SYNC, _WARN, keep_output=True, retime=True),
}


def classify(kind) -> RetryDecision:
    """Return the deterministic :class:`RetryDecision` for ``kind``.

    Fail-closed: an unknown / unclassified kind raises ``ValueError`` rather
    than defaulting to a silent action. The returned decision is validated
    against the taxonomy invariants before it is handed back.
    """
    k = FailureKind(kind)
    decision = _POLICY.get(k)
    if decision is None:
        raise ValueError("no RetryDecision policy for FailureKind %r" % (k,))
    assert_decision_invariants(decision)
    return decision


def assert_decision_invariants(decision: RetryDecision) -> RetryDecision:
    """Prove a decision honors the A-S7 hard invariants; raise on a violation.

    For EVERY decision: it never discards rendered output, never touches the
    frozen audio, never aborts the episode. A WARN decision must additionally
    keep its output (a subjective gate may not drop a beat). A HARD decision is
    never warn-only. Used by :func:`classify` and the tests so a future policy
    edit that breaks an invariant fails loudly.
    """
    if decision.discards_output:
        raise ValueError(
            "RetryDecision must never discard rendered output: %r" % (decision,))
    if decision.touches_audio:
        raise ValueError(
            "RetryDecision must never touch frozen audio: %r" % (decision,))
    if decision.aborts_episode:
        raise ValueError(
            "RetryDecision must never abort the episode: %r" % (decision,))
    if decision.block_class is BlockClass.WARN and not decision.keep_output:
        raise ValueError(
            "a WARN decision must keep its rendered output: %r" % (decision,))
    if decision.block_class is BlockClass.HARD and decision.warn_only:
        raise ValueError("a HARD decision is not warn-only: %r" % (decision,))
    return decision


# NO FALLBACKS (Sprint A rip, 2026-07-02): the fallback-action API
# (build_fallback_decision / restamp_shot_row / append_runtime_fallback_decision
# / format_swap_log) was DELETED with the chain machinery. The ledger's
# runtime_fallback_decisions schema slot survives (stamped never, A5 -- no
# schema churn); this module keeps ONLY the failure-classification role.

__all__ = [
    "DEFAULT_TRANSIENT_IO_RETRIES",
    "BlockClass",
    "FailureKind",
    "HARD_KINDS",
    "WARN_KINDS",
    "block_class_of",
    "RetryDecision",
    "classify",
    "assert_decision_invariants",
]
