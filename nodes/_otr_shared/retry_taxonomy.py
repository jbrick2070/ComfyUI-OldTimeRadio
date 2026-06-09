"""Render-time retry taxonomy with an explicit block-class split (A-S7).

Every video render-time failure is classified into one of two BLOCK CLASSES,
and each class has a single deterministic action:

* ``BlockClass.HARD`` -- renderability / safety-integrity failures. The chosen
  engine cannot produce a usable clip for this shot, so the render node walks
  the declared ``fallback_engine`` chain (humo -> latentsync -> still_kenburns,
  resolved by :mod:`nodes._otr_shared.fallback`) down to the zero-VRAM radio
  floor, which always succeeds. The swap is LOUD: the node logs it and restamps
  the ledger -- never a silent substitution (operator directive 2026-06-07).
  ``dependency_missing`` / ``asset_missing`` / ``license_blocked`` /
  ``invalid_dag`` fail fast and walk the chain; ``oom`` / ``timeout`` tear down
  and re-resolve with zero retries; ``crash_before_load`` retries once;
  ``corrupt_output`` retries once at the same seed then reseeds then walks;
  ``transient_io`` does a small bounded retry. The episode never aborts (the
  floor terminates every chain) and a beat is never dropped.

* ``BlockClass.WARN`` -- subjective quality / coherence / NSFW gates (and the
  A/V-sync guard). These WARN only: the already-rendered clip is RETAINED, a
  warning is logged, and an optional in-render reseed / fallback may run, but a
  WARN gate NEVER discards rendered output, NEVER aborts an episode, and NEVER
  touches the frozen master audio. The offline NSFW frame-QC sampler is
  DEFAULT-OFF and warn-only because SFW is an invariant. The A/V-sync guard does
  a best-effort deterministic retime of the VIDEO frames (never the audio).

This mirrors the SHIPPED audio ``freeze_block_class`` structural/quality split
(``nodes/_otr_freeze_cascade.py``) on the render surface, but is a separate
module -- the frozen audio cascade is read-only and never imported here.

The module is PURE and dependency-free (stdlib only): it classifies a failure,
yields the deterministic :class:`RetryDecision`, and builds the durable ledger
restamp + the LOUD swap-log records. The GPU-side mechanics (taskkill the
sidecar tree, re-probe NVML, actually re-render on the fallback engine) live in
the render node and are exercised at the A-S7.5 soak; everything here is
testable on the CPU box. UTF-8, no BOM, ASCII-only source.
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

    HARD decisions retry (per the kind) then ``escalate_to_fallback`` to the
    next engine in the chain; WARN decisions ``keep_output`` and ``warn_only``.
    The three guard flags ``discards_output`` / ``touches_audio`` /
    ``aborts_episode`` are FALSE for every decision the taxonomy emits -- they
    exist so a test (and :func:`assert_decision_invariants`) can prove no class
    can ever drop a beat, mutate the frozen audio, or abort the episode.
    """

    kind: FailureKind
    block_class: BlockClass
    same_seed_retries: int = 0
    reseed_retries: int = 0
    escalate_to_fallback: bool = True
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
# escalate to the fallback chain (keep_output False -- there is no usable clip);
# WARN kinds keep the rendered clip and warn (escalate False by default; the
# abort-only improvement gates are off).
_HARD = BlockClass.HARD
_WARN = BlockClass.WARN

_POLICY: Dict[FailureKind, RetryDecision] = {
    FailureKind.DEPENDENCY_MISSING: RetryDecision(
        FailureKind.DEPENDENCY_MISSING, _HARD, escalate_to_fallback=True),
    FailureKind.ASSET_MISSING: RetryDecision(
        FailureKind.ASSET_MISSING, _HARD, escalate_to_fallback=True),
    FailureKind.LICENSE_BLOCKED: RetryDecision(
        FailureKind.LICENSE_BLOCKED, _HARD, escalate_to_fallback=True),
    FailureKind.INVALID_DAG: RetryDecision(
        FailureKind.INVALID_DAG, _HARD, escalate_to_fallback=True),
    FailureKind.OOM: RetryDecision(
        FailureKind.OOM, _HARD, same_seed_retries=0, escalate_to_fallback=True),
    FailureKind.TIMEOUT: RetryDecision(
        FailureKind.TIMEOUT, _HARD, same_seed_retries=0,
        escalate_to_fallback=True),
    FailureKind.CRASH_BEFORE_LOAD: RetryDecision(
        FailureKind.CRASH_BEFORE_LOAD, _HARD, same_seed_retries=1,
        escalate_to_fallback=True),
    FailureKind.CORRUPT_OUTPUT: RetryDecision(
        FailureKind.CORRUPT_OUTPUT, _HARD, same_seed_retries=1, reseed_retries=1,
        escalate_to_fallback=True),
    FailureKind.TRANSIENT_IO: RetryDecision(
        FailureKind.TRANSIENT_IO, _HARD,
        same_seed_retries=DEFAULT_TRANSIENT_IO_RETRIES,
        escalate_to_fallback=True),
    FailureKind.QUALITY: RetryDecision(
        FailureKind.QUALITY, _WARN, escalate_to_fallback=False,
        keep_output=True, warn_only=True),
    FailureKind.COHERENCE: RetryDecision(
        FailureKind.COHERENCE, _WARN, escalate_to_fallback=False,
        keep_output=True, warn_only=True),
    FailureKind.NSFW: RetryDecision(
        FailureKind.NSFW, _WARN, escalate_to_fallback=False,
        keep_output=True, warn_only=True),
    FailureKind.AV_SYNC: RetryDecision(
        FailureKind.AV_SYNC, _WARN, escalate_to_fallback=False,
        keep_output=True, retime=True),
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


# --------------------------------------------------------------------------- #
# Durable ledger restamp + the LOUD swap-log (operator directive 2026-06-07:
# every in-render fallback is LOUD -- log the swap + restamp the ledger, never
# silent). All pure dict transforms; the render node persists them.
# --------------------------------------------------------------------------- #
def build_fallback_decision(
    *,
    shot_id: str,
    beat_id: str,
    from_engine: str,
    to_engine: str,
    kind,
    video_revision: int,
    attempt: int = 1,
    detail: str = "",
) -> dict:
    """Build one ``runtime_fallback_decisions`` record (pure).

    The record pins the SAME ``video_revision`` (a fallback restamp never bumps
    the revision -- it is a within-revision degradation, not a re-lock) and
    classifies the trigger by block class so the audit trail is self-describing.
    """
    k = FailureKind(kind)
    return {
        "shot_id": shot_id,
        "beat_id": beat_id,
        "from_engine": from_engine,
        "to_engine": to_engine,
        "failure_kind": k.value,
        "block_class": block_class_of(k).value,
        "video_revision": int(video_revision),
        "attempt": int(attempt),
        "detail": detail,
    }


def restamp_shot_row(shot_row: dict, *, to_engine: str, to_family: str,
                     from_engine: str, kind) -> dict:
    """Return a NEW shot-row dict degraded onto ``to_engine`` (pure).

    Stamps ``engine_id`` / ``family`` to the fallback and appends a human-
    readable hop to ``degradation_trail`` so the trail records the full chain.
    Does not mutate the input and does not touch any revision counter.
    """
    k = FailureKind(kind)
    out = dict(shot_row)
    trail = list(out.get("degradation_trail") or [])
    trail.append("%s->%s (%s)" % (from_engine, to_engine, k.value))
    out["degradation_trail"] = trail
    out["engine_id"] = to_engine
    out["family"] = to_family
    return out


def append_runtime_fallback_decision(video_section: dict,
                                     decision_record: dict) -> dict:
    """Return a NEW ``ledger['video']`` section with the decision appended.

    Fail-closed: the record's ``video_revision`` must equal the section's (a
    fallback restamp stays at the SAME revision); a mismatch raises rather than
    silently re-numbering. ``video_revision`` is left UNCHANGED.
    """
    sec_rev = int(video_section.get("video_revision", 1))
    rec_rev = int(decision_record.get("video_revision", sec_rev))
    if rec_rev != sec_rev:
        raise ValueError(
            "runtime_fallback_decision video_revision %d != section "
            "video_revision %d (a restamp must stay at the same revision)"
            % (rec_rev, sec_rev))
    out = dict(video_section)
    decisions = list(out.get("runtime_fallback_decisions") or [])
    decisions.append(decision_record)
    out["runtime_fallback_decisions"] = decisions
    out["video_revision"] = sec_rev          # explicit: unchanged
    return out


def format_swap_log(decision_record: dict) -> str:
    """Render the LOUD one-line swap log the render node emits on a fallback.

    LOUD by contract (never a silent swap): names the shot/beat, the engine
    swap, the classified reason, the captured failure detail (exception type +
    first line, when present -- so a swallowed GraphExecutionError surfaces in
    the log instead of only the reason code), and reaffirms the frozen audio is
    untouched.
    """
    line = (
        "[OTR video] LOUD FALLBACK: shot=%s beat=%s engine '%s' -> '%s' "
        "reason=%s block_class=%s video_revision=%s (frozen audio untouched)"
        % (
            decision_record.get("shot_id"),
            decision_record.get("beat_id"),
            decision_record.get("from_engine"),
            decision_record.get("to_engine"),
            decision_record.get("failure_kind"),
            decision_record.get("block_class"),
            decision_record.get("video_revision"),
        )
    )
    detail = decision_record.get("detail")
    if detail:
        line += " detail=%r" % (str(detail),)
    return line


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
    "build_fallback_decision",
    "restamp_shot_row",
    "append_runtime_fallback_decision",
    "format_swap_log",
]
