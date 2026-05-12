"""nodes/_otr_lfc_phase_verdicts.py — shared verdict literals.

Single source of truth for the verdict strings returned by the
three standalone LFC phase nodes (Phase 4 / 5 / 6) and any future
phase-level consumer. Pre-fix each standalone node had its own
ad-hoc literal set with overlapping-but-not-identical strings;
downstream branching on the verdict would have needed to know
each node's specific dialect.

G3 fix (LFC commit 12.13, 2026-05-12). Per reviewer go-forward
plan §G3 in docs/2026-05-12-lfc-cleanbreak-goforward.md.

Verdict semantics
  Universal (every phase node can return any of these):
    skipped                  enable=False on the widget
    no_ledger                peek_ledger() returned None
    completed_no_edits       phase ran, applied zero edits
    completed_with_edits     phase ran, at least one edit applied
    failed                   the underlying phase function reported
                             a real failure (LLM crash, parse
                             exhaustion, etc.)

  Interlock (added by commit 12.14, G1 + G2):
    already_ran_in_cascade   the same phase ALREADY ran via the
                             main cascade in this episode (the
                             standalone node refuses to re-edit
                             without an explicit `force=True`).
    missing_prerequisite     a required upstream phase did not
                             run (e.g. standalone Phase 5 without
                             a polish pass on the ledger).

  Phase-specific (preserved because they carry useful nuance):
    completed_with_failures        Phase 4 only -- one or more
                                   scenes' two_step_generate
                                   exhausted parse retries but the
                                   phase as a whole advanced.
    completed_no_edits_after_flag  Phase 5 only -- stats flagged
                                   lines, but the editor chose
                                   LEAVE on all of them.
    completed_no_flagged_lines     Phase 5 only -- stats found no
                                   drift; no LLM call fired.
"""
from __future__ import annotations

from typing import Literal


__all__ = [
    "PhaseVerdict",
    "ALL_PHASE_VERDICTS",
    "UNIVERSAL_VERDICTS",
    "INTERLOCK_VERDICTS",
    "PHASE_SPECIFIC_VERDICTS",
    "phase_already_ran_in_cascade",
    "prerequisite_phase_ran",
]


# ---------------------------------------------------------------------------
# Universal verdicts -- every phase node may return any of these
# ---------------------------------------------------------------------------

UNIVERSAL_VERDICTS = frozenset({
    "skipped",
    "no_ledger",
    "completed_no_edits",
    "completed_with_edits",
    "failed",
})


# ---------------------------------------------------------------------------
# Interlock verdicts (commit 12.14 G1 + G2 -- not yet emitted)
# ---------------------------------------------------------------------------

INTERLOCK_VERDICTS = frozenset({
    "already_ran_in_cascade",
    "missing_prerequisite",
})


# ---------------------------------------------------------------------------
# Phase-specific verdicts (preserve nuance the universal set drops)
# ---------------------------------------------------------------------------

PHASE_SPECIFIC_VERDICTS = frozenset({
    "completed_with_failures",        # Phase 4
    "completed_no_edits_after_flag",  # Phase 5
    "completed_no_flagged_lines",     # Phase 5
})


# ---------------------------------------------------------------------------
# PhaseVerdict Literal -- the union across all three sets
# ---------------------------------------------------------------------------

PhaseVerdict = Literal[
    # universal
    "skipped",
    "no_ledger",
    "completed_no_edits",
    "completed_with_edits",
    "failed",
    # interlock
    "already_ran_in_cascade",
    "missing_prerequisite",
    # phase-specific
    "completed_with_failures",
    "completed_no_edits_after_flag",
    "completed_no_flagged_lines",
]


ALL_PHASE_VERDICTS: frozenset[str] = (
    UNIVERSAL_VERDICTS | INTERLOCK_VERDICTS | PHASE_SPECIFIC_VERDICTS
)


# ---------------------------------------------------------------------------
# Interlock helpers (G1 + G2 -- commit 12.14)
# ---------------------------------------------------------------------------


def phase_already_ran_in_cascade(meta: dict, phase_name: str) -> bool:
    """True if `phase_name` already has a real (non-skipped) record
    in the cascade's bucket lists.

    A phase-record is "real" when it was NOT stamped via
    `_stamp_stub_or_skipped_phase` with a `stub_bypassed` /
    `terminal_skipped` / `enable_false` reason. The G1 + G2
    interlock should NOT fire on a cascade run that left the
    corresponding widget OFF -- the standalone node's whole
    purpose is to fill that gap.
    """
    if not isinstance(meta, dict):
        return False
    skip_reasons = (
        "stub_bypassed", "terminal_skipped", "enable_false",
    )
    for bucket_key in ("audit_passes", "cleanup_passes", "readiness_passes"):
        bucket = meta.get(bucket_key)
        if not isinstance(bucket, list):
            continue
        for rec in bucket:
            if not isinstance(rec, dict):
                continue
            if rec.get("phase_name") != phase_name:
                continue
            failures = rec.get("failures") or []
            was_skipped = any(
                isinstance(f, dict)
                and any(r in (f.get("reason") or "") for r in skip_reasons)
                for f in failures
            )
            if not was_skipped:
                return True
    return False


def prerequisite_phase_ran(meta: dict, prereq_phase_name: str) -> bool:
    """True if the named prerequisite phase ran cleanly (non-skipped)
    in the cascade. Used by Phase 5 / Phase 6 standalone nodes to
    refuse running on a raw writer ledger.

    Same skip-detection rules as `phase_already_ran_in_cascade` --
    the prerequisite must have been a REAL run, not a stub /
    terminal-skip / enable=False record.
    """
    return phase_already_ran_in_cascade(meta, prereq_phase_name)
