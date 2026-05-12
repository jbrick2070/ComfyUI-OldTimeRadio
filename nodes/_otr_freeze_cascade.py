"""nodes/_otr_freeze_cascade.py — Ledger Freeze Cascade orchestrator.

Wraps the existing `_otr_ledger_reviewer.review_ledger` 3-pass with
Phase 0 (gap_audit_pre) at entry and Phase 10 (gap_audit_post +
freeze) at exit. Subsequent sprint commits decompose the reviewer's
internal passes into discrete cascade phases (Phase 1 / Phase 2 /
Phase 9) AND insert the new LLM phases (Phase 3 polish, Phase 4
scene coherence, Phase 4.5 smart suggestion, Phase 5 voice drift,
Phase 6 episode arc, Phase 7 audio readiness, Phase 8 video
readiness). This commit (commit 2 of 14) ships the orchestrator
skeleton only -- Phase 3/4/4.5/5/6/7/8 are no-op holes for now.

ADR mapping (`docs/2026-05-11-multi-turn-polish-adr.md`):

    [Writer ledger]
        |
        v
    Phase 0    gap_audit_pre              (commit 1, deterministic)
    Phase 1    cast audit + repairs       (existing reviewer Pass 1)  <- review_ledger
    Phase 2    script doctor              (existing reviewer Pass 2)  <- review_ledger
    Phase 3    per-line polish            (commit 4, no-op here)
    Phase 4    per-scene coherence        (commit 8, no-op here)
    Phase 4.5  smart suggestion           (commit 11, no-op here)
    Phase 5    per-speaker voice drift    (commit 9, no-op here)
    Phase 6    episode arc                (commit 10, no-op here)
    Phase 7    audio readiness            (commit 5, no-op here)
    Phase 8    video readiness            (commit 5, no-op here)
    Phase 9    cast audit final           (existing reviewer Pass 3)  <- review_ledger
    Phase 10   gap_audit_post + freeze    (commit 1, deterministic)
        |
        v
    [Frozen ledger; meta.cleanup_locked == True]

Verdict mapping (ADR section 9 -- preserves the existing 5-slot output
contract, just renames `reviewer_verdict` -> `freeze_verdict`):

    Reviewer verdict          Freeze verdict
    ----------------          --------------
    clean_no_edits + clean    -> frozen_clean
    clean_no_edits + warns    -> frozen_with_warns
    improved                  -> frozen_with_doctor_edits
    cast_unrecoverable        -> cast_unrecoverable      (preserved)
    too_many_edits            -> too_many_edits          (preserved)
    needs_full_rerun          -> needs_full_rerun        (preserved)
    post_audit_failed         -> post_audit_failed       (preserved)
    Phase 10 raises           -> needs_full_rerun        (gap audit forced)

Failure cascade (ADR section 8): the cascade is skip-and-continue
EXCEPT on Phase 10 critical gaps (hard-fail at the freeze gate).

Status: sprint commit 2 of 14 (2026-05-11).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from . import _otr_ledger_freeze as _LFC
from . import _otr_ledger_reviewer as _OTRLR

log = logging.getLogger("OTR.freeze_cascade")


__all__ = [
    "FreezeDisposition",
    "run_freeze_cascade",
    "REVIEWER_TO_FREEZE_VERDICT",
    "FREEZE_TERMINAL_FAILURE_VERDICTS",
]


# ---------------------------------------------------------------------------
# Verdict mapping
# ---------------------------------------------------------------------------


# Maps `_otr_ledger_reviewer.ReviewerVerdict` literals (Phases 1+2+9)
# to FreezeVerdict literals (Phase 10 / cascade exit). `clean_no_edits`
# is conditionally lifted to `frozen_clean` or `frozen_with_warns`
# depending on Phase 10 warning state -- the table here lists the
# interim mapping; the final stamp happens in run_freeze_cascade.
REVIEWER_TO_FREEZE_VERDICT: dict[str, str] = {
    "clean_no_edits":     "frozen_clean",
    "improved":           "frozen_with_doctor_edits",
    "cast_unrecoverable": "cast_unrecoverable",
    "too_many_edits":     "too_many_edits",
    "needs_full_rerun":   "needs_full_rerun",
    "post_audit_failed":  "post_audit_failed",
}


# Reviewer verdicts that terminate the cascade WITHOUT running Phase 10.
# review_ledger has already restored the ledger to its pre-review state
# for these; running Phase 10 on the restored ledger would either
# re-flag the same pre-existing gaps or stamp `frozen_clean` on an
# unaltered ledger, both of which are misleading.
FREEZE_TERMINAL_FAILURE_VERDICTS: frozenset[str] = frozenset({
    "cast_unrecoverable",
    "too_many_edits",
    "needs_full_rerun",
    "post_audit_failed",
})


# ---------------------------------------------------------------------------
# Disposition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FreezeDisposition:
    """End-of-cascade summary stamped to `meta.freeze_disposition`.

    `reviewer_disposition` is the original 3-pass return -- preserved
    intact for forensic continuity with the pre-cascade
    `meta.reviewer_disposition` shape. `gap_audit_pre` and
    `gap_audit_post` carry the deterministic audit reports from
    Phase 0 / Phase 10. `verdict` is the final FreezeVerdict literal.
    """

    verdict: str  # FreezeVerdict
    reviewer_disposition: Optional[_OTRLR.ReviewerDisposition]
    gap_audit_pre: _LFC.GapAuditReport
    gap_audit_post: Optional[_LFC.GapAuditReport]

    def to_dict(self) -> dict:
        """JSON-friendly view for stamping on meta.freeze_disposition."""
        rev = self.reviewer_disposition
        pre = self.gap_audit_pre
        post = self.gap_audit_post
        return {
            "verdict": self.verdict,
            "reviewer_disposition": (
                rev.__dict__ if rev is not None else None
            ),
            "gap_audit_pre": {
                "errors": list(pre.errors),
                "warnings": list(pre.warnings),
                "info": dict(pre.info),
            },
            "gap_audit_post": (
                {
                    "errors": list(post.errors),
                    "warnings": list(post.warnings),
                    "info": dict(post.info),
                }
                if post is not None
                else None
            ),
        }


# ---------------------------------------------------------------------------
# Phase records (ADR section 6.7 cleanup_passes scaffolding)
# ---------------------------------------------------------------------------


def _hash_lines_text(ledger_data: dict) -> int:
    """Cheap fingerprint of line.text values for idempotency checks.

    The full §6.7 idempotency feature lands in commit 12; here we just
    stamp the entry/exit hash on each phase record so the soak diag
    has the data for free. `hash()` salted process-locally is fine for
    in-process forensic logs.
    """
    lines = ledger_data.get("lines") or []
    return hash(tuple((ln or {}).get("text", "") for ln in lines))


def _stamp_phase_record(
    ledger_data: dict,
    *,
    phase_name: str,
    text_hash_before: int,
    text_hash_after: int,
    started_at: str,
    finished_at: str,
    edits_proposed: int = 0,
    edits_applied: int = 0,
    failures: Optional[list] = None,
) -> None:
    """Append a phase record to `meta.cleanup_passes`.

    Wrapped in best-effort try/except: a stamping failure must never
    break cascade flow (per ADR section 8 skip-and-continue).
    """
    try:
        meta = ledger_data.setdefault("meta", {})
        passes = meta.setdefault("cleanup_passes", [])
        if not isinstance(passes, list):
            # Hard-failed by Phase 10 anyway; don't try to recover.
            return
        passes.append({
            "phase_name": phase_name,
            "started_at": started_at,
            "finished_at": finished_at,
            "text_hash_before": text_hash_before,
            "text_hash_after": text_hash_after,
            "edits_proposed": int(edits_proposed),
            "edits_applied": int(edits_applied),
            "failures": list(failures or []),
        })
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "[LFC] phase-record stamp failed for %s: %s",
            phase_name, exc,
        )


def _isoformat_utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Phase-3/4/4.5/5/6/7/8 stubs (no-op holes filled in later commits)
# ---------------------------------------------------------------------------


def _phase_3_per_line_polish_stub(generate_fn, led) -> None:
    """Phase 3 no-op stub. Commit 4 wires in the per-line polish refactor."""
    log.debug("[LFC:phase_3] stub -- per-line polish not yet wired")


def _phase_4_per_scene_coherence_stub(generate_fn, led) -> None:
    """Phase 4 no-op stub. Commit 8 wires in scene coherence."""
    log.debug("[LFC:phase_4] stub -- per-scene coherence not yet wired")


def _phase_4_5_smart_suggestion_stub(led) -> None:
    """Phase 4.5 no-op stub. Commit 11 wires in deterministic SFX synth."""
    log.debug("[LFC:phase_4_5] stub -- smart suggestion not yet wired")


def _phase_5_voice_drift_stub(generate_fn, led) -> None:
    """Phase 5 no-op stub. Commit 9 wires in voice drift detection."""
    log.debug("[LFC:phase_5] stub -- voice drift not yet wired")


def _phase_6_episode_arc_stub(generate_fn, led) -> None:
    """Phase 6 no-op stub. Commit 10 wires in episode arc audit."""
    log.debug("[LFC:phase_6] stub -- episode arc not yet wired")


def _phase_7_audio_readiness_stub(led) -> None:
    """Phase 7 no-op stub. Commit 5 wires in CMU dict normalization."""
    log.debug("[LFC:phase_7] stub -- audio readiness not yet wired")


def _phase_8_video_readiness_stub(led) -> None:
    """Phase 8 no-op stub. Commit 5 wires in portrait + duration checks."""
    log.debug("[LFC:phase_8] stub -- video readiness not yet wired")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_freeze_cascade(generate_fn, led) -> FreezeDisposition:
    """Orchestrate Phase 0 -> reviewer (Phase 1+2+9) -> Phase 10.

    Behaviour:
      * Phase 0 always runs first; warn-only. Records to
        meta.cleanup_passes.
      * Existing review_ledger runs next (Phases 1, 2, 9 are still
        bundled into this one function; commits 3, 4, 8, 9, 10 split
        them out).
      * If the reviewer verdict is a terminal failure
        (cast_unrecoverable / too_many_edits / needs_full_rerun /
        post_audit_failed), the cascade stops there: the ledger has
        already been restored to its pre-review state, and Phase 10
        would either re-flag the same pre-existing gap or stamp
        frozen_clean on an unaltered ledger -- both misleading.
      * On reviewer success (clean_no_edits / improved), Phase 10
        runs and either freezes the ledger or raises
        FreezeAssertionError. The exception is caught and translated
        to a `needs_full_rerun` verdict; the freeze stamp from
        phase_10 lives on meta.

    Stamps:
      meta.cleanup_passes      list of phase records
      meta.freeze_verdict      final cascade verdict
      meta.freeze_disposition  full FreezeDisposition dict
      meta.gap_audit_pre       Phase 0 report
      meta.gap_audit_post      Phase 10 report (only when Phase 10 ran)
      meta.cleanup_locked      True only on successful freeze

    `led` is a `production_ledger.Ledger`-like object exposing `.data`.

    Honors meta.skip_reviewer (test bypass) -- when set, Phases 1-9
    are skipped via the existing reviewer's own bypass, and the
    cascade still runs Phase 0 / Phase 10 around it. This keeps the
    freeze contract (ledger-health audit) independent of LLM cleanup.
    """
    ledger_data = led.data

    # ---- Phase 0: deterministic warn-mode audit ----------
    started = _isoformat_utc_now()
    hash_before = _hash_lines_text(ledger_data)
    pre_report = _LFC.phase_0_gap_audit_pre(led)
    hash_after = _hash_lines_text(ledger_data)
    _stamp_phase_record(
        ledger_data,
        phase_name="phase_0_gap_audit_pre",
        text_hash_before=hash_before,
        text_hash_after=hash_after,
        started_at=started,
        finished_at=_isoformat_utc_now(),
        failures=[{"line_id": "__phase_0__", "reason": e}
                  for e in pre_report.errors],
    )

    # ---- Phases 3..8: no-op stubs (filled in later commits) ----
    # Threading order matches the ADR phase chain so the no-op
    # contract is locked in commit 2; later commits replace each
    # stub call with the real implementation in place.
    _phase_3_per_line_polish_stub(generate_fn, led)

    # ---- Phase 1 + 2 + 9: existing 3-pass reviewer -------
    started = _isoformat_utc_now()
    hash_before = _hash_lines_text(ledger_data)
    reviewer_disp = _OTRLR.review_ledger(generate_fn, led)
    hash_after = _hash_lines_text(ledger_data)
    _stamp_phase_record(
        ledger_data,
        phase_name="phase_1_2_9_reviewer_composite",
        text_hash_before=hash_before,
        text_hash_after=hash_after,
        started_at=started,
        finished_at=_isoformat_utc_now(),
        edits_proposed=reviewer_disp.doctor_edits_proposed,
        edits_applied=reviewer_disp.doctor_edits_applied,
    )

    # Phase 4, 4.5, 5, 6 -- stubs run AFTER the reviewer because once
    # commit 4/8/9/10/11 land, they operate on post-doctor text. The
    # call order here is the contract; the stubs are no-ops today.
    _phase_4_per_scene_coherence_stub(generate_fn, led)
    _phase_4_5_smart_suggestion_stub(led)
    _phase_5_voice_drift_stub(generate_fn, led)
    _phase_6_episode_arc_stub(generate_fn, led)

    # Phase 7 / 8 deterministic readiness checks -- ALWAYS run
    # (deterministic & cheap), even on reviewer failure paths in
    # later commits. Today they are no-op stubs.
    _phase_7_audio_readiness_stub(led)
    _phase_8_video_readiness_stub(led)

    # ---- Translate reviewer verdict to interim freeze verdict ----
    interim_verdict = REVIEWER_TO_FREEZE_VERDICT.get(
        reviewer_disp.verdict, "needs_full_rerun",
    )

    # ---- Terminal failure -> skip Phase 10 ----------------
    if interim_verdict in FREEZE_TERMINAL_FAILURE_VERDICTS:
        meta = ledger_data.setdefault("meta", {})
        meta["freeze_verdict"] = interim_verdict
        # cleanup_locked stays False -- the reviewer's restore preserves
        # the pre-cascade state and the cascade is not advancing.
        disp = FreezeDisposition(
            verdict=interim_verdict,
            reviewer_disposition=reviewer_disp,
            gap_audit_pre=pre_report,
            gap_audit_post=None,
        )
        meta["freeze_disposition"] = disp.to_dict()
        log.info(
            "[LFC] terminal reviewer verdict %r -- skipping Phase 10",
            interim_verdict,
        )
        return disp

    # ---- Phase 10: hard gate ------------------------------
    started = _isoformat_utc_now()
    hash_before = _hash_lines_text(ledger_data)
    try:
        post_report = _LFC.phase_10_gap_audit_post_and_freeze(led)
    except _LFC.FreezeAssertionError as exc:
        hash_after = _hash_lines_text(ledger_data)
        _stamp_phase_record(
            ledger_data,
            phase_name="phase_10_gap_audit_post_and_freeze",
            text_hash_before=hash_before,
            text_hash_after=hash_after,
            started_at=started,
            finished_at=_isoformat_utc_now(),
            failures=[{"line_id": "__phase_10__", "reason": e}
                      for e in exc.errors],
        )
        meta = ledger_data.setdefault("meta", {})
        # phase_10 already stamped meta.freeze_verdict = needs_full_rerun
        # when it raised (when meta was stampable).
        if not meta.get("freeze_verdict"):
            meta["freeze_verdict"] = "needs_full_rerun"
        disp = FreezeDisposition(
            verdict="needs_full_rerun",
            reviewer_disposition=reviewer_disp,
            gap_audit_pre=pre_report,
            gap_audit_post=exc.report,
        )
        meta["freeze_disposition"] = disp.to_dict()
        log.warning(
            "[LFC] Phase 10 rejected freeze (%d critical gap(s))",
            len(exc.errors),
        )
        return disp

    hash_after = _hash_lines_text(ledger_data)
    _stamp_phase_record(
        ledger_data,
        phase_name="phase_10_gap_audit_post_and_freeze",
        text_hash_before=hash_before,
        text_hash_after=hash_after,
        started_at=started,
        finished_at=_isoformat_utc_now(),
    )

    # ---- Successful freeze -> finalize verdict ------------
    # phase_10 stamped meta.freeze_verdict = frozen_clean OR
    # frozen_with_warns. Lift `improved` reviewer paths to
    # frozen_with_doctor_edits regardless (the doctor-edits result is
    # more informative than the gap-audit warn-state alone).
    meta = ledger_data["meta"]
    if reviewer_disp.verdict == "improved":
        meta["freeze_verdict"] = "frozen_with_doctor_edits"
        final_verdict = "frozen_with_doctor_edits"
    else:
        final_verdict = meta.get("freeze_verdict", "frozen_clean")
    disp = FreezeDisposition(
        verdict=final_verdict,
        reviewer_disposition=reviewer_disp,
        gap_audit_pre=pre_report,
        gap_audit_post=post_report,
    )
    meta["freeze_disposition"] = disp.to_dict()
    log.info(
        "[LFC] freeze landed: verdict=%s reviewer=%s pre_warns=%d "
        "post_warns=%d",
        final_verdict, reviewer_disp.verdict,
        len(pre_report.warnings), len(post_report.warnings),
    )
    return disp
