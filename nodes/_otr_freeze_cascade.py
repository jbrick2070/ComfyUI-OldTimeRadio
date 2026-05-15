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

Verdict mapping (ADR section 9, S33 B2 trim 2026-05-15):

    Reviewer verdict          Freeze verdict
    ----------------          --------------
    clean_no_edits + clean    -> frozen_clean
    clean_no_edits + warns    -> frozen_with_warns
    improved                  -> frozen_with_doctor_edits
    too_many_edits            -> too_many_edits
    needs_full_rerun          -> needs_full_rerun
    Phase 10 raises           -> needs_full_rerun        (gap audit forced)

S33 B2 retired `cast_unrecoverable` (speaker_unknowns rollback gate)
and `post_audit_failed` (post_audit_pass rollback gate). Both were
pipeline cuts; the refined no-auditors rule forbids audit calls that
just gate / halt / rollback without feeding an editor.

Failure cascade (ADR section 8): the cascade is skip-and-continue
EXCEPT on Phase 10 critical gaps (hard-fail at the freeze gate).

Status: sprint commit 2 of 14 (2026-05-11).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from . import _otr_ledger_freeze as _LFC
from . import _otr_ledger_reviewer as _OTRLR

log = logging.getLogger("OTR.freeze_cascade")


__all__ = [
    "FreezeDisposition",
    "run_freeze_cascade",
    "REVIEWER_TO_FREEZE_VERDICT",
    "FREEZE_TERMINAL_FAILURE_VERDICTS",
    "all_phase_passes",
    "build_phase_telemetry",
]


# ---------------------------------------------------------------------------
# Verdict mapping
# ---------------------------------------------------------------------------


# Maps `_otr_ledger_reviewer.ReviewerVerdict` literals (Phases 1+2+9)
# to FreezeVerdict literals (Phase 10 / cascade exit). `clean_no_edits`
# is conditionally lifted to `frozen_clean` or `frozen_with_warns`
# depending on Phase 10 warning state -- the table here lists the
# interim mapping; the final stamp happens in run_freeze_cascade.
#
# S33 B2 (2026-05-15): `cast_unrecoverable` and `post_audit_failed`
# rows retired with the rollback gates that produced them.
REVIEWER_TO_FREEZE_VERDICT: dict[str, str] = {
    "clean_no_edits":     "frozen_clean",
    "improved":           "frozen_with_doctor_edits",
    "too_many_edits":     "too_many_edits",
    "needs_full_rerun":   "needs_full_rerun",
}


# Reviewer verdicts that terminate the cascade WITHOUT running Phase 10.
# review_ledger has already restored the ledger to its pre-review state
# for these; running Phase 10 on the restored ledger would either
# re-flag the same pre-existing gaps or stamp `frozen_clean` on an
# unaltered ledger, both of which are misleading.
#
# S33 B2 (2026-05-15): `cast_unrecoverable` and `post_audit_failed`
# removed -- their rollback gates were retired so neither verdict is
# reachable anymore.
FREEZE_TERMINAL_FAILURE_VERDICTS: frozenset[str] = frozenset({
    "too_many_edits",
    "needs_full_rerun",
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


# Bucket routing per B6 (clean-break 2026-05-12). "Cleanup" reads as
# polish / structural-edit scope; readiness checks are not cleanup.
# Phase 0 + Phase 10 are audit gates -- their own bucket so the
# semantic split is clean across the four record kinds:
#   audit_passes      Phase 0 (pre)  + Phase 10 (post + freeze)
#   cleanup_passes    Phase 1+2+9 (reviewer composite) + Phase 3
#                     (polish) + Phase 4 (scene coherence) + Phase
#                     4.5 (smart suggestion) + Phase 5 (voice drift)
#                     + Phase 6 (episode arc)
#   readiness_passes  Phase 7 (audio readiness) + Phase 8 (video
#                     readiness)
# S30 B4: phase_3 / 4 / 4.5 / 5 / 6 entries DELETED from this table.
# Those phases were removed from the cascade in the same commit.
_PHASE_BUCKETS: dict[str, str] = {
    "phase_0_gap_audit_pre":              "audit_passes",
    "phase_10_gap_audit_post_and_freeze": "audit_passes",
    "phase_1_2_9_reviewer_composite":     "cleanup_passes",
    "phase_7_audio_readiness":            "readiness_passes",
    "phase_8_video_readiness":            "readiness_passes",
}


def _bucket_for_phase(phase_name: str) -> str:
    """Map a phase name to its meta bucket. Unknown phases default
    to cleanup_passes (safest fall-through for soak telemetry; any
    new phase will surface as 'in cleanup_passes' until it's
    classified here).
    """
    return _PHASE_BUCKETS.get(phase_name, "cleanup_passes")


def build_phase_telemetry(meta: dict) -> list:
    """Build a compact per-phase telemetry payload for soak diagnostics.

    C3 of the clean-break go-forward plan (2026-05-12). The cascade's
    `freeze_verdict` STRING output is a single literal -- useful but
    coarse. This helper builds a structured per-phase summary that
    rides on `meta.freeze_phase_telemetry` (NOT the output STRING --
    that stays the literal verdict for graph-canvas readability).

    Each entry shape:
      {
        "phase":    "phase_1_2_9_reviewer_composite",  # phase_name
        "bucket":   "cleanup_passes",            # audit/cleanup/readiness
        "skipped":  bool,                        # did the phase run?
        "changed":  bool,                        # did line.text mutate?
        "warnings": int,                         # count of failure entries
        "edits_proposed": int,
        "edits_applied":  int,
      }

    `skipped` is True when the phase's failures list carries a
    "stub_bypassed" / "terminal_skipped" / "enable_false" reason.
    `changed` is True when text_hash_before != text_hash_after.
    """
    out: list = []
    for bucket_key in ("audit_passes", "cleanup_passes", "readiness_passes"):
        bucket = meta.get(bucket_key)
        if not isinstance(bucket, list):
            continue
        for rec in bucket:
            if not isinstance(rec, dict):
                continue
            failures = rec.get("failures") or []
            skip_reasons = (
                "stub_bypassed", "terminal_skipped", "enable_false",
            )
            skipped = any(
                isinstance(f, dict)
                and any(r in (f.get("reason") or "") for r in skip_reasons)
                for f in failures
            )
            hash_before = rec.get("text_hash_before")
            hash_after = rec.get("text_hash_after")
            changed = (
                hash_before is not None
                and hash_after is not None
                and hash_before != hash_after
            )
            real_failure_count = sum(
                1 for f in failures
                if isinstance(f, dict)
                and not any(
                    r in (f.get("reason") or "") for r in skip_reasons
                )
            )
            out.append({
                "phase": rec.get("phase_name", ""),
                "bucket": bucket_key,
                "skipped": skipped,
                "changed": changed,
                "warnings": real_failure_count,
                "edits_proposed": int(rec.get("edits_proposed") or 0),
                "edits_applied": int(rec.get("edits_applied") or 0),
                "started_at": rec.get("started_at", ""),
            })
    out.sort(key=lambda r: r.get("started_at", ""))
    # `started_at` is a forensic detail; drop from the public-shape
    # output so soak diagnostics see the compact form.
    for rec in out:
        rec.pop("started_at", None)
    return out


def all_phase_passes(meta: dict) -> list:
    """Return the chronological concatenation of phase records across
    all three buckets (audit_passes + cleanup_passes + readiness_passes).

    Soak diagnostics and tests that want "every cascade phase record
    in order" call this instead of indexing a single bucket. Records
    are sorted by `started_at` ISO timestamp so the merged list
    reflects actual run order regardless of bucket.

    Best-effort: a malformed bucket (not a list) is skipped silently
    -- the gap-audit invariant will have already flagged the type.
    """
    merged: list = []
    for bucket_key in ("audit_passes", "cleanup_passes", "readiness_passes"):
        bucket = meta.get(bucket_key)
        if isinstance(bucket, list):
            merged.extend(rec for rec in bucket if isinstance(rec, dict))
    merged.sort(key=lambda r: r.get("started_at", ""))
    return merged


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
    """Append a phase record to its meta bucket (B6 split).

    Wrapped in best-effort try/except: a stamping failure must never
    break cascade flow (per ADR section 8 skip-and-continue).
    """
    try:
        meta = ledger_data.setdefault("meta", {})
        bucket_key = _bucket_for_phase(phase_name)
        passes = meta.setdefault(bucket_key, [])
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
# Phase 7 / 8 readiness stubs (deterministic; Phase 3 / 4 / 4.5 / 5 / 6
# DELETED at S30 B4 -- the standalone OTR_LFCPhase4Scene / 5Voice /
# 6Arc node classes were orphaned from every shipped workflow JSON and
# all five backing functions defaulted OFF on every code path. The
# cascade's main path is now:
#   Phase 0 -> Phase 1/2/9 (reviewer composite) -> Phase 7 -> Phase 8
#   -> Phase 10
# Phase3PolishReport + the five `_phase_*` wrappers + their backing
# files (_otr_lfc_phase_4_scene_coherence, _otr_lfc_phase_5_voice_drift,
# _otr_lfc_phase_6_episode_arc, _otr_lfc_smart_suggestion,
# _otr_lfc_phase_verdicts, _otr_lfc_llm_helpers) all deleted in
# lockstep. B7 adds the symbol names as forbidden-pattern markers.
# ---------------------------------------------------------------------------


# S30 B4: the five phase function wrappers
# (_phase_3_per_line_polish, _phase_4_per_scene_coherence,
# _phase_4_5_smart_suggestion, _phase_5_voice_drift,
# _phase_6_episode_arc) and the Phase3PolishReport dataclass DELETED.
# Standalone OTR_LFCPhase4Scene / 5Voice / 6Arc node classes and their
# backing files (_otr_lfc_phase_4_scene_coherence,
# _otr_lfc_phase_5_voice_drift, _otr_lfc_phase_6_episode_arc,
# _otr_lfc_smart_suggestion, _otr_lfc_phase_verdicts,
# _otr_lfc_llm_helpers) deleted in the same commit. All five phases
# defaulted OFF on every code path; the cascade-side enable widgets
# were already removed in B3.


def _stamp_stub_or_skipped_phase(
    ledger_data: dict,
    *,
    phase_name: str,
    reason: str,
) -> None:
    """Append a zero-cost phase record for stub / terminal-skipped phases.

    B5 fix (commit 12.1): ensures meta.cleanup_passes is contiguous
    even when a phase didn't actually run. `reason` is one of:
      "stub_bypassed"     -- the phase implementation is a no-op stub
                            (Phase 4 / 5 / 6 today).
      "terminal_skipped"  -- the reviewer returned a terminal verdict
                            and the cascade short-circuited (B7 fix).
      "enable_false"      -- phase disabled via the widget toggle.
    """
    hash_now = _hash_lines_text(ledger_data)
    ts = _isoformat_utc_now()
    _stamp_phase_record(
        ledger_data,
        phase_name=phase_name,
        text_hash_before=hash_now,
        text_hash_after=hash_now,
        started_at=ts,
        finished_at=ts,
        failures=[{"line_id": f"__{phase_name}__", "reason": reason}],
    )


# LFC commit 5: Phase 7 + Phase 8 wired through `_otr_readiness`. Both
# phases default ON via cascade kwargs because both are deterministic
# and cheap.
def _phase_7_audio_readiness(led, *, enable: bool = True):
    if not enable:
        log.debug("[LFC:phase_7] disabled (enable=False)")
        led.data.setdefault("meta", {})["audio_readiness"] = {
            "skipped": True, "skipped_reason": "enable=False",
        }
        return None
    from . import _otr_readiness as _LFC_RDY  # type: ignore
    return _LFC_RDY.phase_7_audio_readiness(led)


def _phase_8_video_readiness(led, *, enable: bool = True):
    if not enable:
        log.debug("[LFC:phase_8] disabled (enable=False)")
        led.data.setdefault("meta", {})["video_readiness"] = {
            "skipped": True, "skipped_reason": "enable=False",
        }
        return None
    from . import _otr_readiness as _LFC_RDY  # type: ignore
    return _LFC_RDY.phase_8_video_readiness(led)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_freeze_cascade(
    generate_fn,
    led,
    *,
    polish_generate_fn=None,
    enable_phase_7_audio_readiness: bool = True,
    enable_phase_8_video_readiness: bool = True,
    vram_ceiling_gb: float = 14.0,
) -> FreezeDisposition:
    """Orchestrate Phase 0 -> reviewer (Phase 1+2+9) -> Phase 10.

    Behaviour:
      * Phase 0 always runs first; warn-only. Records to
        meta.cleanup_passes.
      * Existing review_ledger runs next (Phases 1, 2, 9 are still
        bundled into this one function; commits 3, 4, 8, 9, 10 split
        them out).
      * If the reviewer verdict is a terminal failure
        (too_many_edits / needs_full_rerun), the cascade stops there:
        the ledger has already been restored to its pre-review state,
        and Phase 10 would either re-flag the same pre-existing gap
        or stamp frozen_clean on an unaltered ledger -- both
        misleading. S33 B2 (2026-05-15) retired the two rollback-gate
        verdicts (cast_unrecoverable, post_audit_failed) per the
        refined no-auditors rule.
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

    # ---- B2 fix (commit 12.1): VRAM watchdog at cascade entry ----
    # Alarm plumbing only -- single measurement, warn on over-ceiling,
    # continue regardless. Per-phase gating is follow-up wiring once
    # soak data shows where the actual ceiling hits are.
    meta = ledger_data.setdefault("meta", {})
    meta["lfc_vram_ceiling_gb"] = float(vram_ceiling_gb)
    try:
        from . import _otr_lfc_watchdog as _LFC_WD  # type: ignore
        over_ceiling, current_gb = _LFC_WD.vram_over_ceiling(
            ceiling_gb=float(vram_ceiling_gb),
        )
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "[LFC] VRAM watchdog read failed at cascade entry: %s; "
            "stamping 0.0 GB and proceeding", exc,
        )
        over_ceiling, current_gb = False, 0.0
    meta["vram_at_cascade_entry_gb"] = float(current_gb)
    if over_ceiling:
        log.warning(
            "[LFC] WARN VRAM at %.2f GB over %.2f ceiling; cascade "
            "will proceed", current_gb, float(vram_ceiling_gb),
        )

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

    # ---- Phase 3 DELETED (S30 B4) --------------------------------
    # The per-line polish phase ran with `enable=False` on every code
    # path (the cascade widget defaulted OFF; B3 deleted the widget
    # entirely). Composer-inline polish via the writer's
    # `enable_polish_pass` widget covers the same surface; the
    # cascade-side second polish opportunity was never exercised.

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

    # ---- B7 fix (commit 12.1): terminal-verdict check moved here ----
    # Pre-12.1 the terminal check fired AFTER Phase 4.5 / 7 / 8 had
    # already mutated the restored ledger, breaking the "ledger
    # byte-identical to input" guarantee that justifies skipping
    # Phase 10. New order: reviewer composite -> terminal check ->
    # (terminal: stamp remaining phases as terminal_skipped + return)
    # OR (non-terminal: continue through Phase 4..8 + Phase 10).
    interim_verdict = REVIEWER_TO_FREEZE_VERDICT.get(
        reviewer_disp.verdict, "needs_full_rerun",
    )
    if interim_verdict in FREEZE_TERMINAL_FAILURE_VERDICTS:
        meta = ledger_data.setdefault("meta", {})
        meta["freeze_verdict"] = interim_verdict
        # B5: stamp the remaining phases as terminal_skipped so
        # meta.cleanup_passes stays contiguous. S30 B4: phase 4/4.5/5/6
        # names removed -- those phases no longer exist.
        for skipped_name in (
            "phase_7_audio_readiness",
            "phase_8_video_readiness",
            "phase_10_gap_audit_post_and_freeze",
        ):
            _stamp_stub_or_skipped_phase(
                ledger_data,
                phase_name=skipped_name,
                reason="terminal_skipped",
            )
        # Stamp the skipped-phase status on their respective meta keys
        # so downstream readers see a consistent shape.
        meta.setdefault("audio_readiness", {
            "skipped": True, "skipped_reason": "terminal_skipped",
        })
        meta.setdefault("video_readiness", {
            "skipped": True, "skipped_reason": "terminal_skipped",
        })
        disp = FreezeDisposition(
            verdict=interim_verdict,
            reviewer_disposition=reviewer_disp,
            gap_audit_pre=pre_report,
            gap_audit_post=None,
        )
        meta["freeze_disposition"] = disp.to_dict()
        # C3 (clean-break 2026-05-12): compact per-phase telemetry
        # for soak diagnostics. Lives on meta only -- the
        # freeze_verdict output STRING stays the verdict literal
        # so graph-canvas previews are readable.
        meta["freeze_phase_telemetry"] = build_phase_telemetry(meta)
        log.info(
            "[LFC] terminal reviewer verdict %r -- skipping Phase 4..10 "
            "(B7 fix: short-circuit moved before mutation phases)",
            interim_verdict,
        )
        return disp

    # ---- Non-terminal path: Phase 7 / 8 / 10 ---------------------
    # S30 B4: Phase 4 / 4.5 / 5 / 6 DELETED. The standalone
    # OTR_LFCPhase4Scene / 5Voice / 6Arc node classes were orphaned
    # from every shipped workflow JSON and all five backing functions
    # defaulted OFF on every code path. Cascade flow is now:
    #   Phase 0 -> Phase 1/2/9 reviewer -> Phase 7 -> Phase 8 -> Phase 10.

    # Phase 7 / 8 -- deterministic readiness checks (LFC commit 5).
    started_7 = _isoformat_utc_now()
    hash_before_7 = _hash_lines_text(ledger_data)
    p7_report = _phase_7_audio_readiness(
        led, enable=enable_phase_7_audio_readiness,
    )
    hash_after_7 = _hash_lines_text(ledger_data)
    _stamp_phase_record(
        ledger_data,
        phase_name="phase_7_audio_readiness",
        text_hash_before=hash_before_7,
        text_hash_after=hash_after_7,
        started_at=started_7,
        finished_at=_isoformat_utc_now(),
        edits_applied=(
            p7_report.lines_normalized if p7_report is not None else 0
        ),
    )
    started_8 = _isoformat_utc_now()
    hash_before_8 = _hash_lines_text(ledger_data)
    _phase_8_video_readiness(
        led, enable=enable_phase_8_video_readiness,
    )
    hash_after_8 = _hash_lines_text(ledger_data)
    _stamp_phase_record(
        ledger_data,
        phase_name="phase_8_video_readiness",
        text_hash_before=hash_before_8,
        text_hash_after=hash_after_8,
        started_at=started_8,
        finished_at=_isoformat_utc_now(),
    )

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
        # C3 (clean-break 2026-05-12): compact per-phase telemetry
        # for soak diagnostics. Lives on meta only -- the
        # freeze_verdict output STRING stays the verdict literal
        # so graph-canvas previews are readable.
        meta["freeze_phase_telemetry"] = build_phase_telemetry(meta)
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
    # C3 (clean-break 2026-05-12): compact per-phase telemetry
    # for soak diagnostics. Lives on meta only -- the
    # freeze_verdict output STRING stays the verdict literal.
    meta["freeze_phase_telemetry"] = build_phase_telemetry(meta)
    log.info(
        "[LFC] freeze landed: verdict=%s reviewer=%s pre_warns=%d "
        "post_warns=%d",
        final_verdict, reviewer_disp.verdict,
        len(pre_report.warnings), len(post_report.warnings),
    )
    return disp
