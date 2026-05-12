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


@dataclass
class Phase3PolishReport:
    """Per-line polish summary for Phase 3.

    Stamped on `meta.phase_3_polish_record` so soak diagnostics can
    track how many leaky lines fired, how many were repaired, how
    many were rejected (still-leaky / refusal / word-cap miss), and
    which lines were skipped (announcer beats when announcer-polish
    is disabled).
    """

    lines_scanned: int = 0
    lines_examined: int = 0       # tripped needs_polish (would-polish)
    edits_applied: int = 0
    edits_rejected_still_leaky: int = 0
    edits_rejected_refusal: int = 0
    failures: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "lines_scanned": int(self.lines_scanned),
            "lines_examined": int(self.lines_examined),
            "edits_applied": int(self.edits_applied),
            "edits_rejected_still_leaky": int(
                self.edits_rejected_still_leaky
            ),
            "edits_rejected_refusal": int(self.edits_rejected_refusal),
            "failures": list(self.failures),
        }


def _phase_3_per_line_polish(
    generate_fn,
    led,
    *,
    polish_generate_fn=None,
    enable: bool = False,
    polish_announcer_beats: bool = False,
) -> Phase3PolishReport:
    """LFC Phase 3 -- per-line polish wired into the cascade.

    Pre-LFC the polish ran inline inside `compose_line` gated by the
    writer's `enable_polish_pass` widget. The cascade now offers a
    SECOND polish opportunity AFTER the reviewer's 3-pass: any line
    that still trips `needs_polish` gets one polish call with the
    new ADR-section-6 context (beat intent + last 2 lines + announcer
    guard + refusal detector + dedicated polish generate_fn).

    `enable=False` (the default) keeps Phase 3 as a no-op so the
    cascade-widget surface is OFF until soak validates the
    interaction. Tests can opt in with `enable=True`.

    `polish_announcer_beats=False` skips announcer beats entirely
    (per ADR section 6.1 "Either skip Phase 3 entirely OR route to
    _POLISH_SYSTEM_PROMPT_ANNOUNCER"). The composer-side polish_line
    DOES support announcer routing now; the cascade-side default
    stays conservative until soak proves the announcer polish prompt
    behaves on the typical mid-2020s small-LLM corpus.

    Mutation policy:
      * Mutates only `line.text` on lines that polished cleanly.
      * Recomputes `word_count` and `char_count` in lockstep.
      * Never touches skip / char_id / speaker_role.
      * Never adds or removes lines (ADR section 5 hard rule 1).
      * On polish refusal / still-leaky / word-cap miss, leaves the
        pre-polish text in place and records the rejection in the
        Phase3PolishReport.

    Stamps:
      meta.phase_3_polish_record  full report dict for forensics.
    """
    rep = Phase3PolishReport()
    if not enable:
        log.debug("[LFC:phase_3] disabled (enable=False)")
        led.data.setdefault("meta", {})["phase_3_polish_record"] = (
            rep.to_dict()
        )
        return rep

    # Lazy import keeps the orchestrator module load surface clean
    # (regex compilation in _otr_line_composer is non-trivial).
    from ._otr_line_composer import (  # type: ignore
        needs_polish,
        polish_line,
    )

    lines = led.data.get("lines") or []
    cast = led.data.get("cast") or []
    cast_by_id = {
        row.get("char_id", ""): row
        for row in cast
        if isinstance(row, dict)
    }

    rep.lines_scanned = len(lines)
    last_two: list[str] = []

    for idx, ln in enumerate(lines):
        if not isinstance(ln, dict):
            continue
        if ln.get("skip"):
            # Already muted -- nothing to polish. Keep last_two
            # unchanged: skipped lines don't contribute to the
            # rolling 2-line window.
            continue
        text = (ln.get("text") or "")
        if not text:
            continue
        role = (ln.get("speaker_role") or "").strip().lower()
        if role not in ("character", "announcer"):
            # Non-voiced beat (sfx / music). Skip + don't update
            # last_two (those windows are dialogue-only).
            continue
        if role == "announcer" and not polish_announcer_beats:
            # Skip announcer beats per ADR section 6.1 default.
            # Still advance last_two so character beats that
            # follow see the announcer line in their PREVIOUS
            # LINES window.
            last_two.append(text)
            if len(last_two) > 2:
                last_two = last_two[-2:]
            continue
        if not needs_polish(text):
            last_two.append(text)
            if len(last_two) > 2:
                last_two = last_two[-2:]
            continue

        rep.lines_examined += 1
        # Pull the voice card from the cast row.
        cid = ln.get("char_id") or ""
        cast_row = cast_by_id.get(cid) or {}
        voice_card = (
            cast_row.get("voice_card")
            or cast_row.get("name", "")
            or cid
            or "unspecified speaker"
        )
        beat_intent = ln.get("beat_intent") or ln.get("intent") or ""

        try:
            polished = polish_line(
                generate_fn,
                text,
                voice_card,
                speaker_role=role,
                beat_intent=beat_intent,
                previous_lines=tuple(last_two),
                polish_generate_fn=polish_generate_fn,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "[LFC:phase_3] polish_line raised on line_id=%s: %s",
                ln.get("line_id", ""), exc,
            )
            rep.failures.append({
                "line_id": ln.get("line_id", ""),
                "reason": f"{type(exc).__name__}: {exc}",
            })
            last_two.append(text)
            if len(last_two) > 2:
                last_two = last_two[-2:]
            continue

        if polished == text:
            # polish_line falls back to the original on refusal / empty
            # / still-leaky / etc. We can't distinguish refusal vs
            # other rejection types from this side without parsing the
            # log; record as "rejected" and rely on polish_line's own
            # log lines for the breakdown.
            rep.edits_rejected_refusal += 1
        elif needs_polish(polished):
            # Polish output still trips the leak gate -- keep original.
            rep.edits_rejected_still_leaky += 1
        else:
            # Apply the polish.
            ln["text"] = polished
            ln["char_count"] = len(polished)
            ln["word_count"] = sum(1 for _ in polished.split())
            rep.edits_applied += 1
            # Update last_two with the polished text, not the leaky
            # original -- subsequent lines see the cleaned voice.
            text = polished

        last_two.append(text)
        if len(last_two) > 2:
            last_two = last_two[-2:]

    led.data.setdefault("meta", {})["phase_3_polish_record"] = (
        rep.to_dict()
    )
    log.info(
        "[LFC:phase_3] examined=%d applied=%d still_leaky=%d refusal=%d",
        rep.lines_examined, rep.edits_applied,
        rep.edits_rejected_still_leaky, rep.edits_rejected_refusal,
    )
    return rep


def _phase_4_per_scene_coherence_stub(generate_fn, led) -> None:
    """Phase 4 no-op stub. Commit 8 wires in scene coherence.

    B5 fix (commit 12.1): the stub itself only logs; the orchestrator
    stamps a stub_bypassed record on meta.cleanup_passes so
    downstream telemetry / soak diagnostics see a contiguous
    phase-record list with no gaps for stub phases."""
    log.debug("[LFC:phase_4] stub -- per-scene coherence not yet wired")


def _phase_4_5_smart_suggestion(led, *, enable: bool = False, generate_fn=None):
    """LFC Phase 4.5 caller. Wires into _otr_lfc_smart_suggestion."""
    from . import _otr_lfc_smart_suggestion as _LFC_SS  # type: ignore
    return _LFC_SS.phase_4_5_smart_suggestion(
        led, enable=enable, generate_fn=generate_fn,
    )


def _phase_5_voice_drift_stub(generate_fn, led) -> None:
    """Phase 5 no-op stub. Commit 9 wires in voice drift detection.

    B5 fix (commit 12.1): see phase 4 stub note."""
    log.debug("[LFC:phase_5] stub -- voice drift not yet wired")


def _phase_6_episode_arc_stub(generate_fn, led) -> None:
    """Phase 6 no-op stub. Commit 10 wires in episode arc audit.

    B5 fix (commit 12.1): see phase 4 stub note."""
    log.debug("[LFC:phase_6] stub -- episode arc not yet wired")


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
    enable_phase_3_polish: bool = False,
    polish_announcer_beats: bool = False,
    enable_phase_7_audio_readiness: bool = True,
    enable_phase_8_video_readiness: bool = True,
    enable_phase_4_5_smart_suggestion: bool = False,
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

    # ---- Phase 3: per-line polish (LFC commit 4 wiring) -----
    # Phase 3 runs BEFORE the reviewer in the ADR phase chain, but
    # the existing reviewer's snapshot/restore semantics already
    # protect against a polish-introduced phantom (Pass 3 catches
    # the phantom; restore returns to pre-polish state). The
    # mutation order in this orchestrator is therefore:
    #   Phase 0 -> Phase 3 -> Phases 1+2+9 -> Phase 4..8 stubs ->
    #   Phase 10.
    # Default `enable_phase_3_polish=False` keeps the cascade-side
    # polish OFF until soak validates the interaction with the
    # composer's existing inline polish path. When OFF the call is
    # a no-op (Phase3PolishReport with zero counts is stamped).
    started_3 = _isoformat_utc_now()
    hash_before_3 = _hash_lines_text(ledger_data)
    p3_report = _phase_3_per_line_polish(
        generate_fn,
        led,
        polish_generate_fn=polish_generate_fn,
        enable=enable_phase_3_polish,
        polish_announcer_beats=polish_announcer_beats,
    )
    hash_after_3 = _hash_lines_text(ledger_data)
    _stamp_phase_record(
        ledger_data,
        phase_name="phase_3_per_line_polish",
        text_hash_before=hash_before_3,
        text_hash_after=hash_after_3,
        started_at=started_3,
        finished_at=_isoformat_utc_now(),
        edits_proposed=p3_report.lines_examined,
        edits_applied=p3_report.edits_applied,
        failures=p3_report.failures,
    )

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
        # meta.cleanup_passes stays contiguous.
        for skipped_name in (
            "phase_4_per_scene_coherence",
            "phase_4_5_smart_suggestion",
            "phase_5_voice_drift",
            "phase_6_episode_arc",
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
        meta.setdefault("phase_4_5_smart_suggestion", {
            "skipped": True, "skipped_reason": "terminal_skipped",
        })
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
        log.info(
            "[LFC] terminal reviewer verdict %r -- skipping Phase 4..10 "
            "(B7 fix: short-circuit moved before mutation phases)",
            interim_verdict,
        )
        return disp

    # ---- Non-terminal path: Phase 4 / 4.5 / 5 / 6 / 7 / 8 / 10 ----
    # B5: stubs stamp records with reason="stub_bypassed" so the
    # cleanup_passes list stays contiguous.
    _phase_4_per_scene_coherence_stub(generate_fn, led)
    _stamp_stub_or_skipped_phase(
        ledger_data,
        phase_name="phase_4_per_scene_coherence",
        reason="stub_bypassed",
    )
    # Phase 4.5 smart suggestion (LFC commit 11). Default OFF
    # per ADR section 6.17. When enabled, scans for SFX/music
    # implications and appends auto_generated=True entries.
    started_4_5 = _isoformat_utc_now()
    hash_before_4_5 = _hash_lines_text(ledger_data)
    p4_5_report = _phase_4_5_smart_suggestion(
        led,
        enable=enable_phase_4_5_smart_suggestion,
        generate_fn=generate_fn,
    )
    hash_after_4_5 = _hash_lines_text(ledger_data)
    _stamp_phase_record(
        ledger_data,
        phase_name="phase_4_5_smart_suggestion",
        text_hash_before=hash_before_4_5,
        text_hash_after=hash_after_4_5,
        started_at=started_4_5,
        finished_at=_isoformat_utc_now(),
        edits_proposed=(
            p4_5_report.sfx_suggested + p4_5_report.music_suggested
            if p4_5_report is not None else 0
        ),
        edits_applied=(
            p4_5_report.sfx_suggested + p4_5_report.music_suggested
            if p4_5_report is not None else 0
        ),
    )
    _phase_5_voice_drift_stub(generate_fn, led)
    _stamp_stub_or_skipped_phase(
        ledger_data,
        phase_name="phase_5_voice_drift",
        reason="stub_bypassed",
    )
    _phase_6_episode_arc_stub(generate_fn, led)
    _stamp_stub_or_skipped_phase(
        ledger_data,
        phase_name="phase_6_episode_arc",
        reason="stub_bypassed",
    )

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
