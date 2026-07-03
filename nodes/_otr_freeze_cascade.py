"""nodes/_otr_freeze_cascade.py — Ledger Freeze Cascade orchestrator.

Wraps the existing `_otr_ledger_reviewer.review_ledger` 3-pass with
Phase 0 (gap_audit_pre) at entry and Phase 10 (gap_audit_post +
freeze) at exit. Subsequent sprint commits decompose the reviewer's
internal passes into discrete cascade phases (Phase 1 / Phase 2)
AND insert the new LLM phases (Phase 3 polish, Phase 4 scene
coherence, Phase 4.5 smart suggestion, Phase 5 voice drift, Phase 6
episode arc, Phase 7 audio readiness, Phase 8 video readiness).
This commit (commit 2 of 14) ships the orchestrator skeleton only
-- Phase 3/4/4.5/5/6/7/8 are no-op holes for now.

S33 B3 (2026-05-15) retired Phase 9 (the post-edit `audit_cast_contract(
label="post")` LLM call) per the refined no-auditors rule: its only
consumer was the `post_audit_pass` rollback gate that B2 removed.
The phase_name string `phase_1_2_9_reviewer_composite` is retained
for forensic / soak-diagnostic continuity even though the "9"
component no longer fires.

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
    Phase 9    RETIRED in S33 B3 (2026-05-15) -- pipeline cut, no
               editor consumer; rollback gate B2 retired first.
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
from . import _otr_story_critic as _OTRSC
from . import _otr_reroll as _OTRRR
from . import _otr_render_plan as _OTRRP

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
# Freeze WARN taxonomy (story-ledger DRIFT chunk 4, 2026-06-25).
#
# Stop labeling a SHIPPED arc / critic / continuity failure "structural".
# Three deterministic tiers (no LLM):
#   * structural_error       -> a genuinely UNRENDERABLE gap (missing voiced
#                               text, broken structure). BLOCKS at Phase 10
#                               (needs_full_rerun) -- this is the ONLY blocking
#                               tier and it is already enforced by the critical-
#                               gap path; the taxonomy just names it.
#   * story_accuracy_warning -> continuity / canon-divergence / an UNVERIFIED
#                               critic / a ledger-consistency defect. The
#                               episode is RENDERABLE, so it SHIPS -- but NON-
#                               clean (frozen_with_warns) + operator-visible
#                               meta. NEVER a hard block (don't gate audio on a
#                               flaky LLM or a cosmetic-content note).
#   * cosmetic_warning       -> everything else: ships clean-with-warns.
# ---------------------------------------------------------------------------
FREEZE_WARN_TIERS: tuple[str, ...] = (
    "structural_error", "story_accuracy_warning", "cosmetic_warning",
)

#: substrings (casefolded) that mark a finding as genuinely STRUCTURAL
#: (unrenderable) -- kept tight so a content note never reads as structural.
_STRUCTURAL_WARN_KEYS: tuple[str, ...] = (
    "unrenderable", "missing voiced", "no voiced", "empty line", "empty spoken",
    "missing line", "missing beat", "broken link", "no lines", "critical gap",
    "missing audio", "zero-length", "missing speaker",
)
#: substrings that mark a STORY-ACCURACY (content-correctness) finding.
_ACCURACY_WARN_KEYS: tuple[str, ...] = (
    "continuity", "canon", "divergen", "unverified", "inconsistent",
    "consistency", "timeline", "contradict", "prop ", "stance", "premise",
    "sound_palette", "drift",
)


def classify_freeze_warning(text: Any) -> str:
    """Deterministically bucket ONE finding string into a FREEZE_WARN_TIERS
    tier. Structural (unrenderable) wins over accuracy; anything unmatched is
    cosmetic. Pure; never raises."""
    try:
        s = str(text or "").casefold()
        if any(k in s for k in _STRUCTURAL_WARN_KEYS):
            return "structural_error"
        if any(k in s for k in _ACCURACY_WARN_KEYS):
            return "story_accuracy_warning"
        return "cosmetic_warning"
    except Exception:  # noqa: BLE001
        return "cosmetic_warning"


def build_freeze_warn_taxonomy(
    *,
    gap_warnings: Any = (),
    critic_report: Any = None,
    critic_status: Any = None,
    consistency_status: Any = None,
) -> dict:
    """Bucket every non-structural freeze finding into the 3-tier taxonomy.

    Sources: the gap-audit warnings (classified by text), an UNVERIFIED critic
    (critic_status.validated is False -> accuracy), the critic's continuity
    issues (-> accuracy), and any ledger-consistency defects (-> accuracy).
    Returns ``{tier: [messages]}`` for every tier. PURE; never raises."""
    tax: dict = {t: [] for t in FREEZE_WARN_TIERS}
    try:
        for w in (gap_warnings or ()):
            tax[classify_freeze_warning(w)].append(str(w)[:240])
        # an unverified critic is a story-ACCURACY warning (chunk 1 fail-closed),
        # never structural -- the episode renders, the arc just was not checked.
        if isinstance(critic_status, dict) and critic_status.get("validated") is False:
            tax["story_accuracy_warning"].append(
                "story_critic_unverified: the arc was not actually evaluated"
            )
        if critic_report is not None:
            for ci in (getattr(critic_report, "continuity_issues", None) or []):
                issue = (ci.get("issue") if isinstance(ci, dict)
                         else getattr(ci, "issue", "")) or ""
                tax["story_accuracy_warning"].append(
                    f"continuity: {str(issue)[:200]}"
                )
        if isinstance(consistency_status, dict) and not consistency_status.get(
            "clean", True
        ):
            for d in (consistency_status.get("defects") or []):
                fld = d.get("field") if isinstance(d, dict) else str(d)
                tax["story_accuracy_warning"].append(f"consistency: {fld}")
    except Exception:  # noqa: BLE001 -- taxonomy is observability, never fatal
        pass
    return tax


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
#   cleanup_passes    Phase 1+2 (reviewer composite, Phase 9 retired
#                     in S33 B3) + Phase 3 (polish) + Phase 4 (scene
#                     coherence) + Phase 4.5 (smart suggestion) +
#                     Phase 5 (voice drift) + Phase 6 (episode arc)
#   readiness_passes  Phase 7 (audio readiness) + Phase 8 (video
#                     readiness)
# S30 B4: phase_3 / 4 / 4.5 / 5 / 6 entries DELETED from this table.
# Those phases were removed from the cascade in the same commit.
# S33 B3 (2026-05-15): Phase 9 component of the reviewer composite
# retired. The phase_name string `phase_1_2_9_reviewer_composite`
# stays for forensic continuity even though only Phase 1 + 2 actually
# run inside it.
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
# BUG-LOCAL-278: defensive cascade meta persistence
# ---------------------------------------------------------------------------


def _persist_cascade_meta(led) -> None:
    """Persist the in-memory ledger to disk at every cascade exit.

    BUG-LOCAL-278 fix. The freeze cascade modifies `meta` in place
    (`freeze_verdict`, `freeze_disposition`, `story_critic_report`,
    etc.) but does not save the ledger to disk
    at its exit points. Downstream consumers split into two camps:

      * In-memory readers (BatchBarkGenerator, MusicGenTheme, the
        SceneSequencer that flows the ledger socket-to-socket) SEE the
        cascade's stamps from the live `led.data` reference -- the
        bark halt for BUG-LOCAL-276 works correctly.

      * Disk readers (KokoroAnnouncer's load_ledger_safe + write-back
        path, post-run forensic tools, the operator's manual inspection
        of `pending_*_ledger.json`) DO NOT see them. KokoroAnnouncer in
        particular reads the pre-cascade ledger from disk, stamps its
        own line_id fields, and writes back -- silently dropping every
        cascade modification.

    Symptom: Step 8 operator gate criteria (meta.freeze_verdict,
    meta.story_critic_report) are MISSING from the
    persisted ledger even when the in-memory log confirms they were
    stamped. The operator cannot verify the gate criteria from the
    .json after the run; the data exists only in the live log tail.

    Fix: persist the ledger at every cascade exit (terminal-skip,
    Phase 10 rejection, successful freeze, render-plan post-stamp
    [BUG-LOCAL-274 precedent]). Production Ledger.save() never raises
    (Prime Directive 1); defensive getattr handles test fixtures that
    stub `led` as a SimpleNamespace without a save method.

    PD1 (audio is king): N/A; persistence is forensic. Never raises.
    """
    _save_fn = getattr(led, "save", None)
    if callable(_save_fn):
        try:
            _save_fn()
        except Exception as _save_exc:  # noqa: BLE001 -- PD1
            log.warning(
                "[LFC] cascade meta persistence save failed (%s: %s); "
                "cascade meta lives in-memory only -- downstream nodes "
                "still see it, but the .json on disk is stale",
                type(_save_exc).__name__, _save_exc,
            )


# ---------------------------------------------------------------------------
# Terminal-skip disposition (shared by the reviewer-terminal and the
# Sprint 5C reroll-exhaustion exits)
# ---------------------------------------------------------------------------


def _build_terminal_skip_disposition(
    ledger_data: dict,
    *,
    verdict: str,
    reviewer_disp: Optional[_OTRLR.ReviewerDisposition],
    pre_report: _LFC.GapAuditReport,
    block_class: str = "structural",
) -> FreezeDisposition:
    """Stamp the terminal-skip state and build the FreezeDisposition.

    Shared by the two cascade exits that end WITHOUT running Phase 10:
    the reviewer-terminal path (too_many_edits / needs_full_rerun from
    review_ledger) and the Sprint 5C reroll-exhaustion path. Both leave
    the ledger in a state Phase 10 must not freeze, so Phase 7 / 8 / 10
    are stamped `terminal_skipped` and the audio / video readiness meta
    keys carry the same reason. The freeze_verdict STRING, the full
    freeze_disposition dict, and the compact per-phase telemetry are all
    stamped here so both exits leave an identically shaped meta.
    """
    meta = ledger_data.setdefault("meta", {})
    meta["freeze_verdict"] = verdict
    # BUG-LOCAL-300 safety/quality split: record WHY this terminal verdict
    # fired so BatchBarkGenerator can tell a genuinely unrenderable ledger
    # ("structural") from a renderable-but-low-quality one ("quality").
    meta["freeze_block_class"] = block_class
    # B5: stamp the remaining phases as terminal_skipped so
    # meta.cleanup_passes / readiness_passes stay contiguous. S30 B4:
    # phase 4 / 4.5 / 5 / 6 names removed -- those phases no longer exist.
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
    # Stamp the skipped-phase status on their respective meta keys so
    # downstream readers see a consistent shape.
    meta.setdefault("audio_readiness", {
        "skipped": True, "skipped_reason": "terminal_skipped",
    })
    meta.setdefault("video_readiness", {
        "skipped": True, "skipped_reason": "terminal_skipped",
    })
    disp = FreezeDisposition(
        verdict=verdict,
        reviewer_disposition=reviewer_disp,
        gap_audit_pre=pre_report,
        gap_audit_post=None,
    )
    meta["freeze_disposition"] = disp.to_dict()
    # C3 (clean-break 2026-05-12): compact per-phase telemetry for soak
    # diagnostics. Lives on meta only -- the freeze_verdict output STRING
    # stays the verdict literal so graph-canvas previews are readable.
    meta["freeze_phase_telemetry"] = build_phase_telemetry(meta)
    return disp


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
    # Sprint 6 -- critic-to-render coupling widgets (cascade node
    # surface). Defaults match the node's INPUT_TYPES defaults so
    # existing callers / tests are unaffected (render_selection="all"
    # with no flat_lines / arc-block reduces to "stamp every character
    # line_id in ledger order" -- HuMo's filter is then a no-op).
    render_selection: str = "all",
    render_max_n: int = 6,
    protagonist_only: bool = False,
    manual_line_ids: str = "",
) -> FreezeDisposition:
    """Orchestrate Phase 0 -> reviewer (Phase 1+2) -> Phase 10.

    Behaviour:
      * Phase 0 always runs first; warn-only. Records to
        meta.cleanup_passes.
      * Existing review_ledger runs next. Only Phases 1 + 2 fire
        inside it after S33 B3 (Phase 9 retired per refined
        no-auditors rule). The composite phase_name string
        `phase_1_2_9_reviewer_composite` is retained for forensic
        continuity.
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

    # ---- VRAM telemetry at cascade entry (ADR 6.8) ----
    # Single measurement stamped for forensics; no ceiling policy --
    # the tier JSON the operator picks owns the OOM budget now.
    meta = ledger_data.setdefault("meta", {})
    try:
        from . import _otr_lfc_watchdog as _LFC_WD  # type: ignore
        current_gb = _LFC_WD._torch_vram_allocated_gb()
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "[LFC] VRAM read failed at cascade entry: %s; "
            "stamping 0.0 GB and proceeding", exc,
        )
        current_gb = 0.0
    meta["vram_at_cascade_entry_gb"] = float(current_gb)

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

    # ---- Phase 1 + 2: reviewer composite (Phase 9 retired S33 B3) ----
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
        # review_ledger has already restored the ledger to its
        # pre-review state; Phase 10 must not run. Stamp the terminal
        # skip + build the disposition via the shared helper (the same
        # path the Sprint 5C reroll-exhaustion exit uses).
        disp = _build_terminal_skip_disposition(
            ledger_data,
            verdict=interim_verdict,
            reviewer_disp=reviewer_disp,
            pre_report=pre_report,
            # Reviewer cast/contract terminal -> the ledger is structurally
            # unrenderable; Bark must halt (BUG-LOCAL-300).
            block_class="structural",
        )
        log.info(
            "[LFC] terminal reviewer verdict %r -- skipping Phase 4..10 "
            "(B7 fix: short-circuit moved before mutation phases)",
            interim_verdict,
        )
        # BUG-LOCAL-278: persist cascade meta before the terminal exit
        # so downstream disk readers (Kokoro, forensic tools) see the
        # freeze verdict + disposition stamps. See _persist_cascade_meta.
        _persist_cascade_meta(led)
        return disp

    # ---- Sprint 5B: whole-script story-quality critic --------------
    # Post-Script-Doctor (review_ledger above ran Phases 1+2), on the
    # non-terminal path only -- the ledger is cast-clean and
    # structurally sound, so the critic spends its whole budget on the
    # one thing no earlier pass judges: DRAMATIC QUALITY (continuity,
    # voice drift, flat lines, arc verdict -- audit Finding C). For 5B
    # the report is ADVISORY: it changes no line text. It is stamped on
    # meta for Sprint 5C (targeted reroll reads `reroll_targets`) and
    # Sprint 6 (render coupling reads `render_priority` / `flat_lines` /
    # `arc_verdict`). run_story_critic NEVER raises -- on any failure it
    # returns StoryCriticReport.clean(), so a critic failure can never
    # break the freeze (Prime Directive 1). The `# LLM slot: technical`
    # tag lives at the structured_call site inside run_story_critic;
    # the critic reuses the technical model already resident in the
    # cascade -- no new widget, no VRAM swap (Prime Directive 6).
    #
    # BUG-LOCAL-273 fix: refresh ledger_data + meta from led.data BEFORE the
    # critic stamp. The Phase 1+2 reviewer above calls Ledger.save()
    # internally; production_ledger.py:1012 rebinds self.data to a new
    # merged dict on every save, detaching the cascade's L570/L576 locals.
    # Without this refresh the stamp lands on an orphaned dict and the
    # reroll/render_plan/HuMo never see it (live run
    # signal_lost_bioluminescent_trench_descent_20260525_182002: critic
    # produced 4 reroll_targets / arc_verdict=uneven; reroll + render_plan
    # both saw clean()/defaults).
    ledger_data = led.data
    meta = ledger_data.setdefault("meta", {})
    story_critic_report = _OTRSC.run_story_critic(
        generate_fn,
        ledger_data,
        ledger_data.get("cast", []) or [],
    )
    meta["story_critic_report"] = story_critic_report.model_dump()
    # story-ledger DRIFT (2026-06-25): derive observable critic provenance so a
    # fail-CLOSED critic (arc_verdict="unverified" == run_story_critic.clean())
    # is visible, not silent. `validated` is deterministic off the verdict --
    # the critic NEVER emits "unverified" on a real evaluation (it is the
    # clean() failure sentinel); a real verdict is strong/uneven/flat/
    # mid_collapse. Lives on meta only -- the frozen report schema is unchanged.
    _critic_unverified = (story_critic_report.arc_verdict == "unverified")
    meta["story_critic_status"] = {
        "ran": True,
        "validated": not _critic_unverified,
        "failure": "critic_unverified_fail_closed" if _critic_unverified else "",
    }
    log.info(
        "[LFC] Sprint 5B story critic: arc_verdict=%s, %d reroll "
        "target(s), %d flat line(s), %d continuity issue(s), %d line(s) "
        "in render priority (advisory -- no line text changed)",
        story_critic_report.arc_verdict,
        len(story_critic_report.reroll_targets),
        len(story_critic_report.flat_lines),
        len(story_critic_report.continuity_issues),
        len(story_critic_report.render_priority),
    )

    # ---- A3 (story-quality Phase 1): UNCONDITIONAL mechanical floor ----
    # run_story_critic returns StoryCriticReport.clean() identically whether
    # it succeeded, found nothing, or silently failed -- the caller cannot
    # detect a failure. So a deterministic, no-LLM mechanical pass runs
    # UNCONDITIONALLY here and its (line_id, hint) targets are UNIONED into
    # the critic's reroll_targets. This guarantees real repair targets exist
    # for the escalation + reroll loop below whenever the deterministic floor
    # finds a near-duplicate / "What if...?" loop defect, EVEN on a silent
    # critic failure (no failure flag, no clean() mutation, no critic-retry).
    # Character-only: the reroll path cannot act on announcer lines and the
    # announcer taglines are exempt by construction. Never raises.
    try:
        from . import _otr_anti_loop as _OTRAL  # type: ignore
        _existing_ids = {
            str(getattr(t, "line_id", ""))
            for t in story_critic_report.reroll_targets
        }
        _mech_added = 0
        for _lid, _hint in _OTRAL.anti_loop_reroll_targets(ledger_data):
            if _lid in _existing_ids:
                continue
            story_critic_report.reroll_targets.append(
                _OTRSC.RerollTarget(line_id=_lid, hint=_hint)
            )
            _existing_ids.add(_lid)
            _mech_added += 1
        if _mech_added:
            # story-ledger DRIFT (2026-06-25): a report cannot be CLEAN and
            # also DEMAND rerolls. If the mechanical floor adds anti-loop
            # targets to a `strong`/`unverified` report, deterministically
            # downgrade the arc verdict to `uneven` so the verdict matches the
            # repair work it is asking for (closes the strong-but-rerolling
            # contradiction).
            if story_critic_report.arc_verdict in ("strong", "unverified"):
                story_critic_report.arc_verdict = "uneven"
            meta["story_critic_report"] = story_critic_report.model_dump()
            meta["anti_loop_mechanical_targets_added"] = _mech_added
            log.info(
                "[LFC] A3 mechanical floor: unioned %d deterministic "
                "anti-loop target(s) into the critic reroll set "
                "(total now %d, arc_verdict=%s).",
                _mech_added, len(story_critic_report.reroll_targets),
                story_critic_report.arc_verdict,
            )
    except Exception as _a3_exc:  # noqa: BLE001 -- floor must never break freeze
        log.warning("[LFC] A3 mechanical floor failed: %r", _a3_exc)

    # ---- Sprint 5C: targeted reroll loop ---------------------------
    # The Sprint 5B critic above stamped meta.story_critic_report but
    # changed no line text (advisory). Sprint 5C ACTS on it:
    # run_targeted_reroll re-composes every line the critic named in
    # `reroll_targets`, threading the critic's concrete hint in as a hard
    # REVISE instruction, then re-runs the critic -- capped at
    # MAX_REROLL_CYCLES cycles. The re-composition runs on the technical
    # model already resident here (Sprint 5C fork Option A: no
    # creative-slot VRAM swap, no node-surface change). It NEVER raises
    # (Prime Directive 1) -- a failed re-composition keeps the original
    # line, an unexpected error degrades to freeze-what-we-have. If the
    # critic still names targets after the cap, the reroll restores the
    # pre-reroll lines and returns the `needs_full_rerun` terminal
    # verdict, handled here exactly like a terminal reviewer verdict
    # (skip Phase 7 / 8 / 10 via the shared terminal-skip helper).
    # BUG-LOCAL-281 (Sprint 10B Wave 0 follow-on, 2026-05-27):
    # Stage 7 ship-verdict gate on the legacy Sprint 5C reroll loop.
    # Live operator soak 2026-05-27 surfaced a critic-disagreement
    # gate failure: Stage 7 whole-episode critic returned
    # verdict=ship mean=3.70 failing_axes=[] on a freshly composed
    # ledger, but the legacy story critic (run inside the cascade
    # above) independently flagged 2 reroll targets, which forced
    # the Sprint 5C reroll loop, which exhausted both cycles trying
    # to lengthen short-but-shippable lines, which stamped
    # needs_full_rerun, which halted Bark per BUG-LOCAL-276.
    # Stage 7 is the new authoritative critic (Sprint 10A step 7);
    # legacy reroll only fires when Stage 7 has NOT confirmed a ship.
    # When Stage 7 says ship, construct a no-op disposition and skip
    # the reroll call entirely -- the composed lines ship as-is.
    # Interim gate until Sprint 10B Wave 1 Agent C (critic-driven
    # escalation) replaces the legacy reroll wholesale.
    # Sprint 10B Wave 1 Agent C (2026-05-27): critic-driven reroll
    # escalation. Replaces the Wave 0 BUG-LOCAL-281 simple ship-gate
    # with a typed scope decision. Structural failures (premise_clarity
    # / continuity / resolution / emotional_arc) route to whole-episode
    # regenerate immediately -- the line reroll loop cannot fix arc
    # problems, so its cycles would exhaust pointlessly. Local
    # failures fall back to the legacy line reroll. See
    # nodes/_otr_reroll_escalation.decide_escalation_scope and
    # docs/2026-05-26-good-story-writer-architecture__02_design.md
    # Section 4 Wave 1 Agent C.
    from . import _otr_reroll_escalation as _OTRRE
    # Stage-7 whole-episode shadow critic removed (2026-05-29 lean-down): it
    # never stamped a verdict in production, so this signal was always empty.
    # The escalation engine decides from the legacy story-critic targets alone
    # (empty Stage-7 signal -> LINE when targets pending, else NONE).
    # T2 (2026-06-23): OTR_ENABLE_CRITIC_ESCALATION gates whether the 5B critic's
    # arc_verdict drives structural escalation. Default OFF => {} (byte-identical).
    from . import _otr_story_select as _OTRSEL_ESC
    _w0f_s7 = _OTRSEL_ESC.build_escalation_signal(story_critic_report, meta)
    _w1c_escalation = _OTRRE.decide_escalation_scope(
        _w0f_s7,
        story_critic_targets=(
            getattr(story_critic_report, "reroll_targets", None)
            or []
        ),
    )
    meta["reroll_escalation"] = _w1c_escalation.to_dict()
    log.info(
        "[LFC] Wave 1 Agent C escalation: scope=%s reason=%s targets=%d",
        _w1c_escalation.scope.value,
        _w1c_escalation.reason[:120],
        len(_w1c_escalation.target_beat_ids),
    )

    if _w1c_escalation.scope is _OTRRE.EscalationScope.NONE:
        # Stage 7 ship verdict (or equivalent). Skip reroll. Construct
        # a no-op RerollDisposition so downstream code that reads
        # .verdict / .cycles_run / .lines_rerolled keeps working.
        reroll_disp = _OTRRR.RerollDisposition(
            verdict="no_reroll",
            cycles_run=0,
            lines_rerolled=0,
            final_report=story_critic_report,
        )
        meta["reroll_skipped_by_stage7_ship"] = True
        log.info(
            "[LFC] Stage 7 verdict=ship -- skipping Sprint 5C legacy "
            "reroll loop; episode ships as composed."
        )
    elif _w1c_escalation.scope is _OTRRE.EscalationScope.EPISODE:
        # Structural failure -- the arc is broken, not the lines.
        # Stamp needs_full_rerun immediately. Skips the wasted cycles
        # the legacy line reroll would burn trying to fix an
        # unfixable structural problem (BUG-LOCAL-279/280/281
        # family). The downstream `if reroll_disp.verdict ==
        # 'needs_full_rerun'` branch handles the terminal exit.
        reroll_disp = _OTRRR.RerollDisposition(
            verdict="needs_full_rerun",
            cycles_run=0,
            lines_rerolled=0,
            final_report=story_critic_report,
        )
        meta["reroll_skipped_by_structural_escalation"] = True
        if _w1c_escalation.regeneration_hint:
            meta["regeneration_hint"] = _w1c_escalation.regeneration_hint
        log.warning(
            "[LFC] Wave 1 Agent C: structural failure -- skipping "
            "line reroll, stamping needs_full_rerun immediately. "
            "Hint: %s",
            (_w1c_escalation.regeneration_hint or "")[:200],
        )
    else:
        # BEAT or LINE scope -- both currently route through the
        # legacy line reroll. Wave 2 can split beat-from-line if
        # operator A/B finds a quality lift there.
        reroll_disp = _OTRRR.run_targeted_reroll(generate_fn, led)
        log.info(
            "[LFC] Sprint 5C targeted reroll (scope=%s): verdict=%s, "
            "%d cycle(s), %d line(s) re-composed",
            _w1c_escalation.scope.value,
            reroll_disp.verdict, reroll_disp.cycles_run,
            reroll_disp.lines_rerolled,
        )
    if reroll_disp.verdict == "needs_full_rerun":
        # ---- A2 (story-quality Phase 1): repair-then-ship, NEVER refuse ----
        # The repair loop is BOUNDED (run_targeted_reroll caps at
        # MAX_REROLL_CYCLES) and has already run; the critic still names
        # subjective quality targets. We are on the NON-TERMINAL path -- the
        # reviewer already passed the cast + structural audit, so the
        # candidate (run_targeted_reroll restores the reviewed pre-reroll
        # lines on exhaustion) PASSES the structural validators and is
        # renderable. The old behaviour skipped Phase 7..10 and stamped a
        # terminal needs_full_rerun, aborting a finished, renderable episode
        # -- the documented "passes the worst, fails the best" bug. Instead
        # we SHIP it: ordered selection prefers the structurally-valid
        # candidate, we LOG the residual validator findings for the record,
        # and we FALL THROUGH to the normal freeze (Phase 6 render plan +
        # Phase 7/8/10 below). Phase 10's gap audit remains the final
        # structural gate -- a genuinely unrenderable ledger still fails
        # there (block_class=structural, the PD1 audio guard); a renderable-
        # but-low-quality one now ships instead of aborting.
        # BUG-LOCAL-273: refresh ledger_data + meta -- run_targeted_reroll may
        # have called Ledger.save() (rebinding led.data) so the stamp lands on
        # the live dict, not an orphan.
        ledger_data = led.data
        meta = ledger_data.setdefault("meta", {})
        try:
            _a2_resid = _LFC.run_gap_audit(ledger_data, label="a2_ship_through")
            _a2_errs = list(_a2_resid.errors)
            _a2_warns = list(_a2_resid.warnings)
        except Exception as _a2_exc:  # noqa: BLE001 -- never break the ship
            log.warning("[LFC] A2 ship-through gap audit failed: %r", _a2_exc)
            _a2_errs, _a2_warns = [], []
        meta["a2_ship_through"] = {
            "reason": "reroll_exhausted_or_structural_escalation",
            "cycles_run": int(getattr(reroll_disp, "cycles_run", 0) or 0),
            "residual_structural_errors": _a2_errs,
            "residual_warnings": _a2_warns,
        }
        log.warning(
            "[LFC] A2 repair-then-ship: reroll bounded at %d cycle(s), "
            "critic still flagging -- SHIPPING the best candidate through "
            "the normal freeze (%d residual structural error(s), %d "
            "warning(s) logged; never refusing).",
            int(getattr(reroll_disp, "cycles_run", 0) or 0),
            len(_a2_errs), len(_a2_warns),
        )
        # No terminal-skip, no early return -- execution continues to the
        # Phase 6 render plan + Phase 7/8/10 ship path below.

    # ---- Sprint 6: critic -> render coupling -----------------------
    # Compute the render plan AFTER the reroll completes (so the plan
    # sees the FINAL critic report + cycle_count + reroll_history) and
    # BEFORE Phase 7 / 8 / 10 freeze the ledger. The plan rides on
    # `meta.render_plan` -- BatchHumoRender reads it to filter which
    # lines actually get rendered, gate on arc_verdict, and apply
    # render_max_n. build_render_plan NEVER raises (PD1) -- a degraded
    # computation returns None and the stamp is skipped, which makes
    # HuMo fall back to its pre-Sprint-6 render-all behaviour.
    #
    # BUG-LOCAL-273 fix (cont.): refresh ledger_data + meta again before the
    # Sprint 6 stamp. The Sprint 5C reroll above may have triggered an
    # internal Ledger.save() (update_line_text on rerolled lines is followed
    # by save() in some paths), rebinding self.data again. Same root cause
    # as the 5B refresh above.
    ledger_data = led.data
    meta = ledger_data.setdefault("meta", {})
    render_plan = _OTRRP.build_render_plan(
        ledger_data,
        render_selection=render_selection,
        render_max_n=render_max_n,
        protagonist_only=protagonist_only,
        manual_line_ids=manual_line_ids,
    )
    if render_plan is not None:
        meta["render_plan"] = render_plan
        log.info(
            "[LFC] Sprint 6 render plan: mode=%s, %d line(s) (blocked=%s, "
            "applied_max_n=%d, arc_verdict=%s, cycle_count=%d, "
            "%d flat line(s) excluded)",
            render_plan["selection_mode"], len(render_plan["line_ids"]),
            render_plan["blocked"], render_plan["applied_max_n"],
            render_plan["source_arc_verdict"],
            render_plan["source_cycle_count"],
            len(render_plan["excluded_flat_lines"]),
        )
        # BUG-LOCAL-274 fix: persist the render_plan stamp to disk so
        # OTR_BatchHumoRender (a separate node invocation that loads the
        # ledger via _load_ledger_with_path from *_ledger.json) actually
        # sees it. Phase 10's later save would normally cover this, but
        # the belt-and-braces save here makes the contract explicit and
        # survives any future Phase 10 refactor. Production Ledger.save()
        # never raises (Prime Directive 1). Defensive getattr handles
        # test fixtures that stub `led` as a SimpleNamespace without a
        # save method (test_lfc_phase_7_8_readiness, etc.).
        _save_fn = getattr(led, "save", None)
        if callable(_save_fn):
            try:
                _save_fn()
            except Exception as _save_exc:  # noqa: BLE001 -- PD1
                log.warning(
                    "[LFC] post-render-plan save failed (%s: %s); "
                    "render_plan lives in-memory only -- HuMo will "
                    "fall back to render-all", type(_save_exc).__name__,
                    _save_exc,
                )

    # ---- D3 (2026-06-22, story-quality lift): MANDATORY pre-freeze role sweep
    # The FINAL mutation step before the freeze hash + role-dependent routing
    # (Phase 7 audio readiness / scrub / TTS). The reviewer's role_mismatch
    # repair can leave a cast character stamped speaker_role="announcer" (b011
    # "Chandra's Echo": char_id=c02, role=announcer). Force speaker_role=
    # "character" on every line whose char_id is a real cast id. Audit rides
    # meta["role_coercions"] + per-line compose_flags. COERCE-NEVER-CRASH (any
    # failure leaves the ledger untouched -- the freeze never breaks, PD1).
    try:
        from . import production_ledger as _PL  # type: ignore
        ledger_data = led.data
        meta = ledger_data.setdefault("meta", {})
        _d3_cast_ids = _PL.cast_ids_from_ledger(ledger_data)
        _d3_coerced: list[str] = []
        if _d3_cast_ids:
            for _row in ledger_data.get("lines", []) or []:
                _, _ch = _PL.coerce_speaker_role_for_char_id(
                    _row, _d3_cast_ids, source="pre_freeze_sweep",
                )
                if _ch:
                    _d3_coerced.append(str(_row.get("line_id") or ""))
        if _d3_coerced:
            meta["role_coercions"] = {
                "count": len(_d3_coerced), "line_ids": _d3_coerced,
            }
            log.warning(
                "[LFC] D3 pre-freeze role sweep coerced %d row(s) to "
                "character (char_id is a cast id): %s",
                len(_d3_coerced), _d3_coerced,
            )
    except Exception as _d3_exc:  # noqa: BLE001 -- sweep must never break freeze
        log.warning("[LFC] D3 pre-freeze role sweep failed: %r", _d3_exc)

    # D3 CI-only invariant (gated on OTR_TEST_MODE so production never crashes --
    # COERCE-NEVER-CRASH). NOT inside the sweep try/except above (that would
    # swallow the AssertionError). Music rows are separate (not asserted).
    import os as _os_d3
    if _os_d3.environ.get("OTR_TEST_MODE"):
        try:
            _ci_cast_ids = _PL.cast_ids_from_ledger(led.data)
        except Exception:  # noqa: BLE001
            _ci_cast_ids = set()
        for _row in led.data.get("lines", []) or []:
            if not isinstance(_row, dict):
                continue
            _cid = str(_row.get("char_id") or "").strip()
            _role = str(_row.get("speaker_role") or "")
            if _cid in _ci_cast_ids:
                assert _role == "character", (
                    f"D3 invariant: cast char_id {_cid!r} must be role "
                    f"'character', got {_role!r} (line_id="
                    f"{_row.get('line_id')!r})"
                )
            if _role == "announcer":
                assert _cid not in _ci_cast_ids, (
                    f"D3 invariant: announcer role on cast char_id {_cid!r} "
                    f"(line_id={_row.get('line_id')!r})"
                )

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
        # BUG-LOCAL-300: Phase 10 gap-audit rejection = critical structural
        # gaps -> the ledger is genuinely unrenderable. Mark structural so the
        # Bark halt always blocks (never downgraded to a quality warn-through).
        meta["freeze_block_class"] = "structural"
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
        # BUG-LOCAL-278: persist cascade meta on the Phase 10 reject path.
        _persist_cascade_meta(led)
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
    # story-ledger DRIFT chunk 4 (2026-06-25): the freeze WARN taxonomy. Bucket
    # every non-structural finding (gap-audit warns + an UNVERIFIED critic +
    # critic continuity issues + ledger-consistency defects) into the 3 tiers
    # and stamp it operator-visible. A STORY-ACCURACY warning (continuity /
    # canon-divergence / unverified / consistency) ships the renderable episode
    # NON-clean (frozen_with_warns) -- it is NOT a structural block (which is
    # reserved for the genuinely unrenderable Phase-10 critical gap above). This
    # generalizes the chunk-1 unverified->non-clean map; a real, validated,
    # clean arc with no warnings still ships frozen_clean.
    _warn_tax = build_freeze_warn_taxonomy(
        gap_warnings=list(getattr(post_report, "warnings", None) or []),
        critic_report=story_critic_report,
        critic_status=meta.get("story_critic_status"),
        consistency_status=meta.get("consistency_status"),
    )
    meta["freeze_warn_taxonomy"] = _warn_tax
    if _warn_tax["story_accuracy_warning"]:
        meta["freeze_story_accuracy_warnings"] = _warn_tax["story_accuracy_warning"]
    if getattr(story_critic_report, "arc_verdict", "") == "unverified":
        meta["freeze_unverified_critic"] = True
    if final_verdict == "frozen_clean" and (
        _warn_tax["story_accuracy_warning"] or _warn_tax["cosmetic_warning"]
    ):
        final_verdict = "frozen_with_warns"
        meta["freeze_verdict"] = "frozen_with_warns"
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
    # BUG-LOCAL-278: persist cascade meta on the successful-freeze exit.
    _persist_cascade_meta(led)
    return disp
