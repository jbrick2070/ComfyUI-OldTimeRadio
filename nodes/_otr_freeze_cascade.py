"""Final ledger safety, readiness, and freeze orchestration.

Every bank converges here. Inline banks may receive one bounded atomic patch
for the shared narrow spoken-safety policy; producer-owned banks arrive sealed
and are checked read-only. Word count, visual vocabulary, style, craft, and
subjective quality are telemetry or generation guidance only and never affect
liveness. Genuine ledger structure, authorship, provenance, or residual safety
corruption remains fail-closed.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from . import _otr_ledger_freeze as _LFC
from . import _otr_word_delivery as _OTRWD

log = logging.getLogger("OTR.freeze_cascade")


__all__ = [
    "FreezeDisposition",
    "run_freeze_cascade",
    "all_phase_passes",
    "build_phase_telemetry",
]


# ---------------------------------------------------------------------------
# Freeze WARN taxonomy (story-ledger DRIFT chunk 4, 2026-06-25).
#
# Keep renderability errors distinct from non-terminal accuracy warnings.
# Three deterministic tiers (no LLM):
#   * structural_error       -> a genuinely UNRENDERABLE gap (missing voiced
#                               text, broken structure). BLOCKS at Phase 10
#                               (needs_full_rerun) -- this is the ONLY blocking
#                               tier and it is already enforced by the critical-
#                               gap path; the taxonomy just names it.
#   * story_accuracy_warning -> continuity, canon, or consistency telemetry.
#                               The
#                               episode is RENDERABLE, so it SHIPS -- but NON-
#                               clean (frozen_with_warns) + operator-visible
#                               meta. NEVER a hard block (don't gate audio on a
#                               an advisory or cosmetic-content note).
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
    consistency_status: Any = None,
) -> dict:
    """Bucket non-terminal deterministic warnings for operator telemetry."""
    tax: dict = {tier: [] for tier in FREEZE_WARN_TIERS}
    try:
        for warning in gap_warnings or ():
            tax[classify_freeze_warning(warning)].append(str(warning)[:240])
        if isinstance(consistency_status, dict) and not consistency_status.get(
            "clean", True,
        ):
            for defect in consistency_status.get("defects") or ():
                field_name = (
                    defect.get("field") if isinstance(defect, dict)
                    else str(defect)
                )
                tax["story_accuracy_warning"].append(
                    f"consistency: {field_name}"
                )
    except Exception:
        pass
    return tax


# ---------------------------------------------------------------------------
# Disposition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FreezeDisposition:
    """End-of-cascade summary stamped to meta.freeze_disposition."""

    verdict: str
    cleanup_receipt: Optional[dict[str, Any]]
    gap_audit_pre: _LFC.GapAuditReport
    gap_audit_post: Optional[_LFC.GapAuditReport]

    def to_dict(self) -> dict:
        pre = self.gap_audit_pre
        post = self.gap_audit_post
        return {
            "verdict": self.verdict,
            "cleanup_receipt": (
                dict(self.cleanup_receipt)
                if isinstance(self.cleanup_receipt, dict)
                else None
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


@dataclass(frozen=True)
class FreezePolicy:
    """One resolved mutation policy for the final ledger boundary.

    inline_safety_cleanup permits only the shared atomic safety patch and
    deterministic role normalization. content_owned_readonly verifies the
    producer's sealed authorship and structure without changing canonical text.
    A tagged bank that cannot resolve returns a terminal configuration error;
    an untagged legacy ledger uses the inline safety policy for migration.
    """

    name: str
    source: str
    run_inline_safety_cleanup: bool
    terminal_error: str = ""


def resolve_freeze_policy(meta: dict) -> FreezePolicy:
    """Resolve the FreezePolicy from meta.source_bank. Pure decision;
    never raises (a resolution failure is RETURNED as terminal_error so
    the cascade can stamp a truthful receipt before halting)."""
    bank_id = str((meta or {}).get("source_bank") or "")
    if not bank_id:
        return FreezePolicy(
            name="inline_safety_cleanup",
            source="untagged ledger (no meta.source_bank) -- legacy "
                   "route retained for migration",
            run_inline_safety_cleanup=True,
        )
    try:
        from . import _otr_story_routing as _RT
        pack = _RT.resolve_story_pack(bank_id)
        stages = getattr(pack, "prompt_stages", None) or {}
        if str(stages.get("line_composer_system") or "").strip():
            return FreezePolicy(
                name="inline_safety_cleanup",
                source=f"pack for bank {bank_id!r} declares the "
                       "line_composer_system seam",
                run_inline_safety_cleanup=True,
            )
        return FreezePolicy(
            name="content_owned_readonly",
            source=f"pack for bank {bank_id!r} declares NO "
                   "line_composer_system seam -- the lane owns its own "
                   "content loop",
            run_inline_safety_cleanup=False,
        )
    except Exception as exc:  # noqa: BLE001 -- returned, not raised
        return FreezePolicy(
            name="policy_resolution_failed",
            source=f"bank {bank_id!r} is TAGGED but failed to resolve",
            run_inline_safety_cleanup=False,
            terminal_error=(
                f"freeze policy resolution failed for declared bank "
                f"{bank_id!r}: {type(exc).__name__}: {exc} -- refusing "
                "to fail open into legacy content mutation"
            ),
        )


def _sha256_lines_text(ledger_data: dict) -> str:
    """Stable canonical-text fingerprint for the capability receipt
    (unlike `_hash_lines_text`, not process-salted)."""
    import hashlib
    h = hashlib.sha256()
    for ln in ledger_data.get("lines") or []:
        h.update(str((ln or {}).get("line_id", "")).encode("utf-8"))
        h.update(b"\x00")
        h.update(str((ln or {}).get("text", "")).encode("utf-8"))
        h.update(b"\x01")
    return h.hexdigest()


def _sha256_content_authorship(ledger_data: dict) -> str:
    """Validate and fingerprint the generic accepted-artifact receipt."""
    import hashlib
    if not isinstance((ledger_data.get("meta") or {}).get("content_authorship"), dict):
        return hashlib.sha256(b"").hexdigest()
    from ._otr_content_authorship import receipt_sha256
    return receipt_sha256(ledger_data)


def _readonly_structural_validation(ledger_data: dict) -> "list[str]":
    """Read-only structural validation for content_owned_readonly lanes
    (r2 P0.1): generic authorship verification + speaker-to-cast validation,
    WITHOUT any of the legacy repair mutations. Returns error strings;
    any error is terminal for the freeze."""
    errors: "list[str]" = []
    meta = ledger_data.get("meta") or {}
    lines = [ln for ln in (ledger_data.get("lines") or [])
             if isinstance(ln, dict)]
    lines_by_id = {str(ln.get("line_id") or ""): ln for ln in lines}
    # 1. Generic authorship verification: exact voiced-line coverage and
    #    raw UTF-8 hashes from the accepted final artifact.
    try:
        from ._otr_content_authorship import validate_receipt
        validate_receipt(ledger_data)
    except Exception as exc:  # fail closed; caller promotes errors to terminal
        errors.append(f"content_authorship: {exc}")
    # 2. Read-only speaker-to-cast validation (the D3 sweep's CHECK
    #    without its mutation): a cast char_id must carry role
    #    "character". A violation means an unaccounted mutator ran.
    try:
        from . import production_ledger as _PL  # type: ignore
        cast_ids = _PL.cast_ids_from_ledger(ledger_data)
    except Exception:  # noqa: BLE001
        cast_ids = set()
    for ln in lines:
        cid = str(ln.get("char_id") or "").strip()
        role = str(ln.get("speaker_role") or "")
        if cid in cast_ids and role != "character":
            errors.append(
                f"line {ln.get('line_id')!r}: cast char_id {cid!r} has "
                f"speaker_role {role!r} (must be 'character')")
    return errors


def _stamp_capability_receipt(
    led,
    policy: "FreezePolicy",
    *,
    entry_text_sha: str,
    entry_authorship_sha: str,
    skipped_phases,
    executed_phases,
    structural_errors=(),
) -> dict:
    """meta.freeze_capability_receipt (r2 P0.1): policy provenance,
    skipped/executed phases, canonical-text + authorship hashes before
    and after the cascade, and the content-mutation count that makes
    proof invariance testable. Stamped at EVERY cascade exit."""
    data = led.data
    meta = data.setdefault("meta", {})
    exit_text_sha = _sha256_lines_text(data)
    exit_authorship_sha = _sha256_content_authorship(data)
    receipt = {
        "policy": policy.name,
        "policy_source": policy.source,
        "skipped_phases": list(skipped_phases),
        "executed_phases": list(executed_phases),
        "text_sha256_entry": entry_text_sha,
        "text_sha256_exit": exit_text_sha,
        "content_authorship_sha256_entry": entry_authorship_sha,
        "content_authorship_sha256_exit": exit_authorship_sha,
        "content_mutations": 0 if (
            exit_text_sha == entry_text_sha
            and exit_authorship_sha == entry_authorship_sha
        ) else 1,
        "structural_errors": [str(e) for e in structural_errors],
        "terminal_error": policy.terminal_error,
    }
    meta["freeze_capability_receipt"] = receipt
    return receipt


# Inline banks may perform only this content mutation at freeze.
_INLINE_CLEANUP_PHASES = (
    "same_story_safety_cleanup",
    "d3_role_sweep_mutation",
)


def _hash_lines_text(ledger_data: dict) -> int:
    """Cheap fingerprint of line.text values for idempotency checks.

    The full §6.7 idempotency feature lands in commit 12; here we just
    stamp the entry/exit hash on each phase record so the soak diag
    has the data for free. `hash()` salted process-locally is fine for
    in-process forensic logs.
    """
    lines = ledger_data.get("lines") or []
    return hash(tuple((ln or {}).get("text", "") for ln in lines))


# Phase records are grouped by purpose for diagnostics. Only the shared safety
# cleanup can mutate accepted text; readiness and audits are deterministic.
_PHASE_BUCKETS: dict[str, str] = {
    "phase_0_gap_audit_pre":              "audit_passes",
    "phase_10_gap_audit_post_and_freeze": "audit_passes",
    "same_story_safety_cleanup":           "cleanup_passes",
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
        "phase":    "same_story_safety_cleanup",  # phase_name
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
    """Persist cascade receipts at every exit; never raise."""
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
# Terminal-skip disposition (structural reviewer/capability exits only)
# ---------------------------------------------------------------------------


def _build_terminal_skip_disposition(
    ledger_data: dict,
    *,
    verdict: str,
    cleanup_receipt: Optional[dict[str, Any]],
    pre_report: _LFC.GapAuditReport,
    block_class: str = "structural",
    remaining_phase_names: tuple[str, ...] = (
        "phase_7_audio_readiness",
        "phase_8_video_readiness",
        "phase_10_gap_audit_post_and_freeze",
    ),
) -> FreezeDisposition:
    """Stamp a terminal structural or residual-safety disposition."""
    meta = ledger_data.setdefault("meta", {})
    meta["freeze_verdict"] = verdict
    # BUG-LOCAL-300 retains the explicit class for downstream compatibility.
    # Current callers are structural-only: renderable craft defects are repaired
    # and never reach a terminal disposition.
    meta["freeze_block_class"] = block_class
    # B5: stamp the remaining phases as terminal_skipped so
    # meta.cleanup_passes / readiness_passes stay contiguous. S30 B4:
    # phase 4 / 4.5 / 5 / 6 names removed -- those phases no longer exist.
    for skipped_name in remaining_phase_names:
        _stamp_stub_or_skipped_phase(
            ledger_data,
            phase_name=skipped_name,
            reason="terminal_skipped",
        )
    # Stamp the skipped-phase status on their respective meta keys so
    # downstream readers see a consistent shape.
    if "phase_7_audio_readiness" in remaining_phase_names:
        meta.setdefault("audio_readiness", {
            "skipped": True, "skipped_reason": "terminal_skipped",
        })
    if "phase_8_video_readiness" in remaining_phase_names:
        meta.setdefault("video_readiness", {
            "skipped": True, "skipped_reason": "terminal_skipped",
        })
    disp = FreezeDisposition(
        verdict=verdict,
        cleanup_receipt=cleanup_receipt,
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
# Inline same-story cleanup
# ---------------------------------------------------------------------------


def _run_inline_safety_cleanup(generate_fn, led) -> dict[str, Any]:
    """RETIRED 2026-08-05. Stamps the phase; edits nothing.

    This used to run one atomic safety-only patch set over the accepted inline
    story -- rewriting a delivered spoken row whose words matched the
    profanity / weapon / sexual list. It is gone by operator directive: no
    content guardrails on generated episodes, and the inline lanes get the same
    treatment as the adaptation lanes.

    The PHASE is not deleted along with the pass. ``same_story_safety_cleanup``
    is a declared cascade phase with a registry entry, a telemetry group and a
    ledger field that two fable2 artifact tests assert on, so it keeps
    stamping -- with a retired status and zero edits -- exactly the way the
    producer-owned branch already stamps its not-applicable receipt. A ripped
    pass may not leave an unowned field.
    """
    del generate_fn
    ledger_data = led.data
    meta = ledger_data.setdefault("meta", {})
    receipt = {
        "status": "retired_no_content_policy",
        "reason": "content_guardrails_removed_by_operator_directive",
        "patch_count": 0,
    }
    meta["same_story_safety_cleanup"] = dict(receipt)
    _stamp_stub_or_skipped_phase(
        ledger_data,
        phase_name="same_story_safety_cleanup",
        reason="retired_no_content_policy",
    )
    return dict(receipt)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_freeze_cascade(
    generate_fn,
    led,
    *,
    enable_phase_7_audio_readiness: bool = True,
    enable_phase_8_video_readiness: bool = True,
) -> FreezeDisposition:
    """Validate, optionally clean narrow safety, then freeze the ledger.

    Phase 0 records a deterministic preflight. Inline banks receive at most one
    atomic safety-only patch set; producer-owned banks are verified read-only.
    Phase 7/8 readiness and Phase 10 structural/safety gates then run. Advisory
    word, visual, style, craft, and quality observations never affect liveness.
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

    # ---- Freeze policy ---------------------------------------------
    # Inline banks may run one atomic, same-story safety cleanup. Content-owned
    # banks perform that cleanup before sealing their authorship receipt, so the
    # cascade validates them read-only. A tagged bank that cannot resolve is a
    # structural configuration error; it never falls through to another lane.
    policy = resolve_freeze_policy(meta)
    meta["freeze_policy"] = {"name": policy.name, "source": policy.source}
    _cap_entry_text_sha = _sha256_lines_text(ledger_data)
    _cap_entry_authorship_sha = _sha256_content_authorship(ledger_data)
    if policy.terminal_error:
        log.error("[LFC] %s", policy.terminal_error)
        disp = _build_terminal_skip_disposition(
            ledger_data,
            verdict="needs_full_rerun",
            cleanup_receipt=None,
            pre_report=pre_report,
            block_class="structural",
        )
        _stamp_capability_receipt(
            led, policy,
            entry_text_sha=_cap_entry_text_sha,
            entry_authorship_sha=_cap_entry_authorship_sha,
            skipped_phases=_INLINE_CLEANUP_PHASES,
            executed_phases=("phase_0_gap_audit_pre",),
            structural_errors=(policy.terminal_error,),
        )
        _persist_cascade_meta(led)
        return disp

    cleanup_receipt: Optional[dict[str, Any]] = None
    if policy.run_inline_safety_cleanup:
        cleanup_receipt = _run_inline_safety_cleanup(generate_fn, led)
    else:
        ledger_data = led.data
        meta = ledger_data.setdefault("meta", {})
        meta["same_story_safety_cleanup"] = {
            "status": "not_applicable_content_owned",
            "reason": "producer_cleanup_precedes_authorship_seal",
            "patch_count": 0,
        }
        _stamp_stub_or_skipped_phase(
            ledger_data,
            phase_name="same_story_safety_cleanup",
            reason="not_applicable_content_owned",
        )
        readonly_started = _isoformat_utc_now()
        readonly_hash = _hash_lines_text(ledger_data)
        readonly_errors = _readonly_structural_validation(ledger_data)
        _stamp_phase_record(
            ledger_data,
            phase_name="readonly_structural_validation",
            text_hash_before=readonly_hash,
            text_hash_after=_hash_lines_text(ledger_data),
            started_at=readonly_started,
            finished_at=_isoformat_utc_now(),
            failures=[
                {"line_id": "__readonly__", "reason": error}
                for error in readonly_errors
            ],
        )
        if readonly_errors:
            log.error(
                "[LFC] read-only structural validation failed under %s: %s",
                policy.name,
                "; ".join(readonly_errors)[:400],
            )
            disp = _build_terminal_skip_disposition(
                ledger_data,
                verdict="needs_full_rerun",
                cleanup_receipt=None,
                pre_report=pre_report,
                block_class="structural",
            )
            _stamp_capability_receipt(
                led, policy,
                entry_text_sha=_cap_entry_text_sha,
                entry_authorship_sha=_cap_entry_authorship_sha,
                skipped_phases=_INLINE_CLEANUP_PHASES,
                executed_phases=(
                    "phase_0_gap_audit_pre",
                    "readonly_structural_validation",
                ),
                structural_errors=readonly_errors,
            )
            _persist_cascade_meta(led)
            return disp
        log.info(
            "[LFC] policy %s: content-owned ledger is structurally clean",
            policy.name,
        )

    _cap_skipped = (
        () if policy.run_inline_safety_cleanup else _INLINE_CLEANUP_PHASES
    )
    _cap_executed_tail = (
        (
            "phase_0_gap_audit_pre",
            "same_story_safety_cleanup",
            "d3_role_sweep_mutation",
            "phase_7_audio_readiness",
            "word_delivery_telemetry",
            "phase_8_video_readiness",
            "phase_10_gap_audit_post_and_freeze",
        )
        if policy.run_inline_safety_cleanup
        else (
            "phase_0_gap_audit_pre",
            "readonly_structural_validation",
            "word_delivery_telemetry",
            "phase_8_video_readiness",
            "phase_10_gap_audit_post_and_freeze",
        )
    )

    # ---- D3: mandatory pre-freeze role sweep
    # The FINAL mutation step before the freeze hash + role-dependent routing
    # (Phase 7 audio readiness / scrub / TTS). A prior role mismatch
    # repair can leave a cast character stamped speaker_role="announcer" (b011
    # "Chandra's Echo": char_id=c02, role=announcer). Force speaker_role=
    # "character" on every line whose char_id is a real cast id. Audit rides
    # meta["role_coercions"] + per-line compose_flags. COERCE-NEVER-CRASH (any
    # failure leaves the ledger untouched -- the freeze never breaks, PD1).
    from . import production_ledger as _PL  # type: ignore
    try:
        ledger_data = led.data
        meta = ledger_data.setdefault("meta", {})
        _d3_cast_ids = _PL.cast_ids_from_ledger(ledger_data)
        _d3_coerced: list[str] = []
        # r2 P0.1: the D3 MUTATION is a legacy content pass -- under a
        # content_owned_readonly policy the sweep must not write (the
        # equivalent CHECK already ran read-only inside
        # _readonly_structural_validation, where a violation is
        # terminal, not silently repaired).
        if _d3_cast_ids and policy.run_inline_safety_cleanup:
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

    # ---- Non-terminal path: Phase 7 / 8 / 10 ---------------------\n\n    # Phase 7 stamps a pronunciation-only projection and never mutates
    # canonical text. Content-owned producers already stamp their projection,
    # so the cascade keeps their policy skip while preserving the same public
    # readiness surface for inline banks.
    started_7 = _isoformat_utc_now()
    hash_before_7 = _hash_lines_text(ledger_data)
    _p7_enabled = bool(
        enable_phase_7_audio_readiness and policy.run_inline_safety_cleanup
    )
    _phase_7_audio_readiness(led, enable=_p7_enabled)
    if enable_phase_7_audio_readiness and not _p7_enabled:
        led.data.setdefault("meta", {})["audio_readiness"] = {
            "skipped": True,
            "skipped_reason": "producer_owned_delivery_stamp",
        }
    hash_after_7 = _hash_lines_text(ledger_data)
    _stamp_phase_record(
        ledger_data,
        phase_name="phase_7_audio_readiness",
        text_hash_before=hash_before_7,
        text_hash_after=hash_after_7,
        started_at=started_7,
        finished_at=_isoformat_utc_now(),
        edits_applied=0,
    )

    # PBUG-20260721-15: one derived-metric owner at the final text boundary.
    # Phase 0 intentionally preserves the incoming producer diagnosis. After
    # every permitted text mutator has finished, refresh row/root/meta counts
    # once before Phase 10. This is count-only and therefore preserves sealed
    # canonical text, text_for_tts, and content-authorship hashes.
    _PL.refresh_ledger_text_metrics(led)
    ledger_data = led.data
    meta = ledger_data.setdefault("meta", {})

    # Final word observations are stamped after every permitted text mutator.
    # Target metadata may be missing, invalid, under, or over; none of those
    # states can change freeze/media/publication liveness.
    _declared_word_budget = meta.get("word_budget")
    _declared_word_owner = str(
        _declared_word_budget.get("owner")
        if isinstance(_declared_word_budget, dict) else ""
    ).strip()
    _freeze_word_receipt = _OTRWD.stamp_actual(
        ledger_data,
        stage="freeze_pre_media",
    )
    meta["word_delivery_telemetry"] = {
        "status": "telemetry_only",
        "owner": _declared_word_owner,
        **_freeze_word_receipt,
    }


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

    # r2 P0.1 invariant: under a readonly policy NOTHING may have
    # mutated canonical text or the authorship receipt between cascade entry and
    # the freeze gate. A divergence means an unaccounted writer ran --
    # terminal structural error, fail loud, never freeze.
    if not policy.run_inline_safety_cleanup:
        _ro_exit_text = _sha256_lines_text(led.data)
        _ro_exit_authorship = _sha256_content_authorship(led.data)
        if (_ro_exit_text != _cap_entry_text_sha
                or _ro_exit_authorship != _cap_entry_authorship_sha):
            _mut_err = (
                "content mutated under content_owned_readonly policy "
                "(canonical text or authorship hash diverged between "
                "cascade entry and the freeze gate)"
            )
            log.error("[LFC] %s", _mut_err)
            disp = _build_terminal_skip_disposition(
                led.data,
                verdict="needs_full_rerun",
                cleanup_receipt=None,
                pre_report=pre_report,
                block_class="structural",
            )
            _stamp_capability_receipt(
                led, policy,
                entry_text_sha=_cap_entry_text_sha,
                entry_authorship_sha=_cap_entry_authorship_sha,
                skipped_phases=_cap_skipped,
                executed_phases=_cap_executed_tail,
                structural_errors=(_mut_err,),
            )
            _persist_cascade_meta(led)
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
        # G9 (terminal spoken-safety) was deleted 2026-08-05 by operator
        # directive, so a freeze can no longer be blocked on content -- the
        # "safety" class became unreachable and the branch that computed it is
        # gone rather than left as dead code. The FIELD stays, with exactly one
        # owner and a defined value on every path, because soak telemetry and
        # the disposition readers expect it.
        meta["freeze_block_class"] = "structural"
        disp = FreezeDisposition(
            verdict="needs_full_rerun",
            cleanup_receipt=cleanup_receipt,
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
        _stamp_capability_receipt(
            led, policy,
            entry_text_sha=_cap_entry_text_sha,
            entry_authorship_sha=_cap_entry_authorship_sha,
            skipped_phases=_cap_skipped,
            executed_phases=_cap_executed_tail,
            structural_errors=list(exc.errors),
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
    meta = ledger_data["meta"]
    final_verdict = meta.get("freeze_verdict", "frozen_clean")
    warn_taxonomy = build_freeze_warn_taxonomy(
        gap_warnings=list(getattr(post_report, "warnings", None) or []),
        consistency_status=meta.get("consistency_status"),
    )
    meta["freeze_warn_taxonomy"] = warn_taxonomy
    if warn_taxonomy["story_accuracy_warning"]:
        meta["freeze_story_accuracy_warnings"] = (
            warn_taxonomy["story_accuracy_warning"]
        )
    if final_verdict == "frozen_clean" and (
        warn_taxonomy["story_accuracy_warning"]
        or warn_taxonomy["cosmetic_warning"]
    ):
        final_verdict = "frozen_with_warns"
        meta["freeze_verdict"] = final_verdict
    disp = FreezeDisposition(
        verdict=final_verdict,
        cleanup_receipt=cleanup_receipt,
        gap_audit_pre=pre_report,
        gap_audit_post=post_report,
    )
    meta["freeze_disposition"] = disp.to_dict()
    # C3 (clean-break 2026-05-12): compact per-phase telemetry
    # for soak diagnostics. Lives on meta only -- the
    # freeze_verdict output STRING stays the verdict literal.
    meta["freeze_phase_telemetry"] = build_phase_telemetry(meta)
    log.info(
        "[LFC] freeze landed: verdict=%s cleanup=%s pre_warns=%d "
        "post_warns=%d",
        final_verdict,
        (cleanup_receipt or {}).get("status", "not_applicable"),
        len(pre_report.warnings),
        len(post_report.warnings),
    )
    _stamp_capability_receipt(
        led, policy,
        entry_text_sha=_cap_entry_text_sha,
        entry_authorship_sha=_cap_entry_authorship_sha,
        skipped_phases=_cap_skipped,
        executed_phases=_cap_executed_tail,
    )
    # BUG-LOCAL-278: persist cascade meta on the successful-freeze exit.
    _persist_cascade_meta(led)
    return disp
