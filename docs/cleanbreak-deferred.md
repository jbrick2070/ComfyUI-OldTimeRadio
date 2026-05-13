# Voice-Path-Cleanbreak — Deferred Items

Tracking page for sprint tasks that have shipped enough mitigation
to lock the batch but remain open. One entry per deferral with the
reason, the mitigation already in place, and the unblock condition.

## C10 — LFC audit regex extension (DEFERRED, 2026-05-13)

**Status:** deferred. The plan's premise was wrong.

**Reason:** the C10 plan (IMP-37) extended the legacy-audit regex with `\bLFC\b`, `\bLive Freeze Cascade\b`, and `\blfc_` prefix tokens — the assumption being that LFC was a deleted legacy generation needing forensic-marker discipline (like Director). Pre-edit dry-run grep found 159 hits across the repo, and inspection of the top hit-files surfaces the actual situation:

- `OTR_LFCPhase4Scene`, `OTR_LFCPhase5Voice`, `OTR_LFCPhase6Arc` are **registered ComfyUI nodes** in `__init__.py`.
- `nodes/_otr_lfc_phase_4_scene_coherence.py`, `_otr_lfc_phase_5_voice_drift.py`, `_otr_lfc_phase_6_episode_arc.py`, `_otr_lfc_context.py`, `_otr_lfc_llm_helpers.py`, `_otr_lfc_phase_verdicts.py`, `_otr_lfc_smart_suggestion.py`, `_otr_lfc_watchdog.py` are **all live infrastructure**.
- `nodes/OTR_LedgerFreezeCascade.py` (the FreezeCascade itself) names "LFC" in its forensic comments as part of its own naming history.

LFC is the **current** Live Freeze Cascade architecture, not a deleted lineage. Adding LFC tokens to the audit regex would flag every legitimate reference (159 of them) as a violation. The audit's contract is to catch *deleted* surfaces that survived a cleanbreak — LFC doesn't qualify.

**Mitigation already in place:** none specific. The current audit (S15.5.1) catches Director-era surfaces; legacy LFC variants (if any exist) would surface there only if they were ALSO Director-era. None have been observed.

**Unblock condition:** if a future sprint retires the LFC system (replaces LFCPhase4/5/6 with a different cascade architecture), revisit the regex extension. The audit token to add at that point is whatever's deleted, not the generic "LFC" name.

**Plan correction:** the IMP-37 framing of "LFC tokens are legacy" in the S24 plan was an incorrect assumption. Future plans should distinguish between (a) names of currently-live systems that happen to be acronyms (LFC, FLUX, HuMo, LTX -- leave alone), and (b) names of deleted systems (Director, LLMDirector, production_plan_json -- audit).

---

## C8 — CastContract quarantine (DEFERRED, 2026-05-13)

**Status:** deferred. The plan's premise was wrong.

**Reason:** the C8 plan said "Update any internal imports (likely none)" when scoping the move from `nodes/_otr_cast_contract.py` to `nodes/experimental/_otr_cast_contract.py`. The dependency audit at execution time found:

- `nodes/_otr_cast_repair.py:40,312` imports from `_otr_cast_contract` (CharacterEntry, _extract_dialogue_tags, others)
- `nodes/_otr_cast_repair.py` is consumed by `nodes/_otr_ledger_reviewer.py::apply_deterministic_cast_repairs` (live production code path)
- `nodes/_otr_ledger.py:897` + `nodes/_otr_line_composer.py:740` carry forensic references

Cast contract IS wired into production via the `cast_repair → ledger_reviewer` chain. Quarantining to `experimental/` without first untangling those imports would either break `apply_deterministic_cast_repairs` (which IS called at writer-time) or ship a docstring lie ("not wired into production" when it IS).

**Mitigation already in place:** none specific to this. The repair path holds invariants via its own tests (`tests/test_cast_repair.py`, `tests/test_phase3_ledger_reviewer.py`).

**Unblock condition:** one of:
1. Delete the `cast_repair → cast_contract` dependency (move the small helpers cast_repair needs into a separate `_otr_cast_helpers.py`, then quarantine cast_contract clean).
2. Quarantine the full chain `cast_contract + cast_repair + apply_deterministic_cast_repairs` together (large scope; needs a real design call).
3. Accept that cast contract is production-wired and drop the quarantine plan. Update the C8 plan-spec docstring to reflect this.

This is a real architectural call, not a mechanical move. Plan as its own sprint.

---

## S14.2 — Validator auto-invoke (DEFERRED — implementation scheduled for S25+; ADR at docs/2026-05-13-S14_2-active-validation-ADR.md)

**Status:** DEFERRED (was: INDEFINITELY DEFERRED until 2026-05-13).
**Decision locked:** Option B — opt-in `OTR_WorkflowValidator` first-node. See ADR for the full rationale + alternatives + consequences.
**Reason for original deferral:** ComfyUI has no central Python-side workflow loader to wrap. The frontend parses JSON in JavaScript and dispatches per-node; there is no single chokepoint for `validate_workflow_contract()`.

**Mitigation already in place:**
- `tests/test_workflow_live_passes_validator.py` (S16.6) validates the production workflow JSON in CI.
- `tests/test_legacy_audit_clean.py` (S15.5.1) catches legacy Director-era surfaces repo-wide.

**Implementation path (S25+):** new node class `OTR_WorkflowValidator` in `nodes/_otr_workflow_validator.py` (~150 LOC) wired as position-0 in the production workflow JSON. Tests cover the canonical workflow + an adversarial broken workflow fixture. ADR section "Status" lists the estimated scope.
