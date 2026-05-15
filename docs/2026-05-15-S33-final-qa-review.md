# S33 — Final QA Review (Editor-Only Cleanup)

> **Status:** CLOSED 2026-05-15.
> **Branch:** `s33-editor-only-cleanup` @ B6 (pending push).
> **Parent:** `s32-helper-per-subpass-routing` @ B8 (`3261b18`, S32 close).
> **Plan:** `docs/2026-05-14-S33-editor-only-cleanup-sprint-plan.md`.
> **Refined rule applied:** "Audit calls are OK if they USE the audit to develop / edit the story. NOT OK if they just cut the pipeline (gate, halt, fail, rollback, report-only)." (Jeffrey, 2026-05-15).

---

## Summary

S33 is the third sprint in the dual-LLM cleanup arc (S31 → S31.5 → S32 → S33). Subtractive: net codebase shrinks. Eight commits on `s33-editor-only-cleanup` apply the refined no-auditors rule to the cascade reviewer + polish prompt design.

Two cascade rollback gates retired (`speaker_unknowns` driving `cast_unrecoverable`; `post_audit_pass` driving `post_audit_failed`). One LLM call retired (Phase 9, `audit_cast_contract(label="post")` -- no editor consumer post-B2). Two phantom-handler functions retired per B1.5 classification (`apply_phantom_skip_fallback`, `_final_phantom_check`). One polish-prompt constant renamed for symmetric naming (`_POLISH_SYSTEM_PROMPT` → `_POLISH_SYSTEM_PROMPT_CHARACTER`) with a design-lock comment block and behavior tests locking the two-prompt split.

`auto_remap_phantom` SURVIVES (helper to an editor, classified KEEP at B1.5). Phase 1 (`audit_cast_contract(label="pre")`) SURVIVES (its output feeds `apply_deterministic_cast_repairs`, an editor).

Architectural change at B1: the plan's "delete cascade Phase 1/9 auditors" framing was mismatched with the code (no Phase 1/9 methods on the cascade class; the two LLM calls live inside `_otr_ledger_reviewer.review_ledger` as label-distinguished invocations of the shared `audit_cast_contract` function, structurally coupled per ADR D12). Sprint halted at B1, surfaced to Jeffrey, resumed with the refined no-auditors rule. The structural pre-grep coverage from B1 (machine-checkable inventory across 6 dimensions) carried into B2-B5 deletions.

---

## Commit table

| # | Commit | Subject | Hash |
|---|---|---|---|
| 1 | B0 | branch cut + S33 editor-only cleanup plan landing (round-robin integrated) | `4b296a2` |
| 2 | B1 | cascade Phase 1 + Phase 9 inventory (machine-checkable table, downstream consumer + workflow JSON sweep) | `8afec7b` |
| 3 | B1.5 | phantom handler classification per refined rule (KEEP if edits, DELETE if cuts) | `7ab5d1a` |
| 4 | B2 | delete rollback gates (refined-rule application -- gates cut pipeline, not edit story) | `c559a1f` |
| 5 | B3 | delete Phase 9 LLM auditor (no editor consumer post-B2) | `7e3748e` |
| 6 | B4 | delete pipeline-cutting phantom handlers per B1.5 classification | `2b9e8c4` |
| 7 | B5 | polish prompt rename + design lock -- _POLISH_SYSTEM_PROMPT -> _CHARACTER, behavior tests prove semantic differentiation | `a2f9c7b` |
| 8 | B6 | Sprint S33 close -- refined no-auditors rule applied, pipeline-cut gates retired, polish design locked | (this commit) |

---

## Acceptance table (vs plan, post-refined-rule reconciliation)

The original plan's 23-row acceptance table was framed around the pre-refinement mental model. After Jeffrey's refined rule + B1.5 classification, the acceptance shape changed. The revised acceptance table below reflects what S33 actually targets and ships:

| # | Check | Target | Result |
|--:|---|---|---|
| 1 | Canonical pytest count (B5 affected suites) | green | 97/1 skipped (98 tests) |
| 2 | Cascade affected suites (B4) | green | 138/1 skipped (139 tests) |
| 3 | Bug Bible regression | 23 / 1 / 2 | held at every commit boundary |
| 4 | Audio C7 byte-identical (pytest proxy, default config) | holds B2 → B5 | held all four boundaries |
| 5 | Forbidden sweep | 0 runtime hits | 0 hits; 6 new markers integrated |
| 6 | `speaker_unknowns` rollback gate DELETED | ✅ | gone from `review_ledger` |
| 7 | `post_audit_pass` rollback gate DELETED | ✅ | gone from `review_ledger` |
| 8 | `cast_unrecoverable` verdict literal DELETED | ✅ | not in `ReviewerVerdict` Literal / mapping / terminal-set |
| 9 | `post_audit_failed` verdict literal DELETED | ✅ | not in `ReviewerVerdict` Literal / mapping / terminal-set |
| 10 | `audit_cast_contract(label="post")` call DELETED | ✅ | AST walk: exactly 1 call site in `review_ledger` (Phase 1 only) |
| 11 | `apply_phantom_skip_fallback` function DELETED | ✅ | not on module, not in `__all__` |
| 12 | `_final_phantom_check` function DELETED | ✅ | not on module |
| 13 | `phantom_skip_count` field DELETED from `ReviewerDisposition` | ✅ | dataclass has 6 fields (was 7) |
| 14 | `auto_remap_phantom` SURVIVES | ✅ | still on module, still in `__all__`, positive test |
| 15 | `_POLISH_SYSTEM_PROMPT_CHARACTER` exists (renamed from `_POLISH_SYSTEM_PROMPT`) | ✅ | module-level constant |
| 16 | `_POLISH_SYSTEM_PROMPT_ANNOUNCER` exists | ✅ | untouched (existing constant) |
| 17 | Polish prompts text contents non-identical | ✅ | behavior test |
| 18 | Character prompt forbids narration (behavior test) | ✅ | "narration of any kind" check |
| 19 | Announcer prompt allows third-person narration (behavior test) | ✅ | "third-person narration is OK" check |
| 20 | `polish_line()` dispatches different prompts for different `speaker_role` | ✅ | runtime capture test |
| 21 | `_POLISH_SYSTEM_PROMPT_UNIFIED` does NOT exist | ✅ | parametrized hasattr check |
| 22 | `_UNIFIED_POLISH_PROMPT` does NOT exist | ✅ | parametrized hasattr check |
| 23 | Design-lock comment block present in `_otr_line_composer.py` | ✅ | source-file string check |
| 24 | New S33 forbidden-sweep markers | 6 (2 polish unify names + 2 B2 verdicts + 2 B4 phantom funcs) | 6 markers landed |
| 25 | Workflow JSON references to retired symbols | 0 hits | 0 hits across all `workflows/*.json` |
| 26 | Tree-wide string-ref sweep for retired symbols (`nodes/`, `tests/`) | 0 live hits | callout exclusion via +/-12-line window; 0 live hits |
| 27 | ROADMAP refreshed | S33 marked closed | ✅ (this commit) |

(The original-plan B3 row "Phase 2 hard-fails malformed Script Doctor output without Phase 9" is REMOVED from the acceptance table: per Jeffrey's phantom-ship policy, no Phase 2 hardening was required.)

---

## Deviations from plan

Five deviations, all reconciled via the refined rule + Jeffrey's phantom-ship policy:

1. **B1 plan-vs-code mismatch (major).** Plan assumed cascade-class methods + widgets for Phase 1/9. Reality: those don't exist (S30 B3 already deleted the widgets; the LLM calls live inside `review_ledger`). B1 inventory surfaced this; halted; resumed with refined rule.
2. **B3 plan flip (major).** Original plan B3 was "prove Phase 2 hard-fails malformed Script Doctor output" as a test-before-fix gate for Phase 9 deletion. Refined plan's phantom-ship policy retired the need for this proof: the rollback gates are gone whether or not Phase 2 catches every defect. New B3 became "delete Phase 9 LLM call".
3. **B4 scope shift (medium).** Original plan B4 was "delete Phase 9 post-edit auditor". Refined plan's B4 became "delete pipeline-cutting phantom handlers per B1.5 classification" (because B3 absorbed the Phase 9 LLM call deletion).
4. **New B1.5 commit (medium).** Inserted between B1 and B2 to formalize the phantom-handler classification per the refined rule. Outcomes: 1 KEEP, 2 DELETE.
5. **`phase_1_2_9_reviewer_composite` phase_name string retained (minor).** The string literally names the deleted Phase 9 alongside the surviving Phase 1+2. Drift policy: rename is adjacent (not strictly required for any deletion), forensic continuity wins. Defer.

---

## Gate run details

All pytest commands used the Windows venv (`C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe`), background-detached via `start /MIN`, output to a log file, then read back.

| Boundary | Suites run | Pass / Skip / xfail | Notes |
|---|---|---|---|
| B2 | rollback-gate proof + 4 LFC suites | 83 / 0 / 0 | Audio C7 holds |
| B2 (gates) | forbidden sweep + 4 other guardrail suites | 72 / 5 / 0 | Bug Bible 23/1/2xf |
| B3 | B3 proof + B2 carryover + 4 LFC suites + audio | 99 / 1 / 0 | Audio C7 holds |
| B4 | B4 proof + B3/B2 carryover + 5 LFC suites + audio | 138 / 1 / 0 | Audio C7 holds |
| B4 (gates) | forbidden + naming + workflow JSON + phase extinction + Bug Bible | 95 / 6 / 2 | Bug Bible 23/1/2xf |
| B5 | B5 design lock + polish fixes + B4/B3/B2 carryover + audio + forbidden | 97 / 1 / 0 | Audio C7 holds |

---

## Reviewer surface after S33

`nodes/_otr_ledger_reviewer.py`:

* Module docstring updated: Pass 3 + Step 2.5 marked RETIRED.
* `ReviewerVerdict` Literal: 4 entries (`clean_no_edits`, `improved`, `too_many_edits`, `needs_full_rerun`).
* `ReviewerDisposition` dataclass: 6 fields (was 7).
* `__all__`: `apply_phantom_skip_fallback` removed.
* `audit_cast_contract` function: `label: str = "pre"` default; only call site is Phase 1.
* `review_ledger` body: Pass 1 audit (LLM) → speaker_unknowns gate DELETED → deterministic cast repairs (editor) → Pass 2 doctor (LLM) → too_many_edits gate (retained, ordinary edit-cap exceed) → apply_doctor_edits → commit. No Step 2.5; no Phase 9 LLM call; no `post_audit_pass` rollback.
* `apply_phantom_skip_fallback` + `_final_phantom_check` functions: DELETED.

`nodes/_otr_freeze_cascade.py`:

* ADR diagram updated: Phase 9 row marked RETIRED.
* `REVIEWER_TO_FREEZE_VERDICT` map: 4 entries.
* `FREEZE_TERMINAL_FAILURE_VERDICTS` frozenset: 2 entries.
* `phase_1_2_9_reviewer_composite` phase_name string: RETAINED for forensic continuity.
* `_PHASE_BUCKETS` table comment updated.

`nodes/OTR_LedgerFreezeCascade.py`:

* `freeze_verdict` literal-set docstring: 5 entries (was 7).
* `model_id` docstring trimmed: Phase 9 removed from the consumer list.

`nodes/_otr_line_composer.py`:

* `_POLISH_SYSTEM_PROMPT_CHARACTER` (renamed from `_POLISH_SYSTEM_PROMPT`).
* Design-lock comment block (14 lines) immediately before the constant.
* `polish_line()` dispatch references the new name.
* `_POLISH_SYSTEM_PROMPT_ANNOUNCER` unchanged.

`docs/_s28_forbidden_sweep.py`:

* 6 new S33 markers (2 polish unify names + 2 B2 verdicts + 2 B4 phantom funcs).

---

## VRAM / runtime impact (informational)

Per cascade run, the reviewer now invokes 2 LLM calls instead of 3 (Phase 1 audit + Phase 2 doctor; Phase 9 retired in B3). Same `technical_model` slot, same cache; no model unload / reload. Net saving: ~1500 audit-temperature tokens per episode-cleanup pass.

No VRAM ceiling change. No Bark / HuMo / LTX / FLUX interaction.

---

## Forward work (out of scope, surfaced during S33)

* **`phase_1_2_9_reviewer_composite` rename.** Misnamed after B3 retires the "9" component. Cosmetic. Deferred per drift policy.
* **`post_audit_violations` ReviewerDisposition field removal.** Always 0 after B2; carrying it is dead-state-shape debt. Adjacent cleanup; deferred.
* **Sprint E enhancer chain audit** (`arc_enhancer` / `self_critique` / `target_length` revival decisions). Separate sprint, queued post-Sprint C per the original plan.
* **Audio-intentional sprint** (model-author `generation_config.json` respect for polish). Queued.
* **`OTR_LedgerScriptWriter.py` Phase 3 + Step 2.5 comment refs.** Two docstrings + two comments in the writer node reference the retired terms. Cosmetic; deferred.
* **`_otr_ledger_consumers.py:87` "set by Step 2.5" comment.** Cosmetic; deferred.

---

## BUG_LOG entries filed during S33

None. The S33 plan-vs-code mismatch was an architectural surface, not a defect in shipped code. The full architectural finding lives in `docs/2026-05-15-S33-B1-cascade-auditor-inventory.md`.

---

## Sources

* `docs/2026-05-14-S33-editor-only-cleanup-sprint-plan.md` (sprint plan)
* `docs/2026-05-15-S33-B1-cascade-auditor-inventory.md` (B1 inventory)
* `docs/2026-05-15-S33-B1p5-phantom-handler-classification.md` (B1.5 classification)
* `docs/script-writing-architecture-adr.md` (D12 reviewer-function decision)
* `docs/2026-05-14-S32-final-qa-review.md` (parent sprint close)
* `docs/2026-05-14-S31p5-final-qa-review.md` (subtractive sprint format reference)
* `nodes/_otr_ledger_reviewer.py`, `nodes/_otr_freeze_cascade.py`, `nodes/OTR_LedgerFreezeCascade.py`, `nodes/_otr_line_composer.py` (source of truth for the deletions + rename)
* `tests/test_no_rollback_gates_b2.py`, `tests/test_no_phase_9_call_b3.py`, `tests/test_no_phantom_handlers_b4.py`, `tests/test_polish_speaker_prompts_locked.py` (new S33 tests)
* `ROADMAP.md` (S33 CURRENT WORK section)
