# OTR QA Review Handoff — 2026-05-13

**Branch:** `v2.0-alpha` at `7012eb4` (pushed; local == origin).
**Audience:** QA review team. **Goal:** verify the S24 fix sprint wiring shipped correctly, audit the deferrals, vote on the next-sprint plan.

> **2026-05-13 update -- S25 LANDED.** The next-sprint plan in §3 was executed under the playbook handed to Cowork. Branch `s25-musicgen-parity` at `a78e282` (pushed; local == origin). See **`docs/2026-05-13-S25-qa-postmortem.md`** for the single-doc summary of what shipped, the three inline CD-1/CD-2/CD-3 decisions, the +18 test delta, and carry-forward to S26. This original handoff stays as the historical S24 review surface.

This doc is the single entry-point. Use the orientation map below to navigate to deeper docs without re-reading what you don't need.

---

## 0. Orientation map — read this first (2 min)

Three review surfaces. Each has a separate canonical doc; this handoff stitches them together.

| Surface | Question it answers | Canonical doc |
|---|---|---|
| **A. What shipped** | "Did the 14 S24 commits do what they claimed?" | §1 below + `docs/2026-05-13-S24-fix-sprint-qa.md` |
| **B. What's deferred** | "What did we explicitly choose NOT to ship, and why?" | §2 below + `docs/cleanbreak-deferred.md` |
| **C. What's next** | "What's left to lock down for v2.0?" | §3 below + `docs/2026-05-13-S25-plus-sprint-planning-tracker.md` |

If you only have 5 minutes: read §1's commit table, §2's deferral table, §3's S25 sprint package. The rest is depth.

---

## 1. Surface A — What shipped (S24 fix sprint, 14 commits)

**Predecessor HEAD:** `bed3c4a` (end of S15.5-S19 batch).
**Batch HEAD:** `7012eb4` (post-batch ROADMAP/BUG_LOG header refresh).
**Regression:** **2147 passed / 8 skipped / 6 known-fail.** Bug Bible 23/1/2 baseline held; `EXPECTED_FAILED_NODEIDS` set steady at 6. +39 net new tests.

### Commit table

| # | Hash | Subject | What to spot-check |
|--:|------|---------|---------------------|
| 1 | `cf8eb96` | `docs(readme): scrub Director references from README + reference fixture README (S23.10)` | README L33-35 pipeline arrow + L478 "Director JSON Resilience" labeled legacy. Forensic anchors present on every remaining mention. |
| 2 | `2002958` | `fix(audiogen): stamp sfx_render_status, prevent short-output cache poisoning, gate sfx_wav_path on save proof` | `_save_wav` returns explicit bool; short-output fallback writes to `_fallback/` subdir not canonical path; writeback gates `sfx_wav_path` on `save_ok AND os.path.isfile`. |
| 3 | `f7a5ca0` | `fix(musicgen): strict ImportError default + allow_silence_fallback opt-in (matches AudioGen S17.2)` | MusicGen ImportError raises RuntimeError by default; AudioGen widget vector realigned (stale `'{}'` from deleted production_plan_json input removed). |
| 4 | `6d3f893` | `fix(procsfx): stamp fallback_default_type on resolver default + clean stale wav/G7 comments` | Resolver tracks `matched: bool`; default-path stamps `fallback_default_type`. Stale `sfx_wav_path=None` + `[0.25, 12.0]` references gone or forensic-anchored. |
| 5 | `2bfab7f` | `feat(audit): tighten sfx_render_status to known-enum check (expanded set covering C2 + C4)` | `ALLOWED_SFX_RENDER_STATUS` frozenset has 8 values; walker checks enum membership only for this field; other 9 stay string-shape-only. |
| 6 | `0156797` | `test(workflow): pin AudioGen + MusicGen widget vectors to explicit allow_silence_fallback=false` | `test_workflow_audio_widget_vectors.py` has 6 tests covering length, false-pinning, type-alignment. |
| 7 | `493ab8c` | `cleanbreak(imp-31): delete AudioGen _cache_key back-compat alias (matches MusicGen S17.1)` | `_cache_key` deleted from AudioGen + tests; zero external callers confirmed; matches MusicGen S17.1 deletion. |
| 8 | `bb689f2` | `docs(cleanbreak): defer C8 CastContract quarantine -- premise was wrong, cast contract IS production-wired` | **DEFERRAL** -- see §2.A. |
| 9 | `af7e7b1` | `test(imp-33): automate ComfyUI queue-halt assumption smoke for _LLMTimeoutWorkflowPause` | Mock-based smoke; decision doc explains why Option B over A/C. Round-robin deviation noted. |
| 10 | `4e972c7` | `docs(cleanbreak): defer C10 LFC audit regex extension -- LFC is current architecture, not legacy` | **DEFERRAL** -- see §2.B. |
| 11 | `f9f5aa7` | `docs(imp-38): require justification comment per EXCLUDED_PATHS entry in legacy-audit test` | Per-entry `# justification:` comments on all 4 EXCLUDED_PATHS entries. |
| 12 | `d35aa71` | `docs(adr): close S14.2 active-validation design call (implementation deferred to S25+)` | ADR locks Option B (OTR_WorkflowValidator first-node). Round-robin deviation noted. |
| 13 | `fdb164b` | `docs: ROADMAP + BUG_LOG live update for the S24 fix sprint batch` | ROADMAP + BUG_LOG track BUG-LOCAL-209 + BUG-LOCAL-210. |
| 14 | `f529812` | `docs: S24 fix sprint QA document for round-robin review` | Detailed per-commit QA walkthrough (the doc QA reviewers ALSO want to read alongside this handoff). |
| 15 | `f11fee1` | `docs(planning): S25+ master sprint tracker consolidating every outstanding item across all batches` | The forward-looking sprint plan. |
| 16 | `7012eb4` | `docs: ROADMAP + BUG_LOG header refresh for post-batch state` | Header refresh; stack head + planning doc pointers. |

### Plan deviations to scrutinize

The QA team's main job here is to confirm these deviations were justified. Each is documented inline in the relevant commit message + the S24 QA doc §2.

| Deviation | Where | Justification source |
|---|---|---|
| C2 fallback to `_fallback/` subdir (vs. plan's "skip OR `_fallback/`") | `nodes/batch_audiogen_generator.py` short-output path | C2 commit body + S24 QA §1 C2 |
| C2 dropped `render_results` ledger stamp | AudioGen ImportError fallback | C2 commit body |
| C3 widget realignment beyond stated scope | AudioGen widget vector | C3 commit body + BUG-LOCAL-210 |
| C5 includes `"skipped"` forward-compat enum slot | `ALLOWED_SFX_RENDER_STATUS` | C5 commit body |
| **C8 DEFERRED** | Plan premise wrong | §2.A below |
| C9 round-robin skipped | C9 decision doc | Decision doc "Round-robin deviation" section |
| **C10 DEFERRED** | Plan premise wrong | §2.B below |
| C11 per-entry comments added beyond strict plan | `tests/test_legacy_audit_clean.py` | C11 commit body |
| C12 round-robin skipped | C12 ADR | ADR "Round-robin deviation" section |

### New Bible candidates (in `BUG_LOG.md`)

| Bug | Title | General lesson |
|---|---|---|
| BUG-LOCAL-209 | `_save_wav -> None` on both paths | Functions whose return is consumed by a contract MUST declare explicit bool, not implicit None. Audit `-> None` on functions whose callers check truthiness. |
| BUG-LOCAL-210 | AudioGen widget vector stale `'{}'` shifting every subsequent slot | Cleanbreak deleting a REQUIRED INPUT_TYPES entry MUST trim every saved-workflow widget vector at the same index in lockstep. |

### Drift guards (new this batch — 26 contracts pinned by tests)

Full list in S24 QA doc §5. QA spot-check questions:
- Did the C6 widget-vector tests cover the actual production drift (BUG-LOCAL-210), or just synthetic fixtures?
- Does the C9 mock-based queue-halt smoke actually exercise the exception-propagation contract, or just check the class hierarchy?
- Does C5's enum check fire on a typo in soft mode (`audit_post_freeze_writeback(led)` returns the violation) AND raise in strict mode?

→ **For full per-commit mechanics + open round-robin questions per commit:** `docs/2026-05-13-S24-fix-sprint-qa.md`.

---

## 2. Surface B — What's deferred (3 items with corrected premises)

Two of the three deferrals were caused by the plan's premise being wrong — the team should pay special attention to whether the corrected premise is right.

### 2.A — C8: CastContract quarantine (DEFERRED, **needs decision**)

**Plan said:** mechanical move from `nodes/_otr_cast_contract.py` to `nodes/experimental/_otr_cast_contract.py`. "Likely no internal imports."

**Reality:** dependency audit at execution time found:
- `nodes/_otr_cast_repair.py:40,312` imports `CharacterEntry`, `_extract_dialogue_tags` from cast_contract.
- `nodes/_otr_cast_repair.py` is consumed by `nodes/_otr_ledger_reviewer.py::apply_deterministic_cast_repairs` -- **live production code path called at writer-time**.
- 2 other files (`_otr_ledger.py`, `_otr_line_composer.py`) carry forensic references.

Cast contract IS wired into production. Quarantining would either break the repair path at writer-time or ship a "not wired into production" docstring lie.

**Three unblock options** (decision needed before any execution):
1. **Extract helpers** — move just the helpers cast_repair needs into a new `_otr_cast_helpers.py`, then quarantine cast_contract clean. Smallest scope; needs an audit of what cast_repair actually imports.
2. **Quarantine the chain** — move `cast_contract + cast_repair + apply_deterministic_cast_repairs` together. Cleanest separation; largest scope.
3. **Drop the quarantine plan** — accept cast contract is production-wired; reframe C8 as "audit + harden cast_contract as a production module."

**QA team question:** which option fits the project's broader cast-contract direction? Option 1 is the smallest scope but assumes a clean split exists; Option 2 is a large architectural commitment; Option 3 is "we were wrong, embrace it."

→ Full entry: `docs/cleanbreak-deferred.md` §C8.

### 2.B — C10: LFC audit regex extension (DEFERRED, **premise rejected**)

**Plan said:** extend `tests/test_legacy_audit_clean.py` regex with `\bLFC\b` tokens to audit "LFC legacy" surfaces.

**Reality:** dry-run grep found 159 hits. Spot-check showed:
- `OTR_LFCPhase4Scene`, `OTR_LFCPhase5Voice`, `OTR_LFCPhase6Arc` are **registered ComfyUI nodes**.
- 8 `_otr_lfc_*.py` modules are live infrastructure.
- LFC = "Live Freeze Cascade" = the CURRENT system, not a deleted lineage.

Adding LFC to the audit would flag all 159 references as violations. The audit's contract is to catch DELETED surfaces.

**Premise correction documented for future plans:** distinguish between (a) currently-live system names that happen to be acronyms (LFC, FLUX, HuMo, LTX -- leave alone) and (b) deleted system names (Director, LLMDirector, production_plan_json -- audit).

**QA team question:** worth spot-checking if there were any earlier LFC-era class names that were retired? E.g., if a hypothetical `OTR_LFC_Cascade` was retired in favor of the current `OTR_LFCPhase4/5/6`, that specific name would qualify. (Tracked as IMP-46 in the master tracker.)

→ Full entry: `docs/cleanbreak-deferred.md` §C10.

### 2.C — S14.2: Active validation (DEFERRED, **decision locked**, implementation pending)

**Status:** decision locked in S24 / C12 (commit `d35aa71`). Picked Option B (opt-in `OTR_WorkflowValidator` first-node) over Option A (ComfyUI frontend extension).

**Why Option B:**
1. ComfyUI's Python node API is the most stable extension surface.
2. Pure Python keeps validation in OTR's primary skill envelope.
3. Validation-at-execution is sufficient because OTR contributors run the production workflow on every change.
4. Failure mode is observable in the same channel as every other OTR node failure.
5. Option A's "earliest possible moment" advantage is theoretical in OTR's actual usage pattern.

**Implementation scope (S25+):** ~150 LOC new node class + workflow JSON wiring + tests covering canonical + adversarial broken workflow.

**QA team question:** is validation-at-execution acceptable, or should Option A get added on top once B ships (catch save-time drift before queue)?

→ Full ADR: `docs/2026-05-13-S14_2-active-validation-ADR.md`. Full deferral entry: `docs/cleanbreak-deferred.md` §S14.2.

---

## 3. Surface C — What's next (S25+ master sprint plan)

The master tracker (`docs/2026-05-13-S25-plus-sprint-planning-tracker.md`) organizes every outstanding item into 6 tiers and proposes S25+ sprint packaging. QA team can vote on the sprint packaging or flag missing items.

### Tier summary

| Tier | Item count | Examples |
|---|--:|---|
| **T1 — Architectural decisions blocking v2.0** | 2 | C8 unblock decision (T1.1); S14.2 implementation (T1.2) |
| **T2 — Pending-gate (waiting on time/event)** | 2 | S19.3 survival-guide promotion (T2.1, ready); IMP-33b cross-version (T2.2, external) |
| **T3 — Improvements closing known gaps** | 5 | IMP-43 widget-check in validator; IMP-42 EXCLUDED_PATHS AST meta-test; IMP-40 playbook step; IMP-34, IMP-35 |
| **T4 — Forward-compat / nice-to-have** | 6 | IMP-39, IMP-41, IMP-44, IMP-45, IMP-46, IMP-33a |
| **T5 — Rule conflicts (won't ship unless rule changes)** | 1 | S21.3 preset split |
| **T6 — Stretch / skipped** | 1 | S20 stretch items |

### Suggested S25+ sprint packaging

| Sprint | Headline | Items | Scope estimate |
|---|---|---|---|
| **S25** | Implementation sprint | T1.2 (S14.2 validator) + T3.1 (IMP-43 widget check fold-in) + T2.1 (S19.3 promotion side-action) | ~150 LOC new + 30 LOC fold-in + 1 sibling-repo commit |
| **S26** | Decision sprint | T1.1 (C8 round-robin + execute the chosen option) | Round-robin first, then execute (varies by option) |
| **S27** | Audit-test hardening | T3.2 + T3.3 + T3.4 + T3.5 + T4.5 | Mixed; mostly small additions |
| **S28+** | Forward-compat opportunistic | T4.1 / T4.2 / T4.3 / T4.4 / T6.1 | Fold into adjacent sprints |
| **Never** | (rule-conflicting) | T5.1 | Unless `feedback_minimum_json_files` rule changes |
| **External-triggered** | (no sprint) | T2.2 + T4.6 | On next ComfyUI major bump |

**QA team question:** does S25's package size feel right (1 substantial item + 1 fold-in + 1 side-action), or should T3.1 split out into its own sprint?

→ Full tier + per-item detail: `docs/2026-05-13-S25-plus-sprint-planning-tracker.md`.

---

## 4. Closed in S24 (so QA doesn't re-flag these)

| Item | Closing commit | One-line summary |
|---|---|---|
| S23.10 — README rewrite | `cf8eb96` (C1) | README + fixture README scrubbed; forensic anchors on every remaining Director mention. |
| IMP-31 — AudioGen `_cache_key` alias | `493ab8c` (C7) | Deleted. Matches MusicGen S17.1. |
| IMP-32 — `sfx_render_status` enum check | `2bfab7f` (C5) | 8-value frozenset; audit walker enforces. |
| IMP-33 — Queue-halt smoke | `af7e7b1` (C9) | Mock-based test. Real-subprocess = IMP-33a (T4.6); cross-version = IMP-33b (T2.2). |
| IMP-37 — LFC audit | `4e972c7` (C10) | **Rejected** — LFC is live architecture, not legacy. |
| IMP-38 — EXCLUDED_PATHS justification | `f9f5aa7` (C11) | Per-entry `# justification:` comments + module docstring rule. |

---

## 5. Files the QA team should pull up

In priority order for review depth:

1. **This doc** — orientation + the three review surfaces.
2. `docs/2026-05-13-S24-fix-sprint-qa.md` — detailed per-commit mechanics + drift-guard inventory + open round-robin questions per commit.
3. `docs/2026-05-13-S25-plus-sprint-planning-tracker.md` — forward-looking master plan with full per-item entries.
4. `docs/cleanbreak-deferred.md` — per-item drill-down on the 3 active deferrals.
5. `docs/2026-05-13-S14_2-active-validation-ADR.md` — design decision for T1.2 implementation.
6. `docs/2026-05-13-imp33-queue-halt-test-decision.md` — design decision for C9 / IMP-33.
7. `ROADMAP.md` "CURRENT WORK" section — single-paragraph batch summary.
8. `BUG_LOG.md` — BUG-LOCAL-209 + BUG-LOCAL-210 entries with full general-lesson framing.

---

## 6. Acceptance state

All gates green at `7012eb4`:

- [x] `tests/test_legacy_audit_clean.py` — `1 passed`
- [x] `tests/test_workflow_live_passes_validator.py` — `1 passed`
- [x] `tests/test_naming_conventions.py` — `3 passed`
- [x] `tests/test_workflow_audio_widget_vectors.py` — `6 passed`
- [x] `tests/test_audiogen_writeback_hardening.py` — `8 passed`
- [x] `tests/test_musicgen_strict_failure.py` — `4 passed`
- [x] `tests/test_post_freeze_writeback_audit.py` — `21 passed`
- [x] `tests/test_llm_timeout_queue_halt_smoke.py` — `4 passed / 1 skipped` (intentional Option C stub)
- [x] Bug Bible regression — `23 passed / 1 skipped / 2 xfailed` (baseline held)
- [x] Full pytest run — `2147 passed / 8 skipped / 6 known-fail` (exact match to `EXPECTED_FAILED_NODEIDS`)
- [x] Local HEAD == origin HEAD (`7012eb4004b2912c922bf209a755fea687d5db4c`)
- [x] No 0-byte tracked Python files
- [x] No BOM-prefixed tracked Python files
- [x] ROADMAP + BUG_LOG live-updated through `7012eb4`
- [x] QA doc + S25+ master sprint tracker shipped

**S24 fix sprint is LOCKED. Ready for QA review.**
