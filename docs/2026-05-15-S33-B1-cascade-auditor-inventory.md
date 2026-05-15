# S33 B1 — Cascade Phase 1 + Phase 9 Pre-Deletion Inventory

> **Status:** B1 inventory complete. **HALT and surface to Jeffrey before B2.**
> **Branch:** `s33-editor-only-cleanup` @ B0 (`4b296a2`).
> **Parent:** `s32-helper-per-subpass-routing` @ B8 (`3261b18`).
> **Plan:** `docs/2026-05-14-S33-editor-only-cleanup-sprint-plan.md`.

---

## Headline finding

The S33 plan's mental model of Phase 1 / Phase 9 does not match the current code structure. B1 cannot rubber-stamp a clean DELETE row for any Phase 1 or Phase 9 symbol; the standing drift policy fires.

Specifically:

1. **No Phase 1 method exists on `OTR_LedgerFreezeCascade` class.** Plan §B2 says "Delete Phase 1 method(s) from cascade class." There are zero `_phase_1*` / `phase_1*` methods on the class. The only references are docstring + comment mentions of "Phase 1 Auditor" naming.
2. **No Phase 1 widget exists in cascade `INPUT_TYPES`.** Plan §B2 says "Delete any Phase 1 widgets from cascade `INPUT_TYPES`." S30 B3 (2026-05-12 clean-break, see `OTR_LedgerFreezeCascade.py` lines 153-159) already removed every `phase_1_*` / `enable_phase_1_*` / `phase_9_*` / `enable_phase_9_*` widget. Current `INPUT_TYPES.optional` contains only: `script_json`, `news_used`, `estimated_minutes`, `technical_model`, `enable_phase_7_audio_readiness`, `enable_phase_8_video_readiness`, `vram_ceiling_gb`. None of these are Phase 1 / Phase 9 widgets.
3. **No Phase 1 dispatch exists in `OTR_LedgerFreezeCascade.run()`.** The cascade's `run` calls `_LFC_ORCH.run_freeze_cascade(...)`, which dispatches a single composite `phase_1_2_9_reviewer_composite` (one entry in `_PHASE_BUCKETS`, line 188 of `_otr_freeze_cascade.py`). There is no granular Phase 1 or Phase 9 dispatch at the cascade orchestrator level.
4. **Phase 1 + Phase 9 are inseparable at the `review_ledger` level.** Architectural decision D12 (`docs/script-writing-architecture-adr.md` line 23): "Three reviewer LLM calls implemented as a SINGLE `audit_cast_contract(ledger, label)` function called twice (pre + post), plus `run_script_doctor` once." The two audits share one function distinguished only by a `label` kwarg.
5. **Phase 1's output IS consumed by editors.** Plan framing: "Phase 1 auditor — emits issue reports, never rewrites." True for the LLM call itself, but `pre_audit.violations` is consumed downstream by:
   - `apply_deterministic_cast_repairs(candidate, pre_audit, cast_rows)` — Python editor (line 1152 of `_otr_ledger_reviewer.py`)
   - `speaker_unknowns` rollback gate — early-exit on `cast_unrecoverable` (line 1123-1149)
6. **Phase 9's output IS consumed by a rollback gate.** Plan framing: "Phase 9 post-edit auditor — verifies Phase 2 Script Doctor's edits landed correctly." True. The verification result drives the `post_audit_pass` gate (line 1218-1251): if Phase 9 reports violations, Phase 2's edits are REJECTED and the ledger rolls back to `original_snapshot`. So Phase 9 is not just diagnostic — it is the load-bearing veto on Phase 2's commit.

---

## Pre-grep coverage (all 6 dimensions)

### 1. Method / function symbols — cascade module

`nodes/OTR_LedgerFreezeCascade.py`:

| Hit | Line | Context |
|---|---|---|
| `Phase 1` | 90 | docstring listing reviewer-LLM consumers |
| `Phase 1 / 2 / 9` | 154 | comment about S30 B3 widget deletion |
| `Phase 1` | 267 | LLM-slot routing comment |
| `Phase 9` | 91 | docstring |
| `Phase 9` | 154 | same line as above |
| `Phase 9 (post-edit auditor)` | 268 | same comment block |

→ ALL hits are docstring / comment. Zero method, widget, or dispatch references.

`nodes/_otr_freeze_cascade.py`:

| Hit | Line | Context |
|---|---|---|
| `Phase 1 / Phase 2 / Phase 9` | 6-7 | module docstring |
| `Phase 1    cast audit + repairs       (existing reviewer Pass 1)` | 19 | docstring ADR diagram |
| `Phase 9    cast audit final           (existing reviewer Pass 3)` | 28 | docstring ADR diagram |
| `phase_1_2_9_reviewer_composite` | 188, 214, 531 | `_PHASE_BUCKETS` table + telemetry shape + actual dispatch phase_name stamp |
| `# ---- Phase 1 + 2 + 9: existing 3-pass reviewer -------` | 524 | dispatch comment in `run_freeze_cascade` |
| `Phase 1+2+9 (reviewer composite)` | 177 | comment in `_PHASE_BUCKETS` table |
| `Phase 1/2/9 reviewer` | 598 | dispatch-order comment |

→ Phase 1 + Phase 9 are NEVER independently dispatched. The cascade orchestrator's only call site for either is the composite `_OTRLR.review_ledger(generate_fn, led)` (line 524-531). Inside `review_ledger`, the two audits are `audit_cast_contract(generate_fn, candidate, label="pre")` and `audit_cast_contract(generate_fn, candidate, label="post")` — same function, same scope.

`nodes/_otr_ledger_reviewer.py` (the actual reviewer module — NOT in the plan's pre-grep target list, but the only file containing the Phase 1 / Phase 9 LLM call sites):

| Symbol | Line | Role |
|---|---|---|
| `def audit_cast_contract(...)` | 367 | THE LLM auditor function for both Pass 1 (= Phase 1) and Pass 3 (= Phase 9) |
| `pre_audit = audit_cast_contract(generate_fn, candidate, label="pre")` | 1089 | Phase 1 invocation |
| `post_audit = audit_cast_contract(generate_fn, candidate, label="post")` | 1215 | Phase 9 invocation |
| `def apply_deterministic_cast_repairs(...)` | 464 | EDITOR — consumes `pre_audit` |
| `def auto_remap_phantom(...)` | 223 | Levenshtein helper called by `apply_deterministic_cast_repairs` |
| `def run_script_doctor(...)` | 750 | Phase 2 editor |
| `def apply_doctor_edits(...)` | 829 | Phase 2 commit step |
| `def apply_phantom_skip_fallback(...)` | 906 | Step 2.5 deterministic phantom-skip (between Phase 2 and Phase 9) |

### 2. Widget keys in cascade `INPUT_TYPES`

`OTR_LedgerFreezeCascade.INPUT_TYPES.optional` keys after S30 B3:

```
script_json, news_used, estimated_minutes, technical_model,
enable_phase_7_audio_readiness, enable_phase_8_video_readiness,
vram_ceiling_gb
```

Phase 1 / Phase 9 widget keys: **none exist**. Plan §B1 step 2 pre-grep (`enable_phase_1_*`, `phase_1_*`, `enable_phase_9_*`, `phase_9_*`) yields zero hits.

### 3. `meta` dict keys stamped by Phase 1 or Phase 9

Phase 1 (`audit_cast_contract(label="pre")`) itself stamps no meta keys directly. Its result `pre_audit` is consumed by `review_ledger`, which in turn stamps:

| meta key | written at | shape | consumer |
|---|---|---|---|
| `meta.reviewer_verdict` | `review_ledger` lines 1041, 1100, 1133, 1166, 1191, 1231, 1261 | string literal (`clean_no_edits` / `improved` / `cast_unrecoverable` / `too_many_edits` / `needs_full_rerun` / `post_audit_failed`) | Cascade orchestrator (`REVIEWER_TO_FREEZE_VERDICT` table, line 84-91 of `_otr_freeze_cascade.py`) + downstream `freeze_verdict` output socket |
| `meta.reviewer_disposition` | lines 1042-1046, 1113, 1143, 1176, 1201, 1245, 1271 | `ReviewerDisposition.__dict__` (7 numeric counters + verdict) | Cascade telemetry; surfaced on `meta.freeze_disposition.reviewer_disposition` |
| `meta.reviewer_audit_failure_reason` | lines 1101, 1233 | string | Diagnostic only |

`ReviewerDisposition` fields (line 178-187 of `_otr_ledger_reviewer.py`): `verdict`, `pre_audit_violations`, `pre_audit_repairs_applied`, `doctor_edits_proposed`, `doctor_edits_applied`, `post_audit_violations`, `phantom_skip_count`. Three of these (`pre_audit_violations`, `pre_audit_repairs_applied`, `post_audit_violations`) directly carry Phase 1 / Phase 9 outputs.

Phase 9 (`audit_cast_contract(label="post")`) similarly stamps no meta keys directly. Its result `post_audit` drives `post_audit_pass` (line 1218) which gates the commit branch (1253-1277) vs the rollback branch (1227-1251) of `review_ledger`.

The cascade orchestrator additionally stamps composite-level keys:

| meta key | written at | shape | downstream consumer |
|---|---|---|---|
| `meta.cleanup_passes[...]` | `_otr_freeze_cascade.py` _stamp_phase_record path | list of dicts, includes `phase_1_2_9_reviewer_composite` entry | `build_phase_telemetry` (line 203), `all_phase_passes` (line 276) |
| `meta.freeze_disposition` | cascade exit | `FreezeDisposition.to_dict()` | Downstream consumers: see dimension 4 below |

### 4. Downstream consumer grep — every meta key from step 3

Searched `nodes/ scripts/ workflows/ tests/` for reads of each meta key.

`meta.reviewer_verdict` reads:
- `nodes/_otr_freeze_cascade.py` — verdict mapping into `freeze_verdict`. Production path.
- `tests/test_phase3_ledger_reviewer.py` — verdict assertions. Test path.
- ROADMAP.md / BUG_LOG.md — documentation only.

`meta.reviewer_disposition` reads:
- `nodes/_otr_freeze_cascade.py` — packed into `FreezeDisposition.to_dict()` via `disp.reviewer_disposition` field.
- `tests/test_phase3_ledger_reviewer.py` — disposition shape assertions.

`pre_audit.violations` reads (object field, not a meta key):
- `nodes/_otr_ledger_reviewer.py:1090` — `pre_audit_violations = len(pre_audit.violations)` (counter for `ReviewerDisposition`)
- `nodes/_otr_ledger_reviewer.py:1124` — speaker_unknowns rollback gate
- `nodes/_otr_ledger_reviewer.py:1152` — `apply_deterministic_cast_repairs(candidate, pre_audit, cast_rows)` **<= LOAD-BEARING EDITOR INPUT**

`post_audit.violations` / `post_audit.pass_clean` reads:
- `nodes/_otr_ledger_reviewer.py:1218-1222` — `post_audit_pass` rollback gate **<= LOAD-BEARING ROLLBACK SIGNAL**
- `nodes/_otr_ledger_reviewer.py:1232-1234` — audit_failed_reason stamp
- `nodes/_otr_ledger_reviewer.py:1242` — `post_audit_violations` counter

### 5. Workflow JSON grep

Scanned `workflows/*.json` for any widget key containing `phase_1` / `phase_9` / `enable_phase_1` / `enable_phase_9`:

| File | Hits |
|---|---|
| `workflows/otr_scifi_16gb_full.json` | 0 |
| `workflows/otr_humo_smoke.json` | 0 |
| `workflows/otr_humo_only_smoke.json` | 0 |
| `workflows/otr_humo_4x_smoke.json` | 0 |
| `workflows/ltx_2_3_downstream_smoke.json` | 0 |
| `workflows/external_examples/*.json` | 0 |

→ Zero workflow JSON references. S30 B3's clean-break already swept these.

### 6. Test file inventory

Files mentioning Phase 1 / Phase 9 / pre_audit / post_audit / audit_cast_contract / apply_deterministic_cast_repairs in `tests/`:

| File | Role |
|---|---|
| `tests/test_phase3_ledger_reviewer.py` | **Primary tests** — covers `audit_cast_contract` (clean/dirty/LLM-fail/malformed), `auto_remap_phantom` (Levenshtein G8 table), `apply_deterministic_cast_repairs` (every repair kind), end-to-end `review_ledger` for ALL SIX verdicts. **Heavy coverage; cannot be naively deleted.** |
| `tests/test_phase1_composer_prompt.py` | UNRELATED — "Phase 1" here refers to composer-prompt sprint phase 1, not the cascade Phase 1 auditor. Filename collision, not a Phase 1 auditor test. |
| `tests/test_lfc_freeze_cascade_orchestrator.py` | Cascade orchestrator tests — references composite `phase_1_2_9_reviewer_composite` (not individual phases). |
| `tests/test_lfc_c4_news_used_passthrough.py` | Cascade passthrough — references `phase_1_2_9_reviewer_composite` and full cascade chain. |
| `tests/test_lfc_b1_cascade_unload_in_finally.py` | VRAM unload regression — references composite. |
| `tests/test_lfc_phase_7_8_readiness.py` | Phase 7 + 8 readiness — UNRELATED to Phase 1 / Phase 9. |
| `tests/test_lfc_phase_0_10_gap_audit.py` | Phase 0 + 10 deterministic gap audits — UNRELATED. |
| `tests/test_lfc_g4_telemetry_derivation.py` | Telemetry derivation — references composite. |
| `tests/test_freeze_cascade_g6.py` | Cascade G6 contract — references composite. |
| `tests/test_g8_line_id_uniqueness.py` | Line ID uniqueness — references composite. |
| `tests/test_per_cue_sfx_dur.py` | SFX duration — references composite. |
| `tests/test_core.py` | Core test — phase mentions but UNRELATED. |
| `tests/test_workflow_json_guardrails.py` | Workflow JSON guardrails — checks no phase_1/9 widget keys, but does so as a NEGATIVE assertion (already absent). |

---

## Decision table

| Symbol | File:Line | Type | Decision | Reason |
|---|---|---|---|---|
| `_PHASE_BUCKETS["phase_1_2_9_reviewer_composite"]` | `_otr_freeze_cascade.py:188` | composite dispatch entry | **HALT** | Composite covers Phases 1+2+9 together. Cannot delete Phase 1 / Phase 9 entries from this table individually because there is no individual entry — only the composite. Deleting the composite would also delete Phase 2 dispatch (which the plan keeps). |
| `audit_cast_contract` function | `_otr_ledger_reviewer.py:367` | LLM call shared by Phase 1 + Phase 9 | **HALT** | Per ADR D12 the two audits intentionally share one function. Deleting "Phase 1" means deleting `label="pre"` branch usage; deleting "Phase 9" means deleting `label="post"` branch usage. After both deletions the function becomes dead code. The function itself is structurally the same call site for both; the plan's "delete Phase 1 method" framing assumes Phase 1 has its own method, which is false. |
| `pre_audit = audit_cast_contract(..., label="pre")` | `_otr_ledger_reviewer.py:1089` | Phase 1 LLM invocation site | **HALT** | `pre_audit` is consumed downstream by `apply_deterministic_cast_repairs` (editor) and `speaker_unknowns` rollback gate. Deletion breaks both. The plan's drift policy explicitly flags this case: "meta key WITH downstream consumer → flag for separate handling". |
| `post_audit = audit_cast_contract(..., label="post")` | `_otr_ledger_reviewer.py:1215` | Phase 9 LLM invocation site | **HALT** | `post_audit.violations` and `post_audit.pass_clean` drive `post_audit_pass` rollback gate. Phase 9 is the load-bearing veto on whether Phase 2's edits commit. The plan acknowledges this gate exists in §B3 ("Phase 9 cannot be deleted until Phase 2 demonstrably hard-fails malformed output") but the architectural reality is broader: Phase 9 also catches downstream defects (e.g., Phase 2 made a clean rewrite but introduced a new phantom; Phase 2 edited correctly but Step 2.5 phantom-skip left residue). B3's hard-fail-on-malformed-output proof is necessary but not sufficient. |
| `apply_deterministic_cast_repairs(candidate, pre_audit, ...)` | `_otr_ledger_reviewer.py:1152` | Editor consuming Phase 1 output | **DOWNSTREAM CONSUMER** | Plan-edge case: this editor's signal source is Phase 1. If Phase 1 is deleted, this editor either becomes a no-op or needs an alternative signal source (e.g., scan the ledger directly for phantoms without an LLM step). Requires architectural decision before B2. |
| `speaker_unknowns` rollback gate | `_otr_ledger_reviewer.py:1123-1149` | Phase 1-driven rollback gate | **DOWNSTREAM CONSUMER** | Drives `cast_unrecoverable` verdict on high-confidence unknown speakers. If Phase 1 is deleted, this gate never fires; cast_unrecoverable becomes an unreachable verdict literal. Question for Jeffrey: is `cast_unrecoverable` a verdict the cascade should still be able to emit? It's part of the verdict literal set in `OTR_LedgerFreezeCascade.py` line 17. |
| `post_audit_pass` rollback gate | `_otr_ledger_reviewer.py:1218-1251` | Phase 9-driven rollback gate | **DOWNSTREAM CONSUMER** | The veto on Phase 2's commit. If Phase 9 is deleted: Phase 2's edits always commit. The plan's B3 gate (prove Phase 2 hard-fails malformed output) addresses ONE failure mode; this rollback gate addresses ALL post-edit defects, including residue from Step 2.5 phantom-skip and the `final_phantoms` line-1216 check. Question for Jeffrey: do we keep the rollback gate driven by a non-LLM mechanism, or drop the rollback entirely and accept that Phase 2's edits always commit? |
| `meta.reviewer_verdict` literal set | `OTR_LedgerFreezeCascade.py:13-18` + `REVIEWER_TO_FREEZE_VERDICT` map | Verdict literal set | **DOWNSTREAM IMPACT** | After deleting Phase 1 + Phase 9, three verdicts become unreachable: `cast_unrecoverable` (Phase 1 gate), `post_audit_failed` (Phase 9 gate), and `needs_full_rerun` from the Phase 1 audit-failed sentinel branch. Plan implicitly assumes the verdict set survives unchanged. |
| `tests/test_phase3_ledger_reviewer.py` | full file | Phase 1/2/9 + repairs tests | **DOWNSTREAM IMPACT** | Heavy coverage on `audit_cast_contract` (4 tests), `auto_remap_phantom` (5+ tests), `apply_deterministic_cast_repairs` (8+ tests), end-to-end `review_ledger` (~6 tests covering each verdict). Naively deleting Phase 1 / Phase 9 collapses 20+ tests. Plan §B2's "Delete Phase 1 test file(s) per B1 inventory" framing doesn't match; the tests cover a tangled composite, not individual phases. |
| All workflow JSON files | `workflows/*.json` | Widget references | **NONE** | Zero hits. S30 B3 already swept clean. Plan-side projection that workflow JSONs need updating: NOT NEEDED in B2 / B4 (already done). |

---

## Halt rationale (drift policy)

Per S33 plan §Drift Policy:

> Findings needing Jeffrey's architectural decision:
>   - Halt the autonomous run
>   - Document the question
>   - Report to Jeffrey before resuming
>   - Specifically: if B1 inventory surfaces a phase that EDITS content (not just audits), halt before deleting it

Strict reading: Phase 1's LLM call (`audit_cast_contract(label="pre")`) does NOT itself edit. Phase 9's LLM call (`audit_cast_contract(label="post")`) does NOT itself edit. So the literal "phase that EDITS" condition does not fire.

BUT the broader policy clause applies: "meta key WITH downstream consumer → flag for separate handling (may need a Sprint E entry or follow-up commit)". Phase 1's output is a load-bearing input to two editor / rollback paths. Phase 9's output is the load-bearing veto on Phase 2's commit.

The structural mismatch between plan and code makes B2 / B4 unsafe to execute without Jeffrey's decision on:

1. **What does "delete Phase 1" mean given that Phase 1 is just `audit_cast_contract(label="pre")` inside a tightly-coupled `review_ledger` composite?**
   - Option A: Delete only the `pre` branch call site → `pre_audit.violations` becomes empty → deterministic repairs lose signal → speaker_unknowns gate never fires → `cast_unrecoverable` verdict becomes unreachable.
   - Option B: Delete the `pre` branch AND `apply_deterministic_cast_repairs` AND the speaker_unknowns gate → larger surgery, larger test impact, but matches the "audit-only retired" plan intent.
   - Option C: Delete the entire `review_ledger` composite → also deletes Phase 2 (which the plan KEEPS). Bad option.
   - Option D: Other.

2. **What does "delete Phase 9" mean given that Phase 9's output is the veto on Phase 2's commit?**
   - Option A: Delete only the `post` branch call site + the `post_audit_pass` gate → Phase 2's edits always commit, no post-edit verification. B3's "Phase 2 hard-fails malformed output" proof catches structural defects but not "Phase 2 made a clean edit that introduces a new phantom".
   - Option B: Delete the LLM `post` audit but replace the rollback gate with a deterministic post-edit check (e.g., re-run `_final_phantom_check` and rollback on any hits, no LLM needed).
   - Option C: Other.

3. **Do the verdict literals `cast_unrecoverable` and `post_audit_failed` survive S33?** Plan implicit assumption is yes. If yes, Option B from each question above is required (preserve the gate, swap the signal source).

4. **`apply_deterministic_cast_repairs` and `auto_remap_phantom` — do these survive S33?** They are editors (they rewrite content), so per standing directive #1 ("every node must edit the story") they survive. But they currently take `pre_audit` as input; if Phase 1 is deleted, their signature needs to change.

5. **`audit_cast_contract` function (the shared Pass 1 / Pass 3 function) — what is its fate?** After deleting both `label="pre"` and `label="post"` call sites, the function is dead code. Plan says "delete Phase 1 method(s)" and "delete Phase 9 method(s)" — combining both implies deleting the shared function. OK; just naming it out loud.

6. **`tests/test_phase3_ledger_reviewer.py` — how much of it survives?** Tests for `auto_remap_phantom` and `apply_deterministic_cast_repairs` survive only if those functions survive (Option B above). Tests for `audit_cast_contract` are deleted. Tests for `review_ledger` end-to-end need to be rewritten because the verdict surface changes.

---

## Recommendation to Jeffrey

S33 plan needs revision before B2 can proceed. Two viable paths:

**Path 1 (narrow): Reframe S33 as "delete the cast-auditor LLM calls only; keep the deterministic editor and rollback gates with new signal sources."**
- B2 = delete Phase 1 LLM call + auto-derive `pre_audit`-shaped input from a deterministic scan; preserve `apply_deterministic_cast_repairs`.
- B4 = delete Phase 9 LLM call + replace `post_audit_pass` gate with deterministic `_final_phantom_check`-only verification.
- Verdict literal set survives.
- `audit_cast_contract` function deleted.
- Test surgery is moderate: `test_phase3_ledger_reviewer.py`'s LLM-call tests deleted, deterministic-function tests preserved with updated signatures.

**Path 2 (broad): Reframe S33 as "delete the entire cast-audit-and-repair stack."**
- B2 = delete Phase 1 + `apply_deterministic_cast_repairs` + `auto_remap_phantom` + `speaker_unknowns` gate + `cast_unrecoverable` verdict.
- B4 = delete Phase 9 + `post_audit_pass` gate + `post_audit_failed` verdict + `_final_phantom_check` + `apply_phantom_skip_fallback`.
- Massively simpler `review_ledger` — only Phase 2 (Script Doctor) + apply_doctor_edits remain.
- `audit_cast_contract` function deleted.
- Verdict literal set shrinks; the cascade `freeze_verdict` set drops `cast_unrecoverable` and `post_audit_failed`.
- Test surgery is large: most of `test_phase3_ledger_reviewer.py` deleted.

**Path 3 (no-op): Defer S33 to a successor sprint with a revised plan that matches the code.**
- B0 (already shipped) becomes a no-op plan landing.
- B1 inventory (this document) becomes the seed for the successor sprint's planning.
- No B2-B6 in S33.

---

## Other findings (file later, not in S33 scope)

- The two writer-side LLM widgets (`creative_writing_model`, `technical_model`) are the only model widgets per S30 / S32 design. The cascade reviewer LLM call is correctly wired to the broadcast `technical_model` socket (line 271 of `OTR_LedgerFreezeCascade.py`). No drift.
- `_PHASE_BUCKETS` table has `phase_1_2_9_reviewer_composite` listed under `cleanup_passes`. If the cascade-orchestrator-level dispatch entry survives the rework, the name should change to reflect the new composition (e.g., `phase_2_script_doctor_only`). Cosmetic; defer.
- Polish prompt rename (B5) is independent of B2 / B4 and can proceed safely whether or not B2 / B4 are revised. B5 also does not depend on the inventory questions above.

---

## What changed since the plan was written

- Nothing in the code has changed; the plan was finalized 2026-05-14 and S32 closed 2026-05-14 @ 3261b18. The mismatch is in the plan's mental model, not in code drift.
- Round-robin reviewers (Gemini + ChatGPT) did not catch the cascade-vs-reviewer architectural mismatch because the round-robin context was the plan itself, not the cascade source tree. Verifiable architectural assumptions about Phase 1 / Phase 9 living on the cascade class (rather than inside `_otr_ledger_reviewer.review_ledger`) went unchallenged.

---

## Sources

- `nodes/OTR_LedgerFreezeCascade.py` — cascade ComfyUI node, no Phase 1 / Phase 9 methods or widgets present.
- `nodes/_otr_freeze_cascade.py` — cascade orchestrator, dispatches Phase 1+2+9 as one composite.
- `nodes/_otr_ledger_reviewer.py` — actual Phase 1 + Phase 9 LLM call sites and downstream editors.
- `docs/script-writing-architecture-adr.md` line 23 (D12) — single-function-twice design decision.
- `docs/2026-05-14-S33-editor-only-cleanup-sprint-plan.md` — sprint plan (canonical name landed in B0).
- `tests/test_phase3_ledger_reviewer.py` — primary test coverage for the impacted code paths.
