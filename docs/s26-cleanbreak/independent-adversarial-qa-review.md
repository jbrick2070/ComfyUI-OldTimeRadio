# S26 Cleanbreak -- Independent Adversarial QA Review

**Verdict:** CLEANBREAK COMPLETE -- 7 issues (1 cleanup on internally-inconsistent SFX contract, 2 cleanup on sibling-pattern audit misses, 2 cleanup on sweep-pattern gap + downstream of deferred B6, 2 nits)
**Reviewer:** Independent QA pass, adversarial framing
**Branch reviewed:** `s26-cleanbreak` @ `5bf9d3a` (vs `s25-musicgen-parity` HEAD `3393b39`)
**Date:** 2026-05-13
**Scope rule applied:** Cleanbreak target is legacy ledger tolerance code. Production nodes migrate, not delete. Production bugs stay -- they are forward work, not cleanbreak targets.

---

## Headline

A1-A4 + B1 + B3 + B6/post_audio_video_pipeline shipped as **real deletions, not new gates**. No new DeprecationWarnings, no new shims, no new back-compat language was introduced. Production pipeline files intact. Known-fail node ID delta is byte-identical. Only one whole-file deletion (`tests/test_audiogen_legacy_gate.py`); production nodes untouched outside documented ledger-migration scope.

What weakens the run:

1. **The SFX contract is internally inconsistent post-A3.** A3 removed `"sfx": []` from the schema scaffold and from `_REQUIRED_TOP_LEVEL_LISTS`, but the API surface that **writes** the deleted shape survives: `set_sfx()` (creates the key on demand), `apply_sfx_timings()` (subscript-reads `self.data["sfx"]` which is now uninitialized in fresh ledgers), and `_merge_with_disk` still has `"sfx"` in `ROW_KEYED` so any on-disk ledger with sfx rows would forward them into memory. None of these have production callers, so nothing breaks in practice -- but the contract is not actually gone.

2. **B6 enumeration has sibling-pattern misses.** Cowork enumerated `otr_video_plan.py:645` and `batch_humo_render.py:2806` but missed the identical patterns at `otr_shot_duration_calculator.py:287` and `video_composite.py:383`. Same fact, same risk profile, not in the deferred list.

3. **Forbidden-pattern sweep pattern set is too narrow.** Cowork's sweep used the literal-string patterns `back-compat | legacy fallback | shim | ...` -- which misses `otr_legacy_audio_dir()` call sites (legacy by function name, not by inline comment). Several B6 deferrals are downstream of those callers; the enumeration isn't wrong, but the upstream surface didn't show in the sweep.

Deferrals are honest. B4, B5, B6, B2 stop all have named follow-up sprints with documented triggers ((a) non-zero producer/consumer/test audit, or (b) architectural blast radius on the audio path). Production pipeline files are intact. No over-deletion.

---

## Per-issue table

| ID | Severity | Surface | Evidence | Recommended action | Category |
|----|----------|---------|----------|--------------------|----------|
| QA-1 | cleanup | `production_ledger.py` SFX-contract internal inconsistency: `set_sfx` L810, `apply_sfx_timings` L865, `_merge_with_disk` L1041 row-keyed merge | A3 removed `"sfx": []` from `__init__` (`L321`) + `_REQUIRED_TOP_LEVEL_LISTS`. But **(a)** `set_sfx(sfx_rows)` writes `self.data["sfx"] = rows` -- creating the key on demand; 0 production callers, 1 test caller. **(b)** `apply_sfx_timings(timing)` does `for row in self.data["sfx"]:` -- subscript-reads a key the scaffold no longer initializes; would `KeyError` on a fresh ledger; 0 production callers, 1 test caller. **(c)** `_merge_with_disk` has `ROW_KEYED = {"lines": "line_id", "clips": "line_id", "sfx": "cue_id", "music": "cue_id"}` -- if any on-disk ledger has `sfx` rows, the merge forwards them into memory at the `if on_disk_rows and not in_mem_rows: in_mem[arr_name] = on_disk_rows` branch. S25 CD-3 audit established zero producers ever wrote `ledger["sfx"]`, so the disk path is inert in practice -- but the contract still preserves the deleted shape. A3's audit looked at READ paths in production code; it did not audit WRITE paths on the Ledger class itself or the disk-merge preservation list. | Delete `set_sfx` + `apply_sfx_timings` (+ their 2 test callers), remove `"sfx"` from `_merge_with_disk::ROW_KEYED`. Single small follow-up commit (~15 LOC net delete). Update the A3 audit checklist: "when deleting a schema field, audit setter methods, timing methods, AND the disk-merge preserve list." | dead symbol residue + residual coupling |
| QA-2 | cleanup | B6 enumeration miss -- `nodes/video_composite.py:383` `_load_ledger` | Same pattern Cowork's audit DID enumerate at `nodes/batch_humo_render.py:2806` ("Compatibility shim around `_load_ledger_with_path`") -- and deferred. `video_composite.py:383` carries an identical docstring (`"""Backwards-compat shim around _load_ledger_with_path. Returns only the parsed ledger dict for callers that don't need the source path. New code should prefer _load_ledger_with_path..."""`) and is also a back-compat shim. Cowork's B6 audit did extend coverage to unchanged files (`otr_video_plan.py`, `_otr_paths.py`, `video_engine.py`, `video_composite.py:2183`), so the rule was not "changed files only" -- this is a sibling-pattern miss. | Add to the deferred list under "B6 video pipeline misc -> SPRINT #3 (A)" rollup, or delete in a single small commit. The shim is a 3-line wrapper. | sibling-pattern audit miss |
| QA-3 | cleanup | B6 enumeration miss -- `nodes/otr_shot_duration_calculator.py:287` `"shot_id": frame_id, # legacy alias` | Cowork DID enumerate the identical pattern `"shot_id": frame_id, # back-compat: some callers still key on shot_id` at `nodes/otr_video_plan.py:645` in B6 and deferred it. The sibling at `otr_shot_duration_calculator.py:287` is the same fact, same alias, same risk profile -- not enumerated. | Group both into the same "B6 shot_id alias sweep" deferred sprint, or delete in lockstep. Both files write the same composite envelope. | sibling-pattern audit miss |
| QA-4 | cleanup | Forbidden-pattern sweep gap -- `otr_legacy_audio_dir()` call sites not detected | Cowork's sweep regex was `back-compat\|back_compat\|backcompat\|legacy fallback\|legacy path\|legacy_path\|shim\|DeprecationWarning`. This misses `otr_legacy_audio_dir()` callers, which are legacy by function name. The function itself (`_otr_paths.py:201`) is on the deferred B6 list ("B6 path back-compat -- small"), but the 14 callers across `_otr_ledger.py:328`, `batch_audiogen_generator.py:33`, `batch_bark_generator.py:33`, `batch_humo_render.py:65/2840`, `batch_ltx_render.py:82/2090`, `audio_enhance.py:434`, `scene_sequencer.py:879/1226`, `video_composite.py:90/405` are downstream of that deferral and don't appear in the sweep. The B6 path-back-compat sprint should explicitly enumerate them at open. | Extend the audit-results.md "B6 path back-compat" entry with the 14 caller sites so the next sprint has a closed list. Optionally add `\botr_legacy_audio_dir\b` to the forbidden-pattern regex set for future sweeps. | sweep-completeness gap |
| QA-5 | nit | `tests/test_audiogen_ledger.py::test_audiogen_iter_sfx_only` under `-W error::DeprecationWarning` | `deprecation-audit.txt` confirms 1 NEW regression vs baseline known-fail set; classification "likely third-party" was made without traceback evidence (cmd.exe shell terminated before stdout flush -- 3 retries). The "gate held" claim in audit-results.md §Strict DeprecationWarning audit is contingent on a classification that has not actually been performed. | Hand-back item already captured (BUG-LOCAL-221 + final-qa-review §8 step 3). Until the traceback is captured in an interactive shell and classified, the strict-deprecation gate is "passed with hand-wave." | audit accuracy |
| QA-6 | nit | `nodes/scene_sequencer.py:937-993, 1318-1319` SFX-mirror-into-lines walk now a permanent no-op | After A3 deleted the `"sfx": []` scaffold, `_led.get("sfx") or []` returns `[]` on every read. The `for _sfx_idx, ...` loop guarded by `if _sfx_idx >= len(_ledger_sfx): break` now breaks on iteration 0. The BUG-LOCAL-107 SFX-mirror-into-lines block is permanently inert. S25 CD-3 audit established zero producers ever wrote `ledger["sfx"]` in production, so the walk was effectively no-op pre-S26. Audio path is not broken; deferral is honest. Still: A3 could have stripped the dead walk in-commit since it had no producer to break. | Open the named follow-up "B6 sequencer SFX-mirror migration to lines[]-native source" or strip the dead walk directly. | incomplete migration (defensible deferral) |
| QA-7 | nit | Cleanbreak fixture-state runtime gap -- ComfyUI Desktop re-save deferred | `workflows/otr_scifi_16gb_full.json` was textually scrubbed in A4b (widget_values[0]: `"[]" -> "{}"`), but the file has NOT been re-saved through ComfyUI Desktop. Per §11 of the plan, only ComfyUI's own save path normalizes the full widget vector against the current INPUT_TYPES contract. Until that re-save happens, the fixture is "textually clean but not contract-normalized." The fixture is structurally valid JSON (`json.loads` passes), so this is a tail-of-cleanbreak item, not a blocker. | Hand-back captured in final-qa-review §8 step 1 (boot ComfyUI, drag validator, re-save). | runtime gap |

---

## Whole-file deletion audit

| File | Verdict | Evidence |
|------|---------|----------|
| `tests/test_audiogen_legacy_gate.py` | DELETED -- correct | `git diff --name-status s25..s26` shows `D`. Whole 144-line file gone with A1. |
| `nodes/post_audio_video_pipeline.py` | KEPT -- correct as scoped | The S25 ADDENDUM B6 entry called the BACK-COMPAT LINE (L124 flat-layout) "doubly dead" -- referring to the line/branch, not the whole file. The class `PostAudioVideoPipeline` is still registered at `__init__.py:181` with the comment "Kept registered so any old workflow JSON that still references it loads without error; new builds should not use it." Post-S26, the file is 420 lines including a live `INPUT_TYPES + execute` class. Deleting the whole file requires (a) removing the `__init__.py` registration AND (b) validating no workflow JSON references the type -- a separate "fully retire OTR_PostAudioVideoPipeline" sprint. The S26 commit `88bbbe9` correctly stripped only the legacy flat-layout fallback inside `_resolve_ledger_from_input`. Not a husk; not a missing whole-file delete. |
| `nodes/_otr_outline.py` | KEPT -- correct (B2 STOPPED) | 22 hits surveyed; design judgment required (mandate budget vs preserve bare-format); explicit stop logged in `cowork-stop-log.md`. The stop is legitimate per directive §5. |
| `tests/test_otr_workflow_validator.py` | ADDED -- correct (Sprint 3) | 183 lines, 8 tests, all green. |
| `nodes/_otr_workflow_validator.py` | ADDED -- correct (Sprint 3) | 154 lines; registered at `__init__.py:180`; wired into `workflows/otr_scifi_16gb_full.json` as `OTR_WorkflowValidator id=63 pos=[-300,-300]`. |
| `tools/validate_workflow_links.py` | ADDED -- correct (Sprint 2) | 244 lines static link-integrity checker; report clean across all 5 in-repo fixtures. |

No near-empty husks. No missing whole-file deletes.

---

## Production pipeline intactness

All production node files present at expected paths under `nodes/` on `s26-cleanbreak`. FLUX renderer lives in `scripts/render_flux_batch.py`.

The canonical workflow `workflows/otr_scifi_16gb_full.json` still contains the production graph: story writer, scene sequencer, Bark/Kokoro/MusicGen/AudioGen, FLUX render, HuMo render, VideoComposite, LTX render, RTX upscale, post-upscale blend, portrait render, FreezeCascade, plus the new `OTR_WorkflowValidator id=63`.

### Production node files modified -- legitimate ledger-migration changes only

| File | Change | Verdict |
|------|--------|---------|
| `batch_audiogen_generator.py` | A1 + A2-sibling + A4a. 138-line delta. | Legitimate. Render logic intact. |
| `musicgen_theme.py` | A2 (single-tier `_find_cached`). 55-line delta. | Legitimate. Render logic intact. |
| `batch_procedural_sfx.py` | A4a (default flip). 2-line delta. | Legitimate. |
| `production_ledger.py` | A3 + B3. 54-line delta. | Legitimate -- but see QA-1 (sfx contract still internally inconsistent). |
| `batch_humo_render.py` | B3 (`_build_pos_prompt` fallback removed). 11-line delta. | Legitimate. |
| `_otr_ledger.py` | B1 (l2 narrative scrub). 24-line delta. | Legitimate -- pure docstring scrub. |
| `_otr_ledger_freeze.py` | A3 extension (drop sfx from `_REQUIRED_TOP_LEVEL_LISTS`). 14-line delta. | Legitimate -- validator mirror. |
| `post_audio_video_pipeline.py` | B6 (legacy flat-layout removed). 19-line delta. | Legitimate. |

**No production node file shows render-logic modification outside the documented ledger-migration scope.** No over-deletion.

---

## Fresh forbidden-pattern sweep (run by reviewer, not trusted from Cowork)

Command:
```
git --no-pager grep -nE "back-compat|back_compat|backcompat|legacy fallback|legacy path|legacy_path|\bshim\b|DeprecationWarning" \
  s26-cleanbreak -- 'nodes/*.py' 'tests/*.py'
```

Result: 74 hits. Cross-diffed against the same query on `s25-musicgen-parity`.

**New OTR-origin back-compat surfaces introduced by S26: 0.** Independent confirmation.

### Hit classification (full survey, not Cowork-filtered)

| Category | Files | Verdict |
|----------|-------|---------|
| Documented B-group deferrals | `_otr_line_composer.py` (B4), `_otr_ledger_freeze.py` (B5), `_otr_outline.py` (B2 STOPPED), `batch_humo_render.py`, `OTR_LedgerScriptWriter.py`, `story_orchestrator.py`, `_otr_paths.py`, `video_engine.py`, `video_composite.py:2183`, `otr_video_plan.py`, `scene_sequencer.py` (B6) | Documented with named follow-up sprints. |
| Sibling-pattern misses (NOT enumerated) | `video_composite.py:383`, `otr_shot_duration_calculator.py:287` | **QA-2 + QA-3** above. |
| Feature use of trigger words | `_voice_backends/bark.py:17` (forward design intent: "becomes a thin shim that always pre-pins"), `_voice_backends/kokoro.py:11` (forward design intent), `batch_ltx_render.py:1709` ("v0.9 legacy path" -- LTX model version branching, not back-compat), `otr_post_upscale_procgen_blend.py:284` (documents non-green-overlay branch as longstanding default), `otr_save_to_episode_workspace.py:70/75` (documents BUG-LOCAL production directory layout), `batch_procedural_sfx.py:199` (ProcSFX cue tag-matching feature), `_otr_cast_*.py` (cast alias detection feature), `story_orchestrator.py` (char_id alias map -- defined feature) | Not back-compat. Trigger-word match only. Out of cleanbreak scope. |
| Operational error messages | `video_composite.py:2233/2319` ("Set strict_c7=False to allow the legacy fallback chain") -- inside `if strict_c7: raise` paths. The strict_c7 flag itself is a back-compat surface (controls runtime fallback to AAC re-encoding); the cited lines are operator-facing error messages explaining the toggle. Future sprint candidate: "C7 strict-mode default flip." | Out of S26 scope (audio path; requires soak). Worth naming as a deferred sprint, not a S26 audit miss. |
| Test fixtures of forensic / feature references | `tests/test_news_history_ttl.py` (tests news history's LEGITIMATE legacy-file fallback feature, BUG-LOCAL-090), `test_legacy_contract_retired.py:225` (positive test that asserts ZERO new legacy surfaces), `test_audiogen_cache_keys.py` + `test_cache_key_mutations.py` (forensic comments about C7-deleted alias), `test_phase2b_progressive_ledger.py:262`, `test_render_flux_batch.py`, `test_lfc_freeze_cascade_orchestrator.py:358` (forensic), `test_lfc_w4_writer_polish_fn.py:78/86/91` (tests intentional `polish_fallback_to_none` design), `test_otr_video_plan.py:572` (forensic), `test_episode_assembler_offset_shift.py:99` (forensic), `test_post_upscale_procgen_blend.py:42` (positive test for ordering), `test_production_ledger.py:134/137/356` (back-compat TEST + forensic for removed shim), `test_procsfx_*.py` (feature words), `test_cast_*.py` (cast feature) | Tests of production features, forensic references, and positive assertions. Not legacy-tolerance code. |

### Sweep-pattern gap (QA-4)

`otr_legacy_audio_dir()` is the legacy audio-root function declared at `_otr_paths.py:201`. The function itself appears once in the sweep (because its `def` line contains the trigger word in `_otr_paths.py:204`). Its 14 caller sites do NOT appear in the sweep because the call expression `otr_legacy_audio_dir()` does not contain the literal trigger words `back-compat`, `legacy fallback`, `shim`, etc. These callers include:

- `_otr_ledger.py:328` -- `in_flight_ledger_path()` fallback walker
- `_otr_ledger.py:358` -- docstring reference
- `audio_enhance.py:434`
- `batch_audiogen_generator.py:33`
- `batch_bark_generator.py:33`
- `batch_humo_render.py:65, 2840`
- `batch_ltx_render.py:82, 2090`
- `scene_sequencer.py:879, 1226`
- `video_composite.py:90, 405`

All are downstream of the deferred B6 "B6 path back-compat" sprint. Not a separate cleanbreak miss -- but the deferral entry should enumerate them so the next sprint has a closed list.

---

## Detailed check against each requested criterion

### 1. Silent skips
Every A1-A4 + B1/B3/B6 surface named in the S25 ADDENDUM is accounted for: shipped, deferred with named follow-up + documented trigger, or stopped per directive §5 (B2). Sibling-pattern misses inside B6 (QA-2, QA-3) are completeness gaps within already-enumerated sweeps -- not hidden deferrals.

### 2. Incomplete migration
`production_ledger` SFX contract is the main issue (QA-1). The schema scaffold is gone, the validator-required-list is updated, the read paths in production code are dead -- but `set_sfx`, `apply_sfx_timings`, and `_merge_with_disk`'s `ROW_KEYED` still treat sfx as a managed shape. None of these have production callers; the surface is inert. Still: the contract should be fully removed in a small follow-up commit.

`scene_sequencer.py` SFX-mirror walk is the QA-6 deferred no-op (defensible).

### 3. Deferral pattern returns
**Zero new gates/shims/DeprecationWarnings introduced.** Confirmed via fresh sweep. The directive forbidding "gate + schedule" was honored: every A-section deletion is a real delete.

### 4. Disguised deferrals
Each deferral checked against the documented trigger list. All B-section deferrals pass the trigger check ((a) producers non-zero, or (b) audio-path / architectural blast radius). The borderline items (B6 video plan misc, B6 path back-compat) are honest small defers, not disguised. "Hard to update" justification appears nowhere.

### 5. Soft-passing tests
**None found.** Test deltas are real migrations -- `description -> character_description` rename in fixtures, outright deletion of tests pinning removed shims, addition of 1 positive single-tier `_find_cached` test. No assertion relaxation.

### 6. Dead-symbol residue
| Symbol | `git grep` hits | Verdict |
|--------|-----------------|---------|
| `_legacy_sort_key` | 0 | Clean. |
| `legacy_prefix` | 0 | Clean. |
| `_derive_tts_model_from_voice_preset` | 1 forensic comment | Clean. |
| `sfx_rows` | 2 hits in `production_ledger.py:810/812` -- `set_sfx` setter | **QA-1**. |
| `apply_sfx_timings` | 1 def + 1 test caller | **QA-1**. |
| `_merge_with_disk` `ROW_KEYED["sfx"]` | 1 hit | **QA-1**. |
| Path-1 legacy comment / writeback loop | 0 | Clean. |
| `"sfx": []` schema scaffold | 0 | Clean. |

### 7. Workflow fixture drift
All 8 workflow JSONs enumerated. Only `otr_scifi_16gb_full.json` references AudioGen/ProcSFX/MusicGen; widget values confirmed via `json.load`: both `OTR_BatchAudioGenGenerator id=15` and `OTR_MusicGenTheme id=14` have `widgets_values[0] = '{}'`. `OTR_WorkflowValidator id=63 pos=[-300,-300]` confirmed. ComfyUI Desktop re-save deferred per §11 = QA-7 above.

### 8. Whole-file `.py` deletion audit
Covered above.

### 9. Over-deletion
`git diff --name-status s25..s26 | grep '^D'` returns exactly one line: `tests/test_audiogen_legacy_gate.py`. No production node file deleted. Every production-node change matches the audit-results.md scope.

### 10. Regression delta
`fc baseline-known-fail-nodeids.txt final-known-fail-nodeids.txt` -- byte-identical. Pass count delta -20 = sum of intentional test deletions documented in commit messages.

### 11. ROADMAP / BUG_LOG accuracy
- CD-3 marked CLOSED matches reality (A1+A3 shipped).
- BUG-LOCAL-221 + 222 are new and accurately framed. Notably, **QA-1, QA-2, QA-3, and QA-4 are textbook instances of the lesson BUG-LOCAL-222 was trying to capture**: "when a deletion is about a shape, enumerate every code path that produces, consumes, or validates that shape." Same lesson extends to setter methods, timing methods, disk-merge preserve lists, and sibling patterns in shim audits. The Bible promotion should reference these as additional applications of the same lesson.
- BUG_LOG "Last entry" pointer correctly bumped from 220 to 222.
- Stack head pointer in ROADMAP correct.

### 12. Production bugs != cleanbreak fails
Known production failures (sync drift, LTX metadata, etc.) remain in the final known-fail set unchanged. None were touched as cleanbreak targets. Correct.

---

## Rounds remaining estimate

**1 round (small tail).**

Cleanup items left:
- **QA-1:** delete `set_sfx`, `apply_sfx_timings`, drop `"sfx"` from `_merge_with_disk::ROW_KEYED`, + the 2 test callers (~20 LOC net delete, 1 commit). Single highest-value cleanup -- completes the A3 contract removal.
- **QA-2 + QA-3:** add the two sibling-pattern sites to existing B6 deferred enumeration, or delete in lockstep with the broader B6 sweep.
- **QA-4:** enumerate the 14 `otr_legacy_audio_dir()` callers in the deferred "B6 path back-compat" entry so the next sprint has a closed list. Optionally extend the forbidden-pattern regex.
- **QA-5:** re-run strict-deprecation audit in interactive PowerShell; classify the warning; close or open a follow-up commit (~30 min).
- **QA-6:** optional -- strip the now-always-empty `_ledger_sfx` walk from `scene_sequencer.py` in advance of the B6/sequencer follow-up sprint.
- **QA-7:** ComfyUI Desktop runtime pass per §11 hand-back -- already on the queue.

None of these are blockers. The cleanbreak target shipped. The directive ("no gate + schedule") was honored. No deferral-pattern resurfacing.

The "3+ rounds" framing is too dramatic. The audit-completeness gaps are real but small, and the SFX contract inconsistency is internally contained (zero production callers, zero producers per S25 CD-3). One small follow-up commit closes QA-1 and brings the contract removal to consistent state; the rest are enumeration housekeeping inside existing deferred sprints.

---

## Sources

- `docs/s26-cleanbreak/final-qa-review.md`
- `docs/s26-cleanbreak/audit-results.md`
- `docs/s26-cleanbreak/cowork-stop-log.md`
- `docs/s26-cleanbreak/baseline-known-fail-nodeids.txt`
- `docs/s26-cleanbreak/final-known-fail-nodeids.txt`
- `docs/s26-cleanbreak/known-fail-delta.txt`
- `docs/s26-cleanbreak/forbidden-pattern-sweep.txt` (cross-verified with independent re-run)
- `docs/s26-cleanbreak/deprecation-audit.txt`
- `docs/2026-05-13-S25-qa-postmortem.md` (ADDENDUM A/B/C)
- `git log --oneline s25-musicgen-parity..s26-cleanbreak`
- `git diff --name-status s25-musicgen-parity..s26-cleanbreak`
- `git diff --stat s25-musicgen-parity..s26-cleanbreak`
- Independent `git grep` sweeps run against `s26-cleanbreak` ref directly, including the SFX-contract trace (`set_sfx`, `apply_sfx_timings`, `_merge_with_disk`) and the `otr_legacy_audio_dir()` caller graph
- `BUG_LOG.md` (BUG-LOCAL-211..222)
- `ROADMAP.md` ("CURRENT WORK -- S26 cleanbreak" section)
