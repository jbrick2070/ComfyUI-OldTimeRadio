# S34 — Final QA Review (P0/P1 Hotfix)

> **Status:** CLOSED 2026-05-15.
> **Branch:** `s34-p0-p1-hotfix` @ B-final (pending push).
> **Parent:** `s33-editor-only-cleanup` @ B6 (`0297af7`, S33 close).
> **Plan:** `docs/2026-05-15-S34-p0-p1-hotfix-sprint-plan.md`.
> **Runtime status:** NOT PROVEN. Pytest-only structural pass; ComfyUI Desktop smoke deferred by explicit operator decision.

---

## Summary

S34 is a lean four-commit hotfix sprint. Two defects surfaced by S33's post-close round-robin (both verified against actual code state during S34 planning):

* **P0 (HIGH):** `run_script_doctor` in `nodes/_otr_ledger_reviewer.py` silently fail-softed on LLM exception / JSON parse failure / schema validation failure — three return sites all called `ScriptDoctorReport()` which defaulted to `overall_verdict="clean"`, so the caller committed the candidate as if the doctor had succeeded. S33's phantom-ship policy assumed Phase 2 fails LOUD; it didn't. Phase 1 (`audit_cast_contract`) already does this correctly via `_audit_failed_sentinel(pass_clean=False)`.
* **P1 (MEDIUM):** `meta.freeze_unload_ok` stamp landed on `led.data` inside the cascade's finally block AFTER `updated_script_json` had already been serialized. The cascade's own comment at L374 said "the next visual node can branch on the stamp" — but the JSON they received didn't contain it.

Both block Sprint C clean-start. Both are unambiguous fixes. Everything else identified during S34 planning was deferred to Sprint G (filed at B-final in ROADMAP).

**Runtime status:** NOT PROVEN. Pytest-only structural pass; ComfyUI Desktop smoke deferred by explicit operator decision.

---

## Commit table

| # | Commit | Subject | Hash |
|---|---|---|---|
| 1 | B0 | branch cut + S34 P0/P1 hotfix plan landing | `88aa5dd` |
| 2 | B1 | P0 fix -- Script Doctor hard-fails on malformed output (needs_full_rerun, matches Phase 1 pattern) | `7ce9b49` |
| 3 | B2 | P1 fix -- freeze_unload_ok stamp now visible to downstream JSON consumers (reserialize post-finally) | `d7099ce` |
| 4 | B-final | Sprint S34 close -- P0/P1 shipped, runtime NOT PROVEN, Sprint G queued for comprehensive bug sweep | (this commit) |

---

## Acceptance table (per plan)

| # | Check | Target | Result |
|--:|---|---|---|
| 1 | Canonical pytest count | green | green |
| 2 | Wide pytest walk | 2150 / 10 / 0 or higher, +10 new (B1: 7, B2: 3) | **2150 / 10 / 0** in 18.41s (+10 over S33 close) |
| 3 | Bug Bible regression | 23 / 1 / 2 | held at every commit boundary |
| 4 | Audio C7 byte-identical (pytest proxy, default-config happy path) | holds B1 → B-final | held at B1 and B2 boundaries |
| 5 | Forbidden sweep | 0 runtime hits at every boundary | 0 hits |
| 6 | P0: Script Doctor returns `needs_full_rerun` on LLM exception | ✅ (B1) | ✅ `test_script_doctor_llm_exception_returns_needs_full_rerun` |
| 7 | P0: Script Doctor returns `needs_full_rerun` on JSON parse failure | ✅ (B1) | ✅ `test_script_doctor_invalid_json_returns_needs_full_rerun` |
| 8 | P0: Script Doctor returns `needs_full_rerun` on schema validation failure | ✅ (B1) | ✅ `test_script_doctor_schema_validation_returns_needs_full_rerun` |
| 9 | P0: Script Doctor docstring no longer contains exact stale phrases `'fail-soft on the doctor'` AND `'overall_verdict="clean"'` | ✅ (B1) | ✅ `test_script_doctor_docstring_replaces_stale_phrases` |
| 10 | P0: `review_ledger` propagates doctor failure to `verdict="needs_full_rerun"` without committing candidate | ✅ (B1) | ✅ `test_review_ledger_doctor_failure_returns_needs_full_rerun_without_commit` (spies on `apply_doctor_edits`) |
| 11 | P0: Cascade maps reviewer `needs_full_rerun` through `REVIEWER_TO_FREEZE_VERDICT` to returned freeze verdict | ✅ (B1) | ✅ `test_freeze_cascade_maps_reviewer_needs_full_rerun_to_freeze_verdict` (verifies map + terminal-failure set) |
| 12 | P1: `meta.freeze_unload_ok` stamp visible in returned `updated_script_json` | ✅ (B2) | ✅ `test_freeze_unload_ok_stamp_in_returned_json` |
| 13 | P1: Stamp matches actual unload outcome (True on success, False on failure) | ✅ (B2) | ✅ `test_freeze_unload_failure_stamp_in_returned_json` + `test_freeze_unload_stamp_consistent_with_led_data` |
| 14 | No Sprint C surface touched | both checks pass | **file-surface:** 0 hits (`git diff --name-only s33-editor-only-cleanup..HEAD` lists nothing matching `OTR_LedgerScriptWriter.py` / `_otr_reflection.py` / `_otr_visual_prompt_coercion.py`); **content-surface:** 0 hits across `nodes/*.py` + `workflows/*.json` for `meta.story_brief` / `meta.ltx_style_brief` |
| 15 | Happy-path audio C7 byte-identical | YES | held; P0+P1 only change failure paths + add post-finally serialization |
| 16 | ROADMAP refreshed (S34 closed; Sprint G entry filed) | ✅ | ✅ (B-final) |
| 17 | Runtime status NOT PROVEN line present in final QA | ✅ (B-final) | ✅ (Summary section above) |

---

## Gate run details

All pytest runs used the Windows venv. Output captured via background-detached `start /MIN` + log-file readback.

| Boundary | Suites run | Pass / Skip / xfail | Notes |
|---|---|---|---|
| B1 | B1 hardfail proof + phase3 reviewer + cascade orchestrator + audio | 72 / 1 / 0 (73 collected) | Audio C7 holds |
| B2 | B2 visibility proof + B1 carryover + LFC B1 finally + cascade orchestrator + audio | 38 / 1 / 0 (39 collected) | Audio C7 holds |
| B-final | Wide pytest walk (`tests/` collect-all) | **2150 / 10 / 0** in 18.41s | +10 over S33 close baseline (2140 / 10 / 0) |
| B-final (gates) | forbidden sweep + legacy audit + naming + workflow JSON + phase extinction + Bug Bible regression | 95 passed / 6 skipped / 2 xfailed | Bug Bible 23/1/2xf held |

---

## Deviations from plan

None. All four commits landed per the plan's commit subjects:

* `B0: branch cut + S34 P0/P1 hotfix plan landing`
* `B1: P0 fix — Script Doctor hard-fails on malformed output (needs_full_rerun, matches Phase 1 pattern)`
* `B2: P1 fix — freeze_unload_ok stamp now visible to downstream JSON consumers (reserialize post-finally)`
* `B-final: Sprint S34 close — P0/P1 shipped, runtime NOT PROVEN, Sprint G queued for comprehensive bug sweep`

Test counts matched: 7 B1 tests + 3 B2 tests = 10 new = exact target +10 vs S33 baseline.

---

## Files touched (S34 in total)

| File | Change kind | Commits |
|---|---|---|
| `docs/2026-05-15-S34-p0-p1-hotfix-sprint-plan.md` | NEW (plan) | B0 |
| `nodes/_otr_ledger_reviewer.py` | docstring rewrite + 3 return-site edits in `run_script_doctor` | B1 |
| `tests/test_script_doctor_hardfail.py` | NEW (7 tests) | B1 |
| `nodes/OTR_LedgerFreezeCascade.py` | reserialization block inserted between finally and return | B2 |
| `tests/test_cascade_freeze_unload_visible.py` | NEW (3 tests) | B2 |
| `ROADMAP.md` | S34 CURRENT WORK section + Sprint G QUEUED entry | B-final |
| `docs/2026-05-15-S34-final-qa-review.md` | NEW (this document) | B-final |

No Sprint C surface touched (file-surface and content-surface checks both clean).

---

## VRAM / runtime impact

* **B1 P0:** unchanged on the happy path (doctor succeeds, returns the schema-validated `ScriptDoctorReport` it has always returned). On failure paths the candidate is no longer committed corrupted; the cascade returns `needs_full_rerun` instead. No additional LLM calls, no model reload.
* **B2 P1:** adds one `json.dumps(led.data, ...)` call between the finally block and the return. This is the same call already in the cascade body (lines ~346-348) but on a post-stamp `led.data`. Pure dict/string work, no torch tensors. Negligible runtime impact.

---

## BUG_LOG entries filed during S34

None. The two defects fixed in S34 were surfaced by S33's post-close round-robin and described in the S34 plan's "Why this sprint exists" section. They were never reproduced in a soak run (the silent fail-soft would only appear under LLM crash or hallucination conditions; the stamp-invisibility was a passive integration gap, not a runtime failure).

---

## Forward work (deferred to Sprint G — filed at this B-final)

See `ROADMAP.md` "Sprint G — Comprehensive bug sweep + cosmetic cleanup" entry for the full deferred-items list (KNOWN-DEFECT items from S33 forward-work + AUDIT-DRIVEN items that need a B1 inventory pass to enumerate). Sprint G sequencing: after Sprint A or whenever Jeffrey calls. May split into G1/G2 if scope warrants.

**Why Sprint G is deferred rather than fixed now:** Sprint C touches the writer's script-finalization area and LTX consumer code; cosmetic items in that surface get rewritten by Sprint C anyway, so fixing them now is wasted work. Sprint G after Sprint C closes can absorb whatever Sprint C didn't touch, cleanly.

---

## Optional operator action (Jeffrey's discretion)

Between S34 B-final and Sprint C kickoff, Jeffrey can OPTIONALLY run a 5-minute ComfyUI Desktop smoke test:

1. Load canonical `otr_scifi_16gb_full.json`
2. Run one short episode through writer → freeze cascade → audio path
3. Confirm no runtime errors, audio output reaches file

This is **not** a Cowork autonomous task and **not** a Sprint commit gate. If anything breaks, hotfix on `s34-p0-p1-hotfix` branch before Sprint C kickoff.

---

## Sources

* `docs/2026-05-15-S34-p0-p1-hotfix-sprint-plan.md` (sprint plan)
* `docs/2026-05-15-S33-final-qa-review.md` (parent sprint close)
* `nodes/_otr_ledger_reviewer.py`, `nodes/OTR_LedgerFreezeCascade.py` (source of truth for the P0 + P1 fixes)
* `tests/test_script_doctor_hardfail.py`, `tests/test_cascade_freeze_unload_visible.py` (new S34 tests)
* `ROADMAP.md` (S34 CURRENT WORK section + Sprint G entry)
