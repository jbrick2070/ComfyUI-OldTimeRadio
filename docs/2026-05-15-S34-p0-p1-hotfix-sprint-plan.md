# S34 — P0/P1 hotfix sprint (Cowork loop, pytest-only)

> **What this is:** Lean hotfix sprint. Fix the two real defects surfaced by S33's post-close round-robin: Script Doctor silent fail-soft (P0) + freeze_unload_ok stamp invisibility (P1). Nothing else. Cosmetic cleanup and comprehensive bug sweep deferred to a later sprint per Jeffrey's "don't fix too much because Sprint C will rebreak it" principle.

**Status:** PLANNED.
**Branch:** `s34-p0-p1-hotfix`. Cut from `s33-editor-only-cleanup` @ B6 (`0297af7`).
**Sequencing:** S33 ✅ → **S34 (this sprint)** → Sprint C → Sprint E (queued) → Sprint A → Sprint G (queued — comprehensive bug sweep, see deferral list below).
**Loop per commit:** review → code → wire → pytest → commit → push. No ComfyUI execution. No operator gates.

---

## Why this sprint exists

Two defects verified against actual code state during S34 round-robin:

**P0 (HIGH) — Script Doctor swallows all failures.** `nodes/_otr_ledger_reviewer.py:813-842`. Three failure paths return `ScriptDoctorReport()` with default `overall_verdict="clean"`. Caller commits the candidate as if doctor succeeded. **S33's phantom-ship policy assumed Phase 2 fails LOUD; it doesn't.** Phase 1 already does this correctly via `_audit_failed_sentinel(pass_clean=False)`. Phase 2 needs to match.

**P1 (MEDIUM) — `meta.freeze_unload_ok` invisible to JSON consumers.** `nodes/OTR_LedgerFreezeCascade.py:340-410`. Sequence: serialize → finally block stamps `freeze_unload_ok` → return pre-stamp string. Stamp lands on `led.data` AFTER serialization. The comment at line 374-376 explicitly says "the next visual node can branch on the stamp," but the JSON they receive doesn't contain it.

Both block Sprint C clean-start. Both are unambiguous fixes. Anything else is deferred to Sprint G.

---

## Hard rules

1. **Audio C7 byte-identical pytest proxy** must hold on happy path at every commit boundary. P0 changes the FAILURE-path behavior (clean → needs_full_rerun), not happy path; if happy-path audio regresses, something accidental happened.
2. **No legacy back-compat reintroduced.**
3. **No new generate or lifecycle surfaces.**
4. **No widgets.**
5. **Bug Bible regression** 23/1/2xf at every commit boundary.
6. **Forbidden-pattern sweep** stays at 0 runtime hits.
7. **Sprint G scope (see below) is strictly out of bounds** for S34. Cowork tagging any non-P0/P1 finding during this sprint goes to ROADMAP, not into a new commit.
8. **Sprint C surface untouched** — pure deferral filter.

---

## Canonical pytest run

Same as S33. Pre-S34 baseline: wide walk **2140 / 10 / 0** (from S33 B6 close). Target post-S34: **2150 / 10 / 0** (10 new tests).

---

## Commit structure

4 commits total: B0 + B1 + B2 + B-final.

### B0 — branch cut + plan landing (~0.25 d)

**Review.** Confirm parent `s33-editor-only-cleanup` @ `0297af7`. Confirm clean working tree.

**Code.** Cut `s34-p0-p1-hotfix` branch. Land this plan at `docs/<date>-S34-p0-p1-hotfix-sprint-plan.md`.

**Commit subject.** `B0: branch cut + S34 P0/P1 hotfix plan landing`

---

### B1 — P0 fix: Script Doctor hard-fails on malformed output (~0.5 d)

**Review.** Read `nodes/_otr_ledger_reviewer.py:775-843`. Confirm the three soft-fail return statements (lines 824, 834, 842). Confirm Phase 1's `_audit_failed_sentinel` pattern (lines 146-164, 435-455) as the reference architecture.

**Code.**

Change each of the three `return ScriptDoctorReport()` calls in `run_script_doctor` to:

```python
return ScriptDoctorReport(overall_verdict="needs_full_rerun")
```

(`needs_full_rerun` is already in the `ScriptDoctorReport.overall_verdict` Literal type per line 179. The cascade already has consumer logic for it per line 1084.)

Update the docstring (lines 781-788) to:

```python
"""One LLM call. Returns ScriptDoctorReport.

On LLM / JSON / schema failure, returns a report with
overall_verdict="needs_full_rerun" so the caller can branch
on the failure (cascade routes to needs_full_rerun verdict;
caller decides whether to retry the writer or surface the
failure loud).

S33 B3 + B4 retired Pass 3 post-audit and Step 2.5 phantom-
skip fallback. The doctor IS the final structural pass; it
must therefore fail loud with needs_full_rerun so downstream
commits don't ship corrupted candidates. S34 B1 corrected the
prior fail-soft behavior that S33 had assumed was already
loud (which it wasn't).
"""
```

**Wire.** None.

**Pytest.**

| Test | File | Asserts |
|---|---|---|
| `test_script_doctor_llm_exception_returns_needs_full_rerun` | `tests/test_script_doctor_hardfail.py` (new) | Mock `generate_fn` to raise; assert `result.overall_verdict == "needs_full_rerun"` |
| `test_script_doctor_invalid_json_returns_needs_full_rerun` | same | Mock returning non-JSON text; assert verdict |
| `test_script_doctor_schema_validation_returns_needs_full_rerun` | same | Mock returning valid JSON but invalid schema; assert verdict |
| `test_script_doctor_empty_output_returns_needs_full_rerun` | same | Mock returning empty string; assert verdict |
| `test_script_doctor_docstring_replaces_stale_phrases` | same | Read docstring; assert exact stale phrase `'fail-soft on the doctor'` is ABSENT and exact stale phrase `'overall_verdict="clean"'` is ABSENT; assert `'needs_full_rerun'` is PRESENT |
| `test_review_ledger_doctor_failure_returns_needs_full_rerun_without_commit` | same | Mock `generate_fn` so `run_script_doctor` hits a failure path; call `review_ledger`; assert `ReviewerDisposition.verdict == "needs_full_rerun"` AND `apply_doctor_edits` NOT invoked |
| `test_freeze_cascade_maps_reviewer_needs_full_rerun_to_freeze_verdict` | same | Mock `review_ledger` to return a valid `ReviewerDisposition(verdict="needs_full_rerun", ...)` populated with all dataclass-required fields (read the dataclass definition at test-write time — post-S33 it has 6 fields including `verdict`, `edits_applied`, etc.); call cascade's run logic against a valid freeze-cascade ledger fixture; assert `return_tuple[4] == "needs_full_rerun"` (the cascade's `freeze_verdict` return slot at output index 4) |
| `test_audio_c7_byte_identical_b1` | `tests/test_audio_byte_identical.py` | EXISTING reused canary. Happy-path audio byte-identical. |

**Commit gate.** 7 new tests green + audio canary reused. Canonical pytest subset green. Bug Bible 23/1/2xf. Audio C7 happy-path holds. Forbidden sweep clean.

**Commit subject.** `B1: P0 fix — Script Doctor hard-fails on malformed output (needs_full_rerun, matches Phase 1 pattern)`

---

### B2 — P1 fix: freeze_unload_ok stamp visibility (~0.25 d)

**Review.** Read `nodes/OTR_LedgerFreezeCascade.py:340-410`. Confirm `updated_script_json` serialized at line 346, `freeze_unload_ok` stamped on `led.data` at line 393 inside the finally block, return at line 407 uses the pre-stamp string.

**Code.**

After the finally block (after line 397), before the return at line 405, add:

```python
# S34 B2: reserialize led.data so freeze_unload_ok stamp (set
# in the finally block above) is visible to downstream JSON
# consumers. The earlier serialization at L346 happened before
# the stamp; without this reserialization, the comment at L374
# claiming "the next visual node can branch on the stamp" is
# false because the JSON doesn't contain it.
try:
    updated_script_json = json.dumps(
        led.data, indent=2, ensure_ascii=False,
    )
except Exception as exc:  # noqa: BLE001
    log.warning(
        "[OTR_LedgerFreezeCascade] failed to reserialize "
        "post-unload ledger to JSON (%s); freeze_unload_ok "
        "stamp may not reach downstream consumers.", exc,
    )
    # Keep the pre-finally serialization as best-effort fallback
```

**Wire.** None.

**Pytest.**

| Test | File | Asserts |
|---|---|---|
| `test_freeze_unload_ok_stamp_in_returned_json` | `tests/test_cascade_freeze_unload_visible.py` (new) | Run cascade with mocked `unload_llm`; parse returned `updated_script_json`; assert `meta.freeze_unload_ok` present and matches stamp |
| `test_freeze_unload_failure_stamp_in_returned_json` | same | Mock `unload_llm` to raise; verify returned JSON has `freeze_unload_ok=False` (not missing) |
| `test_freeze_unload_stamp_consistent_with_led_data` | same | Assert returned JSON's `meta.freeze_unload_ok` matches `led.data["meta"]["freeze_unload_ok"]` post-finally |
| `test_audio_c7_byte_identical_b2` | `tests/test_audio_byte_identical.py` | EXISTING reused canary. Happy-path audio byte-identical. |

**Commit gate.** 3 new tests green + audio canary reused. Canonical pytest subset green. Bug Bible 23/1/2xf. Audio C7 holds. Forbidden sweep clean.

**Commit subject.** `B2: P1 fix — freeze_unload_ok stamp now visible to downstream JSON consumers (reserialize post-finally)`

---

### B-final — sprint close (~0.25 d)

**Review.** Mirror prior sprint final QA format.

**Code.** Final QA review at `docs/<date>-S34-final-qa-review.md`. ROADMAP refresh — mark S34 closed, file Sprint G with the deferred items list below.

**Mandatory final QA section:** the QA review document MUST include a "Runtime status" line in its Summary or Gate section reading exactly:

> Runtime status: NOT PROVEN. Pytest-only structural pass; ComfyUI Desktop smoke deferred by explicit operator decision.

**Wire / Pytest.** Wide pytest walk: confirm 2150 / 10 / 0 or higher (+10 expected new tests unless unrelated skips/counts shift).

**Commit gate.** All S34 acceptance rows green. Audio C7 held at B1 + B2 boundaries. Runtime-status line present in final QA. Sprint G ROADMAP entry filed. Branch pushed.

**Commit subject.** `B-final: Sprint S34 close — P0/P1 shipped, runtime NOT PROVEN, Sprint G queued for comprehensive bug sweep`

---

## Acceptance table

| # | Check | Target |
|--:|---|---|
| 1 | Canonical pytest count | green |
| 2 | Wide pytest walk | 2150 / 10 / 0 or higher, with +10 expected new tests (B1: 7, B2: 3) unless unrelated skips/counts shift |
| 3 | Bug Bible regression | 23 / 1 / 2 |
| 4 | Audio C7 byte-identical (pytest proxy, default-config happy path) | holds B1 → B-final |
| 5 | Forbidden sweep | 0 runtime hits at every boundary |
| 6 | **P0: Script Doctor returns `needs_full_rerun` on LLM exception** | ✅ (B1) |
| 7 | **P0: Script Doctor returns `needs_full_rerun` on JSON parse failure** | ✅ (B1) |
| 8 | **P0: Script Doctor returns `needs_full_rerun` on schema validation failure** | ✅ (B1) |
| 9 | **P0: Script Doctor docstring no longer contains exact stale phrases** `'fail-soft on the doctor'` AND `'overall_verdict="clean"'` | ✅ (B1) |
| 10 | **P0: `review_ledger` propagates doctor failure to `verdict="needs_full_rerun"` without committing candidate** | ✅ (B1) |
| 11 | **P0: Cascade maps reviewer `needs_full_rerun` through `REVIEWER_TO_FREEZE_VERDICT` to returned freeze verdict** | ✅ (B1) |
| 12 | **P1: `meta.freeze_unload_ok` stamp visible in returned `updated_script_json`** | ✅ (B2) |
| 13 | **P1: Stamp matches actual unload outcome (True on success, False on failure)** | ✅ (B2) |
| 14 | No Sprint C surface touched. Two grep checks, both must pass (using bash `!` inversion so 0 = clean). **File-surface:** `! git diff --name-only s33-editor-only-cleanup..HEAD \| grep -E "nodes/OTR_LedgerScriptWriter.py\|nodes/_otr_reflection.py\|nodes/_otr_visual_prompt_coercion.py"`. **Content-surface (path-scoped to runtime files only — docs/ excluded since the plan itself references these strings):** `! git diff s33-editor-only-cleanup..HEAD -- 'nodes/**/*.py' 'workflows/**/*.json' \| grep -E "meta\.story_brief\|meta\.ltx_style_brief\|meta\.style"`. Note: `_otr_reflection.py` doesn't exist yet (Sprint C creates it); file-surface check is a forward-lock against accidental Sprint-C-zone creation. | both checks pass |
| 15 | Happy-path audio C7 byte-identical | YES (P0+P1 only change failure paths + add post-finally serialization) |
| 16 | ROADMAP refreshed | S34 marked closed; Sprint G entry filed |
| 17 | Runtime status NOT PROVEN line present in final QA | ✅ (B-final) |

---

## Out of scope (deferred to Sprint G — comprehensive bug sweep)

S34 is hotfix-only. The following items were identified during S34 planning but are deferred to a future "Sprint G — comprehensive bug sweep" so S34 stays lean and Cowork doesn't get pulled into broad maintenance work that Sprint C may re-touch anyway.

**Sprint G ROADMAP entry (file at S34 B-final):**

```
Sprint G — Comprehensive bug sweep + cosmetic cleanup
Status: QUEUED. Position: after Sprint A or whenever Jeffrey calls.
Scope:

KNOWN-DEFECT items (from S33 forward-work, verified during S34 planning):
- phase_1_2_9_reviewer_composite phase_name string references retired "9";
  resolve via rename-with-consumer-updates OR documented retention with
  telemetry constraint + regression test.
- post_audit_violations ReviewerDisposition field always 0 post-S33 B2;
  remove field after AST sweep proves no constructor passes it as kwarg.
- OTR_LedgerScriptWriter.py Phase 3 + Step 2.5 comment refs (non-Sprint-C
  zone).
- _otr_ledger_consumers.py:87 "set by Step 2.5" stale comment.

AUDIT-DRIVEN items (B1 inventory pass needed to enumerate):
- Fail-soft pattern audit (find other try/except returning Default()
  that may silently swallow failures the way Script Doctor did).
- Comment/docstring drift for deleted code across S31/S31.5/S32/S33.
- Stale __all__ entries across all nodes/*.py modules.
- Stale imports referencing deleted modules.
- Workflow JSON inventory across all workflows/*.json files (beyond
  otr_scifi_16gb_full.json which has been audited).
- Forbidden-sweep regex coverage gaps + add narrow marker
  `return\s+ScriptDoctorReport\s*\(\s*\)` to lock S34 B1 against
  reintroduction.
- ADR drift (docs/script-writing-architecture-adr.md post-S33 accuracy).
- BUG_LOG hygiene (stale OPEN entries, hash mismatches).
- ROADMAP CURRENT WORK accuracy.
- Test name drift (commit-hash-specific test filenames).
- Stale # noqa / # type: ignore comments orphaned by deletions.
- Magic strings (top 3-5 worst offenders only).

Sequencing:
- After Sprint C (which may obsolete some of the above by rewriting
  the same code).
- May be split into G1/G2 if scope warrants.
- Round-robin reviewed before Cowork execution per established pattern.
```

**Why Sprint G is deferred rather than fixed now:**

- Sprint C touches the writer's script-finalization area and LTX consumer code. Cosmetic items in that surface get rewritten by Sprint C anyway; fixing them now is wasted work.
- Sprint C's plan refresh after S34 close will catch any items in its own blast radius.
- Sprint G after Sprint C closes can absorb whatever Sprint C didn't touch, cleanly.

---

## Optional operator action (Jeffrey's discretion, NOT autonomous)

Between S34 B-final and Sprint C kickoff, Jeffrey can OPTIONALLY run a 5-minute ComfyUI Desktop smoke test:

1. Load canonical `otr_scifi_16gb_full.json`
2. Run one short episode through writer → freeze cascade → audio path
3. Confirm no runtime errors, audio output reaches file

This is NOT a Cowork autonomous task and NOT a Sprint commit gate. Pure operator-discretion safety check. If anything breaks, hotfix on `s34-p0-p1-hotfix` branch before Sprint C kickoff.

---

## Sources

- `docs/2026-05-15-S33-final-qa-review.md` — parent sprint close
- `nodes/_otr_ledger_reviewer.py` — Script Doctor + Phase 1 source of truth
- `nodes/OTR_LedgerFreezeCascade.py` — freeze cascade source of truth
- Round-robin review documents (Gemini + ChatGPT, 2026-05-15)
- ROADMAP.md — Sprint G entry to be filed at B-final
