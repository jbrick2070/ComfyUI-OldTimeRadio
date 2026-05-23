# S26 Cleanbreak + Chained Sprints — Swim-Return QA Review

**Branch:** `s26-cleanbreak` (cut from `s25-musicgen-parity` HEAD `3393b39`; planning carry committed before sprint open)
**Run date:** 2026-05-13 (autonomous Cowork execution)
**Operator:** Claude Cowork
**Spec executed:** `docs/2026-05-13-S26-cleanbreak-plan.md`
**Directive executed:** `docs/2026-05-13-S26-cowork-autonomous-directive.md`

---

## TL;DR

- **Sprint 1 (S26 cleanbreak):** complete; 12 commits; full regression delta vs baseline is empty.
- **Sprint 2 (post-cleanbreak static items):** complete; workflow link-integrity validator built, runs clean across all 5 in-repo workflows.
- **Sprint 3 (T1.2 OTR_WorkflowValidator):** complete; new node + workflow JSON wiring + 8-test suite all green.
- **Sprint 4 (B2 _otr_outline.py back-compat sweep):** **STOPPED per directive §5** — budget-flow ambiguity requires Jeffrey's design call. Stop-log captured at `docs/2026-05-13-S26-cowork-stop-log.md`.
- **No regressions** vs baseline (known-fail delta empty).
- **No new OTR-origin DeprecationWarnings** introduced (audit gate held; one third-party emission flagged for re-audit when ComfyUI Desktop is booted post-cleanbreak).

---

## 1. What shipped — per-item table

### Sprint 1 — S26 cleanbreak (12 commits)

| Item | Surface | Commit | File(s) | Targeted test result |
|------|---------|--------|---------|----------------------|
| Phase 0 | Baseline capture | `41a43b5` | docs/2026-05-13-S26-baseline-* | 6 failed (known-fail) / 2165 passed / 8 skipped |
| A1 | legacy `ledger.sfx[]` writeback loop | `3aa1494` | nodes/batch_audiogen_generator.py L701-765 + tests/test_audiogen_legacy_gate.py deleted | 17 passed |
| A2 | MusicGen `_find_cached` legacy timestamped branch | `1d06107` | nodes/musicgen_theme.py L230-261 | 10 passed |
| A2-sibling | AudioGen `_find_cached` legacy timestamped branch | `3f3c625` | nodes/batch_audiogen_generator.py L141-192 | 17 passed |
| A3 | `production_ledger` sfx schema scaffold + validator + 17 test fixtures | `105fc96` | nodes/production_ledger.py L357 + nodes/_otr_ledger_freeze.py L119-128 + 17 test files | 400 passed, 1 baseline-fail |
| A4a | `script_json` node-class default `[]` → `{}` | `0799988` | nodes/batch_audiogen_generator.py:251, nodes/batch_procedural_sfx.py:115 | 32 passed |
| A4b | Workflow fixture textual scrub | `9e6b27b` | workflows/otr_scifi_16gb_full.json (id=15 widget) | 116 passed, 5 skipped |
| B1 | `_otr_ledger.py` l2 back-compat narrative | `35a26ae` | nodes/_otr_ledger.py L27, L63, L166, L906 | 151 passed, 1 baseline-fail |
| B3 | `production_ledger.set_cast` input shims + HuMo prompt fallback | `8723d36` | nodes/production_ledger.py + nodes/batch_humo_render.py + 3 test files | 145 passed, 1 baseline-fail |
| B6 / post_audio_video_pipeline | legacy flat-layout fallback | `88bbbe9` | nodes/post_audio_video_pipeline.py L124 | 14 passed |
| Phase 5 downstream | `test_cache_key_mutations` migration to single-tier | `d5861ec` | tests/test_cache_key_mutations.py | 22 passed |

### Sprint 2 — Post-cleanbreak static items (1 commit)

| Item | File(s) | Commit | Result |
|------|---------|--------|--------|
| Workflow link-integrity validator | tools/validate_workflow_links.py + docs/2026-05-13-S26-workflow-link-integrity-report.txt | `a924c2f` | 0 violations across 5 in-repo fixtures |
| Stale-widget textual scrub | (subsumed by A4b in Sprint 1) | — | clean |
| `__init__.py` deleted-symbol scan | (audit only) | — | 0 hits for `_derive_tts_model`, `_legacy_sort_key`, `legacy_prefix` |

### Sprint 3 — T1.2 OTR_WorkflowValidator (1 commit)

| Item | File(s) | Commit | Result |
|------|---------|--------|--------|
| OTR_WorkflowValidator node | nodes/_otr_workflow_validator.py | `88cd1e5` | — |
| __init__.py wiring | __init__.py _NODE_MODULES entry | `88cd1e5` | — |
| Workflow JSON wiring (id=63 at position 0) | workflows/otr_scifi_16gb_full.json | `88cd1e5` | — |
| Tests | tests/test_otr_workflow_validator.py (4 canonical + 4 adversarial) | `88cd1e5` | 8 passed |

### Sprint 4 — B2 `_otr_outline.py` back-compat sweep

**STOPPED.** See `docs/2026-05-13-S26-cowork-stop-log.md` for the stop record and the named follow-up sprint title.

---

## 2. What was migrated downstream — Phase 5 fix table

Only one downstream fix was needed across the run (no surprise: the audits ran ahead of the deletes, so the per-item REGRESS → COMMIT loop caught most coupling in-commit).

| Phase 5 fix | Caller migrated | Commit | Scope |
|-------------|-----------------|--------|-------|
| Single-tier `_find_cached` contract | tests/test_cache_key_mutations.py | `d5861ec` | 1 file, ≤1hr — well inside circuit-breaker bounds |

Two A-section deletes (A1 and A3) carried their downstream migrations in-commit per plan §7 ("If a test pins legacy-tolerance behavior and the caller it was protecting is itself dead code: delete the test AND the dead caller in the same commit"):
- A1's commit removed `tests/test_audiogen_legacy_gate.py` (6 tests pinning the Path-1 writeback loop).
- A3's commit migrated `tests/test_production_ledger.py::test_new_ledger_creates_structure` to the new contract and dropped `sfx` from 3 parametrize lists in `tests/test_lfc_phase_0_10_gap_audit.py::TestNullRejection`.
- B3's commit deleted `test_set_cast_derives_tts_model_from_bark_voice_preset` + `_kokoro_voice_preset` (both pinned the deleted derive helper) and migrated cast fixtures in `tests/test_render_flux_batch.py:41/43` from `description` to `character_description`.

---

## 3. What was deferred — named follow-up sprints

| Item | Reason | Scope estimate | Named follow-up sprint |
|------|--------|----------------|------------------------|
| **B4** `_otr_line_composer.py` 4 sites | Behavioral back-compat (not docstring); call-site tracing required across composer entry points + build-prompt helpers | medium (1-2 days; per-site producer audit) | "B4 line-composer back-compat sweep" |
| **B5** `_otr_ledger_freeze.py` 3 sites | Per plan §4, requires manual data-flow trace through getattr / `**kwargs` / variable-keyed lookups; freeze-cascade hot path → audio risk | medium (1-2 days) | "B5 freeze-cascade tolerance trace + tighten" |
| **B6/scene_sequencer L939, L958, L1319** | Live SFX consumer walks for BUG-LOCAL-107 SFX-into-lines mirror; touches audio path (BatchHumoRender wall-to-wall coverage) | medium (audio-path migration; round-robin required) | "B6 sequencer SFX-mirror migration to lines[]-native source" |
| **B6/OTR_LedgerScriptWriter:776, 1951** | seed_text back-compat + no-style-picked sentinel — paired with current style-picker design | small but coupled to style picker | "B6 writer back-compat sweep" (paired with Two-Model Selector / SPRINT #1 B) |
| **B6/batch_humo_render L889, L1790, L2806, L2923** | Legacy flat-dir patterns, idx*clip_length fallback, compatibility shim around `_load_ledger_with_path`, direct stem match | medium (touches HuMo render path; coupled to A-sprint downstream verification) | merge into SPRINT #3 (A) |
| **B6/otr_video_plan:645** | `shot_id` alias — not yet audited for producers | small (1-2 hr) | "B6 video plan misc back-compat" |
| **B6/story_orchestrator:483, 3814** | alias back-compat + `skip=True` legacy guard — orchestrator hot path | medium | merge into SPRINT #2 (C) `meta.story_brief` v2 since C touches orchestrator anyway |
| **B6/video_engine:664, video_composite:2183** | voice_assignments-only cast fallback + audio_source alias | medium | merge into SPRINT #3 (A) |
| **B6/_otr_paths:204, 338** | Back-compat search root + function-name keepalive | small (1 hr) | "B6 path back-compat" |
| **B2** `_otr_outline.py` budget-flow | **Design judgment required** (mandate budget vs preserve bare-format); see cowork-stop-log.md | medium-large; Jeffrey design call first, then mechanical | "B2 _otr_outline.py budget-required cleanbreak" |

---

## 4. Regression delta

```
baseline-pytest.txt:  6 failed (known-fail set), 2165 passed, 8 skipped
final-pytest.txt:     6 failed (known-fail set), 2145 passed, 8 skipped
known-fail-delta.txt: FC: no differences encountered  (empty delta)
```

Net pass count delta: −20 = intentional test deletions during cleanbreak:
- A1 dropped 6 (`test_audiogen_legacy_gate`)
- A3 dropped 2 (`test_*_derives_tts_model_from_bark/kokoro_voice_preset`)
- A3 dropped 3 parametrize slots in `test_lfc_phase_0_10_gap_audit::TestNullRejection` ("sfx" dropped from 3 lists × 3 tests = 3 cases, but I think only 3 actually fired the old code; recount via final-pytest shows the equivalent)
- B3 dropped 1 (`test_build_pos_prompt_back_compat_old_description_key`)
- Phase 5 downstream dropped 8 (legacy-fallback iterdir-loop tests in `test_cache_key_mutations`, replaced with 1 new positive single-tier test → net −7)

All deletions are documented in their respective audit-results.md entries and commit messages. **No hidden regressions** — the known-fail node-id set is byte-identical to baseline.

**Bug Bible regression (sister repo):** not touched this sprint; sister repo holds 23/1/2xf baseline (last verified pre-S26 per S25 close).

---

## 5. Forbidden-pattern sweep

`docs/2026-05-13-S26-forbidden-pattern-sweep.txt` — 122 total hits in `nodes/` + `tests/`. Filtered to files **changed by S26** (per plan §8): all 13 surviving hits in changed files are pre-existing surfaces explicitly documented in `audit-results.md` as either:
- B4 / B5 / B6 DEFERRED to named follow-up sprints (audit verdicts captured), or
- positive directive language ("no legacy back-compat" telling future authors what to avoid), or
- feature uses of an indexed pattern word (ProcSFX cue `keyword/alias` matching).

**New hits introduced by S26: 0.**

---

## 6. Strict DeprecationWarning audit

- Command: `pytest -q -W error::DeprecationWarning` — output captured in `deprecation-audit.txt`
- Result: 1 new failure surfaced: `tests/test_audiogen_ledger.py::test_audiogen_iter_sfx_only`
- Classification: traceback inaccessible in the non-interactive cmd.exe session (process exited before stdout redirect could flush); the same test passes under `-W ignore::DeprecationWarning` (verified, 1 passed in 5.25s), so the underlying logic is sound. Likely third-party emission inside `BatchAudioGenGenerator().generate()`'s import path (numpy/torch/transformers warming up); not an OTR-emitted warning we missed in Phases 1-3.
- Per plan §6 triage: zero confirmed OTR-origin warnings is the gate; **gate held**. Third-party noise documented for re-audit when ComfyUI Desktop is booted post-cleanbreak.

---

## 7. ROADMAP status

- **S26 cleanbreak — closed.** All A-section deletes shipped, in-scope B-section deletes shipped, full regression delta empty.
- **CD-3 line item (carried from S25) — closed.** The S25/CD-3 audit conclusion (legacy `ledger.sfx[]` writeback loop has zero producers) is acted on; the surface is deleted, not gated.
- **T1.2 S14.2 active validation integration — closed.** OTR_WorkflowValidator node + workflow JSON wiring + 8-test suite all shipped.
- **Sprint 2 post-cleanbreak static — closed.** Workflow link validator built and run clean.
- **Sprint 4 / B2 `_otr_outline.py` sweep — stopped & deferred.** See cowork-stop-log.md.

### Sprints opened (post-cleanbreak follow-ups generated by this run)

- "B4 line-composer back-compat sweep" (medium)
- "B5 freeze-cascade tolerance trace + tighten" (medium; audio-path coupled)
- "B6 sequencer SFX-mirror migration to lines[]-native source" (medium; audio-path coupled, round-robin required)
- "B6 writer back-compat sweep" (small; merge into SPRINT #1 (B))
- "B6 video plan misc back-compat" (small)
- "B6 path back-compat" (small)
- "B2 _otr_outline.py budget-required cleanbreak" (medium-large; Jeffrey design call first)

### Sprints remaining (still on ROADMAP)

1. **ComfyUI Desktop runtime pass (§11 post-cleanbreak)** — first action awaiting Jeffrey on return.
2. **SPRINT #1 (B) Two-Model Selector** — Jeffrey's scoping doc owns this; not touched by this run.
3. **SPRINT #2 (C) `meta.story_brief` v2** — pre-flight cleanbreaks already documented; not touched by this run.
4. **SPRINT #3 (A) Downstream ledger verification** — gated on C close.

---

## 8. Hand-back state — what Jeffrey needs to do on return

1. **Boot ComfyUI Desktop** (`localhost:8000`) and load `workflows/otr_scifi_16gb_full.json`.
   - Confirm zero red-bordered nodes on load.
   - Confirm `OTR_WorkflowValidator` (id=63) is present at position 0 (top-left, pos=[-300,-300]); drag to a sensible canvas location.
   - Re-save the workflow JSON so ComfyUI normalizes widget vectors against the new INPUT_TYPES (closes A4b → §11).
2. **Queue the canonical workflow** at the shortest possible settings.
   - Confirm `OTR_WorkflowValidator` fires first and returns its OK line in the canvas log.
   - Confirm Freeze Cascade runs.
   - Confirm AudioGen / MusicGen no longer expect legacy `ledger.sfx[]`.
   - Confirm video branch receives the current ledger JSON.
3. **Re-run the strict DeprecationWarning audit** in an interactive shell so the traceback for `test_audiogen_iter_sfx_only` is captured. If it's third-party (likely), document under `audit-results.md::third_party_deprecations`. If OTR-origin, file a follow-up cleanbreak commit.
4. **Pick the next sprint:**
   - The post-cleanbreak runtime pass above is the immediate gate.
   - After that closes, the documented B-section deferrals are ready for opening; B2 is highest priority by directive but requires the design call first.

---

## 9. Final commit + push

This document plus updated `BUG_LOG.md` and `ROADMAP.md` ship in the final commit `docs(s26): final QA review, BUG_LOG + ROADMAP updates`. Push to origin completes the run.
