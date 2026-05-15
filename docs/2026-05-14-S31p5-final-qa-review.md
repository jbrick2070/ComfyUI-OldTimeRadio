# Sprint S31.5 -- Final QA Review

**Branch:** `s31p5-legacy-residue-cleanup`
**Cut from:** `s31-loader-clean-break` @ `2b0b6dc` (S31 B8 close)
**Closed:** 2026-05-14

## Summary

Sprint S31.5 sweeps the legacy residue revealed by S31's clean
break. Nothing functional changes; the codebase stops carrying
ghosts. Eight commits land on `s31p5-legacy-residue-cleanup`, each
pushed to origin. The headline outcomes: BUG-LOCAL-227 (25 latent
LFC failures from the wide pytest walk, surfaced at S31 B1)
closed via triage; two vestigial test files consolidated into the
canonical guard file; one delegation wrapper eliminated; stale
comments and docstrings sweep-cleaned across the LLM stack
modules.

Plan continuity from S31+S32: all 8 hard rules held at every
commit boundary (audio C7 byte-identical proxy, no legacy
back-compat reintroduced, one generate surface, Bug Bible
23/1/2xf, no version-label bumps, etc.). Pytest gates only;
ComfyUI Desktop runtime verification remains deferred per the
autonomous-run handoff.

S32 picks back up next: re-land S32 B0 fresh on
`s32-helper-per-subpass-routing` against the post-S31.5 baseline
(the orphan B0 at `655dd6a` was reverted on its branch at `4837ed7`
during S31.5 B0).

## Commit log

| # | Hash | Subject |
|---|---|---|
| B0 | `834534b` | S32 B0 revert + S31.5 branch cut + plan landing |
| B1 | `3f2dcf8` | BUG-LOCAL-227 triage -- 16+9+0 classifications + 10 S31-relocation collateral, wide walk 0 regressions |
| B2 | `fc77367` | consolidate vestigial test files (test_b4b_rss_rewire, test_unload_synchronize_guard) into test_no_orchestrator_legacy_symbols |
| B3 | `ddd3ab5` | eliminate `_vram_cleanup_via_loader` wrapper -- `register_vram_cleanup` takes `_otr_model_loader.unload_llm` directly |
| B4 | `cba7b04` | stale comment + docstring sweep -- remove references to deleted symbols, B4b history, cache-rebind workarounds |
| B5 | `ee17607` | S31.5 forbidden-pattern sweep verification -- 0 runtime hits + 1 new marker |
| B6 | `208ed58` | empty, skipped |
| B7 | (this commit) | Sprint S31.5 close -- legacy residue cleared, BUG-LOCAL-227 FIXED, codebase ghost-free |

## Acceptance table

| # | Check | Target | Actual |
|--:|---|---|---|
| 1 | Canonical pytest count | green (count may shift due to deletions) | **243 / 7 / 2 -> 232 / 7 / 2** (test counts dropped from B1 Bucket-A deletions; gate is content of tests, not count) |
| 2 | Wide `pytest tests/` walk | 0 unexpected failures | **2080 / 8 / 0** -- 35 prior unexpected failures (25 BUG-LOCAL-227 + 10 S31-relocation collateral) all resolved |
| 3 | Bug Bible regression | 23 / 1 / 2 | **23 / 1 / 2** PASS |
| 4 | Audio C7 byte-identical (pytest proxy) | holds B1 -> B7 | **PASS** at every commit boundary |
| 5 | Forbidden sweep | 0 runtime hits | **PASS** at every commit boundary |
| 6 | `tests/test_b4b_rss_rewire.py` exists | False (deleted at B2) | **PASS** (deletion + sanity-guard test) |
| 7 | `tests/test_unload_synchronize_guard.py` exists | False (deleted at B2) | **PASS** (deletion + sanity-guard test) |
| 8 | Surviving assertions from #6/#7 present in `test_no_orchestrator_legacy_symbols.py` | True | **PASS** -- importer-pattern guards, tree-wide importer guard, BUG-LOCAL-226 BUG_LOG status check, all folded |
| 9 | `_vram_cleanup_via_loader` audit complete | Outcome A OR B | **Outcome A** -- eliminated. `register_vram_cleanup`'s caller already wraps callbacks in try/except |
| 10 | Stale comments referencing `_LLM_CACHE`/`B4b`/cache-hit short-circuit | 0 in nodes/*.py runtime code | **PASS** (B4 sweep cleared) |
| 11 | `load_llm` docstring describes current (post-B4) behavior accurately | manual review confirms | **PASS** (B4 docstring rewrite) |
| 12 | BUG-LOCAL-227 | FIXED at B1 with classification log | **PASS** (`3f2dcf8`, 16A + 9B + 0C + 10 S31-collateral) |
| 13 | New sweep markers added (if any) | optional, <= 1 | **1 added** (`\b_vram_cleanup_via_loader\b` at B5) |
| 14 | ROADMAP refreshed | S31.5 marked closed | **PASS** (B7 this commit) |

## Triage breakdown (BUG-LOCAL-227)

The 25 LFC failures surfaced at S31 B1 plus 10 S31-relocation
collateral surfaced at the B1 wide-walk re-run, classified:

**Bucket A (DELETE) -- 16:**
* `tests/test_lfc_phase_3_polish_in_cascade.py` (entire file, 15 tests)
  -- Phase 3 deleted at S30 B3.
* `tests/test_lfc_w4_writer_polish_fn.py::test_polish_fallback_to_none_preserves_back_compat`
  (1 test) -- violates CLAUDE.md "no legacy back-compat" rule.

**Bucket B (REFACTOR) -- 19:**
* 4 in `test_lfc_b1_cascade_unload_in_finally.py` -- added
  `technical_model=` kwarg to `inst.run()` calls (post-S30 B3
  require_model fail-loud).
* 4 in `test_lfc_c4_news_used_passthrough.py` -- renamed
  `model_id=` kwarg to `technical_model=`.
* 1 in `test_lfc_freeze_cascade_orchestrator.py` -- updated
  INPUT_TYPES assertion from `"model_id"` to `"technical_model"`.
* 3 in `test_story_orchestrator_vram_calibration.py` -- retargeted
  source file from `story_orchestrator.py` to `_otr_model_loader.py`
  (S31 B2 ported the `>= 14.5` flagship threshold + `_MODEL_CONTEXT_CAPS`
  dict there).
* 3 in `test_bark_ledger.py` + 4 in `test_sequencer_ledger.py` --
  mock targets switched from `story_orchestrator._unload_llm` to
  `_otr_model_loader.unload_llm`.

**Bucket C (SKIP) -- 0:**
No flaky / environment-dependent failures emerged.

## Forward work

* **S32 re-land.** S32 B0 was reverted at S31.5 B0 (`4837ed7` on
  `s32-helper-per-subpass-routing`). When S32 resumes, B0 lands
  fresh against the post-S31.5 baseline (this branch's tip),
  then proceeds B1 -> B8 per the canonical S31+S32 plan.

* **Operator runtime 5080 verification.** S31 + S31.5 changes
  remain pytest-only-verified. Audio C7 byte-identical pytest
  proxy held throughout. Jeffrey runs the runtime verification
  on his own schedule after feature sprints land.

* **Loader API consolidation.** Out-of-scope for S31.5 (per
  plan's "Out of scope" section). Splitting
  `_otr_model_loader.py` (~700+ LOC post-S31) into
  `{__init__.py, bnb_profiles.py, generation.py}` is structural
  refactor, not residue cleanup. Keeps its own sprint.

* **S33 (editor-only cleanup passes)** sequences AFTER S32
  closes, per the locked sprint sequence:
  S31 -> S31.5 -> S32 -> S33.

## Sources

* Plan: `docs/2026-05-14-S31p5-legacy-residue-cleanup-sprint-plan.md`
* Branch: `s31p5-legacy-residue-cleanup` (origin synced through B7 close)
* Parent S31 close: `docs/2026-05-14-S31-final-qa-review.md`
* BUG_LOG updates: `BUG-LOCAL-227` [FIXED `3f2dcf8`].
* S32 B0 revert artifact: `4837ed7` on
  `s32-helper-per-subpass-routing`.
