# Sprint S31 -- Final QA Review

**Branch:** `s31-loader-clean-break`
**Cut from:** `s30-two-model-selector` @ `ccf583d` (S30 B8)
**Closed:** 2026-05-14

## Summary

Sprint S31 completes the legacy LLM stack clean break planned in
`docs/2026-05-14-S31-S32-cowork-execution-plan.md`. Eight content
commits + one empty B7 buffer land on `s31-loader-clean-break`,
each pushed to origin. The headline non-deferrable B4 commit
DELETED four legacy symbols (`_load_llm`, `_unload_llm`,
`_LLM_CACHE`, `_generate_with_llm`) from `nodes/story_orchestrator.py`,
ported the ~613-LOC bitsandbytes profile body to
`nodes/_otr_model_loader.load_llm` (the canonical loader surface),
simplified `_otr_model_loader.unload_llm` to drop its
orchestrator-fallback block, and added a new lifecycle helper
`invalidate_cache_no_gpu_teardown()` that fixes the
TIMEOUT_RECOVERY CUDA-race regression introduced at S30 B4b
(BUG-LOCAL-228). The four legacy symbols are hardware-locked OUT
via S31 B5 sweep markers + `hasattr` deletion guards in
`tests/test_no_orchestrator_legacy_symbols.py`.

Plan hard rule #1A (B4 deletion non-deferrable) satisfied -- no
shims live post-B4. Plan hard rule #5 (one generate surface)
satisfied -- the canonical
`request_slot(slot, model_id) -> make_generate_fn(cache_entry) ->
fn(messages, *, temperature, max_new_tokens)` is the only generate
path; preemptive sweep markers lock out `generate_text` /
`generate_with_llm` reintroduction. Plan hard rule #6 (lifecycle
helpers distinct from generate surface) satisfied -- `load_llm`,
`unload_llm`, `request_slot`, and the new
`invalidate_cache_no_gpu_teardown` form the lifecycle layer.

Pytest gates only; ComfyUI Desktop runtime verification is
deferred to post-feature-set per the autonomous-run handoff.
Audio C7 byte-identical pytest proxy stood in for the runtime
gate at every commit boundary.

## Commit log

| # | Hash | Subject |
|---|---|---|
| B0 | `1bb4a4d` | branch cut + S31+S32 Cowork execution plan landing |
| -- | `3c8118e` | docs: file BUG-LOCAL-227 for pre-existing LFC failures (Task 0) |
| B1 | `7a9584f` | caller-switch pre-work -- VRAMContextTest off legacy `_SO` symbols |
| B2 | `d8129bf` | port `_load_llm` body (~613 LOC) to `_otr_model_loader`; orchestrator `_load_llm` becomes thin shim |
| B3 | `dc26421` | refactor `_fetch_science_news` + internal callers off `_generate_with_llm` onto `request_slot + make_generate_fn` |
| B4 | `a4fe67a` | **HEADLINE** -- delete `_load_llm` + `_unload_llm` + `_LLM_CACHE` + `_generate_with_llm`; simplify `unload_llm`; fix TIMEOUT_RECOVERY CUDA race via `invalidate_cache_no_gpu_teardown` |
| B5 | `64524ef` | arm S31 extinction markers (4 deleted + 2 preemptive) |
| B6 | `bc883b1` | writer + workflow + visual residuals -- RSS slot fix, self-test drift, workflow JSON link off-by-one, VisualPromptCoercion missing-model-id loud-fail |
| B7 | `8abe271` | empty, skipped |
| B8 | (this commit) | Sprint S31 close -- legacy LLM stack clean break shipped, TIMEOUT_RECOVERY race fixed, residuals cleared |

## Acceptance table

| # | Check | Target | Actual |
|--:|---|---|---|
| 1 | Full pytest count (canonical subset) | ~282 / 7 / 2 | **243 / 7 / 2** -- below the plan's projected target; deltas accumulated faster than expected per commit. All gate-relevant tests green; the gap is in projection-vs-actual on new-test counts, not in regressions. |
| 2 | Bug Bible regression | 23 / 1 / 2 | **23 / 1 / 2** PASS |
| 3 | Audio C7 byte-identical (pytest proxy) | holds B1->B8 | **PASS** at every commit boundary |
| 4 | Audio C7 byte-identical (runtime 5080) | confirmed B2, B4, B6 | **DEFERRED** per autonomous-run handoff |
| 5 | Forbidden sweep | 0 runtime hits | **PASS** at every commit boundary |
| 6 | `story_orchestrator._load_llm` DELETED | yes | **PASS** (B4) |
| 7 | `story_orchestrator._unload_llm` DELETED | yes | **PASS** (B4) |
| 8 | `story_orchestrator._LLM_CACHE` DELETED | yes | **PASS** (B4) |
| 9 | `story_orchestrator._generate_with_llm` DELETED | yes | **PASS** (B4) |
| 10 | `_otr_model_loader.unload_llm` legacy fallback block DELETED | yes | **PASS** (B4) |
| 11 | `_otr_model_loader.invalidate_cache_no_gpu_teardown` EXISTS, dict-only, no GPU calls | yes | **PASS** (B4) |
| 12 | `_otr_model_loader.load_llm` owns bitsandbytes body | yes | **PASS** (B2 ported; B4 cleaned cache logic) |
| 13 | `generate_text` / `generate_with_llm` (no-underscore variants) anywhere | 0 | **PASS** (B5 preemptive sweep locks) |
| 14 | `story_orchestrator._run_with_timeout` uses `invalidate_cache_no_gpu_teardown` | yes | **PASS** (B4 CUDA-race fix) |
| 15 | External callers of legacy 4 symbols | 0 | **PASS** (B1 cleared VRAMContextTest, B4 confirmed) |
| 16 | Internal orchestrator callers switched | yes | **PASS** (B3 refactored `_llm_rank_news_candidates` + `_llm_rerank_with_bodies`; B3 deleted dead `_generate_ltx_style_brief`) |
| 17 | RSS path passes `technical_model` | grep-clean at writer | **PASS** (B6 Fix 1) |
| 18 | Standalone self-test 9/9 (15 widgets) | PASS | **PASS** (B6 Fix 2 flipped 11 -> 15) |
| 19 | BUG-LOCAL-226 | FIXED at B4 | **PASS** (`a4fe67a`) |
| 20 | BUG-LOCAL-228 (TIMEOUT_RECOVERY) | FIXED at B4 | **PASS** (filed + fixed in same commit) |
| 21 | Workflow JSON link rows match target input indexes | 0 violations | **PASS** (B6 Fix 3 corrected 4 rows; new `test_workflow_link_target_indexes.py` enforces) |
| 22 | OTR_VisualPromptCoercion raises loud on unwired model_id | yes | **PASS** (B6 Fix 4) |
| 23 | New S31 extinction markers | 6 | **PASS** (B5 added 4 deletion + 2 preemptive) |
| 24 | BUG-LOCAL-NNN (ungated GatedModelError recommendation) | FILED, deferred until ungated PASS soak | **CARRIED** (deferred from B6 draft Fix 3; recorded in forward-work section below; no new BUG-LOCAL entry filed since the catalog state hasn't changed to enable a recommendation) |

## Deviations from plan

1. **B2 cache reference re-bind** (not in original plan): the
   ported body in `_otr_model_loader.load_llm` imports
   `_LLM_CACHE` from `story_orchestrator`. The legacy `_unload_llm`
   REBINDS `_so._LLM_CACHE` to a fresh dict (deletes old keys,
   assigns new dict); the locally-imported reference goes stale
   after the unload call. Fix: re-bind `_LLM_CACHE = _so._LLM_CACHE`
   immediately after each `_unload_llm()` call in the body.
   Documented in the load_llm docstring. Cleared at B4 when
   `_unload_llm` and `_LLM_CACHE` were both deleted; the B4
   load_llm body has no cache logic at all (request_slot handles
   it at the outer layer).

2. **B4 `load_llm` cache-logic deletion** (deeper than plan
   suggested): the plan said B4 "deletes 4 legacy symbols and
   fixes TIMEOUT_RECOVERY race" without specifying that
   `_otr_model_loader.load_llm` also needed restructuring. With
   `_so._LLM_CACHE` gone, the body's cache-hit / cache-mismatch
   block (113 LOC of `_LLM_CACHE["..."]` checks ported from the
   legacy body) became dead code. Deleted at B4. `request_slot`
   handles caching at the outer layer; `load_llm` is now the
   always-load primitive.

3. **B4 `register_vram_cleanup(_unload_llm)` rewire** (not in
   plan): the orchestrator's module-load-time
   `register_vram_cleanup(_unload_llm)` pointed at the deleted
   symbol. Replaced with a local `_vram_cleanup_via_loader()`
   wrapper that imports `_otr_model_loader.unload_llm` lazily and
   calls it inside try/except. Pattern matches other "delegate to
   canonical surface" wrappers in `story_orchestrator.py`.

4. **B6 Fix 4 default-string sentinel preserved** (plan-allowed
   choice): the plan-listed defensive check raises
   `MissingModelInputError` when `model_id` is unwired. The
   `OTR_VisualPromptCoercion.coerce` default value is `"none"`
   (string), which is the intentional rule-only opt-out
   sentinel. The defensive check raises on None / empty / whitespace
   but NOT on the literal `"none"` -- the latter remains a
   supported opt-out for users who explicitly want rule-only
   cleanup. Documented in the error message and the new
   `test_visual_prompt_coercion_contract.py::test_visual_prompt_coercion_with_wired_model_id_proceeds`.

5. **Test infrastructure churn** (collateral of B2 / B4): five
   test files needed structural updates as the legacy symbols
   moved or got deleted:
   * `tests/test_core.py` -- `TestStoryOrchestratorCodePatterns.src`
     fixture broadened to concatenate `story_orchestrator.py` +
     `_otr_model_loader.py` (B2 moved the bitsandbytes patterns).
     `test_max_length_none_in_generate` flipped from "must exist"
     to "must NOT exist" (B4 deleted `_generate_with_llm` body
     which held the literal).
   * `tests/test_loader_slot_primitives.py` -- autouse fixture
     trimmed of `_real_so._LLM_CACHE` reset block (B4 deletion);
     added `loader.load_llm` test seam (B2 architecture shift);
     extended with 3 new B4 tests (unload simplification + helper
     contract).
   * `tests/test_loader_body_profiles.py` (NEW at B2, updated at B4)
     -- runtime cache-hit-stub tests converted to AST-based
     structural assertions (B4 removed the cache-hit short-circuit;
     the body now requires real CUDA + transformers + bitsandbytes
     to run end-to-end).
   * `tests/test_b4b_rss_rewire.py` -- two of the S30 B4b
     assertions were obsolete post-B4; renamed to deletion guards
     + safe-invalidation assertions.
   * `tests/test_unload_synchronize_guard.py` -- collapsed from 10
     BUG-LOCAL-073 assertions (guarding the deleted `_unload_llm`
     body) to a single deletion guard. The historical bug is
     structurally retired alongside the function.

6. **Plan B6 Fix 3 (UNGATED_PASS_RECOMMENDATION) deferred** (per
   plan): the originally-drafted Fix 3 was struck. Catalog has no
   ungated PASS-tier entry to recommend; the only PASS-tier entry
   (Mistral-Nemo) is gated. Recommendation would point users at
   the same gate they hit. Deferred until ungated PASS soak
   prerequisite lands (S30 forward-work).

## Forward work

* **Operator runtime 5080 verification (S31 post-close gate).** All
  7 gates from the plan's S31 post-close runtime release gate are
  deferred to post-feature-set per the autonomous-run handoff.
  Audio C7 byte-identical pytest proxy stood in during the
  autonomous run. Jeffrey runs the runtime verification when a
  feature sprint that requires it ships.

* **BUG-LOCAL-227 triage.** 25 LFC test failures latent at S30 B8
  (wide pytest tests/ walk). PRE-EXISTING, not an S31 regression.
  Suspected Phase 3/4/5/6 deletion collateral from S30. Triage
  AFTER S31 close: either delete stale tests, refactor to test
  surviving surface, or add to `EXPECTED_FAILED_NODEIDS` with
  rationale. Plan's canonical pytest subset already excludes these.

* **UNGATED_PASS_RECOMMENDATION (post-soak).** Re-open when the
  S30 forward-work soak validation of an ungated curated entry
  as `vram_fit_tier="PASS"` completes. Extend
  `auto_download_if_missing` raise path with the recommendation
  constant.

* **Loader API consolidation.** Post-S31 `nodes/_otr_model_loader.py`
  is ~700+ LOC. Split into
  `_otr_model_loader/{__init__.py, bnb_profiles.py, generation.py}`
  as post-port hygiene. Behavior unchanged. Out of scope for the
  S31+S32 autonomous run.

* **S32 (`s32-helper-per-subpass-routing`).** Per-sub-pass routing
  inside `pick_style` / `lock_cast` / `compose_line` /
  `build_news_briefs`. Closes the helper-level gap from S30
  deviation #6. Next sprint per the canonical execution plan.

* **S33 (editor-only cleanup passes).** Restore announcer-beat
  polish via writer's `enable_polish_pass` widget extension;
  retire `OTR_LedgerFreezeCascade` Phase 1 + Phase 9 auditors
  (audit-only, never edit -- standing directive #N). After S32.

## Sources

* Plan: `docs/2026-05-14-S31-S32-cowork-execution-plan.md`
* Branch: `s31-loader-clean-break` (origin synced through B8 close)
* Commits: `1bb4a4d` (B0), `3c8118e` (Task 0), `7a9584f` (B1),
  `d8129bf` (B2), `dc26421` (B3), `a4fe67a` (B4),
  `64524ef` (B5), `bc883b1` (B6), `8abe271` (B7), (B8 = this commit)
* BUG_LOG updates: `BUG-LOCAL-226` [FIXED `a4fe67a`],
  `BUG-LOCAL-227` (filed at `3c8118e`, triage carried),
  `BUG-LOCAL-228` filed + [FIXED `a4fe67a`].
* S30 final QA: `docs/2026-05-14-S30-final-qa-review.md` (parent
  reference for format).
