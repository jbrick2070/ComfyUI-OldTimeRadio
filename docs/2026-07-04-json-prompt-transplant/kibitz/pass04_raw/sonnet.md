# r4 Review -- Sonnet (convergence)

**VERDICT: GO (with one MUST-FIX, mechanical, same-round-resolvable)**

The plan is code-ready. All file:line claims re-verified against real files check out exactly. One genuinely new must-fix surfaced: Chunk 5's harness code calls a method that doesn't exist. It's a one-line fix, not a design defect.

**MUST-FIX**

1. **Chunk 5 harness calls `reg.list_packs("science_news")` -- that method does not exist on `Registry`.** `registry.py` (read in full) exposes `self.packs: dict[tuple[str,str,str], tuple[StoryPack, Path]]` populated at `:77`, with accessor methods `bank()`, `pack()`, `pack_path()`, `style()`, `pipeline()`, `resolve()` -- no `list_packs`. The Chunk 5 test code in pass03_plan.md (`for pack_key in reg.list_packs("science_news")`) will `AttributeError` at collection/run. Fix: iterate `[k for k in registry.packs if k[0] == "science_news"]` directly, or add a small `list_packs(bank_id)` helper to `Registry` in the same Chunk 3/5 commit. Trivial, same-round fixable -- does not block GO, but a coder must not copy the pseudocode verbatim.

**SHOULD-FIX**

- None new. r3's rollback/push-verification/regression commands hold up under direct inspection.

**Grounding table**

| # | Check | Finding | Status |
|---|---|---|---|
| 1 | pass03 vs OTR HEAD file:line drift | `contracts.py:185/232/351` (3 validators), `:270-279` (StoryPromptProfile fields), `profiles.py:67-95` (stage calls), `registry.py:70-72` (explicit root ctor) -- all match exactly | CONFIRMED, no drift |
| 1b | `_otr_outline.py` / `_otr_line_composer.py` constants | `_SYSTEM_PROMPT`:532, `_MACRO_SYSTEM_PROMPT`:1102, `_PHASE_SYSTEM_PROMPT`:1115, `_BEAT_SYSTEM_PROMPT`:1130, `_make_system` closure :1854-1857, line_composer `_SYSTEM_PROMPT`:1174 | CONFIRMED exact |
| 2 | test-name collision | Sibling `tests/` has 7 files (`test_bridge_and_runner.py`, `test_bridge_e2e.py`, `test_compat_drift.py`, `test_matrix_and_leakage.py`, `test_registry_failloud.py`, `test_roundtable_folds.py`, `test_transplant_modules.py`); no `test_phase_a_byte_identity.py`, `test_extractor_coverage.py`. OTR `tests/` has no `test_identity_check_outline.py` | CONFIRMED no collision |
| 3 | Chunk 5 OTR-side pytest needs conftest setup | OTR `tests/conftest.py` sets `CUDA_VISIBLE_DEVICES=""` / `OTR_TEST_MODE=1` at **module import time** (before collection), autouse session fixture -- new test file picked up automatically, no per-test setup needed | CONFIRMED, no gap |
| 4 | Chunk 7 regression commands runnable | Correct venv path pattern used elsewhere in repo docs, `$env:PYTHONUTF8="1"`, `-p no:cacheprovider` flags match CLAUDE.md-documented invocation; Bug Bible path outside this session's reach to verify existence directly, but path/pattern matches project convention | CONFIRMED pattern; bug-bible file existence UNVERIFIABLE (outside connected folders) |
| 5 | New Phase A invariant risk | **`Registry.list_packs()` referenced in Chunk 5 pseudocode does not exist** -- real gap | CONFIRMED -- new MUST-FIX |
| 6 | circular import: `extractor.py` -> `contracts` + `registry` | `registry.py` imports from `.contracts` and `.compat` only; `contracts.py` imports nothing from `.registry`. `extractor.py` importing both is a strict DAG leaf, no cycle | CONFIRMED safe |
| 7 | `TEMPLATE_SEAMS` alias safety | Only 6 references to `TEMPLATE_SEAMS` repo-wide, all inside `contracts.py` itself (3 validators + 2 error-message sites + the definition) -- no external `from ... import TEMPLATE_SEAMS` anywhere in either repo | CONFIRMED safe |

Nothing else rises to a must-fix. The `list_packs` gap is exactly the kind of thing that would otherwise cause a red Chunk 5 push -- worth flagging explicitly to the coder, but it doesn't change the plan's shape or sequencing, so GO stands with that one-line correction folded in before the coder executes Chunk 5.
