# Dead-Code / Lean-Down Candidate Audit -- 2026-05-29 (round 2)

Read-only fan-out audit (5 parallel agents) over the OTR repo, run after the
Phase-2 live-render validation + the BUG-276 Gate-2 fix (`cbee72c`). Findings
below are **reconciled against direct re-verification** -- a couple of agent
claims were over-stated and are corrected here. Nothing was deleted; this is a
candidate list for operator-gated removal.

**Headline:** the codebase is already lean -- `pyflakes` is clean across
`nodes/` + `visual/` (zero unused imports / redefs / unused locals). The prior
lean-down (~19K lines) was thorough. What remains is a small set of dormant /
test-only / workflow-orphaned items plus some duplicate helpers.

Removal discipline (unchanged): each candidate is its own commit; run Bug Bible
+ core + audio byte-identity after each; re-wire workflow JSON if a node surface
changes; never tombstone without the zero-ref proof shown here.

---

## TIER 1 -- Confirmed dormant / test-only (strongest; verified)

| Item | Location | Evidence | Removal note |
|------|----------|----------|--------------|
| Stage-7 shadow-critic dead branch (~54 LOC) | `nodes/_otr_freeze_cascade.py:790-843` (`if "stage1_shadow_attempts" in meta:`) | The gate key `stage1_shadow_attempts` is **never written by any production code** (verified repo-wide: only the read at :790, the comment at :768, and a test). The shadow-pass that used to stamp it was removed in lean-down step 7. | Dead branch -- unreachable in production. Audio-path-adjacent (freeze cascade) -> run the audio byte-identity gate. **Test pin:** `tests/test_stage7_shadow_critic_wiring.py::...test_block_gated_on_stage1_shadow_attempts` asserts the block exists; remove/replace it in the same commit. |
| `_otr_lfc_context.py` (whole module) | `nodes/_otr_lfc_context.py` (+ `tests/test_lfc_context_helpers.py`, 29 cases) | **Test-only**: the production import `from . import _otr_lfc_context` was already removed in a prior sprint (s28 diff shows it deleted); only the test imports it now. | Delete module + its test together. Handoff step-12 had it as "kept (test-covered)" -- this is the fresh zero-ref proof it asked for. |

## TIER 2 -- Registered nodes used by NO workflow (step-12 list, now proven)

Not present in `workflows/otr_scifi_16gb_full.json` (the shipped graph) nor any
other workflow JSON; their only references are `__init__.py` registration + test
guardrail lists + internal docstrings. Each removal = delete node file + drop
the `NODE_CLASS_MAPPINGS` entry + adjust the test guardrail.

| Node | Module | Only-refs |
|------|--------|-----------|
| `OTR_BatchProceduralSFX` | `nodes/batch_procedural_sfx.py` | registration + `test_workflow_json_guardrails.py` voice-node list |
| `OTR_VideoConcat` | `nodes/otr_video_concat.py` | registration + internal docstring only |
| `OTR_SaveCopy` | `nodes/otr_save_copy.py` | registration + internal (QA tee) only |
| `OTR_CheckpointLoaderGated` | `visual/checkpoint_loader_gated.py` | registration + a docstring mention in `visual/flux_prompt_extractor.py` |

## TIER 3 -- Orphaned visual placeholders (never wired)

| Item | Location | Evidence |
|------|----------|----------|
| `WallClockEstimate` + `estimate()` | `visual/wall_clock.py` (~230 LOC) | Day-11 render-time estimator; 0 production callers, test-only pin. |
| `character_regression` module | `visual/character_regression.py` (~150 LOC) | Day-12 SSIM portrait-likeness gate; 0 callers anywhere in the live path. |

## TIER 4 -- Duplicate / overlapping helpers (DRY refactor, lower priority)

These DO run -- they are copies, not dead code. Consolidation reduces surface
but carries regression risk; several are **intentional mirrors** and should be
left alone unless a sprint explicitly takes them on.

| Cluster | Sites | Note |
|---------|-------|------|
| `_resolve_input_still` (code-for-code identical) | `visual/backends/ltx_motion.py:209`, `wan21_loop.py:230`, `florence2_sdxl_comp.py:243` | Highest-value consolidation IF those backends are live; verify backend usage first. |
| `_word_count` | `production_ledger.py:120`, `_otr_craft_floor.py:316`, `_otr_stage3_validators.py:253` | Two regex copies + one split-variant; live writer/cascade paths -> regress carefully. |
| `_resolve_radio_still_path` | `batch_ltx_render.py:804`, `batch_humo_render.py:386`, `video_composite.py:117` | **INTENTIONAL mirror** per BUG-LOCAL-121 -- leave unless coordinated. |
| `_load_ledger[_with_path]` | `batch_ltx_render.py:2225`, `batch_humo_render.py:3134`, `video_composite.py:384` | **INTENTIONAL mirror** per BUG-LOCAL-076 -- leave unless coordinated. |
| `_voiced_beats` | `_otr_beat_validators.py:81`, `_otr_editor_constraints.py:163` | One-line filter dup; trivial. |
| default-model-path resolvers | `visual/backends/flux_anchor.py:85`, `pulid_portrait.py:67/80` | Same try/except pattern, different model names. |

## KEEP -- verified live or intentionally retained (NOT candidates)

- `nodes/project_state.py` -- **LIVE**: `story_orchestrator.py:53` imports `ProjectState`. The *node* `OTR_ProjectStateLoader` is workflow-orphaned, but the module stays and ROADMAP plans a `show_name` widget on the node. (Agent over-claimed "zero refs" -- corrected.)
- 5x `OTR_Visual*` sidecar nodes (`bridge`/`poll`/`renderer`/`prompt_coercion`/`flux_prompt_extractor`) -- self-contained video sidecar subsystem, internally referenced; not wired into the main audio graph but not dead.
- `OTR_VRAMGuardian`, `OTR_VRAMContextTest` -- kept (lean-down step 11); test-pinned.
- `SlotScheduler.for_polish()` (`OTR_LedgerScriptWriter.py:562-579`) -- no prod caller but intentionally retained as a test-pinned slot-routing primitive wrapping the *shared* `make_polish_generate_fn`.
- Live flags: `use_exchange`, `enable_production_stage3_validators` -- both ON in the shipped workflow, fully tested.

## Suggested removal order (each its own commit + full gate run)

1. **Tier 1** -- shadow-critic dead branch + `_otr_lfc_context` (pure dead/test-only; highest confidence, ~230 LOC + 29 tests). 
2. **Tier 2** -- the 4 workflow-orphaned nodes (one commit each; touches `__init__.py` + a test guardrail, no workflow re-wire since they were never wired).
3. **Tier 3** -- the 2 visual placeholders.
4. **Tier 4** -- optional DRY pass on the non-mirror duplicates only.

Verify each with: `rg "<symbol>" --type py | rg -v "^tests/"` -> expect empty,
then Bug Bible + `test_core` + `test_audio_byte_identical` green before commit.
