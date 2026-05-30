# Dead-Code / Unused-Node Audit — Round 3

**Date:** 2026-05-29 | **Branch:** v2.0-alpha | **HEAD:** d893e56
**Scope:** Read-only audit. No code/workflow/config was modified. Output is this report only.
**Method:** ripgrep across all non-`.venv` `.py` plus workflow JSON, docs, scripts, config. Every "removable" claim carries a reproduced zero-reference proof.

---

## Headline

**The codebase is already lean.** The 8-commit lean-down pass (aad4cfb..d893e56) removed the heavy orphans. Of 35 registered nodes, **27 are wired into the canonical workflow** (`workflows/otr_scifi_16gb_full.json`) and the remaining 8 are all KEEP-by-design (VRAM topology, Visual sidecar, project-state loader). Of 120 source `.py` files in `nodes/` + `visual/`, the vast majority are live via direct import, lazy in-method import, ComfyUI node registration, the `visual/backends` factory registry, or subprocess spawn.

**Genuinely removable: 4 items (~1,070 LOC).** Two are clean orphans left by the just-completed teardowns; two are an unused per-shot planner and an unused post-processing filter. A further 2 modules are **test-only** (a superseded model-runtime backend layer and a never-wired voice-backend registry) — flagged but NOT recommended for deletion, because they each carry a dedicated test suite and touch model/provider plumbing (the hard-constraint KEEP zone).

### Critical methodology note — null-byte blind spot

Three production files contain **trailing NUL padding** that makes ripgrep treat them as *binary* and silently skip them unless `-a` is passed:

| File | First NUL @ | Trailing NULs |
|------|------------|---------------|
| `__init__.py` | offset 24533 | 862 bytes |
| `nodes/_otr_freeze_cascade.py` | offset 47075 | 3696 bytes |
| `nodes/_otr_legacy_to_stage1_adapter.py` | offset 25590 | 100 bytes |
| `visual/unload_all.py` | offset 16052 | 10 bytes |

**This is almost certainly why the prior audit mislabeled live modules as orphans.** `_otr_freeze_cascade.py` (the freeze-cascade orchestrator) lazily imports `_otr_lfc_watchdog`, `_otr_readiness`, and `_otr_render_plan` *inside method bodies after the NUL block* — a default rg run never sees those imports and reports the modules as dead. They are all LIVE. This audit re-ran every candidate with `rg -a` (text mode) to defeat the trap. (See "Bonus finding" for the NUL-padding defect itself — worth a separate clean-file fix.)

---

## TIER 1 — High-confidence removable (zero-ref proof each)

### 1. `nodes/_otr_critic_rubric.py` (237 LOC) + `docs/2026-05-26-sprint-10a-whole-episode-critic-rubric.md`
- **What:** Rubric loader (`Rubric`, `load_rubric`, `RubricAxis`, `ShipThreshold`) for the Sprint-10A whole-episode shadow critic.
- **Why dead:** The whole-episode critic (`_otr_whole_episode_critic.py`) was removed in commit `aad4cfb` ("Remove Stage-7 whole-episode shadow critic"). This rubric loader was its only consumer and is now orphaned. The companion rubric markdown is referenced by nothing but this module.
- **Proof (text-mode, null-byte-safe):**
  ```
  rg -a -n "\b_otr_critic_rubric\b" -g '*.py' --glob '!.venv/**' --glob '!**/__pycache__/**' .   | rg -v "/_otr_critic_rubric\.py:"
  -> (no output)
  rg -a -n "load_rubric|RubricAxis|ShipThreshold|\bRubric\b" ... | rg -v "/_otr_critic_rubric\.py:"
  -> (no output)
  rg -al "sprint-10a-whole-episode-critic-rubric" -g '*.py' ...
  -> nodes/_otr_critic_rubric.py   (only self-reference)
  ```
- **Removal note:** Delete both files together; no test, prod, workflow, or other-doc reference remains.

### 2. `visual/planner.py` (513 LOC)
- **What:** Per-shot backend planner (`plan_episode`, `PlannerJob`, `PlannerResult`, `_infer_backend`, non-repeat windowing). Selects which `visual/backends/*` to run per beat.
- **Why dead:** The live sidecar (`visual/worker.py`, spawned by `bridge.py:623`) dispatches backends through `visual/backends.resolve()` directly via the `OTR_VISUAL_BACKEND` env var — it never calls `plan_episode`. No node, no test, no script imports `planner`.
- **Proof:**
  ```
  rg -a -n "\bplanner\b|plan_episode|PlannerJob|PlannerResult" -g '*.py' --glob '!.venv/**' --glob '!**/__pycache__/**' .   | rg -v "/planner\.py:"
  -> visual/backends/wan21_loop.py:58:  (a doc-comment mention only, no import/call)
  rg -al "visual.planner|plan_episode|from visual import planner" tests/   -> (no output)
  rg -a -ln "visual.planner|visual/planner|plan_episode" --glob '!.venv/**' .
  -> visual/planner.py, docs/ROADMAP_HISTORY.md   (self + history doc only)
  ```
- **Removal note:** Remove `visual/planner.py`. Only a stale doc-comment in `wan21_loop.py:58` mentions "the planner"; that comment can stay or be trimmed.

### 3. `visual/postproc/` subtree — `vhs.py` (549 LOC) + `__init__.py` (30 LOC)
- **What:** VHS aesthetic post-processing filter (scanlines, chroma bleed) and its package shell. `__init__.py.__all__ == ["vhs"]`.
- **Why dead:** No backend, worker, renderer, node, test, or script imports `postproc` or `vhs`. The package only re-exports itself.
- **Proof:**
  ```
  rg -a -n "postproc|apply_vhs|vhs_filter|from .vhs|import vhs" -g '*.py' --glob '!.venv/**' --glob '!**/__pycache__/**' .   | rg -v "/postproc/"
  -> (no output)
  rg -a -n "postproc|vhs" visual/backends/*.py visual/worker.py visual/renderer.py   -> (no output)
  rg -al "postproc|vhs" tests/   -> (no output)
  ```
- **Removal note:** Remove the whole `visual/postproc/` directory. Only `docs/s28_diff_tmp.txt` and `docs/ROADMAP_HISTORY.md` (history docs) mention it.

---

## TIER 2 — Registered-but-unwired nodes (8 total) — classification

All 8 registered nodes absent from every workflow JSON were checked. **All 8 are KEEP.** None is removable.

| Registration key | Source module | In workflow? | Classification | Evidence |
|---|---|---|---|---|
| `OTR_ProjectStateLoader` | `nodes/project_state.py` (`ProjectStateLoader`) | No | **KEEP — live in prod** | `nodes/story_orchestrator.py:53 from .project_state import ProjectState`. CLAUDE.md constraint also pins it. |
| `OTR_VRAMGuardian` | `nodes/vram_guardian.py` | No | **KEEP — by-design** | Named KEEP in audit constraints (VRAM topology). Test pin: `tests/` reference `vram_guardian`. |
| `OTR_VRAMContextTest` | `nodes/vram_context_test.py` | No | **KEEP — by-design** | Named KEEP in constraints; pinned in `tests/test_b7_forbidden_sweep.py`, `tests/test_no_orchestrator_legacy_symbols.py`, and `docs/_s28_llm_slot_sweep.py`. |
| `OTR_VisualBridge` | `visual/bridge.py` | No | **KEEP — sidecar root** | Spawns the live worker subprocess (`bridge.py:623 [sidecar_python, _WORKER_SCRIPT, job_dir]`). Whole OTR_Visual* sidecar is a constraint KEEP. |
| `OTR_VisualPoll` | `visual/poll.py` | No | **KEEP — sidecar** | Part of Bridge→Poll→Renderer trio (`poll.py:177` "Job ID from OTR_VisualBridge"). |
| `OTR_VisualRenderer` | `visual/renderer.py` | No | **KEEP — sidecar** | Consumes sidecar STATUS/assets (`renderer.py:97`). Trio member. |
| `OTR_VisualPromptCoercion` | `visual/prompt_coercion.py` | No | **KEEP — sidecar** | Pre-cleans script_json into VisualBridge (`prompt_coercion.py:228`). |
| `OTR_VisualExtractFluxPrompt` | `visual/flux_prompt_extractor.py` | No | **KEEP — sidecar** | Registered FLUX-prompt extractor in the Visual subsystem. |

**Note on "unwired":** these are wire-ready / sidecar-orchestration nodes. The Visual subsystem runs as a subprocess-isolated pipeline driven by `bridge.py`, not by being placed in the main DAG, so absence from `otr_scifi_16gb_full.json` is expected and intentional.

---

## TIER 3 — Test-only modules & dead-branch sweep

### Test-only modules (flagged, NOT recommended for deletion)

#### `nodes/_otr_model_runtime.py` (183 LOC) — test-only, superseded backend layer
- **What:** `get_backend_for_row` + three Transformers adapter classes (`TransformersSafetensorsBackend`, `TransformersMultimodalTextOnlyBackend`, `TransformersGPTQInt4Backend`). Sprint-D D1b experiment.
- **Status:** Only importer is `tests/test_loader_backend_protocol.py:39`. The live loader `nodes/_otr_model_loader.py` loads models directly via `AutoModelForCausalLM.from_pretrained` (lines 437, 512) and uses `_otr_loader_backends` for the duck-typed protocol — it never calls `get_backend_for_row`.
- **Proof:**
  ```
  rg -a -n "\b_otr_model_runtime\b" -g '*.py' --glob '!.venv/**' --glob '!**/__pycache__/**' .   | rg -v "/_otr_model_runtime\.py:"
  -> tests/test_loader_backend_protocol.py:39  (test import)
  -> tests/test_loader_backend_protocol.py:104 (test reads source)
  -> nodes/_otr_loader_backends.py:17          (docstring mention, not an import)
  ```
- **Recommendation:** **KEEP for now.** This is model-runtime/loader-backend plumbing in a model-agnostic project (the hard-constraint KEEP zone) and carries a dedicated protocol test. Removing it would also gut `test_loader_backend_protocol.py`. If the team confirms the duck-typed `_otr_loader_backends` path fully supersedes it, retire module + test together in a deliberate sprint — do not silently delete.

#### `nodes/_voice_backends/` package (346 LOC: `__init__.py` 151, `_protocol.py` 68, `bark.py` 74, `kokoro.py` 53) — test-only voice-engine registry
- **What:** Engine-selection registry (`register`, `get_factory`, `available_engines`, `VoiceBackend` protocol) with self-registering bark/kokoro drivers. Maps engine name → lazy factory.
- **Status:** Only `tests/test_voice_backends.py` imports it. The live TTS paths bypass it: `nodes/batch_bark_generator.py:581` uses `from ._otr_bark_lib import _load_bark`; `nodes/kokoro_announcer.py` lazy-imports the Kokoro lib directly.
- **Proof:**
  ```
  rg -a -n "_voice_backends|VoiceBackend|get_factory\(" -g '*.py' --glob '!.venv/**' --glob '!**/__pycache__/**' --glob '!tests/**' .   | rg -v "/_voice_backends/"
  -> (no output — only tests/ reference the package)
  ```
- **Recommendation:** **KEEP / defer.** This is provider/engine-selection plumbing (the hard-constraint KEEP zone: "fallback/provider plumbing... do NOT flag as dead") AND it has a full test suite. Classify as wire-ready abstraction, not dead. If the team decides the abstraction will never be wired, remove package + test deliberately.

### Dead-branch sweep — clean

Searched `nodes/_otr_freeze_cascade.py` and `OTR_LedgerScriptWriter.py` for gates keyed on never-written meta keys (the Stage-7 pattern that aad4cfb removed). **None found.** The remaining gates read keys that ARE written:
- `meta.get("freeze_verdict")` — written and asserted in `tests/test_lfc_phase_0_10_gap_audit.py` (frozen_clean / frozen_with_warns). Live.
- `meta["render_plan"]` — written by `_otr_freeze_cascade.py:919` via `_OTRRP.build_render_plan`, consumed by `batch_humo_render.py:1690`. Live.

No always-off feature flags or unreachable consumer branches detected post-teardown.

---

## TIER 4 — Duplicate / mirrored helpers (all INTENTIONAL — KEEP)

| Helper | Locations | Marker | Verdict |
|---|---|---|---|
| `_resolve_radio_still_path` | `nodes/batch_ltx_render.py:804`, `nodes/batch_humo_render.py:386` | `BUG-LOCAL-121` hardening comments at both sites; tested in `tests/test_radio_still_resolver.py` | **KEEP — intentional mirror** (per constraint) |
| `_load_ledger` | `nodes/batch_ltx_render.py:2225`, `visual/batch_flux_portrait_render.py:683` | `BUG-LOCAL-076` fallback-chain comments | **KEEP — intentional mirror** (per constraint) |

Both are pinned in the constraints as intentional mirrors. Do not consolidate.

---

## KEEP list — verified live or intentionally retained

These were investigated and proven live; listing the ones most likely to be misjudged by a naive importer scan (all confirmed via `rg -a` text mode to defeat the NUL blind spot):

| Module | Why a naive scan misses it | Proof of liveness |
|---|---|---|
| `nodes/_otr_lfc_watchdog.py` | imported *after* NUL block in `_otr_freeze_cascade.py` | `_otr_freeze_cascade.py:631 from . import _otr_lfc_watchdog as _LFC_WD`; called L632 `_LFC_WD.vram_over_ceiling(...)` |
| `nodes/_otr_readiness.py` | lazy import after NUL block | `_otr_freeze_cascade.py:432,443 from . import _otr_readiness as _LFC_RDY` (Phase 7/8) |
| `nodes/_otr_render_plan.py` | lazy import after NUL block | `_otr_freeze_cascade.py:73 from . import _otr_render_plan`; `build_render_plan` called L911 |
| `nodes/_otr_freeze_cascade.py` | flagged 0-importer by census | lazy-imported by `OTR_LedgerFreezeCascade.py:292 from . import _otr_freeze_cascade as _LFC_ORCH` |
| `nodes/_otr_legacy_to_stage1_adapter.py` | NUL-padded file | `OTR_LedgerScriptWriter.py:1977 from . import _otr_legacy_to_stage1_adapter` |
| `nodes/news_interpreter.py` | not a node, not in any workflow | `OTR_LedgerScriptWriter.py:1968 from . import news_interpreter as _OTRNI` |
| `nodes/_otr_hf_env.py` | in-method import | `_otr_model_loader.py:351 from . import _otr_hf_env` |
| `nodes/_otr_lmfe_compat.py` | in-method import | `_otr_constrained_generate.py:40,218` |
| `nodes/_otr_loader_backends.py` | in-method import | `visual/llm_polish.py:134`, `_otr_model_loader.py:848` |
| `nodes/_otr_memory.py` | in-method import | `video_composite.py:972`, `otr_post_upscale_procgen_blend.py:547` |
| `nodes/_otr_probe.py` | in-method import | `video_composite.py:1370`, `batch_ltx_render.py:1545,2114` |
| `visual/worker.py` | spawned, not imported | `bridge.py:623` Popen of `_WORKER_SCRIPT` |
| `visual/camera_path.py` | imported inside worker | `worker.py:166 from camera_path import zoompan_for_shot` |
| `visual/backends/*.py` (all 8: flux_anchor, pulid_portrait, flux_keyframe, ltx_motion, wan21_loop, florence2_sdxl_comp, video_stack, placeholder_test) | imported inside factory bodies | `visual/backends/__init__.py:106-113 register(...)` with lazy `from . import X` factories |
| `visual/wedge_probe.py` | flagged 0-importer | dedicated suite `tests/test_wedge_probe.py` — KEEP per constraint (tested) |
| `nodes/_otr_humo_tier_loader.py`, `nodes/_otr_deferred_loaders.py`, `visual/unload_all.py`, `visual/flux_branch_gate.py`, `visual/ltx_branch_gate.py` | registered nodes, loaded by ComfyUI registry not Python import | named KEEP-by-design in constraints (VRAM topology / branch gates) |
| `nodes/_otr_workflow_validator.py` (`OTR_WorkflowValidator`) | — | wired in `otr_scifi_16gb_full.json`; test-pinned (`test_default_workflow_validator.py`) |
| `otr_v2/` package | tiny namespace | `worker.py` resolves `from otr_v2.visual import backends`; holds `visual_plan.schema.json` |

---

## Bonus finding (not dead code, but a clean-file defect)

**NUL-byte padding in tracked source files.** Four production/test files carry trailing `\x00` bytes (table in Headline), and four test files too (`test_audiogen_cache_keys.py`, `test_core.py`, `test_g7_consumer_constants.py`, `test_per_cue_sfx_dur.py`, `test_stage7_shadow_critic_wiring.py`). This violates the "UTF-8, no BOM, clean files" standard, breaks default ripgrep/grep auditing (the root cause of prior false-orphan calls), and risks editor/diff corruption. **Recommend a separate sweep to strip trailing NULs** (`data.rstrip(b'\x00')`) and re-verify AST parse. This is the highest-value non-deletion cleanup surfaced by this audit.

---

## Summary table — removable inventory

| Tier | Item | LOC | Action |
|---|---|---|---|
| 1 | `nodes/_otr_critic_rubric.py` + rubric `.md` | 237 + doc | Remove (orphaned by aad4cfb) |
| 1 | `visual/planner.py` | 513 | Remove (no live caller) |
| 1 | `visual/postproc/` (vhs.py + __init__.py) | 579 | Remove subtree |
| 3 | `nodes/_otr_model_runtime.py` | 183 | KEEP (test-only, model plumbing) — retire deliberately if confirmed superseded |
| 3 | `nodes/_voice_backends/` package | 346 | KEEP (test-only, provider plumbing) — retire deliberately if never wired |
| — | NUL-padding sweep | 9 files | Clean-file fix (separate from dead-code) |

**Genuinely removable now: 3 items / ~1,330 LOC** (TIER 1). Two further test-only subsystems (~530 LOC) are flagged but held back under the model/provider-plumbing KEEP constraint.
