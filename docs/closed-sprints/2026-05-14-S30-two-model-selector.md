# PREAMBLE — Problem Statement for Round-Robin Review

**Audience.** ChatGPT, Gemini, and any other reviewer LLM consulted under the round-robin pattern in `CLAUDE.md`. Read this preamble first. The sprint plan below is the document being reviewed. Companion artifact: `workflows/otr_scifi_16gb_full.json` (the canonical ComfyUI workflow JSON this sprint mutates).

## Project context

ComfyUI-OldTimeRadio (OTR) is a ComfyUI custom-node pack that generates multi-character radio drama episodes — audio + video — single-developer, ~2 years old, Python 3.12 on Windows, RTX 5080 Laptop with 16 GB VRAM. **Strict local inference**: no generation-time API calls to OpenAI / Anthropic / hosted endpoints; every LLM, TTS, image, and video pass runs on the user's own GPU. **One-time HuggingFace model fetch** is the standard distribution mechanism for open-weight models — a free public API, no key required for ungated repos.

The default model in the canonical workflow is `mistralai/Mistral-Nemo-Instruct-2407` (gated, soak-tested PASS) **for audio C7 byte-identity continuity** — the regression baseline. `HF_TOKEN` is consumed only when present and only for gated repos; it's never injected as a required parameter.

**Honest first-run-without-token framing (corrected, no overpromise):** OTR doesn't *require* an HF_TOKEN to attempt any model. But the ungated curated alternatives (Qwen2.5-14B, Captain-Eris-12B, Mag-Mell-12B) all carry `vram_fit_tier = WARN` — they're not soak-tested PASS on a 16 GB rig and their full safetensors are 24–28 GB, needing quantization or offload that the S30 loader doesn't ship. **There is no 16GB-PASS no-token path in S30 today.** A no-token user can attempt an ungated model, but the path is unproven on the target rig. The "no-token first-run" claim from earlier passes was too broad; the accurate claim is: "no-token users can attempt alternate models, not guaranteed 16GB-ready until a future soak validates one ungated entry as PASS or the loader gains a quantization path." Once weights are local, OTR runs fully offline. Branch `v2.0-alpha`.

The last four sprints (S24 → S29, closed 2026-05-14) were a **cleanbreak chain** — methodically deleting legacy code paths so the v2.0 contract is the only contract. Current HEAD state:

- pytest: **2146 passed / 8 skipped / 0 failed**
- Bug Bible regression: **23 passed / 1 skipped / 2 xfailed**
- Forbidden-pattern sweep: **0 runtime hits** (31 forensic suppressed)
- Workflow link integrity: **0 violations** across all 8 workflow JSONs (canonical + 4 HuMo/LTX smokes + 3 external examples)
- Audio byte-identical regression: **PASS**
- `docs/cleanbreak-deferred.md`: **deleted** (zero active deferrals)

## The problem this sprint solves

**Ten LLM-model-pick widgets currently exist** scattered across the workflow:

- 1 on `OTR_LedgerScriptWriter` (the writer node)
- 1 on `OTR_LedgerFreezeCascade`
- 3 on the standalone LFC phase nodes (`OTR_LFCPhase4Scene`, `OTR_LFCPhase5Voice`, `OTR_LFCPhase6Arc`)
- 1 on `OTR_VisualLLMSelector`
- 1 on a test bench (`nodes/vram_context_test.py`)
- 2 on non-LLM media nodes (out of scope for this sprint per the Shape-C decision)
- Plus an **eleventh** discovered in the §2b audit: `nodes/story_orchestrator.py` carries a complete parallel `_load_llm` / `_LLM_CACHE` / `_generate_with_llm` stack with 8+ hardcoded model defaults. Dead at runtime — production code only imports `_runtime_log` + `_unload_llm` — but the surface is hostile to the routing audit being clean.

Three separate in-process LLM caches exist:
- `nodes/_otr_model_loader.py::LLM_CACHE` — the modern facade
- `nodes/story_orchestrator.py::_LLM_CACHE` — dead at runtime
- `visual/llm_polish.py::_POLISH_CACHE` — live, can double-load Mistral-Nemo and bust the 14.5 GB VRAM ceiling

Users have no way to know which dropdown drives which behavior. Misconfigurations cause silent model drift between phases. The workflow is a model-pick maze.

## The fix this sprint delivers

Centralize on **two dropdowns**, both on the writer node:

- `creative_writing_model` — narrative passes (outline, cast, dialogue, polish, style invention, scene coherence, visual prompt cleanup).
- `technical_model` — structured passes (JSON validators, GBNF-grammar output, reviewer verdicts, format normalization, critic, news interpreter, style chooser, cast contract).

Every other LLM-consuming node has its `model_id` widget **deleted** (no transition shim, no fallback default) and reads from the writer's broadcast output sockets via wired STRING inputs. New standing rule landed in CLAUDE.md as Prime Directive 6: no new LLM widget anywhere outside the writer, ever. Every new LLM call site must be tagged `# LLM slot: creative` or `# LLM slot: technical` at the call site with a one-sentence reason.

## Hard constraints (non-negotiable)

1. **Audio is king (Prime Directive 1).** Audio output stays byte-identical across the change. If audio drifts → revert immediately. Do not investigate forward; revert and re-plan.
2. **VRAM ≤ 14.5 GB peak (Prime Directive 2).** When Slot 1 ≠ Slot 2, use `_otr_model_loader.unload_llm()` for the full teardown (CPU-move weights + `gc.collect()` + `empty_cache()` + `ipc_collect()` + `synchronize()`) before loading the next slot model. Do NOT use `_flush_vram_keep_llm()` for cross-model transitions — that primitive preserves the LLM cache identity, which is the opposite of what a model swap needs. `_flush_vram_keep_llm()` applies only to same-model phase transitions (clear intermediate activations, keep weights resident). Never `force_vram_offload()` between LLM phases in either case.
3. **Wire it or don't ship it (Prime Directive 3).** If a changed node is **placed in any shipped workflow JSON**, that JSON gets the corresponding update in the same commit. If the changed node is not placed in any shipped workflow JSON (per §2a-bis: LFC Phase 4/5/6, VisualLLMSelector, VisualPromptCoercion), WIRE = NONE for the Python commit and the workflow link validator still runs across all 8 JSONs to catch drift. Prime Directive 3 is honored at the SPRINT level: by B8 close, every Python-side change is reflected in every JSON that places the changed node.
4. **Every LLM call tagged creative or technical (Prime Directive 6, new).** No node other than the writer exposes a `model_id` widget. Enforced by forbidden-pattern sweep.
5. **No legacy back-compat (standing directive since 2026-05-11).** S28+S29 spent two sprints deleting legacy. This sprint must not reintroduce any of it. Specifically forbidden: `_RENAME_ALIASES` entries, fallback-on-unknown-model, "stamp both meta keys for transition", migration scripts that linger after running, `legacy_archive/` directories, soft-landing passthroughs.
6. **Bug Bible regression 23/1/2xf holds at every commit boundary.** Drift → revert.

## Audit findings already folded into this plan

The original scoping doc (`docs/2026-05-13-two-model-selector-scoping.md`) listed 7 LLM-pick sites and 9 sub-passes inside the writer. A full grep across `nodes/`, `visual/`, and `scripts/` for `load_llm(`, `make_generate_fn(`, `make_polish_generate_fn(`, and `AutoModelForCausalLM.from_pretrained` surfaced 3 additional production sites and 5 additional writer sub-passes. All folded into §2b and §2c of the plan. Total inventory: **15 entries, every one slot-tagged**.

## What the round-robin is asked to evaluate

Eight specific questions. Each reviewer returns a structured response per question with **MERGE / MODIFY / REJECT** recommendation, severity (**P0 stop-ship / P1 fix-before-kickoff / P2 future-sprint**), and reasoning.

1. **Per-commit decomposition robustness — 14 commits.** B0 → B1a / B1a2 / B1b / B1c → B2a / B2b / B2c → B3 → B4 → B5 → B6 → B7 → B8. Is any sub-commit still too large? Is any commit ordered wrongly such that re-ordering would reduce risk? Particular attention to B1c (loader slot primitives — `unload_llm`, `request_slot`, `check_vram_fit` all land in one commit) and B2b (slot scheduler implementation).
2. **Audit completeness — any LLM call site missed?** §2 + §2a-bis + §2b + §2c claim to be the full inventory. Is there a category of LLM-call grep that wasn't run (e.g. `importlib.import_module(...)`-based lazy loads, ComfyUI plugin hooks, dynamic class registration)?
3. **VRAM swap pattern + teardown sequence correctness.** §6 says `unload_llm()` (full teardown: CPU-move + `gc.collect()` + `empty_cache()` + `ipc_collect()` + `synchronize()`) for cross-model transitions; `_flush_vram_keep_llm()` for same-model phase transitions only. Documented DAG minimum is ~9 transitions for a 3-beat episode in per-beat-default mode; opt-in `OTR_BATCH_PER_BEAT=1` mode reaches ~3 at the cost of the per-beat critic→polish feedback loop. Does this teardown order and batching tradeoff match real-world Windows / Blackwell / CUDA 13 behavior?
4. **No-back-compat enforcement.** §0a lists 8 forbidden kinds of back-compat. Are any of the new tests in B6 secretly back-compat tests in disguise (e.g. `test_slot1_eq_slot2_reuses_single_model_cache` — phrased as a behavior assertion, but is it really a back-compat probe by another name)?
5. **Audio C7 byte-identity safety.** Slot 1 default == prior Mistral-Nemo. §7 specifies a Python-fixture gate at every commit B0 onward and an end-to-end gate at B3 (first commit where canonical workflow runs on the new contract). Does the two-tier gate catch every scenario where the routing change could non-trivially shift audio output?
6. **Forbidden-pattern sweep correctness.** §B7 lists extinction markers + a structural rule ("`model_*` STRING widget outside writer"). Critical: does the structural rule correctly distinguish between **widget** STRING entries (`("STRING", {"default": "..."})` — rejected outside writer) and **connectable input socket** STRING entries (link-only, no widget args — allowed on consumer nodes, since B3/B4/B5 add `creative_writing_model` / `technical_model` sockets exactly there)? If the sweep doesn't make this distinction, it will false-positive on the very sockets the sprint creates.
7. **Hardware inclusivity constraints.** B1a `validate_model_id` allow-list — does it correctly admit (a) curated models, (b) locally-scanned HF cache repo IDs, (c) arbitrary `org/name` when auto-download is enabled, while rejecting paths / drive letters / traversal / unsafe formats? B1b `HARD_VRAM_CONTEXT_LIMIT` default (8192 on 16 GB) — right ceiling, or too tight for a 16 GB rig with Gemma-4-E2B? B1c `check_vram_fit` — should it return PASS/WARN/UNKNOWN/FAIL tiers rather than a binary `bool` (since rough math on uncurated models can lie)?
8. **Estimate realism — 14 commits now.** With B0 + B1a/B1a2/B1b/B1c + B2a/B2b/B2c + B3 + B4 + B5 + B6 + B7 + B8, what's the realistic sprint length? Plan calls it 5–7 days (§9b). Is any commit optimistically scoped — particularly B2b (slot scheduler implementation against the live writer DAG) and B5 (`_POLISH_CACHE` collapse plus sampling-profile precedence)?

## Round-robin format

Per `CLAUDE.md` round-robin section:

1. **ChatGPT** (gpt-4.1+ via `scripts/_consult_openai.py` or the round-robin driver) — first opinion + critique against the 8 evaluation points above.
2. **Gemini** (gemini-2.5-pro via `scripts/_consult_round_robin.py`) — feed ChatGPT's answer + this preamble + the sprint plan + the workflow JSON; ask for agreement, corrections, additions.
3. **Claude** — synthesize, flag disagreements, decide the grounded answer.
4. **Loop step 2** if the externals disagree on something material — re-prompt with the disagreement spelled out until they converge or there's enough to break the tie with reasoning.

Save every consultation transcript under `docs/2026-05-14-two-model-selector/` so the design history is auditable.

## Companion artifacts the reviewer should also read

- `workflows/otr_scifi_16gb_full.json` — the canonical workflow JSON this sprint mutates. Read alongside the plan to understand what "wire the writer's broadcast output to the consumer's STRING socket" means in JSON terms.
- `CLAUDE.md` — Prime Directives 1–6 + Bug Log Pipeline + Round-Robin Consultation sections.
- `docs/2026-05-13-two-model-selector-scoping.md` — the original 14-section scoping doc (the plan below is its execution arm, with audit additions).
- `docs/2026-05-14-S29-final-qa-review.md` — what the predecessor cleanbreak sprint just closed; sets the baseline state the plan operates against.

---

# Sprint S30 — Two-Model Selector — Per-Commit Sprint Code Plan

**Date:** 2026-05-14
**Branch target:** `v2.0-alpha` (cut `s30-two-model-selector` from `v2.0-alpha @ HEAD-post-S29-merge`)
**Predecessor:** S29 Clean-Slate Gate closed 2026-05-14 (pytest 2146/8/0, Bug Bible 23/1/2xf, forbidden sweep 0 runtime hits, audio-byte-identical PASS)
**Scoping doc:** `docs/2026-05-13-two-model-selector-scoping.md` (read first — this plan is its execution arm, not a rewrite)
**Sequencing:** Sprint #1 of the B → C → A line. Locks the model-pick surface before C3 flips the audio LLM default.

---

## 0a. NO LEGACY BACK-COMPAT — directive for this sprint

S28 + S29 spent two sprints making the v2.0 contract the only contract. This sprint **must not reintroduce** any of what was just cleaned. Concrete rules:

- **No old-workflow-JSON strip loops.** The existing `cleanup_model_id` legacy-strip loop at `OTR_LedgerScriptWriter.py:2470` is itself legacy back-compat we missed. It gets **deleted** in B2, not extended. Workflow JSONs are rewritten clean by the one-shot migration script and then the script is deleted in the same commit.
- **No silent fallback-to-default on unknown `model_id`.** "Unknown" splits into three cases — each gets a tailored, actionable recovery message:
  - **Not-on-disk but resolvable.** A `model_id` that parses as `org/name` (HuggingFace shape) and passes validation but isn't in the local HF cache → the loader **auto-downloads via `huggingface_hub.snapshot_download`** before failing. This is the public-release-friendly path: a stranger opens the canonical workflow, ComfyUI parses the JSON, the user clicks Queue, the loader resolves the model, sees it's missing, fires `snapshot_download`, ComfyUI shows the download progress in its queue UI, and the workflow runs once the weights land. All subsequent runs are fully offline.
  - **Gated repo, no `HF_TOKEN`.** Detected by **pre-flight** check `model_id in GATED_CURATED_MODELS and resolve_hf_token() is None` BEFORE any `snapshot_download` attempt. `resolve_hf_token()` (from `nodes/_otr_hf_auth.py`, defined in B1a2) checks `os.environ` first, then `HKCU\Environment` via `winreg` (gated on `os.name == "nt"`), then returns `None`. Raises `GatedModelError` with this message shape:
    ```
    GatedModelError: 'mistralai/Mistral-Nemo-Instruct-2407' requires HuggingFace authentication (Mistral license acceptance).
    To run OTR end-to-end as designed, free one-time setup (~5 min):
      1. Create HF account at https://huggingface.co/join.
      2. Accept the license at https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407.
      3. Set HF_TOKEN in your environment.
    Once configured, this download fires automatically on first Queue.
    ```
    **Note on ungated alternatives:** the message deliberately does NOT recommend the catalog's WARN-tier ungated entries (Qwen2.5-14B, Captain-Eris-12B, Mag-Mell-12B) as a fallback. Those models exist in the dropdown for users with bigger rigs but none are 16GB-soak-tested PASS and none have a `load_strategy` for quantization/offload in S30's loader. Recommending them in the error message would advertise a path that probably fails on the target hardware. When a `load_strategy` lands in a future sprint (or one ungated entry reaches PASS in soak), the message updates to include verified alternatives. Until then: honest single-path recovery, not a false-choice menu.

    **Unknown remote repo returning 401/403** (a model_id outside `GATED_CURATED_MODELS` that auto-download attempts and fails on auth) → raises `UnknownModelError` (not `GatedModelError`) with the generic recovery hint — because OTR can't reliably predict gating for arbitrary remote IDs.
  - **Genuinely unknown.** A `model_id` that fails validation (path traversal, banned format, doesn't exist on HF after the download attempt) → fail loud with: `UnknownModelError: model_id 'foo/bar' could not be resolved or downloaded. Reason: <specific>. Install via 'huggingface-cli download foo/bar' once, or pick from your installed set: <top 5 installed>.` Never silently substitutes a different model — silent substitution hides bugs and makes the workflow appear to run with the wrong model.
- **No "stamp both" meta keys.** `meta["model_id"]` is **deleted**. Replaced cleanly by `meta["creative_writing_model"]` and `meta["technical_model"]`. Downstream consumers update in the same commit.
- **No transition shims.** Old node classes, old socket names, old widget names — all deleted, no re-export shims, no `_RENAME_ALIASES` (the dict was killed in S29 and `tests/test_init_aliases_empty.py` enforces it stays dead).
- **No "back-compat parity" tests.** Tests assert the new contract. "Same as single-LLM mode" framing is forbidden — instead phrase it as "Slot 1 == Slot 2 → single model cache reuse", which is a behavior assertion, not a legacy promise.
- **No `OTR_VisualLLMSelector` passthrough.** The node is deleted in B5. No softer landing.
- **No keeping `tests/test_two_llm_split.py` for history.** The file is deleted in B6. New contract gets the new test file. Git history is the historical record.
- **No `legacy_archive/`.** Already a standing directive. Restated: nothing in this sprint creates `legacy_archive/old_workflow_*.json`.

Every commit gets reviewed against this list at the REVIEW step. If a hedge or "for back-compat" comment creeps in, it gets stripped before the commit lands.

---

## 0. Why this plan exists separately from the scoping doc

The scoping doc captured design + 6 open decisions. It was authored alongside S28 (HEAD `f11fee1` at the time). Two things now need reconciliation before execution starts:

1. **S28 + S29 added new code-quality gates** that any new sprint inherits:
   - `_RENAME_ALIASES` dict must not exist (`tests/test_init_aliases_empty.py`)
   - `polish_generate_fn` is now **required** on `polish_line` (S29 Phase 2 deleted the fallback)
   - `NODE_DISPLAY_NAME_MAPPINGS` rejects placeholder strings (`[EMOJI]`, `[TODO]`, `[PLACEHOLDER]`, `[FIXME]`)
   - Test-module-level `EXCLUDED_*` / `ALLOWED_*` collections need per-entry `# justification:` comments
   - `docs/_s28_forbidden_sweep.py` is the canonical extinction-marker gate (run alongside Bug Bible at every commit)
2. **Jeffrey's screenshot (2026-05-13) added a scope item not in the original scoping:** delete the `enable_phase_3_polish` / `polish_announcer_beats` / `enable_phase_4_scene_coherence` / `enable_phase_4_5_smart_suggestion` / `enable_phase_5_voice_drift` / `enable_phase_6_episode_arc` widgets from `OTR_LedgerFreezeCascade` alongside the `model_id` widget. These toggles default OFF in the current workflow JSON; their code paths are effectively dead surface. Per `feedback_no_legacy_back_compat` + S29's deletion-bias precedent, they go too.

Decision locked before kickoff (no B3 review-gate, no Option A/B branch mid-sprint):
- **B3** deletes Freeze Cascade `model_id` widget + the six phase-3..6 toggle widgets. Cascade-only. The shared phase functions in `_otr_lfc.py` stay alive at the close of B3 because the standalone Phase 4/5/6 nodes (deleted in B4) still consume them.
- **B4** atomically deletes the standalone Phase 4/5/6 node files + `__init__.py` registrations + the now-orphaned phase functions in `_otr_lfc.py`. One commit, one deletion pass.

This split keeps shared code never half-deleted: at every commit boundary, either the consumers are alive and the functions are alive (pre-B4), or both are gone (post-B4). No window where the consumers reference deleted functions.

---

## 1. Open decisions — resolved

From scoping §10, with screenshot signals applied:

| # | Decision | Choice | Rationale |
|--:|---|---|---|
| 1 | Shape A / B / C for non-LLM models | **C — writer carries only the two LLM slots** | Screenshot shows only `creative_writing_model` + `technical_model` on writer, no separate hub node. Non-LLM consolidation deferred to a follow-up sprint; in this sprint TTS / SFX / music / video keep current widgets. |
| 2 | Red-state UX | **Path A — `[NOT DOWNLOADED]` label suffix** | Zero JS. Path B JS extension deferred. |
| 3 | Auto-download default | **ON** | With `OTR_MODEL_CATALOG_AUTO_DOWNLOAD=0` env-var off switch for offline scenarios. |
| 4 | `vram_context_test.py` | **Skip in this sprint** | Test bench carve-out documented at B6. |
| 5 | `OTR_VisualLLMSelector` | **Delete** | `feedback_no_legacy_back_compat`. Visual consumers wire to writer's `creative_writing_model` directly. |
| 6 | Slot 2 default | **Same as Slot 1 (`mistralai/Mistral-Nemo-Instruct-2407`)** | Preserves audio C7 byte-identity across the change. Users opt into the split by editing Slot 2. |
| 7 | Phase-3..6 toggles on Freeze Cascade + orphan standalone Phase 4/5/6 nodes | **Delete widgets in B3; delete standalone node files + phase functions in B4** | Locked pre-kickoff: B3 strips cascade widgets only; B4 deletes standalone nodes and the phase functions they consumed. No B3 review-gate decision. |

---

## 2. Verified current code surfaces (HEAD post-S29)

All 7 LLM `model_id` sites from scoping §1a still exist (verified 2026-05-14):

| # | File | Line | Action this sprint |
|--:|---|--:|---|
| 1 | `nodes/OTR_LedgerScriptWriter.py` | 1163 | Replace with `creative_writing_model` + `technical_model` (B2) |
| 2 | `nodes/OTR_LedgerFreezeCascade.py` | 148 | Delete widget + delete phase-3..6 toggles (B3) |
| 3 | `nodes/OTR_LFCPhase4Scene.py` | 83 | DELETE node file + `__init__.py` registration in B4 (Option A locked pre-kickoff; nodes orphaned from canonical workflow). No socket rewire. |
| 4 | `nodes/OTR_LFCPhase5Voice.py` | 64 | DELETE node file + `__init__.py` registration in B4. No socket rewire. |
| 5 | `nodes/OTR_LFCPhase6Arc.py` | 63 | DELETE node file + `__init__.py` registration in B4. No socket rewire. |
| 6 | `nodes/vram_context_test.py` | 138 | **Skip** (carve-out) |
| 7 | `visual/llm_selector.py` | n/a | Delete file (B5) |

Non-LLM sites from §1b (`musicgen_theme.py:393`, `batch_audiogen_generator.py:260`) stay untouched per Shape-C decision.

### 2a-bis. Workflow JSON node-presence reality (verified 2026-05-14)

`grep -l "OTR_LFCPhase\|OTR_VisualLLMSelector\|OTR_VisualPromptCoercion" workflows/*.json` returns **zero matches** across all 8 workflow JSONs. The only LLM-pick nodes actually **placed** in any shipped workflow are:

- `OTR_LedgerScriptWriter` — in `workflows/otr_scifi_16gb_full.json` only.
- `OTR_LedgerFreezeCascade` — in `workflows/otr_scifi_16gb_full.json` only.

The standalone LFC Phase 4/5/6 nodes, `OTR_VisualLLMSelector`, and `OTR_VisualPromptCoercion` are registered in `__init__.py` as draggable into a canvas, but **no shipped workflow JSON references them**. This collapses the JSON-migration scope: only writer + cascade entries need rewriting. B4 and B5 below are Python-only (no JSON wiring) since the affected nodes aren't placed.

The 4 HuMo/LTX downstream smoke JSONs and the 3 `external_examples/*.json` files contain no LLM-pick widgets — they're video-pipeline-only and stay untouched by this sprint. Validation in §4 still runs across all 8 to catch any drift.

### 2a-tris. Verified writer widget order (from `otr_scifi_16gb_full.json`)

Current widget order (index 0 → 17):

```
[ 0] episode_title          ""
[ 1] target_words           350
[ 2] num_characters         2
[ 3] seed                   0
[ 4] seed_mode              "randomize"
[ 5] model_id               "mistralai/Mistral-Nemo-Instruct-2407"     ← target of B2a surgery
[ 6] custom_premise         ""
[ 7] include_act_breaks     true
[ 8] act_count              3
[ 9] style                  "let the story decide"
[10] style_custom           ""
[11] creativity             "balanced"
[12] optimization_profile   "Standard"
[13] perfect_run_spacesaver false
[14] min_p                  0.05
[15] repetition_penalty     1.03
[16] max_new_tokens_cap     200
[17] enable_polish_pass     false
```

Post-B2a order (index 0 → 18) — `creative_writing_model` replaces `model_id` at index 5; `technical_model` inserts at index 6; everything shifts +1 from there:

```
[ 0] episode_title
[ 1] target_words
[ 2] num_characters
[ 3] seed
[ 4] seed_mode
[ 5] creative_writing_model
[ 6] technical_model
[ 7] custom_premise
[ 8] include_act_breaks
[ 9] act_count
[10] style
[11] style_custom
[12] creativity
[13] optimization_profile
[14] perfect_run_spacesaver
[15] min_p
[16] repetition_penalty
[17] max_new_tokens_cap
[18] enable_polish_pass
```

`seed_mode` ("randomize") sits at index 4 between `seed` and the model widget — confirm pin in B6 widget-order test against the writer's actual `INPUT_TYPES()`, not against any inferred ordering.

### 2b. Audit findings — sites the scoping doc missed (2026-05-14)

A full grep across `nodes/`, `visual/`, and `scripts/` for `load_llm(`, `make_generate_fn(`, `make_polish_generate_fn(`, `_OTRML.load_llm`, and `AutoModelForCausalLM.from_pretrained` surfaced three production sites the scoping doc §1a inventory did not list. All three are accounted for in this sprint:

| # | File | Status | Action |
|--:|---|---|---|
| 8 | `nodes/story_orchestrator.py` | **Dead at runtime.** Carries its own `_load_llm` / `_LLM_CACHE` / `_unload_llm` / `_generate_with_llm` stack with 5+ hardcoded `"mistralai/Mistral-Nemo-Instruct-2407"` defaults (L1447, L1549, L1644, L1820, L1911, L1974, L3542, L3557). Production importers (`batch_bark_generator`, `_otr_bark_lib`, `scene_sequencer`, `video_engine`) pull only `_runtime_log` + `_unload_llm`. **No code path calls `_load_llm` at runtime.** | **Delete the LLM stack as commit B0** (pre-B1 cleanbreak). Keep `_runtime_log` + `_unload_llm` (or relocate to a thin helper module). |
| 9 | `visual/llm_polish.py` | Live consumer. Has its **own** `_POLISH_CACHE` separate from `_otr_model_loader.LLM_CACHE`. Calls `AutoModelForCausalLM.from_pretrained` directly at L161 + L172. Currently receives `model_id` from `OTR_VisualLLMSelector` (deleted in B5). | **MANDATORY collapse in B5**: delete `_POLISH_CACHE`; route through `_otr_model_loader.LLM_CACHE`. Two parallel LLM caches on a 16 GB card = guaranteed OOM (Prime Directive 2). Not a round-robin question — locked. Rewire to writer's `creative_writing_model` socket. |
| 10 | `visual/visual_prompt_coercion.py` (+ any future `OTR_VisualDirector` / `OTR_VisualCaptioner` per `llm_selector.py` docstring) | Live consumer of selector. | **Rewire to writer's `creative_writing_model` in B5.** Same wiring path as `llm_polish.py`. |

### 2c. Routing table inside `OTR_LedgerScriptWriter` (extended)

The scoping doc §2c routing table named 9 sub-passes. Five more were missed. Complete list of every LLM call site inside the writer, slot-tagged per Prime Directive 6:

| # | Pass | Slot | Reason |
|--:|---|---|---|
| 1 | Outline | creative | narrative |
| 2 | Cast | creative | narrative |
| 3 | Dialogue composer | creative | narrative |
| 4 | Polish | creative | narrative |
| 5 | WORD_EXTEND rescue | technical | structured |
| 6 | FORMAT_NORM | technical | structured |
| 7 | Grammarian | technical | structured |
| 8 | LLM_RESCUE | technical | structured |
| 9 | ANNOUNCER bookends | technical | structured |
| 10 | **`news_interpreter.build_news_briefs` (D.2.5 stage)** | **technical** | GBNF grammar + pydantic schema + V0-V3 validators = structured |
| 11 | **Style picker pass 1 (inventor — 5 snake_case candidates)** | **creative** | recombines seed flavors creatively from article |
| 12 | **Style picker pass 2 (chooser — picks best with tie-break)** | **technical** | rule-based + GBNF + regex grammar |
| 13 | **Cast contract `_otr_casting` schema validation pass** | **technical** | locked pydantic schema, `extra="forbid"`, JSON validators |
| 14 | **Critic pass (`script_critic.py`)** | **technical** | verdict-style structured output |

Every entry above gets a `# LLM slot: creative` or `# LLM slot: technical` tag comment at its call site as part of B2, per Prime Directive 6 obligation 1.

---

## 3. Per-commit execution sequence

Every commit follows the **REVIEW → CODE → WIRE → REGRESS → COMMIT** cycle. Cycle definition:

- **REVIEW** — re-read the target files, grep for cross-references, confirm scope.
- **CODE** — Python changes.
- **WIRE** — workflow JSON re-write to match the new node surface (Prime Directive 3 — work isn't done until it's wired). Per §2a-bis, only `OTR_LedgerScriptWriter` and `OTR_LedgerFreezeCascade` are placed in any shipped workflow JSON, so JSON wiring lands in exactly two commits: **B2a** (writer widgets + outputs) and **B3** (cascade widgets + technical_model input link). B0 / B1 / B2b / B2c / B4 / B5 / B6 / B7 / B8 are Python-only and their WIRE step is "NONE." The remaining 6 workflow JSONs (HuMo/LTX smokes + external examples) contain no LLM-pick widgets and stay untouched; `tools/validate_workflow_links.py` runs across all 8 to catch any drift.
- **REGRESS** — run the four canonical suites:
  ```
  python -m pytest "C:/Users/jeffr/Documents/ComfyUI/comfyui-custom-node-survival-guide/tests/bug_bible_regression.py" -v
  pytest tests/test_dropdown_guardrails.py -v
  pytest tests/test_core.py -v
  pytest tests/v2/test_audio_byte_identical.py -v
  ```
  Plus the forbidden-pattern sweep: `python docs/_s28_forbidden_sweep.py`.
- **COMMIT** — file-tool path for `COMMIT_EDITMSG` (CLAUDE.md commit-message section), then `git commit -F .git\COMMIT_EDITMSG` via Desktop Commander cmd.

### B0 — Delete `story_orchestrator.py` dead LLM stack (pre-cleanbreak)

**Purpose.** The S28+S29 sweep missed this surface. Eight hardcoded `"mistralai/Mistral-Nemo-Instruct-2407"` defaults sit in dead function signatures; the entire `_load_llm` / `_LLM_CACHE` / `_generate_with_llm` machinery is unreachable at runtime. Removing it before B1 keeps the routing audit honest — no third LLM-load path lurking in the codebase while we centralize on the two slots.

**REVIEW — grep is necessary but NOT sufficient for dead-runtime confirmation.** Grep misses dynamic imports, ComfyUI's class-registration discovery, and call chains where the deleted function is reached via a method dispatch that the static analyzer can't follow. Mandatory three-step audit before any deletion:
1. **Grep step:** `grep -n 'from .story_orchestrator import\|import nodes.story_orchestrator\|story_orchestrator\.' nodes/ scripts/ tests/ visual/` — confirm explicit imports.
2. **Vulture step:** `vulture --min-confidence 80 nodes/story_orchestrator.py` (the same gate S29 used at Phase 5). Functions flagged with high-confidence "unused" are deletion candidates; anything not flagged stays under audit.
3. **Module-import smoke step:** in a clean Python process, run `python -c "from nodes import story_orchestrator; print(dir(story_orchestrator))"` AND boot ComfyUI Desktop once with `nodes.story_orchestrator` registered. Walk the registered nodes table; confirm no `OTR_*` class references any of the to-be-deleted symbols. If a node registers but its `INPUT_TYPES`/`run` references a deleted symbol, the import-time error is caught here, not at first user-load.

Only after all three steps return clean does the deletion land. Read `nodes/story_orchestrator.py` lines 1447, 1549, 1644, 1820, 1911, 1974, 2086, 2114, 3093, 3509, 3542, 3557 to identify every function that contains the hardcoded default or touches `_LLM_CACHE` / `_load_llm`.

**CODE.**
- Delete from `nodes/story_orchestrator.py`:
  - `_load_llm` (L1974) — entire function.
  - `_LLM_CACHE` (module-level dict).
  - `_unload_llm` (L3093) — **DELETE outright, do not relocate.** B1c adds the canonical `_otr_model_loader.unload_llm()` (full teardown per §6). Having two unload primitives (one in `_otr_llm_cache_helpers.py` plus one in `_otr_model_loader.py`) invites semantic drift exactly when the LLM↔video transition needs identical teardown sequences. ONE primitive only. The three current importers (`batch_bark_generator`, `_otr_bark_lib`, `scene_sequencer`) get their `from .story_orchestrator import _unload_llm` rewritten to `from ._otr_model_loader import unload_llm` in this same commit. If `_otr_model_loader.unload_llm` doesn't exist yet at B0 (it lands in B1c), B0 imports a thin forward-shim that B1c replaces with the real symbol; the shim is deleted in B1c so no dual surface ships.
  - `_generate_with_llm` (L3542) — entire function.
  - Every function still in the file that defaults `model_id="mistralai/Mistral-Nemo-Instruct-2407"` but is no longer called → delete it too (dead code, hardcoded default).
- Keep `_runtime_log` and any tests-only symbols (`_LLMTimeoutWorkflowPause`, `_LLMTimeout`, `_check_parse_ok`) for now. They are independent of the LLM stack.
- Update test imports if any test reaches into the deleted symbols (likely none after S29 cleanbreak; verify with grep).
- **`__init__.py` forensic-comment cleanup (folded in here, not deferred to B7).** Grep `__init__.py` for any comment block that mentions `_RENAME_ALIASES`, `OTR_LedgerScriptReviewer`, `Gemma4Director`, "registered as an alias", or other language describing alias / back-compat behavior that contradicts S29's `_RENAME_ALIASES` deletion. Rewrite each such comment to clearly state the alias / class is deleted and NOT supported, or delete the comment outright. Stale forensic comments are exactly what this sprint is trying to kill — they don't survive to B7.

**WIRE.** None — no node-side change, no workflow JSON change. This is a pure deletion pass.

**REGRESS.**
- Full canonical suite. Bug Bible 23/1/2xf must hold.
- Forbidden-pattern sweep: pre-flight unchanged. The sweep's new markers (added in B7) catch the deleted symbols on later commits.
- Audio C7 byte-identical must hold (these functions aren't on the audio path at runtime; the byte-identity is a paranoia gate).

**COMMIT.** Subject: `B0: delete story_orchestrator dead LLM stack + __init__.py forensic-comment cleanup (pre-S30 cleanbreak — S29 audit misses)`

---

### B1a — Catalog scan + validation + tests (no loader change yet)

**Smallest catalog commit.** Pure new module + tests. No edits to `_otr_model_loader.py`, no `unload_llm` / `request_slot` primitives. Those land in B1c. Splitting the catalog work into three commits keeps each diff small and individually revertible.

**REVIEW.** Confirm `_otr_model_loader.py` facade unchanged. Grep for any existing `MODEL_CONTEXT_CAPS` consumers (to be touched in B1b).

**CODE.**
- New `nodes/_otr_model_catalog.py` (~150 LOC, catalog-scan + validator surface only):
  - `CURATED_LLM_MODELS` list — each entry is a `CuratedModel` dataclass with explicit honesty fields:
    ```python
    @dataclass(frozen=True)
    class CuratedModel:
        repo_id: str
        requires_auth: bool                                                   # gated repo → True
        loader_backend: Literal["transformers_safetensors"]                    # only one backend in S30; add new variants when their loader actually exists
        vram_fit_tier: Literal["PASS", "WARN", "UNKNOWN", "FAIL"]              # 16 GB soak-tested status
        approx_safetensors_gb: float                                           # download size, not VRAM
        notes: str
    ```
    **Backend support (S30 ships two loader backends):**
    - `transformers_safetensors` — `AutoModelForCausalLM.from_pretrained` + `AutoTokenizer.from_pretrained`. The standard text-only causal-LM path. Used by Mistral-Nemo, Gemma-2, Qwen2.5, and community 12B entries.
    - `transformers_multimodal_text_only` — `AutoModelForImageTextToText.from_pretrained` + `AutoProcessor.from_pretrained`. Loads multimodal models but feeds only text-input prompts (no images). Used by Gemma-4 / Gemma-3n family (matformer architecture: HF docs show `AutoProcessor` + `AutoModelForImageTextToText` usage; loading these via the standard causal-LM path fails). The loader's `generate_fn` adapter handles the processor → tokenizer interface difference so the writer's call sites don't care which backend is active.

    Initial catalog (curated entries — these annotations drive the dropdown labels, error messages, and backend dispatch):
    | repo_id | requires_auth | vram_fit_tier (16 GB) | ~size GB | loader_backend | notes |
    |---|---|---|---|---|---|
    | `mistralai/Mistral-Nemo-Instruct-2407` | YES (Mistral license) | **PASS** | ~24 | `transformers_safetensors` | Audio C7 regression baseline — soak-tested. Default for both slots. |
    | `google/gemma-4-E2B-it` | YES (Google license) | PASS | ~6 | `transformers_multimodal_text_only` | Multimodal architecture (matformer / Gemma-3n family) used in text-only mode. Compact technical-slot option. |
    | `google/gemma-4-E4B-it` | YES (Google license) | PASS | ~9 | `transformers_multimodal_text_only` | Slightly larger technical option, same backend. |
    | `Qwen/Qwen2.5-14B-Instruct` | NO (Apache 2.0) | **WARN** | ~28 | `transformers_safetensors` | Ungated; 14B safetensors needs quantization or offload to fit 16 GB — not soak-tested as PASS yet. Available in dropdown for users with bigger rigs; NOT advertised in gated-error recovery hint until a `load_strategy` lands. |
    | `Nitral-AI/Captain-Eris_Violet-V0.420-12B` | NO (community) | **WARN** | ~24 | `transformers_safetensors` | Ungated community; 12B at the edge, not soak-tested. Same gated-error-hint exclusion. |
    | `inflatebot/MN-12B-Mag-Mell-R1` | NO (community) | **WARN** | ~24 | `transformers_safetensors` | Ungated community; same caveat. Same gated-error-hint exclusion. |

    **Non-curated locally-scanned models** still admit through `validate_model_id` (curated + locally-scanned + valid `org/name`-with-auto-download — the three admit-paths). A user who already has Gemma-2, Llama-3, or any other text-only causal-LM in their `HF_HOME` cache can pick it from the dropdown; it just isn't recommended/curated. The locally-scanned path defaults to `transformers_safetensors` backend; uncurated models requiring the multimodal path will fail at load with a clear backend-mismatch error.

  **Backend compatibility check (pre-catalog-lock):** before any entry ships in the catalog, the implementation runs a one-time backend smoke test that instantiates the catalog-declared model class on a minimal config slice. If the repo requires a different class than its declared `loader_backend` claims, it gets rejected from the catalog with a clear log line. This guardrail catches future Gemma-4-vs-causal-LM mismatches automatically rather than at first user-load.

  **Honesty rule:** only entries with `vram_fit_tier=="PASS"` are advertised in dropdown labels and error messages as "16 GB-ready." WARN entries get labeled `[16GB-FIT: UNTESTED]`. The error-message recovery hint for gated-without-token only proposes PASS-tier ungated alternatives; if no ungated PASS entries exist yet, the message says so plainly: "No ungated curated alternatives are 16 GB-soak-tested yet. Pick Mistral-Nemo (recommended) or accept your Qwen/community pick may need quantization for your hardware."

  **First soak goal post-sprint:** validate at least one ungated PASS-tier entry so the first-run-without-token path has a confident recommendation. Until then, the honest message admits the gap.
  - `scan_local_llm_cache()` — walks `HF_HOME/hub/models--*/snapshots/*`. For each snapshot, reads `config.json` and extracts `max_position_embeddings` (or `n_positions` / `n_ctx` fallback for non-Llama-shape configs). Returns a list of `ScanResult(repo_id, on_disk, snapshot_path, advertised_context)`.
  - `build_dropdown_choices() -> list[DropdownEntry]` — merges curated + scanned, applies `[NOT DOWNLOADED]` suffix to curated entries not on disk.
  - `validate_model_id(model_id: str) -> str` — strips `[NOT DOWNLOADED]` suffix, runs structural rejection (path-traversal, drive letters, backslashes, unsafe formats), and then admits the id if it matches any of:
    1. **Curated** — `model_id in CURATED_LLM_MODELS`.
    2. **Locally scanned** — `model_id` corresponds to a folder under `HF_HOME/hub/models--*/snapshots/*` (the user already has it).
    3. **Arbitrary `org/name`** matching the HF repo-id regex, only when `auto_download` is enabled (default ON via `OTR_MODEL_CATALOG_AUTO_DOWNLOAD=1`) OR when `OTR_MODEL_CATALOG_ALLOW_REMOTE=1` is set.
    If none of the three admit-paths match: raises `UnknownModelError` with the actionable recovery message (the exact text in §0a). No silent fallback. No WARN-log substitution. The point of this expansion: hardware-inclusivity means a user can slot in any HF model their rig handles, not just OTR's curated picks — but structural-rejection rules (path traversal, drive letters, unsafe formats) still raise regardless.
  - `GATED_CURATED_MODELS: frozenset[str]` — derived from `CURATED_LLM_MODELS` entries where `requires_auth=True`. Used by B1a2's pre-flight gated check.
  - Module-level constants used by tests / wiring code so no test ever hardcodes a repo ID string:
    - `DEFAULT_LLM = "mistralai/Mistral-Nemo-Instruct-2407"` — default for both writer slots; matches audio C7 baseline.
    - `TEST_TECHNICAL_LLM = "google/gemma-4-E2B-it"` — used by B6 routing tests + the manual VRAM-profile script to drive Slot 1 ≠ Slot 2.
    - `TEST_OVERSIZED_LLM = "meta-llama/Llama-3.1-70B-Instruct"` — used by B1c VRAM-fit tests as a known-fails-on-16GB target. Not added to the dropdown.
    Any future renames / casing fixes happen in one place. Catalog is the single source of truth.
- New `nodes/_otr_model_inputs.py` (~30 LOC, just the error classes that B1a's validator raises + a placeholder for B1a2):
  - `class MissingModelInputError(RuntimeError)` — raised by consumers when the socket is unwired.
  - `class UnknownModelError(RuntimeError)` — raised by `validate_model_id` on miss (genuinely unknown / banned-format).
  - `def require_model(model_id: str | None, *, slot: str) -> str` — shared resolver / fail-loud helper used by every consumer socket.

  (`GatedModelError` and `InsufficientDiskSpaceError` get added in B1a2 alongside `auto_download_if_missing`.)
- New `tests/test_model_catalog_scan.py` (~90 LOC, offline-only):
  - Catalog returns curated entries even when local cache is empty.
  - `scan_local_llm_cache` walks the fixture HF cache and returns `ScanResult` entries with correct `repo_id` / `on_disk` / `snapshot_path`.
  - `build_dropdown_choices` applies `[NOT DOWNLOADED]` suffix to curated entries not in the local scan.
  - Validator strips `[NOT DOWNLOADED]` suffix before allow-list check.
  - Validator rejects `..`, absolute paths, drive letters, backslashes (structural rejection).
  - Validator raises `UnknownModelError` with recovery-hint message when a `model_id` matches none of curated / locally-scanned / valid `org/name` admit-paths.
  - `pytest-httpx` strict mode: zero outbound HTTP fires during the entire B1a test run.

  (Auto-download tests + `GatedModelError` tests + disk-space pre-check tests + no-required-keys end-to-end smoke + fast-path-during-download-path tests live in B1a2's `tests/test_model_catalog_download.py`.)

**WIRE.** Nothing. Pure module + tests. No `_otr_model_loader.py` touch. **B1a is offline-only**: catalog dataclass + scan + dropdown merge + validator. No HF API surface, no download.

**REGRESS.** Run all four canonical suites + the new catalog test file. Bug Bible 23/1/2xf must hold. `pytest-httpx` strict mode asserts zero outbound HTTP fires during the B1a test run.

**COMMIT.** Subject: `B1a: catalog dataclass + scan_local_llm_cache + dropdown choices + validator (offline-only)`

---

### B1a2 — `auto_download_if_missing` + size estimate + disk pre-check + `GatedModelError`

Splits the network surface off of B1a so failures isolate cleanly — the offline B1a can ship green even if the HF API layer needs round-robin iteration.

**REVIEW.**
- Read `huggingface_hub.snapshot_download` + `HfApi().model_info` signatures + error shapes.
- Round-robin trigger: pre-flight gated detection vs. catching `HfHubHTTPError 401/403` — confirm the right surface (see `GatedModelError` design below).
- Grep for any existing `HF_TOKEN` resolution in `nodes/` and `visual/` to share the `resolve_hf_token()` helper, not duplicate.

**CODE.**
- New helper `nodes/_otr_hf_auth.py::resolve_hf_token() -> str | None`:
  - Check `os.environ.get("HF_TOKEN")` first.
  - **Cross-platform gate:** the `winreg` fallback runs ONLY when `os.name == "nt"`. `import winreg` throws `ImportError` on macOS/Linux; gating both the import and the registry lookup avoids crashing the node pack on non-Windows systems. Skeleton:
    ```python
    def resolve_hf_token() -> str | None:
        env_token = os.environ.get("HF_TOKEN")
        if env_token:
            return env_token
        if os.name == "nt":
            try:
                import winreg
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
                    value, _ = winreg.QueryValueEx(key, "HF_TOKEN")
                    return value or None
            except (ImportError, FileNotFoundError, OSError):
                return None
        return None
    ```
  - Returns `None` if neither resolves. Token-presence checks throughout the codebase use this helper, never `os.environ` directly.
- `nodes/_otr_model_catalog.py` (extending B1a):
  - `estimate_model_size_gb(repo_id)` — free public HF API call; constrained to user-action paths per the rules in §B1a (which now point at this commit).
  - `auto_download_if_missing(repo_id)` — full implementation per the auto-download flow already documented in the previous B1a section. Default ON. `OTR_MODEL_CATALOG_AUTO_DOWNLOAD=0` env-var off switch.
  - **Pre-flight gated check** runs BEFORE `snapshot_download`:
    ```python
    if repo_id in GATED_CURATED_MODELS and resolve_hf_token() is None:
        raise GatedModelError(_format_gated_message(repo_id))
    ```
    This guarantees the user gets the helpful dual-path `GatedModelError` instead of a generic 401 from `snapshot_download`. The 401-from-unknown-remote path still raises `UnknownModelError` because OTR can't reliably predict gating for arbitrary remote IDs.
  - Pre-fetch disk-space + size-estimate + queue-UI announcement (as documented in §B1a's existing auto-download flow).
  - **Download-progress wiring (non-trivial, must be explicit):** `huggingface_hub.snapshot_download` is synchronous; the node's `run()` method runs in ComfyUI's worker thread so the UI thread isn't blocked, but the user sees no progress signal without explicit wiring. Implementation:
    ```python
    from comfy.utils import ProgressBar
    import huggingface_hub

    def _hf_progress_callback(downloaded_bytes: int, total_bytes: int, pbar: ProgressBar):
        if total_bytes > 0:
            pbar.update_absolute(downloaded_bytes, total_bytes)

    def auto_download_if_missing(repo_id: str):
        # ... pre-flight checks (gated, disk space, etc.) ...
        total_gb = estimate_model_size_gb(repo_id)
        pbar = ProgressBar(int(total_gb * 1024**3))
        # huggingface_hub supports tqdm_class override; route its progress through pbar
        huggingface_hub.snapshot_download(
            repo_id=repo_id,
            allow_patterns=ALLOW_PATTERNS,
            tqdm_class=_make_pbar_tqdm_adapter(pbar),
            token=resolve_hf_token(),
        )
    ```
    The `_make_pbar_tqdm_adapter` returns a small class that satisfies `huggingface_hub`'s tqdm interface (`update`, `set_description`, `__enter__`, `__exit__`) and forwards to ComfyUI's `ProgressBar`. Users see a real progress bar in the queue UI during the 24 GB Mistral-Nemo first-run download, not a frozen "running" spinner.
- `nodes/_otr_model_inputs.py`: the four error classes (`MissingModelInputError`, `UnknownModelError`, `GatedModelError`, `InsufficientDiskSpaceError`) all live here. `MissingModelInputError` was already defined; the others added here in B1a2.
- New `tests/test_model_catalog_download.py`:
  - All `auto_download_if_missing` tests (including `OTR_MODEL_CATALOG_AUTO_DOWNLOAD=0` respect, disk-space pre-check, `GatedModelError` pre-flight, no-required-keys end-to-end smoke).
  - `resolve_hf_token` resolves from `os.environ` first, then `HKCU` via `winreg`, then returns `None`. Mocked registry access for cross-platform CI.
  - `pytest-httpx` strict mode for fast-path discipline: scan / dropdown / `INPUT_TYPES` / startup paths fire ZERO HF API calls; `auto_download_if_missing` IS allowed to fire calls (verified via mock).

**WIRE.** Nothing.

**REGRESS.** All four canonical suites. Bug Bible 23/1/2xf must hold.

**COMMIT.** Subject: `B1a2: auto_download_if_missing + size estimate + disk pre-check + GatedModelError + resolve_hf_token`

---

### B1b — Dynamic context-cap replacement (kills the static dict)

Moves the static `MODEL_CONTEXT_CAPS` dict + `DEFAULT_CONTEXT_CAP = 8192` constant out of `_otr_model_loader.py` and replaces them with the dynamic + hardware-clamped catalog lookup. Isolated commit so a regression in context handling attributes cleanly.

**REVIEW.** Grep every caller of `MODEL_CONTEXT_CAPS` and `DEFAULT_CONTEXT_CAP` in `nodes/`, `visual/`, and `tests/`. Confirm all callers funnel through the loader's `context_cap` accessor (not direct dict reads).

**CODE.**
- `nodes/_otr_model_catalog.py` (extending the surface from B1a):
  - `CURATED_CONTEXT_OVERRIDES: dict[str, int]` — explicit per-model effective-context overrides for the curated set ONLY (where the model's `config.json` advertises a context window larger than what the inference pipeline can sanely feed). Defaults to empty; populated as soak-tested.
  - `HARD_VRAM_CONTEXT_LIMIT: int` — hardware-aware ceiling on the effective context window, defaulting to **8192 on the 5080 16 GB target**. Configurable via the `OTR_HARD_VRAM_CONTEXT_LIMIT` env var so a user on bigger hardware (24 GB / 32 GB / multi-GPU) can raise it. Rationale: a small model like Gemma-4-E4B advertises a 128k context in its `config.json`, but feeding 128k tokens through it on a 16 GB card OOMs instantly. The hardware ceiling clamps the upper bound regardless of what the model says it can do.
  - `resolve_context_cap(model_id: str) -> ContextCapVerdict` — returns a tiered verdict mirroring `VRAMFitVerdict`, **not raise**. Verdict struct: `(tier: Literal["PASS", "WARN", "UNKNOWN"], value: int, source: str)`. Tiers:
    - `PASS` — `CURATED_CONTEXT_OVERRIDES[model_id]` is present (soak-tested cap). `value = min(override, HARD_VRAM_CONTEXT_LIMIT)`.
    - `WARN` — `config.json` parses cleanly via `max_position_embeddings` / `n_positions` / `n_ctx` but model isn't curated. `value = min(parsed, HARD_VRAM_CONTEXT_LIMIT)`.
    - `UNKNOWN` — neither source resolves. `value = HARD_VRAM_CONTEXT_LIMIT` (the only safe default — clamp to the hardware ceiling rather than raise, since `check_vram_fit` will also return UNKNOWN for the same model and the combined verdict makes the escalation decision).

    **Why tiered, not raise:** `request_slot` runs `resolve_context_cap` before `check_vram_fit`. If `resolve_context_cap` raised on UNKNOWN, an unparseable model would die at the context-cap gate before the more-permissive fit verdict ever ran — the two systems would disagree on whether to fail-loud or proceed-with-warn for the same input. Aligning both to verdict-tiered semantics + a single escalation decision in `request_slot` (see B1c) keeps the policy coherent.

    The clamp still solves both real failure modes: "model says 4k but we feed 8k" (PASS/WARN: use parsed, clamped down to hardware limit if needed) and "model says 128k, we'd OOM on 16 GB" (PASS/WARN: clamped to limit).
- `nodes/_otr_model_inputs.py`: add `class ContextCapUnknownError(RuntimeError)`.
- `nodes/_otr_model_loader.py`:
  - **Delete** the module-level `MODEL_CONTEXT_CAPS` dict and the `DEFAULT_CONTEXT_CAP = 8192` constant.
  - Replace every `context_cap = MODEL_CONTEXT_CAPS.get(model_id, DEFAULT_CONTEXT_CAP)` lookup with `context_cap = _otr_model_catalog.resolve_context_cap(model_id)`.
- `tests/test_model_catalog_scan.py` (extend):
  - `resolve_context_cap` returns curated override when present.
  - `resolve_context_cap` reads `max_position_embeddings` from a fixture `config.json` when no override.
  - `resolve_context_cap` clamps to `HARD_VRAM_CONTEXT_LIMIT` when advertised > limit (the Gemma-4-E4B 128k case).
  - `resolve_context_cap` returns the advertised value when advertised < limit (the "model says 4k" case).
  - `resolve_context_cap` raises `ContextCapUnknownError` when neither source resolves (no blind 8192 fallback).

**WIRE.** Nothing. Pure Python.

**REGRESS.**
- All four canonical suites.
- **Audio C7 byte-identical must hold** — same model defaults, same context_cap resolution (Mistral-Nemo's advertised ~128k clamped to 8192 should match what the prior static dict was returning). If drift, the static dict was returning a non-8192 value the catalog isn't matching; reconcile before commit.

**COMMIT.** Subject: `B1b: dynamic context-cap (config.json + HARD_VRAM_CONTEXT_LIMIT clamp); delete MODEL_CONTEXT_CAPS / DEFAULT_CONTEXT_CAP`

---

### B1c — Loader slot primitives: `unload_llm` + `request_slot` + `check_vram_fit`

The slot scheduler's load-bearing primitives. Splitting this from B1a/B1b keeps the catalog work cleanly separate from the loader's runtime control surface.

**REVIEW.** Read `nodes/_otr_model_loader.py` in full. Identify the existing single-model load/unload pattern. Confirm `_flush_vram_keep_llm()` lives there (used only for same-model phase transitions).

**CODE.**
- `nodes/_otr_model_loader.py`:
  - **Add `unload_llm()`** — full teardown per §6: `model.to("cpu")` → drop refs → `gc.collect()` → `torch.cuda.empty_cache()` → `torch.cuda.ipc_collect()` → `torch.cuda.synchronize()`. Logs VRAM-after.
  - **Add `request_slot(slot_name: Literal["creative", "technical"], model_id: str) -> CacheEntry`** — the slot-aware entry point. Explicit sequence:
    1. `model_id = validate_model_id(model_id)` — raises `UnknownModelError` if `model_id` fails structural rejection or doesn't match any admit-path. **Returns the normalized repo ID** (strips `[NOT DOWNLOADED]` suffix), so downstream steps see canonical IDs regardless of what the widget value contained.
    2. If currently-loaded model `== model_id`, return the cached entry. Done. (Steps 3–7 only run on a slot transition.)
    3. `auto_download_if_missing(model_id)` — raises `GatedModelError` pre-flight (B1a2) if gated + no token; otherwise fetches lightweight `HfApi().model_info` first, runs `check_vram_fit` verdict against the parsed config, and only proceeds to full `snapshot_download` if verdict is not `FAIL`. See P4 below for the "fetch config, fit-check, then weights" ordering rule.
    4. `ctx_verdict = resolve_context_cap(model_id)` — tiered `ContextCapVerdict`, never raises (per P0-4 alignment).
    5. `fit_verdict = check_vram_fit(model_id, ctx_verdict.value)` — tiered `VRAMFitVerdict`.
    6. **Combined escalation decision** (single coherent policy point):
       - `fit_verdict.tier == "FAIL"` → raise `VRAMFitFailedError` with both verdicts in the message.
       - `fit_verdict.tier == "PASS"` and `ctx_verdict.tier == "PASS"` → quiet success path.
       - Any other combination (WARN/UNKNOWN on either side) → emit a single combined log line: `[Selector] proceeding with caution: ctx_cap={ctx_verdict.tier}@{value}, vram_fit={fit_verdict.tier}@{estimate_gb} GB`. Proceed to step 7.
    7. `unload_llm()` (if a different model was previously resident), then `load_llm(model_id)`.
    8. Log the transition with elapsed time + final verdict tiers.

    **P4 (oversize-download avoidance) rule for `auto_download_if_missing`:** for arbitrary remote `org/name` (not curated), fetch `model_info` (lightweight) FIRST, parse advertised param count from `model_info.config`, run `check_vram_fit` against the estimate; if verdict is `FAIL`, raise `VRAMFitFailedError` **before** any `snapshot_download` of weights. Avoids downloading 80 GB of a 70B model only to reject it at load time. Curated `PASS`-tier entries skip the pre-check fetch since their verdict is already cached.
  - Keep `_flush_vram_keep_llm()` for same-model phase transitions; do not call it from `request_slot`.
- `nodes/_otr_model_catalog.py` (extend):
  - `check_vram_fit(model_id: str, context_cap: int) -> VRAMFitVerdict` — tiered verdict, not a binary. Returns one of: `PASS` (curated model with known soak-tested profile fits the budget — quiet success), `WARN` (scanned arbitrary model; rough math suggests it fits but the profile isn't soak-tested — log a warning, let the load proceed), `UNKNOWN` (scanned model whose param count / dtype / quantization can't be reliably parsed — log informational message, let the load proceed and rely on the OOM safety net), `FAIL` (rough math clearly exceeds the budget by ≥1.5× — fail loud with the estimate-vs-ceiling reason). Hard-failing is reserved for the obvious 70B-on-16GB case; uncurated models with ambiguous size data don't get blocked on potentially-wrong math. Verdict struct includes `(tier, estimated_gb, ceiling_gb, reason, soak_tested: bool)`. A 70B model on a 16 GB card returns `FAIL` with `"estimated 42 GB peak resident vs 14.5 GB ceiling — pick a smaller model"`; an uncurated 13B model with unknown quantization returns `WARN`.

    **Honesty note on `UNKNOWN`:** HuggingFace `config.json` has no standardized `num_parameters` field. Inferring from `hidden_size × num_hidden_layers × ...` is architecture-dependent and frequently unreliable. **`UNKNOWN` is the expected verdict for most uncurated arbitrary `org/name` models**, not the exception. The function is a coarse guardrail against the obvious oversize case, not a precise VRAM oracle. Curated PASS-tier entries are the only models where the verdict is trustworthy; for everything else the load proceeds with a WARN/UNKNOWN log and relies on the runtime OOM safety net + `LibreHardwareMonitor` for the real-time signal.
- New `tests/test_loader_slot_primitives.py` (~80 LOC):
  **Mandatory hard-mock fixture spec (prevents real-world download/load disasters in pytest):** all B1c loader tests use an `autouse=True` fixture in `conftest.py` that patches the following symbols before any test runs. Missing any one of these patches risks pytest triggering a real 24 GB download or a real `from_pretrained` GPU load:
  ```python
  @pytest.fixture(autouse=True)
  def _hard_mock_loader_paths(monkeypatch):
      monkeypatch.setattr("huggingface_hub.snapshot_download", lambda *a, **kw: "/fake/snapshot/path")
      monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained", _FakeModelFactory())
      monkeypatch.setattr("transformers.AutoModelForImageTextToText.from_pretrained", _FakeMultimodalFactory())
      monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", _FakeTokenizerFactory())
      monkeypatch.setattr("transformers.AutoProcessor.from_pretrained", _FakeProcessorFactory())
      monkeypatch.setattr("torch.cuda.max_memory_allocated", lambda *a, **kw: 0)
      monkeypatch.setattr("torch.cuda.empty_cache", lambda: None)
      monkeypatch.setattr("torch.cuda.ipc_collect", lambda: None)
      monkeypatch.setattr("torch.cuda.synchronize", lambda: None)
      monkeypatch.setattr("_otr_model_loader.LLM_CACHE", _FakeCacheRegistry())
  ```
  `_FakeModelFactory` / `_FakeTokenizerFactory` / etc. are minimal stand-ins that record their `(repo_id, kwargs)` call args for assertion + return predictable mock objects with a `.to(device)` method that doesn't actually move tensors. Any test that fails to import this fixture is **rejected at code review**.

  - `test_request_slot_same_model_returns_cached_entry` — calls `request_slot("creative", catalog.DEFAULT_LLM)` twice; assert second call doesn't fire `unload_llm`. Full repo IDs only; lowercase shorthand would fail catalog validation. Mocks ensure no real download.
  - `test_request_slot_different_model_triggers_full_teardown` — calls `request_slot("creative", catalog.DEFAULT_LLM)` then `request_slot("technical", catalog.TEST_TECHNICAL_LLM)`; assert `unload_llm` fires exactly once between them.
  - `test_check_vram_fit_oversized_returns_FAIL_verdict` — uses `catalog.TEST_OVERSIZED_LLM` (a catalog constant pointing at a known 70B model used only by tests; lives in `_otr_model_catalog.py` next to `DEFAULT_LLM`/`TEST_TECHNICAL_LLM`); asserts `check_vram_fit(TEST_OVERSIZED_LLM, 8192).tier == "FAIL"`; verdict struct has `estimated_gb`, `ceiling_gb`, and a human-readable `reason`. No raw repo IDs in the test — catalog is the source of truth.
  - `test_check_vram_fit_curated_mistral_nemo_returns_PASS` — `check_vram_fit(catalog.DEFAULT_LLM, 8192).tier == "PASS"`; only curated soak-tested entries should be `PASS`.
  - `test_check_vram_fit_uncurated_returns_WARN_or_UNKNOWN` — fixture model with parseable param count but no soak profile returns `WARN`; fixture with unparseable config returns `UNKNOWN`. Neither hard-fails the load.
  - `test_unload_llm_calls_ipc_collect` — mock CUDA primitives, assert the teardown sequence calls `empty_cache`, `ipc_collect`, `synchronize` in that order.

**WIRE.** Nothing.

**REGRESS.** All four canonical suites. Audio C7 byte-identical (default workflow doesn't trigger slot transitions yet since neither slot widget exists; baseline behavior unchanged).

**COMMIT.** Subject: `B1c: loader slot primitives (unload_llm + request_slot + check_vram_fit)`

---

### B2a — Writer two-widget surface + output sockets (writer JSON wired here)

**Smallest commit that produces a working dual-slot writer.** No internal-routing change yet. **Strict rule for B2a:** the writer MUST NOT call `request_slot("technical", ...)` anywhere in its internal flow. Internally, every LLM call site still routes through the same legacy generation path using `creative_writing_model` only. `technical_model` is **surfaced** (widget + output socket) and **broadcast** (its value flows out the output STRING socket for downstream consumers to read in B3+B4) but is **NOT consumed internally** by the writer until B2b. This keeps B2a a pure widget-surface change and the audio C7 byte-identity gate easy to attribute if anything moves.

**REVIEW.**
- Read `nodes/OTR_LedgerScriptWriter.py` lines 1163, 120, 330, 857, 1399 in full. The legacy-strip loop at 2470 is **not** touched here — that's B2c's job.
- Confirm `_otr_model_catalog.DEFAULT_LLM` resolves to `"mistralai/Mistral-Nemo-Instruct-2407"`.

**CODE.**
- `nodes/OTR_LedgerScriptWriter.py`:
  - Delete `DEFAULT_MODEL_ID` literal (line 120) — replaced by catalog default.
  - Delete `_MODEL_CHOICES` literal (line 330) — replaced by catalog import.
  - Replace `model_id` widget (line 1163) with two widgets:
    ```python
    "creative_writing_model": (_otr_model_catalog.dropdown_choices(), {"default": _otr_model_catalog.DEFAULT_LLM}),
    "technical_model":        (_otr_model_catalog.dropdown_choices(), {"default": _otr_model_catalog.DEFAULT_LLM}),
    ```
  - Update `run()` signature: rename `model_id` → `creative_writing_model`, add `technical_model`. Both still feed the **same** legacy generation path for now — internal routing change lands in B2b.
  - Add output sockets `creative_writing_model` (STRING) + `technical_model` (STRING) at the end of the `OUTPUT_NAMES` / `RETURN_TYPES` list. **Critical: outputs broadcast the `validate_model_id()`-normalized repo ID, NEVER the raw widget value.** ComfyUI dropdowns persist the literal string the user selected, including any `[NOT DOWNLOADED]` suffix. Returning the raw value would propagate the suffix downstream to consumers and to `meta` keys, breaking validation everywhere it lands. Concrete rule applied at every output site:
    ```python
    creative_id = _otr_model_catalog.validate_model_id(self.creative_writing_model)  # strips suffix, normalizes
    technical_id = _otr_model_catalog.validate_model_id(self.technical_model)
    return (..., creative_id, technical_id)
    ```
  - `meta["creative_writing_model"]` and `meta["technical_model"]` stamps also use the normalized IDs (same rule applies in B2b's routing surgery). No `[NOT DOWNLOADED]` substrings ever land in `meta`.

**WIRE.** `workflows/otr_scifi_16gb_full.json` — writer node only (only canonical has the writer placed; smoke / external workflows have no writer):
- `widgets_values`: insert `"mistralai/Mistral-Nemo-Instruct-2407"` at index 6 (the new `technical_model` slot) so the final order matches §2a-tris post-B2a. The existing value at index 5 stays — it's now `creative_writing_model`. Indices 6..17 shift +1 to 7..18.
- `outputs`: append two new entries with **explicit `slot_index` values** (ComfyUI's JSON schema requires these per output):
  ```json
  {"name": "creative_writing_model", "type": "STRING", "links": null, "slot_index": 4},
  {"name": "technical_model",        "type": "STRING", "links": null, "slot_index": 5}
  ```
  The existing four outputs (`script_text` / `script_json` / `news_used` / `estimated_minutes`) keep slot_index 0..3. New outputs occupy 4 and 5.
- `links` stays `null` for both new outputs in B2a — downstream consumers wire in B3 (cascade gets `technical_model`) and downstream visual consumers wire when they get placed.
- `last_link_id` at the graph root: **unchanged in B2a** since no new links land yet. B3 bumps it.
- Other workflow JSONs untouched (no writer placed).
- `tools/validate_workflow_links.py` across all 8 JSONs — 0 violations.
- New B2a guardrail test: assert writer's output 4 = `creative_writing_model` and output 5 = `technical_model` against the JSON.

**REGRESS.**
- All four canonical suites.
- **Audio C7 byte-identical** — both slots default to the same Mistral-Nemo and feed the unchanged generation path, so audio output must be byte-identical to the pre-B2a baseline. If drift → revert immediately per Prime Directive 1. Drift here points at the widget surgery, not at routing (routing hasn't changed yet).
- `tests/test_workflow_json_guardrails.py::TestWriterStyleSentinelDefault` — fix widget index for `style` in lockstep (shifted +1).
- New `tests/test_writer_b2a_surface_only.py`: AST-walk `OTR_LedgerScriptWriter.run` (and any helpers it calls within the writer module); assert ZERO calls to `_request_slot("technical", ...)` or `request_slot(slot="technical", ...)` exist in this commit's code. The internal routing change is B2b's job; if B2a leaks a technical-slot call, the audio C7 attribution at B2b becomes ambiguous.

**COMMIT.** Subject: `B2a: writer two-widget surface + output sockets (single generation path; technical_model output-only)`

---

### B2b — Writer internal creative/technical routing + new meta keys (Python only)

**REVIEW.**
- Map every LLM call site in `run()` and tag each as creative or technical against the §2c routing table (14 sub-passes).
- Identify hard data dependencies (`news_interpreter` → outline; `cast_contract` → outline; per-beat `critic` → polish). These are the unavoidable interleaving points the slot scheduler has to honor.
- Round-robin trigger (per §5): writer slot scheduler design — confirm partial-batching against the dependency DAG; confirm `unload_llm()` (not `_flush_vram_keep_llm()`) is the correct primitive at slot transitions per §6.

**CODE.**
- `nodes/OTR_LedgerScriptWriter.py`:
  - Replace every direct `load_llm(model_id=...)` / `make_generate_fn(...)` call in `run()` with calls through a new private method `_request_slot(slot: Literal["creative", "technical"]) -> CacheEntry`:
    - Resolves the slot to either `creative_writing_model` or `technical_model` widget value.
    - Calls `_otr_model_loader.request_slot(slot, resolved_model_id)` which handles cache reuse vs. full teardown automatically.
    - Tags the call site with `# LLM slot: creative` or `# LLM slot: technical` per Prime Directive 6.
  - Apply partial-batching where the dependency DAG allows per §6. Document each unavoidable interleave point with `# slot-interleave: <prior> -> <next>` naming the data dependency that forces it. Acceptance is dual-mode (per §6):
    - **Per-beat default mode** (preserves critic→polish feedback loop): `meta["slot_transitions"] == DOCUMENTED_DAG_MIN_PER_BEAT` (~9 for a 3-beat fixture episode). Failing higher means batching opportunities are being missed.
    - **`OTR_BATCH_PER_BEAT=1` opt-in mode** (loses per-beat feedback for VRAM-pressured rigs): `meta["slot_transitions"] <= 3`.
  - **Delete the `meta["model_id"]` stamp.** Replace with two top-level keys (`meta["creative_writing_model"]` + `meta["technical_model"]`, the resolved repo IDs from the writer's widget surface) AND a per-phase stamp under `meta["gen_params_by_phase"][<phase>]`:
    ```python
    meta["gen_params_by_phase"]["outline"] = {
        "slot":  "creative",                              # which slot this phase routed through
        "model": "mistralai/Mistral-Nemo-Instruct-2407",  # resolved repo id (handy for forensics)
        # ...existing per-phase fields (temperature, max_new_tokens, etc.)
    }
    ```
    Explicit two fields, not a concatenated key name. No "stamp both legacy + new" hedge. Every downstream `meta["model_id"]` reader gets updated in this same commit (grep before commit).
- Routing per §2c table:
  - **Creative:** outline, cast, dialogue composer (per beat), polish (per line), style picker pass 1.
  - **Technical:** news_interpreter, style picker pass 2, cast contract validator, critic (per beat), ANNOUNCER bookends, FORMAT_NORM, LLM_RESCUE, WORD_EXTEND, Grammarian.

**WIRE.** None — pure Python. JSON unchanged from B2a end state.

**REGRESS.**
- All four canonical suites.
- **Audio C7 byte-identical** — both slots default to the same model and the loader caches one model regardless of slot, so audio output must still be byte-identical. Drift here points at routing, not widget surface.
- New `tests/test_writer_slot_scheduler.py`: drive `_request_slot()` with a mock loader; assert `unload_llm()` is called exactly at slot transitions and never within same-slot blocks. Transition-count assertions are **computed from the actual enabled-phase graph**, not hardcoded numbers (per P1-5 fix: hardcoded `== 9` / `== 3` is brittle because it depends on episode beats / polish on/off / critic enabled / rescue path / fixture content):
  - Build a `compute_expected_transitions(fixture_config) -> int` helper that walks the enabled-phase DAG (which phases run given the fixture's `enable_polish_pass`, beat count, critic flags, etc.) and counts the actual creative↔technical slot transitions. Use this for the per-beat-default expectation.
  - Per-beat-default fixture (3-beat episode, polish OFF — matching canonical workflow): `meta["slot_transitions"] == compute_expected_transitions(fixture)`. Also assert phase-block order matches the documented DAG (creative-block before per-beat T→C cycle before final-technical-block, etc.) via a recorded phase trace.
  - `OTR_BATCH_PER_BEAT=1` fixture: `meta["slot_transitions"] <= 3` (upper bound — the batched mode collapses to exactly 3 transitions for the canonical writer pipeline, but the assertion is an upper bound to tolerate future scheduler improvements that might collapse further).
  - **Polish-ON fixture** (separate test, not C7 byte-identity gate but exercises the B5 polish path): 3-beat episode with `enable_polish_pass=true`. Asserts (a) polish is invoked once per line as expected, (b) polish routes to `creative_writing_model` slot, (c) `_otr_model_loader.LLM_CACHE` shows exactly one resident model after the run (no `_POLISH_CACHE` duplicate-load drift). Catches B5 polish-path bugs the canonical (polish-OFF) workflow can't.

**COMMIT.** Subject: `B2b: writer internal creative/technical routing + slot scheduler + new meta keys`

---

### B2c — Delete `cleanup_model_id` legacy-strip loop + test residue

**REVIEW.** Grep `cleanup_model_id` repo-wide — confirm hits are limited to the legacy-strip loop in the writer + the tests file that pinned it.

**CODE.**
- `nodes/OTR_LedgerScriptWriter.py:2470` — **delete the entire legacy-strip loop.** It strips `cleanup_model_id` from old workflow JSONs; that loop is itself legacy back-compat S28+S29 missed.
- `tests/test_two_llm_split.py` — delete file outright (pinned the prior `cleanup_model_id` partial implementation that no longer exists). New contract gets the new test file in B6.
- `docs/_s28_forbidden_sweep.py` — add `cleanup_model_id` as a forbidden-pattern marker so reintroductions fail loud.

**WIRE.** None.

**REGRESS.**
- All four canonical suites.
- Forbidden-pattern sweep: 0 runtime hits.

**COMMIT.** Subject: `B2c: delete cleanup_model_id legacy-strip loop + test residue + arm forbidden-pattern marker`

---

### B3 — Freeze Cascade widget + toggle deletion (with phase-function dependency audit)

**Scope clarified post-§2a-bis:** cascade is the only LFC LLM-pick site **placed in the canonical workflow**. Standalone Phase 4/5/6 nodes are handled separately in B4 (Python-only, no JSON wire since they aren't placed). Splitting these makes phase-function deletion safe — we audit dependencies before deleting.

**REVIEW.** Read `nodes/OTR_LedgerFreezeCascade.py` lines 84 (docstring), 148-155 (model_id widget), 156-220 (phase-3..6 toggles), 274-284 (run() signature), 319 (load_llm call), 363-368 (downstream phase-toggle args). No dependency-audit decision tree needed here — B4 (locked Option A) deletes the standalone Phase nodes and the phase functions they consume in the same commit. B3 just strips the cascade.

**Cascade widget order — exact before/after pin (matches the §2a-tris discipline for the writer):**

Current cascade `widgets_values` (index 0 → 9):
```
[0]  model_id                          "mistralai/Mistral-Nemo-Instruct-2407"   ← deleted in B3
[1]  enable_phase_3_polish             false                                    ← deleted
[2]  polish_announcer_beats            false                                    ← deleted
[3]  enable_phase_4_scene_coherence    false                                    ← deleted
[4]  enable_phase_4_5_smart_suggestion false                                    ← deleted
[5]  enable_phase_5_voice_drift        false                                    ← deleted
[6]  enable_phase_6_episode_arc        false                                    ← deleted
[7]  enable_phase_7_audio_readiness    true                                     ← kept
[8]  enable_phase_8_video_readiness    true                                     ← kept
[9]  vram_ceiling_gb                   14.0                                     ← kept
```

Post-B3 cascade `widgets_values` (index 0 → 2):
```
[0]  enable_phase_7_audio_readiness    true
[1]  enable_phase_8_video_readiness    true
[2]  vram_ceiling_gb                   14.0
```

`technical_model` arrives via an input socket (not a widget), so it doesn't appear in `widgets_values`. Pin tested in B6 against the cascade's actual `INPUT_TYPES()` at runtime, not against a hardcoded index list.

**CODE — cascade widgets only. Do NOT delete the shared phase functions in `_otr_lfc.py` here.**
- `nodes/OTR_LedgerFreezeCascade.py`:
  - Delete `model_id` widget (lines 148-155) + `DEFAULT_MODEL_ID` literal.
  - Delete `enable_phase_3_polish` / `polish_announcer_beats` / `enable_phase_4_scene_coherence` / `enable_phase_4_5_smart_suggestion` / `enable_phase_5_voice_drift` / `enable_phase_6_episode_arc` widgets (lines 156-220).
  - Add `technical_model` STRING socket input (no widget).
  - Delete corresponding `run()` params + downstream call args.
  - Routing: cascade phases 1/2/9 (reviewer verdicts) → `technical_model` resolved via `_otr_model_inputs.require_model(..., slot="technical")`. Tag each call site `# LLM slot: technical` per Prime Directive 6.
  - Update `_no_ledger_error_json` if it references removed fields.
- `nodes/_otr_lfc.py` — **untouched in B3**. The phase functions stay alive here because the standalone Phase 4/5/6 nodes (still registered until B4) consume them. B4 deletes the standalone nodes AND the now-orphaned phase functions in one atomic commit. Splitting cascade-widget deletion (B3) from shared-function deletion (B4) avoids half-deleting shared code mid-sprint.

**WIRE.** `workflows/otr_scifi_16gb_full.json` — cascade node only:
- `widgets_values`: delete `model_id` value + 6 phase-toggle values (7 entries removed). Final state matches the post-B3 table above (`[enable_phase_7_audio_readiness, enable_phase_8_video_readiness, vram_ceiling_gb]`).
- `inputs`: add `technical_model` input slot. **Explicit link allocation** (ComfyUI link IDs are monotonic graph-level integers):
  - Read current `last_link_id` from the workflow root (existing canonical workflow's highest is `114`; assert this at script time, fail if drift).
  - Allocate new link ID = `last_link_id + 1` (e.g. `115`).
  - Bump `last_link_id` to the new value at the graph root.
  - New link entry in the graph's `links` array: `[115, <writer_node_id>, 5, <cascade_node_id>, <cascade_input_slot>, "STRING"]` (slot 5 == writer's `technical_model` per B2a's output indices).
  - Update writer's `outputs[5]["links"]` from `null` to `[115]` (back-link).
  - Update cascade's new `inputs` entry: `{"name": "technical_model", "type": "STRING", "link": 115}`.
- Other workflow JSONs untouched (no cascade placed).
- `tools/validate_workflow_links.py` across all 8 JSONs — 0 violations.
- New B3 guardrail test: assert (a) writer's output 5 has exactly one back-link, (b) cascade's `technical_model` input link ID matches the writer's output back-link, (c) `last_link_id >= 115`.

**REGRESS.**
- All four canonical suites.
- **End-to-end audio C7 byte-identical** — this is the first commit where the canonical workflow runs end-to-end on the new contract (writer broadcasts to cascade). If drift → revert immediately. Drift here points at the cascade rewire or the writer broadcast format.
- New canary in `tests/test_workflow_json_guardrails.py` against Python `INPUT_TYPES`: cascade has no `model_id` widget and no `enable_phase_3..6` widgets; the canonical workflow's cascade entry has a `technical_model` input link sourced from the writer.

**COMMIT.** Subject: `B3: cascade — delete model_id + phase-3..6 toggles, wire technical_model socket (canonical JSON re-wired)`

---

### B4 — Standalone LFC Phase 4/5/6 nodes — DELETE (Option A, locked pre-kickoff)

**Decision locked before kickoff, not mid-sprint.** Per §2a-bis these nodes are registered in `__init__.py` but not placed in any shipped workflow JSON, so they're orphaned today. Combined with `feedback_no_legacy_back_compat` (deletion-bias) and the standing directive "if it's not in the canonical contract, it shouldn't survive," the plan locks **Option A: delete the standalone node files entirely**. This avoids a mid-sprint branch point where B3's dependency audit could create implicit "what does B4 do?" ambiguity halfway through the sprint.

Option B (keep as user-facing rerun helpers) was considered. Rejected because:
1. The nodes are orphaned today — zero canonical workflows place them. Their continued existence is dead surface waiting for a future "wait, why does this exist?" cleanup pass.
2. Phase function deletion in B3 can proceed unconditionally if standalone nodes are gone — no branching dependency audit needed.
3. Sprint scope stays linear; no mid-sprint decision points.

**Reversal trigger.** Only if Jeffrey explicitly says BEFORE kickoff that he wants the standalone nodes as rerun helpers. Once the sprint opens with this plan, Option B is off the table.

**REVIEW.** Grep `tests/`, `docs/`, and `scripts/` for any reference to `OTR_LFCPhase4Scene` / `OTR_LFCPhase5Voice` / `OTR_LFCPhase6Arc`. Confirm only the class files themselves + `__init__.py` registrations + their own test files (which also get deleted) reference them.

**CODE — atomic standalone-node + phase-function deletion.**
- Delete `nodes/OTR_LFCPhase4Scene.py`, `nodes/OTR_LFCPhase5Voice.py`, `nodes/OTR_LFCPhase6Arc.py` (the standalone nodes that consumed the phase functions).
- Remove the three entries from `__init__.py` `NODE_CLASS_MAPPINGS` + `NODE_DISPLAY_NAME_MAPPINGS`.
- Delete from `nodes/_otr_lfc.py` (now safe — B3 stripped cascade widgets, this commit removes the last consumers):
  - `_phase_3_per_line_polish` (writer's polish pass already handles this on `creative_writing_model`).
  - `_phase_4_scene_coherence`, `_phase_4_5_smart_suggestion`, `_phase_5_voice_drift`, `_phase_6_episode_arc`.
- `nodes/_otr_lfc_llm_helpers.py`: prune helpers exclusively used by the deleted phases.
- Delete any test files exclusively pinning these standalone nodes' `INPUT_TYPES` / `run()` signatures or the deleted phase functions.

**WIRE.** None — these nodes are not placed in any shipped workflow JSON.

**REGRESS.**
- All four canonical suites.
- Forbidden-pattern sweep — 0 runtime hits (B7 adds `OTR_LFCPhase4Scene` / `OTR_LFCPhase5Voice` / `OTR_LFCPhase6Arc` as extinction markers).
- Workflow link validator across all 8 JSONs — 0 violations.
- New `tests/test_b4_standalone_phase_nodes_extinct.py`: assert the three classes are NOT in `NODE_CLASS_MAPPINGS` and the three module files do not exist.

**COMMIT.** Subject: `B4: atomically delete standalone LFC Phase 4/5/6 nodes + phase-3..6 functions in _otr_lfc.py (orphaned post-B3)`

---

### B5 — Delete `OTR_VisualLLMSelector` + MANDATORY `_POLISH_CACHE` collapse (Python only — no JSON wire needed)

**Two parallel LLM caches on a 16 GB card is a guaranteed OOM.** If `_POLISH_CACHE` and `_otr_model_loader.LLM_CACHE` both load Mistral-Nemo (~12 GB each), peak resident VRAM hits ~24 GB. The 14.5 GB ceiling is busted before any inference runs. Collapse is **mandatory**, not a round-robin question.

**No JSON wire in this commit** — per §2a-bis, neither `OTR_VisualLLMSelector` nor `OTR_VisualPromptCoercion` is placed in any shipped workflow JSON. Deleting the selector class is a Python + `__init__.py` change only. The visual consumers (`visual/llm_polish.py`, `visual/visual_prompt_coercion.py`) get their Python `INPUT_TYPES` rewired to expose a **required** `creative_writing_model` STRING input socket (link-only, not a widget).

**Honest UX note (correcting earlier framing):** dragging a node onto a canvas does NOT auto-wire it to anything. There's no auto-discovery of compatible writer outputs in ComfyUI. The required-input socket means: if a user drags a visual consumer onto a canvas and runs the workflow without wiring `creative_writing_model` from the writer, the node fails loud with `MissingModelInputError` rather than silently using a wrong model. To shorten the manual-wire path for users, B8 ships a small `workflows/example_visual_consumer_wiring.json` showing the writer → visual consumer link as a copy-paste reference.

**B5 audio-path discipline — preserve audio byte-identity, no exceptions.** Prime Directive 1 ("audio is king") governs. S30's job is structural: cache collapse + consumer rewire. Sampling behavior is not touched. No feature flag. No scaffold for a future change. The whole sampling-precedence idea is deleted from S30 scope and re-derived from scratch in a later audio-intentional sprint when it has its own design + tests + baseline-roll discipline.

Pre-flight grep (still mandatory):
```
grep -R "make_polish_generate_fn" nodes/ visual/
```
Confirms the writer's polish pass calls `make_polish_generate_fn`. The B5 changes (collapse `_POLISH_CACHE`; route `visual/llm_polish.py` through `_otr_model_loader.LLM_CACHE`; extend `make_polish_generate_fn` to accept a pre-loaded cache entry) must produce identical generated text on the same inputs as pre-B5. OTR's existing `_POLISH_TOP_P=0.9` / `_POLISH_DO_SAMPLE=True` defaults apply unconditionally, unchanged.

**Acceptance:** B5's audio C7 gate is the same regular gate. If audio drifts at B5, **revert immediately** per Prime Directive 1. Baseline is unchanged from pre-B5.

**REVIEW.**
- Grep for every consumer of `OTR_VisualLLMSelector.model_id`. Per §2b audit: `visual/visual_prompt_coercion.py` AND `visual/llm_polish.py`. Confirm no third consumer.
- Read `visual/llm_polish.py` in full: `_POLISH_CACHE` (module-level), `_load_model()` (L67), `_POLISH_TOP_P` / `_POLISH_DO_SAMPLE` sampling constants, `_load_model` callers in this file.
- Confirm `_otr_model_loader.make_polish_generate_fn(...)` exists and accepts sampling-parameter overrides. The collapse requires the loader facade to support a "polish profile" override on top of its existing cache — different sampling math, same model weights.

**CODE.**
- Delete `visual/llm_selector.py` entirely.
- Remove from `__init__.py` `NODE_CLASS_MAPPINGS` + `NODE_DISPLAY_NAME_MAPPINGS`.
- `visual/visual_prompt_coercion.py`: swap the model_id input source — was wired to `OTR_VisualLLMSelector.model_id`, now wires to writer's `creative_writing_model` directly. Add `# LLM slot: creative — visual prompt prose` tag.
- `visual/llm_polish.py`:
  - **Delete `_POLISH_CACHE` entirely** and every code path that references it.
  - **Delete `_load_model()`** (L67) — the LLM-load duty moves to `_otr_model_loader.load_llm`/`request_slot`.
  - Replace the polish entry point's internal model load with `_otr_model_loader.request_slot("creative", model_id)` followed by `make_polish_generate_fn(cache_entry, ...)`. The hardcoded `_POLISH_TOP_P = 0.9` / `_POLISH_DO_SAMPLE = True` constants are NOT applied unconditionally — see the sampling-precedence rule below.
  - Replace the `model_id` parameter source — was wired to `OTR_VisualLLMSelector.model_id`, now wires from the consumer's `creative_writing_model` STRING socket. Add `# LLM slot: creative — visual prompt cleanup` tag.
  - Delete the `"none"` short-circuit (L294) — the writer always provides a model id; the sentinel is dead legacy.
- `nodes/_otr_model_loader.py::make_polish_generate_fn`:
  - **No sampling-precedence change in S30. No feature flag. No scaffold.** The function continues to apply OTR's hardcoded `_POLISH_TOP_P=0.9` / `_POLISH_DO_SAMPLE=True` defaults unconditionally — identical behavior to pre-B5. S30 is structural cleanup (cache collapse + consumer rewire); changing creative output belongs in a deliberate later audio-intentional sprint with its own baseline-roll discipline. Scaffolding dormant feature-flagged logic is tech debt; the cleaner answer is to delete the idea from S30 entirely and re-derive it from scratch in the audio sprint when it has actual scope.
  - Only signature change in B5: extend the function so it can be called against an already-loaded cache entry rather than re-resolving the model. Sampling behavior is unchanged.

**WIRE.** **NONE.** No workflow JSON touch — `OTR_VisualLLMSelector` and `OTR_VisualPromptCoercion` aren't placed in any shipped workflow JSON (§2a-bis), so deleting the selector class is a Python + `__init__.py` change only.

**REGRESS.**
- All four canonical suites.
- New test `tests/test_polish_cache_collapse.py`: load creative slot, invoke `visual/llm_polish.py` polish entry point, assert `_otr_model_loader.LLM_CACHE` reports exactly one resident model (no duplicate load). Test fails if `_POLISH_CACHE` is reintroduced.
- Forbidden-pattern sweep: `_POLISH_CACHE` is now a runtime extinction marker.
- Audio C7 byte-identical — must still hold against fixture inputs (this commit doesn't touch the audio path).

**COMMIT.** Subject: `B5: delete OTR_VisualLLMSelector + collapse _POLISH_CACHE into single LLM cache (Python only)`

---

### B6 — Test suite — new wiring tests + extend guardrails

**REVIEW.** Read `tests/test_workflow_json_guardrails.py` in full. (The legacy `tests/test_two_llm_split.py` was deleted in B2c.)

**CODE.**
- New `tests/test_two_model_selector_wiring.py` (~180 LOC):
  - `test_slot1_neq_slot2_routes_structured_paths_to_technical` — imports `DEFAULT_LLM` and `TEST_TECHNICAL_LLM` from `nodes._otr_model_catalog`; fixture wires writer with `creative_writing_model=DEFAULT_LLM` and `technical_model=TEST_TECHNICAL_LLM`. Uses the same `_hard_mock_loader_paths` fixture from B1c (no real downloads, no real GPU loads). Runs through a short episode and asserts:
    - `meta.gen_params_by_phase["FORMAT_NORM"]["slot"] == "technical"`
    - `meta.gen_params_by_phase["FORMAT_NORM"]["model"] == TEST_TECHNICAL_LLM`
    - `meta.gen_params_by_phase["outline"]["slot"] == "creative"`
    - `meta.gen_params_by_phase["outline"]["model"] == DEFAULT_LLM`
    Two explicit fields (`slot` + `model`) — no awkward concatenated key name. **Repo IDs imported from the catalog as constants** — hardcoded strings in tests are forbidden because casing drift would cause cosmetic test failures unrelated to the contract under test. The only place a hardcoded repo ID appears is in explicit error-message tests (where the exact message contents are the assertion).
  - `test_slot1_neq_slot2_split_routing_full_phase_trace` — **mandatory Slot1≠Slot2 regression** (audio C7 only proves same-model cache reuse, not split routing). Same setup as above but asserts the full phase-by-phase routing trace: every one of the 14 phases in §2c routes to its documented slot. Table-driven test that iterates the routing table. Mocked loader records each `request_slot` call; assertions compare against the §2c reference table. If any phase routes to the wrong slot, the test fails with the specific phase + slot mismatch.
  - **Manual VRAM soak (separate, run before B8 close):** Jeffrey runs `scripts/vram_profile_slot_swap.py` once with a real Slot1≠Slot2 configuration on the 5080. Output peak-memory trace goes into B8's QA doc as confirmation that the split routing actually works under real memory pressure. Not in pytest count; not a CI gate; one-shot operator verification.
  - `test_slot1_eq_slot2_reuses_single_model_cache` — Slot 1 == Slot 2 → loader caches one model + reuses across phases (behavior assertion against the cache, not a back-compat claim).
  - `test_unknown_model_id_raises_UnknownModelError` — non-curated, non-scanned ID fails loud (no fallback) with the actionable recovery-hint message.
  - `test_missing_creative_writing_model_raises_MissingModelInputError` — drive a consumer with the socket unwired.
  - `test_missing_technical_model_raises_MissingModelInputError`.
  - `test_no_legacy_model_id_in_meta` — runs a full fixture episode and asserts:
    - `"model_id" not in meta` (top level)
    - For every phase entry in `meta["gen_params_by_phase"]`: `"model_id" not in entry`
    - For every value across `meta["creative_writing_model"]` + `meta["technical_model"]` + `meta["gen_params_by_phase"][*]["model"]`: never contains the substring `"[NOT DOWNLOADED]"` (label-poisoning regression catch).
  - `test_all_14_routing_table_phases_present` — iterates §2c routing table (14 sub-passes); for each phase, asserts (a) the writer call site emits a `meta.gen_params_by_phase[<phase>]` entry, (b) the entry's `slot` matches §2c, (c) the entry's `model` matches whichever slot the routing table specifies. Catches any sub-pass that silently routes to the wrong slot or skips meta-stamping.
  - `test_context_cap_clamps_to_hardware_limit` — fixture model advertises 128k in `config.json`; assert `resolve_context_cap` returns `HARD_VRAM_CONTEXT_LIMIT` (8192), not the advertised 128k.
  - `test_vram_fit_precheck_oversized_returns_FAIL_verdict` — `check_vram_fit(catalog.TEST_OVERSIZED_LLM, 8192).tier == "FAIL"` with the estimated-vs-ceiling reason in `verdict.reason`. Catalog constant; no hardcoded repo ID in the test.
  - `test_polish_sampling_respects_generation_config` — fixture `generation_config.json` sets `do_sample=False`; assert the resolved polish profile honors that and does NOT apply OTR's `do_sample=True` override.
  - `test_polish_sampling_falls_back_to_otr_defaults_when_config_silent` — fixture model has no `generation_config.json`; assert OTR's `top_p=0.9` / `do_sample=True` defaults land.
  - `test_auto_download_disk_space_precheck_raises_when_insufficient` — mock `shutil.disk_usage` to return 1 GB free, mock `estimate_model_size_gb` to return 24 GB; assert `InsufficientDiskSpaceError` raises BEFORE any `snapshot_download` call.
- New `scripts/vram_profile_slot_swap.py` — **real VRAM profiler, lives outside the pytest gate**:
  - Standalone script under `scripts/`, NOT under `tests/`. Never run by `pytest` in any configuration; not in the canonical four-suite regression; not in sprint acceptance pytest count.
  - Manual soak tool: Jeffrey runs it ad-hoc on the actual 5080 hardware to validate peak-memory behavior across slot transitions. Output to `outputs/vram_profile_<timestamp>.txt` for tracking peak drift across releases.
  - Real load: `catalog.DEFAULT_LLM` → `_request_slot("technical", catalog.TEST_TECHNICAL_LLM)` → measure `torch.cuda.max_memory_allocated()` at every boundary; assert peak < 14.5 GB. Catalog constants only; no hardcoded repo IDs in the script.
  - Real-hardware HuMo→LLM fragmentation case to validate `ipc_collect()` recovers the IPC-handle pool (compare pre/post `cudaMallocAsync` pool counters where the driver exposes them).
  - **Why outside pytest:** real VRAM behavior on Windows depends on driver state, other GPU allocations, ComfyUI Desktop state, and fragmentation. Including it in pytest creates flaky-CI risk on every machine. Manual soak by Jeffrey is the appropriate audience and frequency.
  - Acceptance: optional manual soak before B8 close; not a CI gate.
- Extend `tests/test_workflow_json_guardrails.py` — **all assertions are structure-based, not value-based** (avoids false-positives on non-LLM media nodes whose widgets legitimately carry repo-like strings: MusicGen, AudioGen, FLUX, HuMo, LTX, Whisper):
  - `TestWriterModelSlotDefaults` — pin writer widget order against the writer's actual Python `INPUT_TYPES()` (per the §2a-tris layout), both slots default to the catalog default. Read the order from `INPUT_TYPES()` at test time; do not hardcode index numbers in the test that the test itself is supposed to verify.
  - `TestNoModelWidgetOutsideWriter` — for every registered node class other than `OTR_LedgerScriptWriter`, assert `INPUT_TYPES()["required"]` and `INPUT_TYPES()["optional"]` contain **no widget keyed** `model_id`, `model_creative`, `model_technical`, `creative_writing_model`, `technical_model`, or any other `model_*` STRING widget. Structure check, not value check.
  - `TestCanonicalWorkflowHasBroadcastLinks` — for the canonical workflow JSON only: assert writer's `creative_writing_model` and `technical_model` outputs are present, and `technical_model` has a link to the placed `OTR_LedgerFreezeCascade` node. Do NOT assert links to phase 4/5/6 or VisualPromptCoercion — those nodes are not placed in any canonical workflow (per §2a-bis).
  - `TestAllWorkflowsValidate` — `tools/validate_workflow_links.py` returns 0 violations across all 8 workflow JSONs.

**WIRE.** Tests only — no workflow JSON touch.

**REGRESS.** Full suite. New test count: +~12 (wiring) +~5 (guardrails) = +~17.

**COMMIT.** Subject: `B6: wiring tests + structure-based guardrails (widget-name not widget-value)`

---

### B7 — Forbidden-pattern sweep — extinction markers

**REVIEW.** Read `docs/_s28_forbidden_sweep.py` to confirm current marker list. Grep `__init__.py` for any forensic comments still describing alias / back-compat behavior that contradicts S29's `_RENAME_ALIASES` deletion.

**CODE.** Append new extinction markers to the sweep (cleanup_model_id was already added in B2c):
- `OTR_VisualLLMSelector` (deleted class — B5)
- `_LLM_MODEL_CHOICES` (deleted symbol — B5)
- `_MODEL_CHOICES` (deleted from writer — B2a)
- `DEFAULT_MODEL_ID` — path-aware: forbidden in **runtime Python under `nodes/` and `visual/`** (B2a deletes from writer; B3 deletes from cascade; B4 deletes from standalone phase nodes under Option A). Allowed only in docs with S30 citation, in tests' own marker-list literals, and in the forbidden-sweep marker config itself. Sweep configured to skip these paths when grepping.
- `_LLM_CACHE` (deleted from `story_orchestrator.py` — B0)
- `_load_llm` — **path-scoped marker**: forbidden only as `nodes/story_orchestrator.py::_load_llm` (deleted in B0). The new public `_otr_model_loader.load_llm` is a legitimate symbol and must NOT be caught by this marker. Sweep implementation: match the function definition `def _load_llm(` AST-walked within `nodes/story_orchestrator.py` only, not substring grep across the codebase.
- `_generate_with_llm` (deleted from `story_orchestrator.py` — B0)
- `_POLISH_CACHE` (deleted from `visual/llm_polish.py` — B5; mandatory collapse)
- `MODEL_CONTEXT_CAPS` (deleted from `_otr_model_loader.py` — B1; dynamic catalog read replaces it)
- `DEFAULT_CONTEXT_CAP` (deleted from `_otr_model_loader.py` — B1; no blind 8192 fallback)
- `enable_phase_3_polish` (deleted widget — B3)
- `polish_announcer_beats` (same)
- `enable_phase_4_scene_coherence` (same)
- `enable_phase_4_5_smart_suggestion` (same)
- `enable_phase_5_voice_drift` (same)
- `enable_phase_6_episode_arc` (same)
- Option A only: `OTR_LFCPhase4Scene`, `OTR_LFCPhase5Voice`, `OTR_LFCPhase6Arc` (deleted classes — B4)

**Structural sweep rule** added to `_s28_forbidden_sweep.py` — **path-aware, syntax-aware (STRING + COMBO/list), name-aware, AND semantic-tagged**:
- AST-walk every `INPUT_TYPES` block in `nodes/**/*.py` and `visual/**/*.py`. Examine each entry's structural form:
  - **Widget form, STRING-typed**: `"name": ("STRING", {"default": "...", ...})`.
  - **Widget form, COMBO/list-typed**: `"name": (["choice_a", "choice_b", ...], {...})` — the first element is a Python list (or list-returning expression like `catalog.dropdown_choices()`), ComfyUI renders this as a dropdown. **The writer's new model dropdowns use this exact form.** Without explicit detection, a future bad LLM dropdown using COMBO would slip past the sweep.
  - **Connectable input form**: `"name": ("STRING", {"forceInput": True})` or `"name": ("STRING",)` (single-element tuple in `INPUT_TYPES["optional"]` / `["required"]` without widget args).
- **Name-matching rule** (independent of type form): the sweep considers any widget/input keyed by one of these names a candidate for the LLM-pick rule:
  - `model_id`
  - `llm_model`
  - `creative_writing_model`
  - `technical_model`
  - Anything matching the regex `model_[A-Za-z_]+` or `[A-Za-z_]+_model` (catches `model_creative`, `polish_model`, `slot_model`, etc.)
- Rejection rules:
  - Candidate name in **widget form** (STRING OR COMBO/list) outside `OTR_LedgerScriptWriter`, on a class **NOT** carrying `NON_LLM_MODEL_WIDGET_OK = True` → REJECT. Catches both `("STRING", {...})` and `(list_of_repo_ids, {...})` forms. A future bad LLM dropdown can't sneak in via COMBO.
  - Candidate name in **connectable input form** outside the writer → ALLOWED. These are the sockets B3/B4/B5 add on consumer nodes to receive the writer's broadcast.
- **Non-LLM media nodes opt into the exemption with a class-level marker, not a hardcoded class list:**
  ```python
  class MusicGenTheme:
      NON_LLM_MODEL_WIDGET_OK = True   # MusicGen / AudioGen / FLUX / HuMo / LTX / Whisper — non-LLM model picks
      ...
  ```
  Adding a new audio/video node with a legitimate `model_id` / `model_path` widget just sets the marker; no edit to `_s28_forbidden_sweep.py` needed. This is durable + extensible vs. the fragile hardcoded class list approach. The sweep grep is:
  ```python
  if has_model_star_widget(node_class) and not getattr(node_class, "NON_LLM_MODEL_WIDGET_OK", False):
      reject(node_class)
  ```
- Enforces Prime Directive 6 obligation 2 without false-positives on the media path and without churning the sweep config every time a new media loader lands.

**`__init__.py` forensic-comment cleanup re-verify.** B0 did the primary `__init__.py` comment scrub. Re-grep here to catch any new stale forensic comments that may have crept into `nodes/` or `visual/` during B1-B6: anything mentioning `_RENAME_ALIASES`, deleted classes, or alias / back-compat behavior in a way that suggests they're still supported. Delete or rewrite. This is the second-pass safety net, not the primary scrub.

**WIRE.** None.

**REGRESS.** `python docs/_s28_forbidden_sweep.py` — 0 runtime hits required. Any forensic comments referencing deleted symbols must cite sprint S30 + commit hash.

**COMMIT.** Subject: `B7: arm forbidden-pattern sweep with S30 extinction markers (+ forensic-comment re-verify)`

---

### B8 — Sprint close — final QA review + ROADMAP refresh

**REVIEW.** Walk the diff `git log --oneline v2.0-alpha..HEAD` and verify each acceptance check passed at its commit boundary.

**CODE.**
- New `docs/2026-05-14-S30-two-model-selector-final-qa-review.md`:
  - Acceptance table (target vs actual for each check in §4 below)
  - Documented deviations from this plan
  - Open follow-ups (Path B JS extension, non-LLM consolidation — Shape B `OTR_ModelHub`)
- Update `ROADMAP.md`:
  - Move Sprint B from "next-up" to "COMPLETE 2026-05-14"
  - Update "Forward work" section with the open follow-ups
  - Update sprint sequencing — Sprint C now in next-up position
- Update `BUG_LOG.md` with any local bugs surfaced during the sprint.

**WIRE.** None — docs only.

**REGRESS.** Full suite one final time. ComfyUI Desktop boot test (Jeffrey's call when ready — not a gate per S29 precedent, but a final operator verification).

**COMMIT.** Subject: `B8: Sprint S30 close — two-model selector shipped, ROADMAP refreshed`

---

## 4. Acceptance criteria (sprint close)

| # | Check | Target |
|--:|---|---|
| 1 | Pytest | Baseline 2146 + new tests (count recorded at B8 from actual final state; not a hardcoded target). 0 failed, 8 skipped (+ any new test-module skip markers must be justified per the EXCLUDED_* / ALLOWED_* rule). |
| 2 | Bug Bible | 23/1/2xf — held at every commit boundary |
| 3 | Forbidden-pattern sweep | 0 runtime hits with B7 markers armed |
| 4 | Workflow link validator | 0 violations across all 8 workflow JSONs |
| 5 | Audio-byte-identical (Python fixture) | PASS at every commit boundary B0 onward |
| 6 | Audio-byte-identical (end-to-end) | PASS at B3 (first commit where the canonical workflow runs end-to-end on the new contract) AND at every commit boundary after, including B5. No baseline roll in S30. |
| 7 | `_otr_model_catalog.py` importable + tested | catalog scan test green; `resolve_context_cap` reads `config.json` dynamically |
| 8 | `_otr_model_loader.py::MODEL_CONTEXT_CAPS` static dict | DELETED (replaced by catalog dynamic read) |
| 9 | `_otr_model_loader.py::DEFAULT_CONTEXT_CAP = 8192` | DELETED (no blind fallback; raises `ContextCapUnknownError` on miss) |
| 10 | Slot scheduler transition count per fixture episode (Slot 1 ≠ Slot 2) | Per-beat-default: equals documented DAG minimum (`meta["slot_transitions"]`), ~9 for a 3-beat episode. `OTR_BATCH_PER_BEAT=1` opt-in: 3. Fail only if measured > documented minimum. |
| 11 | `_POLISH_CACHE` references in `visual/llm_polish.py` | 0 after B5 (collapse mandatory in B5) |
| 12 | `_LLM_CACHE` in `nodes/story_orchestrator.py` | 0 (deleted in B0) |
| 13 | `model_id` widget anywhere outside writer | 0 hits |
| 14 | `OTR_VisualLLMSelector` references | 0 hits in code, only forensic comments with S30 citation |
| 15 | Phase-3..6 toggle widgets in Freeze Cascade | 0 hits |
| 16 | `_phase_3_per_line_polish` / `_phase_4_scene_coherence` / `_phase_4_5_smart_suggestion` / `_phase_5_voice_drift` / `_phase_6_episode_arc` in **runtime Python under `nodes/_otr_lfc.py`** | 0 definitions or call sites. Forensic mentions in `docs/`, `BUG_LOG.md`, `ROADMAP.md`, and `tests/` test-name docstrings allowed only with S30 citation. Sweep config explicitly excludes `docs/` and test-name strings from this rule. |
| 17 | `OTR_LFCPhase4Scene` / `OTR_LFCPhase5Voice` / `OTR_LFCPhase6Arc` in `NODE_CLASS_MAPPINGS` + as module files | 0 hits (Option A locked pre-kickoff) |
| 18 | Workflow JSON loads in ComfyUI Desktop | 0 missing-node warnings, 0 alias firings |
| 19 | `tests/test_init_aliases_empty.py` | Still passes |
| 20 | Writer widget order | Matches §2a-tris post-B2a layout: `[episode_title, target_words, num_characters, seed, seed_mode, creative_writing_model, technical_model, custom_premise, include_act_breaks, act_count, style, style_custom, creativity, optimization_profile, perfect_run_spacesaver, min_p, repetition_penalty, max_new_tokens_cap, enable_polish_pass]`. Pin tested against `INPUT_TYPES()` at runtime, not against a hardcoded index list. |
| 21 | Both writer outputs (`creative_writing_model`, `technical_model`) are link sources for the right consumers | per §2c routing table |
| 22 | `UnknownModelError` raises with actionable recovery-hint message | error includes top-5 installed alternatives |
| 23 | `HARD_VRAM_CONTEXT_LIMIT` clamp enforced in `resolve_context_cap` | `min(advertised, limit)`; default limit 8192 on 16 GB |
| 24 | `check_vram_fit` pre-check at LLM load time | returns `FAIL` verdict for 70B-on-16GB BEFORE the OOM with estimated-vs-ceiling message |
| 25 | `torch.cuda.ipc_collect()` in `unload_llm()` | called alongside `empty_cache()` + `synchronize()` |
| 26 | Polish sampling behavior unchanged in S30 | `make_polish_generate_fn` applies OTR's hardcoded `_POLISH_TOP_P=0.9` / `_POLISH_DO_SAMPLE=True` unconditionally — identical to pre-B5. No feature flag exists. No scaffolded dormant code path. Model-author-config respect is deferred entirely to a later audio-intentional sprint, re-derived from scratch. |
| 26b | `resolve_hf_token` cross-platform gating | `winreg` import + lookup only executes inside `if os.name == "nt":` block. Test on macOS/Linux mock confirms no `ImportError`. |
| 26c | HF download progress wiring | `huggingface_hub.snapshot_download` invoked with `tqdm_class` adapter forwarding to `comfy.utils.ProgressBar`. User sees real progress in ComfyUI queue UI during 24 GB Mistral-Nemo first-run download. Worker-thread placement does not block UI thread. |
| 27 | Pre-fetch disk-space + size-estimate check | `InsufficientDiskSpaceError` raises before `snapshot_download` if free space − download − 5 GB margin < 0 |
| 28 | Download announcement in queue UI | single-line `[OTR] Downloading <repo> — X.X GB → <path> (first run only)` before fetch |
| 29 | `scripts/vram_profile_slot_swap.py` manual soak (NOT in pytest count) | peak resident < 14.5 GB during slot transitions; `ipc_collect()` recovers IPC-handle pool. Optional manual run by Jeffrey before B8 close; not a CI gate. |
| 30 | `grep` of `__init__.py` for "registered as an alias below" | 0 hits (B0 forensic-comment scrub) |
| 31 | `grep` of `__init__.py` for `_RENAME_ALIASES` outside test-pin context | 0 hits |
| 32 | `*.gguf` in `auto_download_if_missing` `allow_patterns` | 0 hits (Transformers loader can't consume GGUF; deferred to future llama.cpp backend sprint) |
| 33 | No required API keys, no required payment | OTR runs end-to-end on at least one ungated curated model with zero auth setup (Qwen2.5 / Captain-Eris / Mag-Mell verified). No paid hosted APIs invoked at runtime. `HF_TOKEN` is consumed only when present and only for gated repos; never injected as a required parameter. |
| 33b | Fast-path / startup discipline (not policy — just speed and test reproducibility) | `HfApi().model_info` does NOT fire from `scan_local_llm_cache`, `build_dropdown_choices`, `INPUT_TYPES`, ComfyUI startup, or pytest paths without `responses` / `pytest-httpx` mocking. Network calls are allowed in user-action paths (`auto_download_if_missing`, explicit "preview size" commands). |
| 34 | `check_vram_fit` verdict tiering | Returns `VRAMFitVerdict.{PASS,WARN,UNKNOWN,FAIL}` enum, not `bool`. `FAIL` reserved for ≥1.5× ceiling overage on parseable param counts; `WARN`/`UNKNOWN` for ambiguous uncurated models. |
| 35 | `validate_model_id` allow-list scope | Admits curated + locally-scanned + valid `org/name` (when auto-download enabled). Structural rejection on path-traversal / drive letters / unsafe formats applies regardless. |
| 36 | B2a internal slot routing leak check | `tests/test_writer_b2a_surface_only.py` AST-walks `OTR_LedgerScriptWriter.run` and asserts ZERO `request_slot("technical", ...)` calls in B2a's commit state. |
| 37 | B7 structural sweep distinguishes widgets from connectable input sockets | `model_*` STRING widgets outside writer → REJECT; `model_*` STRING connectable input sockets (`forceInput: True` or single-element tuple) → ALLOWED. No false-positives on B3/B4/B5 consumer sockets. |

---

## 5. Round-robin trigger points

Per CLAUDE.md round-robin section. Save transcripts under `docs/2026-05-14-two-model-selector/`:

- **Before B1a** — catalog dataclass + scan + dropdown + validator (offline only). Validate: `ScanResult` shape; `validate_model_id` admit-paths (curated / locally-scanned / valid `org/name` when auto-download enabled); structural rejection (path traversal, drive letters, unsafe formats); the `CuratedModel` dataclass shape with `requires_auth` / `vram_fit_tier` / `loader_backend`.
- **Before B1a2** — auto-download UX on Windows HF paths + pre-flight gated detection. Validate: long-path edge cases under `HF_HOME` junction-resolution; download-progress visibility in ComfyUI queue UI; `resolve_hf_token()` resolution from process env first, then `HKCU\Environment` via `winreg`; pre-flight `GatedModelError` raises BEFORE `snapshot_download`; pre-fetch disk-space + size-estimate announcement UX.
- **Before B1b** — dynamic context-cap design. Validate: `max_position_embeddings` / `n_positions` / `n_ctx` extraction across the curated model family's `config.json` shapes; the `HARD_VRAM_CONTEXT_LIMIT` default (8192 on 16 GB); curated overrides for any model whose advertised window exceeds what the pipeline can sanely feed.
- **Before B1c** — loader primitive design. Validate the `unload_llm` teardown sequence order (`empty_cache` + `ipc_collect` + `synchronize`); the `check_vram_fit` estimator (param count × bytes-per-param + KV cache); the `request_slot` cache-hit-vs-teardown decision logic.
- **Before B2b** — writer slot scheduler design. Validate the documented DAG minimum (§6 partial-batching table) against the actual writer call graph; confirm `unload_llm()` (not `_flush_vram_keep_llm()`) is the correct primitive at slot transitions when Slot 1 ≠ Slot 2; confirm the per-beat-default vs `OTR_BATCH_PER_BEAT=1` tradeoff is the right shape (or propose a refinement). The "≤3 transitions" target only applies in opt-in batched mode; per-beat default is ~9 for a 3-beat episode and is the honest baseline.
- **Before B3** — phase-3..6 code-path deletion **dependency audit**. The grep audit is mechanical, but if it finds external callers (specifically the standalone Phase 4/5/6 node files), the path forward (defer function deletion to B4-Option-A or split the work) wants a sanity check.
- **Before B5** — cache collapse + loader signature change + visual rewire + selector deletion is NOT mechanical. Validate: the `make_polish_generate_fn` extension for already-loaded cache entries; the consumer rewire surface in `visual/llm_polish.py` + `visual/visual_prompt_coercion.py`; the polish-ON test fixture (since canonical workflow has polish disabled — see §B5 polish-coverage fixture). Save transcript under `docs/2026-05-14-two-model-selector/`.
Skip the round-robin for B0 / B2a / B2c / B4 / B6 / B7 / B8 — mechanical execution. (B1a / B1a2 / B1b / B1c / B2b / B3 / B5 each have their own trigger above. B4 is mechanical because Option A is locked pre-kickoff per the standalone-phase-node decision in B4's section.)

**No longer round-robin triggers (resolved in-plan):**
- `_POLISH_CACHE` collapse — MANDATORY in B5 per VRAM math (§6). Two parallel LLM caches on a 16 GB card = guaranteed OOM. Not optional.
- Workflow JSON migration sequencing — per-commit wiring works given only writer (B2a) + cascade (B3) are placed in the canonical workflow. No separate "atomic migration" commit needed.
- Unknown-`model_id` behavior — fail-loud with actionable recovery hints; soft-fallback deferred to S24 public-release polish.
- `MODEL_CONTEXT_CAPS` static dict — deleted in B1; dynamic `config.json` read is the only path.

---

## 6. VRAM ceiling — Prime Directive 2

The writer's execution flow is **interleaved by dependency** — `news_interpreter` (technical) feeds outline (creative); cast contract (technical) validates cast (creative) before outline starts; critic (technical) runs per-beat before polish (creative). Full batching ("all creative, then all technical") is not architecturally achievable. The slot scheduler must handle interleaving correctly.

### When Slot 1 == Slot 2 (the default, audio C7 baseline)

The loader caches one model and reuses it across all 14 sub-passes. No swap, no flush, no penalty. This is the audio byte-identical path.

### When Slot 1 ≠ Slot 2

Memory math on a 16 GB card (14.5 GB usable ceiling — target, not guarantee; Windows DWM + background apps can eat more, so live `LibreHardwareMonitor` polling is the real-time check, per `reference_libre_hardware_monitor` memory):

- Mistral-Nemo: ~12 GB VRAM resident
- Gemma-4-E2B-it: ~3 GB VRAM resident
- Both resident simultaneously: ~15 GB → **immediate OOM**

`_flush_vram_keep_llm()` is the WRONG primitive for the different-models case — it preserves the LLM cache identity, which is exactly what you DON'T want when the next call needs a different model. It's designed for same-model phase transitions (free intermediate activations, keep the model weights resident).

**The correct teardown for slot transitions when Slot 1 ≠ Slot 2:**

1. Detect slot mismatch at each LLM call site (loaded slot ≠ requested slot).
2. Full `unload_llm()`:
   - Move model weights + tokenizer to CPU (`model.to("cpu")`).
   - Drop module references (`del self._cache_entry`).
   - `gc.collect()` — purge Python-side references.
   - `torch.cuda.empty_cache()` — return free blocks to the allocator.
   - **`torch.cuda.ipc_collect()`** — release inter-process CUDA IPC handles. Critical when LLM load follows a video-model run (FLUX / HuMo / LTX) — PyTorch fragmentation accumulates IPC handles from the prior pipeline, and `empty_cache()` alone doesn't recover that VRAM. Without `ipc_collect()`, the next `load_llm` can OOM even when the raw byte budget mathematically fits.
   - `torch.cuda.synchronize()` — let any in-flight ops finish before the next allocator request lands.
3. Then `load_llm(requested_slot_model)` to bring up the new model.
4. Log the transition: `[Selector] slot transition: <old> → <new> (full teardown, swap cost ~Xs, VRAM after teardown: XX MB)`.

`_flush_vram_keep_llm()` stays in use **only when** the next call is the same slot as the currently loaded model (intermediate cache clear without unloading weights).

**Note on chained-backend coordination — video models are the dominant VRAM-pressure source, not the LLM swap.** The canonical workflow JSON runs `OTR_BatchHumoRender` (HuMo 14B, ~11 GB on disk) and `OTR_BatchLTXRender` (LTX 22B, ~22 GB on disk) in the same pipeline as the writer's LLM(s). These models load sequentially (not simultaneously — OTR's pipeline unloads between stages) but their PyTorch allocations accumulate fragmentation that persists across stage boundaries.

The `reference_chained_backend_teardown.md` memory documents the canonical teardown sequence (`remove_all_hooks` + CPU-move + `gc.collect()` + `torch.cuda.empty_cache()` + `torch.cuda.ipc_collect()`) for the FLUX / HuMo / LTX sidecar pipelines. **B1c's `unload_llm` must apply the same sequence in the same order** so:
- HuMo → LLM transition (post-video, pre-next-script-LLM-call) starts from a fully-defragmented allocator.
- LTX → LLM transition same.
- LLM → HuMo transition (the audio→video cycle): `unload_llm` runs the same teardown so HuMo's loader sees a clean allocator.

If a future commit refactors `_otr_model_loader.unload_llm`, align with the chained-backend teardown helper rather than re-deriving the sequence. The B1c implementation should reuse the existing helper if one exists in the codebase, or expose a new shared helper that both LLM and video loaders consume — single source of truth for the teardown sequence prevents drift.

### Partial-batching mitigation + measured target (not a fake precision number)

The writer's natural call graph for an N-beat episode interleaves creative and technical:
- Pre-loop: `news_interpreter (T)` → `style_picker_p1 (C)` → `style_picker_p2 (T)` → `cast_contract (T)` → `cast (C)` → `outline (C)` (≥4 transitions even before per-beat work begins).
- Per-beat (×N): `dialogue (C)` → `critic (T)` → `polish (C)` (2 transitions per beat).
- Post-loop: `announcer (T)` → optional `format_norm / llm_rescue / word_extend (T)` → `grammarian (T)` (1 transition).

**Irreducible minimum WITHOUT batching, for a 3-beat episode: ~11 transitions.** That's disk-thrashing territory on a 16 GB card with swap-cost ~12 GB.

**Batching opportunities (the writer's job in B2b to identify and implement):**

| Batch | Members | Tradeoff |
|---|---|---|
| Pre-loop technical block | `news_interpreter` + `style_picker_p2` + `cast_contract` | Free — these run sequentially with no creative interleaving between them. |
| Pre-loop creative block | `style_picker_p1` + `cast` + `outline` | Free — `style_picker_p1` depends only on the article, runs before the technical block; `cast` + `outline` come after the technical context lands and run together. Requires `style_picker_p1` to run BEFORE the technical block, not interleaved with it. |
| Per-beat critic batch | All N critics fire after all N dialogues, all N polishes after all N critics | **Lossy** — changes feedback flow: critic1's findings won't influence dialogue2 generation. Per-beat default preserves the corrective loop; batched mode is a user opt-in via `OTR_BATCH_PER_BEAT=1`. |
| Post-loop technical block | `announcer` + `format_norm` + `llm_rescue` + `word_extend` + `grammarian` | Free — these run sequentially at episode end. |

**Realistic transition counts under batching for a 3-beat episode (Slot 1 ≠ Slot 2):**
- Per-beat default (preserves feedback loop): ~9 transitions (pre-loop block-pair, then per-beat C→T→C ×3, then post-loop block). Documented DAG minimum given the preserved feedback semantics.
- `OTR_BATCH_PER_BEAT=1` (user opt-in for VRAM-pressured rigs): **3 transitions** (pre-loop technical-block → creative-block-including-all-dialogues → final-technical-block-including-all-critics-and-polishes). Loses the per-beat corrective loop in exchange for the swap-cost savings.

**Acceptance — measure-and-report, not a single hardcoded number:**
- The slot scheduler logs every transition: `[Selector] transition #N at <phase>: <old_slot> → <new_slot>`.
- At the end of each fixture episode, total transitions reported in `meta["slot_transitions"]`.
- Test asserts transition count == the documented DAG minimum for whichever batching mode the fixture runs in (per-beat default = ~9 for 3 beats; opt-in batch = 3). Fail only if transition count **exceeds** the documented minimum — that means batching opportunities are being missed.

Implementation: `_otr_model_loader.py` exposes `request_slot(slot_name, model_id)`. The writer's internal scheduler consolidates same-slot calls where the dependency DAG permits + the batching mode allows, then makes one `request_slot()` call per consolidated block. This is a B2b design point that must land alongside the routing surgery, not a follow-up sprint.

---

## 7. Audio C7 gate — Prime Directive 1

Slot 1 default == prior Mistral-Nemo, and Slot 2 default == Slot 1, so under the default workflow audio output is **byte-identical** to the pre-B0 baseline **at every commit boundary in S30**. There is no audio-baseline roll-point in S30. Per Prime Directive 1 + the B5 feature-flag-OFF rule below, audio byte-identity holds end-to-end.

- **B0 / B1a / B1a2 / B1b / B1c / B2c** — Python-fixture audio test runs after each Python-only commit; any drift points at the Python change itself, revert immediately.
- **B2a** — writer JSON gets the two-widget surface + unused output sockets. Both slots default to the same model and feed the unchanged generation path. Audio MUST stay byte-identical. Drift attributes to widget surgery, revert immediately.
- **B2b** — writer's internal routing switches to the slot scheduler. Both slots default to the same model so the loader caches one model and reuses it. Audio MUST stay byte-identical. Drift attributes to routing, revert immediately.
- **B3** — first commit where canonical workflow runs end-to-end on the new contract. Both nodes resolve to the same Mistral-Nemo. Audio MUST stay byte-identical. Drift attributes to cascade rewire, revert immediately.
- **B4** — standalone Phase nodes + phase functions deleted; nodes not in canonical workflow so end-to-end audio unaffected. Python-fixture gate.
- **B5** — `_POLISH_CACHE` collapse + `visual/llm_polish.py` rewire. Sampling-precedence change scaffolded but feature-flagged **OFF** via `OTR_POLISH_RESPECT_MODEL_CONFIG=0` (the S30 default). Under the OFF flag, `make_polish_generate_fn` continues to apply OTR's hardcoded `_POLISH_TOP_P=0.9` / `_POLISH_DO_SAMPLE=True` unconditionally. Audio MUST stay byte-identical. Drift attributes to the cache collapse or the polish rewire, revert immediately. Flipping the flag ON is the job of a later, audio-intentional sprint.
- **B6 / B7 / B8** — Python-fixture gate; audio byte-identical to pre-B0 baseline.

Any drift at any commit → **revert immediately**. Do not investigate forward; revert and re-plan. No "legitimate roll" exemptions in S30.

---

## 8. Re-entry triggers (if sprint pauses)

- **B0 is fully isolated** — pure deletion of dead code + `__init__.py` forensic-comment scrub. Safe stopping point.
- **B1a is fully isolated** — offline catalog dataclass + scan + dropdown + validator; no HF API surface, no loader change. Safe stopping point.
- **B1a2 adds the HF network surface** — `auto_download_if_missing` + `estimate_model_size_gb` + `GatedModelError` pre-flight + disk-space pre-check + `resolve_hf_token()`. No production caller wires to it until B2b (writer's `_request_slot` → `request_slot` → `load_llm` → auto-download path). Safe stopping point.
- **B1b replaces the static context-cap dict** with dynamic + hardware-clamped lookup. Touches `_otr_model_loader.py`. Audio C7 must hold; safe stopping point if it does.
- **B1c adds the slot primitives** (`unload_llm`, `request_slot`, `check_vram_fit`). No production caller yet, so behavior unchanged. Safe stopping point.
- **B2a is the writer-surface widget surgery + writer-JSON wire.** Once landed, the writer's Python and JSON surfaces both expose the two slots, but routing inside the writer still feeds the single legacy generation path. Cascade still has its old `model_id` widget. Safe stopping point if audio C7 holds.
- **B2b is the writer internal routing change** — slot scheduler lands here. Both slots default to the same model so audio is unchanged. Safe stopping point.
- **B2c is the legacy-strip-loop deletion** — pure Python cleanup. Safe stopping point.
- **B3 is the cascade rewire + canonical-workflow end-to-end gate.** First commit where the full pipeline runs on the new contract. If audio C7 drifts here, the cascade rewire or the writer broadcast format is the suspect.
- **B4 (delete standalone phase nodes — Option A locked pre-kickoff) and B5 (visual selector + cache collapse)** — Python-only. Each individually revertible. No canonical workflow uses those nodes, so the system still runs end-to-end on B3's state.
- **B6 / B7 / B8** — tests + sweep + QA close. Each individually revertible.

---

## 9. Open scope questions for Jeffrey before kickoff

1. **Branch name.** `s30-two-model-selector` (plan default) or land directly on `v2.0-alpha`? Plan default cuts a feature branch given the B3 end-to-end audio gate's blast radius; revert window is wider with a branch.
2. **Sample sprint run.** Single-episode soak run between B5 and B6 (HuMo-free, audio-only path, ~5 minutes wall time on 5080) — plan default YES, or trust the regression suite alone?
3. **`meta` key naming.** `meta["creative_writing_model"]` + `meta["technical_model"]` (plan default), or one merged key like `meta["models"] = {"creative": ..., "technical": ...}`?
4. **Post-sprint soak goal:** validate at least one ungated catalog entry as `vram_fit_tier="PASS"` so the first-run-without-token recovery message has a confident recommendation. Not in S30 scope — log as a follow-up sprint after B8.

**Resolved in-plan (no longer open):**

- `_POLISH_CACHE` collapse — MANDATORY in B5. Two parallel LLM caches on 16 GB = guaranteed OOM. Locked.
- VRAM swap primitive when Slot 1 ≠ Slot 2 — `unload_llm()` (full teardown), NOT `_flush_vram_keep_llm()`. Locked in §6.
- Workflow JSON migration sequencing — per-commit wire (B2a writer, B3 cascade); B4/B5 are Python-only because affected nodes aren't placed in any canonical workflow. Locked per §2a-bis.
- Unknown-`model_id` behavior — two-tier per §0a. Not-on-disk-but-valid → auto-download via `snapshot_download` (the public-release first-run UX). Genuinely-unknown (validation fail / banned format / nonexistent on HF / gated without token / download itself fails) → `UnknownModelError` with specific reason + `huggingface-cli download` recovery hint + top installed alternatives. Never silently substitutes a different model. Locked.
- `MODEL_CONTEXT_CAPS` static dict + `DEFAULT_CONTEXT_CAP = 8192` — DELETED. Replaced by dynamic `config.json` read clamped against `HARD_VRAM_CONTEXT_LIMIT` (default 8192 on 16 GB target, `OTR_HARD_VRAM_CONTEXT_LIMIT` env var to raise on bigger hardware). On miss → raises `ContextCapUnknownError`, no blind fallback. Locked in B1.
- VRAM-fit pre-check at load time — `check_vram_fit(model_id, context_cap)` runs before `load_llm`; fails loud BEFORE the OOM if the model is too big for the budget. Surfaces the "70B on a 16 GB card" failure mode without silent substitution. Locked in B1.
- Teardown sequence — `model.to("cpu")` → `gc.collect()` → `torch.cuda.empty_cache()` → `torch.cuda.ipc_collect()` → `torch.cuda.synchronize()`. The `ipc_collect()` step is required to recover IPC-handle pool fragmentation accumulated by prior FLUX / HuMo / LTX runs. Aligned with the chained-backend-teardown memory. Locked in §6.
- Polish sampling profile — model's `generation_config.json` is consulted FIRST; OTR's `top_p=0.9` / `do_sample=True` defaults apply only to keys the model author didn't specify. A math/coding model whose config says `do_sample=False` is honored. Logged at INFO with the resolved profile. Locked in B5.
- Pre-fetch UX safety net — size estimate via `HfApi().model_info(...)` + free-disk check via `shutil.disk_usage` + single-line announcement in ComfyUI queue UI ("Downloading X — Y GB → path") before any bytes hit the wire. `InsufficientDiskSpaceError` raises if free space minus download minus 5 GB margin goes negative. Locked in B1.
- Auto-download default — **ON**, prioritizing public-release first-run UX. Strangers downloading OTR open the canonical workflow, click Queue, and the loader fetches Mistral-Nemo from HuggingFace once; every subsequent run is fully offline. Triggers only at catalog click + LLM-load-during-execution. Never at JSON parse, dropdown build, ComfyUI startup, or CI. `OTR_MODEL_CATALOG_AUTO_DOWNLOAD=0` turns it off for offline scenarios. Format gating + allow-list + path-traversal rejection stay enforced regardless. Locked in B1.
- Writer widget-order pin in tests — runtime-read from `INPUT_TYPES()`, NOT a hardcoded index list inside the test that the test is supposed to verify. Locked in B6.
- Guardrail false-positive risk on non-LLM media nodes — guardrails test **widget structure** (`model_*` STRING widget name outside writer), not raw `widgets_values` strings. Non-LLM media nodes' legitimate repo-like strings are not flagged. Locked in B6.
- Commit decomposition — 14 commits: B0, B1a, B1a2, B1b, B1c, B2a, B2b, B2c, B3, B4, B5, B6, B7, B8. Each commit is small enough to revert individually; load-bearing audio C7 attribution attaches to B2a (writer JSON wire), B2b (routing), B3 (canonical workflow end-to-end), and B5 (cache collapse + polish rewire). **No audio-baseline roll in S30** — sampling-precedence change feature-flagged OFF; audio byte-identical at every commit boundary. Locked.
- B5 sampling-precedence behavior — **the whole idea is deleted from S30 scope**. No feature flag, no scaffolded-but-dormant code path. `make_polish_generate_fn` continues to apply OTR's hardcoded `_POLISH_TOP_P=0.9` / `_POLISH_DO_SAMPLE=True` unconditionally. The model-author-config-respecting behavior is re-derived from scratch in a later audio-intentional sprint with its own design, tests, and baseline-roll discipline. Scaffolding dormant logic is tech debt; deleting the idea entirely is cleaner. Prime Directive 1 governs.
- `resolve_hf_token()` cross-platform safety — the `winreg` fallback is gated on `os.name == "nt"`. macOS/Linux skip the registry branch entirely. The function returns `None` cleanly on non-Windows when `HF_TOKEN` isn't in `os.environ`. Locked in B1a2.
- `check_vram_fit` honesty — `UNKNOWN` is the **expected** verdict for most uncurated arbitrary `org/name` models (HF `config.json` has no standardized `num_parameters` field; inference is architecture-dependent). The function is a coarse guardrail against the obvious 70B-on-16GB case, not a precise VRAM oracle. Curated PASS-tier entries are the only models with trustworthy verdicts. Locked in B1c.
- HF download progress wiring — `huggingface_hub.snapshot_download` runs in ComfyUI's worker thread (UI thread not blocked), but explicit `tqdm_class` adapter forwards progress to ComfyUI's `comfy.utils.ProgressBar` so users see real progress during the 24 GB first-run download. Locked in B1a2.
- Curated catalog (locked):
  - Gemma-4 E2B-it / E4B-it stay in the catalog. They use the new `transformers_multimodal_text_only` backend (`AutoProcessor` + `AutoModelForImageTextToText` consumed in text-only mode). Gemma-2 is NOT curated; users with Gemma-2 already in their HF cache can still pick it via the locally-scanned admit-path.
  - Pre-catalog-lock backend compatibility smoke runs against every entry; mismatches reject the entry from the catalog with a clear log line.
- `ContextCapVerdict` aligned with `VRAMFitVerdict` — `resolve_context_cap` returns a tiered verdict, never raises; `request_slot` makes a single combined escalation decision in step 6 of the 8-step sequence. Locked in B1b + B1c.
- Label-poisoning fix — writer outputs and `meta` keys broadcast `validate_model_id()`-normalized repo IDs, never raw `widgets_values` strings. New regression test `test_no_legacy_model_id_in_meta` also asserts no `[NOT DOWNLOADED]` substring leaks anywhere in `meta`. Locked in B2a + B2b + B6.
- B7 sweep covers both STRING widget form AND COMBO/list (catalog-dropdown) widget form — name-aware (`model_id` / `llm_model` / `*_model` / `model_*` / `creative_writing_model` / `technical_model`) and structurally aware. Connectable input sockets allowed; widget forms rejected outside the writer unless class carries `NON_LLM_MODEL_WIDGET_OK`. Locked.
- ONE unload primitive — B0 deletes `story_orchestrator._unload_llm` outright (B0 ships a thin forward-shim to `_otr_model_loader.unload_llm` if needed; B1c lands the real symbol and removes the shim in the same commit). The three current importers rewire to the canonical path. Locked.
- B0 dead-runtime audit — grep + `vulture --min-confidence 80` + module-import smoke (clean Python process + ComfyUI Desktop boot) all run BEFORE any deletion. Locked.
- WARN models in gated-error recovery hint — REMOVED. The message recommends a single honest path (HF account setup) rather than a false-choice menu of unverified ungated alternatives. Re-added if/when a `load_strategy` lands. Locked.
- Mocked Slot1≠Slot2 split-routing regression — `test_slot1_neq_slot2_split_routing_full_phase_trace` table-tests all 14 §2c phases. Locked in B6.
- Polish-ON test fixture — separate from C7 byte-identity gate; exercises the B5 polish path that the canonical (polish-OFF) workflow can't. Locked in B6.
- Transition-count assertions — `compute_expected_transitions(fixture_config)` helper walks the actual enabled-phase DAG; hardcoded `== 9` / `== 3` replaced by computed expectation + upper-bound. Locked in B2b's slot scheduler test.
- `request_slot` sequence — explicit ordered call: `validate_model_id` → `auto_download_if_missing` (which itself pre-flight-checks gating + lightweight `model_info` + VRAM-fit on uncurated remote IDs before downloading weights) → `resolve_context_cap` → `check_vram_fit` (final) → `unload_llm` (if needed) → `load_llm`. Locked in B1c.
- Oversize-download avoidance — uncurated remote `org/name` paths fetch lightweight `HfApi().model_info` and run `check_vram_fit` BEFORE `snapshot_download` of weights. A 70B model on a 16 GB card never downloads 80 GB only to fail at load. Locked in B1a2 + B1c.
- Standalone LFC Phase 4/5/6 — Option A (delete files + registrations) locked **pre-kickoff**. No mid-sprint Option A/B branch. Rationale: nodes are orphaned from any canonical workflow JSON per §2a-bis; keeping them as live surface contradicts deletion-bias. Reversible only if Jeffrey explicitly says before kickoff he wants them as rerun helpers.
- Auto-download trigger — **queue-time only** (LLM-load during workflow execution). No catalog-click download (would require a JS extension that S30 doesn't include). Dropdown shows `[NOT DOWNLOADED]` red label; selection just stores a string; Queue button fires the download.
- GGUF in `allow_patterns` — **excluded**. The loader path is `transformers.AutoModelForCausalLM.from_pretrained` which cannot consume GGUF. Future llama.cpp backend sprint can add a separate loader + extend `allow_patterns` then.
- `validate_model_id` allow-list — admits curated + locally-scanned + valid `org/name` (when auto-download enabled). Hardware-inclusive: users can slot in any HF model their rig handles, not just curated picks. Structural rejection (path-traversal, drive letters, unsafe formats) still applies regardless.
- `check_vram_fit` verdict — **tiered enum** (`PASS / WARN / UNKNOWN / FAIL`), not binary `bool`. `FAIL` reserved for clearly-oversized cases (≥1.5× ceiling overage on parseable param counts); ambiguous uncurated models return `WARN` or `UNKNOWN` and proceed with a logged caution. Locked in B1c.
- `estimate_model_size_gb` — call sites are constrained for **speed and test reproducibility**, not network avoidance. The HF public API is free and key-less for ungated repos; OTR's stance is "no required keys, no required payment," not "no network calls ever." Allowed in `auto_download_if_missing` and any user-action-driven path. Excluded from `scan_local_llm_cache`, `build_dropdown_choices`, `INPUT_TYPES`, startup, and pytest paths without explicit mocking. Locked in B1a.
- API key + payment stance — no required HF_TOKEN to run OTR end-to-end (three ungated curated models provide that floor). No paid hosted APIs at runtime. `HF_TOKEN` is consumed only when the user has it set and only for gated repos; gated-without-token raises `GatedModelError` with both the "create free HF account" path and the ungated-alternatives path so the user can choose. Locked.
- VRAM profiler placement — `scripts/vram_profile_slot_swap.py` (outside `tests/`); not in pytest count, not in canonical regression. Manual soak by Jeffrey only. Locked in B6.
- B7 structural sweep — distinguishes widget-form STRING entries (rejected outside writer) from connectable-input-socket STRING entries (allowed on consumer nodes). Locked in B7.

---

## 9b. Sprint-length estimate (honest)

**Realistic: 6–9 focused working days.** Earlier passes underestimated (5–7); each of B1a2 / B1c / B2b / B5 / B7 has grown scope through the round-robin synthesis cycle. Per-commit table reflects the current shape:

| Commit | Risk | Estimate |
|---|---|---|
| B0 | Medium — pure deletion + grep + vulture + import smoke + unload-shim wiring | 0.5 d |
| B1a | Low — offline catalog dataclass + scan + dropdown + validator | 0.5 d |
| **B1a2** | **HIGH — auto-download + `resolve_hf_token` (cross-platform gating) + `GatedModelError` pre-flight + disk pre-check + HF progress → ComfyUI ProgressBar wiring** | **0.75 d** |
| B1b | Medium — `MODEL_CONTEXT_CAPS` migration + `ContextCapVerdict` tier alignment + every caller updated | 0.5 d |
| **B1c** | **HIGH — `unload_llm` (replacing the B0 shim) + `request_slot` (8-step sequence including combined verdict escalation) + `check_vram_fit` + multimodal-text-only loader backend + hard-mock conftest fixture** | **1.25 d** |
| B2a | Medium — widget surgery + writer JSON wire with explicit slot indices + label-normalization-at-broadcast + AST leak test | 0.5 d |
| **B2b** | **HIGH — live-writer DAG slot scheduler + measure-from-DAG transitions + polish-ON fixture + 14-phase routing table** | **1.25 d** |
| B2c | Low — `cleanup_model_id` legacy-strip deletion + test residue | 0.25 d |
| B3 | Medium — cascade widget + toggle deletion + canonical JSON wire with explicit link IDs + before/after widget pin | 0.5 d |
| B4 | Low — atomic node + phase function deletion | 0.25 d |
| **B5** | **HIGH — `_POLISH_CACHE` collapse + `visual/llm_polish.py` rewire + `make_polish_generate_fn` cache-entry-arg extension + example-workflow stub** | **1.0 d** |
| B6 | Medium — wiring tests + structure-based guardrails + Slot1≠Slot2 split-routing + no-legacy-meta-id regression + 14-phase table test | 0.75 d |
| **B7** | **HIGH — AST sweep with widget-vs-socket distinction + STRING + COMBO/list detection + `NON_LLM_MODEL_WIDGET_OK` marker rollout + path-aware exemptions** | **0.75 d** |
| B8 | Low — QA doc + ROADMAP update + manual VRAM soak + example workflow shipment | 0.5 d |

**Total: ~9.25 days, range 6–9 depending on round-robin friction at the five HIGH-risk commits (B1a2, B1c, B2b, B5, B7).** Round-robin synthesis at each adds ~0.25 d. Soak runs between commits add the same. Mock fixtures + migration scripts don't exist yet; if they did, ~2 days could shave off.

If the sprint slips past 9 days, the most likely culprits in order: (1) B2b's live-writer DAG mapping turns out to have a dependency that batches-poorly, (2) B5's polish path has a subtle correctness drift the polish-OFF canonical workflow can't catch, (3) B1c's multimodal-text-only backend takes a Gemma-4-specific shape that needs adapter work, (4) B7's COMBO-form sweep false-positives on a media node nobody remembered.

Plan does not promise 2–3 days. Setting the right expectation upfront avoids the trap where slipping past day 3 feels like failure when it's the honest shape.

---

## 10. References

- Scoping doc: `docs/2026-05-13-two-model-selector-scoping.md`
- Prior partial implementation: `tests/test_two_llm_split.py`
- S29 close doc: `docs/2026-05-14-S29-final-qa-review.md`
- Forbidden sweep gate: `docs/_s28_forbidden_sweep.py`
- Workflow link validator: `tools/validate_workflow_links.py`
- CLAUDE.md Prime Directives 1, 2, 3 + Bug-Log Pipeline + Round-Robin Consultation
- ROADMAP.md sequencing — Sprint B → C → A

---

**End of sprint plan.** No code changes here — this is the execution playbook only. Kickoff: Jeffrey signs off on §9 open scope questions; commit **B0** starts (`story_orchestrator.py` dead LLM stack deletion + `__init__.py` forensic-comment scrub).
