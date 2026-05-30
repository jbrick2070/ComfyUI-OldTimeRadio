# OTR Architecture / Cleanup Audit — 2026-05-30

**Branch:** `v2.0-alpha` | **HEAD:** `f4d05b9` | **Pre-req met:** API node-load clean (35 nodes, no import errors), `OTR_WorkflowValidator` reports `widget_vector_drift=0`.
**Scope:** `workflows/otr_scifi_16gb_full.json` (30 nodes, 68 links) + the OTR custom-node Python. Grounded in this session's live headless runs (`babe3b74`, `dfeca7ba`), the round-3 dead-code audit, and the BUG-280/281/291/292/293 fixes just landed.
**Stance:** cleanup, not simplification. Flexibility / fallback / model-agnostic / test-gate / debug surfaces are KEEP by default.

---

## Bottom line

- **Biggest cleanup win:** there is no large dead-code win left — the round-3 lean-down already removed ~1,330 LOC and this session closed the widget-drift + output-split + VRAM bugs. The only real structural win remaining is **consolidating the duplicate LTX LoRA loaders (nodes 60→61, same file at 0.5 then 0.2) into one 0.7 loader** and **retiring the duplicate AppData install** of the node pack.
- **Biggest risk:** **position-based widget serialization + the dual install.** ComfyUI maps `widgets_values` by index, so any change to a node's `INPUT_TYPES` widget set/order silently corrupts the saved workflow — this bit twice this session (BUG-281 node 14 real drift; BUG-293 node 62 false-positive from a COMBO-tuple convention gap). The new `OTR_WorkflowValidator` hard-raise now gates exactly this, so the risk is contained as long as that node stays wired.

---

## 1. Execution path (traced from the live runs)

**Core flow → final `otr/obs` mp4:**

`OTR_LedgerScriptWriter (1)` → `OTR_LedgerFreezeCascade (62)` → audio fan-out [`OTR_BatchBarkGenerator (11)`, `OTR_KokoroAnnouncer (13)`, `OTR_MusicGenTheme (14)`, `OTR_BatchAudioGenGenerator (15)`] → `OTR_SceneSequencer (3)` → `OTR_AudioEnhance (4)` → `OTR_EpisodeAssembler (7)` → video: `OTR_SignalLostVideo (12)` / `OTR_VideoComposite (52)` → FLUX branch [`OTR_DeferredCheckpointLoader (22)` → `OTR_FluxBranchGate (71)` → `OTR_BatchFluxRender (23)`, `OTR_BatchFluxPortraitRender (59)`] + HuMo branch [`OTR_HuMoTierLoader (72)` → `OTR_LtxBranchGate (70)` → `OTR_BatchHumoRender (51)` / `OTR_BatchLTXRender (55)`] → `OTR_RTXUpscale (56)` → `OTR_PostUpscaleProcgenBlend (58)` → **`otr/obs/<ep>_procgen_blended.mp4`**.

**Support / topology (in-DAG but not "story" nodes):** `OTR_WorkflowValidator (63)` (OUTPUT_NODE gate), `OTR_VideoPlan (20)`, `OTR_FixedShotDurationStub (21)`, `OTR_UnloadAll (24)`, `OTR_SaveToEpisodeWorkspace (25)`, `LowVRAMCheckpointLoader (54)`, `LoraLoaderModelOnly (60, 61)`.

**Fallback / utility flow:** the VRAM levers (`free_otr_pipeline_residue`, `_flush_vram_keep_llm`), the two branch gates, the Kokoro announcer bus (announcer lines bypass Bark), the radio-bookend fallback prompt when no ledger still exists, and the `needs_full_rerun` → shots=0 short-circuit.

**Outside every workflow path (by design):** the 8 registered-but-unwired nodes — `OTR_VisualBridge / VisualPoll / VisualRenderer / VisualPromptCoercion / VisualExtractFluxPrompt` (subprocess sidecar, spawned not wired), `OTR_VRAMGuardian`, `OTR_VRAMContextTest`, `OTR_ProjectStateLoader`. These are KEEP-by-design (sidecar/topology/state), confirmed in the round-3 audit.

---

## 2. Cleanup table

| File / Node ID | Item | Why it may be unnecessary | Evidence | Recommendation | Risk | Required test |
|---|---|---|---|---|---|---|
| workflow nodes 60 + 61 | Duplicate `LoraLoaderModelOnly` — same `ltx-2.3-22b-distilled-lora-384-1.1.safetensors`, 0.5 then 0.2, chained 54→60→61→55 | Two sequential loads of the identical adapter = `W + 0.5·Δ + 0.2·Δ = W + 0.7·Δ`; one loader at 0.7 is weight-equivalent and drops a node | Verified in the workflow JSON this session; LoraLoaderModelOnly patches additively | **PROBABLY REMOVE AFTER TEST** | Med (render parity / tiny precision diff) | A/B render: current stack vs single 0.7; keep only if output + stability identical |
| AppData install: `…\AppData\Local\Programs\ComfyUI\resources\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio` | Second copy of the whole pack loads alongside the `Documents` dev repo | Two copies register every node ("All 35 nodes loaded" prints twice); the root cause of the output-base split | Boot log shows both import paths; BUG-292 | **KEEP BUT DOCUMENT** (operator action) | Med | After retiring one copy, re-run API load + a smoke; confirm 35 nodes once, output under one tree |
| `nodes/_otr_model_runtime.py` | `get_backend_for_row` + Transformers adapters; only `tests/test_loader_backend_protocol.py` imports it | Live loader uses `_otr_loader_backends` duck-typed path, not this | Round-3 audit proof (rg -a) | **KEEP** | — | n/a — model/provider plumbing in a model-agnostic project; retire only in a deliberate sprint with its test |
| `nodes/_voice_backends/` | Engine-selection registry (bark/kokoro); only `tests/test_voice_backends.py` imports it | Live TTS bypasses it (`_otr_bark_lib`, kokoro lazy import) | Round-3 audit proof | **KEEP** | — | n/a — fallback/provider plumbing; wire-ready abstraction |
| `_resolve_radio_still_path` (×2), `_load_ledger` (×2) | Mirrored helpers across batch_ltx / batch_humo / batch_flux_portrait | Look like duplication | `BUG-LOCAL-121` / `BUG-LOCAL-076` comments; tested | **KEEP** | — | n/a — intentional mirrors, pinned by constraint |
| `OTR_WorkflowValidator (63)` | "opt-in, S14.2" node, visually disconnected in canvas | Looks vestigial | OUTPUT_NODE that self-resolves; runs every queue; now hard-gates widget drift | **KEEP** | — | already test-pinned (`test_otr_workflow_validator`, `test_workflow_canonical_baseline`, `test_workflow_json_wiring_invariants`) |
| `nodes/scene_sequencer.py:152` `DEFAULT_OUT` | Hardcoded `~/Documents/ComfyUI/output/otr/audio` | Same hardcode class as the BUG-292 `_default_out_dir` (now fixed); a non-portable fallback default | grep this session | **PROBABLY REMOVE AFTER TEST** | Low | Route through `_otr_paths.otr_audio_dir`; gate full suite + a smoke (it's a fallback, normally overridden by `led.out_dir`) |
| `scripts/*` hardcoded `C:\Users\jeffr\…` paths (ltx_motion_smoke, overnight_bug_hunt, qa_*, treatment_scanner, serve_ledger) | Operator dev scripts with absolute paths | Not portable | grep this session | **KEEP BUT DOCUMENT** | Low | n/a — machine-local dev tooling (most are `scripts/_*.py` gitignored); not production node code |
| `_otr_paths` `_legacy_audio/_legacy_stills/_legacy_portraits` fallbacks | Returned when a caller omits `episode_id` | Look like dead branches | Defensive fallbacks for auto-pick callers | **KEEP** | — | n/a |
| `nodes/_otr_critic_rubric.py`, `visual/planner.py`, `visual/postproc/` | — | — | Already removed (round-3, commits `5fc49d5`/`6c80597`/`b5c1e93`) | **DONE** | — | — |

No item rates **SAFE TO REMOVE** outright — the proven-dead code is already gone, and everything remaining is either a tested fallback/plumbing surface or a *test-gated* consolidation.

---

## 3. Serialization & compatibility warnings

This is the live fault line for this pack — read before touching any node surface.

- **`widgets_values` is positional.** Adding, removing, or reordering a widget in any node's `INPUT_TYPES` shifts every later saved value to the wrong slot, silently. Two instances this session: **BUG-281** (node 14 `OTR_MusicGenTheme` carried a stale 5th slot for the `forceInput` `script_json` → fixed to 4) and **BUG-293** (node 62 `OTR_LedgerFreezeCascade` *looked* drifted because the checker didn't count a tuple-form COMBO — false positive, fixed).
- **`forceInput` drops the slot.** A widget-typed input flagged `forceInput: True` occupies **no** `widgets_values` slot (it's socket-only). This is the #1 way the JSON and `INPUT_TYPES` silently diverge. The freeze cascade has five (`script_text/script_json/news_used/estimated_minutes/technical_model`).
- **`seed`/`noise_seed` add a hidden slot.** ComfyUI injects a `control_after_generate` companion immediately after an INT `seed`/`noise_seed` widget — `widgets_values` is one longer than the schema's widget count.
- **COMBO can be list *or* tuple.** Choices declared as `("a","b")` (tuple) count as a widget just like `["a","b"]`; the validator now handles both (the BUG-293 gap).
- **Gate that now enforces all of the above:** `OTR_WorkflowValidator` (node 63) recomputes each OTR node's expected serialized-slot count (mirroring `scripts/otr_api._serialized_slot_names`) and **raises** at queue time on any mismatch. `validate_anyway=False` is the operator bypass. **Keep this node wired** — it is the only automated guard against the position-drift class. Its `IS_CHANGED` correctly keys on path + mtime + inputs.
- **`VALIDATE_INPUTS`:** largely absent across the pack; `OTR_WorkflowValidator` is the de-facto whole-graph contract gate instead. Adding per-node `VALIDATE_INPUTS` is optional hardening, not required.
- **Looks unused but MUST stay** (do not "clean"): the 8 sidecar/topology nodes (§1), `_otr_model_runtime` + `_voice_backends` (model/provider plumbing + tests), the intentional `_resolve_radio_still_path` / `_load_ledger` mirrors, both branch gates, the VRAM levers, and the `_legacy_*` path fallbacks.

---

## 4. Refactor order (test-gated items only)

No SAFE-TO-REMOVE items remain, so this is the order for the **PROBABLY REMOVE AFTER TEST** + operator-action items, each as an isolated commit with full gates (Bug Bible + core + audio + a validator POST + a smoke):

1. **LoRA consolidation (nodes 60/61).** inspect → A/B render current-stack vs single `0.7` loader → if parity holds, edit node 60 to 0.7 + delete node 61 + rewire 60→55 in the JSON → validator POST (`widget_vector_drift=0`, validator OK) → smoke → commit. *(Workflow-JSON change → the validator + `OTR_WorkflowValidator` are the re-wire check.)*
2. **`scene_sequencer.DEFAULT_OUT` → `_otr_paths.otr_audio_dir`.** inspect callers (confirm it's only a fallback when `out_dir` isn't passed) → route through the resolver → full suite + smoke (output stays under one `otr/{episodes,obs}` tree) → commit.
3. **Retire the duplicate AppData install** (operator action, not a code commit). Disable/uninstall whichever copy does NOT match the ComfyUI binary that runs headless → relaunch → confirm "35 nodes loaded" prints **once** and all output lands under the single `Documents\ComfyUI\output\otr` tree.

Nothing above is urgent; the pack is healthy and gated. Items 1–2 are small wins; item 3 removes the dual-install fragility that caused the output split in the first place.
