# SPEC: VRAM residency for the video lanes

**Status:** design draft, revised after a QA pass on 2026-10-14. No code. Not approved.
**Scope:** the video lanes on the 8 GB and 16 GB machines. The writer LLM, Bark and the image engines appear only at their handoffs.
**Gate:** no code until the kibitz arc has run and its findings are folded in (CLAUDE.md, 2026-08-17 amendment). Depth is the driver's call, with a contrarian in each round (CLAUDE.md, 2026-09-11).
**Decisions for the operator:** section 9.

## 0. Operator rulings this design must respect

These are standing rules. The design does not work around them.

- **No caps (2026-09-06).** "remove all caps, just let the system take up as much memory as it needs." A budget that forces partial CPU placement makes renders useless (`nodes/_otr_model_loader.py`, lines 479 to 505).
- **Never refuse on a number (2026-09-23).** "we should never refuse on numbers, only OOM decided." Freeing memory is welcome. Predicting a shortfall and refusing is not (`nodes/_otr_video_engines/motion_common.py`, lines 562 to 588).
- **No reserves (2026-09-26).** "I don't like messing with reserves." "We need to unload models after they are used, and if it OOMs we record it" (`nodes/_otr_shared/boot_contracts.py`, lines 7 to 12).
- **Two boxes (CLAUDE.md section 0B).** Never fix one box by breaking the other.

So this design is not a budget gate and does not refuse before loading. It has four parts: attribution, one policy for when memory is freed, an OOM-driven retry, and recorded decisions. Reintroducing a reserve or a refusal needs a new operator ruling first (open decision O2).

## 1. Problem

Five mechanisms free GPU memory today. They overlap, and none of them computes a budget.

| Mechanism | Where | Verified call sites |
|---|---|---|
| Post-prompt release (ComfyUI `free_memory` queue flag) | `nodes/_otr_vram_levers.py:294` | Requested at prompt start, `nodes/_otr_workflow_validator.py:775`. Applied when the prompt ends |
| Pipeline residue release (writer LLM, Bark, detach of tracked patchers) | `nodes/_otr_vram_levers.py:121` | `_otr_workflow_validator.py:776`, `otr_image_gen_dispatcher.py:2166`, `eng_ltx25.py:1267` |
| Node-output eviction (opt-in) | `nodes/_otr_video_engines/wrapper_bridge.py:477` (`evict_after_use`), `:311` | Only `lumina_image.py:395` and `z_image_turbo.py:391` pass it |
| Reclaim all loaded models | `wrapper_bridge.py:405` | Callers not checked |
| Soft cache flush | `wrapper_bridge.py:295` | Callers not checked |

Measured facts:

1. **Streaming can fail without an OOM.** A 4060 clean-room run on 2026-09-02 loaded a classic diffusion model with 0.00 MB usable and streamed it from host RAM every step: about 2 minutes per step, about 42 minutes per still (docstring, `wrapper_bridge.py:311` to `:334`). Nothing crashed.
2. **The H3 reserve is gone.** The 12 GiB reserve (`--reserve-vram 12`) was removed on 2026-09-26 (`boot_contracts.py` header; `apple/HANDOFF_LOG.md`, the entries for 84c98900 and 205e96ad). The earlier receipts depend on it. The 1344x768 leg peaked at 9.15 GiB with the reserve. The 832x480 leg peaked at 15.39 GiB and failed without it (`apple/evidence/lane_receipts/lane19-h3_low_video.md`, rows at lines 44 to 46). The receipt attributes the difference to the boot flag. The two legs also differ in canvas and length.
3. **The 8 GB H3 failure is unexplained.** On 2026-09-28, `otr_8gb_still` with `minimax_h3_video` died at beat 4 while loading the text encoder. CUDA reported 0 bytes free while PyTorch held 213 MiB (`apple/HANDOFF_LOG.md`, entry "H3 on the 8 GB 4060").
4. **The raw H3 recipe fits the 4060 in isolation.** The lab recipe `minimax_h3_fl2va_raw_recipe` is LAB-PROVEN on the 4060 at 864x480 for 90 frames, with peaks of 7.147, 6.788 and 6.788 GiB (`apple/MACHINE_MATRIX.md:22`). That is a lab recipe, not the OTR engine adapter. My inference, to be tested under P2: the 2026-09-28 failure comes from what else shares the GPU, not from H3's own footprint.
5. **Host RAM binds too.** The LTX 8 GB run (TEST_WAVE row B4) peaked at 7,300 MiB with 0.42 GB of free RAM at the low point, before the per-beat T5 reload fix (PBUG-20260926-01). The post-fix 4060 figure is still owed.

Conclusion: no single component decides when memory is freed. The five mechanisms fire on different triggers, and nothing records which one freed what. This design fixes that.

## 2. Goals and non-goals

**Goals**

- **G1.** One policy decides when memory is freed: at prompt start, before a stage loads, after a node's last use, and at prompt end. It replaces the five overlapping triggers, and every action it takes is recorded.
- **G2.** Attribution. Each stage records what the GPU holds (whole-GPU used memory, PyTorch reserved and allocated, the comfy-loaded models), so an 8 GB failure can be explained from the log.
- **G3.** OOM is the only decider. No VRAM cap, no reserve, no refusal on a forecast. A real OOM is recorded with the allocation it names.
- **G4.** One retry after a real OOM, with smaller compute chunks only. Weight spill to host memory is out, because a forced spill is a cap in effect (open decision O3).
- **G5.** CPU, MPS and cloud lanes are unchanged. The policy is inert without CUDA.
- **G6.** ComfyUI stays the host. App mode, the Registry install path and the canonical graph do not change.

**Non-goals**

- Budgets, reserves, forecast-based refusals or any cap, unless the operator reverses a ruling in section 0.
- Replacing ComfyUI's runtime or the `comfy` library.
- Copying FreeVideo's code (section 8).
- The front end (section 10).

## 3. Current state

### 3.1 Pieces that exist

| Piece | What it does | Status | Evidence |
|---|---|---|---|
| `nodes/_otr_shared/gpu_residency.py` | Cross-process GPU lease, NVML probes, `wait_until_below_mb`, `wait_until_stable` | Imported by two modules only: `otr_image_gen_dispatcher.py:43` and `motion_common.py:37`. Video engines use it through `motion_common.py` | grep |
| LLM placement (`_otr_model_loader.py`) | Placement; OOM classification in `_is_memory_placement_failure` (line 623) | `_plan_max_memory` returns `None` by operator ruling (line 479) | lines 479, 623 |
| `check_vram_fit` (`_otr_model_catalog.py:2477`) | Estimator | Called only from `_otr_model_loader.py:2047`. LLM only. FAIL is advisory | grep |
| `render_driver.py` OOM classifier | Recognises `OutOfMemoryError` and "out of memory" | Exists, around lines 280 to 289 | grep |
| `_assert_beat_affordable` (`render_driver.py:4076`) | Prices a beat from the request | Admission at line 4466. The cost table `QUALIFIED_COST_ROWS` is empty (`motion_common.py:1032`), so it gates nothing | grep |

The video side has an OOM classifier and a pricing function that gates nothing. Both are prior art to reuse. The LLM side has placement.

### 3.2 Engines and measured status

| Lane | Engine | 16 GB (RTX 5080 Laptop) | 8 GB (RTX 4060 Laptop) | Third-party pack |
|---|---|---|---|---|
| `h3_low_video` | `minimax_h3_video` | PROVEN (`apple/MACHINE_MATRIX.md:18`). Lab receipts: 864x480 f90 at 7.28 GiB (lane 19, row 44); 1344x768 f124 at 9.15 GiB, taken with the reserve that has since been removed. Raw recipe at 6,315 MB (FL2VA) and 6,678 MB (REF2VA) (`MACHINE_MATRIX.md:162`) | Raw recipe LAB-PROVEN at 7.1 GiB (`MACHINE_MATRIX.md:22`). The OTR adapter failed at beat 4 on 2026-09-28 | None |
| `h3_low_audio_in` | `minimax_h3_audio_in` | Lane 20 receipt | Not measured | None |
| HuMo | `humo` | PROVEN, 32 published episodes | "run out of memory on 8 GB" (`apple/VIDEO_MODELS.md:214`) | None |
| LTX 2b 0.9.8 | `ltx_8gb` | PROVEN on a 20 GB Ampere card, 1 act | TEST_WAVE B4: peak 7,300 MiB, 0.42 GB free RAM (pre-fix) | None |
| LTX 2.5 | `ltx25` | COMPONENTS MEASURED (`MACHINE_MATRIX.md`) | `otr_8gb_ltx25_foley` completed 21 beats in 6:26:58, no OOM (`HANDOFF_LOG.md:855`, footnote at 861) | None |
| AnimateDiff haunted | `animatediff15_v3_haunted_video` (`GhostSignalV3HauntedEngine`, `eng_ghost_signal_official.py:119`) | Not the target | PROVEN on the 4060 (`MACHINE_MATRIX.md:13`) | ComfyUI-AnimateDiff-Evolved |

The mesh stage and the other ghost-signal variants (for example `eng_ghost_signal_stillin_lab.py`) were not measured. They stay out of Phase 1.

### 3.3 Regression run, 2026-10-14 (`qa-20261014-8gb-regress`)

| Leg | Result | Wall time | Sampled whole-GPU peak | Cause of result |
|---|---|---|---|---|
| low (viz_camera) | PASS, 8 beats | 879 s | 7,605 MiB | |
| still (still_motion) | PASS, 8 beats | 703 s | 7,403 MiB | |
| video | FAIL, exit 1 | 8 s | 2,096 MiB | Never reached the GPU. The launcher sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` (`otr_regress_20261014_server.cmd`, lines 10 to 11). The video lane's weight-metadata check then failed with `OfflineModeIsEnabled` (`server.log:1577`). This is a launcher setting, not a network outage |

Three cautions about these numbers:

- The sampler (`sample_vram.ps1`) reads whole-GPU `memory.used` every 15 seconds. It counts every process on the GPU, and 15 seconds is too coarse to catch a peak.
- The same profile and engine (viz_camera) logged a peak of 4,379 MiB in TEST_WAVE B1. The regression figure may include another process's memory. Attribution (P0) settles it.
- 7,403 and 7,605 MiB are about 7.2 and 7.4 GiB. The 4060 has 8,188 MiB in total (`HANDOFF_LOG.md:1006`), so the headroom is under 1 GiB.

## 4. Proposed design

### 4.1 Ownership

- **The pack owns** when memory is freed, the attribution, the OOM handling and the receipts.
- **ComfyUI and `comfy` keep** model code, samplers, VAEs and the App-mode UI. The `comfy` library still supplies the text-encoder loader (`FreeVideo`'s `encoder_checkpoint.py` is the precedent) and some model-management hooks.

### 4.2 One free-timing policy

`ResidencyPolicy` (name provisional) runs at four seams: prompt start, before a stage loads, after a node's last use, and prompt end. At each seam it chooses which existing free action to run (from the table in section 1), and records the choice. It consolidates the five triggers and adds no new free mechanism.

The video path already has the hook. `run_graph` takes `evict_after_use` (`wrapper_bridge.py:479`). Fourteen modules call `run_graph`, covering the video engines and the image engines. The policy supplies the node set, so video engines start using the same eviction the image engines use today. This is a change to verify in P3, not an assumed fact.

### 4.3 Attribution (first deliverable)

Before any policy change, each stage logs at start and end:

- whole-GPU used memory (nvidia-smi),
- `torch.cuda.memory_reserved` and `torch.cuda.memory_allocated`,
- the names of the comfy-loaded models (`current_loaded_models`),
- the policy actions taken at that seam.

Sampling runs at 1 Hz during a stage. This is the only way to explain the 2026-09-28 gap (CUDA 0 free against PyTorch 213 MiB) and the 7,605 against 4,379 MiB difference.

### 4.4 OOM and retry

- Classify with the existing OOM checks (`render_driver.py`, around lines 280 to 289; `_is_memory_placement_failure`, `_otr_model_loader.py:623`). The comfy check `is_oom` (`apple/PROD_BUG_LOG.md`, near line 13969) is a second input.
- Record the failing allocation and its size. A real OOM says how much it wanted, which is ground truth.
- Retry once with smaller compute chunks: feed-forward, projection and attention chunking. Do not spill weights to host memory. A second OOM fails loud and names the stage.
- Never refuse before loading. Never refuse on a forecast.

### 4.5 Determinism

A retry with smaller chunks can change floating-point reduction order. FreeVideo's planner documents the same effect. The episode report records the placement and the retry. A replay claim covers runs with the same recorded placement only (O4).

### 4.6 Retiring the workarounds

The five triggers are consolidated first, then retired one lane at a time, each with a receipt. A profile flag (`residency = "legacy" | "policy"`) keeps the current path during the proof period. The legacy path is removed after the proof. It is not kept for old saved data.

### 4.7 CPU, MPS and cloud

The policy is inert when CUDA is absent, and for cloud and remote lanes. Mac lanes do not change.

### 4.8 Weight streaming (not proposed)

Streaming H3 weights from host memory is a placement change, and the operator rulings in section 0 disfavour it. It is not proposed here. If the 8 GB H3 target is pursued, it needs an explicit ruling first (O5).

## 5. Scope

- **Phase 1 lanes:** `h3_low_video`, `h3_low_audio_in`, `ltx_8gb`, the 8 GB LTX 2.5 lanes, `humo` (16 GB first), and the AnimateDiff haunted lane. The haunted lane is proven on the 4060 today and must not regress.
- **Handoffs only:** the writer LLM, Bark, and the image engines (FLUX, z-image, Lumina, ideogram). They hold VRAM across the video stage, so the handoff is in scope. Their internals are not.
- **Out:** the mesh stage and the other ghost-signal variants, until measured.

## 6. Profiles and measured numbers

| Profile or lane | Machine | Number | Source | Note |
|---|---|---|---|---|
| `otr_8gb_low` | RTX 4060 8 GB | 7,605 MiB sampled | Regression run | Whole-GPU reading |
| `otr_8gb_low` | RTX 4060 8 GB | 4,379 MiB | TEST_WAVE B1 | Conflicts with the line above. P0 settles it |
| `otr_8gb_still` | RTX 4060 8 GB | 7,403 MiB sampled | Regression run | Whole-GPU reading |
| H3 raw recipe, 864x480 f90 | RTX 4060 8 GB | 7.147 / 6.788 / 6.788 GiB | `MACHINE_MATRIX.md:22` | Lab recipe |
| H3 f90, 864x480 | 16 GB box (lane 19) | 7.28 GiB | Lane 19, row 44 | Lab receipt |
| H3 f124, 1344x768 | 16 GB box (lane 19) | 9.15 GiB | Lane 19, row 45 | Taken with the reserve, which is now removed |
| H3 f107, 832x480 | 16 GB box (lane 19) | 15.39 GiB, FAILED | Lane 19, row 46 | Taken without the reserve |
| 4060 total VRAM | RTX 4060 Laptop | 8,188 MiB | `HANDOFF_LOG.md:1006` | |
| 4060 later peak | RTX 4060 Laptop | 7,579 MiB through 161-frame segments | `HANDOFF_LOG.md:1006` | |

## 7. Proof plan

- **P0, attribution.** Add the stage log and a 1 Hz sampler to one 8 GB leg (`otr_8gb_low` on the 4060). Gate: the log explains the 7,605 against 4,379 MiB difference, and the 2026-09-28 text-encoder gap.
- **P1, 8 GB regression.** Re-run low, still and video. For the video leg, either pre-cache the weight metadata or run without the offline flag, and record which. Gate: 3 of 3 pass, with peaks at 1 Hz.
- **P2, the 2026-09-28 failure.** Reproduce `otr_8gb_still` with `minimax_h3_video` with attribution on. No policy change in P2 unless attribution shows which free action would have prevented the failure. Gate: the failure is explained by numbers.
- **P3, 16 GB regression.** Run `humo`, `ltx25` and H3 at 1344x768 on the 5080 with the policy on. Gate: wall time within 10% of the current receipts, and no new OOM. This also verifies the `evict_after_use` claim in section 4.2.
- **P4, OOM path.** Force a real OOM (for example a larger canvas on the 4060). Gate: the failing allocation is recorded, one chunk-only retry runs, and a second OOM fails loud.
- **P5, determinism.** Run one episode with and without a placement change. Record both hashes. Gate: every difference is explained by a recorded placement change.

Gates across the plan: every OOM recorded and named; no refusal on a number (none exist by design); attribution explains every 8 GB peak; wall time within 10% on the 5080; both boxes pass (CLAUDE.md section 0B).

## 8. Reuse and licensing

- FreeVideo's code is Apache-2.0. This pack is MIT (`LICENSE`, Copyright 2026 Jeffrey Brick). The default is to re-derive the design from our own measurements, not to copy code.
- FreeVideo's model weights fall under the MiniMax H3 Community License, with territorial and acceptable-use restrictions. Our H3 lanes already use MiniMax H3 weights, so this is not new.

## 9. Open decisions (operator)

- **O1. Scope.** The video stage only (recommended), or the whole run now?
- **O2. Reserves.** The 2026-09-26 ruling says no reserves. Keep it (recommended), or reopen it? The design works without one. A headroom target would need a new ruling.
- **O3. Retry scope.** Chunk-only retry after a real OOM (recommended), or also allow weight spill to host memory?
- **O4. Determinism.** Accept a recorded, placement-dependent output after a retry, or never change placement mid-episode?
- **O5. H3 on 8 GB.** Weight streaming is a placement change, which the no-cap rulings disfavour. Consider it at all, or keep H3 as a 16 GB lane and move 8 GB work to AnimateDiff and LTX?
- **O6. Legacy path lifetime.** How long the rollback flag stays after P3 passes.

## 10. Separate track: the front end

The operator finds FreeVideo's workspace much nicer than the ComfyUI App-mode form. FreeVideo's look comes from one scoped CSS token file (`web/theme.css`) inside a ComfyUI web extension, plus a Python mirror of the same palette for its native launcher (`freevideo_engine/branding.py`). This pack already ships a `js/` folder with six JavaScript files and one JSON file. A front end is a separate design with its own kibitz arc. It does not block this work and is not decided here.

## 11. Process

- Architecture needs a design arc before any code (CLAUDE.md, 2026-08-17 amendment). This document is the input to that arc.
- Depth is the driver's call, with a contrarian in each round (CLAUDE.md, 2026-09-11). The earlier "full four-round" wording is superseded.
- The two-box rule applies (CLAUDE.md section 0B).
- Commits name files explicitly.

## Revision note

Revised on 2026-10-14 after a Sonnet QA pass that checked 96 claims and confirmed 74. The findings changed the design. The first version proposed a reserve and a refusal before loading. Both conflict with the operator rulings in section 0. That version also stated the H3 reserve as current, called the video-leg failure a network error, and gave the regression peaks in GiB when they were in MiB.

## Appendix: evidence index

- `nodes/_otr_vram_levers.py:121, 294`
- `nodes/_otr_video_engines/wrapper_bridge.py:295, 311, 334, 405, 477, 479`
- `nodes/_otr_shared/gpu_residency.py`
- `nodes/_otr_shared/boot_contracts.py`, lines 1 to 16
- `nodes/_otr_model_loader.py:479 to 505, 623, 2047`
- `nodes/_otr_model_catalog.py:2477`
- `nodes/_otr_video_engines/motion_common.py:37, 562 to 588, 1032`
- `nodes/_otr_video_engines/render_driver.py:280 to 289, 4076, 4466`
- `nodes/_otr_workflow_validator.py:775 to 776`
- `nodes/otr_image_gen_dispatcher.py:43, 2166`
- `nodes/_otr_video_engines/eng_ltx25.py:1267`
- `nodes/_otr_video_engines/eng_ghost_signal_official.py:119`
- `apple/evidence/lane_receipts/lane19-h3_low_video.md`, rows at lines 36 to 46
- `apple/HANDOFF_LOG.md`: entries for 84c98900 and 205e96ad, the 2026-09-28 H3 entry, lines 855, 861 and 1006
- `apple/MACHINE_MATRIX.md`: lines 13, 18, 22, 162
- `apple/TEST_WAVE.md`: rows B1 (line 117) and B4 (line 120)
- `apple/VIDEO_MODELS.md:214`
- `CLAUDE.md`: section 0B (line 442), the 2026-08-17 amendment, and the 2026-09-11 round-count directive (lines 275 to 278)
- `D:\otr-4060-testing\otr_regress_20261014_server.cmd` (outside the repo)
- `D:\otr-4060-testing\qa-20261014-8gb-regress\` (outside the repo)
- FreeVideo (external): `docs/execution-planning.md`, `web/theme.css`, `freevideo_engine/branding.py`, `freevideo_engine/encoder_checkpoint.py`
