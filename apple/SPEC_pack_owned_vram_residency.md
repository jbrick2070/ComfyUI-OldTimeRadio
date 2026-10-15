# SPEC: pack-owned VRAM residency for the video lanes

**Status:** design draft. No code. Not approved.
**Date:** 2026-10-14
**Scope:** the video lanes on the 8 GB and 16 GB profiles. The writer LLM, Bark and the image engines appear only at their handoffs.
**Gate:** this is a design choice with more than one defensible answer, so CLAUDE.md (2026-08-17 amendment) requires the full four-round kibitz arc before any code. This document is the input to that arc.
**Decisions for the operator:** section 9.

## 1. Problem

Nothing in the video path owns the budget. Five mechanisms decide what stays on the GPU, and none of them computes a budget from the request.

| Mechanism | Where | Fires on | Budget? |
|---|---|---|---|
| Post-prompt release (ComfyUI `free_memory` queue flag) | `nodes/_otr_vram_levers.py:294` | after every prompt | no |
| Pipeline residue release (writer LLM, Bark, FLUX detach) | `nodes/_otr_vram_levers.py:121` | before HuMo | no |
| Dropped-output eviction | `nodes/_otr_video_engines/wrapper_bridge.py:311` | a node output is dropped | no |
| Reclaim all loaded models | `nodes/_otr_video_engines/wrapper_bridge.py:405` | after a render | no |
| Soft cache flush | `nodes/_otr_video_engines/wrapper_bridge.py:295` | after a node | no |

Four measured facts drive this spec.

1. **Streaming can fail without an OOM.** The `wrapper_bridge.py:311` docstring records a 4060 clean-room run on 2026-09-02. A classic diffusion model loaded with 0.00 MB usable and streamed from host RAM every step: about 2 minutes per step, about 42 minutes per still. Nothing crashed, so no existing check caught it.
2. **The one explicit budget is a boot flag, and it decided pass or fail.** The H3 boot contract sets `reserve_vram_gb = 12.0`, which ComfyUI receives as `--reserve-vram 12`. Every H3 leg that cleared 14.5 GiB was booted that way. The one leg without it peaked at 15.39 GiB and failed, on a smaller canvas and a shorter length (lane 19 receipt, section A: `apple/evidence/lane_receipts/lane19-h3_low_video.md`). The reserve forces the 21 GB DiT to stream instead of attempting residency.
3. **8 GB memory is not attributed.** The 2026-09-28 H3 attempt on the 8 GB 4060 died at beat 4 while loading the text encoder. CUDA reported 0 bytes free while PyTorch held 213 MiB (`apple/HANDOFF_LOG.md`, entry "H3 on the 8 GB 4060"). The gap between CUDA-used and PyTorch-reserved was never explained.
4. **Host RAM binds too.** TEST_WAVE row B4 (LTX 8 GB lane) peaked at 7,300 MiB with 0.42 GB of free RAM at the low point, before the per-beat T5 reload fix (PBUG-20260926-01). The 4060 free-RAM figure for that lane is still owed.

Conclusion: the problem is not that ComfyUI owns VRAM. Nothing owns the budget, and eviction fires on "after use" and "after prompt" triggers rather than on need.

## 2. Goals and non-goals

**Goals**

- **G1.** One component computes, per request, the VRAM and host-RAM budget and decides per stage: resident, streamed or refused.
- **G2.** Streaming is a recorded decision, never a silent fallback. A streamed stage has a measured step time and an accept or refuse record.
- **G3.** OOM is the only killer. An OOM in real work gets one lighter retry, then fails loud. A failed probe degrades to the safe path and never fails a render.
- **G4.** CPU, Mac/MPS and cloud lanes run unchanged. The planner stays off when CUDA is absent.
- **G5.** ComfyUI stays the host. App mode, the Registry install path and the canonical graph do not change.
- **G6.** Workarounds are retired one lane at a time, as the planner covers each lane. The post-prompt flag and the Bark special case are the first candidates.

**Non-goals**

- Replacing ComfyUI's runtime or the `comfy` library. FreeVideo keeps `comfy` for its text encoder, VAE and checkpoints. We would too.
- Changing model quality, prompts, graph semantics or the variant generator.
- Copying FreeVideo's code (section 8).
- The front end (section 10).

## 3. Current state

### 3.1 What exists and what it owns

| Piece | Owns | Budget? | Evidence |
|---|---|---|---|
| `nodes/_otr_shared/gpu_residency.py` | Cross-process GPU lease (lockdir and owner file), NVML probes, `wait_until_below_mb`, `wait_until_stable` | Timeout only | Callers: `otr_video_render_batch.py`, `otr_image_gen_dispatcher.py`, `motion_common.py`, `eng_ltx25.py` |
| `nodes/_otr_model_loader.py` (LLM) | Placement; OOM classification in `_is_memory_placement_failure` | No cap, by operator directive (`_plan_max_memory` returns None). CPU overflow only as a retry | Lines 479, 623 |
| `_assert_policy_admits_vram` | Admission through one estimator, `check_vram_fit` | Advisory: FAIL is a recommendation, not a refusal | Line 2005 |
| Reserve (`--reserve-vram`) | Keeps VRAM away from model loading | Static, set per boot contract | Lane 19 receipt, section A |

The LLM side already places models and classifies OOM. The video side has neither.

### 3.2 Engines and measured status

| Lane | Engine | 16 GB (RTX 5080 Laptop) | 8 GB (RTX 4060 Laptop) | Third-party pack |
|---|---|---|---|---|
| `h3_low_video` | `minimax_h3_video` | PROVEN (`apple/MACHINE_MATRIX.md`). Peak 9.15 GiB at 1344x768 with reserve 12 | FAILED 2026-09-28 at beat 4, during text-encoder load | None |
| `h3_low_audio_in` | `minimax_h3_audio_in` | Lane 20 receipt exists | Not measured | None |
| HuMo | `humo` | PROVEN, 32 published episodes | "run out of memory on 8 GB" (`apple/VIDEO_MODELS.md:214`) | None |
| LTX 2b 0.9.8 | `ltx_8gb` | PROVEN on a 20 GB Ampere card, 1 act | TEST_WAVE B4: peak 7,300 MiB, free RAM low 0.42 GB | None |
| LTX 2.5 | `ltx25` | COMPONENTS MEASURED (MACHINE_MATRIX) | 8 GB lanes exist. Status not in the receipts read | None |
| AnimateDiff haunted | `animatediff15_v3_haunted_video` | Rented 24 GB card: 16 GB workflow peaked at 15,990 MB | PROVEN on the 4060 (MACHINE_MATRIX) | ComfyUI-AnimateDiff-Evolved |

The mesh stage and ghost-signal engines were not measured in the receipts read. They stay out of Phase 1 until measured.

### 3.3 Regression run, 2026-10-14 (`qa-20261014-8gb-regress`)

| Leg | Result | Wall time | Sampled peak (MiB) | Note |
|---|---|---|---|---|
| low (viz_camera) | PASS, 8 beats | 879 s | 7,605 | |
| still (still_motion) | PASS, 8 beats | 703 s | 7,403 | |
| video | FAIL, exit 1 | 8 s | 2,096 | Failed at queue time, in a visual-asset metadata fetch (HTTP error). Never reached the GPU |

The sampler (`sample_vram.ps1`) writes every 15 s, too coarse to catch a peak. The per-leg `vram_peak_mb` field was empty.

The two passing legs peaked at 7.4 to 7.6 GiB on an 8 GB card, leaving well under 1 GiB of headroom. That margin is the real constraint, not the average.

## 4. Proposed design

### 4.1 Ownership

- **The pack owns:** the per-request budget (VRAM and host), the resident, streamed or refused decision for each video stage, the handoff between stages, the OOM retry and the memory attribution.
- **ComfyUI and `comfy` keep:** model implementations, samplers, VAEs, text encoders, checkpoint loading and the App-mode UI.

### 4.2 The planner

One component, `ResidencyPlanner` (name provisional), called from the graph executor. In the wrapper-engine path that executor is `run_graph` (`wrapper_bridge.py:477`). Which lanes go through it must be confirmed before implementation.

- **Before a node runs:** compare the node's model working set with the budget. Decide resident, streamed or refused.
- **After a node's last use:** evict through the same registry-safe call that `_evict_dropped_models` uses today. The planner replaces the trigger. It does not replace the eviction mechanism.
- **Between stages:** take the GPU lease, then wait until VRAM falls below the budget, using the existing `wait_until_below_mb` and `wait_until_stable`.

### 4.3 Budgets

- **VRAM budget** = total VRAM minus reserve minus memory held by the desktop or driver. The reserve becomes a per-profile field in place of the boot flag. The H3 value of 12 GiB is the first calibration point. On an 8 GB card a 12 GiB reserve is impossible, so the 8 GB H3 plan is "stream the DiT". That needs its own bound (section 4.4).
- **Host budget** = available RAM minus reserve. The free-RAM low point becomes a gate, not a log line.
- **Working set** per engine, resolution and frame count comes from recorded peaks (receipts), not from name-based estimates. The existing estimator (`check_vram_fit`) serves as a prior and is checked against receipts.
- **Reference numbers, not adopted:** FreeVideo reserves 2.5% of free VRAM, clamped to 0.5 to 1 GiB (0.25 GiB on Windows), and 5% of available host RAM, clamped to 1 to 4 GiB. These are starting points for calibration.

### 4.4 Decision rule

Per stage, in order:

1. Working set fits the VRAM budget: **resident**.
2. It does not fit, the engine supports streaming, and the forecast step time is within the bound: **streamed**. The forecast and the measured step time are recorded.
3. Otherwise: **refuse before loading**, naming the shortfall.

The bound is an explicit per-profile number. The operator rule that OOM is the only acceptable killer, and the rule against crawling, both documented in `_otr_model_loader.py` (lines 479 to 505), mean a stream that makes renders useless must be refused, not run. The bound is open decision O2.

### 4.5 OOM and retry

- Classify with the existing `_is_memory_placement_failure` (`_otr_model_loader.py:623`). The comfy OOM check (`is_oom`, referenced in `apple/PROD_BUG_LOG.md` near line 13969) is a second input.
- Retry once, in a fresh process, with the next-lighter placement: smaller chunks, more blocks streamed. FreeVideo allows two retries. One is proposed because a low leg runs about 15 minutes, so a retry costs real time (O3).
- A retry can change floating-point reduction order. FreeVideo's planner documents the same effect. The episode report records the placement, and a replay claim covers only runs with the same recorded placement (O4).

### 4.6 Attribution (first deliverable)

Before any policy change, each stage logs at start and end: `nvidia-smi` used memory, `torch.cuda.memory_reserved`, `torch.cuda.memory_allocated`, and the list of comfy-loaded models. Sampling runs at 1 Hz, or the peak is tracked in process. This is the only way to explain the 2026-09-28 gap (CUDA 0 free, PyTorch 213 MiB).

### 4.7 Retiring the workarounds

`release_models_after_this_prompt`, `_soft_free`, `reclaim_idle_models` and the Bark and LLM special cases in `free_otr_pipeline_residue` are retired one lane at a time. Each retirement needs a receipt. Until then they stay, behind a profile flag (`residency = "legacy" | "planner"`). The legacy path is a rollback switch for the proof period and is removed after it. It is not kept for old saved data.

### 4.8 CPU, MPS and cloud

The planner returns `legacy` when CUDA is absent, and for cloud and remote lanes. The Mac lanes do not change.

### 4.9 Phase 2 (conditional)

Pack-owned block streaming for the H3 DiT: blocks in pinned host memory with prefetch slots, as FreeVideo does. This starts only if Phase 1 leaves the 8 GB H3 target unmet, and only after the attribution data exists. It is the largest and riskiest part, so it stays parked (O5).

## 5. Scope

- **Phase 1 lanes:** `h3_low_video`, `h3_low_audio_in`, `ltx_8gb`, the 8 GB LTX 2.5 lanes, `humo` (16 GB first), and the AnimateDiff haunted lane. The haunted lane is proven on the 4060 today and must not regress.
- **Handoffs only:** the writer LLM, Bark, and the image engines (FLUX, z-image, Lumina). They hold VRAM across the video stage, so the handoff is in scope. Their internals are not.
- **Out:** the mesh stage and ghost signal, until measured.

## 6. Profiles and measured peaks

| Profile | Machine | Measured peak | Reserve | Status |
|---|---|---|---|---|
| `otr_8gb_low`, `otr_8gb_still` | RTX 4060 8 GB | 7,605 and 7,403 MiB (sampled every 15 s) | TBD | PASS, 2026-10-14 |
| `otr_8gb_video` (LTX 8 GB) | RTX 4060 8 GB, per B4 | 7,300 MiB | TBD | B4 passed. 4060 RAM low owed |
| H3 at 1344x768, reserve 12 | RTX 5080 16 GB | 9.15 GiB | 12 GiB | PROVEN |
| H3 I2V, no reserve | RTX 5080 16 GB | 15.39 GiB | None | FAILED |
| Haunted AnimateDiff, 16 GB workflow | Rented 24 GB card | 15,990 MB | None | PASS |

"TBD" means the reserve is calibrated in P0 and P1.

## 7. Proof plan

- **P0, attribution.** Add the four-number log and a 1 Hz sampler to one 8 GB leg (`otr_8gb_low` on the 4060). Gate: the log explains the 2026-09-28 gap during the text-encoder load.
- **P1, 8 GB regression, no policy change.** Fix the visual-asset metadata failure first. It is a queue-time gate that fetches over the network, so it may share a cause with the open "4060 connectivity issue" thread. Then re-run low, still and video. Gate: 3 of 3 pass, with peaks recorded at 1 Hz.
- **P2, H3 on 8 GB.** Run `h3_low_video` on the 4060 in streamed mode. Gate: the forecast step time is within the O2 bound, and the stage is either accepted with its measured time or refused with a named shortfall.
- **P3, 16 GB regression.** `humo`, `ltx25`, and H3 at 1344x768 on the 5080 with the planner. Gate: wall time within 10% of the current receipts, and peak under budget.
- **P4, fault injection.** Set the budget artificially low. The run must refuse or retry. It must never stream silently.
- **P5, determinism.** Repeat one episode with and without a placement change. Record both hashes. Any difference must be explained by a recorded placement change.

Gates across the plan: zero unexplained OOMs; peaks under budget; no silent streaming; wall time within 10% on the 5080; every determinism difference explained.

Both machines take part, per the multi-box proof requirement in `apple/GO_FORWARD_PLAN.md` (line 912).

## 8. Reuse and licensing

- FreeVideo's code is Apache-2.0. This pack is MIT (`LICENSE`, Copyright 2026 Jeffrey Brick). Re-deriving the design from our own measurements avoids copying code, and with it the notice question. **Default: re-derive.**
- FreeVideo's model weights fall under the MiniMax H3 Community License, which has territorial and acceptable-use restrictions. Our H3 lanes already use MiniMax H3 weights, so this is not a new question.

## 9. Open decisions (operator)

- **O1. Scope.** The video stage only (recommended), or the whole run now?
- **O2. Crawling bound.** What is the most a streamed stage may cost before it is refused? For example, a multiple of the resident step time. The 42-minute still is the failure to prevent.
- **O3. Retries.** One lighter retry (recommended), or two, as FreeVideo does?
- **O4. Determinism after a retry.** Accept a recorded, placement-dependent output, or never change placement mid-episode?
- **O5. H3 on 8 GB.** Build Phase 2 block streaming for the H3 DiT, or keep H3 as a 16 GB lane and move 8 GB work to AnimateDiff and LTX?
- **O6. Reserves.** Calibrate from P0 and P1 first (recommended), or start from FreeVideo's numbers?
- **O7. Legacy path lifetime.** How long the rollback flag stays after P3 passes.

## 10. Separate track: the front end

The operator finds FreeVideo's workspace much nicer than the ComfyUI App-mode form. FreeVideo's look comes from one scoped CSS token file (`web/theme.css`) inside a ComfyUI web extension, plus a Python mirror of the same palette for its native launcher (`freevideo_engine/branding.py`). This pack already ships a `js/` folder of web extensions, six files. A front end is a separate design with its own kibitz arc. It does not block the VRAM work and is not decided here.

## 11. Process

- Per CLAUDE.md (2026-08-17 amendment), this is a design choice with several defensible answers, so the full four-round kibitz (`kibitz-plugin:kibitz`) runs before any code.
- No code is written under this spec until that arc has run and its findings are folded in.
- Commits name files explicitly.

## Appendix: evidence index

- `nodes/_otr_vram_levers.py:121, 294`
- `nodes/_otr_video_engines/wrapper_bridge.py:295, 311, 405, 477`
- `nodes/_otr_shared/gpu_residency.py` (lease and probes)
- `nodes/_otr_model_loader.py:479, 623, 2005`
- `apple/evidence/lane_receipts/lane19-h3_low_video.md`, section A
- `apple/HANDOFF_LOG.md`, entry "H3 on the 8 GB 4060" (2026-09-28)
- `apple/MACHINE_MATRIX.md`, rows for the 8 GB and 16 GB machine classes
- `apple/TEST_WAVE.md`, row B4
- `apple/VIDEO_MODELS.md:214`
- `apple/GO_FORWARD_PLAN.md:912`
- `D:\otr-4060-testing\qa-20261014-8gb-regress\` (outside the repo)
- `scripts/otr_canonical_api_run.py`, `tests/test_canonical_headless_api.py`
- FreeVideo (external): `docs/execution-planning.md`, `web/theme.css`, `freevideo_engine/branding.py`
