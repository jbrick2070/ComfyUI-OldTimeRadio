<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

MUST-FIX

1. [Milestones / M0 PROBE] M0 is probing the wrong GGUF default: “L2 GGUF Q4_K_M” is already dead for full residency. Arithmetic: Q4_K_M file = 16.5 GiB; ceiling = 14.5 GB/NVML ceiling 14500 MB, i.e. about 14.16 GiB if treated as MiB. 16.5 GiB exceeds the ceiling by at least 2.0 GiB before activations, VAE, text_projection, LoRA, allocator overhead. Q4_K_S = 15.6 GiB is also over. Fix: M0 table must list GGUF rows Q2_K 11.6, Q3_K_S 13.0, Q3_K_M 13.7, Q4_K_S 15.6, Q4_K_M 16.5 GiB, and mark Q4 full-resident FAIL unless ComfyUI-GGUF per-layer offload measures NVML <=14500 MB. Realistic full-resident candidates are Q3_K_S/Q3_K_M at most, and even those must pass total NVML, not file-size arithmetic.

2. [Milestones / M0 PROBE + Open questions / pass06 HARDWARE] Add the required M0 decision table, not prose. Minimum columns: lane id; artifact/quant; file size; auxiliary artifacts loaded in phase; encoder placement CPU/GPU; offload/block-swap setting; NVML idle/pre-load/peak/sustained/post MB via machine-wide NVML; wall-time for 1472x832 x ~6s; frame count; quality eyeball vs current 2B baseline; PASS/FAIL; notes. PASS criteria: NVML peak and sustained <=14500 MB; wall-time <=10 min/clip PASS, 10-15 min WARN, >15 min/clip FAIL/operationally unusable; quality >= current 2B baseline. Use `gpu_residency.probe_used_mb()` / `nvml_available()` semantics from grounding, not ComfyUI free-memory.

3. [ARCHITECTURE / AS-3 lease + M2 GRAPH + LANE] Encoder/transformer co-residency is not specified and will not fit. Arithmetic: gemma_3_12B_it_fp8_scaled = 13.2 GB decimal ~=12.3 GiB. With Q3_K_S 13.0 GiB, total ~=25.3 GiB; with Q3_K_M 13.7 GiB, ~=26.0 GiB; with fp8 transformer 23.5 GiB, ~=35.8 GiB. All exceed 14.5. Fix: v1 phase discipline must be explicit: text encode first, preferably CPU/RAM-offloaded for this lane; then call `wrapper_bridge.reclaim_idle_models(reason="ltx_av text-encode phase")`; then load/run transformer; then post-render reclaim. Do not rely on a single monolithic graph unless it is proven to unload the encoder before transformer load. Grounding: `wrapper_bridge.reclaim_idle_models()` is the BUG-291 mechanism; `_soft_free()` is not enough because it does not force model eviction.

4. [ARCHITECTURE / AS-3 lease + motion_common.py grounding] Say exactly what the AS-3 lease wraps. Existing `MotionEngineBase.prepare()` acquires `gpu_residency.acquire()` before load and `teardown()` releases after detach/unload/wait; that is whole-render bracketing, not per-phase. Fix: LTX-AV must mirror this: acquire once before any GPU-heavy phase, keep through text-encode/transformer/render/canonicalize until GPU state is reclaimed, then release and wait below ceiling. If text encoding is CPU-only, it may still run under the whole-render lease in v1 for correctness; do not release between encoder and transformer unless there is a proven no-resident-GPU gap and a re-acquire before loading transformer.

5. [Milestones / M0 PROBE + TESTING / M0 sheet] System RAM is UNKNOWN but required for block-swap/offload. Fix: M0_RESULTS.md must record physical RAM, pagefile status, peak committed bytes, and peak working set during each lane. Failure mode to name: RAM shortfall causes paging, not necessarily immediate OOM, producing wall-time blowup. The wall-time gate above must catch this. For L1 fp8 block-swap/streaming, the 23.5 GiB transformer file plus encoder/offload buffers makes this mandatory.

6. [Milestones / M0 PROBE] Episode budget is missing. Fix: record both per-clip and per-episode budget. Proposed gate: at 1472x832 x ~6s, PASS <=10 min/clip, WARN 10-15, FAIL >15. For a 30-word episode with ~2 talk beats + 1 music open = 3 generated video clips, PASS episode video time <=30 min. Current known cost is one 2B open ~=6 min, so the lane’s acceptable delta is roughly +24 min max at PASS; >45 min episode video time is operationally unusable even if renders succeed.

7. [Milestones / M0 PROBE] L3 NVFP4 should not be a mandatory M0 lane. Arithmetic: NVFP4 file = 21.7 GB/GiB-class artifact, above the 14.5 ceiling before auxiliaries; it is DEV-only, not distilled, so more steps/wall-time; and there is an open Comfy loading-failure report (#11864). Fix: move L3 to optional/stretch row after L1/L2 measurement, or cut from M0 entirely to save probe time. Keep cu130/Blackwell note only; no new pip into cu130.

8. [download_ltx_2_3.ps1] The script’s disk estimate is stale/wrong against judge-verified size. It says “~22 GB free” and “expect ~22 GB”; ground truth fp8 file is 23.5 GiB. Fix: update to at least 24 GiB for cache+link case, and warn that manual copy fallback can require roughly double. Also M0 disk inventory must include non-transformer artifacts: 13.2 GB text encoder, 2.3 GB text_projection, 1.45 GB video VAE, 365 MB audio VAE, 2.7 GB dynamic LoRA if used.

SHOULD-CONSIDER

1. [M2 GRAPH + LANE] Treat Q3_K_M as “maybe,” not automatically safe. Q3_K_M = 13.7 GiB leaves only ~0.8 GB vs a 14.5 GB ceiling, or ~0.46 GiB vs the code’s `VRAM_CEILING_MB = 14500` MiB-equivalent. That margin is likely consumed by VAE/projection/activations unless they are phased/offloaded. M0 must judge total NVML, not transformer file size.

2. [I/O CONTRACTS / two-stage] No hardware reason to revisit pass02 base-only v1. Base+latent-upscale would roughly double render cost and increase residency/offload pressure. Keep v1 base-only at 1472x832.

3. [WIRING / FLUX portraits + gpu_residency.py grounding] Sequential image batch then video batch means no FLUX/LTX co-residency requirement only if the existing pipeline actually releases FLUX before video. [ASSUMPTION] Fix is just an M0 verification row: after FLUX portrait phase, confirm AS-3 lease released and `wait_until_below_mb(14500)` passes before LTX begins. Grounding confirms AS-3 is cross-process and machine-wide; it does not prove pipeline ordering.

4. [ARCHITECTURE / assert_usable] Node/weight gates should report the selected lane and phase in MISSING_MODEL messages: e.g. missing gemma encoder vs transformer vs text_projection vs VAE. Otherwise M0 failures will be ambiguous.

5. [Milestones / M0 PROBE] Include “NVML peak sampler callback” where possible, not only pre/post. Grounding `motion_common.assert_vram_within_ceiling()` exists for mid-render breach detection; peak can occur during sampling.

OPEN-QUESTIONS

1. [WIRING / FLUX CO-RESIDENCY] Verify actual production ordering: are FLUX portraits fully rendered and torn down before any video engine prepares? Grounding does not show render-driver ordering.

2. [M2 GRAPH + LANE] Which exact ComfyUI LTX-AV node graph supports split encode -> reclaim -> transformer load? Verify node IO shapes and whether text embeddings can be materialized without retaining the gemma model.

3. [M0 PROBE] Does ComfyUI-GGUF support per-layer CPU offload for these QuantStack LTX-2.3 files on this install, and does it keep NVML <=14500 MB for Q4_K_S/Q4_K_M? If not measured, Q4 is fail.

4. [M0 PROBE] What is system RAM and pagefile configuration on the box? This is a hard prerequisite for judging L1 block-swap/weight-streaming.

5. [M0 PROBE] Which text encoder is v1 default: fp8 gemma on CPU/RAM offload, GGUF Q3 encoder variant, or another installed artifact? The plan must choose one for the measured lane table.