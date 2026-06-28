# Step B pivot -- no HuMo-14B GGUF exists; is the 17B GGUF the fits-the-ceiling candidate?

Harden the DECISION + the concrete wiring. Diagnostic-only; operator-gated; no eng_humo
edit for the bakeoff.

## What Step A settled (grounded; STEP_A_RESULTS.md)
The operator-preferred HuMo-14B fp8 rides ~15.86 GB REAL card occupancy (NVML stable across
allocator A/B; torch max_allocated under-reports because --cuda-malloc loads weights outside
its stats). So 14B does NOT fit <=13.5/14.5 GB by any evict/allocator trick. final.md Step B
said "quantized HuMo-14B GGUF" as the weight-floor lever.

## The blocker (new finding)
There is NO HuMo-14B GGUF anywhere. The OTR "14B" is Kijai's fp8 repack
(`Wan2_1-HuMo-14B_fp8_e4m3fn_scaled_KJ.safetensors`); the OFFICIAL ByteDance HuMo ships only
1.7B and 17B, so every HuMo video GGUF on HF is 1.7B/17B (VeryAladeen/Wan2_1-HuMo_17B-GGUF,
QuantStack/HuMo-GGUF, calcuis/humo-gguf, Alissonerdx/Wan2.1-HuMo-GGUF). ON DISK already:
`HuMo-17b-Q3_K_M.gguf` (8.4 GB), `Wan2_1-HuMo-17B_Q5_K_M.gguf` (11.86 GB).

## OTR quality history (memory; eyeball-able, not gospel)
14B fp8 = 37.5/45, 17B fp8 = 36/45, 17B GGUF Q5_K_M = 30/45 (an early smoke). So the 17B is
near-14B quality at fp8; the GGUF-Q5 scored lower once but was never operator-eyeballed here.

## Code facts (grounded)
- eng_humo tiers: `humo`/`humo_14B_169` (14B + lightx2v distill LoRA), `humo_1.7B`/
  `humo_1.7B_169` (no LoRA, 20 steps). NO 17B tier exists. The lightx2v distill LoRA is
  14B-shaped -> will NOT apply to a 17B (shape mismatch) -> a 17B leg runs LoRA-FREE.
- `eng_humo._node_candidates` "unet" = `UNETLoader` only; a GGUF leg needs `UnetLoaderGGUF`
  ({"unet_name": <gguf>}) -- the proven pattern is `eng_wan_i2v._loader_mode`/:215-218.
  ComfyUI-GGUF is installed. The bakeoff BUILDER must emit the GGUF leg (no eng_humo edit).
- WanHuMoImageToVideo audio cross-attn must accept a GGUF-loaded 17B (verify via a 33f smoke).

## DECISION to harden (rank + justify)
Should Step B pivot to wiring + bakeoff-testing a **17B-GGUF HuMo leg** (UnetLoaderGGUF,
LoRA-free, ~20-25 steps, ModelSamplingSD3 shift 8, cfg) as the fits-the-ceiling, ~14B-class
candidate -- measure NVML fit (<=13.5 target / <=14.5 hard) + operator-eyeball the clip vs
the 14B/1.7B clips? Or is the quality/effort risk high enough to instead KEEP humo_1.7B and
harden its de-blue (the fallback if no fitting HuMo looks good enough)?

## Want from the panel
1. Ranked rec: 17B-GGUF leg (Q3_K_M 8.4GB and/or Q5_K_M 11.86GB) vs keep-1.7B-harden, with
   the one-line why.
2. The concrete bakeoff wiring for a 17B-GGUF leg in build_humo_bakeoff_workflow.py +
   run_humo_bakeoff.py (loader-agnostic meta, LoRA-free, steps/cfg/shift, the 33f smoke,
   audio cross-attn risk), reusing the Step-A meter + the eng_wan_i2v GGUF pattern.
3. Main risk + how to measure on this box (reuse run_humo_bakeoff.py).
HARD constraints: single resident <=14.5 GB; in-process always-silent; diagnostic-only (no
eng_humo/workflow/profile edit for measurement); 100% local; promotion operator-gated.
