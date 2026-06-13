# LTX 2.3 motion model selection for OTR i2v radio clips on a 5080 16GB

## Goal
Pick the best LTX-Video 2.3 model + sampling config to render **i2v** radio-console
clips with **REAL dynamic motion** (camera dolly + object motion: dial sweep, tube
pulse, grille tremble), where the operator's FLUX scene still MUST remain visible
in the output (i2v ConditionOnly anchor, not pure text-to-video).

## Hard constraints
- GPU: RTX 5080 Laptop, **16 GB VRAM**. Single resident heavy engine must peak
  **<= 14.5 GB** (host NVML). torch 2.10 + cu130, Windows, ComfyUI 0.24.1, DynamicVRAM.
- 100% local, offline-first. i2v anchor required (stills stay in the video).
- Deterministic, seed-keyed. UTF-8/SFW.

## What we run TODAY (too static -- the problem)
- Model: **ltx-video-2b-v0.9** (old 2B). Encoder: t5xxl_fp16.
- Sampler: 30-step `euler`, cfg 3.0, scheduler normal. i2v ConditionOnly strength 1.0.
- Result (measured by mean inter-frame MAD on an isolated smoke harness):
  euler MAD 0.59 (freeze); euler_cfg_pp 0.88 (pan); euler_cfg_pp@257 4.2 but WARPS;
  distilled 8-step chain on v0.9 @ strength 0.75 = 0.6-0.9 (still pan). i2v AND
  text-to-video both stay "pan" on v0.9. Conclusion: v0.9 2B is motion-limited.

## The reference that DOES move: ComfyUI-Goofer (operator's own repo)
Goofer's `GooferBatchVideo` produces the dynamic motion the operator wants:
- Model: **ltx-2.3-22b-distilled-fp8** (LTX-2.3 22B distilled).
- Encoder: **gemma_3_12B_it_fp4_mixed** (NOT t5xxl).
- LoRA: **ltx-2-19b-lora-camera-control-dolly-left** (camera-control / dolly).
- Sampler: distilled 8-step **SamplerCustomAdvanced** (KSamplerSelect euler +
  ManualSigmas + CFGGuider **cfg=1.0** + RandomNoise), sigmas
  `1.,0.99375,0.9875,0.98125,0.975,0.909375,0.725,0.421875,0.0`.
- i2v cond_strength **0.75**, 768x512, fps 35, length up to 257, spatial upscaler x2.

## On disk already (no download needed)
- `ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors` (23.5 GB) -- transformer ONLY (needs separate VAE + gemma CLIP).
- `ltx-2.3-22b-dev.safetensors` (43 GB).
- `ltx-2.3-22b-distilled-lora-384-1.1.safetensors` (7 GB).
- `gemma_3_12B_it_fp4_mixed.safetensors` (8.8 GB).
- `ltx-2-spatial-upscaler-x2-1.0.safetensors`.
- NOT on disk: the camera-control dolly LoRA.

## Questions for the panel
1. **VRAM fit:** can the 22B-distilled fp8 transformer (23.5 GB on disk, fp8) run
   i2v at e.g. 768x512x257 within a 14.5 GB live ceiling on a 16 GB 5080 -- with
   gemma (8.8 GB) also needed for encode? What block-swap / sequential-offload /
   tiled-VAE / fp8-on-the-fly is REQUIRED, and what's the realistic peak? Is a
   34 GB-class model even feasible at 16 GB, or must we offload the encoder first
   (encode -> free gemma -> load transformer)?
2. **Camera LoRA:** is the camera-control dolly LoRA the PRIMARY motion driver
   (vs the base 22B model), i.e. is downloading/using it the highest-leverage
   single change? Or does the 22B-distilled base already move well and the LoRA
   only biases direction?
3. **Best motion-per-VRAM config** for this exact use case (i2v, stills must stay,
   16 GB): which model + encoder + sampler + length + cond_strength?
4. **Better current variant:** is there a newer/leaner LTX-2.3 distilled or fp8
   variant better suited to 16 GB than the 23.5 GB transformer (e.g. a 13B-distilled
   fp8, or an LTX-2 2B successor) that keeps the motion?
5. Anything we're missing about LTX-2.3 i2v motion (e.g. STG/skip-layer guidance,
   the gemma encoder's role in motion, frame_rate's effect)?
