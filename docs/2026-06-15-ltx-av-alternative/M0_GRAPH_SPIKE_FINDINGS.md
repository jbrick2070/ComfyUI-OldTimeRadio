# M0 GRAPH SPIKE -- LTX-2.3 A2V (audio-input) lane [ltx_av]  -- FINDINGS

> Probe-or-park. NO engine code. Captures the real LTX-2.3 audio-conditioned graph on THIS 5080
> (live `/object_info` @ :8000 + the official bundled ComfyUI template) and the VRAM reality.
> Date: 2026-06-15. Branch `v2.0-alpha`. Box: RTX 5080 Laptop, 16 GB VRAM (16303 MiB), 63.4 GB RAM.

## >>> M0-GGUF VERDICT = **GO at Q3_K_M** (2026-06-15, measured on the 5080). <<<
**Q3_K_M re-probe (ltx-2.3-22b-dev-Q3_K_M.gguf, 10.03 GB on disk):** unet loaded 10538 MB on GPU
(within 13468 MB usable), 8 steps in 18 s (2.31 s/it), completed `success`, **PEAK NVML = 13688 MB <= the
14500 cap**, and NO VAE-decode unet unload (room for the VideoVAE -- the Q4_K_S thrash is gone). With the
A2V audio VAE (+~0.34 GB) ~= 14.0 GB, still under ceiling. **Q3_K_M is the production quant for ltx_av:
ceiling-compliant, no decode thrash, fast.** Q4_K_S stays on disk as a quality-step-up only if the ceiling
is relaxed / run solo. Probe ran on a headless server I booted on :8011 (alt port, torn down after) because
the operator's Desktop (:8000) was down at probe time. NEXT = M1-M4 additive build (eng_ltx_av.py).

## UPDATE 2026-06-15 (operator REVIVED via GGUF): M0-GGUF PROBE RAN -> lane VIABLE, quant-gated.

**Result (real, measured on the 5080 via a live :8000 forward):** the LTX-2.3 GGUF audio-lane stack RUNS.
- Tooling: `ComfyUI-GGUF` reads LTX2 metadata (`Found quantization metadata version 1`; Gotcha 2 clear);
  `UnetLoaderGGUF` + `LTXAVTextEncoderLoader(device=cpu)` work. Encoder = `gemma_3_12B_it_fp4_mixed` on
  CPU (`11201 MB`, 0 VRAM = the offload the spec wants) + projection from the on-disk bf16 dev ckpt.
- Probe graph (T2V base stage, 512x288x97, 8 steps): unet `ltx-2.3-22b-dev-Q4_K_S.gguf` loaded FULLY on
  GPU (`12780 MB`), sampled 8 steps in 18 s (2.30 s/it), completed `success`.
- **PEAK NVML = 15594 MB -> OVER the 14500 MB OTR ceiling.** At VAE decode ComfyUI partial-unloaded the
  unet (`Unloaded partially: 3304 MB`) to fit the VideoVAE = decode-time offload thrash.
- **Verdict: VIABLE but quant-gated.** Q4_K_S runs on 16 GB (no OOM) yet breaches <=14.5 GB and thrashes at
  decode (so "Q4 = speed" inverts on 16 GB). **Q3_K_M (~10 GB)** would fit with headroom -> no decode unload
  -> faster AND <= ceiling -> the recommended production quant. Q4_0 (12.72 GB) likely still bumps the cap.
- The A2V audio path adds only the audio VAE (~0.34 GB) over this T2V probe -> same conclusion.
- Assets fetched to `C:\ComfyUI-Models` (unet/, vae/, text_encoders/), sha+license in
  `m0_gguf_model_manifest.json`. NEXT (operator-gated): confirm quant (Q3_K_M recommended), re-probe the
  ceiling, then M1-M4 additive build. Box is the operator's ACTIVE production Desktop -- coordinate GPU.

## (superseded by the GGUF revival above) STATUS: M0 COMPLETE -> **PARK Lane B (22B A2V).** Graph captured + grounded; Lane A stands as production.

**VERDICT (operator-decided 2026-06-15, Route A / no-download worst-case reasoning):** PARK the LTX-2.3
22B audio-input lane. The graph capture is decisive without spending a heavy receipt: the A2V model is a
~23 GB fp8 FULL checkpoint + a ~8.8 GB Gemma-3-12B encoder. Those cannot be single-resident under 14500 MB
on a 16 GB card; the only way it runs is aggressive block-swap / CPU-offload, which (a) only *might* cap
peak NVML at <=14.5 GB and (b) streams weights every denoise step -> too slow for per-beat production = the
"offload thrash" PARK condition. Lightricks lists 32 GB+ as the comfortable target, consistent with this.
Spending the 23 GB fp8 download (Route B) to obtain an empirical NVML receipt for an arithmetic
near-certainty is low ROI, so no heavy forward was run. **Lane A (the golden prompt-only `ltx_video`:
boomerang + ksampler + music_open + 832x480) remains production, untouched. Nothing lost.** The grounded
graph spec below is retained for any future revival.

**ONLY FUTURE LEAD for a 16 GB fit:** a **GGUF-Q3_K_S/Q3_K_M** quant of the 22B (~9-11 GB, community/Unsloth)
-- NOT on disk, and it would need (i) the graph adapted off `CheckpointLoaderSimple` to an `UnetLoaderGGUF`
path, and (ii) verification that the community GGUF even supports the audio-conditioning + Gemma encoder
path. That is a SEPARATE, uncertain investigation -- pursue only if Lane B is explicitly revived. NVFP4 is
CUT (exceeds 16 GB). M1-M4 remain correctly un-started (gated behind an M0 GO that did not occur).

## (superseded) STATUS: GRAPH CAPTURED (M0b DONE). VRAM PROOF (M0c) GATED ON A MODEL-FETCH / HEAVY-RUN DECISION.

The A2V topology is no longer unknown -- it is grounded from two authoritative sources:
1. **Live `/object_info`** on the operator's running ComfyUI Desktop (:8000): 1608 node classes; full
   dump saved at `m0_object_info_full.json`. All LTX-2.3 A2V nodes are INSTALLED.
2. **The official bundled template** `comfyui_workflow_templates_media_video/.../video_ltx2_3_ia2v.json`
   ("Image + Audio -> Video", LTX-2.3). Raw copy saved at `m0_template_ia2v.json`; the real pipeline is a
   53-node / 96-link **subgraph** "Video Generation (LTX-2.3)".

## THE GROUNDED A2V GRAPH (authoritative -- M2 wires WHAT THIS CAPTURED)

**Model stack (from the template's actual widget values):**
- Transformer: **`CheckpointLoaderSimple(ltx-2.3-22b-dev-fp8.safetensors)`** -- a 22B-class FULL fp8
  checkpoint (bundles the video VAE + the audio VAE). ~23 GB. **NOT a UNet/GGUF/block-swap loader.**
- Text encoder: **`LTXAVTextEncoderLoader(text_encoder=gemma_3_12B_it_fp4_mixed.safetensors,
  ckpt_name=ltx-2.3-22b-dev-fp8.safetensors, device=default)` -> CLIP**. (Gemma-3-12B fp4 ~8.8 GB.)
  Optional prompt-enhance branch: `TextGenerateLTX2Prompt` + a Gemma abliterated LoRA + `ComfySwitchNode`
  (PrimitiveBoolean "Enable Prompt Enhance" = True) -- DISABLE for the lean lane.
- Distilled motion LoRA: `LoraLoaderModelOnly(ltx_2.3_22b_distilled_1.1_lora_dynamic..., 0.5)`.
- Audio VAE: **`LTXVAudioVAELoader(ltx-2.3-22b-dev-fp8.safetensors)`** (reads the audio VAE out of the same
  checkpoint -- there is NO standalone LTX-2.3 audio-VAE file).

**Audio-conditioned path (the A2V core):**
- `LoadAudio` -> `TrimAudioDuration` -> **`LTXVAudioVAEEncode(audio, audio_vae)`** -> audio latent.
- `LoadImage` -> resize -> **`LTXVImgToVideoInplace`** (i2v image conditioning into the video latent).
- `EmptyLTXVLatentVideo(W,H,frames,1)` provides the video latent shell.
- **`LTXVConcatAVLatent(video_latent, audio_latent)` -> joint AV latent.**
- `CLIPTextEncode` x2 (pos/neg) -> `LTXVConditioning(frame_rate=24)`.
- Sampling: `SamplerCustomAdvanced(noise=RandomNoise, guider=CFGGuider(1.0), sampler=KSamplerSelect(euler),
  sigmas=ManualSigmas(9-step base), latent=joint AV)`. Two-stage in the template: base ->
  `LTXVLatentUpsampler` (+ `LatentUpscaleModelLoader(ltx-2.3-spatial-upscaler-x2-1.1)`) -> second
  `SamplerCustomAdvanced` (4-step refine ManualSigmas). **For the lane probe: BASE STAGE ONLY, no upscaler.**

**TERMINAL = video-only decode (the V-1 frozen-audio anchor -- CONFIRMED GROUNDED):**
- **`LTXVSeparateAVLatent(av_latent) -> (video_latent, audio_latent)`** splits the joint latent.
- **video_latent -> `VAEDecodeTiled(vae, tile=768, overlap=64, ...)` -> IMAGE** (the template uses 768 tiles;
  the low-VRAM lever is `LTXVSpatioTemporalTiledVAEDecode` / `LTXVTiledVAEDecode` with `working_device`).
- The audio branch `audio_latent -> LTXVAudioVAEDecode` is **DROPPED** in the lane: never decode LTX audio;
  only `OTR_MasterAudioMux` emits audio (V-1 byte-identical). This is exactly the terminal the pass03 plan
  needed; my earlier `LTXVSeparateAVLatent` guess was CORRECT and is now grounded (it exists, twice: #309/#311).

**Low-VRAM levers available (installed, confirmed in /object_info):** `LowVRAMAudioVAELoader`,
`LTXQ8Patch` (fp8 attention + quant presets), `LTX2BlockLoraSelect` (48 blocks),
`LTX2MemoryEfficientSageAttentionPatch` (needs triton -- the historical Blackwell SageAttention gotcha,
treat as OPTIONAL/off by default), `LTXVSpatioTemporalTiledVAEDecode` (CPU/tiled decode floor).

## VRAM REALITY (the crux of probe-or-park)

- This is a **22B-class** model used as a single ~23 GB fp8 checkpoint + a ~8.8 GB Gemma-3-12B encoder.
  Co-resident that is ~32 GB -> impossible on 16 GB. Feasibility depends ENTIRELY on ComfyUI's sequential
  CPU-offload (lowvram): Gemma encode (then evict) -> reclaim -> transformer streamed for the base sample ->
  tiled video VAE decode. This matches the pass03 phasing (encoder -> reclaim -> transformer).
- **No fp8/GGUF Q3 of the 22B is on disk.** What IS on disk:
  - `ltx-2.3-22b-dev.safetensors` -- 42.98 GB **bf16 full** checkpoint (bundles VAE+audioVAE).
  - `ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors` -- 23.49 GB (transformer ONLY,
    no VAE -> needs a standalone LTX-2.3 VAE that is NOT on disk).
  - `ltx-2.3-22b-distilled-lora-384-1.1.safetensors` (7.08 GB), `ltx-2-spatial-upscaler-x2-1.0` (0.93 GB).
  - The template's required `ltx-2.3-22b-dev-fp8.safetensors` (~23 GB) is **ABSENT**.
- RAM (63.4 GB, 32.7 free) can hold either the 23 GB fp8 OR the 42.98 GB bf16 for lowvram streaming.
- **NVFP4 CUT** (would exceed 16 GB) -- consistent with pass03.

## THE GO/PARK FORK (M0c -- the empirical <=14500 MB proof still owes a real forward)

The graph is proven present and runnable in principle; the gate requires a measured peak <= 14500 MB. Two
ways to get the measurement, both heavy GPU work on the operator's (idle-but-live) Desktop box:

- **Route A (no download, WORST-CASE bound):** run the base stage with the on-disk **bf16 dev (42.98 GB)**
  at the decode floor (512x288, ~97 frames, Gemma offloaded, no upscaler, prompt-enhance off). bf16 is
  strictly heavier than fp8, so a pass <=14.5 GB proves GO a fortiori. Slow (heavy CPU<->GPU streaming of a
  42 GB model), but uses only what's here. Run detached + poll NVML (>60s MCP ceiling).
- **Route B (clean, production artifact):** fetch `ltx-2.3-22b-dev-fp8.safetensors` (~23 GB, HF
  Lightricks/LTX-2.3, open weights, $0) -- which the lane MUST have to exist at all -- then probe at the
  floor with the real production checkpoint (faster, representative).

**Either way the lane cannot be BUILT/run without the fp8 dev checkpoint download (Route B's fetch).** So a
GO verdict implies that ~23 GB fetch as a prerequisite regardless.

If the probe OOMs / thrashes / cannot prove <= 14500 MB -> **PARK Lane B, Lane A (golden prompt-only
ltx_video) stands as production. Nothing lost.**

## WHAT IS DONE vs OWED
- DONE: live node-class capture, official template capture, the full A2V graph spec, the terminal
  video-only decode confirmation, the on-disk asset inventory, the VRAM analysis, the artifact-gap finding.
- OWED (M0c): the empirical peak-NVML proof (Route A or B) -> GO or PARK.
- NOT STARTED (correctly gated behind M0 GO): M1 skeleton, M2 frozen-audio V-1, M3 wiring, M4 graduation.
