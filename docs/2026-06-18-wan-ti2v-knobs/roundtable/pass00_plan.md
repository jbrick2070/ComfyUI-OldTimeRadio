# wan_ti2v render-knob optimization -- roundtable spec (grounded)

## Problem
`wan_ti2v` (Wan2.2 TI2V-5B) is OTR's **low-floor, system-agnostic** video engine --
explicitly designed to be accessible on **8GB VRAM, Macs (MPS), and AMD (ROCm/
DirectML)**, not just the 5080. It is the weakest video engine on quality but the
only one that fits the 8GB tier. Its render knobs were set from a build-time
`/object_info` capture, NOT a tuning pass.

GOAL (operator-scoped 2026-06-18): the **MOST SOLID LOW FLOOR** -- the best,
rock-reliable 480p recipe that works for 8GB / Mac / AMD users. We are NOT chasing
quality bells-and-whistles here: **720p, higher-step "quality" tiers, and audio-in
are explicitly OUT OF SCOPE** -- that fancier tier is the LTX audio-in lane, a
separate engine. wan_ti2v's only job is the accessible, portable, dependable floor.

## Current recipe (grounded in `nodes/_otr_video_engines/eng_wan_ti2v.py`)
- Graph: `UnetLoaderGGUF` (Wan2.2-TI2V-5B-Q5_K_M.gguf) -> `ModelSamplingSD3`(shift) ->
  `CLIPLoader`(umt5_xxl_fp8, type=wan) -> `CLIPTextEncode` x2 -> `Wan22ImageToVideoLatent`
  (Wan2.2 VAE) -> `KSampler` -> `VAEDecode`.
- Defaults (all env-overridable `OTR_WAN_TI2V_*`): **steps 30, cfg 5.0, shift 5.0,
  sampler uni_pc, scheduler simple**, render canvas **832x480**, length 25 (min 33),
  denoise 1.0.
- GPU smoke (5080, 2026-06-18): i2v 33 frames @ 832x480, engine vram ~8.2 GB, NVML
  peak 13.1 GB. Renders, but unoptimized + only validated on CUDA.

## Search-first grounding (2026-06-18, sources at bottom)
- A 5B-specific accelerator exists: **`Wan2.2-TI2V-5B-4steps`** (LightX2V / Wan2.2-
  Lightning distill family; native ComfyUI workflows). Also a ~6-step distill variant.
  4 steps vs our 30 = the biggest speed/quality lever.
- The Lightning repo supports **both 480P and 720P**, portrait or landscape.
- Official guidance: scheduler **simple**, **shift 3.0-5.0** (lower=less motion),
  samplers `euler/beta` / `sa_solver/beta` / `lcm/simple`; some report `MoEKSampler`
  beats plain KSampler for this model.

## HARD CONSTRAINTS (the panel MUST respect -- this is the accessible FLOOR)
1. **8GB VRAM floor at 480p.** The recipe must fit ~8GB (offload-tolerant) at 480p.
   720p / higher-step quality tiers are OUT OF SCOPE (that's the LTX audio-in lane).
   Optimize the 480p floor, nothing fancier.
2. **System-agnostic.** Must run on **Mac (MPS) + AMD** as well as CUDA. NO CUDA-only
   ops / custom CUDA kernels. Flag any knob that is NVIDIA-only.
   - OPEN QUESTION for the panel: is the **GGUF (`UnetLoaderGGUF`)** path viable on
     MPS/AMD, or does the floor need an fp8/fp16 safetensors path for portability?
     Is `uni_pc` / `sa_solver` / `MoEKSampler` available + correct on MPS/AMD, or
     should the floor stick to core `euler`/`lcm`?
3. **License: commercial-clean.** The LoRA must be Apache-2.0 / MIT (verify LightX2V
   license) to match wan_ti2v's Apache-2.0 base.
4. **Determinism** (seed-keyed within a render) must hold.
5. **Default-off / additive.** Any new knob (LoRA path, step count) ships behind an
   `OTR_WAN_TI2V_*` env, default preserving current behavior unless promoted.

## Candidate recipes (480p floor ONLY; for the A/B that FOLLOWS this roundtable)
- **A Baseline** -- 30-step uni_pc/simple, cfg 5, shift 5, 832x480 (current).
- **B 4-step Lightning LoRA** @ 832x480 (the lever; cfg per distill, often 1.0).
- **C 6-step distill** @ 832x480.
- **E Non-LoRA control** -- sampler/shift swap (euler/beta or sa_solver, shift 3.0)
  @ 832x480.
(720p variant intentionally DROPPED -- out of scope per the floor-only mandate.)

## Questions for the panel (ALL within the 480p / 8GB / cross-platform floor)
1. Which candidate (A/B/C/E) is the most **solid, reliable 8GB + cross-platform**
   480p DEFAULT? Rank against reliability + portability FIRST, quality second --
   this is the floor, "solid and works everywhere" beats "marginally prettier".
2. GGUF-vs-safetensors for Mac/AMD portability -- does the floor default need to
   move off `UnetLoaderGGUF`? What's the smallest portable path that still fits 8GB?
3. Lightning LoRA: correct strength, cfg, sampler/scheduler pairing for the 5B
   4-step; determinism / VAE-decode gotchas; LICENSE confirmation (must be
   Apache/MIT). Does the LoRA help or HURT reliability on non-CUDA backends?
4. Which samplers are safe cross-platform (MPS/AMD) -- is `uni_pc`/`sa_solver`/
   `MoEKSampler` portable, or should the floor stick to core `euler`/`lcm`?
5. Any reliability traps for the floor (OOM edges at 8GB, offload behavior, frame
   count / length limits, fail-closed gaps) we should harden BEFORE tuning quality?

## Sources
- https://github.com/ModelTC/Wan2.2-Lightning
- https://huggingface.co/lightx2v/Wan2.2-Lightning
- https://huggingface.co/lightx2v/Wan2.2-Distill-Loras
- https://comfyui-wiki.com/en/tutorial/advanced/video/wan2.2/wan2-2
- https://docs.comfy.org/tutorials/video/wan/wan2_2
- https://blog.comfy.org/p/comfyui-wan22-fun-inp-support
