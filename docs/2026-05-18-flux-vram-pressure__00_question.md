# Question -- 2026-05-18

# BUG-LOCAL-231 -- FLUX sampler still ~10x slow after the BUG-LOCAL-230 fp8 fix

## Platform constraints (hard)

- **GPU:** RTX 5080 Laptop, 16 GB VRAM (16303 MB usable), Blackwell sm_120, single GPU. No cloud.
- **Stack:** Windows 11, Python 3.12, torch 2.10.0, CUDA 13.0, SDPA + SageAttention (SageAttention available via KJNodes per-workflow but DISABLED in this workflow; widget value `"disabled"`). Flash Attention 2/3 NOT available on Blackwell sm_120.
- **VRAM ceiling:** 14.5 GiB peak (CLAUDE.md Prime Directive 2). Never use `force_vram_offload()` between LLM phases.
- **Audio is king:** byte-identical output across runs. Do NOT touch any audio-path code as a fix for this.
- **No low-level VRAM optimization work** (no weight streaming, no Flash Attention chasing).
- **Smallest change, largest payoff.** Prefer fixing one root cause cleanly over multiple coupled changes.

## The fix that landed and worked (BUG-LOCAL-230)

`--force-fp16` was being silently passed to ComfyUI across 4 launcher sites. It upcast `flux1-dev-fp8.safetensors` (a natively-fp8 Comfy-Org checkpoint, ~11 GiB on disk) to fp16 (~22 GiB resident), forcing the dynamic offloader to thrash weights to system RAM at 564.99 s/sampler-step. Removing the flag fixed the dtype upcast cleanly.

**Verification telemetry (2026-05-18 21:10 smoke run, `logs/comfy_session_iter_001.log`):**

- L574: `model weight dtype torch.float8_e4m3fn, manual cast: torch.bfloat16` (pre-fix: `torch.float16, manual cast: None`)
- L584: `[DeferredCheckpointLoader] load complete: VRAM allocated=2.13 -> 13.21 GiB (delta=11.08); ckpt=flux1-dev-fp8.safetensors` (pre-fix delta was 22.17 GiB)
- L585: `[FluxBranchGate] fire: VRAM allocated=13.21 GiB`

The architectural axis is PROVEN. The fp8 checkpoint loads correctly at native fp8 with bf16 compute cast. Delta within 0.08 GiB of the predicted ~11 GiB.

## The residual defect (BUG-LOCAL-231) -- what we need to fix

With the checkpoint loaded correctly at 13.21 GiB, the FLUX sampler STILL runs slow:

- **L610:** `5%|1/20 [02:34<48:46, 154.02s/it]` -- sampler step 1 took **154 seconds**. Target is ~10-15 s/step.
- This is 3.6x faster than the pre-fix 564.99 s/it (so the BUG-230 fp8 fix delivered most of the win) but ~10x slower than the architectural-fix-only target.

**LibreHardwareMonitor during sampler step 1 (http://localhost:8085/data.json):**

- GPU Memory Used: **15911 MB** / 16303 MB (97.6%)
- GPU Memory Free: 391 MB
- D3D Shared Memory Used: **1098 MB** (offloader paging to system RAM)
- ComfyUI process RSS: 9119 MB

**Pre-fix comparison:** 10445 MB D3D Shared paging -> 1098 MB. The fp8 fix cut offloader spill by ~10x but did not eliminate it. ~756 MB over the 14.5 GiB CLAUDE.md ceiling, and ~1 GiB still spilling to D3D Shared during sampler activations.

## Pipeline shape (relevant context)

The OTR workflow runs LLM (writer) phase first, then audio phase (BatchBark + Kokoro + EpisodeAssembler), THEN FLUX render via `OTR_DeferredCheckpointLoader` (gated on `EpisodeAssembler.audio_done` signal). The Deferred loader is supposed to fire COLD -- meaning the upstream LLM (`google/gemma-4-E4B-it`, default for both writer slots post-Sprint C C3 baseline shift 2026-05-15) is supposed to be unloaded before FLUX loads. The audio side also loads Bark (~1.5 GiB) and Kokoro and MusicGen.

The smoke log shows:

- L539: `[EpisodeAssembler] emit audio_done signal: audio_done:length_sec=171.97;sample_rate=48000;length_samples=8254742;segments=3`
- L573: `[DeferredCheckpointLoader] fire: VRAM allocated=2.13 GiB; gate_signal len=80; ckpt=flux1-dev-fp8.safetensors`
- L584: `[DeferredCheckpointLoader] load complete: VRAM allocated=2.13 -> 13.21 GiB`
- L585: `[FluxBranchGate] fire: VRAM allocated=13.21 GiB`
- L592: `[BatchFluxRender] pinned MODEL via load_models_gpu`
- L598: `[BatchFluxRender] skip_env_stills=True -- bypassing per-shot env-still FLUX pass; rendering radio bookend only`

So at FLUX fire, VRAM was reported by the loader as **2.13 GiB** -- which is what `OTR_DeferredCheckpointLoader` instruments as "cold". After FLUX load, allocated was 13.21 GiB. But LHM during sampler showed 15911 MB used. The gap between 13.21 GiB at load complete and 15911 MB during step 1 is ~2.7 GiB of additional VRAM that came from somewhere -- sampler activations? CLIP text encoder? Residual LLM tensors?

## Candidate causes (jury still out)

Four candidates ordered by my (Claude's + Jeffrey's) first read. The round-robin is to challenge this ordering:

### (a) Stale writer-LLM cache residency at FLUX entry -- strongest first read

L592's `pinned MODEL via load_models_gpu` keeps FLUX hot in VRAM, but if Gemma-4-E4B-it is still partially resident from the audio branch (or its KV cache isn't fully unloaded), the headroom shrinks by 2-4 GiB. The 1098 MB D3D Shared spill fits "almost-resident, sampler activations push it over the edge" better than the other three. The OTR canonical loader is `_otr_model_loader.unload_llm()` and `invalidate_cache_no_gpu_teardown()` (BUG-LOCAL-228 fix). The question is whether these are actually being called between `EpisodeAssembler emit audio_done` and `DeferredCheckpointLoader fire`, or whether the LLM's transformers KV cache is leaking.

### (b) Sampler-time launch flag candidates -- REJECTED at first read by Jeffrey

`--fast` or `--fast fp8_matrix_mult` ComfyUI flags. **REJECTED** because BUG-LOCAL-230 was caused by a launch flag (`--force-fp16`) added without proof. Don't reach for another launch flag as the first fix. Prove the symptom first, then propose surgery. Re-evaluate only if (a) and (c) are ruled out.

### (c) FLUX CLIP text encoder footprint

FLUX CLIP is ~4 GiB. If it loads alongside the 11 GiB FLUX weights without being offloaded to CPU during the diffusion sampler step, the 15 GiB resident plus sampler activations matches the observed 15911 MB. The question is whether ComfyUI / OTR's `BatchFluxRender` keeps CLIP hot for the whole sampler, or only encodes once and frees.

### (d) FLUX-schnell fallback at 4 steps

Status-12 explicitly retracted FLUX-schnell as the recommended primary fix in favor of the dtype removal. Listed last because: (1) it's a workflow-config swap, not a root-cause fix; (2) FLUX-schnell quality is lower (4 steps vs 20); (3) PASS1=3 portraits feed HuMo lip-sync, so portrait quality matters more than the bookend.

## What I need from you (round-robin)

1. **Order the candidates** by likelihood given the telemetry. Don't accept my ordering uncritically.
2. **Flag any candidate I've missed.** I'm specifically NOT considering: (e) some optimization knob in BatchFluxRender; (f) a sage attention enable that's actually safe at sampler time; (g) PyTorch caching allocator fragmentation; (h) ComfyUI's `--lowvram` / `--normalvram` mode shifting fp8 fp32-mat-mul fallback paths.
3. **First diagnostic to run** before any code change. Probe-design only -- no fix prescription.
4. **What to verify** in the OTR source: which files, which functions, which log markers. Be specific.
5. **Reject any candidate as "would be a fix without proof"** -- per the BUG-LOCAL-230 lesson.

Constraints:
- The only valid output is a probe / diagnostic / RCA recommendation. NOT a code prescription.
- Cite specific files when relevant (the repo is at `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\`; the FLUX render code is in `visual/batch_flux_render.py`; the LLM loader is in `nodes/_otr_model_loader.py`).
- Flag uncertainty. If you don't know, say "verify this against the source"; don't bluff.
- Be concise. 5-8 bullets max per recommendation.
