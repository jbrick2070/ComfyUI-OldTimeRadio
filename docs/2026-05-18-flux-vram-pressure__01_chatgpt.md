# Round A -- ChatGPT (gpt-5.5) elapsed=36.6s

## 1. Candidate ordering by likelihood

1. **(c) FLUX text-encoder / conditioning residency during sampler — highest likelihood now.**  
   The key clue is that `OTR_DeferredCheckpointLoader` reports `torch.cuda.memory_allocated()`-style VRAM at **13.21 GiB** after checkpoint load, but LHM sees **15.9 GB physical use** during step 1. That delta looks more like **CLIP/T5/conditioning/VAE/sampler workspace + allocator reserve** than a fully stale Gemma. Verify in `visual/batch_flux_render.py` whether CLIP/T5 is loaded for prompt encoding and whether it remains GPU-resident through `common_ksampler` / FLUX sampling.

2. **Sampler activation/workspace overhead is probably real, not itself a bug.**  
   A 1.5–2.7 GiB rise between “model loaded” and “first diffusion step” is plausible for FLUX activations, attention workspaces, conditioning tensors, CUDA graphs/workspaces, and PyTorch reserved-but-not-allocated memory. The defect may simply be: **13.21 GiB base leaves insufficient headroom** under a 16 GB card.

3. **Residual audio-model residency — missed candidate, probably more plausible than stale LLM.**  
   Bark/Kokoro/MusicGen run before FLUX. If any CUDA tensors survive, they could consume the 1–2 GiB that causes D3D Shared spill. The loader’s `2.13 GiB` “cold” reading weakens this, but does not fully rule out non-allocated reserved memory or non-torch CUDA allocations. Check audio teardown markers before `EpisodeAssembler audio_done`.

4. **(a) stale writer-LLM cache residency — possible, but weaker than first read.**  
   If Gemma tensors or KV cache were still live as PyTorch CUDA tensors, they should usually appear in the `2.13 GiB` pre-FLUX allocated number at `DeferredCheckpointLoader fire`. So I would not lead with this unless source/logs show `_otr_model_loader.unload_llm()` is missing or cache invalidation is skipped. Still worth verifying because the OTR architecture depends on it.

5. **(b) sampler launch flags — reject for now.**  
   `--fast`, `fp8_matrix_mult`, SageAttention, Flash Attention, etc. are **fixes without proof** in this case. BUG-LOCAL-230 was caused by an unproven launcher flag; do not repeat that pattern.

6. **(d) FLUX-schnell fallback — reject as RCA.**  
   It may reduce runtime by reducing steps, but it does not explain why one FLUX-dev step takes 154s. Treat it as a quality/runtime product tradeoff only after the spill cause is understood.

---

## 2. First diagnostic to run before any code change

Run one smoke with **VRAM phase probes around FLUX only**, no behavior change:

1. Log both:
   - `torch.cuda.memory_allocated()`
   - `torch.cuda.memory_reserved()`
   - optionally `torch.cuda.mem_get_info()`
   at these points:
   - immediately before `OTR_DeferredCheckpointLoader` fires
   - immediately after checkpoint load
   - before prompt/text encoding
   - after prompt/text encoding
   - immediately before sampler call
   - after sampler step 1 if there is an accessible callback/log point

2. In parallel, capture LHM values at the same approximate points:
   - GPU Memory Used
   - D3D Shared Memory Used
   - process RSS

3. The decision rule:  
   - If `allocated` jumps after text encoding and stays high into sampler: suspect **CLIP/T5/conditioning residency**.  
   - If `reserved` jumps while `allocated` does not: suspect **allocator reserve/workspace/headroom issue**, not stale model tensors.  
   - If both are already high before FLUX load: suspect **LLM/audio residual residency**.  
   - If only LHM rises but torch does not: suspect **non-torch CUDA/D3D allocation or driver-level paging**, not a Python tensor leak.

This is diagnostic only; do not add unload/offload behavior yet.

---

## 3. Source verification targets

Verify these files/functions directly; I cannot assert exact line numbers without the repo open.

1. **`visual/batch_flux_render.py`**  
   Look for the function/class producing log marker:  
   - `[BatchFluxRender] pinned MODEL via load_models_gpu`  
   - `[BatchFluxRender] skip_env_stills=True`  
   Verify:
   - whether only `MODEL` is pinned, or whether `CLIP`, `VAE`, conditioning, or text encoder objects are also retained
   - where positive/negative conditioning is encoded
   - whether CLIP/T5 is explicitly kept in scope until after sampling
   - whether any Comfy `load_models_gpu(...)` call includes CLIP/text encoder models, not just diffusion model

2. **`nodes/_otr_model_loader.py`**  
   Verify functions/logs around:
   - `unload_llm()`
   - `invalidate_cache_no_gpu_teardown()`
   - any Gemma / Transformers model cache object
   - any KV-cache retention path  
   Confirm there is a log marker between `EpisodeAssembler audio_done` and `DeferredCheckpointLoader fire` proving LLM unload actually happened.

3. **Deferred checkpoint loader implementation**  
   Search for the log markers:
   - `[DeferredCheckpointLoader] fire`
   - `[DeferredCheckpointLoader] load complete`  
   Verify what “VRAM allocated” means. If it uses only `torch.cuda.memory_allocated()`, it does **not** include all driver/LHM-visible memory and does not prove the GPU is physically cold.

4. **Audio nodes before `EpisodeAssembler`**  
   Search for Bark/Kokoro/MusicGen teardown or cache-clearing logs before:
   - `[EpisodeAssembler] emit audio_done signal`  
   If no teardown marker exists, residual audio CUDA state is a credible missed candidate.

5. **Workflow JSON / node graph**  
   Verify whether FLUX branch uses:
   - separate CLIP loader
   - T5/CLIP text encoder node
   - VAE decode in same branch
   - any KJNodes/SageAttention override accidentally set despite widget saying disabled  
   This is verification only; do not change workflow yet.

---

## 4. Things I would explicitly reject as “fix without proof”

1. Enabling SageAttention or chasing Flash Attention alternatives.  
   Blackwell `sm_120` support is already constrained, and this violates the “no low-level VRAM optimization” rule.

2. Adding `--fast`, `--fp8_matrix_mult`, `--lowvram`, `--normalvram`, or similar launcher flags.  
   The last major defect was a hidden launcher flag. No more flag surgery until telemetry says exactly what is wrong.

3. Switching to FLUX-schnell as the RCA answer.  
   That changes the workload, not the root cause.

4. Touching audio-path code.  
   Audio determinism rule C7 makes this off-limits as a speculative fix. You may inspect/log around audio teardown, but not alter audio generation behavior.

5. Implementing weight streaming, manual offload choreography, or attention kernel work.  
   Those are precisely the low-level VRAM optimizations the owner does not want.

---

## 5. My current RCA hypothesis

The fp8 fix is real and correct. The remaining 154s/step is most likely caused by **FLUX running with only a few hundred MB of physical headroom**, causing about **1 GB D3D Shared spill** during the first sampler step. The strongest next distinction is:

- **text encoder / conditioning / VAE kept hot**, versus  
- **normal sampler workspace exceeding available headroom**, versus  
- **residual audio/LLM state not visible in the current “allocated” log**.

The first diagnostic should therefore measure **allocated vs reserved vs LHM physical memory across FLUX text encoding and sampler entry**. That will tell you where the missing ~2.7 GiB appears before proposing any fix.
