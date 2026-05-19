# Synthesis -- 2026-05-18

**Question:** # BUG-LOCAL-231 -- FLUX sampler still ~10x slow after the BUG-LOCAL-230 fp8 fix

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

---

## ChatGPT (gpt-5.5)

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

---

## Gemini (gemini-3.1-pro-preview-customtools)

## 1. Agreement Statement
I PARTIALLY AGREE with the core recommendation to probe VRAM states and investigate CLIP/T5 residency, but ChatGPT fundamentally misunderstands how PyTorch's caching allocator interacts with OS-visible VRAM, leading to a flawed dismissal of the stale cache hypothesis.

## 2. Factual Errors in the ChatGPT Answer
*   **PyTorch Allocator Misunderstanding:** ChatGPT states: *"If Gemma tensors or KV cache were still live... they should usually appear in the 2.13 GiB pre-FLUX allocated number."* **FALSE.** If Python references to the LLM were deleted but `torch.cuda.empty_cache()` was not called, the VRAM remains held by PyTorch's caching allocator. It will show up in `torch.cuda.memory_reserved()` (and LHM's physical use), but `torch.cuda.memory_allocated()` will drop. The 2.13 GiB `allocated` metric does **not** rule out a massive stale LLM cache holding onto physical VRAM.
*   **ComfyUI Execution Order:** ChatGPT suggests checking if CLIP/T5 is loaded *alongside* the 11GB weights in `DeferredCheckpointLoader`. The log explicitly shows a delta of 11.08 GiB. The `flux1-dev-fp8.safetensors` UNet alone is ~11 GiB. This means the text encoders (T5xxl + CLIP L) are almost certainly NOT in that 13.21 GiB number. ComfyUI evaluates `CLIPTextEncode` nodes *after* the checkpoint loads and right before the sampler. That is exactly where your missing ~2.7+ GiB comes from.

## 3. Important Omissions
*   **The 2.13 GiB "Cold" Baseline:** On a truly cold ComfyUI instance, `allocated` is < 100 MB. A baseline of 2.13 GiB means something is explicitly still resident in PyTorch's active memory. This is likely Bark (~1.5 GiB) + Kokoro. If the audio models aren't offloaded to CPU, they are eating into your 14.5 GiB ceiling before FLUX even starts.
*   **T5xxl Precision:** FLUX requires T5xxl. In fp16, T5xxl is ~9.8 GiB. In fp8, it's ~4.9 GiB. If your workflow is silently loading T5xxl in fp16 during the text encoding phase, it will instantly cause the 1 GiB D3D Shared spill you are seeing.
*   **Garbage Collection / IPC:** ChatGPT didn't explicitly recommend checking for `gc.collect()` and `torch.cuda.empty_cache()` at the exact phase boundaries (LLM -> Audio -> Visual).

## 4. My Recommendation (Next Steps)
Do not touch the audio generation logic (Rule C7). Focus strictly on VRAM lifecycle management at the phase boundaries.

1.  **Probe `memory_reserved` (The real bottleneck):** In `nodes/_otr_model_loader.py` and `visual/batch_flux_render.py`, update your telemetry to log `torch.cuda.memory_reserved()`. This is what LHM actually sees. If `reserved` is 6+ GiB when `allocated` is 2.13 GiB, you have an allocator fragmentation/leak issue from the LLM phase.
2.  **Enforce Hard Boundaries:** Check `_otr_model_loader.unload_llm()` and the end of `EpisodeAssembler`. Ensure they both explicitly call `gc.collect()` followed by `torch.cuda.empty_cache()`. If the audio models (Bark/Kokoro) are holding that 2.13 GiB, ComfyUI's `model_management.unload_all_models()` should be invoked before `DeferredCheckpointLoader` fires.
3.  **Verify T5xxl Precision:** Check your workflow JSON or `BatchFluxRender` node graph. Ensure the `DualCLIPLoader` (or equivalent) is explicitly set to load T5 in `fp8_e4m3fn`. If it defaults to fp16, you will never fit FLUX + T5 + 2.13 GiB of audio models in 16 GiB.
4.  **Check CLIP Offload Behavior:** ComfyUI normally offloads CLIP to system RAM before the `KSampler` starts. If `BatchFluxRender` (L592: `pinned MODEL via load_models_gpu`) is accidentally pinning the *entire* loaded state (including CLIP) instead of just the diffusion model, it will cause this exact spill. Verify exactly what `load_models_gpu` is being passed.

## 5. Uncertainties to Verify Against Source
*   **Checkpoint Composition:** I am uncertain if the `flux1-dev-fp8.safetensors` you are using is a unified checkpoint (UNet + VAE + CLIP/T5) or just the UNet. At 11.08 GiB, it is almost certainly just the UNet (or UNet + VAE). Verify your workflow to see where T5xxl is being loaded.
*   **The 2.13 GiB Identity:** I am uncertain what exactly comprises the 2.13 GiB "cold" allocation. Verify if this is Bark/Kokoro, or if it's a leaked LLM KV cache that wasn't properly dereferenced.

---

## NVIDIA (nvidia/llama-3.3-nemotron-super-49b-v1.5)

### Third Opinion: Tying Together ChatGPT and Gemini Insights

---

#### **1. Agreement with Previous Analyses**
- **VRAM Headroom as Root Cause:** Concur with both that the core issue is insufficient VRAM headroom (14.5 GiB ceiling) during FLUX sampling, causing D3D Shared spill. The 13.21 GiB post-load + 2.13 GiB baseline = **15.34 GiB minimum**, already over the ceiling before activations.
- **Diagnostic Focus on Memory Metrics:** Agree that logging `torch.cuda.memory_reserved()` (not just `allocated`) is critical to distinguish between stale allocations and activation workspace.

---

#### **2. Key Disagreements & Third Path**
**Disagreement on (a) vs (c):**
- **ChatGPT** prioritizes **(c) FLUX text encoder residency** (CLIP/T5).
- **Gemini** argues for **(a) stale LLM/audio cache** due to PyTorch allocator behavior.
- **Third Path:** Both are plausible, but **Gemini’s point about `memory_reserved` is more foundational**. The 2.13 GiB baseline likely includes:
  - **Residual audio models (Bark/Kokoro):** ~1.5 GiB (Bark alone).
  - **Stale LLM KV cache:** If `empty_cache()` wasn’t called, this could persist in `memory_reserved`.
  - **T5xxl in fp16:** If loaded during text encoding, adds ~5 GiB vs fp8.

**Recommendation:** Prioritize verifying the 2.13 GiB baseline composition before blaming text encoder residency.

---

#### **3. Factual Errors/Hallucinations**
- **ChatGPT’s Misunderstanding of PyTorch Allocator:** Gemini correctly flags this. `torch.cuda.memory_allocated()` dropping does **not** imply physical VRAM is freed if `empty_cache()` isn’t called.
- **Gemini’s Assumption About T5xxl Size:** Gemini states T5xxl in fp16 is ~9.8 GiB. This is **overstated**—T5xxl in fp16 is ~5–6 GiB (per HuggingFace model sizes). Still significant, but not 9.8 GiB.
- **File Paths:** Both reference `visual/batch_flux_render.py` and `nodes/_otr_model_loader.py` correctly.

---

#### **4. Important Omissions**
1. **VAE Residency:** If `flux1-dev-fp8.safetensors` includes the VAE (~2–3 GiB), it may stay resident during sampling, contributing to headroom loss.
2. **Workflow Configuration for Text Encoding:** Neither explicitly checks if the workflow uses a separate CLIP/T5 loader or if it’s bundled with the FLUX checkpoint.
3. **PyTorch Graph Mode / CUDA Graphs:** If FLUX uses CUDA graphs for sampling, they may reserve memory not reflected in standard metrics.

---

#### **5. Prioritized Recommendations**
1. **Log `memory_reserved` at Phase Boundaries**  
   - **Files:**  
     - `nodes/_otr_model_loader.py` (around `unload_llm()` and `invalidate_cache_no_gpu_teardown()`)  
     - `visual/batch_flux_render.py` (before/after text encoding and sampler entry)  
   - **What to Log:**  
     ```python
     print(f"Reserved: {torch.cuda.memory_reserved(0)/1e9:.2f} GiB, Allocated: {torch.cuda.memory_allocated(0)/1e9:.2f} GiB")
     ```

2. **Verify T5xxl Precision and Residency**  
   - **Check Workflow JSON/Node Graph:**  
     - Look for `T5xxl` or `CLIPTextEncode` nodes.  
     - Confirm dtype is `fp8_e4m3fn` (not `float16`).  
   - **Code:** If `DualCLIPLoader` is used, check its `dtype` argument in `visual/batch_flux_render.py`.

3. **Enforce Cleanup After Audio Phase**  
   - **Files:**  
     - `nodes/_otr_model_loader.py` (ensure `unload_llm()` calls `gc.collect()` + `torch.cuda.empty_cache()`)  
     - `EpisodeAssembler` code (verify `audio_done` signal triggers `model_management.unload_all_models()`)  

4. **Check VAE Inclusion in Checkpoint**  
   - **Action:** Inspect `flux1-dev-fp8.safetensors` metadata or workflow config to confirm if VAE is included.  
   - **If Included:** Ensure it’s offloaded after encoding (if not needed during sampling).

5. **Probe Text Encoder Lifecycle**  
   - **Code:** In `visual/batch_flux_render.py`, log when CLIP/T5 is loaded/unloaded relative to sampler start.  
   - **Key Question:** Is CLIP kept in `torch.cuda.memory_reserved()` during sampling?

---

#### **6. Uncertainties Requiring Verification**
- **Checkpoint Composition:** Is `flux1-dev-fp8.safetensors` UNet-only or includes VAE/CLIP?
- **T5xxl Precision:** Is it explicitly set to fp8 in the workflow?
- **Audio Model Unload:** Does `EpisodeAssembler` actually trigger `unload_all_models()`?

---

### **Final Diagnostic Prescription**
**Run a smoke test with the following instrumentation:**  
1. **Phase Boundaries (LLM → Audio → FLUX):**  
   - Log `memory_reserved`/`allocated` + LHM values at:  
     - After `EpisodeAssembler.audio_done`  
     - Before `DeferredCheckpointLoader.fire`  
     - After FLUX checkpoint load  
     - Before text encoding  
     - After text encoding  
     - Before sampler step  
     - After sampler step  

2. **Code Changes for Telemetry:**  
   - Add `gc.collect()` + `torch.cuda.empty_cache()` in `_otr_model_loader.unload_llm()` if missing.  
   - Log T5xxl/CLIP load/unload events explicitly in `BatchFluxRender`.

**Expected Outcome:**  
- If `memory_reserved` remains high after audio phase → stale audio/LLM models.  
- If `reserved` spikes during text encoding → T5xxl/CLIP in fp16 or kept hot.  
- If `reserved` drops post-encoding but `allocated` is low → activation workspace is the bottleneck.

---

## To decide (Claude / human)

- [ ] All three agree:
- [ ] Two-vs-one splits:
- [ ] Facts to verify:
- [ ] Final grounded recommendation:
