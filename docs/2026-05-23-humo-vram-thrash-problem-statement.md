# Problem Statement — HuMo per-clip render time degrades catastrophically across a run

**Date:** 2026-05-23
**Component:** `nodes/batch_humo_render.py` (`OTR_BatchHumoRender`)
**Related bug:** BUG-LOCAL-231 (open — "HuMo/FLUX 200-250x slower than reference")
**Status:** for outside consultation. The diagnosis below is a hypothesis, not a confirmed root cause.

---

## 1. Summary

A single OldTimeRadio episode runs a batch of HuMo (Wan2.1-HuMo-14B) lip-sync
clips, one per character dialogue line, on a 16 GB laptop GPU. The **first**
HuMo clip renders in a normal ~9 minutes. The **second** clip took **3 hours
43 minutes**. The render is not crashing — the GPU sits pegged at 100%
utilization the whole time — it is degrading. At this rate a ~24-chunk
episode would take multiple days instead of the expected ~2-3 hours.

We want to know: is this a fixable VRAM-management bug in our render loop, a
mis-tuned ComfyUI offloader, or a hard ceiling that only a lower resolution /
smaller model can solve?

---

## 2. Hardware & environment

- **GPU:** NVIDIA RTX 5080 Laptop, **16 GB VRAM**, Blackwell `sm_120`, single GPU.
- **OS:** Windows 11.
- **Stack:** torch 2.10 + CUDA 13, ComfyUI 0.22.2, Python 3.12.
- Flash-Attention 2 is unavailable on this stack (no prebuilt wheel for
  torch 2.10 / CUDA 13 / Blackwell / Windows); SageAttention + SDPA are active.
- **Self-imposed ceiling:** 14.5 GB peak VRAM (the project's hard budget).
- **Constraint:** 100% local, offline, open-source — no cloud, no API.

## 3. The model stack in the HuMo stage

Render path per episode: FLUX portraits → `OTR_UnloadAll` (frees FLUX VRAM) →
`OTR_BatchHumoRender`.

`OTR_BatchHumoRender` runs in three phases:

- **Phase A** — encode 14 positive + 1 negative text prompts (umt5 / WanTE
  text encoder, ~6.4 GB).
- **Phase B** — encode 24 audio segments via Whisper Large v3 (~1.2 GB).
- **Phase C** — the per-clip / per-chunk HuMo render loop. HuMo (Wan2.1-HuMo-14B
  fp8) **stages at ~16.5 GB** — i.e. the model alone does not fit in 16 GB and
  ComfyUI's dynamic VRAM loader must offload/page.

Render parameters (`INPUT_TYPES` defaults): **480×832**, 25 fps, **6 steps**,
`uni_pc` / `simple`, cfg 1.0, ~7 s clip length. Lines longer than
`HUMO_MAX_FRAMES` (353) are split into multiple chunks (BUG-LOCAL-086), so a
14-line episode becomes ~24 render chunks.

## 4. Observed symptom (soak run 2026-05-23)

Episode `signal_lost_frying_pan_clock_20260523_184158`, 14 character lines.
File timestamps in the episode folder:

| Artifact | Finished | Delta |
|---|---|---|
| Final audio+procgen-video MP4 | 18:43 | — |
| HuMo clip `b002.mp4` | 18:54 | ~9 min after HuMo start |
| HuMo clip `b003.mp4` | **22:37** | **3 h 43 min after b002** |
| (clip `b004` still rendering at 23:15) | — | >38 min so far |

During the 3h43m gap the process was alive: ComfyUI's Python process held,
the GPU read **100% utilization, ~78 W power draw, 15.1 GB used**. The low
power at full utilization is the signature of a memory-bound / PCIe-paging
workload — the GPU is busy moving data, not computing.

This is the same failure *class* the code already documents from a prior
incident (see §6.3): "~88x slowdown (5,284 s/it vs healthy 60 s/it), because
the offloader fell into perpetual swap-to-CPU through PCIe."

## 5. The Phase C render loop — the suspect

The loop renders every chunk of every line. Abridged real code from
`nodes/batch_humo_render.py` (~line 2406 onward):

```python
for entry in plan:                       # one per dialogue line
    ...
    for chunk_idx, chunk in enumerate(_chunks):   # BUG-086 sub-chunks
        humo_out = _call(
            humo_node,                    # WanHuMoImageToVideo
            width=width, height=height,   # 480 x 832
            length=entry["humo_length"],
            positive=entry["positive"],
            negative=negative,
            vae=vae,
            audio_encoder_output=chunk["audio_emb"],
            ref_image=entry["ref_image"],
        )
        humo_pos, humo_neg, humo_latent = humo_out[:3]

        samples = _call(
            sampler,                      # KSampler, 6 steps uni_pc
            model=model, seed=shot_seed, steps=steps, cfg=cfg,
            sampler_name=sampler_name, scheduler=scheduler,
            positive=humo_pos, negative=humo_neg,
            latent_image=humo_latent, denoise=1.0,
        )[0]

        images_out = _call(vae_decoder, samples=samples, vae=vae)[0]
        # ... trim warmup frames, mux chunk mp4 via ffmpeg ...
```

**There is no VRAM reclamation anywhere inside this loop.** A repo-wide grep
for `empty_cache` / `soft_empty_cache` / `unload_all_models` / `gc.collect`
confirms the Phase C loop body (≈ lines 2406-2907) contains none. The loop
renders chunk after chunk, line after line, and never flushes the CUDA
allocator pool between renders.

## 6. What VRAM management *does* exist (and when it runs)

### 6.1 One-time inter-phase reset (Phase B → Phase C), runs **once**

```python
# ---- 8.5. VRAM cleanup between Phase B and Phase C ----
# ... move every Phase A/B output tensor to CPU ...
mm.unload_all_models()
mm.soft_empty_cache(force=True)
```

This is the only routine cleanup. It fires once, before the loop starts.

### 6.2 OOM-only hard reset, runs **only on an exception**

```python
_needs_oom_cleanup = (cuda_hard_reset_on_oom and _is_oom_exception(exc))
...
if _needs_oom_cleanup:
    _hard_reset_cuda_context()   # unload_all_models + gc.collect
                                 # + soft_empty_cache + empty_cache
```

`_hard_reset_cuda_context()` only runs when a line **raises an OOM
exception**. The observed failure is a *slowdown*, not an OOM — the allocator
thrashes but never hard-fails — so this cleanup **never fires**. The code only
reclaims VRAM when it crashes; it does nothing when it merely degrades.

### 6.3 A prior pin attempt was removed for causing this exact slowdown

A comment block at ~line 1813 documents that an earlier version pinned HuMo
on the GPU with `mm.load_models_gpu([model], force_full_load=True)`:

> "The pin was structurally broken on 16 GB devices: HuMo stages at
> 16,531 MB ... Under encoder pressure ComfyUI's dynamic VRAM offloader
> silently violated the pin ... which fragmented the cudaMallocAsync pool
> ... Phase C HuMo then ran at ~88x slowdown (5,284 s/it vs healthy
> 60 s/it), because the offloader fell into perpetual swap-to-CPU through
> PCIe instead of running contiguously in GDDR."

So the team already knows the offloader can fall into PCIe-paging. The 8.5
reset was the fix for the *Phase B → C boundary*. Nothing addresses
fragmentation/accumulation **within** the Phase C loop.

## 7. Hypotheses (ranked, unconfirmed)

1. **No per-clip allocator flush.** The CUDA caching allocator fragments as
   each chunk allocates/frees latents, decoded image tensors, and conditioning.
   Clip 1 runs in a clean pool; by clip 2 the pool is fragmented enough that
   the dynamic offloader can no longer fit HuMo contiguously and starts paging
   weights over PCIe every step. A `soft_empty_cache()` after each chunk/clip
   may keep the pool defragmented.

2. **Conditioning-tensor accumulation.** Section 8.5 moves every
   `entry["positive"]` conditioning tensor to CPU once. But `WanHuMoImageToVideo`
   pulls them back to GPU when it renders, and `plan` retains every `entry` for
   the whole loop. If the cond tensors are not returned to CPU after a line
   finishes, conditioning for *all processed lines* accumulates GPU-resident.

3. **Hard ceiling.** HuMo-14B fp8 stages at ~16.5 GB on a 16 GB card. The
   dynamic offloader is doing PCIe paging *by design*. No amount of loop-side
   cleanup changes the fact that the model does not fit; only lower resolution,
   fewer frames, block-swap tuning, or a smaller model reduces the footprint.

## 8. Candidate fixes (for discussion — not yet decided)

- **A. Per-clip / per-chunk VRAM reclamation.** Add `mm.soft_empty_cache()`
  (and explicit `del` of `samples` / `images_out` / `humo_latent`) after each
  chunk in the Phase C loop. Cheap; directly targets hypothesis 1.
- **B. Lower render resolution.** 480×832 → e.g. 416×720 or 384×640. Smaller
  latents = less VRAM per render and slower fragmentation. Quality cost is the
  open question.
- **C. Per-clip model unload + reload.** Evict HuMo and re-stage it for each
  clip so every clip starts from a clean pool. Trades a reload (~tens of
  seconds) for a guaranteed-clean allocator — likely far cheaper than a
  3-hour thrash.
- **D. Allocator tuning.** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
  or ComfyUI block-swap / `--reserve-vram` tuning.
- **E. Return cond tensors to CPU after each line** (hypothesis 2).
- **F. Fewer steps / shorter clips** to shrink peak working set.

## 9. Questions for outside reviewers

1. On a 16 GB card running a 14B fp8 video model that stages above 16 GB, is
   per-clip `soft_empty_cache()` the right tool, or does it just stall the
   pipeline without curing the fragmentation?
2. Is the "100% util / ~78 W / steady VRAM" signature definitely PCIe paging,
   or could it be something else (e.g. a CPU-bound pre/post step)?
3. Is `expandable_segments:True` known to help or hurt on Blackwell + CUDA 13
   for this kind of repeated large-model render loop?
4. Is a per-clip unload+reload of a 14B model a sane pattern, or is the reload
   cost in practice worse than living with the offloader?
5. At what resolution does Wan2.1-HuMo-14B fp8 comfortably fit a 16 GB card
   with headroom for the VAE decode?

## 10. Notes / non-negotiables for any fix

- Audio is unaffected — HuMo is video-only; a failed/slow HuMo render does not
  corrupt the episode audio, which completes earlier in the pipeline.
- 100% local / offline / open-source. No cloud offload.
- 14.5 GB peak VRAM ceiling is the project budget.
- The fix must not regress the FLUX stages, which currently render fine
  (~1.2 s/it for 20 steps in the same run).

---

## Appendix — reproduction

1. Run a normal OTR episode (`otr_scifi_16gb_full.json`) with ≥ 3 character
   lines so Phase C renders multiple HuMo clips.
2. Watch `output/otr/episodes/<ep>/` — note the wall-clock gap between
   consecutive `bNNN.mp4` clip files.
3. Expected (broken) behaviour: clip 1 ≈ 9 min, clip 2+ degrade by 10-25x.
4. `nvidia-smi` during the slow clip shows ~100% utilization at low power draw.
