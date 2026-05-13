# Problem Statement: HuMo Phase C 88× slowdown on large episodes (alien_whispers, 7 cast / 61 lines / 90 audio chunks)

> **Round-robin consultation request.** Please review independently and respond with: (1) which root-cause hypothesis you find most consistent with the evidence; (2) any disagreement with the proposed surgical fixes; (3) any consideration we missed. Claude will synthesize Gemini's and GPT's responses, flag disagreements, and pick the path forward.
>
> **TEMPORARY DOC** — will be `git rm`'d once fix is verified. Do not link from outside `docs/`.

---

## Hardware envelope (read this first)

- RTX 5080 Laptop, **16 GB VRAM** (15.92 GiB usable), Blackwell sm_120, single GPU
- Windows 11, Python 3.12, torch 2.10.0, CUDA 13.0
- SDPA + SageAttention only (no Flash Attention)
- VRAM real-world ceiling target: **14.5 GiB** peak before alarms fire
- ComfyUI Desktop using `dynamic VRAM loading` (cudaMallocAsync allocator + async weight offloading with 2 streams + pinned memory 25982.0)

## Repo + commit

- `https://github.com/jbrick2070/ComfyUI-OldTimeRadio` branch `v2.0-alpha`
- Incident HEAD: `467969a` (HuMo steps 6→5 widget change; not the cause — slowdown also seen at 6 steps)
- Tag for rollback: `v2.0-alpha-pre-humo-fix-2026-05-08`

## Models

- **HuMo:** `Wan2_1-HuMo-14B_fp8_e4m3fn_scaled_KJ.safetensors` + `lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors` LoRA + `ModelSamplingSD3 shift=8`. Stages at **16,531 MB**.
- **Text encoder:** `umt5_xxl_fp8_e4m3fn_scaled.safetensors` via `CLIPLoader`. Stages at **6,419 MB**.
- **VAE:** `wan_2.1_vae.safetensors`. Stages at **242 MB**.
- **Audio encoder:** `whisper_large_v3_fp16.safetensors` via `AudioEncoderLoader`. Stages at **1,215 MB**.

Sum of staged sizes: **24.4 GB**, against a 16 GB device. Dynamic offloading is mandatory.

---

## 1. Symptom (hard data)

### Control: 4-HuMo smoke (commit 9c6353d, ledger `synthetic_4humo_ledger.json`, 4 character lines × 1 chunk each = 4 audio chunks)

```
clip 1 (l002a): 100%|████| 5/5 [05:09<00:00, 61.74s/it]   →  377 s wall (6:17)
clip 2 (l002b): 100%|████| 5/5 [05:54<00:00, 59.15s/it]   →  374 s wall (6:14)
clip 3 (l002c): 100%|████| 5/5 [05:50<00:00, 58.42s/it]   →  369 s wall (6:09)
soak cap fired (clip 4 not rendered)
Prompt executed in 18:49
```

Per-step time: **~60 s/it**. Healthy. No OOM. No allocator drift across 3 consecutive clips.

### Failure: alien_whispers full run (HEAD 467969a, real episode, 7 cast / 61 dialogue lines / 90 audio chunks after BUG-086 splitting)

```
[BatchHumoRender] episode_id=signal_lost_alien_whispers_20260508_194214 cast=7 lines=61
[BatchHumoRender] BUG-094 estimated 61 line timings across 438.1s episode (avg 7.18s/line)
[BatchHumoRender] Phase A: encoding 59 positive + 1 negative text prompts
   ↳ "Model WanTEModel prepared for dynamic VRAM loading. 6419MB Staged" appears 60×
[BatchHumoRender] Phase B: encoding 90 audio segments via Whisper
   ↳ "Model WhisperLargeV3 prepared for dynamic VRAM loading. 1215MB Staged" appears 90×
[BatchHumoRender] Phase A/B tensors moved to CPU
[BatchHumoRender] Inter-phase VRAM cleanup: unload_all_models + soft_empty_cache
[BatchHumoRender] Phase C: HuMo render loop, 0 lines
Requested to load WanVAE
Model WanVAE prepared for dynamic VRAM loading. 242MB Staged.
Requested to load WAN21_HuMo
Model WAN21_HuMo prepared for dynamic VRAM loading. 16531MB Staged. 1053 patches attached.
100%|██████████| 5/5 [7:20:23<00:00, 5284.62s/it]    ←  clip 1 = 7h 20m
 80%|████████  | 4/5 [5:31:56<1:22:42, 4962.28s/it]   ←  clip 2 at 80%, projected ~7h
```

Per-step time: **~5,284 s/it ≈ 88 min/step** vs. healthy 60 s/it.
**Slowdown factor: ~88×.** First clip took 7 h 20 min (vs. 6 min healthy). At this rate, all 60 character clips would take ~440 hours / 18 days.

The same `BatchHumoRender` codepath, same `git=467969a` commit, on the same machine, runs at full speed for 4 chunks and crawls at 1/88× speed for 90 chunks. The variable is episode size, not the code.

---

## 2. Why this matters

This is the **actual root cause** of months of "HuMo hangs mid-script" reports. Earlier today we incorrectly attributed those to the BUG-LOCAL-126 soak cap (`humo_max_lines_per_process=3`); fixing the cap (commit `02a5749`, set to 0) was correct but secondary. The dominant factor is **Phase C step time blowing up by 88× on large episodes**, which makes any positive cap value irrelevant — the workflow effectively halts even before the cap fires because the user kills it / OS reboots / power cycles.

---

## 3. Existing in-code documentation of the failure mode (BUG-LOCAL-081)

`nodes/batch_humo_render.py` lines 2061–2081 — comment block written when this exact symptom was first hit at smaller scale:

```python
# ---- 8.5. VRAM cleanup between Phase B and Phase C ----
# BUG-LOCAL-081: at 30+ lines, Phase B reloaded Whisper for
# every line (each AudioEncoderEncode call triggered a fresh
# "WhisperLargeV3 prepared" log). At Phase B end, GPU still
# holds: Whisper weights (1.2 GB), umt5_xxl text encoder
# (6.4 GB from Phase A), 30 audio embedding tensors (~10 MB
# each = 300 MB), 30 positive cond tensors (~50 MB each =
# 1.5 GB). Total ~9.4 GB pinned before HuMo even starts to
# load.  Phase C asks for HuMo (16.5 GB staged) on a 16 GB
# card -- ComfyUI's dynamic VRAM loader thrashes pages
# perpetually and never converges to forward progress.
# Symptom: KSampler stuck at "0/6 [?it/s]" for 20+ minutes.
#
# Fix is two-step:
#   1. Move every Phase A/B output tensor to CPU. That
#      releases GPU pages backing positive/negative cond and
#      audio embeddings -- Phase C's WanHuMoImageToVideo
#      moves them back to GPU when it actually needs them.
#   2. unload_all_models + soft_empty_cache to evict Whisper
#      and umt5_xxl from GPU and return pages to the CUDA
#      allocator pool.
```

The 081 fix mitigates up to ~30 lines. **alien_whispers ran 90 audio chunks + 60 positive cond tensors = ~3× the envelope the 081 fix was tested against.** The fix's code is intact and runs (log confirms `Phase A/B tensors moved to CPU` and `Inter-phase VRAM cleanup: unload_all_models + soft_empty_cache`), but the resulting CUDA pool state is still fragmented enough at 60+ chunks that HuMo's first sample step thrashes.

## 4. Phase A code (current)

```python
# ---- 7. Phase A: encode all text prompts up front ----
log.info("[BatchHumoRender] Phase A: encoding %d positive + 1 negative text prompts",
         len(plan))
try:
    negative = _call(text_enc, clip=clip, text=_CHINESE_NEGATIVE)[0]
except Exception as exc:
    raise RuntimeError(f"BatchHumoRender: negative encode failed: {exc}")

for entry in plan:                                              # ← 59-iteration loop
    try:
        entry["positive"] = _call(text_enc, clip=clip, text=entry["pos_text"])[0]
    except Exception as exc:
        log.warning("[BatchHumoRender] %s: text encode failed: %s",
                    entry["line_id"], exc)
        entry["positive"] = None
```

Each `_call(text_enc, ...)` is a `CLIPTextEncode` invocation. On every call ComfyUI's `model_management.load_models_gpu` is invoked, which logs `Model WanTEModel prepared for dynamic VRAM loading. 6419MB Staged.` That's why we see 60 `Model WanTEModel prepared` lines — once per prompt.

**Open question:** is each `prepared` line a real re-stage (PCIe weight transfer of 6.4 GB) or is it a no-op log when the model is already resident? If real, that alone explains the slowdown. If no-op, the slowdown is cumulative tensor pin growth (positive cond tensors, ~50 MB × 59 = ~3 GB never released).

## 5. Phase B code (current)

```python
# ---- 8. Phase B: encode all per-line audio up front ----
_total_audio_segments = sum(len(e["chunks"]) for e in plan)
log.info("[BatchHumoRender] Phase B: encoding %d audio segments via Whisper",
         _total_audio_segments)
for entry in plan:                              # 59 outer iterations
    for chunk in entry["chunks"]:               # 90 inner total
        try:
            chunk["audio_emb"] = _call(
                audio_enc_node,
                audio_encoder=audio_encoder,
                audio=chunk["audio"],
            )[0]
        except Exception as exc:
            log.warning("[BatchHumoRender] %s: audio encode failed: %s",
                        entry["line_id"], exc)
            chunk["audio_emb"] = None
```

Same per-iteration pattern, same `prepared` log spam (90 lines). Whisper is 1.2 GB; PCIe-streaming it 90 times = ~108 GB of PCIe traffic if the `prepared` calls are real re-stages.

## 6. Inter-phase cleanup (current, full)

```python
# ---- 8.5. VRAM cleanup between Phase B and Phase C ----
try:
    import torch
    def _to_cpu(obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().to("cpu", copy=False) if obj.device.type == "cuda" else obj
        if isinstance(obj, list):
            return [_to_cpu(x) for x in obj]
        if isinstance(obj, tuple):
            return tuple(_to_cpu(x) for x in obj)
        if isinstance(obj, dict):
            return {k: _to_cpu(v) for k, v in obj.items()}
        return obj
    negative = _to_cpu(negative)
    for entry in plan:
        if entry.get("positive") is not None:
            entry["positive"] = _to_cpu(entry["positive"])
        if entry.get("audio_emb") is not None:
            entry["audio_emb"] = _to_cpu(entry["audio_emb"])
    log.info("[BatchHumoRender] Phase A/B tensors moved to CPU")
except Exception as exc:
    log.warning("[BatchHumoRender] CPU offload failed: %s", exc)

try:
    import comfy.model_management as mm
    log.info("[BatchHumoRender] Inter-phase VRAM cleanup: unload_all_models + soft_empty_cache")
    mm.unload_all_models()
    mm.soft_empty_cache(force=True)
except Exception as exc:
    log.warning("[BatchHumoRender] inter-phase VRAM cleanup failed: %s", exc)
```

What it does: walks every `positive`, `negative`, `audio_emb` payload (lists / tuples / dicts of tensors) and detaches+moves each tensor to CPU; then `unload_all_models()` + `soft_empty_cache(force=True)`.

What it does **not** do (potential gaps):
- No `gc.collect()` between CPU offload and unload — Python refs may still hold GPU memory through closures
- No `torch.cuda.synchronize()` before `empty_cache` — pending kernel launches may still hold GPU pages
- No `torch.cuda.empty_cache()` (relies on `soft_empty_cache(force=True)` which is ComfyUI's wrapper)
- No `torch.cuda.ipc_collect()` — IPC handles can pin segments
- No `torch.cuda.reset_peak_memory_stats()` (cosmetic, but useful for telemetry)

The BUG-LOCAL-126 hard-reset chain (after caught OOM) does have all of those. It does NOT fire on the inter-phase boundary, only on a `torch.cuda.OutOfMemoryError`. On a healthy-but-fragmented pool there's no OOM, so the hard reset never runs and the lighter cleanup is what we get.

## 7. What ALSO occupies VRAM at Phase C entry

The OTR_UnloadAll node fires before HuMo:
```
[UnloadAll] llm_polish.unload() called
[UnloadAll] comfy.model_management.unload_all_models() called
[UnloadAll] soft_empty_cache(force=True) called
```

Same cleanup primitives. Should leave VRAM nearly empty. Then HuMo's loader chain re-stages WAN21_HuMo (16,531 MB) which immediately starts hitting the dynamic offloader.

LibreHardwareMonitor wasn't queried during the failure run; we don't have a direct VRAM-residency snapshot at the Phase B → Phase C transition. **This is data we should grab on the next reproduction.**

---

## 8. Hypotheses (ranked, with evidence)

### H1 — Allocator fragmentation past the 081 fix's tested envelope (most likely)

**Claim:** the 081 cleanup runs correctly but doesn't fully drain the cudaMallocAsync pool after 60+ encoder invocations + 90+ audio embeds. The pool's free list ends up with thousands of small holes that can't be coalesced for HuMo's large contiguous allocations. HuMo's first KSampler step then triggers continuous `cudaMalloc` / `cudaFree` cycles via the dynamic offloader, each round-trip via PCIe at ~10 GB/s on a saturated Gen5 lane.

**Evidence for:**
- 4-HuMo smoke (4 chunks) runs healthy → pool isn't fragmented at small scale
- alien_whispers (90 chunks) runs at 1/88× speed → 60-fold scale increase yields 88-fold slowdown, consistent with allocator thrashing
- 081 comment block predicts exactly this symptom ("KSampler stuck at 0/6 [?it/s]") when chunk count exceeds tested envelope
- The `prepared` log spam (60 + 90 = 150 lines) suggests each call goes through `load_models_gpu` even if the model is already there, which means PCIe traffic + allocator churn whether or not the bytes physically move

**Evidence against:**
- Hard to confirm without a CUDA pool dump at the Phase C entry
- `soft_empty_cache(force=True)` is supposed to release everything; if it doesn't, that's a ComfyUI bug not specifically ours

### H2 — Phase A/B encoder re-staging is a real PCIe transfer per call, not a no-op log

**Claim:** ComfyUI's `load_models_gpu` is not idempotent on already-resident models — it actually re-stages weights on every call. With 60 + 90 = 150 invocations, 6.4 GB + 1.2 GB respectively, that's ~600 GB of PCIe transfer. At 10 GB/s sustained, that's ~60 seconds of pure PCIe traffic per Phase A/B execution, which by itself doesn't explain 88× slowdown but contributes fragmentation.

**Evidence for:**
- 150 `prepared` log lines is suggestive
- ComfyUI's dynamic VRAM model is known to re-page when other models load between calls — Phase A and Phase B alternating with implicit allocator events could trigger constant eviction

**Evidence against:**
- We don't have ground-truth from `nvidia-smi dmon` or PCIe counters showing actual transfer volume
- ComfyUI's `model_management.py` has a "models_already_loaded" check; the spam might just be noisy logging

### H3 — Cumulative tensor pin growth in Python globals

**Claim:** `_to_cpu` walks recursive structures but may miss tensors held by closures, model patches, or `nn.Module` parameter buffers (e.g., HuMo's 1053 `patches attached`). At 60+ entries those leftover GPU references can sum to several GB, leaving HuMo's ksampler space-starved.

**Evidence for:**
- `Model WAN21_HuMo prepared ... 1053 patches attached` — the 1053 number is large and is per-load. If patches accumulate across loads (i.e., the LoraLoaderModelOnly attaches but never releases), we'd see growth
- 4-HuMo smoke survives 3 clips before soak cap → smaller patch-attach pressure

**Evidence against:**
- The `_to_cpu` walk is aggressive (handles list/tuple/dict recursion)
- `unload_all_models()` should unload patches with the model

### H4 — ComfyUI dynamic loader (cudaMallocAsync) regression / config issue under sustained pressure

**Claim:** the `cudaMallocAsync` allocator + 2-stream async weight offloading is sensitive to allocation patterns. Under heavy short-lived allocations (Phase A/B encoder forwards) followed by a single huge contiguous request (HuMo's KSampler), the async streams may get out of sync and cause artificial waits.

**Evidence for:**
- The "Set vram state to: NORMAL_VRAM" + "Using async weight offloading with 2 streams" + "Enabled pinned memory 25982.0" stack is non-default and Blackwell-specific
- The startup banner shows `Device: cuda:0 NVIDIA GeForce RTX 5080 Laptop GPU : cudaMallocAsync` — caller-tunable

**Evidence against:**
- 4-HuMo smoke uses the same allocator and works fine; the regression only appears at scale
- We can't easily A/B this without restarting ComfyUI Desktop with different flags

### H5 — Mistral-Nemo or other LLM still resident at Phase C entry

**Claim:** OTR_UnloadAll fires `unload_all_models()` which should evict Mistral-Nemo (4-bit NF4, ~7.7 GiB), but if the model has any non-Comfy reference (e.g., held by `StoryOrchestrator` Python globals), it stays resident. HuMo arrives to a 16 GB device with 7.7 GB already taken, can't fit, falls into PCIe streaming.

**Evidence for:**
- `[UnloadAll] llm_polish.unload() called` is a separate explicit call — implies the standard unload doesn't catch it
- The `[BUG-098 tripwire] post-load: linear4bit_count=280 is_loaded_in_4bit=True vram_delta=7.74GiB` tripwire from earlier in the run shows Mistral-Nemo holds 7.74 GiB

**Evidence against:**
- The log shows `LLM unloaded: VRAM allocated=0.03 GiB reserved=0.13 GiB (cpu + gc.collect + empty_cache)` after StoryOrchestrator runs; the unload was confirmed
- This would affect 4-HuMo smoke equally (smoke has no LLM phase, but the cap-fired full-stack runs do)

---

## 9. Surgical fix candidates (ranked, all assume H1+H2 are dominant)

### Fix A — Heavy CUDA reset between Phase B and Phase C (lowest risk, partial)

Replace the current 8.5 cleanup with the BUG-LOCAL-126 hard-reset chain:

```python
import gc, torch
import comfy.model_management as mm

# Existing CPU offload of Phase A/B tensors (keep as-is)

# Heavy reset
mm.unload_all_models()
gc.collect()
torch.cuda.synchronize()
mm.soft_empty_cache(force=True)
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
torch.cuda.synchronize()  # second sync to confirm allocator settled

# Telemetry (optional but cheap)
free_b, total_b = torch.cuda.mem_get_info()
log.info("[BatchHumoRender] Pre-Phase-C VRAM free: %.2f GiB / %.2f GiB",
         free_b / 1024**3, total_b / 1024**3)
```

**Risk:** `torch.cuda.synchronize()` can hang on a bad allocator state (this was the BUG-126 surgical concern). Mitigation: wrap in a thread + 30s timeout; if it hangs, log and proceed.

**Expected impact:** if H1 dominant, this should claw back 30–80% of the slowdown. Won't fix H2 if encoders genuinely re-stage 150 times.

### Fix B — Single-load encoder context managers (real fix for H2)

Pin WanTEModel to GPU once at the start of Phase A, force-pin via `model_management.LoadedModel`, encode all 59 prompts, then explicitly release. Same for Whisper in Phase B.

```python
# Phase A
mm.load_models_gpu([clip], force_full_load=True)
try:
    negative = _call(text_enc, clip=clip, text=_CHINESE_NEGATIVE)[0]
    for entry in plan:
        try:
            entry["positive"] = _call(text_enc, clip=clip, text=entry["pos_text"])[0]
        except Exception as exc:
            log.warning(...)
            entry["positive"] = None
finally:
    mm.unload_all_models()
    torch.cuda.empty_cache()
```

**Risk:** `force_full_load=True` may OOM on the 16 GB device if other models still resident; needs a hard reset before Phase A entry too. Adds API surface dependence on ComfyUI internals (`load_models_gpu` signature could change).

**Expected impact:** if H2 dominant, this collapses 150 `prepared` log lines to 2 (one per encoder). Should also reduce Phase A wall time directly (anecdotally Phase A is currently ~5 min on a large episode; would drop to ~30s).

### Fix C — Hard process boundary via humo_max_lines_per_process + watcher (avoidance)

Set `humo_max_lines_per_process = 6` (or tested empirical safe budget) + `stop_workflow_on_soak_cap = True` + external watcher that re-queues with `resume_from_ledger = True` after the soak cap fires. Each ComfyUI process starts with a fresh Python heap and CUDA pool. Allocator drift can't accumulate.

**Risk:** doubles the model load time per N clips (HuMo + encoders re-load each process restart, ~30s each = 1 min per restart). For a 60-clip episode at cap=6, that's 10 process restarts × 1 min = 10 min overhead. Tolerable.

**Expected impact:** sidesteps the root cause entirely. Production-pragmatic. Doesn't actually fix the underlying allocator issue.

### Fix D — Cap script size (avoidance)

Add a pre-flight in `OTR_LLMScriptWriter` that hard-caps `target_words ≤ 700` and `cast ≤ 5`. The 4-HuMo smoke envelope (4 chunks) is known healthy; tested upper bound from memory is roughly 30 chunks (BUG-LOCAL-081 fix's stated coverage).

**Risk:** narrows OTR's script range; 7-cast space-opera-epic episodes can't be made.

**Expected impact:** zero hangs, but a creative-output ceiling.

### Fix E — Switch to a non-thrashing allocator config (last resort)

Revert to `cudaMallocSync` (set `PYTORCH_CUDA_ALLOC_CONF=backend:native`) or tune `expandable_segments:True` for cudaMallocAsync. Restart ComfyUI Desktop with the env var set.

**Risk:** affects every node, not just HuMo. Could regress LTX or FLUX. Untested on this stack.

---

## 10. Recommended path (Claude's pre-synthesis vote)

Stack **Fix A + Fix B**:
- Fix A is small, safe, testable in isolation against the 4-HuMo smoke; if step time stays at 60s/it and Phase A/B logs drop the spam, we know A was sufficient
- Fix B is the principled fix to the encoder re-staging spam; ship if A alone doesn't restore step time on alien_whispers-class workloads

**Fix C** is shipped already (cap=0 via commit 02a5749 was wrong direction; should set cap=6 and add watcher). We can re-enable as a belt-suspenders backstop.

**Fix D** is the wrong move; OTR's narrative range is the product, not the bug.

**Fix E** is last-resort.

---

## 11. Round-robin questions

To Gemini and ChatGPT, please answer in order:

1. **Hypothesis ranking:** which of H1–H5 do you find best supported by the timing evidence (88× slowdown at 60-line scale, healthy at 4-line scale)? Any hypothesis you'd add that isn't on the list?

2. **150 `prepared` log lines:** is `comfy.model_management.load_models_gpu()` idempotent on an already-resident model in ComfyUI 0.20.x, or does it physically re-stage weights every call? If the latter, that alone could account for some of the slowdown via PCIe saturation; if the former, the spam is misleading and the dominant factor is fragmentation.

3. **Fix A (heavy CUDA reset):** is the proposed reset chain (`unload_all_models → gc.collect → synchronize → soft_empty_cache → empty_cache → ipc_collect → synchronize`) sufficient, redundant, dangerous, or all three? Would `torch.cuda.reset_peak_memory_stats()` add any value? Should we add `torch.cuda.memory._dump_snapshot()` for debugging?

4. **Fix B (single-load encoder context):** is `force_full_load=True` on `model_management.load_models_gpu` a stable ComfyUI-public API? Better path? Any objection to pin-encode-release pattern?

5. **Fix C (process recycling):** at cap=6, watcher restart at every 6th line, 60-clip episode = 10 restarts. Is this within the design intent of `resume_from_ledger=True` + `HumoSoakCapReached` raise pattern (BUG-LOCAL-126)? Any race condition concern with the per-clip ledger save's atomic-write contract from BUG-LOCAL-127?

6. **Mistral-Nemo residency (H5):** can a 4-bit NF4 quantized model held by a non-Comfy Python global escape `mm.unload_all_models()` even after `OTR_UnloadAll.llm_polish.unload()` is called? If yes, what's the surgical detection for "is Mistral-Nemo still on GPU"?

7. **Anything we missed:** orthogonal hypotheses (e.g., kernel cache thrash, FlashAttention-fallback path explosion, Triton autotune cache miss, IPC handle leak, NCCL state, async stream lock contention)?

---

## 12. Appendix: full Phase A/B → Phase C log span (alien_whispers run)

Available on disk at `C:\Users\jeffr\AppData\Roaming\ComfyUI\logs\comfyui.log` (filename consistent across sessions per Cowork memory). Specific markers:

- Run start: `2026-05-08 19:05:57` (got prompt for episode_id=alien_whispers)
- SignalLost video render: `19:42:14 → ~19:45:46` (12,247 frames in 192.3s)
- FLUX bookend + 6 portraits: `19:45:30 → ~19:48:30` (28.6s + 6 × ~26s)
- HuMo Phase A start: `~19:48:30` (60 WanTEModel prepared lines)
- HuMo Phase B start: `~19:50:00` (90 WhisperLargeV3 prepared lines)
- HuMo Phase C start: `~19:55:00` (clip 1 sampling 0/5)
- Clip 1 done: `~03:15:00 next day` (7h 20m later)
- Clip 2 sampling stuck at 4/5 when this writeup was drafted

The full log's "lora key not loaded" warnings (~700 lines) are pre-existing cosmetic noise per BUG-LOCAL-027 / HuMo memory and unrelated.

---

## 13. Disposition

- **Status:** open, gathering second opinions
- **Owner:** Claude (synthesizer) → Jeffrey (decision)
- **Blocking:** v2.0-alpha production release on episodes >5 cast members
- **Workaround in place:** none yet. Current full-episode runs effectively halt at HuMo phase.
- **Delete this doc** once: (1) round-robin synthesized, (2) Fix A/B shipped, (3) alien_whispers-class re-run completes inside 1× wall budget
