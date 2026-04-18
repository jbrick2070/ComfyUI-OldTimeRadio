# Synthesis -- 2026-04-17

**Question:** # Day 4 of the OTR v2.0 video stack sprint

## Platform

- RTX 5080 Laptop, 16 GB VRAM, Blackwell sm_120, single GPU, no cloud
- Windows, Python 3.12, torch 2.10.0, CUDA 13.0, diffusers 0.37.0
- SageAttention + SDPA. Flash Attention 2/3 NOT AVAILABLE. Do not suggest FA chasing.
- VRAM ceiling: 15.5 GB for video sidecars (lifted from 14.5 audio-only)
- Audio byte-identical (C7) must not regress

## Current state

Day 2 shipped `flux_anchor.py`: FLUX.1-dev FP8 (`torch.float8_e4m3fn`) with `enable_model_cpu_offload`, VRAMCoordinator-gated, deterministic per-shot SHA256 seeds, CI stub fallback. Peak ~12 GB on 1024x1024.

Day 3 shipped `pulid_portrait.py`: PuLID-FLUX identity-locked portraits, same harness pattern.

Both run as subprocesses via `multiprocessing.get_context("spawn")`. Characters + refs are per-episode emergent from the LLM script process — there is no fixed character roster.

## Day 4 target

Build `flux_keyframe.py`: scene-keyframe renderer that preserves layout across 3+ prompt variations via ControlNet. Gate: same layout preserved across 3 prompt variations, peak VRAM <= 13.5 GB.

## Two decisions I need your call on

### Q1: Union Pro 2.0 vs. stacked Depth + Canny

`Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0` is a single multi-conditioning ControlNet that accepts a `control_mode` arg at call time (depth / canny / pose / tile / etc.) — one adapter in VRAM, mode-switched per call.

Alternative: load separate Depth ControlNet + Canny ControlNet and stack them with `controlnet_conditioning_scale=[w1, w2]`.

On a 16 GB card already hosting FLUX FP8 + cpu_offload, which is the right default here?

Specifically concerned about:
1. Does Union Pro 2.0 on diffusers 0.37 / torch 2.10 actually work when the base FLUX is in `float8_e4m3fn`, or will it force dequantization and blow past 13.5 GB?
2. Does stacking Depth + Canny push peak VRAM over 13.5 GB?
3. Is Union Pro 2.0's mode switch actually cheap (< 500 ms) or does it re-JIT / re-attach every call?
4. Any known failure modes specific to Blackwell sm_120?

### Q2: Control-image source

Where should the control image come from?

A) **Chain off Day 2 anchor output.** `flux_keyframe` reads `io/visual_out/<job>/shot_XXX/render.png` (the Day 2 anchor), extracts Depth or Canny on the fly (e.g. `transformers` depth pipeline or OpenCV Canny), re-renders variations.

B) **Accept `control_image` path from the shotlist.** The LLM script pipeline emits a per-shot `control_image` entry — could be a storyboard sketch, a separate anchor, whatever the script wants.

C) **Both, with A as fallback** when the shotlist doesn't specify.

Which keeps the pipeline clean while staying flexible for when the LLM script eventually wants to inject storyboards?

## What I want back

A short, decisive answer on Q1 and Q2 with reasoning. If there's a third option I'm missing or a FLUX-FP8-on-sm_120 gotcha I should know, flag it. Under 600 words.

---

## ChatGPT (gpt-5.4)

**Call:**

- **Q1 default:** **Union Pro 2.0 first**, but **use one control mode per render pass** and keep a **Depth-only fallback path**. Do **not** make stacked Depth+Canny the default on this machine.
- **Q2 default:** **C) Both, with A as fallback.** Accept `control_image` from shotlist, otherwise derive from Day 2 anchor.

## Why

### Q1: Union Pro 2.0 vs stacked Depth+Canny

**Pick Union Pro 2.0 as the default** because it gives you the biggest payoff for the smallest change: one adapter loaded, mode switched at inference, simpler harness, less VRAM risk than two separate ControlNets resident together.

My confidence is **moderate**, not absolute:

1. **FP8 compatibility / dequant risk**
   - In diffusers, the **base FLUX transformer** can sit in FP8 while the **ControlNet stays in bf16/fp16**. That does **not automatically imply full dequantization of the base model**.
   - The real risk is not “base explodes to fp16,” but **extra activation / adapter memory** during ControlNet forward.
   - I would **not assume** Union Pro itself runs in float8; treat it as bf16/fp16 sidecar.

2. **Stacked Depth + Canny**
   - On your card, with FLUX FP8 + CPU offload already near ~12 GB at 1024², **two ControlNets is the more likely way to blow past 13.5 GB peak**.
   - Even if weights offload, inference activations and duplicated conditioning paths are the problem.
   - So: **stacking is a “maybe later if profiling proves safe,” not the default.**

3. **Mode switch cost**
   - Union Pro’s `control_mode` is typically just a **runtime branch / embedding selection**, not a reload or recompile.
   - I’d expect it to be **cheap enough** relative to a FLUX denoise pass; **sub-500 ms is plausible**, but I would not promise it on Windows + spawn + first-call warmup.
   - Important distinction: **first call** may pay compile/kernel cache cost; **subsequent calls** should be cheap.

4. **Blackwell sm_120 gotchas**
   - No specific Union-Pro-on-sm_120 blocker I can confidently cite.
   - The likely issues are the same as Day 2/3: **new-kernel path instability, first-run warmup variance, and dtype edge cases** with FP8 + auxiliary modules.
   - So add a **hard fallback**: if Union load/infer fails, drop to **single Depth ControlNet** before giving up.

### Practical recommendation
Implement:
1. `Union Pro 2.0` with `control_mode="depth"` as default.
2. Optional `control_mode="canny"` for edge-heavy shots.
3. **No simultaneous multi-control by default.**
4. Feature-flag stacked controls later only if measured peak stays under budget.

---

## Q2: Control-image source

**Choose C: both, with A as fallback.**

Why:
- **A-only** is clean today, but it hard-codes Day 4 to Day 2 and makes future storyboard injection awkward.
- **B-only** is flexible, but it forces the shotlist/LLM layer to solve a problem you already can solve locally.
- **C** keeps the pipeline clean:
  - If `shot["control_image"]` exists, use it.
  - Else use `io/visual_out/<job>/shot_XXX/render.png`.
  - Then derive depth/canny locally and deterministically.

That preserves your current pipeline while opening the door for storyboard/sketch control later with **zero API break**.

## One extra thing you may be missing

For the gate “same layout across 3 prompt variations,” **Depth is the safer default than Canny**. Canny tends to over-constrain texture/edges and can become brittle when prompts diverge. So my real default is:

- **Union Pro 2.0 + depth**
- `control_image` from shotlist if present, else Day 2 anchor
- fallback to **single dedicated Depth ControlNet** if Union misbehaves on your stack

That is the lowest-risk Day 4 path.

---

## Gemini (gemini-3-pro-preview)

**1. Verdict**
PARTIALLY AGREE: Union Pro 2.0 and Option C are the correct architectural choices, but ChatGPT completely missed the VRAM threat posed by the control-image *preprocessors* (Depth/Canny extraction).

**2. Factual Errors in ChatGPT's Answer**
*   **Union Pro 2.0 Multi-Control:** ChatGPT states "No simultaneous multi-control by default" implying Union Pro can't do it without stacking. Union Pro 2.0 *natively* supports simultaneous multi-control (e.g., Depth + Canny) in a single pass by combining the control embeddings. You don't need two adapters to do multi-control with it.
*   **Mode Switch Mechanics:** ChatGPT calls the mode switch a "runtime branch". It is actually a specific learned embedding vector added to the conditioning based on an integer ID (e.g., Depth = 2). It costs zero extra time (< 5ms), not "sub-500 ms".

**3. Important Omissions**
*   **Preprocessor VRAM Spikes:** Extracting depth requires a model (like DepthAnything V2). If you load DepthAnything into VRAM *while* FLUX FP8 is sitting there, you will blow past 13.5 GB. 
*   **FP8/BF16 Casting Crashes:** In `diffusers`, passing inputs into a `bfloat16` ControlNet attached to a `float8_e4m3fn` base model frequently triggers `RuntimeError: expected scalar type BFloat16 but found Float8_e4m3fn`. 
*   **Caching the Control Image:** If you generate 3 variations, you shouldn't extract the Depth map 3 times. 

**4. My Recommendation**
*   **Use Union Pro 2.0 (bf16) + FLUX (FP8).** It adds ~3.5 GB VRAM. Stacking separate ControlNets is a waste of disk and VRAM.
*   **Strict Preprocessor Sequencing:** In `flux_keyframe.py`, if deriving from the Day 2 anchor, load the Depth estimator, extract the image, save it to disk (`shot_XXX/depth.png`), **`del` the depth model, and call `torch.cuda.empty_cache()`** *before* loading the FLUX pipeline.
*   **Adopt Option C (Both), but read from disk:** The LLM shotlist provides `control_image`. If missing, look for `shot_XXX/depth.png`. If missing, generate it from `render.png`. This keeps the generation loop completely decoupled from the extraction logic.
*   **Explicit Dtype Casting:** Force your control images and ControlNet conditioning scales to `torch.bfloat16` explicitly in the pipeline call to prevent diffusers from downcasting them to match the FP8 base model.
*   **Isolate Audio:** To preserve C7 (byte-identical audio), ensure `flux_keyframe.py` runs in a completely isolated `spawn` context and does not import `torchaudio` or touch the RNG states used by the audio generator.

**5. Uncertainties to Verify**
*   **Diffusers 0.37.0 Union Support:** Shakker-Labs Union Pro 2.0 requires passing a `control_mode` integer array. Verify that `diffusers 0.37.0` `FluxControlNetPipeline` natively accepts the `control_mode` kwarg without requiring a custom pipeline wrapper.
*   **Torch 2.10 / sm_120 / SageAttention:** You are on a bleeding-edge (beta) stack. SageAttention + SDPA on Blackwell might have undocumented kernel panics when routing `bfloat16` ControlNet residuals into `float8_e4m3fn` transformer blocks. If you get CUDA illegal memory accesses, disable SageAttention for the ControlNet blocks specifically.

---

## To decide (Claude / human)

- [ ] Agree:
- [ ] Disagree:
- [ ] Facts to verify:
- [ ] Final grounded recommendation:
