# Day 4 of the OTR v2.0 video stack sprint

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

A) **Chain off Day 2 anchor output.** `flux_keyframe` reads `io/hyworld_out/<job>/shot_XXX/render.png` (the Day 2 anchor), extracts Depth or Canny on the fly (e.g. `transformers` depth pipeline or OpenCV Canny), re-renders variations.

B) **Accept `control_image` path from the shotlist.** The LLM script pipeline emits a per-shot `control_image` entry — could be a storyboard sketch, a separate anchor, whatever the script wants.

C) **Both, with A as fallback** when the shotlist doesn't specify.

Which keeps the pipeline clean while staying flexible for when the LLM script eventually wants to inject storyboards?

## What I want back

A short, decisive answer on Q1 and Q2 with reasoning. If there's a third option I'm missing or a FLUX-FP8-on-sm_120 gotcha I should know, flag it. Under 600 words.
