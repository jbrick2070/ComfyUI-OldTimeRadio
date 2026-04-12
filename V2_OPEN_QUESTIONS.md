# V2.0 Animation Pipeline — Open Questions
## Round-Robin for Team Review

**Date:** 2026-04-11  
**Status:** Pre-flight checklist before first animated test run  
**Format:** Add your initials + date after any item. Decisions get moved to bottom.

---

## Q1: LTX Checkpoint Compatibility

The workflow currently wires `CheckpointLoaderSimple` for `ltx-2.3-22b-distilled-fp8.safetensors`. The ComfyUI-LTXVideo pack has its own `LowVRAMCheckpointLoader` which adds a `dependencies` input for enforced sequential loading.

**Question:** Should we use `LowVRAMCheckpointLoader` instead of `CheckpointLoaderSimple` for the LTX model? The MemoryBoundary node handles VRAM flush explicitly, so the dependency chain may be redundant — but belt-and-suspenders is safer on 14.5 GB.

> [ ]

---

## Q2: LTX VAE Compatibility with Standard ComfyUI

SceneAnimator calls `ltx_vae.encode(anchor_pixel)` and `ltx_vae.decode(samples)` using the standard ComfyUI VAE abstraction. The LTX-Video VAE has a temporal dimension (compresses 8 frames into 1 latent step).

**Question:** Does `CheckpointLoaderSimple` return a VAE that handles the temporal compression correctly for LTX? Or do we need the LTXVideo pack's custom VAE loader? This needs a live test — encode a single image, check output shape.

**Expected:** `vae.encode(image[1,H,W,3])` should return `[1, 128, 1, H//32, W//32]`.  
**If wrong shape:** We need to swap to the LTXVideo custom loader or add a shape adapter in SceneAnimator.

> [ ]

---

## Q3: Latent Channel Count

SceneAnimator creates latents with `_LTX_LATENT_CHANNELS = 128` then calls `comfy.sample.fix_empty_latent_channels()` which may override this. LTX-Video uses 128 channels. Some ComfyUI versions have `fix_empty_latent_channels` that only handles 4/8/16 channel models.

**Question:** Does `fix_empty_latent_channels` leave 128 channels alone, or does it clobber them? Needs a dry-run with print statements.

> [ ]

---

## Q4: comfy.sample.sample() with 5D Video Latent

The existing `_generate_image` function in v2_preview.py uses `comfy.sample.sample()` for 4D image latents `[B, C, H, W]`. SceneAnimator passes a 5D video latent `[B, C, T, H, W]`.

**Question:** Does `comfy.sample.sample()` handle 5D tensors correctly when paired with an LTX-Video model? The model itself knows about temporal, but does the outer sampling loop (noise schedule, sigmas) work with arbitrary tensor shapes?

**If it does not work:** We need to call the LTXVideo pack's `LTXVBaseSampler` directly instead of `comfy.sample.sample()`. SceneAnimator would instantiate that class and call `sample()` on it, passing the required GUIDER/SAMPLER/SIGMAS/NOISE objects.

> [ ]

---

## Q5: STG Guidance vs Standard CFG

LTX-Video was trained with Skip-Timestep Guidance (STG), and the LTXVideo pack uses custom `STGGuiderAdvanced` nodes. SceneAnimator currently uses standard CFG via `comfy.sample.sample()`.

**Question:** Does standard CFG produce acceptable quality with LTX-Video distilled, or is STG required? Distilled models are generally more forgiving of guidance methods, but quality may degrade.

**Fallback plan:** If CFG output is bad, refactor SceneAnimator to use `STGGuiderAdvanced` from the LTXVideo pack. This requires building the guider + sampler + sigmas + noise chain programmatically. Adds complexity but uses the validated path.

> [ ]

---

## Q6: FPS Stretch Strategy

The config has `output_fps_stretch: false` and `output_fps: 24`. If we generate 65 frames at 25fps (2.6s), we can output at 12fps to get 5.4s — a 2x duration stretch for free.

**Question:** Should the first test run use stretched fps to maximize runtime from minimal generation? Or should we validate at native 25fps first, then stretch later?

> [ ]

---

## Q7: Scene Count from Director

Run #3 showed the Director producing only 1 scene for a 23-line script. The SceneSegmenter forces breaks every 4 lines (if >8 total), so a 23-line script would produce 6 scenes. But the Director's visual_plan only had 1 scene.

**Question:** SceneAnimator uses `SceneSegmenter` output (6 scenes) which overrides the Director's 1-scene plan. Is this correct? Should we trust the Director or the Segmenter?

**Current behavior:** Segmenter chunks dialogue into N scenes. PromptBuilder then tries to pull `visual_prompt` from the Director's visual_plan. If scene count mismatches, it redistributes the Director's prompts across Segmenter scenes.

> [ ]

---

## Q8: MemoryBoundary Placement in Execution Order

ComfyUI executes nodes based on the dependency graph, not their visual position. The MemoryBoundary node receives images from VisualCompositor (node 18) and passes them to SceneAnimator (node 27). The LTX CheckpointLoader (node 26) has NO dependency on MemoryBoundary.

**Question:** Will ComfyUI try to load the LTX checkpoint BEFORE MemoryBoundary fires? If the loader runs first, we get SD3.5 + LTX co-resident = OOM.

**Fix if needed:** Add a `dependencies` input from MemoryBoundary to the LTX checkpoint loader (use LowVRAMCheckpointLoader instead of CheckpointLoaderSimple, which has this input). Or add a dummy link from MemoryBoundary's STRING output to the LTX loader to enforce ordering.

> [ ]

---

## Q9: Fallback Clip

The action plan specifies `input/signal_lost_prerender.mp4` as the fallback on OOM. This file does not currently exist.

**Question:** Should we pre-render a 3-second static-noise clip (like SignalLostVideo style) and commit it to `v2/assets/`? Or just use the black-frame fallback already coded in SceneAnimator?

> [ ]

---

## Q10: Widget Order and workflow JSON

Adding `video_clips_json` as an optional STRING input to ProductionBus changes the widget count. ComfyUI is sensitive to `widgets_values` array length vs actual widget count.

**Question:** Does the updated workflow JSON have the correct widget count for ProductionBus after adding the new input? This needs validation on first ComfyUI load.

> [ ]

---

## Decisions Log

| Date | Decision | Who |
|------|----------|-----|
| | | |

---

*Generated 2026-04-11 — ComfyUI-OldTimeRadio v2.0 animation pipeline pre-flight*
