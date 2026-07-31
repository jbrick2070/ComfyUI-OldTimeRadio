# PROBLEM STATEMENT -- our 8 GB video tier cannot render, and we think we know why

Pasteable, self-contained, code-grounded. Written 2026-07-31 for a cross-check
window (kibitz / roundtable / another model). **Goal: find where we are WRONG.**

---

## WHAT WE SHIP

`ComfyUI-OldTimeRadio` -- a ComfyUI custom-node pack that generates old-time-radio
episodes end to end (script -> cast -> TTS -> images -> video -> published MP4).
It ships VRAM "tiers" as config profiles. The 8 GB tier
(`config/profiles/otr_8gb_wan.json`) routes all three visual roles to engine
`wan_ti2v` (public id `wan_8gb`) = **Wan 2.2 TI2V-5B**.

Weights, actual on-disk sizes:

| file | GiB |
|---|---:|
| `Wan2.2-TI2V-5B-Q5_K_M.gguf` (UNET) | 3.549 |
| `umt5-xxl-encoder-Q5_K_M.gguf` (text encoder) | 3.861 |
| `wan2.2_vae.safetensors` | 1.313 |
| **total** | **8.722** |

Target card: generic 8 GB consumer NVIDIA (8,192 MB).

## THE SYMPTOM

An 8 GB leg dies with `MotionBudgetError` BEFORE rendering a frame. Recorded
production failure 2026-07-23, leg `wan_8gb__lumina_image__media_archive`.

## THE CODE THAT REFUSES

`nodes/_otr_video_engines/motion_common.py`:

```python
_FRAME_COST_REF_PIXELS = 1472 * 832            # 1,224,704
FRAME_COST_MODEL = {"wan_ti2v": (7000.0, 185.0)}   # (overhead_mb, per_frame_mb)
_BUDGET_MARGIN = 0.85
FRAME_MOTION_FLOOR = {"wan_ti2v": 17}

# compute_real_frame_budget(...)
per_frame_at_res = per_frame * (pixels / _FRAME_COST_REF_PIXELS)
budget_mb  = free_vram_mb * margin
affordable = int((budget_mb - overhead) / per_frame_at_res)
if affordable < snapped:
    raise MotionBudgetError(...)      # no silent resize, by design
```

The provenance of `(7000.0, 185.0)` is ONE code comment, the only one of its kind
in the repo (`motion_common.py:253-254`):

```
#: Telemetry reference resolution the per-frame cost is measured at (wan_ti2v
#: render-phase peak 10277 MB @ 17 frames @ 1472x832 -> 7000 + 185*17 ~= 10145).
```

That single 2026-06 reading was taken on a 16 GB RTX 5080, and `VramPeakProbe`
samples MACHINE-WIDE NVML, so it includes whatever else was resident.

## THE ARITHMETIC THAT MAKES 8 GB IMPOSSIBLE

At the intended 832x480: `pixels = 399,360`; ratio `= 0.3261`;
`per_frame_at_res = 185 * 0.3261 = 60.33 MB`.

Required FREE VRAM `= (7000 + 60.33*n) / 0.85`:

| frames | required |
|---:|---:|
| 1 | 8,306 MB |
| 17 (our motion floor) | **9,442 MB** |
| 33 | 10,647 MB |

The overhead term ALONE is `7000 / 0.85 = 8,235 MB` -- more than an 8 GB card
has in total. **The tier can never render, at any length, by construction.**

## OUR THREE CLAIMS (attack these)

**CLAIM 1 -- the estimator is the wrong SHAPE, not a wrong constant.**
`overhead + per_frame*frames` is a CO-RESIDENT model. The real low-VRAM
technique is STAGED (encode -> release -> sample -> release -> decode), so peak
should be a max over stages, not a sum:

    peak = max over s( W_s + L_s + A_s + WS_s + S_s + D_s ) + R

(resident weights, live latents, activations, backend workspace, dequant
scratch, allocator slack, safety reserve). Our model cannot express staging at
all, so it cannot model the mechanism that makes 8 GB work.

Supporting evidence: ComfyUI's official Wan 2.2 workflow totals ~18 GB of model
files and its docs say it *"should fit well on 8GB vram with the ComfyUI native
offloading"* (https://docs.comfy.org/tutorials/video/wan/wan2_2).

**CLAIM 2 -- our GGUF opts OUT of the mechanism that makes that true.**
ComfyUI v0.16.0 (5 Mar 2026) made Dynamic VRAM (`comfy-aimdo`) default-on: a
custom allocator that streams weights JIT from pinned host memory. But
`ComfyUI-GGUF` defines `class GGUFModelPatcher(comfy.model_patcher.ModelPatcher)`
-- NOT `ModelPatcherDynamic` -- and `clone()` force-reassigns `__class__` back
(https://raw.githubusercontent.com/city96/ComfyUI-GGUF/main/nodes.py L35).
Confirmed as a known limitation in Comfy-Org/ComfyUI#13953 (18 May 2026).

We ship Q5_K_M GGUF for BOTH the UNET and the text encoder. The official 8 GB
workflow uses **fp8 scaled safetensors**. So we believe it fits *because* it is
safetensors, and that our stack is on the legacy 2025 path.

Related: ComfyUI's `comfy/cli_args.py` says `--lowvram` *"Doesn't do anything if
dynamic vram is enabled"*, and `text_encoder_device()` short-circuits on
`aimdo_enabled` and returns the GPU regardless.

**CLAIM 3 -- the engine never declares its canvas, so it renders 3x too big.**
Our `WanTi2vEngine` has NO `render_canvas` attribute. The sibling `ltx_8gb`
declares `render_canvas = (512, 288)` statically. Without it,
`render_driver.py:2494` falls through to
`os.environ.get("OTR_VIDEO_LANDSCAPE_CANVAS", "1472x832")`. The profile's
`render.canvas_w/h = 832x480` is read by NOTHING on the render path, and the
`launch.env` twin only binds if the server was booted with it -- which a
submitted-to-a-running-server production leg never guarantees. So a plain
canonical run renders at 1,224,704 px instead of 399,360 -- **3.07x**.

## WHAT WE ARE LEAST SURE ABOUT -- please attack these hardest

1. Is CLAIM 2 actually true in the way we think? We verified the *mechanism* in
   source but have NO benchmark isolating GGUF-vs-fp8 under Dynamic VRAM on an
   8 GB card. Is there a path where GGUF still gets streamed (DisTorch2 /
   ComfyUI-MultiGPU, block swap, something newer)? Would moving to
   `umt5_xxl_fp8_e4m3fn_scaled.safetensors` + fp8 UNET actually help, or would
   the larger fp8 weights lose more than the dynamic patcher gains?
2. Is `max(stages) + reserve` sufficient, or do overlapping stages (prefetch,
   async offload with 2 CUDA streams by default) mean two stages can be resident
   simultaneously and we need a pairwise max?
3. What is the honest floor for Wan 2.2 TI2V-5B on 8 GB -- resolution, frames,
   steps? We could not find a single published `max_memory_allocated` figure for
   this model on an 8 GB card anywhere. Does one exist?
4. Should we keep the 5B at all? Two independent 8 GB reports have an offloaded
   Wan 2.2 **14B** Q4 + Lightning LoRA (4-6 steps) beating the 5B on BOTH quality
   and wall clock -- because no Lightning/step-distill LoRA exists for the 5B
   (all `lightx2v/Wan2.2-Lightning` variants are A14B). Is a 14B+LoRA 8 GB tier
   actually more sensible than a 5B tier?
5. Is there a better 2026 model for this slot that we dismissed? We rejected
   LTX-2.3 (22B + Gemma-3-12B encoder, ~20 GB weights, ComfyUI rates its
   activation factor 5.5 vs Wan's ~1.38), HunyuanVideo-1.5 (VAE decode OOMs a
   12 GB card at 121 frames), Motif-Video 2B (its own card says ComfyUI needs
   High-VRAM mode), and MobileWan (no GGUF, no ComfyUI node, Qualcomm RAI
   licence). Wan 2.5/2.6/2.7 are NOT open weights -- verified against the
   Wan-AI HF org. Did we miss something?
6. We plan to cache text embeddings (`WanVideoTextEncodeCached`,
   `use_disk_cache=True` by default) to remove the 3.861 GiB encoder -- which is
   LARGER than the 3.549 GiB UNET -- from the budget entirely. Our prompt set is
   bounded and repetitive across episodes. Any reason this is a bad idea, or a
   better first-class way to do it?

## CONSTRAINTS THAT ARE NOT NEGOTIABLE

- **No silent degrade.** An engine that cannot honour a request must RAISE with a
  named error. We do not clamp, resize, or substitute. A "null engine" that
  returns silence instead of refusing is banned.
- **No fallbacks.** No model, device, or provider substitution.
- 100% local, open source, offline-first. Licence must suit a shipped node pack
  (Apache-2.0 preferred; research/RAI licences are a problem).
- Windows, RTX 5080 laptop 16 GB is the DEV box -- the 8 GB target is our users'
  hardware, which we do not have. So any recommendation must be verifiable by
  measurement we can actually run, or clearly labelled as unverifiable here.
- Episode length is never a pass/fail gate.

## WHAT WOULD SETTLE IT

A 4-cell clamped sweep (`OTR_HEADLESS_RESERVE_VRAM_GB=8`) at the declared canvas,
copying the shape that produced trustworthy numbers for our `ltx_8gb` tier:

| cell | text encoder placement | tiled VAE |
|---|---|---|
| A | cpu | off |
| B | gpu | off |
| C | cpu | on |
| D | gpu | on |

judged on the SPREAD across clip lengths, not the minimum -- an 8 GB tier needs a
ceiling a long beat cannot grow through. Then re-fit the cost model from real
`(stage, frames, peak_mb)` triples.

**Tell us what is wrong with the above, what we have missed, and whether the
sweep as designed would actually answer the question.**
