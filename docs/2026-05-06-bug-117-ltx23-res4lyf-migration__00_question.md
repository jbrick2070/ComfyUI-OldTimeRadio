# Question -- 2026-05-06

# Question -- LTX 2.3 + RES4LYF integration into OTR's BatchLTXRender

## Context

OTR's current LTX renderer (`nodes/batch_ltx_render.py`, ~1275 LOC) walks
non-character ledger lines and renders each as an LTX video clip via
ComfyUI nodes called inline through a tiny `_call(name, **kwargs)` helper
that wraps `NODE_CLASS_MAPPINGS[name]()` and dispatches `FUNCTION`.

Today's pipeline (LTX 2B v0.9):

    CheckpointLoaderSimple(ltx-video-2b-v0.9) -> MODEL/CLIP/VAE
    CLIPLoader(t5xxl_fp16, type=ltxv) -> CLIP                     (separate)
    LoadImage -> radio_bookend.png -> IMAGE
    CLIPTextEncode(positive prompt) -> CONDITIONING
    CLIPTextEncode(negative prompt) -> CONDITIONING
    LTXVConditioning(positive, negative, frame_rate=25)
    EmptyLTXVLatentVideo(832x480, 8n+1 frames)
    LTXVImgToVideoConditionOnly(vae, image, latent, strength=0.75)
    RandomNoise(seed)
    CFGGuider(model, positive, negative, cfg=1.0)
    KSamplerSelect("euler")
    SamplerCustomAdvanced(noise, guider, sampler, sigmas=LTX_DISTILLED_SIGMAS,
                          latent_image)
    VAEDecodeTiled(samples, vae, tile_size=512, overlap=64,
                   temporal_size=4096, temporal_overlap=8)
    -> ffmpeg-encode silent .mp4 per line

LTX_DISTILLED_SIGMAS (hardcoded constant, 9 values, 8 sampling steps):

    [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]

Constraints:
- Audio path is C7-byte-identical to v1.5 baseline; this node never touches
  audio. Negative space.
- VRAM ceiling 14.5 GB on RTX 5080 16 GB. HuMo (16.5 GB peak) runs in a
  prior phase with strict teardown (`unload_all_models` + `gc.collect` +
  `cuda.empty_cache` + `cuda.synchronize`). LTX node loads after HuMo
  releases, sequenced via a deliberate STRING-typed DAG edge from HuMo's
  `clips_dir` output to the LTX node's `humo_clips_dir` input.
- All visual generation is INLINE in the ComfyUI graph (not subprocess) -
  the existing pattern works fine for v0.9 (~9 GB) and tonight's smoke
  proved it also works for 2.3 22B BF16 (42 GB on disk, streams from RAM).

## Tonight's empirical results (RTX 5080 16 GB)

1. Stock Lightricks LTX 2.3 workflow (CheckpointLoaderSimple +
   LTXAVTextEncoderLoader + 2x LoraLoaderModelOnly + LTXVImgToVideoConditionOnly
   + ClownSampler_Beta + MultimodalGuider + GuiderParameters + ManualSigmas +
   SamplerCustomAdvanced + LTXVTiledVAEDecode):
   T2V smoke: motion. I2V smoke (radio_bookend.png anchor):
   "perfect subtle zoom in" - clean, no glitching.

2. My minimal smoke (same as #1 but with sampler swapped from
   ClownSampler_Beta to KSamplerSelect("euler") + CFGGuider, holding
   everything else constant including the same 9-value distilled sigma
   schedule via ManualSigmas):
   I2V smoke: motion present but "a lil glitchy" - frame-level temporal
   incoherence, less smooth than #1.

So: euler unlocks motion on LTX 2.3 distilled, but res_2s is visibly
cleaner because the distilled LoRA was trained against that sampler family.

## The migration question

Jeffrey wants the smoother RES4LYF res_2s output in OTR's production
episodes (6+ non-character lines per episode, glitches would compound
across the timeline). I need to integrate ClownSampler_Beta and friends
into BatchLTXRender's existing inline ComfyUI graph pattern.

## Design questions

1. **Engine selector pattern.** Should BatchLTXRender expose an `engine`
   widget with values like `"v0_9_euler" | "v2_3_clownsampler"`, or
   auto-detect from the loaded checkpoint's filename, or just hard-cut to
   2.3+RES4LYF and delete the v0.9 path? Round-trip: which gives the
   safest rollback if the next 2.3 episode regression surprises us?

2. **RES4LYF node call shapes.** `ClownSampler_Beta` from RES4LYF takes
   `guides` and `options` inputs (not standard ComfyUI sampler shape). The
   stock workflow wires:

       ClownSampler_Beta(guides=null, options=null) -> SAMPLER

   Then SamplerCustomAdvanced takes (noise, guider, sampler, sigmas,
   latent_image). The guider is built from MultimodalGuider(model,
   positive, negative, parameters=GuiderParameters(...)) but for
   video-only we shouldn't need MultimodalGuider's audio paths - can we
   substitute plain CFGGuider(model, positive, negative, cfg=1.0) and
   feed the ClownSampler_Beta SAMPLER straight into SamplerCustomAdvanced?

3. **Sigma schedule.** The 9-value distilled schedule worked with
   ClownSampler_Beta in tonight's stock-workflow run. But it wasn't fed
   via OTR's `torch.tensor(LTX_DISTILLED_SIGMAS)` tensor - it came from
   the workflow's ManualSigmas node that takes a comma-separated string.
   The two should be equivalent (both produce torch.tensor[9]
   float32 SIGMAS), but is there any subtlety I'm missing?

4. **Encoder swap.** Today's OTR uses CLIPLoader+t5xxl_fp16. Tonight's
   smoke used LTXAVTextEncoderLoader+gemma_3_12B_it_fp4_mixed. LTX 2.3
   was trained with Gemma. Two questions:
   (a) Does t5xxl produce coherent output with LTX 2.3, or does the
   encoder change have to come along with the model change?
   (b) The FP4 Gemma file caused a "Linear has no attribute weight" crash
   when fed to plain CLIPTextEncode but worked fine after my smoke loaded
   it via LTXAVTextEncoderLoader with full 3-widget config
   ["gemma_3_12B_it_fp4_mixed.safetensors", "ltx-2.3-22b-dev.safetensors",
   "default"]. Can OTR's existing CLIPLoader path handle the Gemma file
   if I just swap its widget, or does the loader node type also have to
   change?

5. **VRAM.** Tonight's stock workflow ran the full 2.3 graph (model +
   2 LoRAs + Gemma encoder + RES4LYF chain + tiled VAE decode) at peak
   ~14.5 GB. OTR adds: per-line teardown (no, kept across loop iterations),
   ledger I/O (negligible), per-episode batch over 6+ lines. Is there a
   reason ClownSampler_Beta would have different VRAM characteristics
   than KSamplerSelect("euler") on the same model? I'm specifically
   worried about samplers that retain extra state across steps.

6. **Backward compat.** OTR's existing `otr_scifi_16gb_full.json`
   workflow JSON has the v0.9 chain wired in. If I add a v2.3 engine
   path inside BatchLTXRender Python, do I also need to update the
   workflow JSON to load 2.3 + LoRAs + Gemma, or can the Python node
   accept the EXISTING v0.9 inputs and detect/upgrade internally? My
   instinct: the workflow JSON has to be edited too (model file widget,
   add LoRA chain, swap encoder) because BatchLTXRender takes
   model/clip/vae as inputs. I'd rather break-then-fix the JSON
   intentionally than have BatchLTXRender silently disagree with what
   the workflow loaded.

## What I'm leaning toward (stake the position so you can challenge)

- Engine selector via env var `OTR_LTX_ENGINE=v2_3` (default), `v0_9` for
  rollback. NOT a workflow widget - widget drift is a recurring OTR bug
  source (BUG-LOCAL-097/113/097). Env var is invisible to ComfyUI's widget
  saver.
- Hard cut workflow JSON to 2.3 chain (CheckpointLoaderSimple ->
  ltx-2.3-22b-dev.safetensors, +2x LoraLoaderModelOnly, swap CLIPLoader
  to LTXAVTextEncoderLoader). Keep v0.9 file on disk for emergency-rollback
  but don't dual-wire the workflow.
- BatchLTXRender's `engine` env-var selector branches to either the
  existing _render_one_line_v0_9() euler path OR a new
  _render_one_line_v2_3() RES4LYF path. The new path:
  - Uses LoRAs already applied to the input MODEL (loaders in the workflow
    chain to BatchLTXRender) - no LoRA calls inside Python.
  - Calls ClownSampler_Beta(guides=null, options=null) for SAMPLER.
  - Keeps CFGGuider (skip MultimodalGuider - we're video-only).
  - Keeps the 9-value LTX_DISTILLED_SIGMAS as a torch tensor (proven
    tonight).
  - Replaces VAEDecodeTiled with LTXVTiledVAEDecode (the 2.3-specific
    variant) since the stock workflow uses it and we don't want to dig
    into whether they're equivalent at 832x480x41f.
- Per-episode regression on `sirens_print` proves: motion is smooth, audio
  C7 byte-identical, VRAM under 14.5 GB peak, wall time per line ~6-10 min.

## Reply format

For each design question above, give:
- Your recommended answer
- Why (the load-bearing reason)
- Specific failure mode if I do the opposite
- Any subtle gotcha I'm missing

End with: "WHAT I'D DO DIFFERENTLY FROM CLAUDE'S LEAN" - bullet list of
disagreements with my proposed approach above.
