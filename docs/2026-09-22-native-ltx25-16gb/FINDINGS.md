# The native LTX 2.5 lane on 16 GB -- what the night measured

Written 2026-09-22 overnight, from the 5080 and two rented pods. Every number
here comes from a log on disk; where a number is an estimate it says so.

## THE HEADLINE

**The native lane is not slow because its graph is wrong. It is slow because
the weights do not fit alongside the work, so ComfyUI streams them.** The
symptom the operator kept seeing -- VRAM pinned at the ceiling, ~62 W, 100%
"utilization", memory controller near idle -- is a card waiting on transfers,
not a card computing and not a card swapping.

Two independent sightings of the same mechanism:

* **The 4090 pod, mid-episode.** `server_4090.log`: `Model LTXAV prepared for
  dynamic VRAM loading. 20484MB Staged.` then `12%|1/8 [01:03<07:26,
  63.80s/it]` at 23,108 MiB / 72 W / **0% util**. The same int8 weight rendered
  a whole clip in 101.3 s when it was the only thing on that card. With an
  episode's writer, TTS, music and image models resident, one sampling step
  costs 63.8 s.
* **The 5080, isolated.** `decode_only_probe.py` with 11 GB of ballast and
  nothing else -- no DiT, no encoder, no sampler, just the video VAE and a
  synthetic stage-2 latent -- reproduces the exact signature: 15.8 GB, 62 W,
  100% util, 2% memory. The ballast is inert; it only occupies VRAM.

## MEASURED, ON THE 5080 (16 GB, Blackwell)

### Generic single-stage stock LTX 2.5 graph, 832x480, 97 frames, 8 steps

Stock nodes only, no OTR recipe, one sampling pass, written to mp4.

    weight            total    sampler   decode   notes
    mix4x8-13.8       46.1 s    18.7 s   11.4 s   also 55.1 s on a second run
    w4a8-11.7         51.7 s    22.8 s   11.5 s
    nvfp4-12.6        52.1 s    23.1 s   11.7 s
    w4a4-10.46        DID NOT LOAD *AT THE TIME* -- the download was still in
                      flight (7.76 of 10.46 GiB). It is complete and
                      header-valid now; it has simply never been run.

**The three weights are indistinguishable in speed.** The spread between them
is 6.0 s; the spread of mix4x8 against ITSELF across two identical runs is
9.0 s. The noise is larger than the signal, so the 16 GB weight choice is a
quality / size / gating decision, not a speed one. `w4a4` needs re-downloading
before it can be judged at all.

Loaders are not the problem either: `UNETLoader` 1.8-2.0 s, `CLIPLoader` 1.6 s.

### The real two-stage native graph, 1664x960 terminal

Per-node, `ltx25_native_foley_16gb` with mix4x8:

    neg (text encode, CPU)      28.9 s
    pos (text encode, CPU)      29.3 s
    sampler      8 steps        38.6 s   at 832x480
    latent_upscale               1.9 s   <- stage 2 upscale is FREE
    refine_i2v                   0.7 s
    refine_concat                0.3 s
    refine_sampler  3 steps     61.2 s   at 1664x960
    refine_separate              0.3 s
    decode                     400+ s   KILLED, never finished

Going into that decode, 11,458 MB was already allocated on a 15.92 GiB card.

**For comparison, from vram-recipe-lab on this same card:** a GGUF two-stage
render at the same 1664x960 / 97 frames finishes **end to end in 255.4 s**
(`LAB_MINI_REQ_ltx25_two_stage_grid_comparable.md`, peak 15.516 GiB). The
native lane's decode ALONE exceeds the GGUF lane's entire render.

Two sub-findings worth keeping:

* **The refine pass is proportional, not buggy.** 61.2 s for 3 steps at
  1664x960 against 38.6 s for 8 steps at 832x480 is ~20 s/step versus
  ~4.8 s/step -- four times the pixels, four times the per-step cost.
* **The latent upscale is 1.9 s.** The suspicion that stage 2's upscale was
  the problem is measured false.

## THE ONE CLEAR LEVER, NOT YET MEASURED ON THIS LANE

`LTX25_SAMPLER = "euler_ancestral_cfg_pp"` (`ltx25_recipe.py:101`) with all
three CFGs at 1.0 (`:124-126`).

At cfg 1.0 there is nothing for CFG++ to do, but the `_cfg_pp` variant sets
`disable_cfg1_optimization=True` (`comfy/k_diffusion/sampling.py:1352`), which
defeats the `uncond_ = None` shortcut at `comfy/samplers.py:610-614`. So every
step runs TWO forward passes of a 22B model instead of one -- in both stages.

* **No official LTX 2.5 recipe uses `_cfg_pp`** -- not Comfy-Org's template,
  not Lightricks' own graph.
* **OTR's own LTX 2.3 lane already moved off it**
  (`eng_ltx_video.py:551-552`), after a lab A/B on this card measured
  **74.8 s -> 56.5 s**.
* **It is one constant**, and one `KSamplerSelect` feeds both samplers
  (`eng_ltx25.py:1347` and `:1374`), so the change reaches stage 1 and the
  refine pass together.

**Caveat that makes this an ear call, not a suite call:** the same change was
REJECTED on quality on the lab's audio lane -- "the speaker turns toward the
monitor and hides the mouth." Our foley lane is joint-AV, the side where it
lost. `sampler_ab.sh` is written and ready; the verdict is the operator's.

## RULED OUT, WITH RECEIPTS

* **The native lanes are not secretly GGUF.** All 30 node slots of both native
  engines resolve to stock `comfy_extras` / core `nodes` classes. `--> GGUF
  slots: NONE`. The claim in `apple/MACHINES.md:36-38` that they require
  ComfyUI-GGUF is generator drift: `otr_dropdown_matrix.py` assigns
  dependencies at MODULE granularity, so every engine in `eng_ltx25.py`
  inherits the GGUF row.
* **Decode tile geometry is not a time lever.** Ours is already Lightricks'
  own `512/64/64/16`. The lab A/B'd a change to it and measured **+0.8 s on
  one lane and -28.8 s on another** -- inconsistent in direction, i.e. noise.
  It is a real VRAM lever (-0.69 GiB) and "visually indistinguishable".
* **The second stage is already the official one.**
  `LTX25_INGRAPH_UPSCALE_ALLOWED = True` and `LTX25_REFINE_SIGMAS = "0.85,
  0.7250, 0.4219, 0.0"` are Lightricks' three-step schedule digit for digit.
  There is no cheaper proven stage 2 to go find; we have it.

## THE DECODE A/B -- ANSWERED, AND IT IS THE WHOLE STORY

Identical VAE, identical synthetic stage-2 latent (1,128,13,30,52), identical
shipped tiling 512/64/64/16. The ONLY variable is how much VRAM is already
occupied.

    empty card          DECODE OK in  30.0 s     peak 8,080 MB
    11 GB of ballast    killed at    412 s       never finished

**The decode needs 8.1 GB.** The DiT is 12.86 GB. On a 15.92 GiB card that is
21.0 GB of demand against 15.92 GB of card, so ComfyUI streams and the decode
takes at least fourteen times longer -- a floor, since the ballast arm never
finished at all. The ballast is a dead `torch.empty` tensor; it does no work.
It only occupies memory, and that alone reproduces the production symptom
exactly.

So the 400 s decode is not a recipe defect, not tile geometry, and not the
weight. It is residency: the DiT is still holding the card when the VAE needs
it.

THE FIX IS TO RELEASE THE DiT AFTER `refine_sampler` AND BEFORE `decode`. On
this card that is worth roughly 380 seconds per beat. `eng_ltx25.py:1611`
passes `keep={"unet", "modality", self._TERMINAL}`, so `free_after_use` never
drops the unet -- but whether that Python reference is what PINS the weights,
or whether ComfyUI would evict them anyway and something else is holding them,
is the one link still unmeasured. That question is out with a reviewer; it
decides whether the fix is `keep` or an explicit unload call.

## OPEN

1. **int8 on 16 GB.** The operator wants one int8 lane for every tier. 20 GB on
   a 16 GB card is exactly the streaming case above, so it needs measuring, not
   predicting.
2. **The CPU text encoder.** 58.2 s of the 16 GB lane's budget, by choice
   (`_native_te_device = "cpu"`). The generic probe did the same work on GPU in
   9.5 s and still fit in 15.8 GB, because the encoder is freed before the DiT
   loads. Whether that holds in the two-stage graph is unmeasured.
3. **`w4a4` has never been RUN** -- the file is complete at 10.46 GiB; the
   earlier "truncated" note measured a download in progress and is withdrawn.

4. **CURED, and the cure is committed (8069154d).** Two attempts, and the
   first one failed usefully:

       drop "unet" from run_graph(keep=...)   9.61 GB still allocated, decode
                                              still 400+ s. A Python reference
                                              is not what pins weights.
       unload_all_models() + soft_empty_cache()
                                              11,595 MB -> 806 MB freed,
                                              decode 34.7 s, whole render
                                              196.3 s with foley muxed.

   196.3 s against the GGUF lane's 255.4 s at the same geometry on this same
   card. The seam is conditional on free VRAM against the measured 8.1 GB
   need, so a machine with room never pays.

5. **The eviction on 24/32 GB is unmeasured.** The condition can fire there
   too. The 4090's int8 leg ran 101.3 s without it; the same leg with it has
   not been run.

6. **56.1 s -- 29% of the render -- is CPU text encoding.** The generic probe
   did the same work on GPU in 9.5 s. The eviction pattern applies one stage
   earlier: encode on GPU, evict before the DiT loads.

7. **`--cuda-malloc` may fight the dynamic-VRAM allocator.** The launcher
   passes it (`_otr_soak_server_launch.cmd:192`); it sets
   PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync before ComfyUI installs its
   own pluggable allocator, and the interaction is undocumented.

## A CAVEAT ON EVERY NUMBER ABOVE: THE TEXT ENCODER WAS NOT THE LANE'S OWN

Every probe tonight was run with `OTR_LTX25_NATIVE_TE=
gemma4-12b-with-proj-ltx-2.5-nvfp4.safetensors` (10.43 GiB), because that is
what was on disk. The lane DECLARES `gemma4-12b-ltx25-comfy-w4a8.safetensors`
(9.88 GiB), which was not.

The canonical runner's preflight caught it on the first episode attempt and
refused before rendering rather than failing part-way -- which is the behaviour
that rule exists for, and it worked.

WHAT IT DOES AND DOES NOT INVALIDATE. The DiT comparison stands: all four arms
used the SAME encoder, so the bake-off's variable was still only the DiT. What
is NOT established is the absolute wall-clock with the shipped stack, because
the encoder differs in size (10.43 vs 9.88 GiB) and in format.

AND THE FORMAT IS THE PART THAT MATTERS. The on-disk encoder is **nvfp4 --
Blackwell-only**. Retargeting the lane at it would have silently made the
"portable, any Ampere+" 16 GB lane Blackwell-locked through its TEXT ENCODER
while every note about it still said portable. That is the kind of defect that
passes every test and is discovered by someone whose card cannot run it. The
declared w4a8 encoder is being fetched instead; it is ungated (verified
`gated: False` on joeygambino/LTX-2.5-Quantized today).

The 205.3 s figure is therefore provisional until one leg runs on the declared
stack end to end.
