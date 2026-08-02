# Deep research brief: local video-model limits we cannot settle from our own code

**For an external deep-research pass. Self-contained -- no repo access needed.**

Everything in section 1 is MEASURED on our hardware. Everything in section 3 is a
question about how these MODELS behave, which our own code cannot answer because
the answer lives in the model architectures, their papers, and other people's
measurements. That is what we need researched.

## 0. THE SYSTEM, briefly

A local, offline pipeline generates 1940s-style radio-drama episodes: it writes a
script, casts characters, does text-to-speech, then renders video per dialogue
beat and joins them into an episode. **Every second of audio must get original
generated video** -- no mirroring, no ping-pong/boomerang, no held frames.

When a beat's audio is longer than one render call can produce, we split it into
segments and join them. Two join modes:

* **CHAIN** -- segment N+1 begins on segment N's real last frame (frame-exact).
  Requires the model to accept a true first-frame lock.
* **JUMP** -- segments are independent clips cut together. Used when the model
  only accepts a *reference/identity* image rather than a first-frame lock.

**Hardware: one RTX 5080 Laptop, 16 GB VRAM, Blackwell sm_120, Windows,
PyTorch 2.10 + CUDA 13, SageAttention + SDPA (Flash Attention 2 has no wheel for
this stack). Real-world working ceiling 14.5 GB.** Everything must run locally
and offline; no cloud APIs, no paid services.

## 1. WHAT WE MEASURED (facts, for context)

Reference beat = 442 frames = 17.68 s at 25 fps.

| our name | model file | frame ladder | join | 442-frame split |
|---|---|---|---|---|
| wan_ti2v / wan_8gb | `Wan2.2-TI2V-5B-Q5_K_M.gguf` + `wan2.2_vae` | 17-177 step 4 | chain | `[177,177,93]` |
| fastwan_8gb | same + `Wan2_2_5B_FastWanFullAttn_lora_rank_128_bf16` | 17-177 step 4 | chain | `[81]x5+[45]` under an 81 cap |
| ltx_8gb | LTX 2B class | 9-161 step 8 | chain | 3 segments |
| ltx23_16gb_video | `ltx-2.3-22b-dev` (+ distilled LoRA) | 169-169 step 8 | chain | 3 segments |
| ltx23_16gb_audio_in | `ltx-2.3-22b-dev` + `_audio_vae` + `_video_vae` | 9-497 step 8 | single | **renders 449, shows 442** |
| humo (portrait) | `Wan2_1-HuMo-14B_fp8_e4m3fn_scaled_KJ` + `lightx2v_I2V_14B_480p_cfg_step_distill_rank64` + `whisper_large_v3_fp16` | 33-177 step 4 | **jump** | 3 segments, **480x832** |
| humo_1.7B | `humo_1.7B_fp16` | 33-177 step 4 | jump | 3 segments, 480x832 |
| humo_1.7B_169 | `humo_1.7B_fp16` | 33-177 step 4 | jump | 3 segments, 832x480 |
| **humo_14B_169** | **same 14B file as `humo`** | **33-49 step 4** | jump | **`[49]x7 + [33]x3` = 10 cuts** |

**The measurement that drives most of our questions.** On the FastWan 5B lane at
832x480, peak VRAM was essentially FLAT across clip length:

    17 frames -> 6563.1 MiB      49 frames -> 6531.1 MiB      81 frames -> 6563.1 MiB

An LTX 2B lane with TILED VAE decode was likewise flat (37 MiB spread); the same
lane UNTILED scaled with length (about 5024 MiB spread). Our hypothesis is that
a bounded temporal decode window (we run `vae_temporal=16`) makes decode peak
independent of total frames. **We have NOT measured above 81 frames at any
canvas**, and a separate measurement at 1472x832 gave ~10,720 MiB for just 17
frames.

## 2. THE CONTRADICTION WE MOST NEED SETTLED

`humo` and `humo_14B_169` load the **identical checkpoint**
(`Wan2_1-HuMo-14B_fp8_e4m3fn_scaled_KJ.safetensors`) at the **identical pixel
count**: 480x832 and 832x480 are both 399,360 pixels.

* `humo` (portrait) is **uncapped**, allowed up to 177 frames.
* `humo_14B_169` (landscape) is **capped at 49 frames**, which is what forces a
  17.68-second beat into ten cuts.

The cap's justification in our source claims the 14B fp8 tier "rides ~15.9 GB at
832x480" -- above our own 14.5 GB working ceiling -- and cites an internal
bake-off document **that does not exist in our repository**. So we cannot check
it, and one of these two numbers must be wrong.

## 3. RESEARCH QUESTIONS

### A. HuMo / Wan 2.1 VRAM and frame scaling
1. For **HuMo-14B at fp8** (the Kijai `Wan2_1-HuMo-14B_fp8_e4m3fn_scaled_KJ`
   conversion), what peak VRAM do others report at 480x832 and 832x480, and at
   what clip lengths? Does anyone run it above 49 frames on a 16 GB card?
2. **Does orientation matter at equal pixel count** for these DiT/attention
   video models? Our assumption is that attention cost follows latent sequence
   length, which is identical at 480x832 and 832x480 -- so a cap that differs by
   orientation alone would be an error, not physics. Is that assumption right?
   Any published counterexample (e.g. rotary/positional handling, tiling
   granularity, or a non-square attention window making one orientation costlier)?
3. Is HuMo's frame ladder genuinely **4n+1**, and is 33 a real floor or a local
   superstition? We believe 4n+1 comes from the Wan 2.1 VAE temporal compression.
   Confirm the rule and its origin.

### B. The flat-VRAM hypothesis
4. Is it established that **tiled VAE decode with a bounded temporal window makes
   peak VRAM independent of total frame count** for Wan-family VAEs? We measured
   flat to 81 frames and want to know whether flat-to-81 licenses flat-to-177, or
   whether some other allocation (attention over the full latent sequence,
   conditioning, or the sampler's own buffers) starts to scale beyond a point.
5. Which allocation actually dominates peak in these pipelines -- sampling
   attention, VAE decode, or text/audio conditioning -- and how does each scale
   with frame count?

### C. Audio-driven face models
6. **HuMo takes a `ref_image` rather than a `start_image`.** Confirm: is the
   reference an identity hint only, with no first-frame lock, so consecutive
   clips cannot be frame-chained? Is there any supported way to chain HuMo
   segments continuously?
7. We have a documented defect: **the audio consistently leads the lips by
   100-200 ms**, with the face visibly static for the first 3-6 frames while the
   first phonemes play. Is this a known, reported characteristic of HuMo (or of
   Whisper-conditioned audio-driven face models generally)? **What is the
   community's fix** -- leading-silence padding then trimming the pad frames
   after decode, a conditioning offset, or something else? Any published numbers
   on the correct pad length?
8. When a HuMo clip restarts from the same reference portrait, **does it always
   resume at the reference pose**, or does audio conditioning carry motion phase?
   (This determines whether a 10-segment beat visibly snaps back ten times.)

### D. LTX-2.3 22B
9. What peak VRAM does **`ltx-2.3-22b-dev` with the audio VAE** need for a single
   ~449-frame clip, and at what canvas is that feasible on 16 GB? Ours plans one
   449-frame render and we consider it the largest untested risk in the system.
10. Is LTX-2.3's length ladder genuinely **9 + 8k**, and does the audio-conditioned
    variant share it?

### E. Identity across cuts
11. For jump-cut segments of the SAME character, what is current best practice for
    **identity preservation** in a fully local pipeline -- shared reference image,
    fixed seed, IP-Adapter, PuLID, InstantID, or something newer? We currently
    share one portrait across all segments of a face beat and have no identity
    conditioning at all. What would measurably improve identity stability without
    a cloud service?
12. Is there a local technique for choosing **cut points at inter-word silences**
    rather than blind frame counts, or is that purely our own scheduling problem?

## 4. WHAT WE HAVE ALREADY RULED OUT (do not re-propose)

* **Any mirroring, ping-pong, boomerang, or held/frozen frame** to fill audio.
  Operator ruling, unconditional. Every second must be original video.
* **Flash Attention 2.** No wheel exists for torch 2.10 + CUDA 13 + sm_120 on
  Windows. Do not suggest installing it.
* **Cloud/API services of any kind** for the local lanes. Offline-first, no keys,
  no paid services.
* **Async CUDA streams / queue refactors.** Sequential execution only.
* Simply **refitting our VRAM cost coefficients from a synthetic bench** -- our own
  review rejected that because the bench uses a different execution path than
  production.

## 5. WHAT AN ANSWER LOOKS LIKE

For each question: the finding, the source (paper, model card, issue thread,
benchmark, or repo), how much to trust it, and whether it was measured on
comparable hardware. **A cited measurement on a 16 GB consumer card beats a
confident generalisation.** Where the answer is "nobody has published this," say
so plainly -- that tells us to measure it ourselves, which is expensive but
decisive.

Highest value first: **question 2** (does orientation matter at equal pixel
count) and **question 7** (the lip-sync onset fix). Those two unblock the most
work.
