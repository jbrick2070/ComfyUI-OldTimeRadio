# PROBLEM STATEMENT -- pick 1-2 still-to-video models that run under 8 GB VRAM

Written 2026-07-31 for external deep research. Self-contained: you do not need
our repo. **Goal: tell us where we are WRONG, and give us a short answer we can
act on.** We are not looking for a survey.

---

## 1. WHAT WE SHIP AND WHAT THE SLOT IS

`ComfyUI-OldTimeRadio` is a ComfyUI custom-node pack that generates old-time-radio
episodes end to end: script -> cast -> TTS -> still images -> video -> published
MP4. It ships VRAM "tiers" so people with small cards can run it.

The slot in question is the **beat renderer**: a locked still image plus a text
prompt becomes a short moving clip, a few seconds long, one per story beat. Both
of our current 8 GB-tier engines are **image-to-video** adapters -- they animate
a still. Text-only rendering is a fallback, not the product.

We already have engines we are happy with for 16 GB and up, and for audio-driven
talking heads. **Do not redesign our pipeline and do not propose a full
replacement stack.** We want one or two models for the small-card tier.

## 2. HARD CONSTRAINTS -- an answer that violates any of these is useless to us

- **100% local, fully offline.** No cloud APIs, no hosted inference, no API
  keys, no paid services, no phone-home. Weights must be downloadable and run on
  the user's own machine forever.
- **Runs under ComfyUI on Windows.** Either native ComfyUI nodes or a
  well-maintained custom-node pack. A model that only exists as a research repo
  with a bespoke inference script is not shippable for us.
- **Licence: MIT preferred; Apache-2.0 or BSD acceptable.** No non-commercial
  terms, no revenue-threshold clauses, no "open weights" licences with bespoke
  conditions. **Check the licence at BOTH levels** -- the base model AND any
  distilled/quantised/fine-tuned derivative you recommend. They differ often.
  This constraint has already eliminated one otherwise-good candidate for us
  (see section 5), so treat it as load-bearing, not boilerplate.
- **Safe for work.** No models whose primary distribution is NSFW-tuned.
- **VRAM envelope: the target is UNDER 8 GB.** 6-12 GB is the band we will read
  about, but under 8 GB is the decision. Call out separately anything that also
  fits **6 GB**, which is our stretch tier.

## 3. WHAT WE HAVE ALREADY MEASURED -- argue with this data, not with intuition

On 2026-07-31 we ran a twelve-cell bench on an RTX 5080 Laptop (16 GB) with
ComfyUI launched at `--reserve-vram 8`, which makes ComfyUI's model manager
behave as though only ~8 GiB is available. Direct-node graphs, seed 42, the box
reset between every leg, machine-wide NVML sampled throughout, every asset
validated by `ffprobe -count_frames` rather than by container header.

We grade on **peak VRAM delta** (peak minus that cell's own desktop baseline,
which ranged 2147-2222 MiB), against a bar of **7168 MiB** = 8192 minus a 1 GiB
display allowance.

| model | canvas | steps | 17f | 49f | 81f |
|---|---|---:|---|---|---|
| Wan 2.2 TI2V-5B (Q5_K_M GGUF) | 832x480 | 30 | 6568.2 MiB / 76.2 s | 6563.1 / 145.7 s | 6563.1 / 221.5 s |
| Wan 2.2 TI2V-5B (Q5_K_M GGUF) | 512x288 | 30 | 6524.6 / 40.4 s | 6578.5 / 60.3 s | 6486.9 / 81.1 s |
| LTX-Video 0.9.8 distilled 2B | 512x288 | 8 | 6691.1 / 20.4 s | 6755.3 / 15.4 s | 6819.1 / 20.4 s |
| Wan 5B + fp8 scaled text encoder | 832x480 | 30 | 7907.1 / 50.5 s | 7811.1 / 125.8 s | 7715.3 / 201.4 s |

Every row above rendered successfully. The fp8-encoder row is the only one over
the bar.

**Four findings you should treat as established, and may attack with evidence:**

1. **Wan 2.2 TI2V-5B at Q5_K_M fits comfortably**, about 600 MiB under the bar,
   at either canvas.
2. **Frame count is nearly free.** Going 17 -> 81 frames moved Wan's VRAM by
   **-5.1 MiB** -- inside noise. Not "a little"; nothing.
3. **Pixels are nearly free too.** 2.71x fewer pixels (832x480 -> 512x288) moved
   it by -43.6 / +15.4 / -76.2 MiB, straddling zero. Across this range the cost
   is the resident model and essentially nothing else.
4. **The 2B model is not intrinsically faster than the 5B.** At the same canvas,
   Wan is FASTER per iteration than LTX at 17 and 49 frames (0.474 vs 1.74 s/it;
   1.07 vs 1.19) and 18% slower at 81 (1.42 vs 1.20). The apparent 10x gap was
   3.75x fewer sampler steps times 2.71x fewer pixels. **At a fixed canvas the
   5B also uses LESS VRAM than the 2B** -- 166 / 177 / 332 MiB less.

**Two limits on the above, stated plainly so you do not over-read it:** this is a
16 GB card told to reserve 8 GiB, not a physical 8 GB card; and these are
whole-window peaks, not per-stage measurements, so we cannot see whether a
staged load/offload scheme would change the picture.

## 4. THE ACTUAL QUESTION

Given that a permissively-licensed 5B model already fits under 8 GB and is
cheaper and faster per step than the 2B distilled model we also ship:

**A. Is there a better still-to-video model under 8 GB than Wan 2.2 TI2V-5B at
Q5_K_M?** "Better" means, in priority order: (1) MIT/Apache licence at both
levels, (2) fits under 8 GB with headroom, (3) fewer sampler steps for
comparable output quality -- step count is where the wall-clock actually lives,
(4) accepts a start image, (5) motion quality and temporal coherence at 2-7
seconds, 24 fps.

**B. What is the permissively-licensed replacement for the fast lane?** Our
8-step distilled model has a licence we cannot clear (section 5). We want its
speed with a clean licence. Step-distilled / few-step / turbo variants of any
permissive base are the obvious place to look. Name the specific weight file and
its licence, not the family.

**C. What, if anything, works at 6 GB?** Same constraints. A separate, shorter
answer is fine.

**D. Given findings 2 and 3, are we wrong about where the VRAM goes?** If frames
and pixels are both nearly free at this scale, the whole cost is resident
weights plus fixed buffers. Does that match what is known about these
architectures, and does it imply we should be choosing on parameter count and
quantisation alone? If a staged load/offload scheme (encode, free, sample, free,
decode) genuinely changes the ceiling for a model that otherwise would not fit,
say so and cite where it is implemented in ComfyUI today.

## 5. ALREADY RULED OUT -- do not recommend these back to us

- **Wan 2.2 I2V 14B** and other 14B-class models. Ruled out by standing decision,
  not by measurement. Do not propose promoting a 14B model to this tier.
- **The fp8 scaled text encoder route.** We tested it: it cost about 1.25 GiB
  MORE than the GGUF encoder on the identical graph and was the only bench row to
  fail. Refuted, not untested.
- **LTX-2.3.** Rejected earlier in this project.
- **Turbo-GGUF step-distilled weights.** CC BY-NC-SA upstream, and the sampling
  contract is not ordinary few-step KSampler execution.
- **LTX-Video 0.9.8 distilled 2B -- the licence, not the model.** The
  "LTX-Video Open Weights License" carries a revenue-threshold clause that admits
  two readings, so we cannot clear it. The model itself works and is benched
  above. **This is the specific gap question B is asking you to fill.**
- **Anything requiring a paid API, an account, or an internet connection at
  render time.**

## 6. WHAT A USEFUL ANSWER LOOKS LIKE

- **1-2 named recommendations**, ranked, with the exact weight file(s) we would
  download, the repository, and the licence of each file. Not a family name.
- **The ComfyUI path**: which nodes or which custom-node pack loads it, and
  whether that pack is currently maintained.
- **VRAM evidence with its provenance labelled.** Say plainly whether each number
  is (a) measured by you, (b) reported by the model authors, (c) a community blog
  or video, or (d) your estimate. We have been burned by (c) presented as (a). A
  candidate with an honest "unmeasured" is more useful than one with a confident
  unsourced figure.
- **The step count and the canvas** any timing claim was made at. A seconds
  figure without both is meaningless -- see finding 4.
- **Where we are wrong.** If the honest answer is "you already have the right
  model, stop looking," say that. That is a valid and welcome answer.

Do not pad the answer to look thorough. Short and correct wins.
