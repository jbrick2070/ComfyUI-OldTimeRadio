# MEASUREMENT: `ltx23_16gb_audio_in` peak VRAM vs frame count

**79 samples, production path, harvested 2026-08-02 from logs already on disk.**
Nobody measured anything new for this. The data had been accumulating since
2026-07-23 and had never been aggregated -- which is the same failure that lost
`docs/2026-06-27-humo-bakeoff`, pointed the other way: evidence we had, never
read.

## Provenance -- why this is admissible

The standing ruling forbids refitting `FRAME_COST_MODEL` from BENCH data and
requires instrumenting the real `prepare()` + `render_clip()` lifecycle
(`docs/2026-08-01-fastwan-frame-cap-TWO-STRIKES.md` constraint 1; codex
`gpt-5.6-sol` high, 2026-08-01). **These are that lifecycle.** Every sample is a
headless canonical run through the production adapter, logged by
`VramPeakProbe`, not a stock-node bench graph under the section 0A carve-out.

Source logs (all under `tmp/`):

    gpu_lane_all_models_20260728_060646/server_humo.log
    gpu_lane_all_models_20260728_060646/server_ltx.log
    otr_headless_53109.log
    otr_headless_57283.log
    overnight_comfy_campaign_20260723_012809/server_humo.log
    overnight_comfy_campaign_20260723_012809/server_ltx.log
    overnight_comfy_campaign_20260723_012809/server_ltx_styles.log
    _fastwan_live_gate.log
    _gemma_night.log

Recipe throughout: `ia2v_canonical`, `unet=ltx-2.3-22b-dev-Q3_K_M.gguf`,
`quant=Q3_K_M`, `lora=True`, `canvas=832x480`. **Note the quantization:** our
production path is the Q3_K_M GGUF, not the bf16 `ltx-2.3-22b-dev.safetensors`
(46.1 GB) the external research priced. Any claim about "the 22B model" must say
which one.

`VramPeakProbe` reports MACHINE-WIDE peak during the render window -- it includes
resident state from earlier phases. That is the right quantity for an admission
gate that reads live free VRAM, and the wrong quantity to compare against a
baseline-subtracted bench delta.

## The result: peak does not track frame count

| frames | n | mean MB | max MB |
|---|---:|---:|---:|
| 0-99 | 7 | 13745 | 13905 |
| 100-199 | 21 | 13522 | 14203 |
| 200-299 | 36 | 13743 | 16166 |
| 300-399 | 7 | 14006 | 14455 |
| **400-499** | **8** | **13274** | **13504** |

Overall: **min 12988, mean 13660, max 16166 MB across 79 samples.**

**The longest renders have the LOWEST mean peak of any bucket.** Within a single
run the inversion is unmistakable:

    otr_headless_53109.log:   frames=25  -> 13598 MB
                              frames=497 -> 12999 MB

A 20x longer render, 599 MB CHEAPER. The total spread across every frame count
from 25 to 497 is about 730 MB, which is smaller than the spread between two
runs at the SAME length in different campaigns.

## What this settles

1. **`ltx23_16gb_audio_in` at 497 frames completed on this box** -- 12,999 MB,
   under the 14.5 GB ceiling. The engine this repo called "the largest
   unexploded shell" was fired on 2026-07-28 and logged.

   **Corrected wording (kibitz r3, 2026-08-02):** this document first concluded
   "449 is smaller than 497 and is not a risk", which does not follow from data
   whose central finding is that frame count does NOT predict peak -- the same
   dataset contains 201-frame peaks of 14,576 and 16,166 MB. What the evidence
   supports is narrower and should be stated as: **449 adds no demonstrated
   length-specific risk.** Total OOM safety remains conditional on a correctly
   wired live-free admission check, which this engine does not yet have.
2. **The linear `overhead + per_frame * frames` model does not describe this
   engine on the production path.** A per-frame term fitted to this data is
   indistinguishable from zero, and its sign is not even stable.
3. **This is the third engine to show it** -- `fastwan_8gb` flat at 17/49/81
   (6563/6531/6563 MiB) and tiled `ltx_8gb` flat within 37 MB. What is now
   different is that this one is PRODUCTION-PATH evidence, at 20x the frame
   range, with 79 samples.

## What it does NOT settle

**Peak is driven by resident context, not by length -- and that risk is real.**
Four samples breached the 14.5 GB ceiling, one of them at **16,166 MB**, above
the card's usable total. Every one of them:

    frames=201  peak=14532 MB   overnight_comfy_campaign_20260723/server_ltx.log
    frames=257  peak=14546 MB   overnight_comfy_campaign_20260723/server_ltx_styles.log
    frames=201  peak=14576 MB   overnight_comfy_campaign_20260723/server_ltx_styles.log
    frames=201  peak=16166 MB   overnight_comfy_campaign_20260723/server_ltx_styles.log

**All four are from ONE campaign, and all four are SHORT-to-MID renders.** Not a
single 400+ frame sample breached. So the breach is a property of resident
context, not of the clip being long.

**But the CAUSE of that context is unresolved, and my first answer was wrong.**
This document originally attributed the breaches to BUG-LOCAL-265's cross-phase
residue (writer LLM / Bark / FLUX still resident from earlier phases). Kibitz r3
checked the logs and refuted it: **the breached campaign logged SUCCESSFUL
pre-render residue cleanup with no failed steps, and started from nearly
identical free VRAM -- 14.34-14.36 GiB** (`server_ltx.log:4191`,
`server_ltx_styles.log:2992,4659`). The peaks then rose across CONSECUTIVE
SAME-ENGINE beats, where inter-engine reclaim is deliberately skipped
(`render_driver.py:3572`).

So the surviving candidates are same-engine residency, allocator fragmentation,
or cleanup drift -- not the cross-phase residue that BUG-LOCAL-265 describes.
The resident-context conclusion stands; its mechanism does not. Settling it
requires each sample to carry a cleanup receipt, pre-admission free VRAM, beat
ordinal, previous engine and peak, and the effective model identity.

Recording the correction rather than quietly fixing it: reaching for a
remembered lesson that FIT the shape but not the evidence is the same error as
re-deriving a solved problem, just wearing the opposite costume.

So the admission gate must key on **live free VRAM at the moment of admission**,
which it now does, and NOT on a per-frame price. And a length cap is close to
useless as an OOM guard here: the 201-frame renders were the dangerous ones.

## Recommended cost row -- WITHDRAWN

This document first proposed an overhead-dominated row for `ltx_audio_in` with
the overhead taken from the observed peak distribution. **That is wrong, and it
is wrong in exactly the way the phantom 15.9 GB was wrong: it puts a USED-VRAM
number into a model that compares against FREE VRAM.**

`VramPeakProbe` reports machine-wide VRAM *already in use*.
`compute_real_frame_budget` compares modelled cost against
`live_free_vram * margin`. They are not interchangeable, and substituting one for
the other would refuse known-good renders: the breached campaign reported about
14.35 GiB FREE, which after the 0.85 margin is about 12.2 GiB -- **below even the
12,988 MB minimum peak in this dataset.** Every run would have been refused by a
row fitted from its own successful history.

To calibrate properly, each sample needs total and free VRAM captured immediately
before admission, with incremental demand derived as `peak_used minus
pre_admission_used`. The 79 samples here do not carry that, so they cannot set a
row -- they can only refute the linear shape, which they do decisively.

Two further blockers, both real:

* **The row would not protect this engine anyway.** `ltx_audio_in` is absent from
  `PLANNING_CAP_ENGINES`, and its single-clip path returns through `render_shot`
  before the admission boundary is reached. The guard must cover both paths first.
* **`FRAME_COST_MODEL` is keyed by engine name alone**, while this engine's
  recipe, UNet, quantization, LoRA, reserve and temporal-decode window are all
  env-configurable. A Q3_K_M/IA2V row would be applied silently to a materially
  different stack. A measured row needs a calibration IDENTITY, not a name.

## Reproducibility -- owed

These 79 rows live in `tmp/`, which is swept. The aggregate buckets above survive
in this document; the dataset does not. A machine-readable manifest -- one row per
PLAN/peak pair with source log and line number, configuration identity, frame
count, peak, and the extraction script's hash -- is owed before any row cites
this measurement. Otherwise this becomes the next `docs/2026-06-27-humo-bakeoff`:
a number everyone quotes and nobody can re-derive.

## The durable lesson

This measurement existed for ten days before anyone aggregated it. The matrix now
carries an evidence column naming the receipt behind every cap; a peak
distribution belongs in the same place. **A number that lives only in a log is
one cleanup away from becoming the next `docs/2026-06-27-humo-bakeoff`.**
