# LTX 2.5 delivery: a fine mesh/tiling texture, and it is NOT the render

**Filed 2026-08-20 by the coder window, at the operator's report. NOT FIXED --
this is a localisation receipt so the fix does not start in the wrong place.**

**Operator's report, verbatim:** *"flciker and tiled outopit"* on
`otr/obs/signal_lost_beneath_the_silvery_boughs_20260820_002734_silent_procgen_blended_captioned_with_credits_final.mp4`,
with a screenshot at **00:37** boxing HELENA's neck/shoulder and labelled
**"tiled"**. And: *"i toghty it loodk good in teh lab so eiter it changed teh
graph and kobs cfg from teh lab to here or ist teh input stilkl res we didn
test"* -- two hypotheses, and **the second one is right.**

## THE ANSWER IN ONE LINE

The LTX 2.5 render is CLEAN. The mesh is introduced downstream, in the
**832x480 -> 1920x1080 upscale/composite stage, which the lab never evaluated.**

## WHAT WAS MEASURED (read-only, no GPU, no re-render)

| stage | resolution | mesh present? |
|---|---|---|
| `shot_b009_character_video_ltx25_video.mp4` (raw render) | 832x480 | **NO -- smooth** |
| `..._silent.mp4` (composite) | 1920x1080 | **YES** |
| `..._silent_procgen_blended.mp4` | 1920x1080 | yes (unchanged) |
| `..._final.mp4` (published) | 1920x1080 | yes (unchanged) |

**The procgen/CRT blend is INNOCENT** -- the mesh is already in `silent.mp4`,
one stage before it. That matters because "CRT scanlines" is the intuitive
suspect and it is the wrong one.

**The comparison is content-matched, not eyeballed across different shots.** The
first attempt compared two different shots and was invalid. The published frame
at t=37 was downscaled to 832x480 and correlated against every raw shot:
`shot_b009` matched at **0.971**. The crop region was then mapped through the
same scale factors (1020,600,520x420 at 1920x1080 -> 225,187 at 442,267 on
832x480) and both were magnified to equal display size.

**Frame-difference sweep over the whole episode:** median inter-frame luma change
**1.48**, mean 2.01, with 20 isolated single-frame spikes -- those are SHOT CUTS,
not flicker. There is no oscillating temporal signal. So of the operator's two
words, **"tiled" is confirmed and "flicker" is not** -- at least not as a
measurable temporal instability in the delivered file.

**Spectral character:** the mesh is a broad quasi-periodic texture in the
**8-14 px** band at 1080p (peaks only ~3.2x median, smeared), NOT a hard seam at
a fixed pitch and NOT h264 macroblocking (which is a rigid 16 px grid).

## WHY HYPOTHESIS 1 (lab -> production drift) IS RULED OUT

* `tests/test_ltx25_recipe_matches_lab_golden.py` **passes** in the full suite run
  taken this session (11237 passed / 114 skipped / 1 xfailed). The recipe is
  pinned to the lab's golden JSON and matches it.
* **The Q3 quant is not drift, it is the lock.** The ledger records
  `"quant": "Q3_K_M"`, and `ltx25_recipe.py:33-36` says Q3 is LOCKED and chosen
  deliberately because the lab measured `Q5_K_M` breaching the clamp at 832x480
  (15.58 GiB). Production is running exactly what the lab settled on.
* The raw 832x480 output is visually clean, which is the direct evidence: if the
  graph or CFG had drifted, the damage would be IN the render.

## WHY THE LAB SAW IT LOOK GOOD

The lab's scope is the 832x480 clip. That is genuinely clean. The delivered file
is **2.31x larger** (1920/832 = 2.308, 1080/480 = 2.25 -- note the aspect also
shifts, 1.733:1 -> 1.778:1, so there is a stretch or crop as well as a scale).
A soft render magnified 2.3x and shown full-screen is a different artifact
regime, and nothing in the lab looked at it.

## WHAT IS STILL OPEN -- do not guess this, measure it

**Which downstream operation adds the texture is NOT yet pinned.** Three
candidates, none eliminated:

1. **Real-ESRGAN x2plus** (`nodes/_otr_upscale_engines/eng_spandrel_esrgan.py`,
   `_model_filename = "RealESRGAN_x2plus.pth"`). ESRGAN-family upscalers are
   known to hallucinate fine cross-hatch on smooth, low-detail regions -- which
   is exactly where this appears (skin), and exactly what it looks like.
2. **The resize/composite filter** -- `_otr_upscale_engines/_pipeline.py:254`
   uses a bicubic `F.interpolate` to land the final size.
3. **The 1080p encode** -- `silent.mp4` is 5.18 Mbit/s at 1080p25.

**A quantitative check was run and it is NOT decisive: fine-band energy is
1213.9 in the production composite vs 948.2 in a bicubic upscale of the raw
render -- 28% higher, consistent with added texture but far from proof.** The
control is imperfect because the two frames are not the same instant of the shot.
Stated as measured rather than dressed up.

**THE LEDGER DOES NOT RECORD WHICH UPSCALE ENGINE RAN.** `meta.post_upscale_blend`
stamps only the source and procgen paths. There is no `upscale_stage` engine
name, model filename or digest anywhere in the ledger, so the question "did
ESRGAN run on this episode?" **cannot be answered from the artifact**. That is a
forensics gap in its own right and it is why this receipt stops here.

## THE CHEAPEST NEXT STEP

Re-composite ONE existing episode's shots twice -- once with `upscale_stage=off`
and once with `spandrel_esrgan` -- and compare the same content-matched crop. No
LTX re-render is needed; the 832x480 shots are already on disk. That is a
minutes-long CPU/GPU job that names the guilty stage outright, and it costs no
story generation.

**Operator constraint that governs any fix here:** `CLAUDE.md` -- *"The recipes
are not on the table."* This finding does NOT propose a recipe change; the LTX
recipe is exonerated. Whatever is adjusted lives in the DELIVERY chain, after the
render, which the recipe lock does not cover.
