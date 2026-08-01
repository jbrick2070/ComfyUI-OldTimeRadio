# fastwan_8gb's frame cap: two fixes, two failures. Panel before the third.

**TWO-STRIKES GATE (CLAUDE.md, hard).** I have set this cap twice and it has failed
twice. A third guess from the same model of the problem is exactly what this rule
exists to stop. Everything below is measured on this box.

---

## THE TWO STRIKES

**Strike 1 -- cap = 17** (from kibitz r4: "ship at the floor, the bench does not
qualify a production cap"). Live canonical run REFUSED:

    fastwan_8gb was handed a coverage-planned segment of 177 frame(s) but this
    tier pins its render ceiling at 17 ... NO FALLBACK

17 is `FRAME_MOTION_FLOOR` -- the motion FLOOR used as a CEILING. A chainable engine
gets coverage-planned segments up to the contract max, and floor==ceiling cannot
serve them.

**Strike 2 -- cap = 81** (the highest rung the four-arm bench MEASURED, where VRAM
was flat: 6563.1 / 6531.1 / 6563.1 MiB at 17 / 49 / 81 @832x480). Fable found the
contradiction:

    tier cap                     : 81 frames
    FRAME_COST_MODEL["fastwan_8gb"] = (7000.0, 185.0)   eng_fastwan_8gb.py:296
    per-frame @832x480           : 60.33 MB
    free needed to admit 81      : (7000 + 81*60.33)/0.85 = 13,984 MB
    free MEASURED mid-campaign   : 13,575 MB
    => affordable there          : 75 frames  ->  highest 4n+1 = 73

So a 77- or 81-frame SINGLE-CLIP beat refuses with `MotionBudgetError` on an engine
that has shipped a published episode. The 45-word leg survived only because its beats
landed under 75 -- a ~400 MB margin nobody knew about.

## WHY I THINK MY MODEL OF THE PROBLEM IS WRONG

The obvious third move is "set the cap to 73." I do not trust it, because the
contradiction is not a bad NUMBER -- it is two authorities disagreeing:

* The cap is STATIC and per-tier (`video.max_render_frames`), chosen from measured
  VRAM.
* `compute_real_frame_budget` (`motion_common.py:338-379`) is DYNAMIC and reads
  LIVE free VRAM through a linear model.

At 12,500 MB free the model affords 57; at 14,500 it affords 85. **Any** static cap
contradicts the model at some free-VRAM level. Picking 73 just moves the knife edge --
it would refuse at 13,000 MB free, which this box also reaches.

Compounding it, the model is wrong in BOTH directions:

* At 1472x832 it predicts 10,145 MB for 17 frames while the campaign measured
  render-phase peaks of 12,181 / 12,614 / 10,774 MB -- it UNDER-predicts.
* At 832x480 it predicts 11,887 MB for 81 frames while the bench measured ~6,563 MiB
  for the whole render -- it OVER-predicts, by more than its own fixed overhead.

And the mechanism for the flatness is known: `vae_temporal=16` bounds the temporal
decode window, so peak is gated by window size, not total frames. `ltx_8gb` TILED is
flat too (37 MB spread); UNTILED scales (5024 MB). It is not "some engines scale" --
it is "untiled scales, tiled-with-a-bounded-window does not." **The linear
`per_frame` term does not describe this engine at all.**

## CONSTRAINTS THE THIRD FIX MUST RESPECT

1. **The standing ruling forbids refitting `FRAME_COST_MODEL` from BENCH data**, and
   the four-arm SPEC requires instrumenting the real `prepare` + `render_clip` path
   first. codex reaffirmed this. So "just fix the coefficients" is not available
   today.
2. `compute_real_frame_budget` raising is the ONLY preflight OOM guard on this path.
   `VramPeakProbe` is telemetry-only ("no ceiling assert -- sampled + logged, never
   enforced"). Loosening it is wrong in the dangerous direction: an in-process CUDA
   OOM corrupts the allocator.
3. The 82-177 frame range is UNMEASURED at any canvas. Flat-to-81 does not license
   flat-to-177.
4. `fastwan_8gb` is in `PLANNING_CAP_ENGINES`, so this cap ALSO narrows the coverage
   planner -- it is not only a render ceiling. Whatever it becomes must serve both.
5. The operator's bar: **every beat's video must cover its audio.** A cap that is too
   low forces mirror-fill or refusal; a cap that is too high refuses via the budget.
   Both fail the bar in opposite directions.

## WHAT I AM ASKING

1. **Is a static per-tier cap the right mechanism at all**, given a dynamic
   live-VRAM predictor sits behind it? If both must exist, which is authoritative,
   and how should they be reconciled so they cannot contradict?
2. **What should `fastwan_8gb`'s cap be TODAY**, and what is the reasoning that makes
   it not-a-guess? Name the free-VRAM reference it is safe at, and say what happens
   below that.
3. **Is there a shape that removes the contradiction entirely** -- e.g. deriving the
   cap from the same function the admission gate uses, or making the gate aware of
   the tiled-decode window -- without refitting coefficients before instrumentation?
4. **What is the smallest change that stops this shipping silently?** A latent
   cap-vs-model contradiction produced a green suite, a published episode, and a
   defect. What guard turns that into a build-time failure?
5. If the honest answer is "instrument first, cap stays conservative until then",
   **say what conservative means numerically** and what the instrumentation run is.

## WHAT I WILL NOT DO WITHOUT A RULING

Set the number to 73 because it happens to fit one measured free-VRAM reading.
