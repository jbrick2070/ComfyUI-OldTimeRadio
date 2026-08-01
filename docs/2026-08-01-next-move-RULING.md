# RULING: fix first, in a specific order. The 120-word campaign is not the next move.

Panel: **codex** `gpt-5.6-sol` (kibitz), **Fable** (structural fan-out), **Sonnet 5 x2**
(instrument value; fix risk). Antigravity produced no file this round. Driver and
sole judge: Claude. Every claim below re-verified against the real files.

**All four reviewers independently said: do not run the 120-word campaign now.**
That unanimity is the finding, not a tiebreak.

---

## THE TWO DISCOVERIES THAT CHANGE THE PLAN

### 1. `fastwan_8gb` carries a contradiction between two of MY OWN decisions (Fable)

    tier cap (otr_g4_fastwan.json)      : 81 frames
    FRAME_COST_MODEL["fastwan_8gb"]     : (7000.0, 185.0)   <- eng_fastwan_8gb.py:296
    per-frame @832x480                  : 60.33 MB
    free VRAM needed to admit 81        : (7000 + 81*60.33)/0.85 = 13,984 MB
    free VRAM MEASURED mid-campaign     : 13,575 MB
    => affordable at that free          : 75 frames

**The cap exceeds what the engine's own cost model affords.** A single-clip beat of
77 or 81 frames refuses with `MotionBudgetError` on the engine this project has been
calling proven. The 45-word episode shipped only because its beats landed under 75 --
it cleared a ~400 MB margin nobody knew existed.

I raised the cap to 81 from the bench (where VRAM is flat) and left the linear cost
model in place. Both decisions were mine and they disagree.

### 2. `ltx_video` is NOT un-root-caused. It is a base-class defect. (codex)

`MotionEngineBase.prepare` returns `{"engine_id", "lease", "patchers"}` --
**`session_ctx` is DROPPED** (`motion_common.py:456-457`). `BeatSession` passes it in
(`beat_session.py:165-167`), and only `eng_wan_ti2v.py:458` and `eng_humo.py:452`
re-add it locally. So `ltx_video`, `ltx_8gb`, `ltx_av` and `wan_i2v` never see
`multi_clip`, `_loop_fill_allowed` enables the boomerang, and the multi-clip
boomerang defect at `eng_ltx_video.py:433-470` fires.

Its regression test fabricates the missing structure instead of driving `BeatSession`
(`tests/test_ltx_boomerang.py:171-177`), which is why the suite never caught it.

This is the repo's own documented failure class: a fix that landed on one adapter and
never reached its siblings. **Zero GPU to diagnose; it was readable all along.**

---

## MY DOC WAS WRONG IN FIVE PLACES

| # | Claim | Verdict |
|---|---|---|
| 1 | "coverage planning is exactly where `wan_ti2v` broke" | **FALSE.** `compute_real_frame_budget` has ONE caller -- `eng_wan_ti2v.py:788`, inside `_floor_length`, the SINGLE-CLIP path. `_planned_length` never consults the predictor |
| 2 | longer episodes produce more coverage-planned segments | **FALSE.** Per-beat frames come from cumulative AUDIO samples (`otr_shot_lock.py:572`), not word count. 120 words = more beats, not longer beats |
| 3 | "2A + 2B share one root" | **FALSE** (codex). Canvas authority and admission arithmetic are independent gates that merely compound |
| 4 | the A/B fork | **CONFLATED** (codex). The g4 campaign profiles are 16 GB / 14.5 GB ceiling; 2C/2D are 8 GB / 6.8 GB. A g4 campaign can neither validate nor repair the 8 GB tier |
| 5 | "Nothing is projected" | contradicted by my own "6-10 h" and "2-4x beat count" estimates |

**And the risk model was inverted.** Coverage-planned segments BYPASS the only
preflight VRAM gate. At 120 words more beats go multi-clip and render up to 177
frames at 1472x832 **with no VRAM protection at all** -- a regime nothing has
measured. More coverage planning means less protection, not more failures.

---

## THE ORDER

### Step 1 -- `ltx_video`, zero GPU
Fix the `session_ctx` ownership so every adapter sees `multi_clip`, not just the two
that re-add it. Add an integration test that drives `BeatSession` instead of
fabricating `prepared`. Also set `sage_attention: false` on the g4 LTX legs: all six
g4 profiles ship `true` and `eng_ltx_video.py:559` calls `assert_sage_not_patched`.
No boot path appears to consume the flag today, so it is probably dormant -- fix it
anyway, it costs nothing.

### Step 2 -- the `fastwan_8gb` cap/cost contradiction
Make the tier cap and the admission model agree. This is a live latent bug in a
shipped engine and it is mine.

### Step 3 -- 2A, declare `wan_ti2v.render_canvas = (832, 480)`
**RISK: LOW.** `fastwan_8gb` already declares this exact value and shipped a
published episode. No driver test asserts a wan_ti2v canvas. **Every profile that
selects wan_ti2v already promises 832x480** in `render.canvas_w/h` -- the 1472x832
renders are the dead-channel defect, not a decision. Canvas is not in the frozen
recipe dict, so the freeze holds.
*Owed:* one canonical 45-word wan_ti2v leg, a profile/declaration drift test, and an
eyeball on the thin pillarbox -- the init still is minted at 1472x832 by
`_landscape_still_dims()` (`otr_meta_brief_image_prompt.py:437-447`), a FOURTH canvas
channel that does not follow the declaration. Note `fastwan_8gb` already has this
mismatch and its episode looked fine.

### Step 4 -- 2B, and NOT YET the coefficients
Both codex and the four-arm SPEC forbid refitting from bench data; the SPEC requires
instrumenting the real `prepare` + `render_clip` path. **Do not choose numbers yet.**
Collect production observations keyed to canvas, requested frames, session lifetime,
baseline/free VRAM and peak across repeats. The telemetry fix landed earlier today
means published manifests already carry per-clip `vram_peak_mb` + `render_canvas`, so
some of that evidence exists on disk -- but not enough to calibrate.

Constraints when the fix does come: scope it to the per-engine rows, never
`_DEFAULT_FRAME_COST`; keep a non-zero conservative slope beyond the tested 81-frame
envelope; keep the refusal mechanism. **The row is wrong in BOTH directions** -- it
under-predicts at 1472x832 (10,145 predicted vs 12,181-12,614 measured) and
over-predicts at 832x480 -- so a naive loosening is wrong in the dangerous direction.
`VramPeakProbe` is telemetry-only; this refusal is the only OOM guard on the path.

*Why the flatness is real:* `vae_temporal=16` bounds the temporal decode window, so
peak is gated by window size rather than total frames. `ltx_8gb` TILED is flat too
(37 MB spread) while UNTILED scales (5024 MB). It is not "some engines scale" -- it is
"untiled scales, tiled-with-a-bounded-window does not."

### Step 5 -- writer-only 120-word probe, CPU-cheap
Verifies the unproven `n_ctx=4096` fix and the writer's behaviour at longer output
before any GPU leg.

### Step 6 -- THEN the 120-word campaign, as a real instrument
Its two genuinely unique unknowns: the real per-beat line-length distribution actual
prose produces, and unattended multi-hour VRAM/ledger stability across many more
model-load cycles. Neither is reachable by a smoke test -- and neither is reachable at
all while step 2 and 3 eat the run.

## DEFERRED, WITH REASON
* **2C** (estimator ignores quant/ctx) -- real, but it only WARNs at the 14.5 ceiling,
  so it does not block the campaign tiers. Its own change.
* **2D** (8 GB tier) -- a different tier at a different ceiling. Even with 2C fixed,
  8.03 GB does not fit 6.8. Needs an operator decision on model or ceiling, and cannot
  be qualified without physical 8 GB hardware.
* **2E** (stateless repair ladder) -- needs a capacity budget first; the
  `ltx_audio_in` capacity failure proves output room is already thin.

## CHEAPER PROBES THAT REPLACE CAMPAIGN LEGS
`scripts/_otr_single_engine_smoke.py --engine wan_ti2v --frames 17/49/81/133` at a
fixed canvas reproduces the flat-VRAM finding **in production code, in minutes** -- a
wan_ti2v leg measures 72.7 s standalone against the 90 minutes it burned in the
campaign.
