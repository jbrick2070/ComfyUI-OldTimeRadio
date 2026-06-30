# CODER KICKOFF -- S-A: DELIVERY-QUALITY FLOOR (clip-fill + legibility)

**Branch v2.0-alpha. Video-only. Master audio FROZEN (`test_audio_byte_identical` stays GREEN).
Keep EVERY engine user-selectable -- this is a QUALITY FLOOR, not choice-limiting / not routing.
Suite + Bug Bible + B7 green AND push per green chunk. No workflow-JSON change (composite-internal;
no new widget -- V-11 holds).**

## The bug (reproduced on 3 episodes, 2026-06-29 coverage soak)

`weight_of_the_blueprints_163656`, `steel_against_skin_170522` (+ a 3rd eyeball): the closing
announcer beat is a **murky, barely-moving, blurry portrait**. NOT a routing or missing-image bug:

- The announcer portrait IS present at render: `[portrait_ledger] still_b001/b005 ... recorded via
  ledger['images']`. HuMo got real image data.
- `humo_1.7B` UNDERRUNS the beat: **`[OTR.composite] CLIP UNDERRUN (LOUD): beat shot_b005 engine
  'humo_1.7B' rendered 177 frame(s) for a 434-frame target (41%) -- the composite will HOLD the last
  frame for the rest of the beat ... investigate 'humo_1.7B'.`** 177 = HuMo per-clip frame ceiling
  (`eng_humo.py:61`, `_HUMO_14B_SAFE_RENDER_FRAMES` / `_HUMO_MAX_FRAMES`) vs long 405-434f announcer
  beats. The held static last-frame IS the murky/dead plate.
- Completion gates (obs ships, audio byte-identical) PASS -- a visually broken clip slips through the
  non-visual gates (same class as the duration-gate bug fixed `3991c019`).

## What already exists (do NOT rebuild)

- `nodes/otr_silent_composite.py:243` `_warn_clip_underrun(row, target_n)` -- detects the underrun,
  **warn-only**, with an env `OTR_CLIP_UNDERRUN_FRAC`. Comment cites "the wan_ti2v 17/280 freeze".
- A `loop` ROW FLAG is already honored (`_warn_clip_underrun` EXEMPTS `row.get("loop")` -- "a loop-fill
  row repeats to fill by design ... clip-fill Piece 5"). So a loop-fill path EXISTS; HuMo's underrun
  rows are just not being routed into it.
- `nodes/_otr_video_engines/wrapper_bridge.py` `fit_frames_to_target` (trim/mirror-extend) +
  `motion_common.py` -- the frame-fit helpers.

## Fix (priority order)

1. **CLIP-FILL (primary).** When a real motion-engine clip underruns its beat target, fill to the
   target by **ping-pong / boomerang** (preferred over a hard loop -- a talking head looping back to
   frame 0 jump-cuts) instead of holding the last frame. Route HuMo (and any motion engine that can
   underrun: wan_*, ltx_*) through the existing `loop`/clip-fill path; pick the fill in the composite
   when `frame_count < target` and the engine is a motion family. Seed-keyed/deterministic. The
   underrun warn then only fires for a TRUE no-fill miss.
2. **LEGIBILITY GUARD (catch).** After each generated clip: sharpness = variance-of-Laplacian RATIO
   vs the source still (RELATIVE/catastrophic only -- HuMo 480x832 is inherently softer; an absolute
   "blurrier than source" check would flag every HuMo beat); motion via freezedetect; subject/face
   presence = phase 2 (heavier). On FAIL, deliver the clear source still + subtle parallax/pan via the
   EXISTING humo->still_parallax LOUD durable-restamp chain.
3. **PROVENANCE.** Record `attempted_engine` / `delivered_engine` / `fallback_reason` in the ledger
   (the A2 restamp pattern) -- the dropdown choice is preserved as "attempted".
4. **SECONDARY / forensic (not the cause).** Preserve `ledger['images']` durably
   (`nodes/production_ledger.py` `_merge_with_disk` drops top-level `images`) + stamp per-beat
   `init_image_used` / `init_source`. Aids diagnosis; demote below 1-3.

## Acceptance

- A 30-word HuMo announcer beat shows MOTION across the whole beat -- `ffmpeg freezedetect` shows no
  multi-second frozen tail (today: ~59% of shot_b005 is a held frame).
- Eyeball: a live moving face, not a held murky plate, on `weight_of_the_blueprints` / `steel_against_skin`.
- `test_audio_byte_identical` GREEN; full suite + Bug Bible + B7 green; HEAD==origin after push.
- Every engine still selectable; the legibility fallback is LOUD (logged + ledger restamp), never silent.

## Pointers

- Composite: `nodes/otr_silent_composite.py` (`_warn_clip_underrun` :243, the `loop`/clip-fill path,
  `plan_timeline_segments`). HuMo ceiling: `nodes/_otr_video_engines/eng_humo.py:61`.
- Evidence: `docs/2026-06-29-coverage-soak/server.log` (grep `CLIP UNDERRUN`), `RETEST_LIST.md` B2,
  `GO_FORWARD_PLAN.md` S-A. HuMo phrase-chunking (S-C) attacks the same root from the engine side.
