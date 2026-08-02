# wan_ti2v: the canvas is the defect, the zero-slope hole is the trap under it

Operator triage, 2026-08-02: "(1) fix zero slope (2) canvas smoking gun -- yes
an r2-r4 sounds appropriate." This document is the corrected premise for that
arc. **The previous plan (measure the cost row, then flip topology) was built on
a premise that has since collapsed** -- see section 1.

## 1. THE SMOKING GUN: wan_ti2v RENDERS AT 3x ITS OWN PROFILE'S PIXELS

The 268-minute failure was:

    MotionBudgetError: static frame budget 173 (snapped 173) exceeds the
    cost-model's affordable 24 frames (free=13481 MB, margin=0.85)

Back-solving that 24 gives the answer. `compute_real_frame_budget` computes
`per_frame_at_res = per_frame * (pixels / _FRAME_COST_REF_PIXELS)` where the
reference is `1472*832`:

    affordable = (13481*0.85 - 7000) / per_frame_at_res = 24
    -> per_frame_at_res = 185.8  ->  pixel ratio = 1.004  ->  canvas ~ 1472x832

So the render was priced at **1472x832**, while
`config/profiles/otr_g4_wan_ti2v.json` declares `render.canvas = 832x480`. At
the profile's own canvas the same beat prices at **73 affordable frames**, not
24.

### Why: three channels name this canvas and the campaign profile uses none

`build_request_from_shot` overwrites the canvas to the shared landscape default
(`OTR_VIDEO_LANDSCAPE_CANVAS`, default `1472x832`,
`render_driver.py:2494`), with deliberate per-engine branches after it for
`ltx_video` and `ltx_audio_in` and NONE for `wan_ti2v`. A declared
`render_canvas` is applied LAST and wins -- and:

| engine | `declared_render_canvas` | `launch.env` pin | effective canvas |
|---|---|---|---|
| `fastwan_8gb` | **(832, 480)** | none | 832x480 |
| `ltx_8gb` | (512, 288) | -- | 512x288 |
| `ltx_video` | (832, 480) | -- | 832x480 |
| **`wan_ti2v`** | **None** | **none in `otr_g4_wan_ti2v`** | **1472x832** |
| `wan_ti2v` under `otr_8gb_wan` | None | `832x480` | 832x480 |

**`fastwan_8gb` is a SUBCLASS of `wan_ti2v` with bit-identical base weights, and
it declares the canvas its parent does not.** Its own comment says why, and names
this exact failure class:

> THE CANVAS, DECLARED (kibitz r2/r3) ... The reason to declare is
> BOOT-INDEPENDENCE, not VRAM: the incumbent reaches the same 832x480 only
> through `launch.env.OTR_VIDEO_LANDSCAPE_CANVAS`, which binds only if the
> server was booted with that profile -- the PBUG-20260723-02 dead-channel class.

`otr_8gb_wan` has that env pin. **`otr_g4_wan_ti2v` has no pin on any channel**,
so its `render.canvas: 832x480` is a dead field and the engine silently renders
3.07x the pixels it was configured for.

### What this retracts

Three claims made earlier today are now withdrawn:
* "the cost row overstates per-frame by ~35x" -- that compared a slope measured
  at 832x480 against a row defined at 1472x832. At 832x480 the runtime charges
  60.3 MiB/frame, not 185.
* "the mirror is load-bearing because wan_ti2v is priced at 24 frames" -- it is
  priced at 24 frames only at the WRONG canvas.
* the estimator-fit run in progress was measuring 832x480, which is not the
  canvas this profile actually renders at. **Stopped mid-ladder** rather than
  produce a confidently wrong number.

## 2. THE TRAP UNDERNEATH: a near-zero slope disables the guard entirely

Found independently by BOTH r3 lanes (codex MUST-FIX 3; antigravity OPTIONAL 1).
`compute_real_frame_budget` performs the affordability calculation AND the
refusal entirely inside `if per_frame_at_res > 0` (`motion_common.py:370-380`).

So a measured row with `per_frame == 0` -- which the envelope fit can legally
produce, because it clamps slope at `>= 0` and the low end of the ladder is
nearly flat -- would mean **no refusal ever fires, even when `overhead` alone
exceeds the budget**. The only enforcing guard on the path silently switches off.

This CANNOT fire today (the shipped slope is 185). It becomes live the moment a
measured row lands, which is exactly why it is a PREREQUISITE of the cost-row
commit rather than an emergency of its own.

Fix shape: check fixed-overhead affordability FIRST and unconditionally; only
then treat frame count. Antigravity's alternative -- floor the stored slope at
0.01 instead of 0 -- is a workaround that leaves the structural hole open for
any future row that legitimately measures flat, and it makes the stored row a
lie about the hardware. Prefer the structural fix; the panel should rule.

## 3. THE PROPOSED ORDER (this is what the arc must break)

1. **C1 -- declare `render_canvas = (832, 480)` on `wan_ti2v`.** One line,
   matching what its own subclass already ships, making the profile's stated
   canvas true and boot-independent. Both axes /32-legal (26 x 15).
2. **C2 -- re-measure at the canvas that actually ships.** The stopped run's
   832x480 points become valid the moment C1 lands, because 832x480 IS then the
   render canvas. Without C1 the fit must be run at 1472x832 instead.
3. **C3 -- fix the zero-slope hole** BEFORE any measured row is stored.
4. **C4 -- store the measured row**, normalized to the 1472x832 cost reference
   (dividing the measured 832x480 slope by 0.3261), with both numbers stamped.
5. **C5 -- re-run the leg.** Only if it STILL refuses do the topology flip,
   `assert_frame_affordable`, and mirror deletion become necessary.

## 4. WHAT THE PANEL MUST BREAK

1. **Is C1 safe on the incumbent?** `wan_ti2v`'s RECIPE is frozen. Is its canvas
   part of that freeze? Declaring changes the shipped output resolution of every
   `otr_g4_wan_ti2v` episode from 1472x832 to 832x480 -- the composite scales up
   either way, but is there a quality or aspect regression, and does any test,
   profile or acceptance gate pin the current 1472x832 behaviour?
2. **Does C1 actually reach the render?** `declared_render_canvas` is applied
   last in `build_request_from_shot`, but verify nothing downstream re-derives
   dimensions (`_dims`, `_aspect_plan`, `_render_dims`) and silently restores
   the landscape default.
3. **Is 832x480 RIGHT for this engine, or merely what the profile says?** The
   8 GB profile pins it and `fastwan` declares it -- but `fastwan` is the
   distilled 3-step arm. Is the 30-step incumbent's quality acceptable at 480p,
   or was 1472x832 an intentional quality choice that the profile field
   contradicts?
4. **Does C1 alone fix the leg?** At 832x480, `affordable = 73` and the failing
   beat wanted 173. So a 173-frame single-clip beat STILL refuses. Does that
   mean coverage planning is required regardless -- i.e. C5 always lands -- or
   does the corrected cost row (C4) raise 73 far enough?
5. **The zero-slope fix.** Structural (check overhead first) vs floored slope
   (0.01). Which, and what does the refusal message say when overhead alone is
   unaffordable?
6. **Ordering risk.** C1 changes VRAM demand ~3x. Does any OTHER engine, tier or
   acceptance gate depend on `wan_ti2v` currently rendering at 1472x832?

## CONSTRAINTS

`wan_ti2v`'s sampler recipe does not move. Every second of audio gets video; no
mirror/ping-pong/re-used frames; fail loud. The only workflow JSON is
`workflows/otr_canonical.json`. 16 GB RTX 5080, 14.5 GB real-world ceiling.
100% local. **Do not launch renders or boot a server.**
