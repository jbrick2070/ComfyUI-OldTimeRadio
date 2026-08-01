# Low-VRAM video: tier motion budget + fastwan_8gb wiring (r3 input)

**Branch** `v2.0-alpha`, HEAD `3abf25e7`. No code written yet.
**Predecessors:** r1 (`docs/HANDOFF_2026-08-01-fastwan-8gb-WIRING.md`), r2
(`docs/2026-08-01-fastwan-8gb-r2-KIBITZ-FINAL.md`).

**OPERATOR CONSTRAINT, HARD, GOVERNS EVERY ANSWER:** the operator has delegated the
decision with one condition -- **it must not fail the workflow.** `wan_8gb` is live
with 68 clips across 7 published episodes. Any proposal that risks a regression on the
existing render path loses to a smaller one that does not. Sequencing matters as much
as the change itself. Prefer changes that are provably inert until deliberately
switched on.

**The two shipped video paths that must work:** `wan_8gb` (internal `wan_ti2v`) and the
new `fastwan_8gb`.

---

## 1. THE FINDING THAT REFRAMES THIS ROUND

`config/profiles/otr_8gb_wan.json` pins both:

    video.max_render_frames                = 17
    launch.env.OTR_WAN_TI2V_MAX_FRAMES     = 17

17 is `FRAME_MOTION_FLOOR["wan_ti2v"]` -- the MINIMUM legal render
(`motion_common.py:274`). At 25 fps that is **0.68 s of real motion.** Delivered clips
on the 2026-07-23 episode are 200-381 frames (8.0-15.2 s), so the remainder is
`wrapper_bridge.extend_frames_to_target` ping-pong: a mirror cycle of period `2N-2` =
32 frames = 1.28 s, repeated ~12x for a 15 s beat.

The pin descends from `PBUG-20260723-02`: an 8 GB leg inherited the 177-frame engine
max, asked for a whole beat, and died in the cost model. `f914f0a4` (2026-07-24) pinned
the ceiling to the floor so it can never over-ask again. The cost was nearly all the
motion.

**`native_frame_count` / `extension_mode` (added 2026-07-29, `wan_shared.py:533-573`)
exist to make this visible and are null on every shipped clip because every shipped
render predates them.** No published episode can currently be asked how much of it is
mirrored.

## 2. THE COST MODEL, AND WHY CANVAS IS THE MOTION LEVER

`compute_real_frame_budget` (`motion_common.py:321-363`):

    per_frame_at_res = per_frame * (pixels / _FRAME_COST_REF_PIXELS)   # ref = 1472*832
    affordable       = (free_mb * 0.85 - overhead) / per_frame_at_res

with `FRAME_COST_MODEL["wan_ti2v"] = (overhead 7000.0, per_frame 185.0)`.

| canvas | MB/frame | affordable 4n+1 @ 10 GB free | @ 12 GB | @ 14 GB |
|---|---:|---:|---:|---:|
| 1472x832 | 185.0 | 5 (0.2 s) | 17 (0.7 s) | 25 (1.0 s) |
| **832x480 (today)** | **60.3** | 21 (0.8 s) | 53 (2.1 s) | 81 (3.2 s) |
| 640x384 | 37.1 | 37 (1.5 s) | 85 (3.4 s) | 129 (5.2 s) |
| 512x288 | 22.3 | 65 (2.6 s) | 141 (5.6 s) | 217 (8.7 s) |

Two consequences:

1. **Canvas buys MOTION, not headroom.** 512x288 yields 3.3x the native frames of
   832x480 for the same VRAM. The operator's "render small, upscale later" instinct is
   correct, and this -- not peak VRAM -- is where it pays.
2. **The fixed 7000 MB overhead dominates.** Below ~8.2 GB free the model refuses every
   canvas and the floor (17) wins. On a true 8 GB card the tier gets the floor no
   matter what canvas is chosen. Canvas only helps cards with real headroom.

**FastWan does not add motion.** Same base weights, same canvas, same per-frame
activation; step count does not change peak. The bench measured identical peak delta.
FastWan is a 2.76x THROUGHPUT win, not a motion win, and should be labelled as such.

## 3. SETTLED BY r2 -- do not re-litigate

- `FastWan8gbEngine(WanTi2vEngine)`; sampler to `nodes/_otr_video_engines/dmd_sampler.py`,
  registered in the pack, bench helper's copy DELETED (key collision otherwise).
- Declare `render_canvas` on BOTH adapters -- the reason is boot-independence
  (`OTR_VIDEO_LANDSCAPE_CANVAS` is boot-only), NOT VRAM.
- The incumbent already renders 832x480 (measured, 7/7 clips). The live-vs-bench VRAM
  spread is a measurement-scope change (`2b095143`, 2026-07-20), not canvas.
- `compute_real_frame_budget` never shortens; it RAISES. Ping-pong is driven by the
  tier ceiling, not the predictor.
- Do NOT touch `_FRAME_COST_REF_PIXELS` (calibration reference).
- Nine r2 must-fixes stand, notably: `_vaedecode_inputs` hardcodes `W("ksampler", 0)`;
  recipe accessors read MODULE-level constants a subclass cannot override;
  `_clip_from_raw` cannot carry `quant` / `use_lora` / `render_canvas`; the LoRA patcher
  is untracked and leaks; `ManualSigmas` takes a comma-separated STRING.

## 4. THE PROPOSAL TO ATTACK

**P1 -- un-pin the ceiling (the biggest output-quality win).** 17 is the floor used as
a ceiling, so a 12 GB card running this profile gets 0.68 s of motion when ~2.1 s is
affordable. Replace the fixed pin with a per-tier value matched to the hardware target,
keeping the cost model as the fail-loud backstop it already is.
*Risk to manage:* the pin exists because over-asking KILLED a leg. Any un-pinning must
not resurrect `PBUG-20260723-02`.

**P2 -- lower the low-VRAM canvas to 640x384.** 1.6x the motion of today, a 3.0x
upscale to 1920x1080 instead of 2.3x. Both axes /32-legal (20 x 12).
*Risk to manage:* softness. GO_FORWARD item 9 records 512x288 LTX output as reading
"like a fuzzy mess" at 36.9% of A/C's pixels -- a different model, but a real warning
about aggressive upscale. This is a sharpness-vs-motion taste call.

**P3 -- declare `render_canvas` on both adapters.** Inert today (it equals what already
renders); removes the boot coupling.

**P4 -- build `fastwan_8gb`** per the r2 must-fixes, labelled as the throughput tier.

**P5 -- backfill the three telemetry fields** (`quant`, `use_lora`, `render_canvas`) in
`_clip_from_raw`, operator-authorized, telemetry-only. Note this is a SHARED file both
WAN adapters ride.

## 5. WHAT THIS ROUND MUST ANSWER

### Q1 -- SEQUENCING UNDER THE NO-REGRESSION CONSTRAINT
Order P1-P5 so that no step can break the live `wan_8gb` path, and name for each step
the exact evidence that it did not. Which of these are provably inert until switched
on, and which genuinely change live render behaviour? If any step should be gated
behind a live leg before the next begins, say which and why.

### Q2 -- THE MOTION / SHARPNESS CALL
Is P2 right, wrong, or wrongly scoped? Specifically: does the 7000 MB overhead term
make canvas reduction pointless on the cards that actually need it (true 8 GB), such
that P2 only helps 12 GB+ users who could instead just get P1? Would a per-tier canvas
(different canvas at different VRAM targets) be better than one lowered default, or is
that a proliferation trap? Ground the upscale-quality risk against
`nodes/rtx_upscale.py` rather than asserting it.

### Q3 -- THE CEILING'S REAL SHAPE
What SHOULD `max_render_frames` be as a function of the hardware target, given the cost
model already refuses loudly? Is a static per-tier number right, or should the tier
declare its VRAM target and let the existing model compute the ceiling? Name the
failure mode of whichever you reject, and address `PBUG-20260723-02` explicitly.

### Q4 -- FASTWAN WIRING ORDER
Given the r2 must-fixes touch SHARED files (`wan_shared.py:_clip_from_raw`, the recipe
accessors in `eng_wan_ti2v.py`), what is the safe order to land them so `wan_ti2v` is
never transiently broken? Which changes to the incumbent are behaviour-preserving
refactors versus real behaviour changes, and what pins each one?

### Q5 -- WORKFLOW INTEGRITY
`workflows/otr_canonical.json` is the source of truth; `widgets_values` is POSITIONAL
and only ever appended. Adding an engine to the dropdown is registry-derived, not a
widget edit. State exactly what must change in that file, what must NOT, and the audit
that proves no widget drift (widget count vs live `INPUT_TYPES`, every wired input-name
present, link referential integrity, JSON round-trip).
