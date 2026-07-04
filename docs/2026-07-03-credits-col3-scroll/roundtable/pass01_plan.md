# OTR credits COL-3 SCROLL — FIX SPEC (R1 converged; for kibitz R2-R4)

Repo `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio`, branch v2.0-alpha.
File: `nodes/otr_credits_roll.py`. R1 (roundtable: Claude anchor + GPT-5.5 + Gemini 3.1 Pro) CONVERGED
unanimously on the fix below, grounded on a REAL obs render (col-3 static + clipped on a 104-word
episode). kibitz R2 (coding) / R3 (wiring) / R4 (convergence) to harden the implementation.

## The bug (grounded, confirmed)
On a real obs final, COL 3 does NOT scroll: transcript static at the top, tail lines CLIPPED. Two causes:
1. **Model:** `render_credits_clip` scrolled only the OVERFLOW (`scroll_px = canvas_h - view_h`) and
   padded short canvases to view_h -> short episodes barely overflow -> static + clipped.
2. **Mechanics:** the ffmpeg `crop=W:H:0:'clip((t-LEAD)*pps,0,scroll_px)'` had NO `eval=frame`; crop
   defaults `eval=init` (evaluate x/y ONCE at t=0) -> y pinned at 0 -> static. (The looped backdrop
   ghosting through at alpha 225 created a ping-pong hash that masked the static text.)

## The fix (3 parts)

### F1 — CLASSIC always-roll model
- `compute_credits_duration_s(roll_px, view_h, pps)` now takes the FULL roll distance
  `roll_px = content_h + view_h` (content enters from below the viewport, exits above the top; EVERY
  line passes through -> nothing clipped; ALWAYS rolls regardless of length). dur = LEAD(3) +
  roll_px/pps(60) + TAIL(4); over the 120s ceiling -> SPEED UP eff_pps, never truncate. (DONE.)
- `render_credits_clip`: pad the scroll canvas view_h transparent at TOP and BOTTOM
  (padded_h = content_h + 2*view_h); `roll_px = padded_h - view_h = content_h + view_h`; crop window
  (view_h tall) travels y = 0 .. roll_px. Remove the old "pad-to-view_h-if-short / static hold" path.

### F2 — per-frame animation
- Crop filter MUST evaluate per-frame: `crop=w={W}:h={view_h}:x=0:y='{yexpr}':eval=frame` where
  `yexpr = clip((t-LEAD)*eff_pps, 0, roll_px)`. (Alternative considered: animate the overlay
  y-expression instead of crop -- overlay y is per-frame by default -- but overlay does not clip to the
  col-3 viewport, so the padded-canvas + eval=frame crop is the chosen mechanism.)

### F3 — SYSTEM moves INTO the col-3 scroll (operator: "all the same details" in the scroll)
- `build_credits_layout`: REMOVE `[ SYSTEM ]` from `col1` (static) and prepend it to `col3_flow` as the
  FIRST block. Col-3 scroll order becomes: `[ SYSTEM ]` -> `[ STORY SPINE ]` -> `[ CLASSIFIED
  TRANSCRIPT ]` (FULL) -> `>> SOURCE INTERCEPT` -> `>> DIAGNOSTIC` (bottom info). Col 1 keeps title /
  MODELS / [PRODUCTION LEDGER] / footer (it gains room -- let it breathe, no duplication).
- `_scroll_render_ops`: render a `[ SYSTEM ]` bracket header + dim-label/bright-value grid at the top
  of the scroll (host/cpu/ram/gpu/cuda from collect_system_specs, soft probe -> "(unknown)").

## Tests (strengthen -- the current test MISSED the real bug)
- The scroll test must prove the col-3 TEXT moves. Render with a STATIC (constant gray) backdrop so any
  col-3 change = text motion (not backdrop), and assert frame(after LEAD) != frame(mid) for a SHORT
  transcript (proving it ALWAYS rolls now, not just long ones).
- Assert the full transcript line-count is present in the scroll canvas (nothing dropped).
- Assert the crop filter string contains `eval=frame` (guard the mechanical regression).
- Keep: silent tail, declared duration to the credits-aware mux, no-fallback raises, node surface.

## Invariants
No-fallback (receipts raise; [SYSTEM]/VRAM soft; story facts omit-if-absent). Silent tail. Declared
duration to node 85 (credits-aware guard) -- a long roll must not trip it. Backdrop = looped last clip
(BUG-410). UTF-8 no BOM. Suite + Bug Bible + B7 green + push per green chunk.

## kibitz focus
R2 (coding): is the padded-canvas + eval=frame crop correct + the duration math sound? R3 (wiring): does
moving SYSTEM out of col1 into col3_flow touch anything else (col1 layout, tests, the durable read)?
R4: residual defects + does the strengthened test actually catch the static-text regression?
