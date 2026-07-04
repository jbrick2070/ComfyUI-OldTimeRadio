# OTR credits COL-3 SCROLL — LANDED (kibitz R3 wiring / R4 convergence)

The fix is IMPLEMENTED + committed @ 63ec024f (branch v2.0-alpha). Review the LANDED code
`nodes/otr_credits_roll.py` + `tests/test_credits_roll_spec.py` for wiring/integration residuals.
Full suite 6119/0 + Bug Bible 16 green; visual preview confirms.

## What landed
- **F1 classic always-roll:** `render_credits_clip` pads the scroll canvas view_h transparent TOP+BOTTOM;
  `roll_px = scroll_img.height - view_h = content_h + view_h`; `compute_credits_duration_s(roll_px,
  view_h)` -> LEAD + roll_px/pps + TAIL (ceiling -> speed up). Content enters below the viewport, exits
  above -> ALWAYS rolls, nothing clipped. Proven by `test_col3_text_scrolls_even_for_short_episode`
  (constant gray backdrop so col-3 change = TEXT motion; short fixture still rolls).
- **F2 mechanics:** the invalid `crop ... eval=frame` was REMOVED (this ffmpeg's crop has no eval option
  and evaluates x/y per-frame already; kibitz R2 + the real ffmpeg error confirmed). The crop y-expr
  `clip((t-LEAD)*eff_pps,0,roll_px)` scrolls per-frame.
- **F3 SYSTEM in the scroll:** `[ SYSTEM ]` removed from static col 1; prepended to `col3_flow` as the
  first block; `_scroll_render_ops` renders it (kind in ("system","spine")) using `col_w` directly (not
  `_draw_grid`/`_COL_W_FOR`, which returns col1 width). Col-3 scroll order:
  SYSTEM -> STORY SPINE -> full CLASSIFIED TRANSCRIPT -> SOURCE INTERCEPT -> DIAGNOSTIC.
- **Panel-caught bugs fixed:** `sysd.get("gpu_vram")` -> `sysd.get("vram")` (real total VRAM shows);
  `sysd.get("host")` -> `sysd.get("hostname")` + os (real hostname shows).
- **Tests:** col3_flow order updated (system first); col1-has-no-SYSTEM guard; classic-roll-distance
  guard; the scroll-motion render test; the `eval=frame` string guard was CUT (kibitz R2).

## Invariants held
No-fallback (receipts raise; [SYSTEM]/VRAM soft; story facts omit-if-absent). Silent tail. Declared
duration to the credits-aware mux (node 85). Backdrop = looped last clip. Node I/O + workflow JSON
UNCHANGED (zero JSON delta). UTF-8 no BOM.

## kibitz focus (R3 wiring / R4 convergence)
R3: does moving `[SYSTEM]` out of col1 into col3_flow leave any DANGLING reference (col1 layout math,
the `_draw_grid`/`_COL_W_FOR` path, the durable `sysd` read, any other consumer of the old col1 SYSTEM
block)? Is the classic-roll padding + crop travel arithmetic exactly consistent (roll_px vs padded
height vs crop window)? R4: residual defects / does the scroll-motion test actually fail if the model
regresses to overflow-only; any invariant at risk. If clean, this is convergence.
