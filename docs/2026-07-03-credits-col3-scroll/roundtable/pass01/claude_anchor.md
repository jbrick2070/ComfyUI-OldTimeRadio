# Claude anchor review — R1 (arc / correctness)

## VERDICT: the plan is directionally right but MUST reframe around the real-obs bug.

The console renders correctly; the failure is that COL 3 DOES NOT SCROLL on a real
episode. Grounded on a real obs final (secrets_of_the_vault, 74.6s, 104-word):
col-3 TEXT is static at the top, transcript CLIPPED at the viewport bottom. Multi-frame
col-3 hashes across the tail change only in a ping-pong (t66==t70, t64==t72) = the
LOOPED BACKDROP boomeranging THROUGH the semi-transparent console, NOT the text moving.

## MUST-FIX

1. **CONFIRMED (mechanics): ffmpeg `crop` evaluates x/y ONCE.**
   `render_credits_clip` uses `crop=W:VIEW_H:0:'clip((t-LEAD)*pps,0,scroll_px)'` with NO
   `eval=frame`. ffmpeg crop defaults `eval=init` -> the `t`-expression is evaluated at
   t=0 -> y pinned at 0 -> STATIC. Fix: either add `eval=frame` to crop, or (cleaner)
   drop the crop-scroll and animate the OVERLAY y-expression instead (overlay y IS
   per-frame by default): overlay the full tall scroll canvas and move it up with
   `y='H0 - (t-LEAD)*pps'`, cropped by the overlay's main-input frame. This is why the
   synthetic 60-line test *looked* like it scrolled (huge scroll_px + fade masked it) but
   the real short episode is dead static.

2. **CONFIRMED (model): overflow-only scroll clips short episodes.**
   `scroll_px = canvas_h - view_h`; a short transcript barely overflows -> ~0 scroll ->
   the tail is CLIPPED, not revealed. The OLD credits always rolled (classic:
   content enters from BELOW the viewport, exits the TOP; distance = canvas_h + view_h).
   Operator wants the FULL content visible, nothing clipped, a visible roll regardless of
   length. RECOMMEND: classic always-roll (content starts just below the viewport top or
   fully below, scrolls until its bottom clears the top), duration = (canvas_h + view_h)/pps
   + small holds; a short episode still rolls (just briefly); a long one rolls longer.
   Nothing is ever clipped because every line eventually passes through the viewport.

3. **CONFIRMED (content, operator): SYSTEM belongs IN the col-3 scroll.**
   Move `[ SYSTEM ]` out of static col 1 into the TOP of col3_flow (above STORY SPINE), so
   the scroll = SYSTEM -> STORY SPINE -> FULL TRANSCRIPT -> SOURCE INTERCEPT -> DIAGNOSTIC
   (bottom info). No duplication. Col 1 keeps title / MODELS / [PRODUCTION LEDGER] / footer
   (it has room). All the same details as the old scrolling right panel.

## SHOULD-FIX
- Verification must be a REAL frame-diff on the col-3 TEXT region that excludes backdrop
  motion (mask the backdrop, or diff a text-only render), plus a full-transcript
  line-count-in-canvas assertion (nothing dropped). The current test passed on a synthetic
  clip and MISSED the real bug -- strengthen it.
- Duration must still declare to the credits-aware mux; a long roll must not trip the guard.

## UNVERIFIABLE / verify-at-build
- Exact readable pps for an 800-word transcript (needs an eyeball on a real long roll).
