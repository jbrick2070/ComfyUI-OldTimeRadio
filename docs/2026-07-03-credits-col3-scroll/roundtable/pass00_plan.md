# OTR credits COLUMN-3 SCROLL — spec to harden

Repo `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio`, branch v2.0-alpha.
Renderer: `nodes/otr_credits_roll.py` (the late terminal node OTR_CreditsRoll, 93->95->85). Design
source: `docs/Credits Overlay - plain.html` + `docs/2026-07-03-credits-enrichment/CREDITS_OVERLAY_BUILD_PLAN.md`.

## What exists today (verified)
- OTR_CreditsRoll renders a 3-column 1920x1080 CRT console over the looped last-clip backdrop, silent
  tail, appended after node 93, feeding the credits-aware mux (node 85) a DECLARED tail duration.
- COL 1 (STATIC): episode-title HERO + "SIGNAL LOST" 50% subtitle; MODELS (image/video-per-role w/
  family suffix / music); `[ PRODUCTION LEDGER ]` (vram/frames/seed/commit/rev); `[ SYSTEM ]`
  (host/cpu/ram/gpu/cuda); footer.
- COL 2 (STATIC): CAST & VOICES (delivered stamps + signature); `[ WRITER / LLM CONFIG ]`; footer.
- COL 3 (SCROLLS): `col3_flow` = [ STORY SPINE ] -> [ CLASSIFIED TRANSCRIPT ] (full dialogue) ->
  ">> SOURCE INTERCEPT" (news, optional) -> ">> DIAGNOSTIC" (seeded, no fabricated numbers).
- Mechanics (VERIFIED working): `render_scroll_canvas` builds a tall RGBA strip; `render_credits_clip`
  ffmpeg-overlays it via `crop=W:VIEW_H:0:'clip((t-LEAD)*pps,0,scroll_px)'`; a 60-line transcript ->
  3204px canvas, scroll_px 2599, dur ~50s, early-vs-late col-3 frames provably differ. Short scripts
  (canvas < viewport) pad to view_h -> static hold. `compute_credits_duration_s` = LEAD(3) +
  scroll_px/pps(60) + TAIL(4), ceiling 120s -> SPEED UP (never truncate).
- No-fallback: receipts RAISE if missing; story facts (spine/news) omit-if-absent; [SYSTEM]/VRAM are
  soft probes ("(unknown)").

## CRITICAL real-obs finding (2026-07-03, grounded on a real render)
Inspecting a REAL obs final (`signal_lost_secrets_of_the_vault_...`, 74.6s, a 104-word episode):
the 3-column console renders perfectly with real data, BUT **col 3 does NOT scroll** -- tail frames at
t=68.6s and t=73.6s are IDENTICAL, the transcript sits static at the TOP, and the last lines are
CLIPPED at the viewport bottom (not rolled up). Two root causes to fix:
1. **Model:** the redesign scrolls only the OVERFLOW (`scroll_px = canvas_h - view_h`); a short
   episode barely overflows -> reads static + clips the tail. The OLD credits always rolled
   (classic credits-roll: content enters from below, exits the top; `(roll_px + viewport_h)/pps`).
   DECISION NEEDED: classic always-roll vs overflow-only-scroll. Operator wants a VISIBLE scroll that
   shows the FULL content (nothing clipped) regardless of length.
2. **Mechanics (suspected):** `render_credits_clip`'s ffmpeg `crop=W:H:0:'clip((t-LEAD)*pps,0,scroll_px)'`
   does NOT set `eval=frame`; ffmpeg crop defaults `eval=init` (evaluate x/y ONCE at t=0) -> y pinned
   at 0 -> STATIC. Must add `eval=frame` (or move the scroll to the `overlay` y-expr, which IS
   per-frame). VERIFY which filter actually animates per-frame.

## Operator change (the reason for this roundtable)
The scrolling COL 3 must carry, IN THE SCROLL, "all the same details as before": **SYSTEM info +
STORY SPINE + FULL dialogue + info at the bottom**. Today `[ SYSTEM ]` lives in the STATIC col 1, not
the scroll. The operator wants the col-3 scroll to read like the OLD credits' scrolling right panel
(which scrolled SYSTEM specs + the transcript). Only col 3 scrolls; cols 1-2 stay static.

## Proposed change (draft — harden this)
1. Move `[ SYSTEM ]` OUT of col 1 and into the TOP of the col-3 scroll (above STORY SPINE). No
   duplication. Col 1 keeps title / MODELS / `[ PRODUCTION LEDGER ]` / footer.
2. Col-3 scroll order: `[ SYSTEM ]` -> `[ STORY SPINE ]` -> `[ CLASSIFIED TRANSCRIPT ]` (full) ->
   `>> SOURCE INTERCEPT` -> `>> DIAGNOSTIC` (bottom info).
3. Keep the verified scroll mechanics + duration model; nothing dropped.

## Open questions for the panel
1. Exact section order + spacing so the col-3 scroll reads as ONE telemetry feed (bracket headers,
   inter-section gaps, whether SYSTEM should be a grid vs single-column in a narrow scrolling column
   ~608px wide).
2. Should `[ SYSTEM ]` ALSO remain in col 1, or move ENTIRELY into the scroll (no duplication)? Col 1
   would gain vertical room if SYSTEM leaves — what fills it, or does it just breathe?
3. Scroll pace (pps=60), LEAD/TAIL holds, and the readability ceiling (120s) before speed-up — right
   for a viewer reading a full 800-word transcript?
4. How to PROVE it (a frame-diff test that early!=late) and that NOTHING is dropped (full transcript
   line-count in the canvas).
5. No-fallback / durable-stamp implications: SYSTEM is a probe (soft); every other scroll field must
   come from the durable ledger or omit — confirm nothing forces a placeholder.

## Deliverable
The final col-3 scroll spec + the exact code deltas in `nodes/otr_credits_roll.py`:
`build_credits_layout` (move SYSTEM into `col3_flow`), `render_scroll_canvas` /
`_scroll_render_ops` (render a SYSTEM grid block at the top of the scroll), and any duration tweak.
