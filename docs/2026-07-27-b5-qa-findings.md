# B5 QA record -- the canvas seam, and the design the panel sent back

Two fan-outs, before the code and before the push: 2 Sonnet lenses + 1 agy
(`Gemini 3.6 Flash (High)`) each round, every lens in ONE block. $0 external,
no codex spend. Claude judged.

## THE POST-CODE PANEL SENT THE DESIGN BACK, AND IT WAS RIGHT

B5 was written, green, and mutation-proven with 10 mutants when a seat pointed
at a document I had read and mis-weighted: **`docs/2026-07-26-o1-canvas-arc-judgment.md`
-- one of the three authorities GO_FORWARD names for this very step -- forbids
the design I built.** Verified against the file, not taken on the seat's word:

* Its channel table lists `render.canvas_w/h -> canonical_canvas` as the one
  **DEAD** channel of five.
* Its "WHAT SHIPS" item 2: *"`ltx_8gb` declares its render canvas STATICALLY --
  beside its existing `frame_contract`, `render_aspect` and `target_fps`. Not an
  env var, **not a ledger read**, not a fourth inline branch."*
* Its item 6 gives the profile channel exactly one remaining job: *"A drift
  guard on the dead channel. A test pinning `otr_8gb_ltx.json`'s `canvas_w/h`
  equal to the engine declaration."*

I had built the ledger read -- following the later 8gb judgment's B5 paragraph,
which says to derive from `canonical_canvas` and never reconciles itself with
the seam-specific judgment written the same day. Two authorities disagreed and I
took the wrong one.

**The panel also supplied the case that decides it on the merits, which is why
this is not a coin-flip between two docs.** `tmp/_run_canonical_engine_matrix_20260723.py`
routes `ltx_8gb` onto the CANONICAL 832x480 workflow through profile
`role_overrides` and copies no canvas -- and its author had already special-cased
the WAN sibling for exactly this reason, with the comment *"Applying only the
engine name silently discarded its 832x480/17-frame render contract."* Under the
ledger-reading design that harness inherits a 26:15 canvas and must either
pillarbox or be REFUSED -- a live QA campaign with an open, owed requalification
leg, broken by a gate meant to protect it. Under the declaration it renders at
512x288, which is the whole point.

**A declaration cannot be displaced by where it is pointed.** That sentence is
the design.

## What shipped

1. **`Ltx8gbEngine.render_canvas = (512, 288)`** -- static, beside the frame
   contract and the aspect, with the reasoning in the declaration itself.
2. **`render_driver.declared_render_canvas(engine_id)`** -- pure, no ledger, no
   env, no I/O. Returns `None` for every engine that declares nothing (all of
   them but one), and validates the DECLARATION: positive, /32 on both axes,
   and a genuine `(w, h)` pair. That last check is shape-before-value on
   purpose -- a string is indexable, so `"512x288"` would otherwise parse as
   5x1 and be refused for the wrong reason.
3. **Last in `build_request_from_shot`'s canvas chain**, so nothing can clobber
   it and it clobbers nothing, plus a pre-flight in `render_beat_coverage`
   before `BeatSession` opens.
4. **The drift guard the judgment asked for**: tests pinning both the profile's
   `render.canvas_w/h` and the 8 GB variant workflow's director widgets equal to
   the declaration, so the unconsumed channel cannot silently disagree.

## What the ledger-reading draft got wrong, kept as the lesson

It was fail-closed in the wrong direction. It refused an episode because the
EPISODE's canvas was 26:15, when the engine had a perfectly good canvas of its
own and no need to consult the episode at all. I defended it in the code
comments as "the decided architecture (pillarbox: never)" -- and a seat correctly
answered that the exact-16:9 clause was *"a quality judgment wearing a
structural gate's clothes"*: the render would have completed, the asset would
have existed, the ledger would have stayed usable. What was being refused was
the LOOK of a composite. Under the declaration there is no cross-engine refusal
at all, and the only remaining error is a code-integrity check on a broken
declaration -- the shape of `FrameContract.__post_init__`, not of an audit.

## Also found, and recorded rather than fixed here

- **`render_single` and both HTTP entry points never reach this seam.** They use
  the older `build_request(shot, assets, frame_count, canvas)`, which takes no
  ledger and no engine declaration, and default to `OTR_VIDEO_RENDER_CANVAS`
  (832x480). The O1 judgment already deferred this explicitly ("`render_single`
  parity ... useful, not on the authoritative ledger path"). Consequence worth
  keeping in view: **the 7d-preflight that "proved the GPU" ran through that
  harness at 832x480, not through the production canvas** -- the production
  canvas for `ltx_8gb` has still never been exercised live.
- **The ShotLock WRITE-side validation the O1 judgment also asked for is not
  built.** `otr_shot_lock.py` still stamps `canonical_canvas` unvalidated from a
  possibly-empty policy. With the canvas no longer read from that stamp this is
  no longer load-bearing for the render, which is why it is recorded rather than
  bundled in; the drift guard covers the disagreement that matters today.
- **`docs/ENGINE_MATRIX.md`'s `ltx_8gb` resolution column** reads
  "canvas-negotiated (`_aspect_plan`)". One seat called it stale; the other
  traced the generator and showed the label is produced by a `hasattr(engine,
  "_aspect_plan")` MRO check that B5 does not touch. The second reading is
  correct -- no change -- but the label is unhelpful now that the engine
  declares a fixed canvas, and the generator is the place to fix it.

## Mutation proof

11 mutants: 9 DEFECT all red -- including the resolver answering `None`, the
engine declaring the landscape canvas, the engine declaring nothing, a
stringly-typed declaration slipping the shape check, and the PROFILE drifting
from the declaration -- and 2 CONTROL green. Baseline and restore green,
125 focused tests.

## Two errors of mine the tests caught, worth naming

`512` does not divide `1920`; the scale is 3.75x. My first "zero pad area"
assertion checked divisibility and was simply wrong arithmetic -- the property
that matters is that the two rectangles are the same SHAPE, and it now asserts
`w * 1080 == h * 1920`. And the malformed-declaration check originally ran after
the int conversion, so a string parsed as 5x1 and failed the grid check instead
of the shape check -- right refusal, wrong reason, and a reader debugging it
would have looked at the latent grid.
