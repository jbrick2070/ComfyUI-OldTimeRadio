# pass08 finishing -- Claude panelist review (before reading the panel)

VERDICT: CONVERGED -- build-ready, with two sequencing notes a coder
should not have to ask about.

BUILD-BLOCKING: none found on my sweep. Contradiction checks done:
flag name OTR_ENABLE_LTX_AV consistent across sections; chain orders
match eng_humo:99 / eng_ltx_video:70 grounding; role lists consistent
with VIDEO_SLOT_ROLES; tuple deltas (a)/(b) consistently name both
engines for canvas and music-only-plus-sibling for prompts; thresholds
(14500 MB, 10/15 min, 497 frames, 240 chars, 2s pad) each appear once
with one value.

SEQUENCING NOTES (resolve in the ticket text, not the plan):

1. M0/M1 PARALLELISM: M1 is CPU-safe and dark and may start before or
   alongside M0 (the touch list never depends on the M0 sheet), but M2
   MUST consume M0's winning lane + max_frames. The ticket cut should
   make that explicit so a coder window doesn't block M1 on the
   operator's GPU evening.
2. GOLDEN CAPTURE ORDER: the dark-lane semantic-projection goldens
   must be captured from a tree WITHOUT the driver deltas (i.e. the
   fixture-capture commit precedes the render_driver edits in the SAME
   ticket), otherwise the goldens bake the new behavior in and the
   guard proves nothing. One sentence in the M1 ticket.

TICKET CUT (proposal):

- CW-LTXAV-1 "Dark skeleton + contracts" (M1): av_dims + eng_ltx_av
  dark skeleton (no graph) + schemas/role_compat/__init__/registry
  edits + tests test_av_dims/test_video_ltx_av (minus graph-dependent
  cases) + fallout membership edits. Done: suite+Bug Bible green, both
  engines visible-dark in the dropdown, byte-identical untouched.
- CW-LTXAV-2 "Driver wiring + goldens" (M1/M3): capture goldens FIRST,
  then driver deltas a-i + test_ltx_av_driver_wiring + storm-line
  tests. Done: suite green, goldens prove dark-lane bit-identity,
  flag-off degrade test green.
- CW-LTXAV-3 "Graph + lane" (M2, AFTER M0 GO): winning-lane graph in
  the shared core + pre-flight + phasing + silent encode + trim/pad +
  max-frames pin + graph-dependent tests. Done: suite green; M2 gates.
- CW-LTXAV-4 "Live gates" (M4, GPU): forced-lane smoke + master-hash +
  greps + NVML; then M5 docs/Bug Bible row/tracker/parity check.
  Done: acceptance greps all green; operator look-QA package ready.

Sanity: no ticket spans a suite-red intermediate state; CW-1 and CW-2
are independently green; CW-3 is the only M0-gated ticket; CW-4 is
operator/GPU.

SHOULD-CONSIDER (one-liners):
- The M0 operator checklist should open with the launcher gate and
  close with "fill the parity table" so the sheet is complete in one
  sitting.
- Ship notes should state explicitly that INERT-everywhere at M0 still
  pays CW-1/CW-2 (they are harmless dark scaffolding) -- the operator
  decides whether to keep or revert them; recommend keep (zero runtime
  cost, registry-clean).
