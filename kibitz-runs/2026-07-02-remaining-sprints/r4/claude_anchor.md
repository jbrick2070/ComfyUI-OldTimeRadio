# Claude anchor review -- r4 (convergence / residual defects)

VERDICT: yes. The r2+r3-hardened plan is build-ready. No new must-fix found on
this pass; residuals are the carried verify-at-build items, which are correctly
labeled as build-time checks rather than plan gaps.

Convergence check against the code (all previously grounded this session):
- Sprint A cites match the real render_driver / otr_video_soak / fallback.py /
  test-consumer sites; internal ordering (A4 -> A3b -> A2 -> A1 -> A3rest)
  eliminates the mixed-state window.
- Sprint B pins match the ImageEngine protocol, dispatcher _coerce_pixels
  contract, CAPABILITIES invariant, and the estimated_usd bridge behavior.
- Sprint D pins match the voice seam (AUDIO dict return), the selector's
  no-hidden-auth row, and the _LEGACY_FIRST_ENGINES dropdown surface.
- Sprint E carrier design covers both request builders; the explicit
  test-fixture profile keeps the no-silent-defaults directive intact.

Residual watch-items (NOT must-fix; confirm during Sprint A/B builds):
1. The retry_taxonomy/fallback.py split -- if retry_taxonomy imports symbols
   from fallback.py, the rip order within A1 matters; grep imports first.
2. B5's xfail list is the only mechanism keeping unbuilt rows visible --
   the test must FAIL (not skip) when an xfail entry unexpectedly passes,
   so a shipped adapter forces the xfail removal.
3. E1's required field will touch many test fixtures; budget a full-suite run
   for that chunk alone.

No scope creep detected; nothing further to cut. Stop at convergence.
