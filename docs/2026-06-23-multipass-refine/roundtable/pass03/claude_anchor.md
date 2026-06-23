# Claude anchor review -- R3 (wiring / integration / sequencing)

Grounded vs `OTR_LedgerScriptWriter.run()` (the real insertion site), `_otr_outline.generate_outline`,
`_otr_freeze_cascade`, `_otr_story_quality_l12`.

## VERDICT
WIRES CLEANLY as a wrapper around `generate_outline`. v0 is internal to the writer node -> ZERO
workflow-JSON change. The one real hazard is the RNG-seed plumbing for candidate diversity.

## MUST-FIX (R3)
1. **Insertion point: wrap `generate_outline` (run() ~L2707), nothing downstream changes.** CONFIRMED the
   writer does outline -> budget check -> F2 `build_sq_data` -> canon -> compose. If best-of-N returns the
   WINNING `Outline` at the same variable `outline`, every downstream stage (budget, build_sq_data, canon,
   compose, freeze, audio) runs UNCHANGED on the winner. This is the lowest-risk wiring: a selector
   function `select_best_outline(generate_fn, outline_req, meta, seed, n) -> Outline` that internally calls
   `generate_outline` N times + scores. Flag OFF => N=1 => the exact current single call => byte-identical.
2. **Candidate diversity must come from the RNG, and the cast/style RNG is seeded per-episode.** CONFIRMED
   `_resolve_cast_rng_seed`/`_resolve_style_rng_seed` derive per-episode seeds (OS entropy unless
   OTR_CAST_SEED/STYLE_SEED pinned). For N DISTINCT candidates the selector must vary the structural RNG
   per candidate (`sha256(episode_seed:n)`) WITHOUT disturbing the pinned-seed C7 byte-identity path
   (when OTR_CAST_SEED is set for regression, N must still be deterministic). Verify the outline RNG is
   threadable per-call.
3. **Local-only gate reads the RESOLVED backend, not an env guess.** CONFIRMED `resolved[
   "creative_writing_model"]` + the slot scheduler backend identify local vs OpenRouter. Gate: if the
   creative writer resolves to a paid/remote backend, force N=1 (loop disabled). LOUD log.

## SHOULD-FIX (R3)
- Score each candidate's `build_sq_data` on a COPY of the beats (build_sq_data mutates beat.intent) so a
  losing candidate's mutations don't leak. CONFIRMED build_sq_data mutates intent in place -> score on
  fresh Outline objects (N separate generate_outline calls already give that).
- Cap N at 3-5 for v0; the 30-word smoke proved one outline ~5 beats composes in seconds, so N=3 outline
  generations (no compose) is cheap locally.

## CONFIRMED / UNVERIFIABLE
- CONFIRMED: downstream-on-winner needs no JSON change (selector is internal to the writer node).
- UNVERIFIABLE (verify-at-build): whether `generate_outline`'s RNG is cleanly re-seedable per call without
  global-state bleed -- check the outline RNG threading at build.
