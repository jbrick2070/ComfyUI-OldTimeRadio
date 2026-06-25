# R3 CLAUDE ANCHOR -- dynamic coda segue, wiring (grounded)

VERDICT: yes-with-fixes. The wiring is clean (a dedicated function + an early branch
leaves the OFF path untouched). Three confirmations + one simplification.

## MUST-FIX (confirmations)
1. **Byte-identity via the early branch.** `compose_news_coda` is a NEW function,
   only reachable when `_style_grammar_on AND nc_brief`. OFF-flag and no-brief both
   call the EXISTING `compose_announcer_outro` with its current args (UNCHANGED) ->
   byte-identical. The shared placeholder at :4483 and `patch_line_text` at :4638
   stay common to both branches. Confirm the branch wraps ONLY the compose call.
2. **cast_seed scope.** `cast_seed` binds at :2878 and the outro is composed later in
   the SAME run() frame -> in scope at the coda call (:4615-4634). No rebinding.
3. **Reroll isolation.** The announcer outro is composed POST-LOOP, NOT via the
   in-loop line path -> `build_reroll_line_request` (:3922) does NOT touch it. So the
   coda needs NO meta-threading (unlike a line-level field). The coda's own reroll is
   INTERNAL to `compose_news_coda` (prompt variation). Confirm the freeze-cascade
   reroll operates on body lines, not the announcer outro.

## SIMPLIFICATION (fold into the build)
**The main campaign's STEP F climax-line DECOUPLING is now unnecessary.** STEP F
passed the CLIMAX line (not the last line) to the outro so the outro wouldn't
restate the wrong beat. But the coda redesign makes the ON-flag outro a pure
premise->real-news pivot that NEVER touches the fictional climax at all -> Job 2
("protect the character climax") is satisfied BY CONSTRUCTION. The OFF-flag path is
unchanged (byte-identical), so it doesn't need the decoupling either. => DROP the
climax-line lookup + `climax_character_line` param from the build (one fewer edit,
one fewer verify-at-build). Keep `_climax_beat_id` for KILL 3's own later build.

## SHOULD-FIX
1. **Coordinate with the main campaign.** The coder builds the coda-segue version
   (`compose_news_coda`) and REMOVES STEP F's `NEWS_CODA_LEAD_IN` +
   `validate_news_coda_line` + the climax-line decoupling. Update pass04_plan.md
   STEP F + CODE_MAP.md C3 accordingly so there is ONE coda design, no dead code.
2. **Refine loop.** The refine loop re-invokes run() per pass (forced cast_seed) ->
   each pass composes its own coda deterministically. No special handling; confirm
   the fallback pool's cast_seed key is the same forced seed across passes.

## CUT
- Nothing new. (The R2 spec is already minimal: one function, a length-capped
  validator, a deterministic fallback.)

## ASSUMPTIONS
- [ASSUMPTION] the freeze cascade does not separately re-compose the announcer outro
  (it re-rolls body character lines). Verify by grepping the freeze/reroll path for
  the announcer/outro beat id; if it DOES, route it through `compose_news_coda` too.
