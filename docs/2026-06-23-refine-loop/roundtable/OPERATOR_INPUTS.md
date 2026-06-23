# Operator inputs during the refine-loop roundtable (2026-06-23)

Captured verbatim-intent so they survive the campaign and land in pass02+.

1. CONFIRMED: recursive passes until the story reaches the target grade ("until B+", "until B").
   The bar is the PRIMARY stop; a HARD CAP is the safety backstop (a weak local model is not
   guaranteed to reach the bar; never an infinite loop).

2. GRADE TARGET = a NODE DROPDOWN widget on OTR_LedgerScriptWriter: choices C+ / B / B+ / A.
   - DEFAULT = B (reachable for a local model => the loop actually terminates; A often unreachable
     => always hits the cap => slow. A stays selectable.).
   - This is a WIDGET => per CLAUDE.md S0 it goes IN workflows/otr_scifi_16gb_full.json in the SAME
     change as the code. Map: C+ ~= 68, B ~= 75, B+ ~= 80, A ~= 90 (0-100 grader scale). The env
     OTR_STORY_REFINE_BAR overrides for headless; widget is the user-facing control.
   - "no-improvement plateau" early-stop => OPT-IN (default OFF) so the default behavior keeps trying
     until the bar or the cap (honors "until B+/B").

3. KEY ARCHITECTURE FORK (operator: "rewrite the story just enough to get an A, not start from
   scratch ... not sure what's best"): TARGETED REVISION vs REGENERATE-FROM-SCRATCH per pass.
   - Operator PREFERS targeted revision (cheaper; keeps the good parts).
   - Constraint: a STRUCTURAL weakness (flat arc / no turning point / wrong premise) cannot be fixed
     by local line edits -> needs a fresh outline. A LOCAL weakness (a few weak/on-the-nose lines)
     can be revised in place.
   - LEAN (to be hardened by R3/R4 panel): HYBRID -- grader classifies the weakness as LOCAL vs
     STRUCTURAL; LOCAL -> targeted revise (reuse/extend the freeze-cascade doctor-edit machinery,
     build_reroll_line_request etc.); STRUCTURAL -> regenerate the critique-informed outline. Keep-best
     by grade across attempts. This realizes "rewrite just enough" while still being able to fix a
     broken structure.
   - This is THE central R3 question. Do not silently pick regenerate-only.

4. FORK RESOLVED BY OPERATOR (2026-06-23): it is ALWAYS a REWRITE of the EXISTING story, NEVER
   write-from-scratch -- even when the SPINE needs changing ("if it needs to rewrite the spine
   great ... but it's a rewrite not write from scratch ... at least it has some ideas to start
   with"). So:
   - Each pass REVISES the CURRENT best (prior outline + composed story) using it + the grader
     critique as the BASIS, producing an improved version that keeps the working parts and fixes
     the weaknesses. The spine CAN change, but it is seeded by the prior story, not regenerated
     blank. This is iterative REVISION toward the bar, not independent re-rolls (that is v0).
   - Implication: need a `revise_story(prior_outline/story, critique, target_grade) -> improved
     outline/story` capability -- a prompt that takes the existing material + weakness and rewrites
     it (spine included when the weakness is structural), NOT a fresh cast_seed-keyed independent
     outline. The prior story is the seed; the critique is the steer.
   - R3/R4 now harden HOW to implement "revise the existing (incl. spine) given prior + critique",
     NOT whether to revise vs regenerate (operator decided: revise).
   - v1 is DISTINCT from v0: v0 = best of N INDEPENDENT drafts (breadth/one-shot); v1 = iteratively
     REVISE ONE evolving draft toward the bar (depth). Keep-best still guards against a revision that
     scores worse (never ship a regression).
