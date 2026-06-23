ROUND 3 -- CLAUDE ANCHOR (wiring / integration / sequencing; code-grounded)

VERDICT: yes-with-fixes. Path C targeting is now correct, but WINNER-COMMIT SEQUENCING, the
critique->outline interface contract, and the widget->bar propagation need pinning.

MUST-FIX BEFORE BUILD:
1. [Pass isolation / Build chunk 3 -- winner-commit sequencing] The current run() runs the post-compose
   steps ONCE, inline, AFTER the compose loop: post-compose title regen + `episode_canon` write (~L4356),
   the FreezeCascade reroll-context meta threading (~L3440-3606), ledger save. In v1 these must run ONLY on
   the WINNER, AFTER the refine loop. Fix: define the boundary precisely -- `_build_and_compose` ENDS at
   {composed pass-local ledger + structural_score + grade}; title regen / canon write / meta threading /
   ledger save happen ONCE post-loop on the winner. The refactor must MOVE those steps out of the per-pass
   path. Sequence: loop builds N isolated candidates -> pick winner -> commit (title/canon/meta/save) on
   the winner only.
2. [Grader -> revision overlay -- interface contract] The grader critiques the COMPOSED story, but the
   overlay feeds the critique to the OUTLINE (pre-compose) macro/beat prompts. A LINE-level critique ("the
   line 'Override the protocols' is on-the-nose") is NOT actionable at the outline level. Fix: the grader
   rubric must return the single biggest STRUCTURAL/dramatic weakness (arc, stakes, premise grounding,
   character want), NOT a line edit -- so the outline reviser can act on it. (Line-level fixes are already
   the downstream freeze-cascade doctor's job; do not duplicate.)
3. [prior_macro construction + sequencing] Define exactly how `prior_macro` is built from the prior
   winner's outline (Title/Premise/Setting + voiced beat intents, length-capped) and capture it from the
   `raw_outline` BEFORE `build_sq_data` mutates the intents -- else the digest carries grounded/substituted
   nouns, not the real prior. Sequence: snapshot prior_macro from raw_outline at the end of each pass.
4. [Widget -> bar propagation] The dropdown is a NEW INPUT_TYPES widget on OTR_LedgerScriptWriter; thread
   it: JSON widgets_values (APPEND at END, positional, BUG-LOCAL-097) -> INPUT_TYPES decl -> run() param
   (default) -> `resolve_refine_passes` bar. Re-validate widget-count vs live INPUT_TYPES + link integrity
   after the JSON edit (CLAUDE.md S0). The widget change lands IN otr_scifi_16gb_full.json in the SAME
   commit as the code (chunk 4).

SHOULD-FIX:
5. [Env vs widget precedence] State precedence: env `OTR_STORY_REFINE_BAR`/`PASSES` OVERRIDE the widget
   (headless); else the widget value. Document in resolve_refine_passes.
6. [best_of_n collision wiring] Resolve the collision STRUCTURALLY: the loop calls `generate_outline`
   directly (never `select_best_outline`), so best-of-N is bypassed by construction -- not by mutating the
   `OTR_STORY_BEST_OF_N` env. Assert no `best_of_n` telemetry key when refine is active.
7. [Grader composed_text timing] Extract `composed_text` from the PASS-LOCAL ledger after compose and
   before any commit -- never from the live writer ledger.

[ASSUMPTION] The post-compose title/canon/meta/save block is cleanly separable from the per-beat compose
loop. Verify the compose loop (~L3280-4060) and the post-compose block (~L4356+) share no interleaved
state that resists extraction into the winner-commit step.
