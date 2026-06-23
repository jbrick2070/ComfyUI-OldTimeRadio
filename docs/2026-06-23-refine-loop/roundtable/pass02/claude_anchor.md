ROUND 2 -- CLAUDE ANCHOR (coding plan / implementability; code-grounded)

VERDICT: yes-with-fixes. pass01 is codeable and mirrors v0's proven patterns, but the pass-isolation
refactor and the grader I/O contract must be pinned before code.

MUST-FIX BEFORE BUILD:
1. [Build order 1 / _build_and_compose] This is the riskiest chunk and is under-specified. The current
   compose path is INTERLEAVED in run(): the per-beat loop spans ~L3280-4060 with announcer intro/outro
   special cases (compose_announcer_intro/outro), `slot_scheduler.helper_context("compose_line")`, the
   reroll-context meta threading for FreezeCascade (~L3440-3606), and post-compose title regen + canon
   write (~L4356). Extracting a pass-local helper that COMMITS NOTHING is non-trivial. Fix: define the
   helper signature explicitly -- inputs (outline, cast_rows, resolved, budget, slot fns, prior_critique)
   -> output a `Candidate` dataclass {composed_ledger, structural_score, grade, meta_delta} built on a
   FRESH ledger + a pass-local meta dict; enumerate EVERY writer-state mutation in the current path
   (led/ledger rows, meta keys, canon, title) so none leak. Chunk 1 must be byte-identical for passes=1.
2. [Loop step 5 / grade_story] Pin the I/O contract: input = the composed spoken-line text (specify HOW
   it is pulled from the ledger), premise, meta -> `StoryGrade(score:int 0-100, biggest_weakness:str)`.
   Route it through the EXISTING `structured_call` ladder (the same one _otr_outline uses) with a small
   pydantic schema -- NOT a raw generate_fn -- so robust JSON parse + bounded retry come for free. State
   which slot (creative vs technical) generates it. Seed before the call.
3. [Loop steps 3-4 / deep-copy] State which object each step uses: `score_outline` runs on the ORIGINAL
   outline's RAW intents (before any build_sq_data); `build_sq_data` runs on a `copy.deepcopy(outline)`
   so the per-beat intent mutation cannot leak to the next pass. Verify deepcopy on the pydantic Outline
   deep-copies the Beat list (it does) -- add a test asserting pass i+1 sees un-mutated intents.
4. [Loop step 6 / prior_critique] Add `prior_critique: str = ""` at the END of the frozen OutlineRequest
   (after diversity_hint, same append rule v0 used); set via `dataclasses.replace`. Render in
   `_build_user_prompt` in a SEPARATE block from diversity_hint, only when non-empty. Assert BOTH-empty
   => byte-identical to today's prompt.

SHOULD-FIX:
5. [grade parse] The "deterministic low score" fallback fires only when the structured_call ladder
   exhausts; define the floor (e.g. 0) so keep-best still orders correctly.
6. [telemetry] All plain JSON scalars; `grade_delta = score[i]-score[i-1]` (None at i=0).
7. [attribution] Wrap each pass's compose + grade in `slot_scheduler.helper_context("refine_compose"/
   "refine_grade")` so slot accounting stays clean (mirrors existing helper_context usage).

CUT: nothing new beyond R1's cuts.

[ASSUMPTION] `structured_call` (from _otr_structured_call, imported by _otr_outline) is reusable for the
grader. Verify it is importable at the grade point in the writer.
