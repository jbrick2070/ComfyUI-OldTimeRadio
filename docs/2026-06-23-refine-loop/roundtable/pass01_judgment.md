# R1 judgment (arc / creative coherence)

Panel: GPT-5.5, Gemini-3.1-pro, DeepSeek-v4-pro. Spend $0.0931. All 3 VERDICT=no (not build-ready
as-is) -- on ADDRESSABLE underspecification, not a flawed concept. Strong convergence.

## ACCEPTED (grounded CONFIRMED -> folded into pass01)
- Pass ISOLATION (GPT#3, Gemini#2): build_sq_data mutates beat.intent in place (CONFIRMED L654-659) ->
  deep-copy outline per pass; only winner commits. New hard invariant. (Best catch of the round.)
- Separate `prior_critique` field, NOT diversity_hint reuse (Gemini#1) + bounded normalization
  (GPT#2, DeepSeek#1): CONFIRMED v0 diversity_hint is index-keyed structural. New field, render-when-
  non-empty, `critique_to_hint` (prefix + <=200 chars).
- One outline per pass, best-of-N NOT nested (all 3): CONFIRMED O(N*M) blowup. effective_n=1 in loop.
- New lean read-only `grade_story`, not the downstream `_otr_story_critic` (all 3): CONFIRMED critic is
  a separate node (OTR_LedgerFreezeCascade) with reroll side-effects + circular-import risk. Model on
  run_story_brief_reflection (CONFIRMED exists ~L4530).
- Corrected invariant: build_sq_data runs once PER CANDIDATE (isolated), freeze+audio once on winner
  (GPT#7). CONFIRMED.
- REFINE_BAR -> explicit `OTR_STORY_REFINE_BAR` (default 80) (GPT SHOULD#3).
- Provider gate in resolve_refine_passes, fail-closed for unknown handles (DeepSeek#4, GPT#8).
- compose-ok/grade-fail => retain candidate w/ low grade (GPT#4); never discard a shippable story.
- Early-stop also on "grade flat/down 2 passes" + optional wall-clock budget (DeepSeek SHOULD#1, GPT#7).
- Telemetry: + normalized_hint + grade_delta per pass (GPT SHOULD#2) to answer "did it improve?".
- HONEST FLOOR + governance: don't advertise quality lift until the soak proves it (GPT SHOULD#1,
  DeepSeek SHOULD#3, Gemini ASSUMPTION).

## CUT (consensus)
- Remote opt-in (build step 4); nested best-of-N; loser-ledger retention; downstream-critic reuse.

## VERIFY-AT-BUILD (UNVERIFIABLE from R1 grounding)
- cast_seed in scope at loop; per-pass reseed actually re-rolls generation+grader (v0 carryover);
  all full-compose providers use only openrouter:/comfy: prefixes; deep-copy fully isolates intent
  mutation.

## CONVERGENCE
R1 surfaced material structural fixes (isolation, separate field, grader sourcing) -> NOT converged;
proceed to R2 (coding plan) on the hardened pass01. No re-loop of R1 needed.
