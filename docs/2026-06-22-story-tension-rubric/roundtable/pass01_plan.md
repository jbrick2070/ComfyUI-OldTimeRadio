# STEP 5/6 -- CONVERGED build spec (roundtable R1). Build-ready.

Panel: GPT-5.5 (gpt-5.5-20260423), Gemini-3.1-pro (gemini-3.1-pro-preview), DeepSeek-v4-pro
(deepseek-v4-pro-20260423) + Claude code-grounded anchor + judge. R1 spend ~$0.21
(GPT/DeepSeek needed --max-tokens 12000; reasoning models burn the 2k default on hidden
reasoning -- known gotcha). Converged R1: all three + anchor agree the load-bearing fix is
the meta dramatic-frame stamp; the rest is well-specified detail. No R2 needed.

## THE UNIFYING INSIGHT (all panelists + anchor)
A SINGLE per-line meta stamp of the dramatic frame solves THREE problems at once:
1. STEP 6 tension delivery to the composer.
2. Critic VISIBILITY -- the critic renders ledger LINES (`_render_lines_for_doctor`), not
   LineRequests, so the tension target must live on the line/meta to be judged.
3. Reroll RECONSTRUCTION -- CONFIRMED (GPT#2, grounded): `build_reroll_line_request`
   (`_otr_reroll.py:339`) reconstructs the LineRequest with arc_phase but OMITS
   beat_objective/obstacle/turn/subtext/dramatic_question/next_turn/beat_tension -> a
   rerolled line today loses the entire dramatic frame the first-pass line had.

## STEP 6 -- escalating beat_tension + the meta dramatic-frame stamp
1. **Deterministic tension (character beats only).** Over the CHARACTER-beat ordinal i
   (0-based, n = count of character beats): `n <= 1 -> 3`; else
   `beat_tension = clamp(round(1 + 4*i/(n-1)), 1, 5)`. Monotone nondecreasing, peak at the
   final character beat. NO easing (cut -- contradicts "escalating"). Announcer/music/sfx
   EXCLUDED from the curve and from any critic tension check.
2. **Stamp the frame on meta (frozen-wire-safe).** At first-pass compose (writer ~L3452),
   in addition to passing the existing objective/obstacle/turn/subtext, pass
   `beat_tension=<derived>` into the LineRequest AND write a per-line frame to free-form
   meta: `meta["line_dramatic_frame"][line_id] = {objective, obstacle, turn, subtext,
   tension, dramatic_question, next_turn}`. META ONLY -- no new Pydantic line field; the
   ledger {cast,lines,meta} wire format stays frozen.
3. **Reroll reads the frame back.** `build_reroll_line_request` reads
   `meta.line_dramatic_frame[line_id]` and reconstructs beat_objective/obstacle/turn/
   subtext/dramatic_question/next_turn/beat_tension into the reroll LineRequest (fixes the
   asymmetry). Missing frame -> empty defaults (PD1 never-raise).
4. **Critic sees the target.** `_render_lines_for_doctor` (the critic's row renderer)
   renders `target_tension=N/5` per character line from the meta frame.

## STEP 5 -- flat rubric + failed_dimension
1. **Critic SECTION 3 rubric (prompt only).** A `character` line is flat iff it does NONE
   of {change knowledge, shift pressure, move relationship, force/avoid a decision,
   raise/clear an obstacle} AND fails to advance its `beat_intent`. Use **beat_intent**, NOT
   `line_job` (GPT#3: line_job is not in the critic's input; beat_intent IS). "Be sparing --
   most competent lines pass."
2. **Tension judged by APPROPRIATENESS, not raw intensity (GPT#8).** Add level definitions
   to the prompt: 1 = orientation/unease, 3 = active pressure/choice, 5 = irreversible
   decision/reveal/cost. The critic judges whether a line MEETS its target_tension level --
   a calm setup line at target 1 is NOT flagged for failing to be explosive.
3. **failed_dimension = OPTIONAL enum on RerollTarget ONLY** (reroll acts on reroll_targets;
   one authoritative field -- cut the FlatLine/dual-field uncertainty). Default
   `"unspecified"` so old `meta.story_critic_report` dicts + `StoryCriticReport.clean()`
   still validate (GPT#9 back-compat). Enum: the 5 dimensions + `tension` + `unspecified`.
4. **NO deterministic reroll mapper (DeepSeek#2 + GPT#5).** The critic's `hint` is already
   concrete + threaded verbatim to `compose_line(reroll_hint=...)`. failed_dimension is
   metadata/telemetry (optionally a short hint PREFIX) -- do NOT build a dimension->REVISE
   re-craft. Drop the word "parser" from the design.
5. **Over-flag calibration (GPT#10).** Only emit a reroll_target when a line fails BOTH
   beat-advancement AND a named dimension. (Keeps the residual-flag count from rising.)

## CUTS (convergent)
Easing on last beats; any deterministic CODE gate for flatness (stays LLM judgment);
SceneArcContext; the dual FlatLine+RerollTarget enum; the reroll dimension->hint mapper;
"climax phase" language (budget phases are setup/complication/resolution -- say "peak at
final beat").

## VERIFY-AT-BUILD
- `run_targeted_reroll` FUNCTION docstring (~L453) may still say lines are "restored" on
  cap -- contradicts the STEP-4 repair-then-ship behavior; update it (GPT SHOULD#6).
- `_render_lines_for_doctor` is the right surface for `target_tension` (confirm it is the
  critic's renderer, not only the doctor's -- it is shared).
- Regression fixtures for 1, 2, and 13 character-beat episodes (tension edge cases).
- post_validator render_priority completeness -- optional lenient warn (GPT SHOULD#5); defer.

## INVARIANTS
No workflow-JSON/node/widget change. Ledger {cast,lines,meta} wire format frozen (frame
rides free-form meta). speaker_role is the ONLY role source. Tension derivation deterministic
+ seed-stable. Flatness stays LLM judgment. STEP-4 convergence invariant + approved-line
preservation intact (only reroll_targets recomposed). Suite + Bug Bible green per chunk. 100%
local. Build order: STEP 6 (tension + meta frame + reroll read + critic render) FIRST, then
STEP 5 (rubric + failed_dimension) -- STEP 5 references the target_tension STEP 6 exposes.
