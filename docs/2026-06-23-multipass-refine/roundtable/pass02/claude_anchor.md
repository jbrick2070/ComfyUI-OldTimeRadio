# Claude anchor review -- R2 (coding plan / implementability)

Grounded vs `_otr_outline.py`, `OTR_LedgerScriptWriter.py`, `_otr_freeze_cascade.py`, `_otr_story_quality_l12.py`.

## VERDICT
IMPLEMENTABLE as best-of-N selection; the iterate-until-good v1 is implementable but riskier. The single
biggest coding decision is the mutable-spine path.

## MUST-FIX (R2)
1. **Spine patch path -- pick ONE and it must be re-validated.** CONFIRMED `generate_outline` returns a
   strict `Outline` (Pydantic) that then passes `validate_outline_against_budget` (arc_phase membership +
   monotonic order + per-phase word/beat budgets) and gets `dialogue_slot_id` stamped. A best-of-N
   candidate spine MUST re-run those validators or it will desync the budget/slot invariants. RECOMMEND:
   N candidates = N independent `generate_outline` calls with seed-varied RNG + a structural constraint in
   the prompt (NOT in-place beat surgery, which would bypass the combiner + slot stamping). Each candidate
   is a fully-validated `Outline`; score, keep-best, then the normal compose path runs ONCE on the winner.
   This makes "re-slug" trivial: the winner just flows through the existing compose loop -- no double
   freeze, no partial re-entry.
2. **Deterministic rubric must be a pure function over the (Outline + composed lines), reusing existing
   signals.** CONFIRMED `_otr_story_quality_l12` already computes `count_ungrounded_crisis` + the beat_role
   sequence + distinct conflict slots. The rubric is: ungrounded_crisis_density below a threshold AND an
   on-stage irreversible_choice present AND >= K distinct conflict objects. No new LLM grade. Score BEFORE
   audio so audio renders once on the winner.
3. **best-of-N happens at the SPINE stage (cheap, text-only), NOT after full compose of every candidate.**
   Grading a spine needs the beats, not the rendered dialogue -- so score candidate OUTLINES (the
   structural signals are all outline-derivable), keep-best, compose the winner ONCE. This is what keeps
   N passes actually cheap (no N full TTS or even N full line-composes).

## SHOULD-FIX (R2)
- Hard cap N small (3-5) for v0; keep-best comparator = (rubric_pass desc, ungrounded_density asc,
  distinct_conflict desc), deterministic tie-break on seed.
- The local-only gate reads the resolved backend; add a one-line LOUD log of N + winner score.

## CONFIRMED / UNVERIFIABLE
- CONFIRMED: outline validation + slot stamping are post-`generate_outline`; bypassing them via in-place
  surgery is the trap.
- UNVERIFIABLE (R3): exact insertion point for "compose only the winner" vs the current single-outline
  assumption in `OTR_LedgerScriptWriter.run()`.
