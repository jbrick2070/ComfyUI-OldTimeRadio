# STEP 5/6 design -- escalating beat_tension + flat rubric / failed_dimension

OTR is an offline ComfyUI text->radio-drama pipeline. STEPs 1-4 of the story+cast
fix shipped (cast role source, auditor scope, voice fail-closed, scoped reroll
convergence + repair-then-ship); the minimal re-soak PASSED with the bypass OFF
(both episodes ship `frozen_with_doctor_edits`/`frozen_with_warns`, reroll
diverged=False/2 cycles). The residual gap to pristine `frozen_clean` is the
critic still naming 1-3 SUBJECTIVE quality flags -- the STEP 5/6 craft levers.
This document hardens the STEP 5/6 DESIGN before coding. Craft levers, model-
agnostic, deterministic where possible. NO workflow-JSON change.

## Grounded facts (verified vs the real source + the live re-soak ledgers)
1. **The composer is already arc-rich.** `_otr_line_composer.LineRequest`
   (~L571-725) carries arc_phase, dramatic_question, beat_objective, beat_obstacle,
   beat_turn, beat_subtext, next_turn, outline_spine, current_beat_block. DO NOT add
   a SceneArcContext (would duplicate these).
2. **arc_phase ESCALATES, validated monotonic.** Outline beats are grouped by
   arc_phase in budget order (`_otr_outline.py` validation #5). Real re-soak arcs:
   `setup xN -> complication xM -> resolution xK`. Budget arc_phases are
   setup/complication/resolution (no separate climax/falling phase).
3. **beat_objective / beat_turn are STRONG + concrete** (A5 news-driven slot
   contract, wired at `OTR_LedgerScriptWriter.py:~3452`). Real intents: "secretly
   broadcasts a distress call...", "tenders his resignation in protest".
4. **THE GAP: `beat_tension` is UNWIRED.** Repo-wide grep: `beat_tension` only
   appears as the LineRequest default (`0 = unset; renders only when 1..5`,
   composer L719/L1213) + a writer comment "(beat_subtext / beat_tension) stay
   empty for Path A". There is NO `beat_tension=` at the writer LineRequest
   construction. So the composer's "Tension: N/5" cue NEVER renders -- no per-beat
   intensity gradient reaches the line writer, though the PHASE gradient exists.
5. **The critic's flat judgment is VAGUE.** `_CRITIC_SYSTEM_PROMPT` SECTION 3:
   "Flag lines that are dramatically inert: a fact delivered instead of a moment
   played." `FlatLine` = {line_id, reason} (free string). No shared operational
   definition; composer + critic do not share a flatness target.
6. **The critic sees LEDGER LINES, not LineRequests.** `run_story_critic` renders
   ledger lines (arc_phase, beat_intent, mood, word_count via _render_lines_for_
   doctor). beat_tension is NOT on the ledger line today, so if STEP 5 is to judge
   "did this line hit its target tension," the target must be made VISIBLE to the
   critic (couples STEP 5 <-> STEP 6).
7. **Frozen invariant:** the ledger `{cast,lines,meta}` WIRE format is frozen
   (audio is the first consumer; byte-identical). New fields ride free-form `meta`,
   NOT new Pydantic line fields (the R2 ledger-schema rule). The critic's OWN output
   schema (StoryCriticReport) is INTERNAL -- extending it is allowed.

## STEP 6 design question -- the beat_tension curve
Derive a deterministic escalating `beat_tension` (1..5) from data already on the
validated outline beat (arc_phase + beat index), wire it into the LineRequest at
writer ~3452. Candidate shapes:
- A) Phase-band: setup->1..2, complication->3..4, resolution->5. Simple, coarse.
- B) Smooth ramp: tension = round(1 + 4 * (beat_ordinal / (n_voiced_beats-1)))
  -- climbs 1->5 across the whole episode, peak at the final beat.
- C) Phase-index + within-phase ramp: base per phase + a small rise across the
  beats inside each phase; peak at resolution start (classic climax), optional ease
  on the last 1-2 beats. (arc_phases have no falling phase, so "ease" is a choice.)
Open: peak placement (resolution-start vs final beat); should the LAST line
(announcer close is separate) ease? does over-specifying tension over-constrain a
short 420w/13-line episode?

## STEP 5 design question -- flat rubric + failed_dimension
Replace SECTION 3's vague flatness with a shared 5-dimension rubric in the critic
PROMPT: a `character` line is flat unless it does >=1 of {change knowledge, shift
pressure, move relationship, force/avoid a decision, raise/clear an obstacle} AND
advances its slot line_job (and, if STEP 6 lands, moves toward its target tension).
Add a `failed_dimension` enum to the FlatLine (and/or RerollTarget) schema; update
the `_otr_reroll.py` hint parser/consumer in the SAME change so the reroll
instruction is dimension-specific. Rubric-guided LLM judgment, NOT a deterministic
code test (flatness is literary). Open: exact enum values; whether failed_dimension
rides FlatLine or RerollTarget (or both); how to avoid over-flagging on a
competent line; how the hint consumer maps a failed_dimension to a concrete REVISE.

## Invariants the design MUST respect
No workflow-JSON / node / widget change. Ledger {cast,lines,meta} wire format
frozen (new fields ride free-form meta). speaker_role is the ONLY role source.
beat_tension derivation deterministic + seed-stable. Critic flatness stays LLM
judgment (no code gate). Reroll preserves approved lines + the STEP 4 convergence
invariant. Regression suite + Bug Bible green per chunk. 100% local. Model-agnostic
(every gate is one opus passes -> lifts the weak end, never rewrites the good).
