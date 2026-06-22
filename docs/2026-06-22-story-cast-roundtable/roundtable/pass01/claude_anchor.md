# R1 ANCHOR REVIEW (Claude, code-grounded) -- story+cast fulfillment problem statement

VERDICT: yes-with-fixes. The plan is a well-grounded fulfillment frame (real ledger
schema, real night-run verdicts), but it conflates two failure classes that need
different fixes, and it omits the single most important high-level question: is the
clean-freeze bar even achievable, or is the CRITIC miscalibrated?

MUST-FIX BEFORE BUILD:

1. [S5 / whole framing] The R1 question "how do we get LLMs to fulfill it" wrongly
   lumps a CODE-CONTRACT BUG in with a CREATIVE-CRAFT problem. The cast
   `role_mismatch` (S4: voice-engine names `kokoro`/`bark` landing in the role
   field, "not in allowed roles") is NOT something better prompting fixes -- it is
   the audit comparing the wrong fields (engine vs role). CONFIRMED in the night
   log: `OTR_LedgerReviewer] role_mismatch violation on line_id=b001 suggested
   expected='kokoro' not in allowed roles [...]`. Fix: split the fulfillment ask
   into (A) story-CRAFT, prompt/critic-side, the LLM's job; (B) cast/voice CONTRACT,
   a code correctness fix, NOT the LLM's job. Round 1 should only own (A); (B) is an
   R2/R3 code item.

2. [S1 vs S4] Missing the highest-leverage high-level question: **is `frozen_clean`
   achievable at all, or is the critic bar miscalibrated?** CONFIRMED from the run:
   arc_verdict was "strong" 5/55 times (so the arc axis CAN pass), yet `frozen_clean`
   was 0/18 -- i.e. a clean freeze needs every axis (arc AND no-flat AND no-
   continuity-issue AND no-voice-drift) simultaneously and that conjunction never
   held. Before we ask the panel "how to make the writer better," we must ask
   "could ANY writer clear this bar, or does the gate demand perfection?" Fix: add
   this as the first creative question; it changes whether the fix is writer-side or
   critic-recalibration-side.

3. [S5.3] "An effective reroll" is named but the plan states no HYPOTHESIS for why
   it fails, so the panel will guess blindly. Candidate root causes to put in front
   of them (verify: OTR_Reroll + OTR_StoryCritic source): (a) the critic is
   non-deterministic -- flags a different set each pass, so re-composing never
   "clears" a stable target list; (b) "flat" has no operational definition the
   re-composer can act on; (c) the reroll re-composes the line but the critic
   re-judges the WHOLE episode, so new flats appear as old ones are fixed (whack-a-
   mole). The night log supports (c)/(a): cycle1=3 targets -> cycle2=3 targets, same
   count, never converging. Fix: state these candidates so the panel critiques the
   real loop.

SHOULD-FIX:

4. [S2/S3] The plan shows the ledger SCHEMA but not the current WRITER PROMPT
   STRATEGY (how the LLM is actually asked to fill slot_drama_contracts today --
   per-slot? whole-episode? with what context?). The panel cannot propose "prompt it
   better" without seeing the current prompt shape. Fix: add a short "current writer
   approach" subsection (verify: OTR_LedgerScriptWriter compose path).

5. [S4] Voice-DRIFT (33 notes -- a craft problem: keeping each character's register
   distinct) and `voice_preset=None` on 2/4 characters (a cast-BINDING gap) are
   listed together but are unrelated. CONFIRMED `voice_preset=None` in the sampled
   ledger (SHERLOCK STEELE, QUINN SPENDER). Separate them: drift -> craft/R1;
   missing preset -> binding/R2.

6. [S1] "no flat lines" as a goal needs an operational definition or it is
   unfalsifiable. The sampled dialogue was competent noir yet flagged "flat" -- so
   either the bar is taste-subjective or "flat" means something specific (no plot
   advancement? no subtext?). Pin the definition or the writer can never target it.

OPTIONAL: state whether the 17 "successes" are operator-watchable (eyeball), since
the critic calls all of them imperfect but the prose reads fine -- this calibrates
how much is real vs critic-strictness.

CUT: nothing. The plan is lean; do not add the video stack (already scoped out).

[ASSUMPTION] I have not yet read OTR_StoryCritic / OTR_Reroll / cast_lock source for
this anchor -- claims about the reroll/critic MECHANISM are runtime-log-grounded, not
code-grounded yet; flagged "verify:" above and to be confirmed in R2.
