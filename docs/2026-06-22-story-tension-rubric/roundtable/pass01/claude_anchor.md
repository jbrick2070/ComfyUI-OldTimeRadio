<!-- Claude code-grounded anchor review, R1 (design/arc coherence) -->
VERDICT: yes-with-fixes. The GO/NO-GO is GO; the design is sound but has ONE
load-bearing coupling the plan understates (beat_tension must be visible to the
critic) plus two over-engineering risks to cut.

MUST-FIX BEFORE BUILD:
1. [fact 6 -- the coupling] beat_tension must be STAMPED where the critic can see
   it, or STEP 5's "moves toward its target tension" is unobservable. CONFIRMED:
   the critic renders LEDGER LINES, not LineRequests; beat_tension is only on the
   LineRequest. Fix = when STEP 6 derives beat_tension, stamp it onto the ledger
   line's free-form meta (NOT a new Pydantic field -- frozen wire format) AND pass
   it to the LineRequest. Then `_render_lines_for_doctor`/critic prompt can surface
   "target_tension=N" per line. Without this, STEP 5 and STEP 6 are decoupled and
   STEP 5's tension clause is dead.
2. [STEP 6 curve] Pick a curve that ESCALATES but does not demand peak intensity on
   connective beats. Recommend Option C (phase-base + small within-phase ramp,
   peak at resolution-start = the climax), but treat beat_tension as GUIDANCE the
   line should move TOWARD, never a hard numeric gate -- a 13-line/420w episode has
   too few beats for a strict 1..5 ladder. Deterministic from arc_phase index +
   within-phase ordinal; seed-stable.
3. [STEP 5 schema] Add `failed_dimension` (enum of the 5) to BOTH FlatLine
   (diagnosis) and RerollTarget (so the reroll hint is dimension-specific), and
   update the `_otr_reroll.py` hint consumer in the SAME change -- a dimension ->
   concrete REVISE template. CONFIRMED the critic output schema is internal
   (StoryCriticReport Pydantic), so extending it is allowed; the LEDGER schema is
   untouched. invalid/missing enum -> deterministic fallback (never silent).

SHOULD-FIX:
1. The 5 dimensions are well-formed as the plan states {change knowledge / shift
   pressure / move relationship / force-or-avoid decision / raise-or-clear
   obstacle}; keep "flat = does NONE of these AND fails to advance line_job" so the
   gate is permissive (most competent lines pass) -- guards against over-flagging
   (the operator's repeated "lift the weak end, never rewrite the good").
2. Keep the existing SECTION 4 arc_verdict; failed_dimension is per-line, not a new
   arc axis.

CUT THESE (over-engineering):
1. Any falling-action "ease" on the last beats -- the budget arc_phases have no
   falling phase; a synthetic ease adds a knob for no dramatic payoff. Peak at
   resolution and stop.
2. A deterministic CODE test for flatness -- the plan + invariants say flatness is
   literary (LLM judgment). Do not add a numeric flatness gate; the rubric guides
   the LLM only.

[ASSUMPTION] beat_tension on the ledger-line meta is read by the critic prompt
renderer -- VERIFY at build that `_render_lines_for_doctor` (the critic's row
renderer) is the right surface to add "target_tension" to.
