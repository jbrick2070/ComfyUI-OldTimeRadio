<!-- Claude grounded anchor review -- R4 (convergence / residual defects). Written before panel output. -->

VERDICT: yes-with-fixes. The plan has converged on direction, candidate set, and wiring; the only
residual risk is that Candidate 2 still carries Tier-2 machinery (PREMISE enum, fingerprints, caps)
that is NOT needed for the first shippable increment and should be fenced harder.

MUST-FIX BEFORE BUILD:

1. [Build order / C2] Make the FIRST shippable increment explicit and tiny, so the PREMISE/fingerprint
   machinery cannot creep into it. Increment 1 = C0 (probe) + C1 (pitch room + greenlight) + C2-Tier1
   ONLY (swap the refine loop's revision trigger from `grade_story.biggest_weakness` to the 5B critic's
   `failing_axes`, and turn `enable_critic_escalation` ON) + C3 (use_exchange flip) + C4 (deterministic
   staging penalty). Increment 2 (separate sprint) = everything PREMISE: the enum value, the
   PREMISE_AXES split, fingerprints, exclusion input, caps, re-pitch. Fix: relabel the plan's sprints
   as Increment 1 / Increment 2 with that boundary; nothing from Increment 2 ships in 1.

SHOULD-FIX:

1. [C2 Tier 1 wiring] The concrete wire is: after each refine pass, read `meta.story_critic_report`
   from the pass result (`_refine_loop`'s `last`) and build `prior_critique` from its `failing_axes` +
   `regeneration_hint` instead of (or in addition to) `grade_story.biggest_weakness`. Verify `last`
   exposes the ledger meta; if not, that is the one new plumbing item for Increment 1.

2. [C0] State the abort action crisply: if no local short-episode clears 75 twice, C0's single output is
   `OTR_ENABLE_FRONTIER_GREENLIGHT=true` proposed to the operator + the success-label rename; do not
   auto-enable frontier (operator gate, per the kickoff S5 ceiling decision).

CUT (final sweep):

1. [C2 Increment 2] Confirm Tier 2 re-pitch is OUT of the first build (already phased) -- and with it
   the fingerprint tuple, `PitchRequest.excluded_fingerprints`, `OTR_STORY_REPITCH_MAX`, and the
   PREMISE enum. They only earn their place once Increment 1 shows the pitch room raises grades.

VERIFY-AT-BUILD (residual):
- exact `..._PALETTE` + `BEAT_ROLE` symbol names/publicness in `_otr_story_quality_l12`.
- `_refine_loop` `last` exposes `meta.story_critic_report` for the Tier-1 trigger swap.
- macro-prompt length tolerance for a richer `script_brief`.
- `use_exchange` JSON field name + precedence; N=3 effective-config assertion.
- all `score_outline` callers updated for the optional `penalty` kwarg; byte-identical when None.

[ASSUMPTION] None new; the above are the carried flags. Direction + candidate set are converged.
