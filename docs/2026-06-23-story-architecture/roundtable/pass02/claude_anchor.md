<!-- Claude grounded anchor review -- R2 (coding plan / implementability). Written before panel output. -->

VERDICT: yes-with-fixes. The candidate set is buildable, but three integration points are under-specified in ways that bite at the keyboard: the pitch->brief handoff has a hard char/schema limit, the two-tier escalation needs a scope the enum does not have AND a re-entry point the cascade does not have, and the "beat-must-turn" check cannot live where the plan implies.

MUST-FIX BEFORE BUILD:

1. [C1 handoff] "Rewritten `script_brief` feeds `_otr_outline`" collides with the real schema. `news_interpreter.py` defines `script_brief` as <= 350 chars inside the `NewsBriefs` pydantic model; the selected pitch carries logline + protagonist + emotional_core + theme_sentence + final_20_seconds -- far more than 350 chars. Fix: do NOT overload `script_brief`. Add a new optional `planning_brief` (or `pitch_selected`) field consumed by `_otr_outline`, leaving `news_interpreter`/`NewsBriefs` intact; map PitchCandidate fields explicitly into the outline macro-premise inputs.

2. [C2 Tier 1] "Re-OUTLINE on the same premise" has no re-entry point. `decide_escalation_scope` runs INSIDE the freeze cascade, which executes AFTER composition; the cascade's EPISODE branch stamps `needs_full_rerun` (terminal) -- there is no mid-cascade hook that re-runs only the outline and re-composes. Fix: either (a) make Tier 1 a full rerun flagged "reuse premise, redraw outline with axis penalty" (honest: nearly as expensive as a full rerun), or (b) add an explicit writer re-entry that the cascade can request. Pick (a) for v1; do not pretend Tier 1 is cheap.

3. [C2 scope] `EscalationScope` today is NONE / EPISODE / (LINE). The two-tier plan needs to distinguish re-outline vs re-pitch. Fix: add `EscalationScope.REPITCH` (Tier 2) and treat EPISODE as Tier 1 (re-outline, same premise); `decide_escalation_scope` keys REPITCH off `premise_clarity`/console-standoff verdicts, EPISODE off the other structural axes. Cap each scope's cycles independently in `meta.reroll_escalation`.

4. [C4] "beat must turn as a `score_outline` penalty" cannot live in `score_outline`: that function is explicitly PURE (no torch, no LLM, never mutates -- per its module docstring) and runs on RAW beat intents before grounding. Detecting a power/status/knowledge turn from intent text is an LLM judgment, not a pure metric. Fix: implement C4 as a SEPARATE pre-composition LLM outline-critic (mirroring `run_story_critic`'s injected `generate_fn` pattern) that emits per-beat turn flags; feed its result to `score_outline` as a numeric penalty input, keeping the scorer pure.

SHOULD-FIX:

1. [C0] The local-ceiling probe is "$0" only in API terms -- compose->grade requires the full GPU writer path for ~10 short episodes. Fix: state it as a local GPU experiment (short word budget) with a fixed seed set + the >=75 pass bar; not instant, but cheap.

2. [C1 greenlight] Routing greenlight to the frontier lane reuses the existing opt-in OpenRouter writer slot + cost guard. Fix: name which slot (creative vs technical) and confirm the per-episode added remote call stays inside the existing cost-guard budget; fail-closed to local if the lane is disabled.

3. [C1 divergence seeds] "conflict-type from `_otr_story_quality_l12`" -- confirm the palette is a public, enumerable symbol (the conflict palette / beat_role sequence) and not a module-private constant; if private, expose a public accessor rather than reaching in.

[ASSUMPTION] I have not re-read `_otr_reroll_escalation.py` line-by-line this pass (it is panel grounding); the exact `EscalationScope` members + `decide_escalation_scope` signature are verify-against-grounding. The PURE-scorer and 350-char-brief claims are CONFIRMED from earlier reads.
