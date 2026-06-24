<!-- requested_model: deepseek/deepseek-v4-pro | resolved_model: deepseek/deepseek-v4-pro-20260423 -->

VERDICT: yes-with-fixes — Build-order inconsistency (C2 Tier 1 depends on C4, but C4 is scheduled after) and several unquantified or underspecified integration points would block implementation.

MUST-FIX BEFORE BUILD:
1. [Build‑order] C2 Tier 1 needs “Add the staging penalty (C4) to the re‑outline selection”, yet the build order puts C4 staging **after** C2 Tier 1. This is a direct dependency violation. Fix: reorder C4 **before** C2 Tier 1, or provide a zero‑penalty stub in C2 Tier 1 that is replaced when C4 lands, with a clear integration note. Without this, the plan is self‑contradictory.
2. [C0] Durable flag mechanism is unspecified. “Emits a DURABLE operator flag OTR_ENABLE_FRONTIER_GREENLIGHT” does not say *how* C0 persists and how C1 reads it. Fix: define that C0 writes an environment‑variable export (e.g., `OTR_ENABLE_FRONTIER_GREENLIGHT=true`) into a well‑known shell‑source file (e.g., `$OTR_RUNROOT/.gate_flag`) and that the conductor sources it before C1.
3. [C0] “Reduced word budget” and “grade the best few twice” are unquantified. Fix: set max 900 words per episode; generate up to 5 pitches; grade the top 3 by score; each gets two independent grades; pass if any has both ≥ 75.
4. [C1] The local genre+protagonist‑archetype axis is underspecified (“palette is domain‑keyed, not genre‑keyed”). An implementor could misinterpret how to fuse it with the domain conflict palette. Fix: supply a concrete rule: for each pitch, randomly select genre from `[‘thriller’,’drama’,’sci‑fi’,’noir’]` and archetype from `[‘reluctant hero’,’anti‑hero’,’naive idealist’,’jaded veteran’]`; combine with the domain‑keyed conflict via a templated logline prompt (include the template). This ensures deterministic divergence.
5. [C2] The refine loop currently uses only `grade_story.biggest_weakness`; there is no specification for *storing* the 5 B critic’s `failing_axes`/`regeneration_hint` across passes so that they can be swapped into `prior_critique`. Fix: after grading, run the critic, persist its output in `meta.story_quality.critic_failing_axes` and `critic_regeneration_hint`. In the refine loop, read the last stored critic output and format it as a text string for `prior_critique`. Define the concatenation format.
6. [C4] The staging penalty is not defined. “Add an explicit optional penalty=None kwarg” gives no computation or application rule. Fix: specify a function `_otr_staging_penalty(outline)` that returns 50 if the BEAT_ROLE “irreversible_choice” beat is missing intent, else 0. (Cut the beat‑turn heuristic for v1.) `score_outline` subtracts penalty from raw score, and `select_best_outline` receives it. Document weight so that selection behaves predictably.
7. [C1] The mapping from `PitchCandidate` to “CONCISE script_brief” is left to the implementor; without a template, length may exceed downstream macro‑prompt limits. Fix: provide a fixed template (e.g., `“{logline} Protagonist: {protagonist}. Conflict: {conflict_type}, {setting_class}. Emotional core: {emotional_core}. Final 20s: {final_20_seconds}.”`) and hard‑truncate to 200 tokens. Confirm against the macro‑prompt budget.
8. [C2] The new critic axis `console_standoff` is required for `PREMISE_AXES` but the plan does not state how the 5 B critic will produce it. Fix: add `console_standoff` to the critic’s rubric (e.g., a severity 0‑2) and include it in the output schema; document where it appears in `failing_axes`.
9. [C1] `OTR_GREENLIGHT_MODEL` is referenced as an OpenRouter model but never defined as an environment variable. Fix: specify env‑var `OTR_GREENLIGHT_MODEL` (default `None`) and state that when `OTR_ENABLE_FRONTIER_GREENLIGHT` is true, the greenlight call uses that model; if unset, fall back to a hardcoded frontier model (e.g., `openrouter:anthropic/claude-3.5-sonnet`).

SHOULD-FIX:
1. [C0] The operator decision step (“decides frontier vs accept‑B”) has no automated protocol; describe a simple interactive prompt and recorded outcome to make it auditable.
2. [C2] Include a regression test that exercises the refine loop with the critic‑swapped `prior_critique` against a known scenario to confirm keep‑best still works.
3. [C4] Audit all callers of `score_outline`/`select_best_outline` for the new penalty kwarg; list them in the plan so the implementor doesn’t miss any.
4. [C1] Add a validation that the generated concise script_brief does not exceed the macro‑prompt token limit (automated at build time).

OPTIONAL / NICE-TO-HAVE:
- The beat‑turn heuristic in C4 could be deferred (cut complexity) and replaced by a simple check “voice = character/announcer” with `intent`; the penalty defined in MUST-FIX 6 is sufficient for v1.
- Provide example loglines for the pitch room to guide prompt tuning.

CUT THESE:
1. [C4] The beat‑turn heuristic (percentage of beats with intent) – safe to cut because the irreversible‑choice penalty already enforces the climax staging; the heuristic is vague and adds integration risk.
2. [C2 Tier 2] Already deferred, no action.

VERIFY-AT-BUILD checklist (from earlier rounds + new):
- Confirm exact `…_PALETTE` and `BEAT_ROLE` symbol names and their publicness in `_otr_story_quality_l12`.
- Verify that `_refine_loop` body re‑invokes outline+compose each pass (confirmed) and that a critique‑source swap from `grade_story.biggest_weakness` to