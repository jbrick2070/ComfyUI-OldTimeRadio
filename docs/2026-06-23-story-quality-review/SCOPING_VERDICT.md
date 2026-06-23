# Story-Quality SCOPING VERDICT (2026-06-23)

Operator ask: review the R3 soak stories myself, run a roundtable for the panel's thoughts on the stories, then
run a 4-round roundtable for how to improve -- WITHOUT QA rounds that don't actually help the story.

Done. This is the answer. Full detail: `STORY_REVIEW.md` (my grounded review of 18 episodes),
`roundtable/passA_STORY_CRITIQUE_SYNTHESIS.md` (panel on the stories), `roundtable/pass04_plan_FINAL.md`
(the converged build plan). Total panel spend ~$0.43.

## The verdict in one paragraph
The real problem is bigger and more specific than "imperative command-shouting." All 18 episodes are
dramatically the SAME scene -- a "console standoff" where every premise (classroom AI, fossils, coal law,
astronomy) collapses into 2-3 people fighting over a lever/key/console while a gauge climbs and a countdown
runs; the decisive moment happens off-stage and the announcer narrates the news outcome. Four frontier models
(GPT-5.5, Gemini-3.1-pro, DeepSeek-v4-pro, Grok-4.3), reading the stories cold and independently, ALL located
the root cause in the BEAT PLANNER (not the writer model, not the line composer) and ALL independently said a
flag-and-reroll QA gate WILL NOT work -- the strongest possible confirmation of your instinct. The fix is to
move the lever UPSTREAM and make it DETERMINISTIC.

## Why your three candidates landed where they did (grounded in the soak data)
- **(a) bare-imperative reroll gate** -> RE-SCOPED. The flatness is real, but a reroll gate just makes the weak
  model regenerate the same standoff (panel unanimous). The deterministic version that works is at the composer:
  strip narrated-action via an `ACTION:` marker + regex (L3), no reroll.
- **(b) best-of-N** -> DEFERRED/CUT for v0. It's a generate-N-score-keep-one SELECTION gate (same shape you want
  to avoid), costs N local generations, and cannot fix the STRUCTURAL sameness (all N candidates share the same
  beat). Kept on record; revisit only after the structural fix lands.
- **(c) stronger frontier writer** -> CONTRADICTED by the data. The 3 best ("strong") episodes were all the
  LOCAL gemma-12b; frontier grok (via API, reasoning-low) UNDERperformed it. Model choice changes sentence
  texture, not dramatic architecture. Re-scoped to "default to gemma-12b" (a free win), gated on a bake-off.

## The real lever (ranked) -- all UPSTREAM + deterministic, no QA reroll gate
1. **L1+L2 (the structural core, ship together):** in the beat planner, (L1) give each beat a Python-chosen,
   premise-specific `conflict_object`/`conflict_type` from a domain palette + deterministically substitute the
   generic crisis nouns (override/purge/lever/console...) the weak model defaults to; (L2) make a phase a
   dramatic FUNCTION via a real `beat_role` sequence (setup -> personal_stake -> pressure -> irreversible_choice
   on-stage as the last beat -> consequence), filled deterministically with a fallback when the model under-
   delivers. Neither alone works (new words in the same structure, or the same threat-noise in relabelled slots).
2. **L3 composer action/dialogue strip** (deterministic `ACTION:` marker + regex) -- kills "Jettisoning module,
   bracing for impact." and the EP18 director-note-spoken-aloud leak, no reroll.
3. **L4 minimal transcript sanitizer** (regex hygiene for prompt-leak/quotes).
4. **L5a (do FIRST):** fix two measurement bugs so we can even tell if this works -- the edit-cap that silently
   terminates (and never grades) the BEST writer's dense prose, and a telemetry under-count. No story change.
5. **Deferred (your call):** L5b gemma-12b default (after a bake-off); L6 best-of-N (after the structural fix).

## What we will NOT do (panel + operator, unanimous)
No flag-and-reroll critic gate; no more soft prompt nudges (already in the code, already ignored); no "just hit
883 words" (length is a symptom -- longer incoherence is worse); no "buy a frontier writer" as the primary fix.

## Status
This was a SCOPING window -- no production code was written. The plan is build-ready with a short verify-at-build
checklist (pass04_plan_FINAL.md). The R3 spine stays shipped + default-ON. prod/main + tags GATED. Awaiting your
GO to build (suggested order: L5a -> L1/L2 scaffolding flag-off -> render-on + re-soak -> L3/L4).
