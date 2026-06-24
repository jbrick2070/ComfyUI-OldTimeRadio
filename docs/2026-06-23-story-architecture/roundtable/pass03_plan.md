# OTR Story Architecture -- Near-final Plan (R3 synthesis: wiring/sequencing)

Candidate set stable since R1. R3 fixed the integration seams. Key grounded discovery: the WRITER
already has the re-plan loop -- `OTR_LedgerScriptWriter._refine_loop` re-runs the full writer body
(re-outline + recompose) up to N passes, revising via `prior_macro` (same premise) + `prior_critique`
(from `grade_story.biggest_weakness`), keep-best. So Candidate 2 Tier 1 LARGELY EXISTS; the new work
is feeding the 5B critic's axes into it, plus Tier 2 (re-pitch) as a later sprint.

## Build order (locked)

C0 gate -> [operator ceiling decision] -> C1 pitch room -> C2 Tier 1 (ride refine loop) -> C4 staging
-> C2 Tier 2 (re-pitch) -> ; C3 (use_exchange) parallel/any time (config-only).

## Candidate 0 -- GATE: local-ceiling probe (own mini-pitch; grade composed output)

- Standalone: a temp `generate_pitches()` (NOT C1's node) seeded from the raw script_brief + divergence
  seeds; outline; compose SHORT full episodes (reduced word budget) -> `grade_story`. (grade_story needs
  composed story shape, so compose; bound cost via short budget. Outline-only score is a cheap
  pre-filter, not the ceiling signal.)
- Pass bar: any >= 75 (B). Grade the best few TWICE; pass = both >= 75. Log model id / seeds / temp.
- Emits a DURABLE operator flag `OTR_ENABLE_FRONTIER_GREENLIGHT` (true/false) that C1 reads. If no
  local episode clears 75 -> operator decides frontier vs accept-B; if local-only, relabel success.

## Candidate 1 -- pitch room + greenlight (IN-CONDUCTOR stage, not a graph node)

- Lives as `_otr_pitch_room.py`, called inside `OTR_LedgerScriptWriter.run()` AFTER news briefs,
  BEFORE `generate_outline` (the `news_interpreter` pattern; injected generate_fn).
- Force divergence: seed 3 pitches from the REAL `_otr_story_quality_l12` palette (domain/conflict,
  via a `load_conflict_palette()` adapter -- resolve the exact symbol first) PLUS a local
  genre + protagonist-archetype axis (palette is domain-keyed, not genre-keyed).
- Schemas (Pydantic via `structured_call`): `PitchCandidate(id:int, logline, protagonist,
  antagonist_or_pressure, genre_mode, emotional_core, theme_sentence, final_20_seconds, conflict_type,
  setting_class, surprise:int, human_want:int, stageability:int, console_standoff_risk:int,
  why_different)`; `GreenlightDecision(selected_id:int, ranking:list[int], rationale)`. Validate
  selected_id in ids, ranking is a permutation, >= 3 valid candidates (else regenerate); tie-break on
  numeric console_standoff_risk then id (NOT prose).
- Greenlight resolution: LOCAL greenlight (same rubric prompt on the local model) is the DEFAULT/
  fallback; `OTR_GREENLIGHT_MODEL` (openrouter:...) upgrades it when `OTR_ENABLE_FRONTIER_GREENLIGHT`;
  timeout 10s, 1 retry, fail-CLOSED to local on disable/timeout/unparseable.
- Handoff: build a CONCISE script_brief from the winner; `dataclasses.replace(outline_req,
  script_brief=brief)` (OutlineRequest is FROZEN). Stamp raw news_seed + full pitch + fingerprint in
  `meta.story_quality.pitch`. Length-bound the brief (verify macro-prompt tolerance).

## Candidate 2 -- critic -> re-plan, riding the refine loop (Tier 1 now, Tier 2 later)

- TIER 1 (now, mostly wiring): feed the 5B critic's `failing_axes`/`regeneration_hint` into the refine
  loop's `prior_critique` (today it uses only `grade_story.biggest_weakness`). Same premise; re-outline
  + recompose is the existing pass. Add the staging penalty (C4) to the re-outline selection.
- Escalation routing fix (grounded collision): in `_otr_reroll_escalation`, SPLIT `STRUCTURAL_AXES`
  into `PREMISE_AXES = {premise_clarity, console_standoff}` (new critic axis) vs
  `EPISODE_AXES = {resolution, emotional_arc, continuity}`; add `if premise_hits: -> PREMISE` BEFORE
  the structural->EPISODE block. Add `EscalationScope.PREMISE`. All behind `enable_critic_escalation`
  (turn ON in canonical workflow as part of this sprint).
- TIER 2 (later sprint): on PREMISE, the next refine pass DROPS `prior_macro`'s premise and calls the
  pitch room with a `PitchRequest(showrunner_note=regeneration_hint, excluded_fingerprints=...)`.
  Fingerprint = normalized (conflict_type, setting_class, norm(antagonist_or_pressure),
  hash(final_20_seconds)), threaded in `meta`. Until Tier 2 ships, route PREMISE -> EPISODE
  (temporary) so no consumer crashes. Caps `OTR_STORY_REPITCH_MAX`(1) / `OTR_STORY_REPLAN_MAX`(2) as
  counters STORED IN meta (survive the refine reruns); on exhaustion keep-best.
- End-to-end enum rollout order: enum value -> decide_escalation_scope -> _otr_freeze_cascade routing
  -> meta serde -> exhaustion->keep-best -> regression cases (failing_axes=[premise_clarity]->PREMISE,
  [resolution]->EPISODE).

## Candidate 3 -- flip `use_exchange` (config-only, GPU N=3)

- Verify exact JSON field name + precedence (writer runtime must not override). N=3 harness runs the
  canonical workflow with ONLY that diff, asserts effective `use_exchange=True`, VRAM <= 14.5 GB, zero
  slot drift. Validate on a SEPARATE run from C1/C2 (one variable at a time -- attributable grades).

## Candidate 4 -- staging enforcement (deterministic; inside the selection loop)

- POST-outline, PRE-composition, computed INSIDE the best-of-N candidate loop so it influences
  `select_best_outline` (a post-selection validator cannot steer selection).
- Deterministic first: enforce the existing `BEAT_ROLE` "irreversible_choice-on-stage-as-the-last-beat"
  (climax = final voiced beat is character/announcer with a decisive `intent`; missing intent ->
  penalty) + a beat-turn heuristic. Add an explicit optional `penalty=None` kwarg to `score_outline`
  + `select_best_outline` (byte-identical when None; audit ALL callers + regression). Expose BEAT_ROLE
  via one import from `_otr_story_quality_l12`, not a copied literal. LLM outline-critic only if the
  deterministic version proves insufficient (CUT for v1).

## Deferred (unchanged): B3/B4 prose->ledger parser (spike), multi-seed assignment desk, character
interviews, listener-taste critic, distinct voices, repo survey (Open-Theatre only), refine-loop
hardening (the loop's non-monotonicity is REAL regression -- each pass is a fresh compose+grade, not
grader-noise; revisit after the levers land).

## Residual verify-at-build (for R4)

- exact `..._PALETTE` + `BEAT_ROLE` symbol names/publicness in `_otr_story_quality_l12`.
- `_refine_loop` body re-invokes outline+compose each pass (CONFIRMED) and accepts a critique source
  swap from grade-weakness to critic-axes without breaking keep-best.
- `decide_escalation_scope` caller set + `_otr_freeze_cascade` scope switch coverage for PREMISE.
- macro-prompt length tolerance for a richer `script_brief`.
