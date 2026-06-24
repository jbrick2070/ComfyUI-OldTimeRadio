# OTR Story Architecture -- SPEC (A+ stories)

Status: CONVERGED via a live 4-round roundtable (2026-06-23). Panel = openai/gpt-5.5-20260423,
google/gemini-3.1-pro-preview-20260219, deepseek/deepseek-v4-pro-20260423; Claude = grounded panelist
+ sole judge. Total panel spend ~$0.286. Raw reviews + per-round judgments: `roundtable/`.

Goal (operator): get OTR stories to A+ or as close as the model ceiling allows; rank the option space
to the best candidates and sequence them. Result below: 1 gate + 4 build candidates, in 2 increments.

---

## The decision (ranked)

The quality apparatus already exists and is wired (critic, targeted reroll, structural escalation,
grouped-exchange, best-of-N outline selection, keep-best refine). It REMOVES flaws; it cannot
MANUFACTURE a good story because the one thing it never varies is the PREMISE. Root cause (triangulated
by two external reviews + prior roundtables): beat-planner / premise sameness -- every premise
collapses into a console standoff, climax off-stage, announcer narrates the outcome.

Ranked candidates (leverage on root cause / cost / risk):

1. **C1 -- Pitch room + greenlight (PRIMARY).** Generate 3 forcibly-divergent premises, taste-select
   one. The only lever that changes WHAT story gets told. Highest leverage.
2. **C2 Tier 1 -- Critic axes drive the existing refine loop.** Make the critic the pipeline already
   runs actually buy a better re-plan. Mostly wiring (the refine loop already re-outlines + recomposes).
3. **C4 -- Deterministic staging penalty.** Enforce the existing BEAT_ROLE "irreversible choice
   on-stage" so the climax lands on-mic. Fixes a named symptom premise-divergence will not.
4. **C3 -- Flip `use_exchange` (quick win).** Grouped-exchange dialogue is built + tested + dark;
   config-only after a GPU N=3. Independent of the others.

GATE: **C0 -- local-ceiling probe (do FIRST).** Cheap experiment that decides whether the campaign
can promise A+ on the local model or needs the frontier lane. **Operator decision required after C0.**

---

## Increment 1 (ship first). Build order: C0 -> [operator ceiling decision] -> C1 -> C4 -> C2 Tier 1; C3 in parallel.

### C0 -- local-ceiling probe (GATE)
- Standalone temp `generate_pitches()` (NOT C1's node), seeded from the raw `script_brief` + divergence
  seeds; outline; compose 5 episodes at a realistic word budget (NOT tiny -- graders penalize short for
  pacing; if you must shorten, inject a "short-format test, do not penalize length" note into the
  `grade_story` prompt for the probe only) -> `grade_story`. Grade the 3 best; near the 75 line, grade
  twice (pass = both >= 75). Log model id / seeds / temperature / word budget.
- Output: a recommendation to the operator. If some episode clears 75 (B), local is viable. If none do,
  the operator decides frontier-lane vs accept-B; if accept-B, relabel success = sameness reduction +
  median lift (rename the grade-label map). The operator sets `OTR_ENABLE_FRONTIER_GREENLIGHT` +
  `OTR_GREENLIGHT_MODEL` (env, HKCU pattern) read by the conductor; C0 does NOT auto-enable frontier.

### C1 -- pitch room + greenlight (in-conductor stage)
- `_otr_pitch_room.py`, called in `OTR_LedgerScriptWriter.run()` AFTER news briefs, BEFORE
  `generate_outline` (the `news_interpreter` pattern; injected `generate_fn`).
  Contract: `run_pitch_room(outline_req, *, generate_fn, local_model, frontier_cfg, seed_context, meta)
  -> (OutlineRequest, PitchMeta)`.
- Force divergence: each of 3 pitches seeds from the REAL domain/conflict palette in
  `_otr_story_quality_l12` (via `load_conflict_palette()` -- resolve the exact symbol first) PLUS one
  genre from {thriller, drama, sci-fi, noir} and one archetype from {reluctant hero, anti-hero, naive
  idealist, jaded veteran}, combined in a templated logline prompt.
- Schema (Pydantic via `structured_call`): `PitchCandidate(id:int unique in [1,2,3], logline,
  protagonist, antagonist_or_pressure, genre_mode, emotional_core, theme_sentence, final_20_seconds,
  conflict_type, setting_class, surprise:int, human_want:int, stageability:int,
  console_standoff_risk:int, why_different)`; `GreenlightDecision(selected_id:int, ranking:list[int]
  permutation, rationale)`. Validate selected_id in ids + ranking permutation + >= 3 valid candidates;
  max 2 regenerations then fall back to original `script_brief` + stamp pitch.status=failed_fallback.
- Greenlight resolution: LOCAL greenlight (same rubric prompt on the local model) is the DEFAULT.
  When `OTR_ENABLE_FRONTIER_GREENLIGHT` + `OTR_GREENLIGHT_MODEL` set, the taste call upgrades to
  frontier (timeout 30s, 1 retry, fail-CLOSED to local on disable/timeout/unparseable).
- Tie-break: lower `console_standoff_risk` wins, then lower id (numeric, not prose).
- Handoff: build a concise `script_brief` from the winner via a fixed template
  ("{logline} Protagonist: {protagonist}. Conflict: {conflict_type}, {setting_class}. Emotional core:
  {emotional_core}. Final 20s: {final_20_seconds}."), hard-truncate ~200 tokens, then
  `dataclasses.replace(outline_req, script_brief=brief)` (OutlineRequest is FROZEN). Stamp raw seed +
  full pitch in `meta.story_quality.pitch`.

### C4 -- deterministic staging penalty (build BEFORE C2 Tier 1)
- Computed INSIDE the best-of-N candidate loop (so it steers `select_best_outline`), POST-outline,
  PRE-composition. v1 = ONE high-signal deterministic rule (beat-turn heuristic CUT): the climax beat
  (the existing BEAT_ROLE "irreversible_choice-on-stage-as-the-last-beat", imported from
  `_otr_story_quality_l12`, not copied) must be the final voiced beat (character/announcer) with a
  non-empty decisive `intent`; else penalty.
- `_otr_staging_penalty(outline) -> float` (e.g. 50.0 when violated, else 0.0). Add optional
  `penalty: float | None = None` to `score_outline` + `select_best_outline`; subtract directly from the
  final score; `None` => byte-identical (audit ALL callers + regression test proving identity).

### C2 Tier 1 -- critic axes drive the refine loop
- The writer's `_refine_loop` already re-runs the full body (re-outline via `prior_macro` + recompose),
  grades, keep-best -- it is the re-plan mechanism. Swap/augment its revision trigger: after each pass,
  read the 5B critic output the freeze cascade already stamps and persist `failing_axes` +
  `regeneration_hint` to `meta.story_quality.critic_*`; build `prior_critique` from them (fall back to
  `grade_story.biggest_weakness` if critic absent). Same premise; the C4 staging penalty rides the
  re-outline selection. Turn `enable_critic_escalation` ON in the canonical workflow as part of this.
- Premise-axis handling in Increment 1: `decide_escalation_scope` returns EPISODE (existing) for
  `premise_clarity` -- do NOT add the PREMISE enum value yet (keeps exhaustive switches safe).
- Add a keep-best monotonicity smoke test (the loop is known non-monotonic) so the critique-source
  swap does not worsen it.

### C3 -- flip `use_exchange` (parallel, config-only)
- Verify the exact JSON field + precedence (writer runtime must not override). N=3 harness runs the
  canonical workflow with ONLY that diff and asserts effective `use_exchange=True`, VRAM <= 14.5 GB,
  and zero slot drift (defined: identical slot count/order/ids before vs after). Validate on a SEPARATE
  run from C1/C2 (one variable at a time). Pass -> config-only PR; fail -> stay OFF.

## Increment 2 (later sprint): C2 Tier 2 -- premise re-pitch

Only after Increment 1 shows the pitch room raises grades. Adds: `EscalationScope.PREMISE` end-to-end
(enum -> split `STRUCTURAL_AXES` into `PREMISE_AXES={premise_clarity, console_standoff}` vs
`EPISODE_AXES={resolution, emotional_arc, continuity}`, intercept premise_hits BEFORE the structural
block -> `_otr_freeze_cascade` routing -> meta serde -> exhaustion->keep-best -> regression cases); a
new `console_standoff` critic rubric axis; on PREMISE, the next refine pass DROPS the premise and calls
`run_pitch_room` with `PitchRequest(showrunner_note=regeneration_hint, excluded_fingerprints=...)`;
fingerprint = normalized (conflict_type, setting_class, norm(antagonist_or_pressure),
hash(final_20_seconds)) in meta; caps `OTR_STORY_REPITCH_MAX=1` / `OTR_STORY_REPLAN_MAX=2` as meta
counters that survive the refine reruns.

## Deferred (out of scope)

B3/B4 whole-scene/whole-episode prose -> ledger parser (separate spike; the risk is SILENT
mis-attribution, not a crash -- gate on deterministic attribution: speaker-prefixed draft or re-derive
beats + DIFF the outline, any unmatched line = loud halt). Multi-seed assignment desk, character
interviews, listener-taste critic, distinct-character-voices, external-repo survey (keep Open-Theatre
prompt-mining only), refine-loop hardening (its non-monotonicity is REAL regression -- fresh compose
each pass -- revisit after the levers land).

## Verify-at-build checklist

1. exact `..._PALETTE` + `BEAT_ROLE` symbol names + publicness in `_otr_story_quality_l12`; wrap behind
   `load_conflict_palette()` / a single BEAT_ROLE import.
2. `_refine_loop`'s per-pass result (`last`) exposes the stamped `meta.story_critic_report` (failing_axes
   + regeneration_hint) for the Tier-1 trigger swap; if not, add the meta plumbing.
3. `OutlineRequest` is `@dataclass(frozen=True)` and `dataclasses.replace(..., script_brief=...)` is valid.
4. macro-prompt length tolerance for the richer `script_brief`; set the hard cap before build.
5. every `score_outline` / `select_best_outline` caller updated for `penalty`; byte-identical when None.
6. `use_exchange` JSON field name + precedence; N=3 effective-config + VRAM + slot-drift assertions.
7. critic output field names are exactly `failing_axes` / `regeneration_hint` (else add an adapter).

## Grounded facts (verified by the judge; do not re-derive)

- `_refine_loop` (OTR_LedgerScriptWriter) re-runs the whole writer body per pass (re-outline via
  prior_macro + recompose), keep-best; non-monotonic by design (a live gemma pass went 72 -> 65).
- `OutlineRequest` (frozen) already has optional `script_brief` (takes precedence) + `diversity_hint`
  (best-of-N already varies "which stake opens the story, who"). Outline-level diversity EXISTS.
- `_otr_story_quality_l12` has a domain-keyed conflict `..._PALETTE` + a `BEAT_ROLE` sequence including
  "irreversible_choice-on-stage-as-the-last-beat".
- `_otr_reroll_escalation`: `EscalationScope{NONE,EPISODE,BEAT,LINE}`, `STRUCTURAL_AXES` frozenset
  (premise_clarity, continuity, resolution, emotional_arc), `enable_critic_escalation` default OFF,
  EPISODE -> needs_full_rerun (terminal).
- `_otr_story_select.score_outline(outline, meta, roster) -> StoryScore` is PURE; `select_best_outline`
  steers via `dataclasses.replace(outline_req, diversity_hint=hint)`.
- `grade_story`: B = 75, B+ = 80. `_otr_story_critic.run_story_critic` runs unconditionally (5B) in the
  freeze cascade; `_otr_reroll` 5C + `_otr_anti_loop` A3 + Wave-1C escalation are wired.
