# OTR Story Architecture -- Hardened Plan (R2 synthesis: implementability)

Candidate SET unchanged from R1 (it converged). R2 hardened the build details and corrected two
grounding errors -- one in the panel (the conflict palette is real), one in my R1 anchor (the outline
already has the handoff field). Real-symbol facts are now baked in so R3 can wire against them.

## Grounded facts (verified by the judge this pass; cite these, do not re-derive)

- `_otr_outline.OutlineRequest` (frozen dataclass) ALREADY has optional `script_brief` (takes
  precedence over raw news_seed) AND `diversity_hint` (best-of-N; the prompt already says "vary which
  stake opens the story, who..."). So outline-level structural diversity EXISTS -> the new lever is
  strictly PREMISE-level, above the outline.
- `_otr_story_quality_l12.py` HAS a domain-keyed conflict palette (`..._PALETTE`: education / energy /
  medicine / law / finance / environment / labor / space / astronomy / paleontology / agriculture ...,
  each with a concrete conflict noun + "a fight over ...") AND a `BEAT_ROLE` sequence including
  "irreversible_choice-on-stage-as-the-last-beat" (climax ON-stage). (Panel R2 MF "no palette" =
  MISREAD from an import line -> DISCARDED.)
- `_otr_reroll_escalation.EscalationScope` = {NONE, EPISODE, BEAT(~LINE today), LINE};
  `STRUCTURAL_AXES` frozenset (premise_clarity, continuity, resolution, emotional_arc, ...);
  `decide_escalation_scope` reads critic `verdict` / `failing_axes` / `regeneration_hint`; gated by
  `enable_critic_escalation` (default OFF). EPISODE -> `needs_full_rerun` (terminal, full rerun).
- `_otr_story_select.score_outline(outline, meta, roster) -> StoryScore` is PURE/frozen; takes NO
  penalty; `select_best_outline` steers via `dataclasses.replace(outline_req, diversity_hint=hint)`.
- `grade_story`: B = 75, B+ = 80 (not "75 = B+").

## Candidate 0 -- GATE: local-ceiling probe (FIRST; break the circular dependency)

- Implement a TEMPORARY local `generate_pitches()` (not the full Candidate 1 node) -> for each of ~10
  fixed seeds, take the greenlit premise -> outline -> compose ONE scene (not a full episode) ->
  `grade_story`. Grade outlines + one scene, NOT 10 full GPU episodes (panel: compute wall).
- Pass bar: any reach >= 75 (B). If NONE: STOP, escalate frontier-vs-accept-B to operator; if
  local-only, relabel success = sameness reduction + median lift (find + rename the grade-label map).
- Fixed seeds / temperature / model id logged; grade the best few TWICE (grader is itself LLM-noisy).

## Candidate 1 -- PRIMARY: pitch-room premise divergence + frontier greenlight

- **Force divergence** by seeding each of 3 pitches from the REAL `_otr_story_quality_l12` palette
  (domain/conflict) PLUS a new small genre + protagonist-archetype axis (the palette is domain-keyed,
  not genre-keyed -- add that axis as a local constant in the pitch module).
- **Schemas (Pydantic, parsed via the repo `structured_call` ladder):**
  `PitchCandidate(id:int, logline, protagonist, antagonist_or_pressure, genre_mode, emotional_core,
  theme_sentence, final_20_seconds, conflict_type, setting_class, why_different)`;
  `GreenlightDecision(selected_id:int, ranking:list[int], taste_rationale, risk_flags,
  failed_premise_fingerprints, brief_for_outline)`.
- **Greenlight** = explicit rubric (surprise/freshness, human want, audio-stageability, ending
  promise, OTR fit, console-standoff risk), forced ordinal ranking with deterministic tie-break
  (lower console-standoff risk, then id); require >= 3 valid candidates (else regenerate); short
  rationale per axis (bound length; evidence-quote is CUT -- parse fragility).
- **Frontier greenlight:** resolve via a new `OTR_GREENLIGHT_MODEL` (openrouter:...) reusing the
  existing OpenRouter slot + cost guard; fail-CLOSED to local if disabled/unparseable. Drafting may
  stay local; only the taste call goes frontier (gated by Candidate 0).
- **Handoff:** map the winning PitchCandidate into the EXISTING `OutlineRequest.script_brief` (it
  already takes precedence). No new outline field. Keep the brief concise enough that the macro prompt
  is not diluted (verify length tolerance at build).

## Candidate 2 -- close the critic -> re-plan loop, TWO tiers (behind enable_critic_escalation)

- Add `EscalationScope.PREMISE`. Map: `premise_clarity` (and console-standoff fingerprint) -> PREMISE
  (Tier 2, re-pitch); `resolution` / `emotional_arc` / `continuity` -> EPISODE (Tier 1, re-outline,
  SAME premise). Both stay behind `enable_critic_escalation`.
- Tier 1 (EPISODE) is HONESTLY a full rerun that reuses the premise brief and re-steers the outline
  via `diversity_hint` + the failing axis as a penalty (thread an optional penalty through
  `select_best_outline`; byte-identical when empty -- add the regression test). Not "cheap."
- Tier 2 (PREMISE) re-enters Candidate 1 with the critic `regeneration_hint` injected as a showrunner
  note + `failed_premise_fingerprints` excluded. Fingerprint = normalized tuple
  (conflict_type, setting_class, antagonist_class, hash(final_20_seconds)); stored in meta.
- Update `_otr_freeze_cascade` to route the new scope; cap PREMISE re-pitches (`OTR_STORY_REPITCH_MAX`,
  default 1) and EPISODE re-outlines (`OTR_STORY_REPLAN_MAX`, default 2) SEPARATELY; on exhaustion,
  keep-best. Phase: ship Tier 1 FIRST, add Tier 2 once pitch state + fingerprints exist (panel CUT).

## Candidate 3 -- flip `use_exchange` (B2): config-only after GPU N=3 (unchanged from R1)

## Candidate 4 -- staging enforcement (build on existing BEAT_ROLE; deterministic FIRST)

- POST-outline, PRE-composition (you cannot critique an outline before it exists -- panel).
- Start DETERMINISTIC: enforce the EXISTING `BEAT_ROLE` "irreversible_choice-on-stage-as-the-last-beat"
  (climax on-mic = final voiced beat is character/announcer with a decisive intent) + a beat-turn
  heuristic over `beat.intent`; feed as a numeric penalty INTO `score_outline` inputs (keep the scorer
  pure). Only add an LLM outline-critic if deterministic proves insufficient. May fold into C2 Tier 1.

## Deferred (unchanged): B3/B4 prose->ledger parser (separate spike, silent-mis-attribution gate),
multi-seed assignment desk, character interviews, listener-taste critic, distinct voices, repo survey
(keep Open-Theatre only), refine-loop hardening (diagnose grader-noise vs regression first).

## Verify-at-build / carry to R3 (wiring)

- Exact `..._PALETTE` symbol name + whether it is public in `_otr_story_quality_l12`.
- Where the pitch-room + greenlight live: new node(s) in the workflow JSON vs methods inside
  `OTR_LedgerScriptWriter` D.2->D.3 (news_interpreter sits between style-resolution and cast-lock).
- `_otr_freeze_cascade` routing for `EscalationScope.PREMISE`; the `enable_critic_escalation` widget.
- `select_best_outline` penalty-param threading + byte-identical regression.
- `use_exchange` N=3 harness; canonical workflow JSON config-only change.
