# R2 judgment (coding plan / implementability)

Panel: gpt-5.5-20260423, gemini-3.1-pro-preview, deepseek-v4-pro. Spend this pass: ~$0.0955
(grounding excerpts added input tokens). Claude = grounded panelist + judge.

## Discarded after grounding (judge overrides panel)

- "No conflict palette in `_otr_story_quality_l12` / invented API" (GPT MF3, Gemini MF1): MISREAD. The
  panel inferred from the IMPORT line in `_otr_story_select.py` (3 helpers), not the file. The file HAS
  a domain-keyed `..._PALETTE` + a `BEAT_ROLE` sequence (climax-on-stage). Plan now cites the real
  symbol; adds only a genre/archetype axis on top (the palette is domain-keyed, not genre-keyed).

## Accepted (folded into pass02_plan.md)

- EscalationScope has no re-plan-tier values -> add `PREMISE`; map premise_clarity->PREMISE(re-pitch),
  resolution/emotional_arc/continuity->EPISODE(re-outline). Behind `enable_critic_escalation`
  (default OFF). (GPT MF7/8, Gemini MF2/3, DeepSeek MF4, anchor MF3.)
- Tier 1 re-outline is a FULL rerun (reuse brief, re-steer diversity_hint + penalty); not cheap.
  Thread an optional penalty through `select_best_outline`; byte-identical regression test. score_outline
  stays PURE. (Unanimous + anchor MF2/MF4.)
- Candidate 4: deterministic FIRST (enforce existing BEAT_ROLE + beat-turn heuristic as a score_outline
  penalty input), LLM critic only if needed; POST-outline/PRE-composition (Gemini MF4: cannot critique
  before generating). (GPT CUT1, DeepSeek MF5, anchor MF4.)
- Candidate 0 circular dependency + compute wall -> temp local generate_pitches; grade outlines + ONE
  scene, not 10 full episodes; grade best few twice. (GPT MF5, Gemini SF1, DeepSeek SF6.)
- grade threshold fix: 75=B, 80=B+ (GPT MF6, grounded).
- Concrete schemas: PitchCandidate.id + GreenlightDecision + fingerprint tuple + OTR_GREENLIGHT_MODEL
  + caps OTR_STORY_REPITCH_MAX/REPLAN_MAX. (GPT MF1/2/10/SF7, DeepSeek MF1/2.)
- Greenlight: require >=3 valid, tie-break, drop evidence-quote (parse fragility). (GPT SF1/2/CUT3.)

## Corrected my own R1 anchor

- Handoff does NOT need a new outline field: `OutlineRequest.script_brief` exists, is optional, and
  "takes precedence". Reuse it. (anchor MF1 overstated the risk; the 350-char cap is on NewsBriefs,
  not OutlineRequest.)

## Convergence call

Candidate set stable (R1). R2 added no new candidates, only build hardening -> proceed to R3 (wiring:
node placement, freeze_cascade scope routing, use_exchange JSON, sprint order).
