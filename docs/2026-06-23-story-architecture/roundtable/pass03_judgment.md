# R3 judgment (wiring / integration / sequencing)

Panel: gpt-5.5-20260423, gemini-3.1-pro-preview, deepseek-v4-pro (grounded on _otr_reroll_escalation.py).
Spend this pass: ~$0.0534. Claude = grounded panelist + judge.

## Accepted (folded into pass03_plan.md)

- Escalation routing COLLISION (grounded): premise_clarity is already in STRUCTURAL_AXES->EPISODE;
  must split into PREMISE_AXES vs EPISODE_AXES and intercept premise_hits BEFORE the structural block.
  (GPT MF4, Gemini MF1.)
- console-standoff must be a named critic AXIS, not a fingerprint-flag input (decide_escalation_scope
  reads only verdict/failing_axes/regeneration_hint). (GPT MF5.)
- score_outline penalty = explicit optional kwarg (byte-identical when None), audit ALL callers + regr;
  NOT via mutable meta. (GPT MF6, DeepSeek MF4; overrides Gemini SF2's meta-dict suggestion.)
- OutlineRequest is FROZEN -> dataclasses.replace for the handoff. (Gemini MF3.)
- C4 penalty computed INSIDE the best-of-N loop (pre-selection), integrated into C2 Tier-1 re-outline.
  (GPT MF7, DeepSeek MF3.)
- Tier 2 plumbing: pitch-room INPUT object (showrunner_note + excluded_fingerprints); fingerprint via
  meta; antagonist normalized; caps as meta counters that survive reruns. (GPT MF2/MF3/SF5, DeepSeek MF2.)
- PREMISE enum end-to-end order; temporary PREMISE->EPISODE until Tier 2 ships (no crash). (GPT MF1,
  DeepSeek SF1.)
- Local greenlight fallback must be a real mechanism (same rubric on local model); timeout+retry+
  fail-closed; validate selected_id/ranking/>=3. (DeepSeek MF1, GPT SF1, Gemini SF1.)
- C0: own mini-pitch (not C1 greenlight); durable OTR_ENABLE_FRONTIER_GREENLIGHT flag; grade composed
  short episodes (grade_story needs story shape) with outline-grade as a cheap pre-filter; grade twice.
  (Gemini MF2/CUT1, GPT MF10/MF11, DeepSeek SF2.)

## Phasing (accepted CUTs -> sprint order)

- Tier 2 PREMISE re-pitch -> LATER sprint (GPT CUT1, DeepSeek SF1). Tier 1 ships first (largely exists).
- LLM outline-critic -> deferred; deterministic staging first (GPT CUT2).
- failed_premise_fingerprints output field -> only when Tier 2 lands (GPT CUT3).

## Judge note (my own grounding this pass)

`_refine_loop` re-runs the full writer body each pass (re-outline via prior_macro + recompose),
keep-best -- so Tier 1 "re-outline same premise" is the EXISTING mechanism; the new work is swapping
the revision trigger from grade-weakness to the 5B critic axes. This de-risks Candidate 2 materially.

## Convergence call

No new candidates; R3 produced only wiring/sequencing fixes (all grounded, no hallucinations). Plan is
near-final -> R4 to confirm no residual must-fix, then SPEC.md.
