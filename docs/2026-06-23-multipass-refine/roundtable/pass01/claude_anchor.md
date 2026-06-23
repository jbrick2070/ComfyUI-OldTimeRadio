# Claude anchor review -- R1 (high-level arc / creative coherence)

Grounded against the real repo (Desktop Commander, Windows venv): `_otr_outline.py`,
`OTR_LedgerScriptWriter.py`, `_otr_freeze_cascade.py`, `_otr_reroll.py`, `_otr_story_critic.py`,
`_otr_story_quality_l12.py` (this session). Labels: CONFIRMED / MISREAD / UNVERIFIABLE.

## VERDICT
PROMISING-WITH-MAJOR-CAVEATS. The structural-spine-rewrite framing is materially stronger than the
line-reroll the panel already rejected, and the local-only "passes are free" constraint removes the cost
objection that killed best-of-N (L6) in v0. But the load-bearing assumption -- that a weak local model can
GRADE its own story honestly AND produce a genuinely better SPINE -- is unproven and is exactly where this
can collapse back into "rephrase the same standoff at the outline layer."

## MUST-FIX (R1)
1. **The grader and the writer must not be the same weak model judging itself, or the rubric is theater.**
   CONFIRMED the writer resolves a single local model (`resolved["creative_writing_model"]`, e.g.
   gemma-12b) for creative work and a technical slot for critique. A 12B model asked "is my own story a
   B?" tends to say yes. R1 question: does the rubric need to be DETERMINISTIC/measurable (reuse the L1/L2
   `ungrounded_crisis` density + on-stage-climax presence + distinct-conflict signals we already compute)
   rather than an LLM letter grade? A hybrid -- deterministic gate decides PASS/FAIL, LLM only proposes the
   structural fix -- is far more robust than an LLM letter grade.
2. **"Update the spine" must be defined as a STRUCTURAL delta, or it is cosmetic.** CONFIRMED the spine is
   `_otr_outline.generate_outline` -> `Outline.beats` (each `Beat`: speaker/intent/arc_phase/target_words).
   "Better spine" has to mean different beat FUNCTIONS or a different conflict object -- not the same beats
   with nicer intents. This is precisely what this session's deterministic L2 `beat_role` sequence already
   imposes. The loop must build ON TOP of L1/L2, not re-litigate it.
3. **Convergence + no-regression is non-negotiable.** Iterate-until-good on a weak model WILL wander. Need
   a hard cap AND keep-best (a later pass can never ship worse than the best-graded earlier pass). Without
   keep-best this degrades to mush -- the panel's original failure mode.

## SHOULD-FIX (R1)
- Prefer best-of-N SELECTION over open-ended iteration as the v0: since local passes are free, generate N
  spines (seed-varied) and keep the best by the deterministic rubric. Selection cannot drift; iteration
  can. The operator's "loop until good" is the v1 once selection proves the rubric discriminates.
- The local-only gate is good cost hygiene but note the irony: local IS the weak writer, so the loop is
  most needed exactly where the grader is least trustworthy. This strengthens MUST-FIX 1.

## CONFIRMED facts
- `run_story_critic` + `run_targeted_reroll` (capped at MAX_REROLL_CYCLES) already run pre-audio at the
  freeze cascade -- a bounded LINE-level loop. CONFIRMED.
- L1/L2 (`_otr_story_quality_l12.build_sq_data`) shapes the beat plan deterministically upstream; default
  OFF. CONFIRMED (shipped this session, HEAD 41aed49).
- Audio spine frozen; text-changing work is flag-gated default-OFF with golden re-baseline (L3/L4 pattern).
  CONFIRMED.

## UNVERIFIABLE (verify-at-build)
- Whether a re-slug from an updated spine re-enters the SAME freeze/scrub/critic path cleanly (no double
  freeze) -- needs the wiring round (R3).
