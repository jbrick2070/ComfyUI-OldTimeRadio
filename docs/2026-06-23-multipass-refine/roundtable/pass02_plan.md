# Local-Only Multi-Pass Structural Story-Refine -- R1-hardened plan (R2 input)

**Operator:** Jeffrey Brick. **Date:** 2026-06-23. Hardened from R1 (GPT-5.5 + DeepSeek-v4-pro + Claude
anchor; Gemini truncated). Operator confirmations folded in: rewrite the SPINE **in line with the story
seed** (same premise, new structure), then rewrite dialogue, loop until ~B+.

## Reframe (R1 convergence)
This is NOT "ship a loop." It is, in order:
1. **MEASURE FIRST (prerequisite, no new code).** Run the L1/L2 ON-vs-OFF soak already deferred to the
   operator and quantify residual cross-episode sameness (`meta.story_quality.ungrounded_crisis` density +
   distinct `conflict_object`/`type` counts). **If L1/L2 already collapses the sameness, this loop is
   CUT.** Only build if measurable residual sameness remains.
2. **IF residual remains: build a LOCAL-ONLY best-of-N SELECTION over seed-varied spines, gated by a
   DETERMINISTIC rubric, pre-freeze/pre-audio, keep-best.** Iterate-until-good is the v1 once selection
   proves the rubric discriminates.

## Purpose also = reliability (operator, 2026-06-23)
The loop is ALSO a never-hard-fail mechanism: rather than ABORT an episode whose story is weak, keep
refining (bounded by a hard cap) and ship the BEST candidate. "It may run a long time getting a good
story, but no failures." GUARDRAIL (reconciles with the 2026-06-16 no-fallbacks rule): "no failures" means
NO quality-floor ABORT -- it never masks a genuine ERROR. Real errors (crashes, missing models, malformed
ledger) STILL fail LOUD. The loop only removes the "story not good enough -> abort" outcome by replacing it
with keep-best-after-cap. The hard cap bounds "a long time"; keep-best guarantees a shippable result.

## The mechanism (v0 = selection; the operator's "loop until B+" is v1)
- **Grade = HYBRID, deterministic gate decides PASS/FAIL.** Reuse the signals L1/L2 already computes:
  `ungrounded_crisis` density (lower better), on-stage-climax present (the `irreversible_choice` last beat
  carries the decisive action), distinct conflict object/type, character-want clarity. Map to a legible
  letter (>= B == pass). The LLM does NOT assign the grade; it only PROPOSES the structural fix when the
  deterministic gate fails. (Kills "weak model grades itself a B" theater.)
- **"Rewrite the spine in line with the story seed."** The mutable artifact is the `Outline` (premise +
  beats[].intent/mood/arc_phase/speaker), anchored to the SAME `outline.premise` + story seed -- the
  rewrite changes STRUCTURE (beat functions / conflict), never the premise. Build ON TOP of L1/L2: after
  any spine rewrite, RE-APPLY `build_sq_data` so the grounded conflict + beat_role survive.
- **best-of-N (local, free text passes):** generate N candidate spines via seed-derived variation
  (`sha256(seed:n)`), score each by the deterministic rubric, KEEP-BEST. N text-only passes are cheap
  locally; no drift (selection, not iteration).
- **Re-slug = re-compose dialogue from the chosen spine** (selective: only beats whose intent changed),
  then the EXISTING `run_story_critic` + `run_targeted_reroll` act as the line-level second gate.
- **Audio rendered ONCE after the text converges (JUDGE NOTE).** The grade is on TEXT. Re-rendering TTS
  per pass is GPU-expensive and breaks the "passes are cheap" premise -- so the loop converges the text to
  >= B FIRST, then audio renders once. (Flagged for operator confirm; the alternative literal
  "rewrite audio each pass" is rejected on cost unless the operator overrides.)

## Wiring (pre-freeze, pre-audio)
- Runs on the COMPOSED LEDGER CANDIDATE, AFTER outline+compose, BEFORE the Phase-10 LFC freeze and BEFORE
  audio readiness (GPT: `run_story_critic` already sits pre-Phase-10; this loop sits at/just before it, at
  the structural layer). The freeze then hashes the FINAL converged text -> byte-identity holds when OFF.
- **Local-only gate:** read the RESOLVED writer model/backend (`resolved["creative_writing_model"]` +
  backend), not an env guess; if the writer is a paid frontier/OpenRouter model, the loop is DISABLED
  (single pass) so API spend never multiplies.

## Invariants (may not break)
- No disguised flag-and-reroll QA gate that just re-asks the weak model to "try again" on the same beats:
  the bright line is the rewrite is STRUCTURAL (new spine) + the GATE is DETERMINISTIC, not an LLM grade.
- Audio spine frozen; loop flag-gated default-OFF; golden re-baseline on enable.
- 100% local for the loop; determinism (seed-keyed, temp-0 improve call); LOUD on any pass failure;
  keep-best + hard cap; UTF-8 no BOM; SFW; zero workflow-JSON change unless a node is wired same-change.

## Open for R2 (coding plan)
- Exact mutable `Outline` patch path: rerun `generate_outline` with a structural constraint, vs patch the
  existing `Outline` in place, vs a new `StructuralRevision` step. Preserve budget/cast/`dialogue_slot_id`.
- The deterministic rubric's exact thresholds (what density / which signals == B).
- N for best-of-N; the keep-best comparator; the hard cap.
- How re-slug re-enters compose/critic without a double freeze.
