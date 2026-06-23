# Local-Only Multi-Pass Structural Story-Refine Loop -- design brief to harden

**Operator:** Jeffrey Brick. **Date:** 2026-06-23. **Status:** pre-build concept; roundtable input.

## The idea (operator's words, normalized)
A pre-audio LLM stage that REFINES the whole episode, iterating until the story is good:

1. **Grade pass.** One LLM call reads the WHOLE frozen ledger (all beats + lines, in order) and grades
   it on a CONCRETE, legible rubric -- "is this a good story? 10th-grade-English-class B or above, yes/no?"
   A single threshold (>= B) is the gate, not a vibes score.
2. **Improve pass (only if below threshold).** A second call asks "how do we make this story better?" and
   returns concrete, structural feedback (not line nitpicks).
3. **Update the SPINE.** Apply that feedback by rewriting the OUTLINE/BEAT SPINE -- the structural layer --
   not just rewording individual lines. (This is the key: the defect is structural sameness; rephrasing
   lines cannot escape it.)
4. **Re-slug.** Re-generate the dialogue ("slugs"/lines) from the UPDATED spine.
5. **Loop** back to the grade pass; repeat until >= B or a hard cap.

**HARD OPERATOR CONSTRAINT:** this loop runs ONLY when the writer models are LOCAL (gemma-12b /
mistral-nemo via Ollama or llama-server). Local passes are essentially free, so multiple passes /
best-of-N are cheap -- cost is NOT a constraint here. If the run uses a paid frontier writer
(OpenRouter), the loop is DISABLED (a single pass) so we never multiply API spend.

## Why this is on the table (grounding the panel MUST weigh)
- **This session's 4-frontier-model + operator panel UNANIMOUSLY concluded a flag-and-reroll QA gate does
  NOT fix the cross-episode "console standoff" sameness** -- every premise collapses to people fighting
  over a lever/key/console + countdown, climax off-stage. The weak local model, asked to reroll, rephrases
  the SAME underlying beat plan. That verdict is WHY this session built deterministic UPSTREAM beat-shaping
  instead (premise-anchored conflict objects/types + a beat_role dramatic-function sequence with the
  irreversible_choice on-stage as the last beat + crisis-noun grounding) -- shipped as L1/L2 behind
  `OTR_STORY_QUALITY_L12`.
- **The crucial distinction for THIS idea:** the operator's loop updates the SPINE (structural) and
  re-slugs -- it is NOT line-level reroll. That directly targets the panel's objection ("you can't escape
  the standoff by rewording"). The open question is whether a WEAK local model can actually produce a
  BETTER spine, or whether it just reshuffles into another standoff.
- **A bounded line-targeted critic already exists pre-audio:** `run_story_critic` (reads the whole frozen
  ledger) -> `run_targeted_reroll` (re-composes flagged LINES, re-scores, capped at MAX_REROLL_CYCLES).
  This new idea is a SUPERSET at the structural layer. The panel must say how they relate (replace? layer
  above? feed each other?) and how to avoid two competing loops.
- **The audio spine is FROZEN** (`test_audio_byte_identical`). Anything that rewrites text pre-audio
  changes what ships -> it must be flag-gated default-OFF with a deliberate golden re-baseline, exactly
  like L3/L4.

## Questions for the panel to converge
1. **Worth building at all?** Given the panel's own "reroll-until-good doesn't work on weak models" finding
   -- does moving the rewrite to the SPINE level + a concrete rubric threshold change that verdict? Or is
   this the same trap one layer up?
2. **Structural vs cosmetic.** Exactly how does "update the spine" work so it is genuinely structural
   (different beat functions / different conflict) and not a reshuffle into another standoff? What is the
   minimum structural delta that counts as "better"?
3. **Rubric + convergence.** Is a single "10th-grade B+ yes/no" gate enough, or does it need a small fixed
   rubric (e.g. premise-specificity / on-stage climax / character want clarity / non-genericness)? Hard cap
   on passes? A NO-REGRESSION guard so a later pass can't ship worse than an earlier one (keep-best)?
4. **best-of-N vs iterate-until-good.** Since local passes are free: generate N spines and keep the
   best-graded (selection), or iterate-and-improve a single spine (refinement), or both?
5. **Not-a-disguised-QA-gate test.** The operator's hard rule is NO flag-and-reroll gate. State the bright
   line that keeps this loop on the right side of it (the answer is probably: it REWRITES the spine
   deterministically-seeded from feedback, it does not just re-ask the same model to "try again").
6. **Wiring.** Where does it sit relative to (a) the existing `run_story_critic`/reroll loop, (b) the new
   L1/L2 upstream levers, (c) the freeze cascade + audio? Does it run BEFORE freeze (so the freeze hashes
   the final text) and BEFORE audio (so byte-identity holds when off)? Local-only gate: read the resolved
   writer model/backend, not an env guess.
7. **Determinism + invariants.** Seed-keyed so a fixed seed reproduces the same final story; 100% local;
   LOUD on any pass failure; UTF-8 no BOM; SFW; flag default-OFF => byte-identical; zero workflow-JSON
   change unless a node is wired in the same change.

## Invariants the panel may NOT break
- No disguised flag-and-reroll QA gate on the weak model that just rephrases the same beats.
- Audio spine frozen; loop is flag-gated default-OFF; golden re-baseline on enable.
- 100% local for this loop (disabled on paid frontier writers); determinism (seed-keyed); LOUD fallbacks.
- Canonical workflow `workflows/otr_scifi_16gb_full.json` is the only graph; any node/widget change goes IN
  it in the same change as the code.
