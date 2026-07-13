# Why did five production bugs in one day, in one lane, all come from the same place?

**Date:** 2026-07-13
**Repo:** ComfyUI-OldTimeRadio, lane `original_codex56sol` (a 9-pass structured-LLM
story pipeline: slate -> triage -> truth map -> fair-play audit -> score -> script ->
blind listener -> retake -> final audit; each pass is a typed pydantic artifact with a
3-rung repair ladder).
**Status:** open question for a frontier panel. Not a coding plan.

## The record (all live, all today, all in one lane)

| # | Bug | Killed by | Cause |
|---|---|---|---|
| 10 | P9 final audit could BLOCK on the manifest -- a Python-compiled artifact no retake can change | dead end at the last gate, 570s in | a gate whose authority exceeded its repair route |
| 11 | `clue_plan` `min_length=3` stated in no prompt; `_repair_rules` had no P1 branch | ladder exhausted, 121s in | a bound the model was never told |
| 12 | P4 seam said "set blocking=false"; model wrote that STRING into `detail` and dropped the boolean field | ladder exhausted | a prompt that phrased a field as prose |
| 13 | my new coordinate gate fail-closed on `truth_map.` prefix, then on a `thread_id` shared by two collections BY DESIGN | two dead episodes | **a guard I added while fixing #12** |
| 14 | P4 graded "clue-before-reveal order" -- but `AudibleTruthMap` has no order to grade | blocked 3/3 runs; retake could never satisfy it | an audit grading a property its artifact cannot express |

**Bugs 12, 13, and 14 were introduced or exposed by the fixes for 10-12.** Each fix was
individually reasonable, passed a full 7,800-test suite, passed a local 2-agent review
panel (4 rounds), and then killed a live episode.

## The question for the panel

Not "was each fix correct" -- they were, individually. The question is **why this system
keeps producing this failure shape, and what discipline would have prevented it**.

Candidate patterns I can see, and I want them attacked:

1. **Guards are cheap to add and expensive to be wrong about.** Bug 13 is the purest
   case: I added a fail-closed validator whose verdict the repair path never reads. It
   could not improve any outcome, and it killed two episodes. Proposed rule: *before
   adding a fail-closed check, name what a caller would DO differently with its answer;
   if nothing, it is not a guard, it is a liability.* Is that the right rule? Is it
   enough?

2. **Every pass is an LLM grading an artifact, and nobody checked whether the artifact
   can answer the question.** Bug 14: the fair-play audit asked about ordering; the truth
   map has no ordering. Proposed rule: *an audit may only grade properties its artifact
   can express.* How would you enforce that mechanically rather than by good intentions?

3. **The model-visible contract and the Python-enforced contract drift apart silently.**
   Bug 11 (a bound in no prompt) and bug 12 (a field phrased as prose). We now have a
   test that every bounded field is named in the seam or the repair rules. Is that the
   right invariant? What is the general form -- "the set of rules Python enforces must
   equal the set of rules the model is told"? Can that be generated rather than tested?

4. **A repair route that always fires is a second authoring pass in disguise.** The P4
   retake fired on 3/3 runs. That was the signal that exposed bug 14 -- but only because
   I happened to look. What metric should make this visible automatically?

5. Is the 9-pass, audit-heavy architecture itself the problem? Every audit pass is a
   place for an LLM to invent a verdict that a deterministic validator then has to
   adjudicate. Would fewer, larger passes with stronger deterministic validation be
   strictly better, or does the audit chain earn its keep?

## Constraints (non-negotiable, from the operator's rules)

- Python may never author story content. Models own prose, causality, clue placement.
- Deterministic validators are the authority; a model verdict may not overrule one.
- Fail-closed must survive for genuinely unrepairable defects. No silent coercion.
- Root-cause fixes only. No shims, no retry-count inflation, no cap inflation.
- Every live failure must remain reproducible and admissible as a logged production bug.

## What I want back

Rank the patterns by how much future breakage each would prevent. Name the ONE discipline
change with the best ratio of breakage-prevented to effort. Attack my proposed rules --
especially #1 and #2 -- and say where they would misfire. If the architecture itself is
the root cause, say so plainly and say what you would replace it with.
