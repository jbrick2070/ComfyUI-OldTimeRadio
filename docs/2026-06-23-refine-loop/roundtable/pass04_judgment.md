# R4 judgment (convergence / residual defects) -- CONVERGED

Panel: GPT-5.5 + Gemini-3.1-pro (DeepSeek-v4-pro ERRORED: empty content / reasoning-token exhaustion).
Spend $0.0704 (campaign total ~$0.5202). Both returning models = yes-with-fixes with only spec-tightening
left -> CONVERGED (no architecture change; the REVISE design held all 4 rounds).

## ACCEPTED (folded into pass04_FINAL)
- Use `grade.score_0_100` everywhere (not `.score`) (GPT#1).
- Pass-count LOCKED: effective_passes = TOTAL incl pass 0; disabled=1; enabled min 2; clamp
  REFINE_MAX_PASSES=5 total; loop range(effective_passes) (GPT#2).
- Collision/disabled EXPLICIT branches: refine off => existing path incl v0 best-of-N preserved; refine on
  (>=2) => bypass best-of-N + assert no key (GPT#3).
- Grade-fail semantics LOCKED: ok=True (shippable), score_0_100=0, grade_error_type, normalized_hint="";
  error_type only for gen/compose fail; comparator includes all ok=True (GPT#4).
- grade_delta = score - pass0_score (grade-fail uses 0) (GPT#5).
- Grade-fail pass STILL revises (macro "improve structure, preserve spine" variant) -- honors "always a
  revision" (GPT SHOULD#4). Good catch.
- Truncation EXACT: first 2000 + "\n...\n" + last 2000 (Gemini#1, the one build-blocker).
- critique_to_hint strips newlines -> single line (Gemini OPTIONAL).
- Widget default LOCKED Off; removed the "or default B" ambiguity; B = recommended pick (GPT SHOULD#6/CUT#2).
- Cancellation = required chunk-4 step w/ verify (GPT#6, Gemini verify).
- CUT RefineCandidate.title (winner runs existing title regen) (GPT CUT#4).
- canon: test losing candidate leaves canon equal; deep-copy only if mutated (GPT SHOULD#3, Gemini verify).
- target_reached:bool in telemetry (GPT OPTIONAL). per-pass progress log + cap_reached_below_bar warn
  (Claude anchor #1/#2). 4-combo overlay byte-identical test (Claude anchor #3). chunk-2 blocks on ledger
  row shape (Claude anchor #4, GPT SHOULD#5).

## REJECTED / kept
- GPT CUT#3 (drop `provider` from telemetry): KEPT -- cheap + useful for the soak audit. Minor.

## DeepSeek R4 failure
Not a plan defect -- a reasoning model spent max_tokens on hidden reasoning (finish_reason=length). R1-R3
DeepSeek reviews stand; R4 had GPT + Gemini + the Claude anchor = sufficient convergence quorum.

## CONVERGENCE CALL
CONVERGED. pass04_plan_FINAL.md is build-ready (6-chunk order 0-5) pending the verify-at-build checklist.
Stop the campaign (do not grind a 5th pass). Await operator GO to build.
