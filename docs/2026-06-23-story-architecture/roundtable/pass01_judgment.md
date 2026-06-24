# R1 judgment (arc/creative)

Panel: openai/gpt-5.5-20260423, google/gemini-3.1-pro-preview-20260219, deepseek/deepseek-v4-pro-20260423.
Spend this pass: ~$0.0706. Claude = grounded panelist (claude_anchor.md) + sole judge.

## Accepted (folded into pass01_plan.md)

- Cut B3/B4 prose->ledger parser from the campaign -> separate spike (UNANIMOUS + anchor). Failure mode
  reframed: silent mis-attribution, not crash; spike needs deterministic-attribution gate.
- Two-tier escalation: Tier 1 re-OUTLINE same premise (staging failure) vs Tier 2 re-PITCH new premise
  (premise unsalvageable). Gemini MF1 -- corrects the kickoff's + my anchor's single-tier route. KEY.
- Inject critic report (arc_verdict/flat_lines) into the re-pitch as a showrunner note (DeepSeek MF1).
- Greenlight node defaults to FRONTIER lane even if drafting stays local (Gemini SF1 + GPT MF6).
- Force premise divergence via conflict-type/protagonist/setting seeds from `_otr_story_quality_l12`
  (anchor MF2 + Gemini/DeepSeek assumptions).
- Local-ceiling PROBE as Candidate 0, done FIRST (DeepSeek SF4 + GPT MF3) -- a $0 experiment.
- Concrete I/O contracts: PitchCandidate[] + GreenlightDecision (GPT MF1).
- Cut multi-seed "3 headlines" from MVP -- 3 takes from one script_brief (GPT MF5).
- Fold "theme & ending first" into PitchCandidate FIELDS, not a standalone step (GPT SF1).
- Split success hypotheses: premise divergence (C1) vs staging (C4 outline critic) (GPT MF7).

## Rejected / downgraded

- None of the panel's code-fact claims were hallucinations; no discards. Panel critiqued PLAN
  underspecification (legitimate), not false code facts.
- "Route ALL structural failures to the pitch room" (kickoff S2 + my anchor SF3): DOWNGRADED -- too
  blunt; replaced by the two-tier split.

## Verify-at-build (carried to R2/R3)

- script_brief richness compatible with `_otr_outline`/`score_outline`.
- escalation EPISODE branch accepts Tier1 re-outline / Tier2 re-pitch without breaking the cascade.
- planner determinism degree (tighten "same shape" -> "same shape class").
- use_exchange N=3 harness + pass/fail.

## Convergence call

R1 CONVERGED on the candidate SET + cut list (3 build candidates + 1 gate + 1 supporting). The open
items are coding-plan + wiring concerns -> proceed to R2 (implementability) then R3 (wiring).
