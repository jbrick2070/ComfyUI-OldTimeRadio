# R4 judgment (convergence / residual defects)

Panel: gpt-5.5-20260423, gemini-3.1-pro-preview, deepseek-v4-pro. Spend this pass: ~$0.0666.
All three returned yes-with-fixes (zero "no") -> the plan has CONVERGED on direction, candidate set,
and wiring. Residual items were SPEC-precision, all folded into ../SPEC.md.

## Accepted (folded into SPEC.md)

- Build-order bug: C4 staging must precede C2 Tier 1 (Tier 1 uses the penalty). Reordered. (DeepSeek MF1,
  GPT SF1.)
- Temporary PREMISE routing contradiction: do NOT add the PREMISE enum in Increment 1; premise_clarity
  -> EPISODE (existing). PREMISE enum is entirely an Increment 2 concern. (Gemini MF3, GPT MF5.)
- Penalty math: `penalty: float|None`, subtracted from final score, None=byte-identical; staging penalty
  = fixed value when the irreversible-choice climax beat is not on-mic. (GPT MF7, Gemini MF4, DeepSeek MF6.)
- Durable flag: operator sets OTR_ENABLE_FRONTIER_GREENLIGHT + OTR_GREENLIGHT_MODEL (env); C0 only
  recommends. (GPT MF1, DeepSeek MF2/MF9.)
- Tier-1 critic wiring: persist critic failing_axes/regeneration_hint to meta; refine loop reads them for
  prior_critique. (GPT MF4, DeepSeek MF5.)
- C0 quantified: 5 pitches, grade best 3, grade-twice near the line; address grader length-bias (realistic
  budget or a short-format directive). (GPT SF2/3, Gemini MF2, DeepSeek MF3.)
- Tie-break ascending console_standoff_risk then id; frontier timeout 30s. (Gemini MF1/SF1, GPT SF5.)
- Concise script_brief template + ~200-token cap. (DeepSeek MF7, GPT SF6.)
- keep-best monotonicity smoke after the critique-source swap. (GPT SF10, DeepSeek SF2.)

## CUT (final sweep)

- C4 beat-turn heuristic -> CUT; keep only the irreversible-choice-on-mic rule. (GPT CUT3, DeepSeek CUT1.)
- C0 outline-only pre-filter -> CUT. (GPT CUT4.)
- console_standoff critic axis + all PREMISE/fingerprint machinery -> Increment 2.

## Convergence call

CONVERGED at R4 (no "no" verdicts; no new candidates across R2-R4; only precision fixes). Stopping at the
4-round arc per CLAUDE.md S8 -- no 5th pass. Deliverable: ../SPEC.md. Total panel spend across R1-R4
~$0.286 ($0.0706 + $0.0955 + $0.0534 + $0.0666).
