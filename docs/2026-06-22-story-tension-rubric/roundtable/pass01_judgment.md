# R1 judgment -- STEP 5/6 design

## Convergence: YES (R1). All three panelists + anchor agree on the core; the
## panel ADDED real grounded material (the reroll-reconstruction gap), now folded.

## Accepted (grounded CONFIRMED)
- Meta dramatic-frame stamp as the single mechanism for tension delivery + critic
  visibility + reroll reconstruction. [anchor, Gemini#1, DeepSeek#1, GPT#1]
- Reroll reconstruction gap: `build_reroll_line_request:339` OMITS the dramatic
  frame -- CONFIRMED by reading the real construction. [GPT#2] HIGH VALUE.
- Use beat_intent, NOT line_job, in the rubric (line_job not in critic input). CONFIRMED. [GPT#3]
- failed_dimension OPTIONAL on RerollTarget only, default "unspecified", back-compat
  with clean()/_coerce_report. [GPT#4, GPT#9]
- NO reroll dimension->hint mapper (critic hint already actionable). [DeepSeek#2, GPT#5]
- Tension judged by APPROPRIATENESS to target level, not raw intensity; add level
  defs to the prompt. [GPT#8] -- prevents flagging a calm setup line.
- One deterministic tension curve: character-beat ordinal ramp, n<=1->3 else
  round(1+4*i/(n-1)) clamped, peak at final beat, no easing. [GPT#6, DeepSeek#3]
- Over-flag calibration: reroll only when a line fails BOTH advancement AND a
  dimension. [GPT#10]

## Rejected / cut
- Easing / falling action (no falling phase exists). [GPT, DeepSeek, anchor]
- Deterministic flatness code gate (stays LLM). [all]
- Dual FlatLine+RerollTarget enum (one authoritative field). [GPT#1 cut]
- SceneArcContext (already confirmed unnecessary). [all]
- "climax phase" language -- budget phases are setup/complication/resolution. [GPT#3 should]

## Verify-at-build
- run_targeted_reroll FUNCTION docstring still says "restored" (STEP-4 left the
  module + disposition docstrings updated but maybe not this one). [GPT SHOULD#6]
- short-episode fixtures (1/2/13 beats). render_priority completeness validator (defer).

## Panel mechanics note
GPT-5.5 + DeepSeek returned empty at the 2000-token default (reasoning models burn
budget on hidden reasoning); re-ran those two at --max-tokens 12000 (both fine).
Gemini answered at 2k but its saved output was truncated after MUST-FIX 1 -- its
VERDICT ("data-flow conflict between the frozen ledger and the shared line
renderer") still corroborates the core fix.

## Spend
R1 ~$0.21 total (pass1 ~$0.0866 + reasoning re-run ~$0.1192). No R2 (converged).
