# R4 judgment (Claude, judge) -- CONVERGED

Panel: GPT-5.5, Gemini-3.1-pro, Grok-4.3 (DeepSeek empty/length again) (~$0.081). All 3 verdicts =
"yes-with-fixes"; the remaining items are SMALL/specific (pin a formula, name a flag, delete a stale mention) --
NO new architecture-level must-fix. Lever set stable since pass02. **CONVERGED -- stop the arc (no R5).**

ACCEPTED (folded into pass04_plan_FINAL.md):
- DROP the advisory grade-only critic: `run_story_critic` needs `generate_fn`, absent in the reviewer module
  (Gemini #2 / GPT #3 / Grok, grounded `_otr_freeze_cascade.py:756`). L5a = scale `compute_edit_cap` only +
  downstream tolerates a missing `story_critic_report`.
- `Beat` has NO `meta` field (verified `_otr_outline.py:84-135`) and defaulted/excluded Pydantic fields risk
  JSON/schema drift -> hold SQ per-beat data in a WRITER-SIDE `dict[beat_id->sq]`, zero Beat change (judge
  synthesis over Gemini's exclude=True + GPT's schema caution).
- Pin `compute_edit_cap = max(3, min(12, ceil(voiced_beats*0.6)))` with test values (GPT #2).
- Telemetry: `setdefault().update()` + count from saved rows; define the flag contract; keep action_split vs a
  new action_strip counter separate (GPT #1/#4, Gemini optional).
- Delete the stale `choice_summary` mention; pin acceptance key `meta.story_quality.ungrounded_crisis=
  {matches,total}`; L1 mutable target = `beat.intent` only (Grok cleanups).
- Mojibake -> verify/no-mutation test only (GPT/Gemini).

REJECTED/NONE: no panel claim was discarded as a misread this round; all were valid refinements.

Total campaign spend: A $0.111 + R1 $0.056 + R2 $0.095 + R3 $0.091 + R4 $0.081 = ~$0.434.
Final coder kickoff: `pass04_plan_FINAL.md`. Residual = a short verify-at-build checklist (not blockers).
