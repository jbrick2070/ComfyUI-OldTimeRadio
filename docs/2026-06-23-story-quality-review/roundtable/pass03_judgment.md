# R3 judgment (Claude, judge)

Panel: GPT-5.5, Gemini-3.1-pro, Grok-4.3 (DeepSeek returned empty/length -- dropped this round) (~$0.091).
Grounded against `GROUNDING_R3.md`. R3 caught build-breaking WIRING bugs; all folded into pass03_plan.md.

ACCEPTED (folded):
- L5a as framed is impossible: `too_many_edits` is terminal + rolls back the ledger before the critic runs
  (GPT/Gemini/Grok, grounded at freeze_cascade:599/730). Recast: scale `compute_edit_cap` + advisory grade-only
  critic before the terminal stop + downstream tolerates missing report.
- Pydantic leak: `Beat` default fields serialize via `model_dump()` -> ride `beat.meta` or `Field(exclude=True)`
  (Gemini #2, GPT #2).
- Telemetry blind-overwrite at scrub:1006 -> `.setdefault().update()` (Gemini/GPT #11); aggregate from the
  final persisted rows (Grok #3 -- ties EP16 undercount to the same restore-ordering root cause).
- Flag gates the MUTATION not just the prompt; flag-off = no population = no drift (GPT #10).
- Adding fields != threading them: update the LineRequest call site; sequence the new role validator AFTER
  deterministic population + fallback, preserving first-failure (GPT #3/#5, Grok #1/#2).
- personal_stake: no structured field exists -> the deterministic (speaker,domain) table is MANDATORY, write to
  beat.meta (Gemini #3, GPT #7).
- L1 crisis-repair: drop the arbitrary cap, repair ALL ungrounded nouns, field allowlist only (Gemini cut / GPT #8).
- L3 `ACTION:` marker + insert before persistence; L4 final-text-only before freeze/TTS (GPT #9).
- Explicit outline build sequence adopted (GPT #3).

CUT (R3-confirmed): choice_summary outro family (Grok -- polish), L1 cap, L4 mojibake, structured personal_stake
discovery. DEFERRED: L5b, L6.

JUDGE NOTES: Grok's "OutlineBeat separate from Beat" assumption is FALSE -- grounding shows only `class Beat`
(:84); the beat class is `Beat`. Residual verify-at-build carried: outline->ledger serialization path (decides
exclude=True vs meta-only); whether allowed_people/allowed_things are populated today (L1a).

Convergence call: R3 surfaced NEW must-fixes (cascade sequencing, Pydantic, telemetry merge) -> NOT converged
at R3. R4 confirms whether the wiring-corrected plan has residual blockers.
