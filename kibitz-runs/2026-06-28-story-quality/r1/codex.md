VERDICT: no. The defect set is real, but the plan is not build-ready because S1/S5/G1 misstate existing architecture and the acceptance criteria do not prove narrative coherence.

MUST-FIX BEFORE BUILD:
1. [S1] False premise: “the per-line composer/brief never binds dialogue to the dramatic_question + wants” contradicts current code. `LineRequest` already carries `dramatic_question`, `beat_objective`, `beat_obstacle`, `beat_subtext`, etc. in `nodes\_otr_line_composer.py:877-883`; `_build_user_prompt` renders them in the DRAMATIC FRAME at `nodes\_otr_line_composer.py:1416-1504`; the writer threads them at `nodes\OTR_LedgerScriptWriter.py:4273-4279`. The `dance_of_keys` ledger also has `meta.line_dramatic_frame` populated. Concrete fix: rewrite S1 as “existing dramatic-frame / L12 grounding is not strong enough or is mis-aimed,” then specify which existing mechanism to tighten.

2. [G1] The plan says to “extend the 3.4 keep-better logic to the one_breath/anchor rerolls,” but that logic already includes both. `_QUALITY_HINT_PRIORITY` includes `one_breath` and `anchor_stuffing` at `nodes\_otr_line_composer.py:2287-2295`; `_quality_flags_for_line` flags them at `nodes\_otr_line_composer.py:2312-2321`; the reroll re-scores both drafts and keeps the fewer-defect result at `nodes\_otr_line_composer.py:2495-2528`. Concrete fix: change G1 from “add keep-better” to “change what better means,” e.g. grammar/semantic-damage scoring instead of flag-count-only.

3. [G1, §5] The lead goal mixes length and craft, but the measures mostly prove counter movement. `docs\2026-06-28-story-quality-kibitz\scan_soak.json` reports low final `anchor_stuffing_total` and `one_breath_violation_total` while scripts can still be bad; `plancks_vanishing_horizon` shows many `anchor_stuffing_retry,one_breath_retry,body_gate_reroll` flags and still ships malformed/noun-salad lines. Concrete fix: add a small golden-ledger acceptance set for the frontier/enrichment failures, with required before/after checks on final text quality, not just `length_ratio`.

4. [S5] “Thread each speaker’s speech_signature into the per-line prompt” is already implemented. Casting creates/diversifies `speech_signature` in `nodes\_otr_casting.py:1717-1720`; `build_voice_card` renders `speaks: ...` in `nodes\_otr_line_composer.py:1081-1130`; the system prompt tells the model to match it at `nodes\_otr_line_composer.py:1183-1187`; the writer passes `all_voice_cards` at `nodes\OTR_LedgerScriptWriter.py:3958-3960` and `4260`. Concrete fix: redefine S5 as an enforcement/measurement problem, not another prompt-threading feature.

5. [§0, S2] The “default-OFF / byte-identical” invariant is underspecified for news-coda changes. `compose_news_coda` has no `story_quality_v2` or subflag argument in `nodes\_otr_line_composer.py:3278-3326`; the writer calls it under `_style_grammar_on` at `nodes\OTR_LedgerScriptWriter.py:4769-4775`. Concrete fix: state the exact flag that gates S2 behavior, or explicitly scope byte-identical to flag-off paths only.

SHOULD-FIX:
1. [S3] Body-gate reroll acceptance is too narrow in the plan. The current writer accepts `_bg_res.text` when grounding validates at `nodes\OTR_LedgerScriptWriter.py:4528-4538`; that does not require “no hard compose flags” or “no run-on/grammar leak.” Concrete fix: define body-gate acceptance as grounding pass plus no hard leak/grammar flags, and stamp rejects.

2. [S2] “Premise-derived deterministic bridges” is a concept, not a design. `NEWS_CODA_POOL` is currently three generic prefixes at `nodes\_otr_line_composer.py:3210`; replacing it safely needs a deterministic template strategy and validator, not raw noun extraction. [ASSUMPTION] Otherwise this can just produce awkward noun-fragment bridges.

3. [§5] Add final-arc residuals to the scan. `scripts\story_quality_scan.py:440-456` aggregates line/coda/register counters, but not final `story_critic_report.arc_verdict`, `reroll_verdict`, or “needs_full_rerun” rate, even though ledgers like `dance_of_keys` and `ledger_ink_runs_dry` ship with `reroll_verdict="needs_full_rerun"`.

OPTIONAL / NICE-TO-HAVE:
- [S4] A deterministic de-cliche rewrite is reasonable, but define the rewrite floor narrowly so it cannot flatten good idiomatic lines.
- [§1] Keep the enrichment episodes in the census artifact or cite them as separate fixtures; right now `scan_soak.json` is the 29-leg soak, while the G1 lead evidence leans heavily on enrichment runs.

CUT THESE (scope / over-engineering):
1. [S6] Cut from the first build. Phantom detection already exists and is observable through `phantom_name:*` compose flags in `nodes\_otr_line_composer.py:2627-2635`; the plan itself ranks S6 low.

2. [S5] Cut from the first G1/S2/S3 chunk. The prompt-threading part already exists; text-level voice enforcement should be a later pass after the gate/reroll degradation is fixed.

3. [§6] Cut the operator-question block from the build plan. Convert these to explicit defaults for R2, with only true product choices escalated.