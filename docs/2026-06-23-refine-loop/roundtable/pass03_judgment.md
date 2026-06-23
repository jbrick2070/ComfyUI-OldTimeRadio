# R3 judgment (wiring / integration / sequencing)

Panel: GPT-5.5, Gemini-3.1-pro, DeepSeek-v4-pro. Spend $0.1777 (cum $0.4498). GPT=no, Gemini=no,
DeepSeek=yes-w-fixes. Dense, well-grounded wiring catches.

## ACCEPTED (CONFIRMED against code -> folded into pass03)
- meta MUST be a DEEP-COPY of the INCOMING meta, not fresh/empty (Gemini#1, DeepSeek#7): score_outline ->
  premise_texts(meta) reads news/brief context. CONFIRMED. (Correctness catch.)
- Wire revision steering into MACRO + BEAT only, NEVER the phase prompt (speaker assignment) (Gemini#3,
  GPT#2). CONFIRMED phase prompt is speaker-only. Speaker-routing revision OUT OF SCOPE for v1.
- structured_call has NO `temperature` kwarg; use base_temperature/structural_retry_temperature/
  max_new_tokens(128)/max_attempts(2)/repair_prompt_factory/helper_name (GPT#5, DeepSeek#5). CONFIRMED.
- raw_outline = model_copy(deep) immediately after generate_outline; score on raw_outline; separate
  working_outline for build_sq_data/compose (GPT#3). prior_macro from raw_outline pre-mutation (Gemini#2,
  GPT#3, DeepSeek#3). CONFIRMED build_sq_data mutates intents.
- THREE seed points per pass: generate / compose / grade, namespaced (DeepSeek#2/#4, GPT#6). composer may
  sample -> seed before compose too.
- Frozen dataclass: no __post_init__ direct assign (GPT#13, Gemini#2) -> skip trimming there; normalize in
  critique_to_hint. CONFIRMED frozen.
- Winner-commit: helper touches NO self state; pass-local containers; commit once post-loop on winner
  (GPT#4). pass-0 MANDATORY baseline; CUT the "all passes failed" fallback (GPT#7/CUT#3) -- pass-0 success
  = automatic never-fail.
- Widget default Off (preserve byte-identical default-OFF + audio golden); B = recommended; absent widget
  => Off (old-JSON safe) (GPT#8). Operator decides default-on-B vs Off at delivery.
- best_of_n collision: bypass resolve_best_of_n/select_best_outline when refine>=2 (GPT#9). Provider remote
  => skip grading + single path (GPT#10).
- Grader: prepend SPEAKER to composed_text (Gemini SHOULD#1); ~4000-char cap + truncation (DeepSeek#1);
  grader_unparseable => normalized_hint="" (GPT#11); map .score->.score_0_100 (GPT#12).
- Narrow the prompt-injection regex (GPT SHOULD#5). Cancellation/interrupt hook between passes (GPT SHOULD#4,
  verify API). CUT meta_delta (GPT CUT#2), elapsed_s telemetry (GPT CUT#1), read-only canon deep-copy
  (Gemini CUT#1).

## VERIFY-AT-BUILD
- exact exception classes; ledger row->spoken-text shape; per-pass reseed re-rolls all 3 stages; canon
  read-only during compose; ComfyUI cancellation API; model_copy(deep) isolation.

## CONVERGENCE
R3 was all WIRING PINS + 2 correctness catches (meta deep-copy, phase exclusion) -- NO architecture change
(the REVISE design held). Proceed to R4 (convergence) to confirm no new MUST-FIX on pass03.
