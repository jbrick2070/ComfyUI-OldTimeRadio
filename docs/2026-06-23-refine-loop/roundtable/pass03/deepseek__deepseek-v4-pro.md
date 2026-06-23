<!-- requested_model: deepseek/deepseek-v4-pro | resolved_model: deepseek/deepseek-v4-pro-20260423 -->

VERDICT: yes-with-fixes. Several wiring gaps (missing grader truncation spec, composition seeding, macro format, grader seed, token limits, exception taxonomy, meta content) must be resolved before build.

MUST-FIX BEFORE BUILD:
1. [Grader section] The `grade_story` spec lacks a concrete max-char cap and truncation/summarization algorithm for `composed_text`. A missing cap risks context overflow or inconsistent grading. Define the exact character limit (e.g., 4000 chars) and the truncation rule (e.g., keep last spoken lines, simple head/tail).  
2. [Revision loop / Sequencing] Per-pass determinism is broken: the plan seeds before `generate_outline` and before the grader, but NOT before line composition. If the line composer uses any randomness (sampling, temperature>0), identical outlines may produce different composed texts, making the grader score non-deterministic and the keep-best comparison unreliable. MUST-FIX: either explicitly seed the RNG before composition with the per-pass seed, or guarantee the composer is deterministic when `temperature=0`.  
3. [Revision overlay wiring] The format of `prior_macro` is undefined. `_build_macro_user_prompt` will embed it, but no structure is given. MUST-FIX: specify that `prior_macro` is a string like `"Title: {title}\nPremise: {premise}\nSetting: {setting}"` (or another fixed format) so that the REVISE block is well-formed.  
4. [Grader / Sequencing] The grader’s seed derivation is not specified. The plan says “Seed before the call” but omits the seed string. MUST-FIX: seed the grader with `_seed_rngs(f"{cast_seed}:grade:{i}")` immediately before the `grade_story` call to ensure determinism across passes.  
5. [Grader / structured_call] The `structured_call` for grading does not set `max_new_tokens`. This may default to a large value, wasting tokens or causing failures. MUST-FIX: set `max_new_tokens` to a small value (e.g., 100) suitable for the grading schema.  
6. [Failure taxonomy] The plan says “Verify-at-build: enumerate the exact exception classes” but does not list them. The never-fail logic (compose-fail skip, missing-model LOUD) depends on knowing which exceptions are which. MUST-FIX: provide a concrete list (e.g., `OutlineFailedError`, known `ValueError` from composer for compose-fail; `RuntimeError`, `OSError` for missing-model) before building chunk 3.  
7. [Pass isolation] The fresh `meta` dict used during a pass must contain the keys that `score_outline` → `premise_texts(meta)` expects (e.g., `"news_seed"`, `"style"`). If those are missing, the scorer will crash. MUST-FIX: ensure the per-pass `meta` copy includes those fields from the writer’s current state.

SHOULD-FIX:
- [Grader] Set `max_attempts` for the grading `structured_call` to 1 or 2 to avoid costly retries on a low-stakes parse.  
- [Grader] Note that the grader re‑uses the creative `generate_fn`; a future optimization would be a separate (faster) grader model, but not block‑worthy.

OPTIONAL / NICE-TO-HAVE:
- Plateau early‑stop default OFF (already cut, can re‑add later).
- Configurable grader model via env var.

CUT THESE (over‑engineering):
- (None beyond the already‑cut items listed in the document.)

[ASSUMPTION] The line composer does not introduce its own random calls; if it does, the composition seeding fix becomes a MUST.