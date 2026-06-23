# R2 judgment (coding plan / implementability)

Panel: GPT-5.5, Gemini-3.1-pro, DeepSeek-v4-pro. Spend $0.1790 (cum $0.2721). GPT=no, Gemini=yes-w-fixes,
DeepSeek=no. Grounding `_otr_outline.py` paid off with the round's biggest catch.

## HEADLINE (CONFIRMED against real code)
- GPT#5: production `generate_outline` is Path C (`_build_macro/phase/beat_user_prompt`); `_build_user_prompt`
  is back-compat/test-only (L1023 banner; callers L2311/2329/2390). => **shipped v0 `diversity_hint` is DEAD
  (rendered only in `_build_user_prompt`)** -- best-of-N candidates varied only by RNG seed (explains the
  smoke TIE). Folded: v1 prior_critique wires into Path C; build chunk 0 fixes v0's dead diversity_hint.

## ACCEPTED (folded into pass02)
- Grader needs the LLM callable (`generate_fn`) + temperature=0.0 for determinism (Gemini#2, DeepSeek#3/#4,
  GPT#9). Route via `structured_call` + pydantic schema. CONFIRMED structured_call exists.
- Early-stop uses `grade.score` (0-100), NOT the structural score (DeepSeek#1). CONFIRMED.
- Exact keep-best key tuple (GPT#7, DeepSeek#2) -- pinned with max(...).
- Call `generate_outline` directly, not `select_best_outline` (it reseeds with `:outline:0` + is best-of-N)
  (Gemini#3). CONFIRMED via v0 _seed_rngs.
- Flag collision: refine>=2 forces best_of_n effective_n=1 (Gemini#1, GPT#4). CONFIRMED.
- Pass isolation must deep-copy ALL mutable inputs (meta/canon/ledger/roster), defer telemetry to winner
  commit (GPT#1/#2). CONFIRMED meta read by score_outline + telemetry merge.
- `composed_lines` extraction rules (spoken rows only) (GPT#9, Gemini ASSUMPTION). CONFIRMED ledger is rows.
- StoryGrade frozen + clamp 0..100 + fallback score=0/"grader_unparseable" (GPT SHOULD#1).
- prior_critique appended as final OutlineRequest field (GPT#6). CONFIRMED frozen dataclass + append rule.
- Provider gate matches v0 EXACTLY -- drop my "fail-closed unknown handle" (no such predicate exists)
  (GPT#3). CONFIRMED.
- critique sanitize vs prompt-injection + word-boundary trim (GPT SHOULD#2, Gemini SHOULD#1).
- `outline.model_copy(deep=True)` for pydantic v2 (GPT SHOULD#3).
- Byte-identical test compares the 3 Path C PROMPTS + ledger, not just suite (GPT SHOULD#6/#9).
- grade_delta None at pass0 (all 3). meta is a dict (GPT#10) CONFIRMED from smoke.

## CUT (consensus)
- `OTR_STORY_REFINE_MAX_SECONDS` (hard cap suffices; re-add if soak shows long passes) (GPT CUT#1);
  no-improve early-stop (opt-in, default OFF -- also operator's pref) (GPT CUT#3).

## VERIFY-AT-BUILD
- exact exception classes (compose vs config) for the taxonomy (GPT#8); per-pass reseed actually re-rolls
  Path C + grader; cast_seed in scope; row->spoken-text extraction; model_copy isolation.

## CONVERGENCE
R2 surfaced the Path C / dead-diversity_hint catch (material) + many implementability pins -> NOT
converged; proceed to R3 (wiring) on pass02. The architecture pivoted to REVISE (operator) -- R3 must
harden the Path C revision wiring + the dropdown widget -> JSON.
