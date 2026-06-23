<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: yes-with-fixes. The pass-isolation and loop boundaries are solid, but missing LLM handles and flag collisions will break the implementation.

MUST-FIX BEFORE BUILD:
1. [Lead design / Flags] Flag Collision. The plan does not specify how `OTR_STORY_REFINE_PASSES` interacts with the existing `OTR_STORY_BEST_OF_N`. If both are > 1, the system either blows up O(N*M) or creates ambiguous state. Fix: Explicitly mutually exclude them in `resolve_refine_passes` (e.g., if `effective_passes >= 2`, force `best_of_n = 1` and log a warning).
2. [The loop - Step 5] Missing LLM Handle in Grader. `grade_story(composed_lines, premise, meta)` is defined as a "local LLM reflection call" but lacks the callable to actually run the model. Fix: Pass `generate_fn` (or the equivalent LLM handle used by `generate_outline`) into `grade_story`.
3. [The loop - Steps 1 & 2] Seed Overwrite vs. Error Handling. If the loop calls `select_best_outline(n=1)` to get the outline, `select_best_outline` will internally overwrite your `refine:{i}` seed with its hardcoded `sha256(f"{cast_seed}:outline:0")`. If the loop calls `generate_outline` directly to avoid this, it loses the `OutlineFailedError` try/except safety net. Fix: Call `generate_outline` directly in the refine loop and wrap it in `try/except OutlineFailedError -> LOUD log + skip pass`.

SHOULD-FIX:
1. [The loop - Step 6] Blind String Trimming. `critique_to_hint` trims `biggest_weakness` to `<= 200 chars`. A hard slice will cut words/sentences in half, feeding garbage syntax into the next pass's prompt. Fix: Enforce the length limit in the `grade_story` LLM JSON schema (e.g., `Field(max_length=200)`), or trim to the last complete punctuation mark.
2. [Telemetry] Undefined `grade_delta` baseline. The telemetry specifies `grade_delta` per pass, but pass 0 has no prior pass to delta against. Fix: Define `grade_delta` as `score - passes[i-1].score` (and `None` or `0` for pass 0).

OPTIONAL / NICE-TO-HAVE:
- Include the elapsed wall-clock time per pass in the telemetry array to help operators tune `OTR_STORY_REFINE_MAX_SECONDS` during the validation soak.

CUT THESE (over-engineering):
1. [None] The R1 consensus already cut the major over-engineering risks (remote best-of-N, nested loops, ledger retention). The remaining scope is appropriately lean. 

[ASSUMPTION] I am assuming `_build_and_compose` has access to the full line-composer dependencies (which are normally in `OTR_LedgerScriptWriter.run()`) and that `composed_lines` passed to `grade_story` is a list of plain text strings extracted from the ledger rows, not the raw audio/metadata objects.