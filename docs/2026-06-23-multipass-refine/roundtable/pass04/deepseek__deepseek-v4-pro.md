<!-- requested_model: deepseek/deepseek-v4-pro | resolved_model: deepseek/deepseek-v4-pro-20260423 -->

VERDICT: yes-with-fixes — plan is converged and build-ready; minor clarifications needed for implementor precision, but no build-blocking defects.

MUST-FIX BEFORE BUILD: None — plan converged.

SHOULD-FIX:
1. [score_outline] Clarify `roster` parameter: explicitly state it is the list of character names (excluding ANNOUNCER) from the locked cast, used by `premise_noun_palette`. The plan currently says `score_outline(outline, meta, roster)` but does not define `roster`; the implementor could infer it from the L1/L2 module, but explicit is safer.
2. [score_outline] Define `distinct_conflict_nouns` computation precisely: tokenize each voiced beat’s intent using the same `_TOKEN_RE` as `premise_noun_palette`, keep tokens that are in the grounded palette, and count distinct tokens across all beats. The current wording (“distinct premise-grounded content nouns”) is clear enough but could be misinterpreted as requiring a noun extractor.
3. [score_outline] Handle division by zero in `ungrounded_crisis_density`: if total voiced-intent words is 0, return 0.0 (or a sentinel). The plan does not specify edge-case behavior; a reasonable implementor would handle it, but stating it avoids ambiguity.
4. [diversity_hint] Ensure the `diversity_hint` is rendered in the user prompt only, and that the system prompt remains unchanged. The plan says “render it in `_otr_outline._build_user_prompt` only when non-empty” — this is sufficient, but a note that it must not alter the system prompt would prevent accidental drift.

OPTIONAL / NICE-TO-HAVE:
- The “legible grade” mapping (>= B is informational) is not essential for the selector’s operation; it could be cut to reduce scope. If kept, define the grade thresholds (e.g., A: all metrics excellent, B: above average, etc.) to avoid subjective interpretation.

CUT THESE:
- None — the plan is already lean.

VERIFY-AT-BUILD checklist (from earlier UNVERIFIABLE flags, now with concrete steps):
1. **RNG re-seeding per candidate**: Confirm that `torch.manual_seed` and `random.seed` before each `generate_outline` call actually affect the model’s sampling (no generator threading bleed). Test by running with `OTR_STORY_BEST_OF_N=2` and a fixed `cast_seed`; verify that candidate 0’s outline is identical to the flag-off outline, and candidate 1 is different.
2. **`count_ungrounded_crisis` non-zero on real outline**: Run the scorer on a production outline (flag-off) and assert `ungrounded_crisis_density > 0`; if zero, the metric cannot discriminate and the selector is useless.
3. **Flag-off byte-identity**: With `OTR_STORY_BEST_OF_N` unset/0, assert exactly one `generate_outline` call, no `meta.story_quality.best_of_n` key, and the generated outline prompt is byte-identical to the pre-change baseline (compare prompt strings).
4. **`build_sq_data` called once**: After the selector runs with N>1, verify `build_sq_data` is invoked exactly once (on the winning outline) and `_enrich_intent` does not double-append (check intent strings for repeated clauses).
5. **Local gate clamp**: With `creative_writing_model` set to an `openrouter:` or `comfy:` handle, assert N is forced to 1 before any candidate generation, and a LOUD log line is emitted.

[ASSUMPTION] The `generate_outline` function uses only the global torch/random RNG state for sampling; if it uses a separate generator or numpy, the seeding may not be effective. The verify-at-build step 1 will surface this.