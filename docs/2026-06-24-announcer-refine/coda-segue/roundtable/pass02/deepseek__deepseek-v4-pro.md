<!-- requested_model: deepseek/deepseek-v4-pro | resolved_model: deepseek/deepseek-v4-pro-20260423 -->

VERDICT: yes-with-fixes — the split design is implementable, but the plan omits critical concrete details (prompt text, subject extraction, validator thresholds, pool list) that must be specified before any code can be written.

MUST-FIX BEFORE BUILD:
1. [Design point 2 / Open Question] Missing `_NEWS_CODA_SYSTEM` prompt content. The plan says to create it but provides no text. Must define the exact system prompt string with rules: OTR host voice, pivot from drama to real subject, no factual claims, no moralizing, etc.
2. [Design point 2 / Open Question] Missing news subject extraction. The plan says to pass a short news subject/noun-phrase to the bridge prompt but does not specify how to derive it from `news_close_brief`. Must implement a function `extract_news_subject(brief: str) -> str` that returns a short phrase (e.g., first sentence truncated to N words, or a simple noun-phrase extraction).
3. [Design point 2] Missing bridge prompt user content template. The plan does not define what story context (script_brief, intro_text) to include in the user message alongside the news subject. Must specify the exact user prompt template, e.g., "Tonight's story: {script_brief}\nReal-world subject: {news_subject}\nWrite a bridge clause..."
4. [Design point 4] Missing validator specificity: ">=1 news content token" requires a definition. Must specify how to tokenize the news subject and check for presence in the bridge (e.g., any word from the subject phrase appears in the bridge). Also, the blacklist of generic openers must be defined as a concrete list of strings.
5. [Design point 6] Missing fallback rotating prefix pool. The plan says "a small CLOSED prefix pool" but does not list the phrases. Must define the pool (e.g., ["The real story:", "The true account:", "From tonight's headlines:"]).
6. [Design point 2] Missing de-dup logic against appended brief. The plan says "de-dup against the appended brief" but provides no algorithm. Must specify a check: if the bridge contains a substring of the cleaned brief longer than, say, 10 words, it's invalid and triggers a reroll.
7. [Design point 5] Missing length caps. The plan says "bounded length" for the bridge and "brief cleaning/length cap" but no numbers. Must define max character/word limits for the bridge (e.g., 40 words) and for the cleaned `news_close_brief` (e.g., 200 chars) to prevent LLM rambling or truncation.
8. [Design point 5] Missing compose_flags taxonomy. The plan mentions new flags like `news_coda_bridge`, `news_coda_fallback`, `news_coda_no_brief` but does not define them. Must define these flag strings explicitly so they can be used in `LineResult.compose_flags`.
9. [Design point 3] Missing integration point specification. The plan says to create a separate `_NEWS_CODA_SYSTEM` prompt and suppress the resolved-fiction branch, but does not specify the new function signature or how it replaces the existing outro call. Must define `compose_news_coda(creative_fn, news_close_brief, script_brief, intro_text, cast_seed, ...) -> LineResult` and the exact branching logic at the call site (around :4615-4634) when `_style_grammar_on` and `news_close_brief` non-empty.
10. [Design point 3] Missing gating condition. The plan says "behind `story_scaffold` / `_style_grammar_on`" but does not specify where to check the flag. Must add a condition at the outro composition point to branch to news-coda logic only when `_style_grammar_on` is true and `news_close_brief` is non-empty; otherwise, use existing outro.

SHOULD-FIX:
1. [Design point 5] The "does not assert an outcome (best-effort)" check should be backed by a list of outcome-asserting phrases (e.g., "resulted in", "the outcome was") to make the validator more robust.
2. [Design point 2] The bridge prompt should include the opening line (`intro_text`) to allow tone echoing, as the existing outro prompt does. Specify that `intro_text` is passed to the bridge prompt.
3. [Design point 5] The reroll mechanism should specify how the seed is altered (e.g., appending "_retry1" to the seed string) to ensure deterministic behavior.
4. [Design point 6] The `news_coda_no_brief` flag should be included in the `compose_flags` of the fallback `LineResult` when `news_close_brief` is empty, for observability.

OPTIONAL / NICE-TO-HAVE:
- Define a "specificity threshold" (e.g., minimum bridge length in words) to avoid overly terse bridges, but the length cap and content token check may suffice.
- Add a check that the bridge does not contain the entire `news_close_brief` verbatim beyond the de-dup substring check.

CUT THESE:
- Any reference to a fixed `NEWS_CODA_LEAD_IN` constant; if it exists in the codebase, remove it; if not, no action needed. The plan already replaces it.
- The old STEP F logic (validate body has no lead-in, prepend lead-in) is superseded; ensure no dead code remains.

[ASSUMPTION] The functions `clean_one_line`, `_announcer_generate`, and the creative function’s seed-alteration mechanism exist as implied by the grounding; the plan does not need to redefine them.