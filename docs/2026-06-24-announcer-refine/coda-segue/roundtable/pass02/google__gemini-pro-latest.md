<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: no. Unimplementable validation rules, missing function arguments for rerolls, and a contradiction in how the LLM gets the "news subject" without leaking facts. 

MUST-FIX BEFORE BUILD:
1.  **[Key Open Question / Design 1] The "News Subject" Extraction Paradox.** You cannot pass a "short news subject/noun-phrase" because no such variable exists in the `run()` scope, and extracting one requires another LLM call. If you pass the full `news_close_brief`, the weak LLM will bleed facts into the bridge, defeating the Core Pivot.
    *   *Fix:* Do NOT pass `news_close_brief` to the bridge prompt at all. Pass only the fiction context (`script_brief`, `ending_change`). Instruct the LLM to write a transitional pivot *away* from the fiction. The appended payload provides 100% of the news specificity.
2.  **[Design 4 & 5] Unimplementable ">=1 news content token" rule.** If the bridge LLM doesn't see the news brief (per Fix 1), it cannot reliably generate a matching token. Even if it did, "news content token" is undefined programmatically without an NLP library.
    *   *Fix:* Drop the ">=1 news content token" requirement. Rely entirely on the blacklist to prevent generic openers, and the LLM's fictional context to make the pivot specific to *tonight's drama*.
3.  **[Design 3] Starving the Pivot Context.** You instruct to "EXCLUDE `ending_change` from the coda LLM call". If the LLM doesn't know how the fiction ended, it cannot write a coherent bridge pivoting *away* from tonight's drama.
    *   *Fix:* Pass `ending_change` to the bridge prompt so it understands the fictional resolution, but SUPPRESS the instruction at :2854 ("State this outcome plainly...") when `_style_grammar_on` is true.
4.  **[Design 5] Unimplementable "does not assert an outcome" validation.** A programmatic validator cannot evaluate semantic assertions without another LLM call. 
    *   *Fix:* Remove this from the python validator (`validate_news_coda_bridge`). Enforce it strictly via the system prompt and a tight character limit (e.g., `max_chars=60`).
5.  **[Design 5 / R2 Wiring] Missing `cast_seed` for Rerolls.** You specify "One reroll (altered seed) on failure." The target function `compose_announcer_outro` (:2778) does not currently accept `cast_seed` or `seed`.
    *   *Fix:* Add `cast_seed: int` to the signature of `compose_announcer_outro`. On validation failure, pass `cast_seed + 1` to the retry call.
6.  **[Design 6] Broken Fallback Routing on Empty Brief.** You state: "EMPTY `news_close_brief` => ... skip news-coda mode, use the existing outro fallback". If there is no news brief, the system should generate a standard *fictional* outro, not force a fallback.
    *   *Fix:* If `news_close_brief` is empty, bypass the News Coda logic entirely and execute the existing `_ANNOUNCER_OUTRO_SYSTEM` happy-path generation.

SHOULD-FIX:
1.  **[Design 6] Rotating Pool Implementation.** To ensure deterministic, safe selection from the fallback pool, define it as a hardcoded tuple.
    *   *Fix:* `pool = ("The real story:", "The true account:", "From tonight's headlines:")`. Select via `pool[hash(str(cast_seed)) % len(pool)]`.
2.  **[Design 2] Payload Cleaning.** `news_close_brief` might not be capitalized or punctuated correctly to follow a bridge clause.
    *   *Fix:* Ensure `news_close_brief_clean` has its first character capitalized if the bridge ends in terminal punctuation, or lowercased if the bridge ends in a comma/colon.

OPTIONAL / NICE-TO-HAVE:
-   **Bridge Length Cap:** Enforce a strict `max_chars=80` on the bridge in `validate_news_coda_bridge` to physically prevent the LLM from rambling into factual territory.

CUT THESE:
1.  **[Design 4] ">=1 news content token"** - Safe to cut because the appended payload guarantees news specificity, and the blacklist prevents generic bridges.
2.  **[Design 5] "does not assert an outcome (best-effort)"** - Safe to cut from the programmatic validator; it's unimplementable without an LLM-in-the-loop.