<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. The design is close, but the current wiring spec has ordering and interface holes that can break the build or violate the “bridge never sees outcome / fallback deterministic” requirements.

MUST-FIX BEFORE BUILD:
1. [CALL-SITE WIRING] The “EARLY BRANCH” can be ordered before variables needed by the existing outro path are built. Grounding says `_outro_final_char_line` is built at writer :4619-4623 and `_outro_ending_change` is also part of the current outro call at :4615-4634. If the new `if _style_grammar_on ...` branch is inserted before those assignments, the `else` path cannot pass existing args unchanged. Concrete fix: either place the branch after the existing `_outro_final_char_line` / `_outro_ending_change` construction, or move those computations inside the `else` block before calling `compose_announcer_outro`.

2. [NEW FUNCTION] / [CALL-SITE WIRING] / [NON-NEGOTIABLE] The spec assumes `script_brief` is “SETUP, not outcome,” but the grounding only shows `script_brief = briefs.script_brief`; it does not prove that this field excludes outcome. Grounding also notes `fallback_announcer_intro(script_brief)` echoes `script_brief` verbatim, so `intro_text` can also leak outcome if `script_brief` contains it. This directly breaks “bridge never sees the fictional outcome.” Concrete fix: verify `briefs.script_brief` construction. If it can include resolution/outcome, create/pass a setup-only field instead, and omit or sanitize `intro_text` when it was generated from an outcome-bearing brief. [ASSUMPTION] The risk depends on unshown `briefs.script_brief` construction.

3. [VALIDATOR] The validator contract is internally inconsistent: `validate_news_coda_bridge(text) -> (ok, cleaned)` is specified, but one required check needs the cleaned `news_close_brief` for the “no >=5-token verbatim run copied from the cleaned brief” guard. The bridge validator cannot perform that check with only `text`. Concrete fix: change signature to `validate_news_coda_bridge(text, *, news_close_brief="")` and clean the brief inside, or cut the n-gram guard entirely.

4. [ASSEMBLY] / [FALLBACK FLOOR] The fallback snippet returns `f"{prefix} {fact}"`, but `fact` is only defined in the assembly snippet, not in the fallback snippet. If implemented literally inside the failure branch before fact assembly, this is an undefined-variable/order bug. Concrete fix: in `compose_news_coda`, clean/cap `news_close_brief` into `fact` at the top, before LLM generation, validation, reroll, or fallback.

5. [FALLBACK FLOOR] `abs(hash(str(cast_seed))) % len(POOL)` is not deterministic across Python processes because built-in `hash()` is salted. This violates [NON-NEGOTIABLE] deterministic fallback. Concrete fix: use stable hashing, e.g. `int(hashlib.sha256(f"news-coda:{cast_seed}".encode("utf-8")).hexdigest(), 16) % len(POOL)`.

6. [ASSEMBLY] / [VALIDATOR] The bridge is required to end with `:` or an em dash, while existing announcer prompts in grounding forbid colons for normal intro/outro lines. If implementation reuses existing announcer line validation/sanitization, it may reject every valid coda bridge or strip the delimiter. Concrete fix: implement a coda-specific sanitizer/validator that allows a bridge-final `:` or `—`, but still rejects leading speaker labels like `ANNOUNCER:`. Do not reuse `validate_announcer_line` unless verified compatible. [ASSUMPTION] Existing validator behavior is not shown, but current intro/outro prompt contracts conflict with the coda delimiter requirement.

7. [EMPTY news_close_brief] / [CALL-SITE WIRING] The no-brief path says “also stamp `news_coda_no_brief`,” but `compose_announcer_outro` must remain untouched and the off/no-brief text path must stay byte-identical. Concrete fix: append the flag only at the caller after receiving `outro_res`, without changing the text. Verify whether `LineResult` is mutable/frozen; if frozen, use `dataclasses.replace(outro_res, compose_flags=outro_res.compose_flags + ("news_coda_no_brief",))` or construct a new `LineResult` with identical text. verify: `LineResult` definition.

SHOULD-FIX:
1. [NEW FUNCTION] Although the caller is supposed to skip coda mode when `news_close_brief` is empty, `compose_news_coda` itself should still guard against empty/whitespace input. Concrete fix: if cleaned `fact` is empty, return `LineResult(text="", compose_flags=("news_coda_no_brief",))` or raise a local `ValueError` caught by the caller; do not produce `"The real story:"` with no fact.

2. [REROLL] The retry prompt must not add `news_close_brief` or outcome fields while asking for more specificity. Concrete fix: append only to the existing setup-only user prompt: `"Attempt 2 -- use different wording; be more specific to the tale."` Do not include the cleaned fact in retry messages.

3. [ASSEMBLY] Normalize the bridge delimiter before final concatenation and then re-run the coda-specific one-line/label/cap checks. Concrete fix: strip trailing `.?!,;`, append `:` or ` —`, then assemble `f"{bridge} {fact}"`.

4. [CALL-SITE WIRING] Keep the feature under `_style_grammar_on` only. Grounding says `_apply_story_scaffold_env(...)` and `_style_grammar_on = _OTRCFG.style_grammar_enabled()` are the single flag path. Concrete fix: do not introduce a second env var or independent news-coda switch unless it is derived from `_style_grammar_on`.

5. [VERIFY-AT-BUILD] Confirm `nc_brief` is read before the outro branch and is the same real journalistic note currently passed as `news_close_brief` into `compose_announcer_outro`. Grounding says it is read from `meta["news"]` at writer :3949 and call happens at :4615-4634, but the exact local variable assignment is not shown here.

OPTIONAL / NICE-TO-HAVE:
- Add unit tests for: happy bridge, reroll success, fallback success, empty brief skip, process-stable fallback index, and bridge with colon not rejected as a speaker label.
- Add a test where `script_brief` contains `ending_change` text to ensure the coda prompt does not receive it.

CUT THESE (over-engineering):
1. [VALIDATOR] Cut the news-brief n-gram copy guard unless you are willing to pass `news_close_brief` into the validator. The bridge never sees the brief by design, so the guard is low-value and currently creates an interface mismatch.

2. [VALIDATOR] Cut the optional outcome-verb blocklist if it causes false positives or requires more prompt plumbing. The hard requirement is keeping outcome data out of the prompt; a tiny verb list is only weak defense after the fact.