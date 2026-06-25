<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. Open activation/data-shape questions and missing call signatures make this not build-ready; fallback/reroll/cast_seed wiring is undefined against the grounded current APIs.

MUST-FIX BEFORE BUILD:
1. [NON-NEGOTIABLE / R3 WIRING FACTS] News-coda activation is undefined. Current `compose_announcer_outro(...)` has no `_style_grammar_on`, `story_scaffold`, or `news_coda_enabled` parameter, and the call site at :4615-4634 only passes existing outro inputs. Concrete fix: add an explicit keyword such as `news_coda_enabled: bool = False` to `compose_announcer_outro`, pass `_style_grammar_on` from the writer call site, and keep the old code path byte-identical when false.

2. [DESIGN 6 / R2 WIRING FACTS] Deterministic fallback requires `cast_seed`, but `compose_announcer_outro` currently has no `cast_seed` parameter. The fallback cannot be keyed as specified. Concrete fix: add `cast_seed: int | None = None` to `compose_announcer_outro`, pass the in-scope `cast_seed` from :2878 / :4615-4634, and define fallback behavior when `cast_seed is None` for backward compatibility.

3. [DESIGN 5] “One reroll (altered seed)” is not implementable as written. Grounding shows `_announcer_generate(creative_fn, messages)` is called without any seed argument; no API for changing seed is shown. Concrete fix: either verify and cite a real `creative_fn` seed API, or implement reroll by changing only the prompt content deterministically, e.g. append `Attempt: 2. Use a different bridge wording.` Do not claim altered seed unless the API exists.

4. [KEY OPEN QUESTION / DESIGN 4-5] The bridge subject/content-token contract is unresolved. `>=1 news content token`, “short news subject/noun-phrase,” and “de-dup against appended brief” have no data shape, source, or algorithm. Concrete fix: define `news_subject: str` source and fallback extraction from `news_close_brief_clean`, plus exact tokenization:
   - lowercase;
   - Unicode word tokens length >= 4;
   - remove fixed stopword set;
   - require intersection between bridge tokens and subject tokens, not full brief tokens;
   - no stemming unless implemented.
   If no subject tokens exist, bypass LLM bridge and use deterministic fallback.

5. [DESIGN 2 / DESIGN 5] Payload cleaning and length cap are unspecified, while current code uses `clean_one_line(news_close_brief or "", max_chars=0)` with no cap. Appending an uncapped real brief can produce overlong announcer lines and may include multiple sentences/newlines before cleaning. Concrete fix: define `clean_news_close_brief(news_close_brief) -> str` with an explicit max char/word cap, one-line normalization, no brackets/speaker labels, terminal punctuation normalization, and behavior when truncation would alter meaning. Use it before both happy path and fallback.

6. [DESIGN 5 / _otr_line_composer.py:2778-2867] Validator scope is incomplete. Validating “BRIDGE ONLY” does not ensure the final emitted line is one line, SFW, within resource bounds, or free of formatting after `f"{bridge} {news_close_brief_clean}"`. Concrete fix: add a cheap final-output sanitizer/check after append: one line, no leading speaker label/bracket, max chars, valid UTF-8 string, and non-empty payload. This must not reject factual content for not being “bridge-like.”

7. [DESIGN 3 / _otr_line_composer.py:2778-2867] The coda branch control flow is underspecified around the existing resolved-fiction branch. Current code appends the fictional `ending_change` instruction and mutates `system_content` when `resolved and ending`. Concrete fix: branch early:
   - if `news_coda_enabled and close_clean`: run new coda path and do not add `ending_change`, `final_character_line`, or resolved-fiction prompt text;
   - else run existing current outro path byte-identically.
   This is safer than trying to “suppress” individual lines later.

8. [DESIGN 3 / DESIGN 1] Excluding only `ending_change` is not enough to avoid fictional blending. Current outro user prompt may include `script_brief`, `intro_text`, and `final_character_line`; the final character line is explicitly harvested from the last character line at :4619-4623 and may contain the fictional resolution. Concrete fix: define the exact LLM inputs for news coda. Minimal safe set: `opening/tonight drama hint` optional, `news_subject`, and no `ending_change` / no `final_character_line` / no full `news_close_brief`.

9. [DESIGN 6] Empty `news_close_brief` behavior contradicts current control flow. Current `compose_announcer_outro` only falls back when both `brief` and `close` are empty; if `script_brief` exists and `close` is empty, it will still call the LLM. The spec says empty `news_close_brief` should “skip news-coda mode, use the existing outro fallback, LOUD flag `news_coda_no_brief`.” Concrete fix: in the coda-enabled branch, if `close_clean` is empty, return a deterministic fallback result immediately with flags including `news_coda_no_brief`, or explicitly delegate to the old composer path and document that it may use LLM. Pick one.

10. [DESIGN 6 / compose_flags taxonomy] Required flags are not defined. The spec mentions `news_coda_no_brief` but not flags for happy path, bridge validation failure, reroll success, fallback prefix, cleaning/truncation, or skipped mode. Concrete fix: define exact `compose_flags` tuples, e.g.:
   - `("news_coda_bridge",)`
   - `("news_coda_bridge_reroll",)`
   - `("news_coda_fallback", "news_coda_bridge_invalid")`
   - `("news_coda_no_brief", "announcer_outro_fallback")`
   Keep old flags unchanged when `news_coda_enabled=False`.

11. [DESIGN 4 / DESIGN 6] Fallback prefixes intentionally include phrases equivalent to blacklisted generic openers. That is acceptable only if the blacklist is bridge-only, but the spec does not state this invariant strongly enough. Concrete fix: name the blacklist `BRIDGE_GENERIC_OPENERS` and ensure `validate_news_coda_bridge` is never called on deterministic fallback prefixes.

12. [DESIGN 5] “Does not assert an outcome” is too vague to implement reliably. Concrete fix: make it explicitly best-effort regex/blocklist only, with a named list of forbidden constructions, e.g. `proved`, `revealed`, `confirmed`, `ended with`, `was found`, `was killed`, `was arrested`, `will`, `did`, etc. Do not overpromise semantic detection.

SHOULD-FIX:
1. [DESIGN 1 / _NEWS_CODA_SYSTEM] The new system prompt is described but not specified. Concrete fix: include the exact `_NEWS_CODA_SYSTEM` text in the plan, including output length, no facts, name subject only, no brackets, no speaker label, no quotes, one clause/sentence, and no dramatic outcome.

2. [DESIGN 2] Appending with `f"{bridge} {news_close_brief_clean}"` can produce bad punctuation, e.g. `bridge` ending in period followed by payload sentence, or a bridge fragment without a colon/comma. Concrete fix: define bridge terminal punctuation rules. Example: validator normalizes bridge to end with `:` or an em dash, then append payload.

3. [DESIGN 4] The blacklist examples are not enough to prevent trivial variants. Concrete fix: normalize lowercase, strip punctuation, collapse whitespace, and compare against a list of normalized prefixes for startswith matching.

4. [KEY OPEN QUESTION] “De-dup against the appended brief” can accidentally remove the only subject reference or alter factual wording. Concrete fix: do not mutate `news_close_brief_clean`; only reject/reroll a bridge if it copies a long n-gram from the payload, e.g. any 5-token sequence.

5. [NON-NEGOTIABLE] “100% local” is not enforceable in this function if `creative_fn` may call a remote model. [ASSUMPTION] If the project already treats `creative_fn` as local, no change. Otherwise add a caller-side assertion or document that coda uses the existing configured model path.

6. [R3 WIRING FACTS] Refine/reroll rebuild implications are not addressed for any new metadata used by line-level regeneration. If news subject or coda mode can affect rerolled announcer lines, verify `build_reroll_line_request` preserves those fields. Concrete fix: either mark coda non-rerollable or stamp needed coda fields on meta and rebuild them there.

OPTIONAL / NICE-TO-HAVE:
- Add unit tests for: byte-identical disabled path, empty brief, overlong brief, generic bridge rejection, bridge with no subject token, reroll success, deterministic fallback stability by `cast_seed`, and resolved-fiction `ending_change` not appearing in coda prompt.
- Log the rejected bridge reason in debug metadata, not user-visible text.

CUT THESE (over-engineering):
1. [KEY OPEN QUESTION] Cut semantic “de-dup” beyond simple n-gram copy detection. It is brittle and risks altering the deterministic factual payload; simple bridge rejection is enough.
2. [DESIGN 5] Cut sophisticated “asserts no outcome” NLP. Use a conservative regex/blocklist and fallback on failure. Semantic outcome detection will be unreliable and expensive for this weak-model path.
3. [DESIGN 4] Cut stemming/lemmatization for news content tokens unless already available in the repo. A fixed stopword list plus exact normalized token intersection is sufficient and deterministic.