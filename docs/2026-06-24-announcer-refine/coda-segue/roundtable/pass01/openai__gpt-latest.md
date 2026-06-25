<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. The hybrid direction is plausible, but the current plan still asks the weak model to perform the exact unsafe act: write the whole real-news coda, while the existing outro prompt and resolved-ending branch still push it toward fictional-outcome restatement.

MUST-FIX BEFORE BUILD:
1. [PROPOSED RESOLVING DESIGN 1 / GROUNDED compose_announcer_outro:2778-2867] The plan says the LLM writes the WHOLE coda, but the current `compose_announcer_outro` path may append the resolved-ending instruction: “State this outcome plainly,” where `ending_change` is explicitly the FICTIONAL outcome. That directly contradicts the non-negotiable that the coda must never restate fiction as real. Concrete fix: add a distinct news-coda mode under the existing scaffold gate that suppresses the resolved-ending outcome instruction entirely and treats `ending_change` only as forbidden/contrast material for validation.

2. [GROUNDED _ANNOUNCER_OUTRO_SYSTEM:2519-2555 / PROPOSED RESOLVING DESIGN 1-2] The proposed teaching coda conflicts with the current outro system prompt, which requires a concrete final image and forbids “news-summary” / lesson framing. A “real story” coda is inherently a journalistic/news-summary move. Concrete fix: create a separate `_NEWS_CODA_OUTRO_SYSTEM` or gated prompt branch for the coda, instead of trying to reuse `_ANNOUNCER_OUTRO_SYSTEM` unchanged.

3. [PROPOSED RESOLVING DESIGN 3] The proposed validator is too weak to support the stated reliability claim. “>=1 real news content token” can pass a line that mentions one news noun while still blending or inventing the rest; “low overlap with `ending_change`” does not prove the real fact was delivered. Concrete fix: make the real payload deterministic: have the LLM write only a short bridge/segue clause, then append the cleaned `news_close_brief` verbatim or near-verbatim. Validate the bridge separately, and validate that the final coda contains the real close payload.

4. [PROPOSED RESOLVING DESIGN 2] “Teachability via consistent POSITION + SHAPE” is an unsupported premise for weak local models. The model is not learning across episodes; it only sees the current prompt. Position in the final beat may teach the listener, but it does not reliably teach mistral/gemma not to blend fiction into news. Concrete fix: encode the structure inside the prompt/output contract for this single generation, e.g. “write a bridge away from the drama, then the following real closing brief is appended exactly,” rather than relying on cross-episode pattern.

5. [PROPOSED RESOLVING DESIGN 3 / ASK 3] The “blend” gate is underspecified and not conceptually sufficient. Semantic blend can happen with synonyms or causal framing while having low token overlap; valid news-specific bridges can also share nouns with `ending_change` if the fiction was built from the same news seed. Concrete fix: narrow what is being validated: forbid the bridge from asserting any outcome; require it to be a pivot only; keep the factual assertion in deterministic `news_close_brief`.

6. [NON-NEGOTIABLE / R3 WIRING FACTS] The plan says “behind `story_scaffold`, byte-identical off,” but does not specify the actual gate. Grounding says new announcer/contract gates must hang on `_style_grammar_on`, not a new env. Concrete fix: explicitly state that the news-coda branch is only entered when `_style_grammar_on` is true; when false, `compose_announcer_outro` must follow the current path byte-identically.

7. [PROPOSED RESOLVING DESIGN 4] The fallback floor is described as “fixed lead-in + `news_close_brief`,” but no empty/malformed `news_close_brief` behavior is defined. The non-negotiable says the coda delivers the real fact; if `news_close_brief` is empty, that is impossible. Concrete fix: define empty-close behavior explicitly: either do not enter news-coda mode and use the existing outro fallback, or emit a deterministic non-news closing line flagged as missing-news-coda. Do not generate a fake “real story.”

8. [ASK 4 / PROPOSED RESOLVING DESIGN 1] Anti-generic is an ask, not a design. “No fixed prefix” alone does not stop the LLM from producing “And now, the real story...” every episode. Concrete fix: add a deterministic generic-phrase blacklist and/or require bridge specificity before the appended payload, such as at least two content tokens from `news_close_brief` in the bridge if the bridge is dynamic. If payload is appended deterministically, bridge specificity can be optional but generic phrases should still be rejected.

SHOULD-FIX:
1. [PROPOSED RESOLVING DESIGN 3-4] “One reroll on failure” has no observability plan. Without compose flags/reasons, you will not know whether dynamic coda is actually working or whether fallback carries most episodes. Concrete fix: add distinct `LineResult.compose_flags` for dynamic success, reroll success, fallback, and validation failure reason categories.

2. [PROPOSED RESOLVING DESIGN 3] “NO strong content-token overlap with `ending_change`” needs a precise threshold before build. Otherwise different implementers will encode different gates. Concrete fix: define the threshold and whether overlap is counted against the bridge only or the final full coda. If `news_close_brief` is appended deterministically, the overlap check should apply to the dynamic bridge only.

3. [GROUNDED token machinery / PROPOSED RESOLVING DESIGN 3] The plan says `_content_tokens` / `_TOKEN_RE` / `_strip_possessive` are reusable writer-side, but the coda composition lives in `_otr_line_composer.py`. [ASSUMPTION] Importing from `_otr_story_quality_l12.py` may create undesirable coupling. Concrete fix: specify whether the token helper is moved to a shared utility module, duplicated locally, or called from writer-side validation after composition.

4. [PROPOSED RESOLVING DESIGN 1] “Low temperature” is not a reliability mechanism and is not quantified. Concrete fix: state the actual temperature/top-p policy or remove this as a claimed guardrail. The deterministic validator/fallback should be the reliability mechanism.

5. [PROPOSED RESOLVING DESIGN 3] “word band” is undefined and may conflict with appending `news_close_brief`, whose length is not bounded in this document. Concrete fix: define final coda length rules and deterministic truncation/sanitization behavior for the brief.

6. [ASK 2] The plan does not choose whether a minimal recurring anchor exists. Concrete fix: decide now. If using deterministic appended payload, a very small structural anchor can be safe, e.g. a variable bridge followed by a fixed separator before the real brief. If avoiding all fixed phrase, then rely on blacklist + payload append, not “teachability.”

OPTIONAL / NICE-TO-HAVE:
- Add a small offline fixture set of `news_close_brief` / `ending_change` pairs that specifically test blend, shared nouns, synonym blend, empty close, and generic bridge cases.
- Add a metric for dynamic-coda acceptance rate by local model repo so the operator can see whether the happy path is real or mostly fallback.

CUT THESE (scope / over-engineering):
1. [PROPOSED RESOLVING DESIGN 1] Cut “LLM writes the WHOLE coda.” It is the unsafe part. Let the LLM write only the bridge; append the real news deterministically. This preserves dynamic/news-specific flavor while removing the highest-risk factual step.

2. [GROUNDED KILL-2 / KILL-4 / R2 WIRING FACTS] Do not include style grammar, StoryContract relocation, opening status quo, or broad beat-enrichment work in this coda build. They are unrelated to whether the final announcer line safely pivots to real news.

3. [GROUNDED _ANNOUNCER_OUTRO_SYSTEM] Cut “lightly echo the opening line’s tone” from the coda-specific prompt. It does not serve the real-news payload and increases pressure to write a literary closing instead of a clear fact pivot.

4. [PROPOSED RESOLVING DESIGN 2] Cut the claim that position alone is a teachability mechanism for the model. It is safe to remove because the build should rely on prompt contract, deterministic payload insertion, validation, and fallback.