# pass03 judgment (WIRING, C1+C2) -- Claude = judge. CONVERGED.

Panel: my grounded critique + GPT-5.5 + Gemini-3.1-pro + DeepSeek-v4-pro. Spend $0.10.

## ACCEPTED (grounded)
- Idempotency via META FLAGS (`meta["_specificity_anchors_injected"]` /
  `_central_object_injected`), NOT a header substring scan -- avoids false
  suppression when the phrase occurs naturally. (DeepSeek)
- central_object positive rule: accept ONLY if it has >=1 entirely-lowercase
  alphabetic token (GPT); reject Title-Case multi-word proper nouns ("James Webb
  Telescope") -- every-alpha-token-capitalized (Gemini); single-cap = `len(split)==1
  and [0].isupper()` (DeepSeek); plus cast-token INTERSECTION (GPT, like anchors).
- _cast_tokens adds sub-tokens only when len>2 (Gemini: "a"/"j"/"will" would drop
  "a new car"). Cast filter casefolded + applied first.
- Defensive never-raise normalization (None/str/non-iterable) for key_terms/cast/
  meta values; coerce existing meta values before injection. (GPT)
- sanitize = collapse whitespace + strip control/newlines only; NO length cap
  (key_terms short; a char cap slices words) + no vague delimiter stripping. (Gemini/GPT)
- Extract PURE inject_* helpers so the wiring is unit-testable without the writer. (GPT)
- Inject BEFORE `meta["canon_header"]=...` (@3255) + the line loop. (GPT, grounded)
- Source = `meta["news"]["key_terms"]` canonical (fallback key_terms_tuple). (DeepSeek)
- compose_announcer_outro has NO input length limit (`clean_one_line(...,max_chars=0)`)
  -> brief append is safe. (verified, resolves DeepSeek/Gemini length concern)
- If a writer-metadata snapshot test asserts exact meta keys, update it same-commit. (GPT)

## CUT
- The reroll gate, outline nudge, LLM opt-in, news proper-noun regex, scan-only
  proper-noun diagnostic, central-object ranking, length cap, broad delimiter strip.

## CONVERGENCE
pass03 produced only build-level robustness items (sentinels, tokenizer edges,
defensive coercion) -- no new architecture. The 3-pass C1+C2 campaign has CONVERGED.
pass03_plan.md is build-ready -> `docs/2026-06-22-story-quality-c1c2/SPRINT_PLAN.md`.
