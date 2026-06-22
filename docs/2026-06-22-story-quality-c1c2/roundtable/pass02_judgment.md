# pass02 judgment (CODING, C1+C2) -- Claude = judge

Panel: my grounded critique + GPT-5.5 + Gemini-3.1-pro + DeepSeek-v4-pro. Spend $0.08.

## ACCEPTED (grounded)
- `key_terms` source RESOLVED: `meta["news"]["key_terms"]` / `briefs.key_terms`
  (the curated salient entities) -- grounded in the writer; the derivations take it.
- CUT the news proper-noun/number regex; anchors come SOLELY from key_terms (clean,
  no garbage). (Gemini; GPT/DeepSeek lean same)
- central_object = a CONCRETE deterministic exclusion rule (reject ALL-CAPS /
  single-cap entity / cast-name / org-place suffix / numeric; accept first
  descriptive phrase; else "") -- the conservative resolution of the
  "regex-can't-judge-centrality" objection (Gemini/DeepSeek). OMIT, never invent.
- Idempotency: derive ONLY when the key is ABSENT (`not in meta`); a present []/""
  is NOT recomputed. (GPT/DeepSeek)
- Anti-duplication on resume/retry: a SENTINEL guard ("Specificity anchors:" /
  "Central object" not in the target string) + the once-before-loop placement, and
  inject on TRUTHINESS (`if meta.get(...)`) so []/"" cleanly skip. (Gemini/GPT)
- SANITIZE every injected value (collapse whitespace, strip newlines/delimiters,
  cap length) -- prompt-injection safety. (GPT)
- Cast-name exclusion flattens names + tokens (handle dict/object rows; partial
  names). (Gemini/GPT)
- meta values are RUNTIME list[str] / str (not JSON-encoded strings). (GPT)
- One home: pure `nodes/_otr_specificity.py` (the extractor IS the diagnostic --
  cut the separate scan-only helper). (DeepSeek)

## CUT
- The LLM opt-in entirely (deterministic only -- no dormant path, no flag). (GPT)
- The news regex (Gemini). - The scan-only proper-noun diagnostic (DeepSeek).
- central-object ranking beyond first-accepted. (GPT)

## OPEN -> pass03 (wiring)
- Exact writer edit points (canon_header @~2745 before the line loop;
  news_close_brief before compose_announcer_outro @~3275; the meta-absent guard).
- Prompt-snapshot tests proving the anchors reach `_build_user_prompt` and the
  central object reaches the outro brief, with no duplication on a second pass.
