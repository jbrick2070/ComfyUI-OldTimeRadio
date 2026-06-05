# Pass 01 judgment log

**Panel landed:** Grok-4.3 (complete), Gemini-3.1-pro (partial -- truncated at the
2000-token cap). Opus/GPT/Sonnet/DeepSeek errored on pass 1 (empty content:
hidden reasoning ate the 2k cap) and the 12k-token re-run stalled with no output;
ChatGPT covered via the manual paste package (`CHATGPT_PASTE.md`).

## Accepted (verified against code)
- B (take-first-5) is NOT one line: `StylePick.candidates Field(min=max=5)` +
  distinctness validator + `StyleGenerationFailedError` docstring all assume 5.
  (Grok #1 -- CONFIRMED at `_otr_style_picker.py:126`.)
- C (re-tag inventor) is staged, not available: `pass1_slot` default "creative"
  (line 138); paired contract routes BOTH passes through `creative_fn` today.
  Needs the staged technical dispatch finished, no new widget. (Grok #2 -- CONFIRMED.)
- A-JSON changes the inventor's line-based contract (template+parser+pydantic must
  migrate). (Grok #4 -- CONFIRMED.) -> prefer A-GBNF to avoid the migration.
- VRAM: two resident local models break 14.5 GB; one slot must be served.
  (Gemini partial + Grok should-fix #2 -- CONFIRMED by the ceiling invariant.)
- CUT the full runtime bake-off and the narrative-re-bakeoff from build scope.
  (Grok cuts #1, #2 -- ACCEPTED.)

## Rejected / downgraded (judge grounding)
- Grok #3 "llama.cpp/non-Ollama has no integration point" -- PARTIAL MISREAD.
  The backend's `OPENROUTER_BASE_URL` already points the lane at ANY OpenAI `/v1`
  (that is how it reached Ollama), so llama-server / LM Studio drop in via the
  SAME lane with no new backend. Downgraded to a verify-at-build: confirm the
  backend's OpenRouter headers don't trip a local server (extra headers normally
  ignored). NOT a blocker.

## Still open (pending ChatGPT manual + reasoning panel)
- Exact blast radius of other exact-count passes (grep). 
- Product call on relaxing the strict 5-count gate (B).
- Current GBNF/json_schema support per runtime (verify; fast-moving).
