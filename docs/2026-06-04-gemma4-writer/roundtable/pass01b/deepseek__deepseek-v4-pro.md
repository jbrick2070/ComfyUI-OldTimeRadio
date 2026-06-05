<!-- requested_model: deepseek/deepseek-v4-pro | resolved_model: deepseek/deepseek-v4-pro-20260423 -->

VERDICT: **yes-with-fixes** – the root cause is correctly identified and candidate B is a practical, low-risk fix; but the plan as written lacks concrete guardrails for B, makes unverified claims about GBNF grammar availability through the OpenRouter lane, skips VRAM confirmation for C, and ignores other exact‑count passes that gemma will likely break.

MUST‑FIX BEFORE BUILD (severity order):

1. **[Approach B – leniency in _parse_inventor_output]**  
   The proposal to “take the first 5 that satisfy the distinctness rule” does not specify what happens when fewer than 5 such lines can be selected after filtering for root‑word distinctness.  
   **Fix:** Add an explicit fallback rule: if after consuming all valid lines the distinct‑selected list is shorter than 5, raise `StyleGenerationFailedError` (fail‑loud). The implementation must be documented and tested.

2. **[Runtime question / GBNF grammar dependence]**  
   The plan relies on GBNF grammar exposure through the OpenAI‑compatible `/v1/chat/completions` endpoint (llama‑server) and on the existing OpenRouter lane to propagate that grammar. The grounding code shows no support for a `grammar` parameter; only `response_format` is handled.  
   **Fix:** Before building A, verify that the chosen runtime (e.g., llama‑server build X) actually accepts a `grammar` key in the chat‑completion payload when called via its OpenAI endpoint, **and** extend `OpenRouterBackend.generate` (or provide an alternative local endpoint) to pass a grammar object. If this cannot be confirmed for the required version/windows/offline setup, remove A from the starting position.

3. **[(C) VRAM math for dual‑model routing]**  
   The plan proposes gemma‑4‑12b (creative) + mistral‑nemo (technical) with one “served remote‑style” and one local. No numbers are given; two 7‑8 GB models will exceed the 14.5 GB ceiling if both are GPU‑resident.  
   **Fix:** Provide concrete VRAM measurements for each model on the target Blackwell/Windows setup under the intended serving method (Ollama GPU vs CPU, etc.). If the total exceeds 14.5 GB, discard C or redesign so that only one model uses GPU at a time (e.g., load/unload per pass, though that adds latency and dev work).

4. **[Gap – other exact‑count passes]**  
   The document acknowledges that gemma would “likely recur on every OTR pass that demands an exact count or a strict shape” but does not enumerate those passes or plan fixes. Only the style‑picker inventor is addressed.  
   **Fix:** Audit the writer’s call chain (e.g., news curators, deep curators, outline passes, etc.) for the same `exactly N items` contracts. For each such pass, either apply leniency (like B), re‑tag it to technical and enforce JSON schema, or document that it must run with a compliant model. Without this audit, gemma will fail later in the pipeline.

5. **[Re‑tagging inventor for constrained‑decoding (A(b))]**  
   If the JSON‑schema route is chosen, the inventor pass must be re‑tagged from creative to technical so that the existing `response_format` seam can enforce `minItems=maxItems=5`. The current code (`pick_style`) always calls `_run_inventor(creative_fn, ...)`.  
   **Fix:** Decide whether to implement A(b). If yes, change `pick_style` to route inventor through `technical_fn` (or introduce a per‑pass flag) and update the prompt template to ask for a JSON array. This must be done before coding A.

SHOULD‑FIX:

- **[Approach A – GBNF grammar vs JSON schema]** The plan favours GBNF over JSON‑schema despite the latter already having a seam (`schema_to_response_format`). GBNF introduces a new code path and runtime dependency; re‑evaluate whether JSON schema (A(b)) suffices for exact‑count passes before committing to GBNF.
- **[Monitoring after B]** The leniency could hide a model that is completely ignoring the count instruction. Add a log warning when the raw output contains more than 5 lines, so operators can detect drift.
- **[Integration of llama‑server with OTR lane]** If A is kept, document the required changes to `OpenRouterBackend` (or a new local backend) to pass grammar, including how to set the grammar string and how to validate the output.

OPTIONAL / NICE‑TO‑HAVE:

- Benchmark gemma’s narrative quality against mistral‑nemo once it can complete a full episode.
- Investigate whether Ollama’s “json” structured output (which supports `minItems`/`maxItems` via system prompt?) can achieve the exact‑count constraint through the same `/v1` lane; this would avoid llama‑server entirely. [ASSUMPTION: Ollama’s structured output implementation and OTR lane compatibility are not proven in the grounding; needs verification.]

CUT THESE (over‑engineering):

1. **Full GBNF grammar via llama‑cpp** – Approach A(a) is heavier than needed. B (post‑parse leniency) solves the immediate problem for the style picker. For other exact‑count passes, re‑tagging them to technical and enforcing JSON schema with the existing `response_format` seam is simpler and leverages already‑working code. The plan does not demonstrate a case where GBNF is strictly required.
2. **Runtime ranking exercise** – The plan asks to rank Ollama vs llama.cpp vs LM Studio for GBNF support, but the decision can be deferred. Only one runtime needs to be picked if A is retained; B requires no runtime change. The ranking adds analysis paralysis without direct value.

Mark [ASSUMPTION] where we inferred beyond the document or grounding:
- The claim that llama‑server exposes GBNF grammar through its OpenAI‑compatible endpoint is an assumption not verified in the grounding.
- The claim that other OTR passes demand exact counts is inferred from the plan text; the grounding only shows the style picker. The audit step is therefore an assumption‑driven recommendation.