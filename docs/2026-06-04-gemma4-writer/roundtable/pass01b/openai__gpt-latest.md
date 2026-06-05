<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. The recommended path depends on local OpenAI-compatible behavior that the grounded backend does not currently provide, and B is not specified tightly enough to preserve the existing picker contract.

MUST-FIX BEFORE BUILD:
1. [Context / Invariants 2 / Runtime question] The claim that the existing OpenRouter lane can simply be pointed at Ollama/llama.cpp/LM Studio is not true as a shipping local/offline path in the grounded code. `_otr_openrouter_backend.openrouter_enabled()` requires both `OPENROUTER_API_KEY` and `OTR_ENABLE_OPENROUTER=1`, and `OpenRouterBackend.generate()` raises if `OPENROUTER_API_KEY` is absent. It also sends OpenRouter-specific `provider.require_parameters` whenever `response_format` is used. Concrete fix: add an explicit “local OpenAI-compatible” mode/backend or provider flag that:
   - does not require a real API key for localhost,
   - does not require `OTR_ENABLE_OPENROUTER=1` as a cloud gate,
   - sends no OpenRouter-only `provider` object to non-OpenRouter endpoints,
   - allows an explicit local model slug/name instead of falling back to OpenRouter recommended cloud slugs,
   - has live smoke tests against the selected runtime’s `/v1/chat/completions`.

2. [A / Runtime question] A(a) “GBNF grammar through the existing lane” is not implementable through the grounded seam as written. `OpenRouterBackend.generate()` accepts `response_format` but has no `grammar`, `gbnf`, or runtime-specific constrained-decoding parameter, and `make_openrouter_generate_fn()` exposes only `response_format`. Concrete fix: either cut GBNF from the first build and use only JSON schema through the existing `response_format` seam, or extend the generate function/backend payload contract with a grammar parameter and runtime-specific mappings/tests for the chosen server.

3. [A(b) / Recommended starting position 2] Re-contracting the inventor to JSON schema is not just a prompt/runtime change. `_run_inventor()` currently calls `creative_fn(messages, temperature, max_new_tokens)` and then `_parse_inventor_output(raw)` for newline text; it never passes `response_format` and does not parse JSON. Concrete fix: add a new inventor output model, e.g. object shape `{ "descriptors": [...] }` with min/max 5 and item pattern; route that call through a schema-capable generate path; parse JSON before validation; keep the existing descriptor grammar/distinctness validation after decoding.

4. [A] Do not treat grammar/schema as a full replacement for the current validator. JSON schema/GBNF can force count and basic string shape, but the current pairwise “max one shared root word” rule is semantic and is enforced in `_parse_inventor_output()` / `StylePick._candidates_grammar_and_distinct()`. Pydantic `field_validator` logic will not be represented in `schema_to_response_format()`’s `model_json_schema()`. Concrete fix: keep post-decode validation and retry/fallback policy even when constrained decoding is enabled.

5. [B / Recommended starting position 1] “Take-first-5-distinct” is under-specified and currently contradicts the inventor’s documented exact-count fail-loud contract. `_parse_inventor_output()` validates `len(lines) == 5` before grammar/distinctness and raises otherwise. Concrete fix: define the exact algorithm and tests before changing it. Minimum safe algorithm:
   - normalize lines exactly as today,
   - reject empty output,
   - for each normalized line, require `DESCRIPTOR_RE`,
   - greedily append only if not duplicate and pairwise shares <= 1 root with already accepted candidates,
   - return when 5 accepted,
   - raise if fewer than 5 accepted,
   - log/count ignored extra valid lines for observability.
   Add tests for exactly 5 unchanged, >5 valid selects first 5 distinct, duplicates skipped, similar-root candidates skipped, fewer than 5 still fails, invalid mixed output policy explicitly covered.

6. [B / _otr_style_picker.py docs] If B ships, update the stale fail-loud documentation in `_otr_style_picker.py`. The module header and `StyleGenerationFailedError` comments say chooser mismatch raises and the workflow halts, but grounded `_run_chooser()` now retries and falls back to `candidates[0]`. Concrete fix: update the docstring/error comments and add tests matching the actual policy so future reviewers do not preserve a false invariant.

7. [Goal / Open questions] The plan only fixes the shown style inventor. The document itself says exact-count/strict-shape failures likely recur, but it does not inventory any other writer passes. Concrete fix: before declaring gemma usable as writer, grep/audit all LLM call sites for exact counts, JSON-only parsing, regex-only parsing, “choose one of N”, and bounded list requirements; list each pass and assign B/A/no-change. Mark gemma unsupported for any unaudited exact-count pass.

8. [C / Invariant 1] The C interim default is not build-ready against the 14.5 GB VRAM ceiling. “Remote-style” localhost serving does not mean zero VRAM; Ollama/llama-server/LM Studio can keep weights resident outside ComfyUI, while the grounded OpenRouter backend’s `unload()` is a no-op. Concrete fix: do not ship gemma-local creative + mistral-local technical as a default until measured with the actual runtime on the target card using `nvidia-smi` or equivalent. If C is kept, require one of: one model CPU/offloaded, one true remote/zero-GPU endpoint, explicit unload between slots, or a measured two-model residency budget below 14.5 GB. [ASSUMPTION] Two quantized local writer models are likely to exceed or fragment the stated budget unless one is not GPU-resident.

SHOULD-FIX:
1. [Runtime question] Runtime ranking for this specific use should be conditional on live `/v1/chat/completions` tests, not feature pages:
   - 1: llama.cpp `llama-server`, if the target version accepts the needed JSON schema and/or grammar parameter through `/v1`.
   - 2: LM Studio, if GUI management is acceptable and its OpenAI server passes the same schema tests.
   - 3: Ollama, unless its OpenAI `/v1` endpoint is proven to honor `response_format` JSON schema; native Ollama `format` support is not enough if OTR only sends OpenAI-style `response_format`.
   Concrete fix: add a small runtime conformance script that asks for exactly five descriptors and verifies status code, payload compatibility, exact count, regex pattern, and no extra prose.

2. [Runtime question / Ollama] The plan says Ollama supports `format` JSON-schema structured output, but the grounded backend sends `response_format`, not Ollama native `format`. Concrete fix: verify Ollama’s OpenAI-compatible endpoint maps `response_format={type:json_schema,...}` correctly; otherwise add an Ollama-specific payload adapter or do not list Ollama as schema-capable for A.

3. [A(b)] Avoid a top-level JSON array as the inventor response. Some OpenAI-compatible structured-output implementations only reliably accept object schemas. Concrete fix: use an object wrapper such as `{ "descriptors": ["..."] }`.

4. [A / B] Keep `_INVENTOR_MAX_TOKENS` under review. For newline mode it is currently 80; for JSON object mode this may be tight on local runtimes without the OpenRouter backend’s 1024-token floor. Concrete fix: set a schema-mode-specific budget and test no `finish_reason=length` / truncation equivalent.

5. [D / _run_inventor] The module header says inventor stops on blank line, but grounded `_run_inventor()` does not pass `stop` to `generate_fn`. Concrete fix: either pass an appropriate `stop` sequence for line mode or remove the claim. Do not use a newline stop for JSON mode unless proven not to truncate valid JSON.

6. [B] Add observability when leniency fires. Concrete fix: log raw parseable count, accepted five, rejected duplicates/similar lines, and model id. This preserves the ability to detect wildly non-compliant models instead of silently normalizing them.

7. [C] Do not add a finer per-pass routing knob unless the audit proves the two-slot creative/technical split is insufficient. Existing `pick_style()` already accepts `creative_fn` and `technical_fn`; retagging exact-count passes technical is the smaller change.

OPTIONAL / NICE-TO-HAVE:
- Add a bake-off metric specifically for “completion rate without parser repair” so B does not make non-compliance invisible.
- Store a `style_pick.pass1_overgeneration_count` forensic field if B accepts more than 5 parseable lines.
- Add a developer doc showing exact env/config for each supported local runtime.

CUT THESE (over-engineering):
1. [A] Cut GBNF from the first build. The current grounded seam already has `response_format`; GBNF requires new API surface and runtime-specific payload handling. JSON schema plus existing post-validation is the smaller first step.

2. [Runtime question] Cut LM Studio from the automated/release path. It can remain a user option, but GUI-managed runtime is weaker for reproducible CI/headless conformance than llama-server.

3. [C] Cut “two resident local models” as an interim default. It violates the simplicity of the VRAM invariant and ComfyUI cannot manage VRAM held by an external localhost server through `OpenRouterBackend.unload()`.

4. [C] Cut any new per-pass model-pick widgets. They conflict with the stated two-slot routing invariant; use existing creative/technical routing and code-level pass tags.