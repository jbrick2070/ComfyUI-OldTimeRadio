<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: yes-with-fixes
The core retry ladder is sound, but the repair mechanism is flying blind because it never actually shows the model the target schema, and the "lower temperature" retry is fundamentally broken for systematic format variance.

MUST-FIX BEFORE BUILD:
1. [Candidate levers D / _otr_repair_prompts.py] **The typed repair factory cannot access the schema.** The `RepairPromptFactory` protocol only accepts `original_prompt`, `failed_output`, and `error`. It does not receive the `schema` class. Therefore, `schema_field_repair` cannot inject the literal JSON schema into the prompt, making Candidate D impossible to implement. 
   *Fix:* Update the `RepairPromptFactory` protocol signature in `_otr_structured_call.py` to accept `schema: type[BaseModel]`. Update `schema_field_repair` to dump and append `schema.model_json_schema()` to the `CRITICAL` directive.

2. [Hard constraints 2 vs Candidate levers C] **Byte-identity contradiction.** You cannot implement Candidate C (Schema-in-the-prompt up front in the BASE prompt) without altering the base prompt string. Altering the base prompt changes the token sequence, which will break local byte-identity for inputs that already validate today.
   *Fix:* Reject Candidate C for the *base* prompt. Inject the schema strictly in the Attempt 3 repair turn (Candidate D) where byte-identity is already lost.

3. [Current architecture / _otr_structured_call.py] **Attempt 2 token burn on systematic variance.** The "2B principle" assumes structural hallucination is a sampling glitch. But if a frontier model systematically prefers its own keys (e.g., Opus emitting `lever` instead of the schema's key), Attempt 2 feeds it the *exact same prompt* at a *lower* temperature. This guarantees the model will confidently repeat the exact same wrong keys, burning tokens.
   *Fix:* Short-circuit Attempt 2. If the Attempt 1 error is a `ValidationError` (missing fields) rather than a `JSONDecodeError` (syntax glitch), skip Attempt 2 and go directly to Attempt 3 (typed repair with the schema). 

4. [What broke] **`extra="ignore"` hides the root cause from the repair prompt.** You note Pydantic ignores extras by default. If Opus emits `index` instead of `beat_index`, Pydantic drops `index` and raises "missing `beat_index`". The `schema_field_repair` prompt echoes this error. The LLM never learns that `index` was rejected, only that `beat_index` is missing, leading to confused retries.
   *Fix:* Enforce `model_config = ConfigDict(extra="forbid")` on all structured schemas. This forces Pydantic to include "Extra input 'index' not permitted" in the `ValidationError`, giving the LLM the exact mapping context it needs in the Attempt 3 error string.

SHOULD-FIX:
1. [Candidate levers A] **Pydantic Aliases for known historical variants.** While fuzzy matching is bad, explicit Pydantic aliases (`Field(alias="index")` for `beat_index`) are deterministic, native, and zero-token-cost. 
   *Fix:* Implement Candidate A *only* as a hardcoded whitelist of known aliases in the Pydantic models via `Field(validation_alias=AliasChoices(...))`. Do not build a custom pre-validation normalizer.
2. [_otr_structured_call.py / _clamp_overlong_strings] **Clamping logic mutates the error context.** If `_clamp_overlong_strings` fixes a length issue but another validation error remains, it re-validates and raises the *new* error. The new error loses the original raw input context, which might confuse the repair prompt echo.
   *Fix:* Ensure the `failed_output` passed to the repair factory is always the *original* `last_raw` string, not a serialized version of the partially clamped dict. (The current code does pass `last_raw`, so this is just a strict invariant to maintain).

OPTIONAL / NICE-TO-HAVE:
- [Candidate levers F] Migrate the stragglers. Having multiple hand-rolled JSON parsers/ladders guarantees this bug will happen again in unmigrated nodes. Prioritize this.

CUT THESE (scope / over-engineering):
1. [Candidate levers E] **Lenient extractor pre-pass.** Safe to cut. Building a custom fuzzy-mapping coercion layer before Pydantic is an endless treadmill, duplicates Pydantic's job, and risks silent-wrong data mapping (violating Constraint 4).
2. [Candidate levers B] **Relax required->optional-with-default.** Safe to cut. If a model is omitting fields, defaulting them silently degrades narrative/downstream quality. Fail loud and repair instead.

[ASSUMPTION] The existing base prompts contain *some* human-readable English description of the required JSON keys. If they do not, local models are only passing via sheer luck or over-fitting, and you have a larger prompt-engineering deficit not mentioned in the doc.
[ASSUMPTION] The OpenRouter transport layer (`_invoke_slot`) correctly strips Markdown fences (````json`) when `response_format={"type": "json_object"}` is used, or `_otr_json.parse_first_json_object` is robust enough to handle them. (The doc states it handles fenced output, assuming this works flawlessly).