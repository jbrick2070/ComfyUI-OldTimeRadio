<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: yes-with-fixes. Implement Option A only if C4 is wired now; Option C is not build-ready because it breaks the documented 15 call sites, and Option B is acceptable if zero-risk shipping is preferred.

MUST-FIX BEFORE BUILD:
1. [C4 as pass04 wrote it / grounding finding] `schema` must not be required on `make_dispatching_repair_factory`. The current grounded signature takes only `deterministic_repair`; the caller inventory says 15 call sites across 9 modules currently omit `schema`, including one that passes only `deterministic_repair`. Making `schema=` required creates an immediate TypeError/build break at every existing call site. Concrete fix: change the proposed signature to `make_dispatching_repair_factory(*, schema: type[BaseModel] | None = None, deterministic_repair: Optional[DeterministicRepair] = None)`, and make `schema=None` produce byte-identical behavior.

2. [Invariants 1 + Option A] The schema snippet must be gated strictly on `schema is not None`. Existing 14 no-arg callers plus the one deterministic caller must not receive altered repair prompts. Concrete fix: in the `ValidationError` branch, keep the current `payload_null_repair` and `schema_field_repair` dispatch unchanged unless `schema` is supplied; only wrap/append the `schema_field_repair(...)` result for opt-in schemas.

3. [Option A / proven-failure path] Do not reuse the pass04 `messages[0]["content"]` append logic for `radio_editor`’s local `_default_repair_prompt`, because the provided context says that factory returns a string, not a messages list. That is an interface mismatch. Concrete fix: expose/use a helper that returns only the schema constraint text, e.g. `_build_schema_snippet(RadioEditPlan) -> str`, and append it to the local string prompt as `"\n\nSchema constraints:\n" + snippet`. Do not force the local path through the dispatching factory unless the Guard 1/2/3 directive is ported exactly and verified.

4. [make_dispatching_repair_factory / deterministic_repair] The proposed type-check `isinstance(resolved, schema)` only works when `schema` is provided. With `schema=None`, there is no exact model class to check against. Concrete fix: guard the exact check: `if resolved is not None and schema is not None and not isinstance(resolved, schema): raise TypeError(...)`. Do not add a new unconditional check that changes the existing deterministic caller’s behavior unless separately tested.

5. [_build_schema_snippet proposal] Do not implement the cap as naive `json.dumps(sort_keys=True)[:2000]` without tests. With `sort_keys=True`, `$defs` can sort before root `properties`/`required`, so a prefix truncation can spend the budget on definitions and omit the top-level contract the model needs. Concrete fix: either ensure the root schema keys are included before truncating `$defs`, or add an offline test proving the selected first opt-in schema’s snippet still contains top-level `type`, `properties`, `required`, and relevant nested constraints after truncation.

6. [_build_schema_snippet proposal] The recursive stripper must strip only the named bloat keys: `description`, `title`, `examples`, `default`. Dropping other JSON Schema keys is unsafe. Concrete fix: preserve at least `$defs`, `$ref`, `properties`, `required`, `type`, `items`, `anyOf`, `oneOf`, `allOf`, `enum`, `const`, `minimum`, `maximum`, `exclusiveMinimum`, `exclusiveMaximum`, `minLength`, `maxLength`, `pattern`, `minItems`, `maxItems`, and `additionalProperties`. Add a fixture test against a pydantic v2 `model_json_schema()` output.

7. [Invariants 2] Before wiring the radio/editor local prompt to import `_build_schema_snippet`, verify the import graph. Grounding only shows `_otr_repair_prompts.py` imports `_otr_structured_call`; it does not show `radio_editor` imports. Concrete fix: verify no circular import from `radio_editor`/writer modules back into `_otr_repair_prompts`. If there is a cycle, move the pure schema-snippet helper to a lower-level pure module or keep a local helper beside the radio editor schema. [ASSUMPTION]

SHOULD-FIX:
1. [Question 2 / dispatcher branches] Keep schema snippets out of `json_syntax_repair`. If the model did not produce parseable JSON, the immediate contract is “return one valid JSON object”; schema detail is secondary noise. Add schema only after JSON parses and pydantic rejects fields.

2. [Question 2 / payload_null_repair] Do not append the generic schema snippet to `payload_null_repair` by default. The grounded comment says the dedicated payload-null directive exists because generic field repair was too vague. Adding schema text risks diluting the precise instruction: supply a non-null replacement string or omit the row.

3. [Question 2 / PostValidationError branches] Do not append the schema snippet to cast-membership, forbidden-name, narration-leak, or too-long branches initially. Those failures are runtime/content constraints, not necessarily represented in pydantic JSON Schema. For cast membership especially, the locked cast is runtime state and the grounded factory says the rejection error already quotes the cast list.

4. [__all__ / helper visibility] If `radio_editor` or other modules import `_build_schema_snippet`, decide whether it is public. Concrete fix: either add it to `__all__` as `build_schema_snippet` / `_build_schema_snippet` intentionally, or keep the import private but add a test so accidental removal is caught.

5. [Invariants 1 / tests] Add byte-identical regression tests for `make_dispatching_repair_factory()` with `schema=None` for at least: `JSONDecodeError`, generic `ValidationError`, payload-null `ValidationError`, recognized `PostValidationError`, and fallback. The expected message content should match current output exactly.

6. [Question 5] If C4 is wired now, the highest-value first opt-in is `RadioEditPlan`, because the context names the radio editor editor-plan path as the proven-relevant path and also says it does not use the shared factory. Do not opt in the other 15 factory call sites without captured failures.

7. [Question 6] Avoid trying to infer schema automatically inside `make_dispatching_repair_factory`. The grounded `RepairPromptFactory` inputs are only `original_prompt`, `failed_output`, and `error`; the schema class is not part of that protocol. Any “free” inference would require a registry, error-title guessing, or changing `_otr_structured_call`’s protocol. That is higher-risk than explicit opt-in.

OPTIONAL / NICE-TO-HAVE:
- Add a small pure helper such as `_append_schema_constraints_to_messages(messages, schema)` for the shared factory and a separate `_schema_constraints_block(schema) -> str` for string-returning local prompts.
- Add a truncation note that is not JSON, e.g. `"... [schema truncated]"`, so the model is not misled into thinking the snippet is a complete schema when capped.
- Add a unit test that `_build_schema_snippet` handles lists, dicts, scalars, and repeated objects without recursion errors.

CUT THESE:
1. [Option C] Cut the required-`schema` big-bang across all 15 call sites. It is safe to cut because C0/C1 already fixed the proven failure, and Option A allows opt-in without changing existing callers.

2. [Question 3] Cut migrating the radio editor local `_default_repair_prompt` to `make_dispatching_repair_factory` for this pass. It is safe to cut because the local string prompt already carries Guard 1/2/3 wording; appending a schema block to that string is the smaller interface change.

3. [Question 2] Cut schema snippets on JSON syntax, payload-null, and PostValidationError branches for now. It is safe to cut because those branches already have targeted repair directives grounded in `_otr_repair_prompts.py`, and schema text does not address their primary failure mode.

4. [Question 6] Cut any global schema registry or automatic schema discovery shim. It is safe to cut because explicit `schema=` opt-in at the call site is clearer, avoids hidden import dependencies, and preserves the current `RepairPromptFactory` contract.