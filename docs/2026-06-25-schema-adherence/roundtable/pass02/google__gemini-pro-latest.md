<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: yes-with-fixes. The strict-first invariant is sound and preserves byte-identity, but the plan introduces a protocol signature mismatch and requires explicit control-flow changes to the retry ladder that are missing from the spec.

MUST-FIX BEFORE BUILD:
1. **[Step 4] Protocol Signature Mismatch crashes the repair turn.** The plan states "Extend the protocol... so the repair turn appends `schema.model_json_schema()`". If `structured_call` passes `schema=schema` to the factory, it will raise a `TypeError` on every existing factory. You MUST update the `RepairPromptFactory` protocol in `_otr_structured_call.py` to `__call__(..., schema: type[BaseModel])` AND add `schema: type[BaseModel]` to the signatures of all 7 factories in `_otr_repair_prompts.py` (including the inner function of `make_dispatching_repair_factory`).
2. **[Step 3] Attempt 2 unconditional execution.** The plan says "Route a ValidationError... STRAIGHT to typed repair... keep the structural rung only for JSONDecodeError". However, `structured_call` is strictly linear. If Attempt 1 fails, Attempt 2 will execute because it is only guarded by `if attempts_run < max_attempts:`. You MUST explicitly wrap Attempt 2's execution block in `if isinstance(last_error, json.JSONDecodeError):` to actually bypass it for schema/content errors.
3. **[Step 2b] Heuristic collision in `_parse_and_validate`.** The plan adds `_normalize_field_keys` to the `except ValidationError` arm. Currently, that arm calls `_clamp_overlong_strings` and immediately raises if nothing was clamped (`if repaired is None: raise`). If an output has BOTH an overlong string and a missing/aliased key, applying one heuristic and re-validating will fail on the other. You MUST apply both `_clamp_overlong_strings` and `_normalize_field_keys` to the `data` dictionary sequentially BEFORE calling `schema.model_validate(repaired_data)`.

SHOULD-FIX:
1. **[Step 4] Computing rejected keys without `extra="forbid"`.** The plan says to compute rejected keys "NOT by flipping schemas to extra='forbid'". Since Pydantic's default `extra="ignore"` silently drops extra keys, a `ValidationError` will not contain them. You must compute this by diffing the raw keys against the schema fields: `set(data.keys()) - set(schema.model_fields.keys())` inside `_parse_and_validate`, and pass that list in the `error` or directly to the factory.
2. **[Step 2a] Pydantic v2 AliasChoices compatibility.** Ensure `populate_by_name=True` is placed in the `model_config = ConfigDict(populate_by_name=True)` of the BaseModel, not as a Field argument.

OPTIONAL / NICE-TO-HAVE:
- **[Step 6] Telemetry hooks.** `structured_call` currently uses standard `logging`. To get "per-helper counters", you will need to inject a telemetry/metrics client or emit structured log payloads that a log-aggregator can parse.

CUT THESE (over-engineering):
1. **[Step 5] "Audit every structured pass... EITHER migrating... OR factoring".** Cut the "OR factoring" path. Do not build a parallel `parse_validate_tolerant` helper for legacy hand-rolled passes. Force migration to `structured_call` for any pass that needs tolerance. Maintaining two separate implementations of the retry/repair ladder guarantees divergent behavior and bugs.

[ASSUMPTION] I am assuming the `_normalize_field_keys` whitelist will be passed down from the caller into `structured_call` or attached to the schema via a custom Pydantic JSON schema extra, as `structured_call` currently has no way of knowing which aliases are valid for a generic `type[T]`.