<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: no. The proposed C4 wiring introduces a redundant dependency, risks prompt corruption via malformed JSON, and chases a ghost execution path. Defer C4 entirely (Option B).

MUST-FIX BEFORE BUILD:
1. [The fork / Q6] **Redundant Interface Dependency**. Forcing 15 callers to pass `schema=` to `make_dispatching_repair_factory` is an architectural failure. `_otr_structured_call` *already holds the schema* (it requires it to perform the validation that triggers the repair). 
   *Fix*: Do not add `schema=` to the factory. If C4 must be built, implement it as a one-line shim inside `_otr_structured_call`'s `ValidationError` except-block: append the snippet to the messages returned by the factory. This gives all 14 callers the snippet for free with zero call-site edits.
2. [C4 as pass04 wrote it] **Data Format Corruption**. "cap ~2000 chars + truncation note" applied to `json.dumps(sort_keys=True)` will slice a JSON string in half, leaving unclosed strings, arrays, and braces. Injecting malformed JSON into a repair prompt meant to fix a schema violation will degrade model performance.
   *Fix*: Never truncate serialized JSON. Prune the dictionary keys *before* serialization to guarantee a valid JSON string, or drop the truncation requirement entirely.
3. [Context / The fork] **Dead Execution Path**. You state the proven failure (`normalize_length`) is already fixed deterministically on attempt 1 via C0/C1. Therefore, the Attempt 3 repair turn *will never execute* for this failure. Wiring C4 into `radio_editor` first (Option A) to "fix the proven failure" tests dead code.
   *Fix*: Defer C4 (Option B). You have no active failing test case to verify this belt-and-suspenders logic actually works. Wait for a real un-aliased drift.

SHOULD-FIX:
1. [The fork / Q2] **Message Wiring Noise**. Injecting a schema snippet into a `JSONDecodeError` (`json_syntax_repair`) or a `PostValidationError` (`cast_membership_repair`, `narration_leak_repair`) is an interface mismatch. If the model failed basic JSON syntax, dumping a schema distracts it. If it failed a semantic post-validator, the schema shape is already correct.
   *Fix*: If implemented, the schema snippet must ONLY be appended during `schema_field_repair` and `payload_null_repair` (pure Pydantic `ValidationError`s).
2. [The fork / Q4] **Schema Recursion Trap**. Pydantic v2 `model_json_schema()` uses `$defs` for nested models. If you recursively strip keys but fail to correctly traverse and preserve the `$defs` dictionary at the root, `$ref` pointers will resolve to nothing, rendering the schema snippet useless to the LLM.
   *Fix*: Ensure the stripping utility explicitly preserves the root `$defs` object and traverses into it to strip descriptions/titles from the nested definitions.

OPTIONAL / NICE-TO-HAVE:
- If you ever migrate `radio_editor`'s local string-returning factory, update it to return `list[dict[str, str]]` to match the `RepairPromptFactory` protocol defined in `_otr_repair_prompts.py`. [ASSUMPTION: `_otr_structured_call` currently has a type-check or branch to handle the legacy `str` return].

CUT THESE (over-engineering):
1. **Option C (Full C4 per pass04)**. Safe to cut because a 9-module big-bang refactor for an unproven belt-and-suspenders feature violates your incremental philosophy and risks breaking 14 currently working call sites.
2. **Option A (Incremental via factory args)**. Safe to cut because passing `schema=` through the factory is architecturally backward when the caller (`structured_call`) already has it.