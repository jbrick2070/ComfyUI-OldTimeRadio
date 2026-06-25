<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: yes-with-fixes. The core logic is sound, but a stale state bug in the failure arm will crash the pipeline, and a dependency inversion violation will cause a circular import.

MUST-FIX BEFORE BUILD:

1. **[C2] Stale ValidationError prevents clamping of normalized keys (Sequencing)**
   If `_normalize_field_keys` maps an alias to a canonical key, Pydantic never validated the alias value. Therefore, the original `ve.errors()` contains a `missing` error for that field, NOT a `string_too_long` error. `_clamp_overlong_strings` will silently skip the newly moved key, and the final `model_validate` will crash with a length error.
   *Fix:* If `nk` mutates the dict, you MUST run a `try: schema.model_validate(work)` to catch a fresh `ValidationError` and pass *that* new error to `_clamp_overlong_strings`.

2. **[C4] Circular import / Dependency Inversion violation (Wiring)**
   The plan states: "`structured_call` passes `schema=schema` into the dispatcher builder". `_otr_repair_prompts.py` already imports from `_otr_structured_call.py`. If `structured_call` imports the builder, it creates a circular import. Furthermore, `structured_call` is designed to accept a pre-built factory, not to build it.
   *Fix:* `structured_call` must remain ignorant of the builder. The *caller* (e.g., the hand-rolled site migrating to the ladder) must call `make_dispatching_repair_factory(..., schema=schema)` and pass the resulting callable into `structured_call(repair_prompt_factory=...)`.

3. **[C1] IndexError on empty `loc` (Interface Contracts)**
   The plan specifies "target field = `loc[0]`". Pydantic root validators or whole-model errors emit an empty `loc` tuple `()`. Accessing index 0 will raise an `IndexError`, crashing the failure arm.
   *Fix:* Add `if not loc: continue` before attempting to read `loc[0]`.

SHOULD-FIX:

1. **[C3] Misleading attempt logs**
   If Attempt 2 is skipped (because the error is a `ValidationError`), `attempts_run` correctly remains 1, and then increments to 2 in Attempt 3. However, the Attempt 3 log hardcodes the rung number: `"attempt 3/%d"`. It will log "attempt 3/3" when it is actually the 2nd attempt run.
   *Fix:* Change the log string in Attempt 3 to dynamically use the counter: `"attempt %d/%d: typed repair", attempts_run, max_attempts`.

OPTIONAL / NICE-TO-HAVE:

- **[C4] Appending the schema string without signature churn:** To strictly honor the "Zero churn to the other factories" invariant, `make_dispatching_repair_factory` does not need to pass `schema` into `schema_field_repair`. It can simply call `messages = schema_field_repair(...)` and then append the schema string to the returned payload: `messages[0]["content"] += "\n\nSchema:\n" + schema_str`.

CUT THESE (over-engineering):

- **[C4] Rejected keys reporting:** Safe to cut for v1 as explicitly noted. The strict-first + whitelist-alias approach covers 99% of the structural drift without needing to teach the LLM about dropped extra keys.