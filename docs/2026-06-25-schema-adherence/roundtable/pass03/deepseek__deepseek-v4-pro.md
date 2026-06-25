<!-- requested_model: deepseek/deepseek-v4-pro | resolved_model: deepseek/deepseek-v4-pro-20260423 -->

VERDICT: build-ready as-is? no. C2 clamp step uses original ValidationError and will miss string_too_long errors introduced by key normalization, causing overlong strings to fail validation instead of being clamped.

MUST-FIX BEFORE BUILD:
1. [C2] Defect: After key normalization, `_clamp_overlong_strings` is called with the original `ve`, which only contains errors from the first strict validate. If a field was missing (synonym case) and its value is overlong, the original `ve` has no `string_too_long` error for that field. The clamp step will not fire, and the subsequent `model_validate(work)` will raise a `ValidationError` for `string_too_long`, bypassing the tolerance. Fix: After key normalization, perform a preliminary `model_validate(work)` to obtain a fresh `ValidationError`; pass that to `_clamp_overlong_strings`. Alternatively, refactor `_clamp_overlong_strings` to inspect the schema’s `max_length` directly instead of relying on `ve.errors()`.
2. [C3] Defect: The plan states “Guard Attempt 2 … run it ONLY when `isinstance(last_error, json.JSONDecodeError)`” but does not specify that `attempts_run` must only be incremented for actually executed attempts. The current ladder increments `attempts_run` at the start of each attempt block; if the guard is added as a conditional inside that block without moving the increment, a skipped attempt 2 would still count against `max_attempts`, potentially preventing the typed-repair attempt from running when it should (e.g., `max_attempts=2`). Fix: Restructure the ladder so `attempts_run += 1` occurs only inside the branch that actually executes the attempt.

SHOULD-FIX:
1. [C4] The filtered schema string appended to the repair prompt (`json.dumps(…)`) has no length bound. A large schema could overflow the model’s context window, causing the repair to fail. Add a character limit (e.g., 2000) and truncate with a note.
2. [C3] When attempt 2 is skipped, the log label for the typed-repair attempt should reflect the actual attempt count (e.g., “attempt 2/2” instead of “attempt 3/3”). Update logging to compute the label dynamically based on executed attempts.
3. [C2] The `touched` list accumulates both `(synonym, canonical)` tuples and plain field-name strings, leading to a messy log line. Normalize all entries to strings for clarity.

OPTIONAL / NICE-TO-HAVE:
- C4: Pass `rejected_keys` (extra keys ignored by `extra="ignore"`) into the repair prompt to help the model see dropped fields (deferred per plan).
- C6: Telemetry via an optional `on_event` callback as described.

CUT THESE: none beyond the plan’s own cut list.

[ASSUMPTION] The `__otr_field_aliases__` class attribute will be added to the relevant pydantic models before C1 is exercised.
[ASSUMPTION] Hand-rolled sites identified in C0 can be migrated to `parse_validate_tolerant` without altering their existing error-handling contracts.
[ASSUMPTION] `make_dispatching_repair_factory` will be extended to accept `schema` and the closure will correctly inject the filtered schema into the `schema_field_repair` prompt only.