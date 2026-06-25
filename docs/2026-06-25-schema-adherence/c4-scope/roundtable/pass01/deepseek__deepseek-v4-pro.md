<!-- requested_model: deepseek/deepseek-v4-pro | resolved_model: deepseek/deepseek-v4-pro-20260423 -->

VERDICT: yes-with-fixes. Option A (schema optional) is the correct incremental approach, but the wiring plan as described lacks critical guards and branch restrictions that would cause incorrect snippet injection or runtime crashes.

MUST-FIX BEFORE BUILD:
1. [C4 as pass04 wrote it / Option A] The snippet must be appended ONLY in the `ValidationError → schema_field_repair` branch, not in `json_syntax_repair`, `payload_null_repair`, or `cast_membership_repair`. The factory dispatch must check `isinstance(error, ValidationError) and not _is_payload_null_validation_error(error)` before appending. Injecting the schema into syntax‑repair or null‑payload prompts adds noise and risks confusing the model.
2. [Questions for the panel #4] `_build_schema_snippet` must include a recursion guard (e.g., a `seen` set of `$ref` URIs) to prevent infinite loops on `$ref` cycles in `model_json_schema()`. Without it, a self‑referencing schema will crash the process.
3. [Invariants #2] The radio_editor local `_default_repair_prompt` modification must import `_build_schema_snippet` from `_otr_repair_prompts`. Verify that this does not create a circular import: radio_editor → _otr_repair_prompts → _otr_structured_call, and _otr_structured_call does not import radio_editor. Should be safe, but must be confirmed.
4. [C4 as pass04 wrote it] The factory must enforce the `TypeError` guard: before appending the snippet, check that the returned `messages` is a non‑empty list whose first element has a string `"content"`. This guard must be active only when `schema is not None`.

SHOULD-FIX:
1. [Questions for the panel #3] Migrate the radio_editor repair path to use `make_dispatching_repair_factory(schema=RadioEditPlan, deterministic_repair=…)` instead of augmenting the local string‑returning `_default_repair_prompt`. This unifies the snippet‑appending logic and avoids maintaining two separate mechanisms.
2. [Questions for the panel #1] Document the incremental opt‑in process: the 14 existing callers remain unchanged (no schema argument), and a tracking issue should be created to add `schema=` to each caller as real un‑aliased drifts are observed, not in a big‑bang.
3. [Grounding finding] Explicitly test that when `deterministic_repair` returns a resolved instance and `schema` is provided, the factory does NOT attempt to append the snippet (it returns the instance directly). The current logic already bypasses snippet appending, but a regression test is warranted.

OPTIONAL / NICE-TO-HAVE:
- Add a build‑time verification (e.g., a test) that greps for `make_dispatching_repair_factory(` calls and asserts that any call with a `schema=` argument passes a valid `BaseModel` subclass, catching typos.
- When the snippet is truncated to ~2000 characters, log a warning so operators know the schema was too large to be fully included.

CUT THESE (over-engineering):
- Option C (schema required, update all 15 callers) is over‑engineering and contradicts the incremental philosophy; it should be cut.
- Appending the snippet to all repair branches (json_syntax, payload_null, etc.) is unnecessary noise and should be cut.

[ASSUMPTION] `_build_schema_snippet` will be implemented in `_otr_repair_prompts.py` and importable. The radio_editor module can import it without circular dependency. The factory dispatch will be modified to conditionally append the snippet only when `schema is not None` and the error is a non‑null‑payload `ValidationError`.