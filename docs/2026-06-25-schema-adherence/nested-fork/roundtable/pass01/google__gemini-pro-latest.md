<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: yes-with-fixes. Candidate A is the only viable path for nested schemas, but combining it with pass04's post-failure exception-arm coercion creates a split-brain architecture that must be simplified.

MUST-FIX BEFORE BUILD:
1. [Candidate A / Question 4] Pydantic v2 class decorator compilation hazard. Do not use a class decorator to inject `mode="before"` validators; Pydantic v2 compiles the `__pydantic_core_schema__` at class creation time, and post-creation decorators routinely fail to register validators correctly. 
   *Fix:* Implement a shared helper function `apply_field_aliases(data: Any, aliases: dict) -> Any`. Call it explicitly from a standard `@model_validator(mode="before")` defined directly on `BeatEdit` (and any future tolerant schemas).
2. [Candidate A / Invariant 1] Missing type guard in the shared validator. `mode="before"` validators can receive model instances, JSON strings, or other types during internal Pydantic operations (like `model_copy`). If the shared validator assumes a dict, it will crash.
   *Fix:* The shared helper MUST replicate the existing `if not isinstance(data, dict): return data` guard from `_accept_field_aliases` before attempting key lookups.
3. [Candidate A / Question 1] Split source of truth. Running a before-validator for nested schemas but keeping `_normalize_field_keys` in the `except ValidationError` arm for top-level schemas is schizophrenic. If a top-level schema needs aliasing, it should just use the exact same before-validator helper.
   *Fix:* Drop `_normalize_field_keys` from the `except ValidationError` block entirely. Use the before-validator uniformly for any schema (top-level or nested) that defines `__otr_field_aliases__`.

SHOULD-FIX:
1. [Candidate A / Performance] Dict allocation overhead. The existing `_accept_field_aliases` uses `data = {**data, "beat_index": data["index"]}`. For a `RadioEditPlan` with 100+ edits, this allocates a new dictionary for every alias found.
   *Fix:* In the shared helper, do a single `data = data.copy()` upon finding the first applicable alias, then mutate in-place (`data[canonical] = data.pop(alias)`).

OPTIONAL / NICE-TO-HAVE:
- [Question 3] `action: ("lever",)` is a safe whitelist entry. Because `action` is load-bearing, Pydantic still enforces the type (string), and `post_validate_plan` (Guard1) enforces the exact value against `ALL_ACTIONS`. An invalid aliased value will fail-loud at the content-check phase, satisfying Invariant 2.
- [Question 5] Collision rule: "Explicit canonical always wins." If `action` and `lever` are both present, ignore `lever`. If `lever` and `action_synonym2` are both present but `action` is missing, pick the first one defined in the alias tuple and ignore the rest to maintain determinism.

CUT THESE (over-engineering):
1. [Candidate A] Pass04's C5 `validate_tolerant_data` and C2 `_normalize_field_keys`. 
   *Why it is safe to cut:* By moving tolerance into the schema's `mode="before"` validator, the schema *itself* becomes tolerant. `schema.model_validate(data)` will now handle the aliases natively. You do not need a bespoke `validate_tolerant_data` wrapper or a post-failure coercion loop for aliases. The existing `_clamp_overlong_strings` loop in `_parse_and_validate` (lines 231-245) is sufficient for post-failure mutation.
2. [Candidate C] Nested path-walking in the exception arm. 
   *Why it is safe to cut:* Fragile, requires manual list-index reassembly, and duplicates Pydantic's native recursive validation tree. Candidate A's approach is vastly superior for nested structures.