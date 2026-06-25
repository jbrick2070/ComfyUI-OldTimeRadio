<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. Core API/control-flow/data-shape changes are underspecified and at least one proposed path contradicts the strict-first invariant.

MUST-FIX BEFORE BUILD:
1. [4] `RepairPromptFactory` schema extension is not implementable as written. Current protocol and all grounded factories accept only `original_prompt`, `failed_output`, `error`; passing `schema=` will raise `TypeError` in `default_repair_prompt_factory`, every typed factory, and the dispatcher closure. Concrete fix: choose one interface and update all call sites/factories. Smallest safe change:
   - define a new protocol signature:
     `__call__(*, original_prompt: Any, failed_output: str, error: BaseException, schema: type[BaseModel], rejected_keys: Sequence[str] = ()) -> Any`
   - update `default_repair_prompt_factory`, all factories in `_otr_repair_prompts.py`, and `make_dispatching_repair_factory.factory`
   - update `structured_call` to pass `schema=schema`
   - update/verify deterministic repair callback typing if it needs schema access.
   Alternative: keep the existing protocol and make a wrapper builder that closes over `schema`; do not mix both.

2. [3] The “skip structural retry for ValidationError/PostValidationError” control flow is undefined and conflicts with the current linear 3-rung ladder. Current code always does Attempt 2 before typed repair for any caught `ValueError`, including `PostValidationError`. Concrete fix: replace the linear ladder with explicit branching and define attempt numbering:
   - Attempt 1 fails with `json.JSONDecodeError` -> run structural retry if attempts remain.
   - Attempt 1 fails with `ValidationError` or `PostValidationError` -> run typed repair immediately as the next actual call.
   - If structural retry then fails with `ValidationError`/`PostValidationError` -> typed repair if attempts remain.
   - If structural retry fails with `json.JSONDecodeError` -> decide whether typed repair sees the JSON error or terminal-fails; test it.
   Also update docs/logs so “Attempt 3” is not hard-coded when typed repair may be the second actual call.

3. [3] Current exception handling catches broad `ValueError`, but the plan wants routing by `JSONDecodeError` vs `ValidationError` vs `PostValidationError`. Because `PostValidationError` subclasses `ValueError`, and arbitrary programming errors can also be `ValueError`, the implementation will misroute unless narrowed. Concrete fix: catch only `(json.JSONDecodeError, ValidationError, PostValidationError)` in ladder branches; let unrelated `ValueError` propagate as programming errors.

4. [2(a), CORE INVARIANT] Native pydantic aliases contradict “new tolerance fires ONLY inside the except ValidationError arm.” `Field(validation_alias=AliasChoices(...))` fires during the first `schema.model_validate(data)`, not in the except arm. It can also change behavior for an otherwise valid object containing both canonical and alias keys. Concrete fix: either cut native aliases and rely only on `_normalize_field_keys` after failure, or add a collision rule and tests:
   - canonical field value always wins if both canonical and alias keys exist
   - no alias may override a canonical value
   - multiple aliases for the same target must fail-loud unless identical by policy
   - `AliasChoices` order must put the canonical field name first if used.
   Also note `populate_by_name=True` is model config, not a `Field` argument; implement as `model_config = ConfigDict(populate_by_name=True)` or current pydantic equivalent.

5. [1, 2(b)] The whitelist taxonomy has no concrete data model, so `_normalize_field_keys` cannot be coded consistently. Concrete fix: define an exact registry shape before implementation, for example:
   `FIELD_KEY_SYNONYMS: dict[type[BaseModel], dict[str, tuple[str, ...]]]`
   for top-level fields only, or a path-aware equivalent if nested fields are required. Define helper signature:
   `_normalize_field_keys(data: object, schema: type[BaseModel], ve: ValidationError) -> Optional[tuple[dict[str, Any], list[tuple[str, str]]]]`
   Required behavior:
   - only run when `data` is a dict
   - inspect `ve.errors()` for `type == "missing"`
   - only fill target fields that are actually missing
   - only from exact whitelisted aliases
   - never fuzzy-match
   - never positional-map
   - fail-loud on alias collisions.

6. [1] “Non-load-bearing deterministic default” is specified as policy but not as code. Changing pydantic fields from required to optional would fire during first validation and break the strict-first design. Concrete fix: either cut defaults from this build or implement a separate failure-only helper, e.g. `_apply_whitelisted_defaults(data, schema, ve)`, backed by an explicit registry:
   `FIELD_DEFAULTS: dict[type[BaseModel], dict[str, Callable[[], Any] | Any]]`.
   It must run only after initial `ValidationError`, only for `missing` errors, and must log/telemetry every default applied.

7. [2(b)] Scope of normalization is ambiguous for nested schemas. `_clamp_overlong_strings` is explicitly top-level only; the plan says “one table per structured schema” and the harness includes nested-wrapper cases. Concrete fix: state whether build v1 supports only top-level aliases. If yes, tests must assert nested alias shapes fail-loud. If no, define path syntax and list handling, e.g. `("items", "*", "beat_index")`, before coding.

8. [4] “Append `schema.model_json_schema()`” is underspecified. `model_json_schema()` returns a dict, not prompt text. Concrete fix: serialize deterministically:
   `schema_text = json.dumps(schema.model_json_schema(), ensure_ascii=False, indent=2, sort_keys=True)`.
   Define truncation or a max schema prompt budget for large schemas. Also include whether `$defs` are kept. Without this, repair prompt output and token cost are unstable.

9. [5] “Audit every structured pass” is not a buildable migration plan. Grounding says `structured_call` currently shipped module/tests only and no call sites are converted. Concrete fix: produce an inventory with file/function/schema/failure-mode/migration choice for each structured pass before claiming model-agnostic coverage. At minimum verify the grounded list from the module comment: story brief reflection, cast contract, critic, news interpreter. [ASSUMPTION] Other hand-rolled passes may exist.

10. [4, VERIFY-AT-BUILD] The grounding confirms `make_dispatching_repair_factory` routes `PostValidationError` by message substrings: `"locked cast"`, `"named_character"`, `"dialogue_verb"`, `"plot_verb"`, `"too_long"`. The plan leaves this as verify-only, but it is now verified. Concrete fix: replace prose substring dependencies with stable error codes/constants emitted by post validators, e.g. `OTR_ERR_LOCKED_CAST`, `OTR_ERR_NAMED_CHARACTER`, and dispatch on those codes.

SHOULD-FIX:
1. [2(b), 4] If rejected/extra keys are included in repair text, define an alias-aware helper. With pydantic default `extra="ignore"` [ASSUMPTION], unknown keys are not available from `ValidationError`. Compute them from raw `data.keys()` minus canonical field names and whitelisted aliases; do not rely on pydantic errors.

2. [6] Telemetry is not specified as an interface. This module is documented as pure, so avoid hidden global mutable counters unless thread-safety and reset behavior are defined. Concrete fix: use structured logging first, or add an optional callback like `on_event: Callable[[StructuredCallEvent], None] | None`.

3. [3] Define behavior for `max_attempts=1` and `max_attempts=2` after the ladder change. Current code supports arbitrary `max_attempts`; the new branchy ladder must still fail predictably and report accurate `attempts_run`.

4. [2(a)] If native aliases remain, test pydantic v2 behavior explicitly for:
   - canonical only
   - alias only
   - canonical plus alias with conflicting values
   - alias plus unrelated extra
   - serialization/model_dump field names.
   This is needed because alias precedence is easy to get wrong.

5. [6] Harness fixture names are listed, but expected outcomes are not. Define per-fixture expected result: “validates via strict”, “validates via key normalization”, “validates via clamp”, “goes to typed repair”, or “fails loud”.

6. [4] Schema-in-repair may expose long descriptions/examples if schemas include them. [ASSUMPTION] If schemas contain prompt text or examples, filter schema JSON to only `properties`, `required`, `type`, constraints, and `$defs` needed for validation.

OPTIONAL / NICE-TO-HAVE:
- [6] Add per-helper event IDs in logs so soak output can be aggregated without parsing English log text.
- [2(b)] Return the list of normalized fields from `_parse_and_validate` only in debug/telemetry paths; do not change public return type.
- [GROUNDED CORRECTIONS] Fix the “4-attempt retry ladder” comment as stated; this is straightforward but not blocking relative to the API/control-flow issues above.

CUT THESE (over-engineering):
1. [2(a)] Cut native pydantic aliases for the first build. The failure-arm `_normalize_field_keys` can repair alias-key outputs in the same LLM attempt without token cost, while avoiding alias precedence/collision risks on already-valid payloads.

2. [1] Cut deterministic defaults from the first build unless a concrete field registry already exists. Safe because missing required fields should remain fail-loud by default, and key aliases solve the stated model-variance case.

3. [4] Cut rejected/extra-key reporting from the first build. It is useful repair context but not required for deterministic key normalization or schema-aware repair; computing it correctly across aliases/nested models is extra surface area.

4. [6] Cut nested-wrapper “repair” unless a real migrated pass needs it. Keep the fixture as fail-loud coverage if wrapper unwrapping is not explicitly whitelisted.