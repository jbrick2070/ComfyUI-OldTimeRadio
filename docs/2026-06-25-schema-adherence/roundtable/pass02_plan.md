# MODEL-AGNOSTIC SCHEMA ADHERENCE -- BUILD-READY CODING PLAN (pass02, post-R2)

R2 panel (GPT-5.5 / Gemini-3.1-pro / DeepSeek-v4-pro) grounded vs the real
_otr_structured_call.py + _otr_repair_prompts.py. R1+R2 spend ~$0.26.

## INVARIANT (reworded after R2) -- "canonical-valid is untouched"
Byte-identity guarantee = **any input that validates under the CANONICAL schema
(no coercion needed) returns the same instance it does today.** All new tolerance
runs ONLY inside the `except ValidationError` arm of `_parse_and_validate`, after
the first `schema.model_validate(data)` has already failed. (This is exactly how
`_clamp_overlong_strings` already behaves -- we extend that arm.)

## DECISIONS (panel forks the judge resolved)
- **Native pydantic aliases (R1 lever 2a): CUT from v1.** AliasChoices fires
  DURING `model_validate`, not in the except arm (GPT#4) -> it muddies the
  invariant + adds collision/precedence rules. v1 uses ONE mechanism: the
  failure-arm `_normalize_field_keys`. (Native aliases = deferred optimization
  only if the extra failed-validate cycle ever shows up as a cost.)
- **Schema-into-repair: closure, not a protocol change.** Do NOT alter the
  `RepairPromptFactory` Protocol or the other 6 factories (passing `schema=` would
  `TypeError` them all -- CONFIRMED). Instead `make_dispatching_repair_factory`
  CLOSES OVER `schema` at construction and injects it into the `schema_field_repair`
  branch only. Zero churn to the other factories. (all 3 panel endorsed this as the
  low-churn option.)
- **Hand-rolled coverage: one shared CORE, not a second ladder.** Gemini's worry
  (two ladders diverge) is answered by making the shared unit the strict-first
  parse+coerce CORE (NO LLM, NO retry) -- `parse_validate_tolerant` -- that BOTH
  `structured_call` AND each hand-rolled site call. One tolerance implementation,
  reused; the ladder stays only in `structured_call`.

## THE BUILD (ordered; each chunk: suite + Bug Bible green, commit+push)

### C0 -- Inventory + taxonomy (gates everything; pure docs+data)
- Audit every structured-JSON pass: file / function / schema / on-`structured_call`
  vs hand-rolled / failure-mode. Start from the module-comment list (story-brief
  reflection, cast contract, critic, news interpreter) + grep for
  `model_validate` / `parse_first_json_object` call sites. Output: a checklist.
- Define the taxonomy data shape, co-located with the schema to avoid registry
  drift (DS#4): a class attribute on each pydantic model:
  `__otr_field_aliases__: dict[str, tuple[str, ...]]`  # canonical_field -> synonyms
  Only top-level fields. The helper reads `getattr(schema, "__otr_field_aliases__",
  {})`; absent attr => no normalization (so unannotated schemas are byte-identical).

### C1 -- `_normalize_field_keys` (the deterministic key fix; mirrors `_clamp_overlong_strings`)
```python
def _normalize_field_keys(
    data: object, schema: type[BaseModel], ve: ValidationError
) -> Optional[tuple[dict, list[tuple[str, str]]]]:
    """On a ValidationError carrying `missing` errors, snap WHITELISTED synonym
    keys present in `data` onto the schema's missing field names. Top-level only.
    Whitelist-exact: no fuzzy, no edit-distance, no positional mapping. Returns
    (repaired_dict, [(synonym, canonical), ...]) or None when nothing maps."""
```
- Source the whitelist from `schema.__otr_field_aliases__`.
- For each `ve.errors()` with `type == "missing"`, target field = `loc[0]`; if that
  field has synonyms AND exactly one synonym key is present in `data` AND the
  canonical key is NOT already present -> move it. COLLISION (canonical already
  present, or 2+ synonyms present) -> do NOT map that field (let it stay failing /
  fail-loud). Deterministic, first-listed-synonym order.

### C2 -- Compose the except arm (order matters; one re-validate)
Refactor `_parse_and_validate`'s `except ValidationError` so BOTH coercions apply
to `data` SEQUENTIALLY, THEN a single re-validate (Gemini#3 -- today
`_clamp_overlong_strings` short-circuits with `if repaired is None: raise`, which
would block the key fix):
```
except ValidationError as ve:
    work = data; touched = []
    nk = _normalize_field_keys(work, schema, ve)        # keys first (fixes 'missing')
    if nk: work, moved = nk[0], nk[1]; touched += moved
    ck = _clamp_overlong_strings(work, ve)              # then clamp (fixes 'too_long')
    if ck: work, clamped = ck[0], ck[1]; touched += clamped
    if not touched: raise                                # nothing to coerce -> fail loud
    instance = schema.model_validate(work)               # ONE re-validate; other errors propagate
    log.warning("[OTR_StructuredCall] coerced %d field(s): %s", len(touched), touched)
```
(`_clamp_overlong_strings` keeps working on the post-key-fix dict; its own
`ve.errors()` reference is fine because string_too_long locs are unaffected by key
moves. Verify-at-build: if a clamp needs the *re-raised* errors, compute them.)

### C3 -- Skip the structural rung on a non-syntax failure (the token fix, Gemini#3)
The ladder is linear; Attempt 2 is gated only by `attempts_run < max_attempts`.
- Narrow the ladder's `except` to `(json.JSONDecodeError, ValidationError,
  PostValidationError)` so unrelated `ValueError` propagates as a programming error
  (GPT#3 -- PostValidationError subclasses ValueError).
- Guard Attempt 2 (structural retry): run it ONLY when
  `isinstance(last_error, json.JSONDecodeError)`. For a ValidationError/
  PostValidationError go straight to the typed-repair attempt. Keep `attempts_run`
  accurate and ensure the typed-repair block is entered exactly once (DS#3). Update
  the logs so the attempt label reflects the real call, and fix the
  "4-attempt retry ladder" comment (it is 3).

### C4 -- Schema-aware typed repair (closure)
- `make_dispatching_repair_factory(..., schema: type[BaseModel])` closes over
  `schema`; its `schema_field_repair` branch appends a deterministic, FILTERED
  schema string:
  `json.dumps({k: v for k,v in schema.model_json_schema().items() if k in
  ("properties","required","type")}, ensure_ascii=False, sort_keys=True)`
  (drop `$defs`/descriptions/examples bloat; bound length; GPT#8). `structured_call`
  passes `schema=schema` into the dispatcher builder, NOT into the factory call --
  Protocol + other 6 factories untouched.
- OPTIONAL (defer): also pass `rejected_keys = sorted(set(data.keys()) -
  set(schema.model_fields.keys()) - <aliases>)` into the repair text so the model
  sees which of its keys were dropped (the `extra="ignore"` blind spot). Computed
  from raw keys, NOT from the ValidationError. Cheap but not v1-critical.

### C5 -- Shared core for hand-rolled passes
- Promote a public `parse_validate_tolerant(raw: str, schema: type[T], *,
  post_validator=None) -> T` = the strict-first + C1 + C2 core (NO LLM, NO ladder).
  `structured_call._parse_and_validate` becomes a thin caller of it.
- Migrate each hand-rolled site (from the C0 inventory) to call
  `parse_validate_tolerant` (or onto `structured_call`). One worked example in the
  PR. This is the cross-cutting guarantee -- without it, tolerance lives only in
  `structured_call`.

### C6 -- Offline conformance harness + telemetry
- `tests/test_schema_adherence_conformance.py` (pure; no GPU/network). Fixture
  corpus with EXPECTED OUTCOME per shape: canonical-valid -> validates via strict
  (unchanged); alias-key (real Opus `{index,...,beat_index}`) -> validates via key
  normalization; overlong-string -> validates via clamp; nested-wrapper / 2+
  synonyms / canonical+synonym collision -> FAIL-LOUD; prose-wrapped JSON ->
  validates; genuinely-unparseable -> raises. Pins the regression model-agnostically.
- Telemetry WITHOUT global mutable state (module is pure, GPT SHOULD#2): structured
  log payloads (or an optional `on_event` callback) -- per pass: failure class,
  key-normalize fired?, clamp fired?, LLM repair fired?, attempts.

## CUT (v1)
Native aliases (use the failure-arm normalizer); wholesale required->optional
defaults (fail-loud; revisit per-field only with a taxonomy row + test); fuzzy /
edit-distance / positional key mapping; nested-key coercion (top-level only ->
nested fails loud); rejected-key reporting (OPTIONAL, deferred); a global alias
registry (use the schema classvar).

## VERIFY-AT-BUILD
- `make_dispatching_repair_factory` routes PostValidationError by error-message
  substrings ("locked cast"/"named_character"/"too_long" -- CONFIRMED by GPT#10).
  The PRIMARY split is `isinstance` (robust); only sub-routing sniffs prose. Add
  stable error codes for the schema-repair path WITHOUT rewriting the 6 working
  branches -- scope tight; do not let it bloat v1.
- Does the byte-identity regression corpus cover ALL structured passes (DS)? Widen
  if not -- the strict-first gate is necessary but coverage must back it.
- The exact normalize_length StorySpine schema + the real failing fixture (GPT#6)
  -- read it; add as the canonical alias-key conformance fixture.
