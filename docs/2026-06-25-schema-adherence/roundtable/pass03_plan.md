# MODEL-AGNOSTIC SCHEMA ADHERENCE -- BUILD-READY PLAN (pass03, post-R3 wiring)

R3 panel (GPT-5.5 / Gemini-3.1-pro / DeepSeek-v4-pro) grounded vs the real code +
a 217-site structured-pass inventory. R1+R2+R3 spend ~$0.44. Convergence is HIGH:
all three independently caught the same two wiring bugs (C2 stale-ve, C4 ownership/
circular-import).

## INVARIANT (unchanged): canonical-valid is untouched
First `schema.model_validate(data)` succeeds -> return it unchanged. All tolerance
lives in the `except ValidationError` arm. A schema with NO `__otr_field_aliases__`
is a byte-identical no-op -> the rollout is INCREMENTAL (annotate high-value
schemas first; the rest are unaffected).

## C0 -- Inventory + taxonomy (gates everything)
- Inventory the structured surface (grounded: 217 sites / 21 files;
  `_otr_radio_editor.py` 111, `_otr_outline.py` 27, `_otr_story_spine.py` 19
  [normalize_length], `_otr_reroll.py` 11, `_otr_continuity.py` 10). Record
  file/function/schema/on-structured_call-vs-hand-rolled per pass.
- Taxonomy = a pydantic v2 CLASSVAR on each schema (GPT#6):
  `__otr_field_aliases__: ClassVar[dict[str, tuple[str, ...]]] = {...}`
  (canonical_field -> synonyms; top-level only). Helper reads
  `getattr(schema, "__otr_field_aliases__", {})`; absent => no-op.

## C1 -- `_normalize_field_keys` (deterministic key fix)
```python
def _normalize_field_keys(data, schema, ve):  # -> Optional[tuple[dict, list[str]]]
    if not isinstance(data, dict): return None
    out = dict(data)                                  # COPY before move (GPT#7)
    aliases = getattr(schema, "__otr_field_aliases__", {})
    moved = []
    for err in ve.errors():
        if err.get("type") != "missing": continue
        loc = err.get("loc") or ()
        if len(loc) != 1 or not isinstance(loc[0], str): continue  # empty/nested loc guard (Gemini#3 + GPT SHOULD#1)
        field = loc[0]
        if field not in schema.model_fields or field in out: continue
        syns = [s for s in aliases.get(field, ()) if s in out and isinstance(s, str)]
        if len(syns) != 1: continue                   # 0 or collision (2+) -> leave failing
        out[field] = out.pop(syns[0]); moved.append(field)
    return (out, moved) if moved else None
```

## C2 -- Except arm = coerce-then-REVALIDATE loop (the unanimous R3 fix)
The original `ve` does NOT carry errors for fields a key-move just introduced
(e.g. a moved value that is now `string_too_long`). So re-derive the error between
coercions:
```python
except ValidationError as ve:
    work, touched, cur_ve = data, [], ve
    for _ in range(2):                                # keys, then clamp; bounded
        step = None
        nk = _normalize_field_keys(work, schema, cur_ve)
        if nk: work, m = nk; touched += m; step = "k"
        else:
            ck = _clamp_overlong_strings(work, cur_ve)
            if ck: work, c = ck; touched += [str(x) for x in c]; step = "c"
        if step is None: break
        try:
            instance = schema.model_validate(work); break   # success
        except ValidationError as nve:
            cur_ve = nve                              # FRESH error for the next coercion
    else:
        instance = schema.model_validate(work)        # final try; propagate if still bad
    if not touched: raise                             # nothing coerced -> fail loud
    if "instance" not in dir(): instance = schema.model_validate(work)
    log.warning("[OTR_StructuredCall] coerced field(s): %s", touched)
```
(Implementor: tidy the control flow; the REQUIREMENT is -- after each coercion,
re-validate to get a fresh `ValidationError` before the next coercion; fail loud if
nothing coerced. all 3 panel.)

## C3 -- Skip the structural rung on a non-syntax failure (token fix)
- Narrow EVERY attempt's `except` to `(json.JSONDecodeError, ValidationError,
  PostValidationError)` so a plain `ValueError` from slot_fn/post_validator/factory
  propagates as a programming error (GPT#4). Test it.
- Structural retry (Attempt 2) runs ONLY when `isinstance(last_error,
  json.JSONDecodeError)`; a ValidationError/PostValidationError goes straight to the
  typed-repair attempt. `attempts_run += 1` ONLY inside the branch that actually
  executes (DeepSeek#2 -- a skipped rung must not consume the budget). Log label is
  DYNAMIC: `"attempt %d/%d", attempts_run, max_attempts` (Gemini/DS -- not hardcoded
  "3/3"). Tests: max_attempts=2 with JSONDecodeError (structural runs) vs
  ValidationError (structural skipped, typed-repair runs once); plain ValueError
  propagates. Fix the stale "4-attempt retry ladder" comment.

## C4 -- Schema-aware typed repair (at the CALL SITE -- corrected wiring)
GROUNDED CORRECTION: `structured_call` does NOT build the dispatcher (callers pass
a pre-built `repair_prompt_factory`), and `_otr_repair_prompts` already imports
`_otr_structured_call` -> structured_call must NOT import the builder (circular).
So:
- The CALL SITE builds `make_dispatching_repair_factory(..., schema=schema)` and
  passes the result into `structured_call(repair_prompt_factory=...)`.
  `structured_call` stays ignorant. (GPT#1 + Gemini#2)
- Inside `make_dispatching_repair_factory`, the `ValidationError` branch calls the
  UNCHANGED `schema_field_repair(...)`, then APPENDS a capped, filtered schema
  string to `messages[0]["content"]` (Gemini optional + GPT SHOULD#4): keep only
  validation keys (type/enum/const/required/properties/min/max/maxLength), drop
  description/examples/title/$defs, `json.dumps(..., sort_keys=True)`, and CAP at
  ~2000 chars with a truncation note (GPT SHOULD#3 + DS SHOULD#1). schema_field_repair
  + the Protocol + the other 6 factories are untouched.
- Type-check the deterministic-repair return (GPT#5): in the closure, if the
  callback returns a non-None object that is `not isinstance(resolved, schema)`,
  raise `TypeError` (fail-loud) -- never pass a wrong BaseModel through as `messages`.

## C5 -- Shared core for hand-rolled passes
- `validate_tolerant_data(data: object, schema, *, post_validator=None) -> T` = the
  strict-first + C1 + C2 core (NO LLM, NO ladder), incl. the `post_validator` ->
  `PostValidationError` step (GPT SHOULD#6). `parse_validate_tolerant(raw, schema,
  ...)` = `validate_tolerant_data(parse_first_json_object(raw), ...)`. Sites that
  already hold a parsed dict call `validate_tolerant_data`; raw-string sites call
  `parse_validate_tolerant` (GPT SHOULD#5). `_parse_and_validate` becomes a thin
  caller of the core.
- Migrate the HIGH-VALUE schemas a writer-swap actually hits FIRST (outline
  macro/phase/beat, story_spine normalize_length, casting, news_interpreter,
  story_critic); one worked example in the PR. `_otr_radio_editor.py` (111) + the
  long tail are a DEFERRED follow-up chunk -- NOT v1. (incremental: the core is a
  no-op until a schema declares aliases.)

## C6 -- Conformance harness + telemetry (land BEFORE the C5 opt-ins)
- `tests/test_schema_adherence_conformance.py` (pure; no GPU/net). PARSE fixtures
  with expected outcome (canonical -> strict/unchanged; alias-key [the real Opus
  normalize_length object] -> key-normalized; overlong -> clamped; alias+overlong-on-
  moved-field -> BOTH fire; 2-synonym / canonical+synonym collision / nested ->
  fail-loud; prose-wrapped -> validates; unparseable -> raises). PLUS C3 SEQUENCING
  tests with a fake `slot_fn` (GPT SHOULD#7): JSONDecodeError -> structural runs;
  ValidationError -> structural skipped + typed-repair once; plain ValueError ->
  propagates; max_attempts=2 accounting.
- Telemetry: structured LOGS only for v1 (stable keys: helper_name, attempt,
  failure_class, key_normalized, clamped, llm_repair). No global mutable state, no
  `on_event` callback in v1 (GPT CUT#2). Module stays pure.

## CUT (v1)
Native pydantic aliases (failure-arm normalizer instead); wholesale
required->optional defaults; fuzzy/edit-distance/positional mapping; nested-key
coercion (top-level only); rejected-key reporting (deferred); error-code dispatch
layer (already isinstance-routed -- GPT CUT#1); on_event telemetry callback; global
alias registry; migrating `_otr_radio_editor.py` + the long tail (deferred chunk).

## VERIFY-AT-BUILD
- The exact StorySpine `normalize_length` schema in `_otr_story_spine.py` -- read
  it, set its `__otr_field_aliases__` from the real Opus shape, ship that object as
  conformance fixture #1.
- Enumerate `make_dispatching_repair_factory(` callers; each must forward `schema=`.
- Confirm the byte-identity regression corpus covers the migrated passes (DS).
