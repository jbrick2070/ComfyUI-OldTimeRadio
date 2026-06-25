# MODEL-AGNOSTIC SCHEMA ADHERENCE -- FINAL HARDENED PLAN (pass04, CONVERGED)

4-round roundtable, LIVE: GPT-5.5 + Gemini-3.1-pro + DeepSeek-v4-pro each round,
Claude grounded judge+panelist, every claim checked vs the real
_otr_structured_call.py + _otr_repair_prompts.py. Total spend ~$0.66
(R1 .11 / R2 .15 / R3 .18 / R4 .22). CONVERGED at R4: no new architecture; the
remaining items are exact code contracts, folded below.

## INVARIANT (reworded after R4 -- GPT#1 caught the imprecision)
Canonical-valid output is returned unchanged (every model, every transport). The
EXISTING `_clamp_overlong_strings` tolerance runs for ALL schemas as it does today
(NOT alias-gated). The NEW behavior -- deterministic key normalization -- is the
only alias-gated piece: a schema with no `__otr_field_aliases__` gets NO key
normalization, so it is byte-identical to today. All new tolerance fires only in
the `except ValidationError` arm.

## C0 -- Annotate StorySpine + name the v1 set (NOT a 217-site gate)
- v1 annotates ONLY the schema with a PROVEN failure: the StorySpine
  `normalize_length` schema in `_otr_story_spine.py`. Other high-value schemas
  (outline macro/phase/beat, casting, news_interpreter, story_critic) are annotated
  LATER, each from a REAL captured failure -- do NOT invent synonyms (GPT#3). The
  full 217-site inventory is a follow-up appendix, not a v1 build gate (GPT/DS CUT).
- Taxonomy = pydantic v2 classvar: `__otr_field_aliases__:
  ClassVar[dict[str, tuple[str, ...]]] = {...}` (canonical_field -> synonyms,
  top-level only). Verify-at-build: read the real schema; set the mapping from the
  real Opus object (likely `{<canonical>: ("index","beat_index",...)}` -- confirm).

## C1 -- `_normalize_field_keys(data, schema, ve) -> Optional[tuple[dict, list[str]]]`
COPY first (`out = dict(data)`); read `getattr(schema, "__otr_field_aliases__", {})`.
For each `ve.errors()` with `type=="missing"`: `loc = err.get("loc") or ()`; skip
unless `len(loc)==1 and isinstance(loc[0],str) and loc[0] in schema.model_fields`
(empty/nested-loc guard); skip if the canonical key is already present; gather
synonym keys present in `out` (str keys only); if EXACTLY ONE, move it
(`out[field]=out.pop(syn)`); 0 or >=2 (collision) -> leave the field failing.
Return `(out, moved)` or None. Deterministic; no fuzzy/positional.

## C2 -- coerce-then-REVALIDATE loop (clean control flow; Gemini's rewrite)
```python
except ValidationError as ve:
    work, touched, cur_ve, instance = data, [], ve, None
    for _ in range(2):                        # keys, then clamp; bounded
        step = None
        nk = _normalize_field_keys(work, schema, cur_ve)
        if nk: work, m = nk; touched += m; step = "k"
        else:
            ck = _clamp_overlong_strings(work, cur_ve)
            if ck: work, c = ck; touched += [str(x) for x in c]; step = "c"
        if step is None: break
        try:
            instance = schema.model_validate(work); break
        except ValidationError as nve:
            cur_ve = nve                        # FRESH error feeds the next coercion
    if not touched: raise ve                    # nothing coerced -> fail loud, original ctx
    if instance is None: raise cur_ve           # coercions ran but still invalid -> fresh ctx
    log.warning("[OTR_StructuredCall] coerced field(s): %s", touched)
# (then the existing post_validator step, unchanged)
```

## C3 -- skip the structural rung on a non-syntax failure (token fix)
- Narrow EVERY attempt's `except` to `(json.JSONDecodeError, ValidationError,
  PostValidationError)`; a plain `ValueError` propagates (GPT#4). Update the
  `PostValidationError` + `structured_call` DOCSTRINGS to state only those three are
  recoverable. Verify `_otr_json.parse_first_json_object` raises `JSONDecodeError`
  (not plain `ValueError`) for unparseable output (GPT verify).
- Structural retry runs ONLY when `isinstance(last_error, json.JSONDecodeError)`;
  else go straight to typed repair. `attempts_run += 1` ONLY inside the branch that
  executes (DS#2). Log label dynamic: `"attempt %d/%d", attempts_run, max_attempts`.
- Grep plan + code for "4-attempt" -> "3-attempt" (DS#1). Tests: max_attempts=2 for
  JSONDecodeError (structural runs) vs ValidationError (structural skipped, typed
  repair once); plain ValueError propagates.

## C4 -- schema-aware typed repair, wired at the CALL SITE (no circular import)
- The CALL SITE builds `make_dispatching_repair_factory(*, schema: type[BaseModel],
  deterministic_repair=None)` and passes the result into
  `structured_call(repair_prompt_factory=...)`. `structured_call` stays ignorant +
  does NOT import `_otr_repair_prompts` (CONFIRMED would be circular).
- Inside the factory, in the `ValidationError`/`schema_field_repair` branch ONLY
  (not json-syntax, not payload-null): call the UNCHANGED `schema_field_repair(...)`,
  then append `_build_schema_snippet(schema)` to `messages[0]["content"]` as
  `"\n\nSchema constraints:\n" + snippet`. If `messages` is not a non-empty list
  whose first item has a string `content`, raise `TypeError`.
- `_build_schema_snippet(schema) -> str`: `d = schema.model_json_schema()`;
  recursively STRIP only the bloat keys (`description`, `title`, `examples`,
  `default`) -- PRESERVE structure incl. `$defs`/`$ref`/`items`/`anyOf`/numeric +
  length constraints, so nested models + arrays survive (Gemini#2 + GPT#3 R4);
  `json.dumps(d, ensure_ascii=False, sort_keys=True)`; cap ~2000 chars with a
  truncation note.
- Type-check the deterministic-repair return IN the closure, immediately after
  `resolved = deterministic_repair(...)`: if `resolved is not None and not
  isinstance(resolved, schema)` -> raise `TypeError` (never pass a wrong model
  through as `messages`) (GPT#5 + DS#3).

## C5 -- shared tolerant core for hand-rolled passes (concrete signatures)
In `_otr_structured_call.py` (no circular risk -- it imports neither
`_otr_repair_prompts` nor the writer):
- `validate_tolerant_data(data: object, schema: type[T], *, post_validator=None) -> T`
  = strict-first + C1 + C2 + the `PostValidationError` step. `_parse_and_validate`
  becomes a thin caller of it (preserve PostValidationError behavior exactly).
- `parse_validate_tolerant(raw: str, schema: type[T], *, post_validator=None) -> T`
  = `validate_tolerant_data(parse_first_json_object(raw), schema, ...)`.
- Migrate the v1 schema(s) (StorySpine first); raw-string sites call
  `parse_validate_tolerant`, already-parsed-dict sites call `validate_tolerant_data`.
  One worked migration in the PR. Import path documented for migrated sites.

## C6 -- conformance harness (LAND FIRST) + minimal telemetry
- `tests/test_schema_adherence_conformance.py` (pytest, no conftest deps, no
  GPU/net). PARSE fixtures + expected outcome: canonical -> unchanged; alias-key
  (the real StorySpine/Opus object) -> key-normalized; overlong -> clamped;
  alias+overlong-on-the-moved-field -> BOTH (exercises the C2 fresh-`ve` loop);
  2-synonym / canonical+synonym / nested -> fail-loud; prose-wrapped -> validates;
  unparseable -> raises. LADDER fixtures with a fake `slot_fn`: JSONDecodeError ->
  structural runs; ValidationError -> structural skipped + typed repair once; plain
  ValueError -> propagates; max_attempts=2 accounting. + a test that
  `_normalize_field_keys` does not mutate the input dict.
- Telemetry v1 = the existing `log.warning` coercion lines only; a structured-key
  log schema + `on_event` callback are DEFERRED to v2 (GPT/DS CUT).

## CUT (v1)
Native pydantic aliases; wholesale required->optional defaults; fuzzy/positional
mapping; nested-key coercion (top-level only); rejected-key reporting; error-code
dispatch layer (isinstance already routes); structured-log/telemetry schema +
on_event; global alias registry; the 217-site mass migration (`_otr_radio_editor.py`
+ tail deferred); annotating schemas beyond StorySpine before real failures appear.

## BUILD-CHECKLIST VERIFY-AT-BUILD (concrete)
- Read `_otr_story_spine.py` normalize_length schema; set `__otr_field_aliases__`
  from the real Opus shape; fixture #1 at `tests/fixtures/conformance/` (the exact
  alias mapping confirmed against the captured object, not guessed).
- `grep -n "make_dispatching_repair_factory(" nodes/` -> every caller forwards
  `schema=`; document the count.
- Byte-identity: run the conformance suite + the existing structured-pass golden
  tests for the migrated schema(s); zero diff vs baseline.
- Confirm `parse_first_json_object` raises `json.JSONDecodeError` (not plain
  `ValueError`) so the C3 narrowing is correct.

INVARIANTS (reject any change that breaks one): canonical-valid byte-identical;
whitelist-exact aliases only; load-bearing fields fail-loud; no circular import;
no forced transport/model; offline-verifiable; UTF-8 no BOM; SFW.
