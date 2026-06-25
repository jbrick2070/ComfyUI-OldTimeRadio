# NESTED-ALIAS FORK -- CONVERGED RESOLUTION (pass01, build-ready)

1 round, LIVE: GPT-5.5 + Gemini-3.1-pro + DeepSeek-v4-pro, Claude grounded
judge+panelist, every claim checked vs the real `_otr_structured_call.py` +
`_otr_radio_editor.py`. Spend ~$0.1294. CONVERGED: all 3 panel + the anchor
return yes-with-fixes on Candidate A; cut B and C. The only split (keep vs drop
pass04's separate except-arm `_normalize_field_keys`) is resolved by the judge
with grounding. No new architecture -> one round suffices (stop at convergence).

## DECISION

Adopt **Candidate A**, in the simplified form the grounding supports:
deterministic alias normalization lives in a **shared pure helper called from a
per-schema `@model_validator(mode="before")`** -- NOT a mixin base, NOT a class
decorator, NOT a recursive core path-walker. This fixes the proven NESTED
`BeatEdit.action` failure during pydantic's own recursive validation, stays
byte-identical on canonical input, and keeps the reusable `validate_tolerant_data`
core for the binary lane. This SUPERSEDES pass04 C1's "top-level-only except-arm
`_normalize_field_keys`" (which could not reach the nested proven failure).

## MECHANISM (the shared helper + per-schema before-validator)

`_otr_structured_call.py` (stdlib + pydantic only -> no import cycle), add a PURE
helper:

```python
def apply_field_aliases(aliases, data):
    """Deterministic, whitelist-exact key normalization for ONE model's own
    top-level keys. No-op on canonical input (byte-identical). Returns `data`
    unchanged for non-dicts (mode='before' may receive non-dict inputs)."""
    if not isinstance(data, dict) or not aliases:
        return data
    out = None  # copy-once on first move (perf: no per-alias re-alloc)
    for canonical, synonyms in aliases.items():
        if canonical in data:            # explicit canonical ALWAYS wins
            continue
        present = [s for s in synonyms if s in (out if out is not None else data)]
        if len(present) != 1:            # 0 or >=2 (collision) -> leave failing
            continue
        if out is None:
            out = dict(data)
        out[canonical] = out.pop(present[0])   # move + drop the synonym key
        log.debug("[OTR_StructuredCall] aliased %r -> %r", present[0], canonical)
    return out if out is not None else data
```

Per annotated schema, a one-line before-validator (explicit -- no inheritance /
decorator hazard; pydantic v2 compiles the core schema at class-creation, so a
post-hoc decorator can fail to register):

```python
__otr_field_aliases__: ClassVar[dict[str, tuple[str, ...]]] = {...}

@model_validator(mode="before")
@classmethod
def _otr_alias(cls, data):
    return apply_field_aliases(cls.__otr_field_aliases__, data)
```

Pydantic recurses into `List[BeatEdit]` and runs this on every nested BeatEdit
during `RadioEditPlan.model_validate` -> the nested fix needs NO path-walking.

## C0 -- annotate the proven-failure schema = `BeatEdit` (NOT RadioEditPlan)

In `_otr_radio_editor.py` (add `ClassVar` to the typing import):

```python
class BeatEdit(BaseModel):
    __otr_field_aliases__: ClassVar[dict[str, tuple[str, ...]]] = {
        "beat_index": ("index",),
        "merge_with_index": ("merge_with",),
        "action": ("lever",),
    }
    beat_index: int = Field(..., ge=0, ...)
    action: str = Field(..., ...)
    ...
    @model_validator(mode="before")
    @classmethod
    def _otr_alias(cls, data):
        return apply_field_aliases(cls.__otr_field_aliases__, data)
```

REPLACE the body of the shipped `_accept_field_aliases` (BUG-LOCAL-303) with the
shared call -- KEEP the validator METHOD on BeatEdit (do not delete the method;
just swap its body / rename to `_otr_alias`). Behavior is a strict superset of
today (adds `lever`->`action`); `index`/`merge_with` behavior is preserved
exactly (canonical-wins, dict-only guard). v1 annotates ONLY BeatEdit.

## C1 (RE-SPEC) -- aliases via the before-validator; DROP the separate except-arm normalizer

pass04 C1 specified `_normalize_field_keys(data, schema, ve)` running top-level-
only in the `except ValidationError` arm. Grounding shows that is (a) unable to
reach the nested proven failure, and (b) REDUNDANT once every annotated schema
carries the before-validator (the strict-first `model_validate` already remapped,
so the except arm never fires for an alias). Resolution: DO NOT build a separate
except-arm alias normalizer. The before-validator is the SINGLE alias mechanism,
uniform for top-level AND nested annotated schemas (Gemini MUST-FIX 3, judged
correct). A future top-level schema gets aliases the SAME way (declare the map +
the one-line validator) -- no second code path.

## C2 -- the except arm keeps CLAMPING only (the existing shipped coercion)

`_clamp_overlong_strings` (top-level `string_too_long` -> clamp to max_length,
shipped 2026-06-18) STAYS in the `except ValidationError` arm; it genuinely needs
the post-failure `ve` (the max_length ctx). No alias logic in the except arm.
Keep the single-pass clamp as-is (it already clamps ALL overflowing top-level
fields in one pass); a bounded multi-coercer loop is unnecessary now that aliases
left the except arm. KNOWN LIMITATION (documented, not a v1 blocker -- DeepSeek
MUST-FIX 3): the before-validator remap and the top-level clamp operate on
different dicts (post-remap-internal vs the external parsed dict), so they do not
compose for a field that is BOTH aliased AND a top-level length-capped string. No
such field exists today (BeatEdit has no top-level capped strings; the clamp was
already top-level-only). Do NOT declare an alias for a top-level `max_length`
string until the clamp reads the post-remap dict.

## C5 -- shared tolerant core (concrete signatures; what the binary lane reuses)

In `_otr_structured_call.py`:

```python
def validate_tolerant_data(data, schema, *, post_validator=None):
    """strict-first model_validate; on ValidationError, clamp overlong strings +
    revalidate; then the PostValidationError step. The reusable core."""
def parse_validate_tolerant(raw, schema, *, post_validator=None):
    return validate_tolerant_data(parse_first_json_object(raw), schema,
                                  post_validator=post_validator)
```

`_parse_and_validate` becomes a thin caller of `validate_tolerant_data`
(preserve PostValidationError behavior EXACTLY -- it must still run AFTER
schema-validation and raise `PostValidationError`, caught by the ladder). The
binary lane's 1-field `Literal["A","B"]` schema declares no `__otr_field_aliases__`
and no before-validator -> wholly unaffected; `apply_field_aliases({}, data)`
returns `data`. `validate_tolerant_data` treats a schema with no aliases as a
plain strict+clamp validate (byte-identical to today for every existing schema).

## C3 / C4 -- UNCHANGED from pass04 (still build them)

C3 (skip the structural rung on a non-`JSONDecodeError`; narrow the except arms
to `(json.JSONDecodeError, ValidationError, PostValidationError)`; `attempts_run`
only-on-execute; "4-attempt" docstring -> "3-attempt") and C4 (call-site
`make_dispatching_repair_factory(*, schema=...)` + `_build_schema_snippet`,
injected at the radio_editor call site, core never imports `_otr_repair_prompts`)
proceed as pass04 specifies. The nested-alias work does not touch them. Note the
radio_editor call site passes `repair_prompt_factory=_default_repair_prompt`
TODAY -> C4 swaps it to the dispatching factory carrying `schema=RadioEditPlan`.

## COLLISION / FAIL-LOUD RULE (deterministic; all sources converged)

- canonical key present -> leave data unchanged; canonical WINS even if a synonym
  also exists and disagrees (matches the shipped `index` vs `beat_index`).
- canonical absent + EXACTLY ONE synonym present -> move it to canonical.
- canonical absent + >=2 synonyms present -> NO mapping; let pydantic fail the
  required field (fail-loud). (Reject Gemini's "pick the first synonym" -- it
  risks silent-wrong on a load-bearing field and contradicts pass04 C1's
  collision rule.)
- An alias only RENAMES a present value; a genuinely missing `action` (no `lever`,
  no `action`) still raises. `action` is `str` (not Literal), so `lever`->`action`
  can make a pydantic-valid-but-wrong object; Guard1 (`post_validate_plan`, run as
  the structured_call `post_validator`) rejects an out-of-`ALL_ACTIONS` value
  LOUDLY -> fail-closed, never silent-wrong.

## C6 -- conformance tests (offline, no GPU/net) -- land with the core chunk

`tests/test_schema_adherence_conformance.py`:
- proven failure: `RadioEditPlan.model_validate({"edits":[{"index":14,"lever":
  "SHORTEN_LINE","beat_index":14}],"projected_word_total":120})` ->
  `edits[0].beat_index==14`, `edits[0].action=="SHORTEN_LINE"`.
- byte-identity: canonical RadioEditPlan (correct `beat_index`/`action`/
  `merge_with_index`) -> `model_dump()` identical before/after the change.
- canonical-wins: `{"index":1,"beat_index":9,"action":"KEEP"}` -> beat_index==9.
- no-fabrication: missing both `action`+`lever` -> ValidationError; missing both
  `beat_index`+`index` -> ValidationError.
- fail-loud value: `{"beat_index":0,"lever":"NOT_AN_ACTION"}` validates at the
  pydantic layer (action=="NOT_AN_ACTION") but `post_validate_plan`/Guard1 rejects
  it (assert the Guard1 GuardError / post_validator error string).
- helper purity: `apply_field_aliases` does not mutate its input dict.
- binary-lane no-op: a `Literal["A","B"]` schema with no alias map validates
  unchanged; `validate_tolerant_data` on it is strict+clamp only.

## VERIFY-AT-BUILD CHECKLIST

1. Read the exact `BeatEdit` (HEAD) before editing; confirm `ALL_ACTIONS`
   contains `SHORTEN_LINE`/`SPLIT_LINE` so `action:("lever",)` is grounded.
2. `grep -n "RadioEditPlan(" + "RadioEditPlan.model_validate"` across `nodes/` ->
   confirm EVERY production validation path runs the Guard1 post_validator (the
   fail-loud backstop for the load-bearing `action`); if any path validates
   RadioEditPlan WITHOUT post_validator, note it (GPT MUST-FIX 6 / DeepSeek #6).
3. `_otr_radio_editor.py` imports `ClassVar` (add to the `from typing import ...`).
4. Run the conformance suite + the existing radio_editor self-test / golden tests
   for the length pass; zero diff vs baseline on canonical input.
5. Confirm `parse_first_json_object` raises `json.JSONDecodeError` (for C3) --
   already confirmed in `_otr_json.py`.
6. `grep -n "make_dispatching_repair_factory(" nodes/` (for C4 wiring count).

## INVARIANTS (reject any change that breaks one)

canonical-valid byte-identical; whitelist-exact aliases only (no fuzzy/positional);
load-bearing fields fail-loud (alias renames, never fabricates; Guard1 backstops
the value); no circular import (`_otr_structured_call` imports neither the writer
nor `_otr_repair_prompts`); model/transport-agnostic; deterministic/offline;
the binary lane reuses `validate_tolerant_data` unaffected; UTF-8 no BOM; SFW;
NO workflow-JSON change (env/code only).
