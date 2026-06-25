# C4 SCOPE FORK -- schema-in-repair: required vs optional vs defer (decide best practice)

A scope/API-design fork inside the model-agnostic schema-adherence sprint. The
load-bearing tolerance (C0-C3, C5, C6) is SHIPPED + pushed; this is about the
remaining C4 (schema-aware typed repair) and how to wire it WITHOUT a risky
big-bang across 9 modules. Ground every claim against the appended real
`_otr_repair_prompts.py` + the caller inventory below.

## Context (what is already done)

- Lever 1 = tolerance. SHIPPED so far (2 green chunks, pushed to v2.0-alpha):
  - C0/C1: `apply_field_aliases` shared helper + `BeatEdit.__otr_field_aliases__`
    (`lever->action`, `index->beat_index`, `merge_with->merge_with_index`) routed
    through BeatEdit's `mode="before"` validator. **The proven nested Opus
    `normalize_length` failure now validates deterministically on attempt 1.**
  - C2: the existing `_clamp_overlong_strings` stays in the except arm.
  - C3: the structural retry is JSON-syntax-only (a ValidationError /
    PostValidationError skips straight to typed repair -- the Opus token-burn fix);
    except arms narrowed to `(JSONDecodeError, ValidationError, PostValidationError)`.
  - C5: `validate_tolerant_data` / `parse_validate_tolerant` shared core.
  - C6: conformance + ladder tests (offline).
- So the PROVEN failure is already fixed deterministically; the retry ladder no
  longer wastes tokens. **C4 is belt-and-suspenders only**: put a stripped JSON
  schema snippet INTO the typed-repair turn so an UN-anticipated drift on ANY
  user-chosen model can self-correct on the repair attempt.

## C4 as pass04 wrote it

`make_dispatching_repair_factory(*, schema: type[BaseModel],
deterministic_repair=None)` -- schema REQUIRED. In the `ValidationError` ->
`schema_field_repair` branch ONLY (NOT json-syntax, NOT payload-null): call the
unchanged `schema_field_repair(...)`, then append `_build_schema_snippet(schema)`
to `messages[0]["content"]` as `"\n\nSchema constraints:\n" + snippet`; if
`messages` is not a non-empty list whose first item has a string `content`, raise
`TypeError`. `_build_schema_snippet(schema)`: `schema.model_json_schema()`,
recursively strip ONLY the bloat keys (`description`/`title`/`examples`/`default`),
PRESERVE `$defs`/`$ref`/`items`/`anyOf` + numeric/length constraints,
`json.dumps(sort_keys=True)`, cap ~2000 chars + truncation note. Type-check the
deterministic-repair return: if `resolved is not None and not isinstance(resolved,
schema)` -> `TypeError`. Verify-at-build: "grep make_dispatching_repair_factory(
-> every caller forwards schema=; document the count."

## The grounding finding (the scope problem)

`make_dispatching_repair_factory` has **15 call sites across 9 modules**:
- 14 call it with NO ARGS: `news_interpreter` (1), `_otr_casting` (1),
  `_otr_continuity` (1), `_otr_creative_qa` (1), `_otr_ledger_reviewer` (3),
  `_otr_outline` (2 -- one no-arg, one with deterministic_repair), `_otr_pitch_room`
  (2), `_otr_story_critic` (1), `_otr_story_brief` (1), `_otr_story_select` (1).
- 1 call (`_otr_outline` phase stage) passes `deterministic_repair=
  _phase_cast_phantom_repair`.
Making `schema=` REQUIRED breaks all 15; "every caller forwards schema=" = edit 9
modules in one chunk (the operator-flagged "sticky wiring").

ALSO: the PROVEN-failure path -- radio_editor `normalize_length` / `run_radio_editor`
-- does NOT use this factory. It passes a LOCAL `_default_repair_prompt(*,
original_prompt, failed_output, error) -> str` (returns a STRING, not a messages
list) that names the Guard 1/2/3 / word-band violation. So "append the snippet to
messages[0]['content']" does not even apply to the proven path as written.

pass04's OWN C0 philosophy: "incremental per-pass opt-in via a ClassVar so you
harden one schema at a time, NEVER a big-bang flip." C4-as-written (schema
required -> update all 15) contradicts that.

## The fork (decide the best practice)

### Option A (anchor's lean) -- schema OPTIONAL + incremental
`make_dispatching_repair_factory(*, schema: type[BaseModel] | None = None,
deterministic_repair=None)`. schema=None (all 14 current callers + the
deterministic one) -> NO snippet, behavior byte-identical. Append the snippet only
when schema is provided. Wire `schema=` on the proven-relevant path FIRST (the
radio_editor editor-plan path -- which means augmenting its local
`_default_repair_prompt` to append `_build_schema_snippet(RadioEditPlan)`, since it
returns a string and knows its schema), and let the other 14 opt in later as each
is hardened from a real captured failure. Matches the incremental philosophy; zero
risk to the 14.

### Option B -- defer C4 entirely
C0-C3 + C5 + C6 already fix the proven failure deterministically and stop the token
burn. Ship Lever-1 without C4; treat schema-in-repair as a separate follow-up;
proceed to G1 (the binary-lane gate). Revisit C4 only when a real un-aliased drift
appears that the deterministic alias + clamp do not catch.

### Option C -- full C4 per pass04
schema REQUIRED; edit all 15 callers across 9 modules to forward their schema in
one chunk + add the snippet + the type-check. Most faithful to pass04's
verify-at-build; highest blast radius; contradicts the incremental philosophy.

## Invariants the resolution MUST guard

1. The 14+1 current callers stay byte-identical (a repair turn only fires on a
   failing input; but the schema snippet must not change behavior for callers that
   do not opt in).
2. No circular import: `_otr_structured_call` imports neither the writer nor
   `_otr_repair_prompts`; the factory is built at the call site.
3. The proven failure is ALREADY fixed (C0/C1) -- C4 must not regress it.
4. Deterministic, offline-verifiable, model/transport-agnostic, UTF-8 no BOM, SFW.
5. NO workflow-JSON change.

## Questions for the panel

1. For a SHARED repair factory with 15 callers, is a REQUIRED schema param
   (force-update all callers now) or an OPTIONAL schema param (incremental opt-in)
   the better engineering practice here? Weigh blast radius vs "every caller should
   declare its schema".
2. Is appending the schema snippet ONLY in the `ValidationError ->
   schema_field_repair` branch correct, or should json-syntax / payload-null /
   cast-membership repairs also carry it? Where does the snippet help vs add noise?
3. The proven-failure path uses a STRING-returning local `_default_repair_prompt`,
   not the dispatching factory. Best way to give it the schema snippet -- augment
   the local factory with `_build_schema_snippet(RadioEditPlan)`, or migrate it to
   `make_dispatching_repair_factory(schema=RadioEditPlan, deterministic_repair=...)`
   (and re-express the Guard directive there)? Which is lower-risk + idiomatic?
4. `_build_schema_snippet`: strip `description/title/examples/default`, preserve
   `$defs/$ref/items/anyOf` + numeric/length constraints, `sort_keys`, cap ~2000.
   Any failure mode (recursion on `$ref` cycles, dropping a load-bearing key,
   pydantic v2 `model_json_schema()` shape) to guard?
5. Given C0/C1 already fix the proven failure deterministically, is C4 worth ANY
   wiring now, or is "defer until a real un-aliased drift appears" the disciplined
   call? If worth it, which ONE schema is the highest-value first opt-in?
6. Any best-practice angle missed (e.g. a one-line shim so the 14 callers gain the
   snippet for free once their schema is known, without per-call-site edits)?
