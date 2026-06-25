<!-- Claude grounded anchor (R3 wiring/integration). Written before fan-out,
grounded vs the real _otr_repair_prompts.py + the 15-caller inventory. -->

VERDICT: yes-with-fixes. Option A's MECHANISM (schema is an OPTIONAL enrichment
param; default None == today, byte-identical for all 15 current callers) is the
correct best practice; a REQUIRED schema (Option C) is a big-bang flip across 9
modules that contradicts pass04's own "incremental per-schema opt-in, never a
big-bang flip" rule and is exactly the wiring the operator flagged. But be honest
about value: C0/C1 already fix the proven failure deterministically, so the
snippet's marginal benefit NOW is low -- the highest-value, lowest-risk C4 is to
land the OPTIONAL infra + the deterministic-repair TYPE-CHECK (a real safety fix
independent of the snippet) and wire AT MOST one schema, not force 15.

MUST-FIX BEFORE BUILD:

1. [API design / Q1] Make `schema` OPTIONAL: `make_dispatching_repair_factory(*,
   schema: type[BaseModel] | None = None, deterministic_repair=None)`. With
   `schema=None` the factory's output is IDENTICAL to today (no snippet) -> all 14
   no-arg callers + the 1 `deterministic_repair` caller (outline phase) are
   byte-identical, no edits. CONFIRMED: a required param breaks all 15
   (grounded -- 14 call `make_dispatching_repair_factory()`, 1 calls it with
   `deterministic_repair=` only). Append the snippet only when `schema is not None`.

2. [snippet placement / Q2] Append the schema snippet ONLY in the
   `isinstance(error, ValidationError)` non-payload-null branch (the
   `schema_field_repair` path). NOT json-syntax (the JSON did not parse -- a field
   schema is noise), NOT payload-null (it has a dedicated directive), NOT
   cast-membership / content (PostValidationError -- the locked-cast list is the
   relevant context, not the JSON schema). CONFIRMED against the factory's existing
   isinstance routing.

3. [deterministic-repair type-check / Q from pass04 C4] Land the type-check
   regardless of the snippet: immediately after `resolved =
   deterministic_repair(...)`, if `resolved is not None and schema is not None and
   not isinstance(resolved, schema)` -> raise `TypeError`. (Only enforce when a
   schema is supplied; the current `deterministic_repair` caller passes no schema,
   so its behavior is unchanged unless it opts in.) This prevents a wrong-type
   deterministic "fix" being returned to `structured_call` as a finished instance.
   Safe + cheap; the single genuine correctness hardening in C4.

4. [proven path / Q3] The radio_editor editor-plan path uses a STRING-returning
   local `_default_repair_prompt`, not the dispatching factory. Do NOT migrate it
   to the dispatching factory (it carries a tuned Guard-1/2/3 + no-new-noun
   directive that must survive). If wiring the snippet there, AUGMENT
   `_default_repair_prompt` to append `_build_schema_snippet(RadioEditPlan)` to its
   returned string. But note (SHOULD-FIX 1) the marginal value is low.

5. [`_build_schema_snippet` safety / Q4] Guard the recursion: `model_json_schema()`
   on a self-referential model emits `$ref` into `$defs` (BeatEdit is not
   recursive, but be general) -- strip bloat keys by walking dict/list nodes only,
   never following `$ref` targets (they live in `$defs`, walked once); cap depth or
   rely on the JSON tree being finite (model_json_schema returns a finite tree with
   `$ref` strings, so a plain recursive dict/list strip terminates). Cap ~2000
   chars with a truncation marker. PRESERVE `$defs/$ref/items/anyOf/enum` +
   `minimum/maximum/minLength/maxLength`.

SHOULD-FIX:

1. [value honesty / Q5] Because C0/C1 fixed the proven RadioEditPlan failure, the
   snippet on radio_editor's repair only helps a RESIDUAL non-alias field error
   (rare). Recommend: land the optional infra + the type-check + `_build_schema_
   snippet` + tests, and EITHER wire exactly the RadioEditPlan opt-in (1 line in
   `_default_repair_prompt`) as the worked example OR wire none and document the
   one-liner. Do NOT speculatively annotate the other schemas (same discipline as
   C0: opt in from a REAL captured failure, not a guess).

2. [no per-caller churn / Q6] A required-schema rollout is unnecessary precisely
   because the snippet is opt-in value, not a correctness fix. The optional param
   IS the "shim" that lets any caller gain the snippet later with a one-token
   `schema=` add, no factory change.

OPTIONAL / NICE-TO-HAVE:
- A conformance test: `make_dispatching_repair_factory(schema=RadioEditPlan)` on a
  ValidationError appends "Schema constraints:" containing `beat_index`/`action`;
  with `schema=None` the output is byte-identical to today; json-syntax + payload
  -null branches never carry the snippet.

CUT THESE (over-engineering):
1. Option C (required schema + edit all 15 callers). Big-bang across 9 modules for
   opt-in value; contradicts the incremental philosophy; the operator-flagged
   sticky wiring. The optional param delivers the same capability incrementally.
2. Speculatively wiring `schema=` into the other 14 callers now -- no captured
   failure justifies them (C0 discipline).

[ASSUMPTION] Option B (full defer) is acceptable if the operator prefers zero new
surface now -- but the deterministic-repair type-check (MUST-FIX 3) is a real
safety fix worth landing even then; folding it in is cheap.
