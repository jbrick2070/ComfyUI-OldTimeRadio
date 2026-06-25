CLAUDE ANCHOR -- R2 (coding plan / implementability). Grounded vs _otr_structured_call.py (read full) + _otr_repair_prompts.py (header).

VERDICT: yes-with-fixes. The design is implementable and grounded in two existing patterns, but four coding-level seams need pinning before the keyboard.

MUST-FIX BEFORE BUILD:
1. [Step 4 -- RepairPromptFactory protocol extension] Adding `schema` to
   `RepairPromptFactory.__call__(*, original_prompt, failed_output, error)` is a
   BREAKING change to a Protocol every factory in _otr_repair_prompts.py (6-7
   classes) implements + every call site constructs. FIX: make it ADDITIVE --
   `structured_call` passes `schema=` as a keyword the factory MAY accept; either
   (a) add `schema: type[BaseModel] | None = None` to the protocol with a default
   so existing factories that ignore it still satisfy it, or (b) have
   `make_dispatching_repair_factory` close over `schema` at construction (it is
   already a factory-of-factories) so the Protocol signature is untouched. Prefer
   (b): zero churn to the other six factories; only `schema_field_repair` reads it.

2. [Step 2b -- `_normalize_field_keys` contract + hook] Pin the signature to mirror
   `_clamp_overlong_strings`: `_normalize_field_keys(data, ve, alias_map) ->
   Optional[tuple[dict, list[str]]]`, called inside the SAME `except
   ValidationError` arm of `_parse_and_validate`. Two coding decisions: (a) source
   of `alias_map` -- attach it to the schema as a classvar (e.g. `__otr_aliases__:
   dict[str,tuple[str,...]]`) so the generic helper looks it up off `schema`, NOT a
   global registry that needs syncing (DS#4). (b) Composition order with
   `_clamp_overlong_strings`: normalize keys FIRST (fixes `missing` errors), then
   clamp (fixes `string_too_long`), then ONE re-validate; any remaining error
   propagates. Define this so the two coercions never double-raise or fight.

3. [Step 2a vs 2b -- overlap collapses the build] Native pydantic
   `Field(validation_alias=AliasChoices(...))` resolves at `model_validate` time --
   so an aliased key NEVER raises a ValidationError and NEVER reaches 2b. That means
   2a and 2b are NOT parallel: 2a is primary (declarative, zero-token, catches the
   known synonyms on attempt 1), and 2b is only the fallback for passes that cannot
   be statically aliased (the hand-rolled / dynamic ones). FIX: state this
   explicitly -- "add AliasChoices to the migrated schemas; `_normalize_field_keys`
   is the cross-cutting net for the stragglers." This SHRINKS the build (mostly
   schema-field annotations) and removes the risk of two mechanisms disagreeing.

4. [Step 5 -- shared seam for hand-rolled passes] `structured_call`'s
   `_parse_and_validate` is already the right unit, but it is module-private + tied
   to the ladder. FIX: promote a public `parse_validate_tolerant(raw, schema, *,
   post_validator=None) -> T` that does strict-first + normalize + clamp + ONE
   re-validate (no LLM, no ladder), used by BOTH `structured_call` AND each
   hand-rolled site. The pass inventory (plan step 1) lists the hand-rolled call
   sites that must adopt it. Without this, "cross-cutting" is unimplementable -- the
   tolerance would live only inside `structured_call`.

SHOULD-FIX:
1. [Step 4 -- error dispatch] verify: does `make_dispatching_repair_factory` route
   by prose-substring-matching the error text (GPT#7)? If yes, the
   schema_field_repair branch is brittle. Add a stable code (attribute on the
   exception or a small enum) and match that; scope it tight (do not rewrite the
   other six branches if they work).
2. [Step 6 -- harness] Pin it as a pure offline unit test
   (`tests/test_schema_adherence_conformance.py`), fixtures inline, assertion =
   each shape -> validates OR raises the RIGHT error class. Include the real
   normalize_length failing fixture once read.
3. [Determinism] If `alias_map` lists multiple source keys for one field, define
   first-present-wins in a fixed order so the coercion is deterministic.

CUT:
1. [Global alias registry] Cut it -- attach the alias map to the schema (classvar);
   a registry is a second source of truth that drifts from the schema (DS#4's sync
   worry made concrete).

[ASSUMPTION] _otr_repair_prompts.py dispatch internals (prose vs structured error
matching) not yet read -- gates SHOULD-FIX#1; verify before scoping error codes.
