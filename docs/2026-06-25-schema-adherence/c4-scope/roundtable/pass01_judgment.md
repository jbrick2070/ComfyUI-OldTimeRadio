# C4-scope judgment -- best-practice convergence

Panel: GPT-5.5 + Gemini-3.1-pro + DeepSeek-v4-pro (R3 wiring), Claude grounded
judge+panelist. Spend ~$0.0975. 1 round.

## DECISION (judge): DEFER the schema-in-repair wiring (Option B), with the
## OPTIONAL-param recipe (Option A) documented for a real future failure; CUT
## Option C.

Rationale, grounded:
- **Option C (schema REQUIRED, edit all 15 callers across 9 modules): CUT --
  UNANIMOUS** (GPT cut #1, Gemini cut #1, DeepSeek cut). A 9-module big-bang for an
  opt-in, belt-and-suspenders feature; breaks all 15 current callers; contradicts
  pass04's own "incremental, never big-bang" rule; the operator-flagged sticky
  wiring.
- **The proven failure is already fixed by C0/C1 (deterministic `lever->action`),
  so the `normalize_length` repair turn is now DEAD CODE** (Gemini MUST-FIX 3,
  CONFIRMED -- C0/C1 validate on attempt 1, the ladder never reaches repair).
  Wiring C4 into the radio_editor path "first" would test dead code. GPT concedes
  "Option B acceptable if zero-risk shipping is preferred." DeepSeek's incremental
  A is fine too but offers no proven case to verify against.
- **`structured_call` ALREADY holds `schema`** (it is a required param it validates
  against). Gemini's grounded catch: forcing the schema through the repair FACTORY
  is redundant. The clean mechanism, IF/when built, is a pure `_build_schema_snippet`
  in `_otr_structured_call` + a guarded append in the typed-repair rung when
  `isinstance(last_error, ValidationError)` -- universal, zero call-site edits, no
  circular import (the helper is local to structured_call, which never imports
  `_otr_repair_prompts`). This SUPERSEDES pass04's "factory schema param + edit call
  sites." (Recorded as the future recipe; NOT built now -- it is a broad behavior
  change against currently-passing repair paths with no failing test to anchor it.)
- C0 discipline applies: opt in from a REAL captured failure, never speculatively.
  No structured pass other than the (now-fixed) normalize_length has a captured
  schema-drift failure, so there is nothing to wire today.

## ACCEPTED into the future recipe (when a real un-aliased drift appears)
- schema is OPTIONAL, default None == byte-identical (GPT MUST-FIX 1+2, DeepSeek #4).
- snippet ONLY on a non-payload-null `ValidationError` repair; NOT json-syntax,
  payload-null, or PostValidationError branches (all 3 panel converged).
- PREFER the `structured_call` shim (Gemini) over the factory param: structured_call
  has the schema, gives all callers coverage free, no per-caller edits. Put
  `_build_schema_snippet` in `_otr_structured_call` (pure: `model_json_schema()` +
  recursive bloat-key strip), NOT in `_otr_repair_prompts` (avoids the circular
  import pass04 worried about). For the payload-null exclusion, structured_call can
  carry its own tiny `is_payload_null` check or accept that the snippet reinforces
  the payload directive (mild).
- `_build_schema_snippet`: do NOT naive-truncate `json.dumps(sort_keys=True)` --
  with sort_keys, `$defs` can sort before the root contract, so a prefix slice
  drops `properties`/`required` AND leaves invalid JSON (GPT MUST-FIX 5, Gemini
  MUST-FIX 2). PRUNE keys BEFORE serialize; if a cap is needed, cap by dropping
  `$defs` entries, never by slicing the JSON string. Preserve `$defs/$ref/items/
  anyOf/oneOf/allOf/enum/const/min*/max*/pattern/additionalProperties` (GPT #6).
  Recursion guard on `$ref` cycles (DeepSeek #2). Strip only `description/title/
  examples/default`.
- deterministic-repair return type-check: `if resolved is not None and schema is
  not None and not isinstance(resolved, schema): raise TypeError` (GPT #4) -- only
  when a schema is supplied, so the current outline `deterministic_repair` caller is
  unchanged.

## REJECTED / CUT
- Option C (required schema big-bang): CUT (unanimous).
- Speculatively wiring `schema=` into the other 14 callers now: no captured failure
  (C0 discipline; GPT SHOULD-FIX 6, DeepSeek SHOULD-FIX 2).
- Auto-inference of schema INSIDE the factory (registry / error-title guessing):
  CUT (GPT #7) -- but note this does NOT apply to the structured_call shim, which
  has the REAL schema, not a guess.
- Migrating radio_editor's local string `_default_repair_prompt` to the dispatching
  factory now: CUT (GPT cut #2) -- it carries a tuned Guard 1/2/3 directive; don't
  disturb it for dead-code value.

## CONVERGENCE CALL
Converged at pass01 (1 round). Best practice = do NOT big-bang (cut C) and do NOT
speculatively wire belt-and-suspenders against a now-dead code path; DEFER with the
structured_call-shim recipe ready. Lever-1's load-bearing goal (C0-C3, C5, C6) is
SHIPPED. NEXT = G1 (the offline abstain-residual count gating the binary lane).
