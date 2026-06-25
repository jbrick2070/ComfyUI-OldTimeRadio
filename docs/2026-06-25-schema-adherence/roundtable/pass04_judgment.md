# R4 JUDGMENT -- CONVERGED

VERDICT: converged. All three R4 panelists returned "yes-with-fixes" where every
"fix" is an exact code-contract / wording / scope refinement, NOT new
architecture. That is the convergence signature -- the spine held across all four
rounds; the panel is now polishing signatures.

FOLDED (R4):
- Invariant reworded: `_clamp_overlong_strings` is NOT alias-gated (runs for all
  schemas as today); only KEY NORMALIZATION is the alias-gated no-op (GPT#1 --
  CONFIRMED the clamp already runs regardless).
- C2 clean control flow replacing the `dir()` smell: `instance=None` init;
  `if not touched: raise ve`; `if instance is None: raise cur_ve` (Gemini#1 + DS#2).
- C4 schema snippet = recursively STRIP bloat (description/title/examples/default)
  but PRESERVE `$defs`/`$ref`/`items`/constraints so nested models + arrays survive
  (Gemini#2 + GPT#3 -- a whitelist would break them); concrete `_build_schema_snippet`
  + cap; inject in the schema_field_repair branch ONLY; deterministic-repair
  type-check placed in the closure (DS#3 + GPT#5).
- C5 concrete signatures + module placement for `validate_tolerant_data` /
  `parse_validate_tolerant` (DS#2).
- Scope tightened: v1 annotates ONLY StorySpine `normalize_length` (others wait for
  real captured failures -- no invented synonyms, GPT#3); the 217-site inventory and
  the structured-log/telemetry schema are follow-ups, not v1 gates (GPT/DS CUT).
- Doc hygiene: "4-attempt" -> "3-attempt" everywhere; PostValidationError +
  structured_call docstrings reflect the narrowed recoverable set; verify
  parse_first_json_object raises JSONDecodeError.

REJECTED / MISREAD at R4: none -- every R4 point grounded true and was folded or
correctly scoped as a deferred cut.

CONVERGENCE CALL: stop at R4 (the standing 4-round arc). pass04_plan.md is the
build-ready, model-agnostic schema-adherence plan: strict-first byte-identity,
deterministic whitelist key-normalizer (extends `_clamp_overlong_strings` +
`cast_membership_repair`), skip-structural-on-ValidationError (the token fix),
call-site-wired schema-aware repair (no circular import), one shared tolerant core
for the hand-rolled passes, incremental classvar opt-in, offline conformance
harness. Build order C0->C6, bisectable. prod/main GATED; do not auto-build.
