# R1 JUDGMENT

ACCEPTED (folded into pass01_plan):
- strict-first invariant = the byte-identity guarantee (GPT#5; matches existing
  `_clamp_overlong_strings` behavior, CONFIRMED).
- deterministic key-normalizer mirroring `_clamp_overlong_strings` +
  `cast_membership_repair` (anchor + all panel).
- native pydantic `Field(validation_alias=AliasChoices)` as the preferred alias
  form (Gemini SHOULD#1).
- skip the structural-retry rung on a ValidationError -> straight to typed repair
  (Gemini#3; explains the ~90k-token burn).
- schema into the typed-repair turn via an extended RepairPromptFactory protocol
  (GPT#3 + Gemini#1; CONFIRMED the factory carries no schema today).
- per-pass load-bearing field taxonomy (all 3 + DS#2).
- cover the hand-rolled passes via migrate-or-shared-helper (GPT#5 + DS#5).
- offline conformance harness + telemetry counters (anchor + GPT#6).
- fix the "4-attempt" entrypoint comment (GPT SHOULD#1; CONFIRMED 3-attempt impl).

REJECTED (with reason):
- GPT MUST#1 "remove the OpenRouter response_format branch" -- MISREAD. CONFIRMED
  it is `json_object` mode (model-agnostic "return JSON"), not provider
  `json_schema`/tool mode; byte-identical no-op for local; correctness does not
  depend on it. Kept + reworded the constraint instead.
- Gemini#4 global `extra="forbid"` -- partially rejected: the INSIGHT (the dropped
  key is hidden from the repair error) is correct + folded, but flipping every
  schema to forbid would fail benign extras on attempt 1 + risk byte-identity. We
  surface rejected keys by COMPUTING them for the repair text, not by changing the
  validation policy.
- Lever B wholesale, Lever C in the base prompt, Lever E fuzzy/positional (all 3 +
  anchor converged): silent-wrong / byte-identity / token bloat.

VERIFY-AT-BUILD: dispatch error-string substring matching (GPT#7); the exact
normalize_length schema + real fixture (GPT#6); regression-corpus coverage of all
passes (DS); base-prompt English key descriptions (Gemini).

CONVERGENCE: HIGH at R1. Anchor + 3 panels independently agree on the spine
(strict-first; deterministic aliases/normalizer; schema-in-repair-only; skip
structural rung on ValidationError; taxonomy; cover hand-rolled). No architectural
disagreement remains. R2 pressure-tests the CODING plan: the protocol-extension
signature, the normalizer contract, error-code standardization, and the shared
parse_validate_tolerant seam.
