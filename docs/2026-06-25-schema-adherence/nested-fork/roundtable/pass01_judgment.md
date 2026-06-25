# pass01 judgment -- nested-alias fork

Panel: GPT-5.5 + Gemini-3.1-pro + DeepSeek-v4-pro (R2 implementability), Claude
grounded judge+panelist. Spend ~$0.1294. CONVERGENCE: all 3 + anchor =
yes-with-fixes on Candidate A; cut B + C. One round (no new architecture).

## ACCEPTED (folded into pass01_plan.md)
- Candidate A, simplified to a shared `apply_field_aliases` helper called from a
  per-schema `@model_validator(mode="before")`. NOT a mixin base, NOT a class
  decorator (Gemini: pydantic v2 compiles the core schema at class creation; a
  post-hoc decorator can fail to register -- CONFIRMED; matches anchor MUST-FIX 3).
- Keep the validator METHOD on BeatEdit; swap its body to the shared helper, do
  NOT delete it (GPT MUST-FIX 2 -- avoids a discovery/ordering risk). CONFIRMED.
- Collision rule: canonical-wins / exactly-one-synonym / >=2 -> fail-loud. All
  sources converged; matches the shipped `index` vs `beat_index` + pass04 C1.
- Type guard `if not isinstance(data, dict): return data` in the helper (Gemini
  MUST-FIX 2: mode='before' can receive non-dicts during model_copy). CONFIRMED
  vs the existing `_accept_field_aliases` guard.
- copy-once-then-mutate (Gemini SHOULD-FIX 1 perf; matches pass04 C1 "COPY first").
- `action:("lever",)` safe because Guard1 fails loud on a non-`ALL_ACTIONS` value
  (Gemini OPTIONAL, DeepSeek #6, anchor MUST-FIX 4). CONFIRMED vs the excerpt.
- Concrete C5 signatures `validate_tolerant_data` / `parse_validate_tolerant`;
  `_parse_and_validate` delegates; PostValidationError behavior preserved (GPT
  MUST-FIX 7 / SHOULD-FIX 5). CONFIRMED -- these are NEW functions, not "unchanged".
- Binary lane: missing alias map treated as empty -> unaffected (GPT SHOULD-FIX 6).
- Full conformance test surface incl. the proven-failure fixture, byte-identity,
  no-fabrication, fail-loud value, helper purity (all 3).

## JUDGE-RESOLVED SPLIT
- Keep vs DROP pass04's separate except-arm `_normalize_field_keys`:
  GPT + DeepSeek = keep (top-level-only, same helper); Gemini = drop (split-brain;
  the before-validator already makes the schema tolerant). JUDGED: DROP. Grounded
  reason -- once every annotated schema carries the before-validator, the
  strict-first `model_validate` remaps before any ValidationError, so an except-arm
  alias normalizer is dead code for annotated schemas and only "helps" a
  misconfigured one. Dropping it removes a code path from the core the operator
  said to touch carefully. `validate_tolerant_data` (the shared core) is KEPT
  (Gemini wrongly lumped it into the cut -- it is the clamp+post_validator
  orchestration the binary lane reuses, not just aliases).

## REJECTED (with reason)
- Gemini "pick the first synonym when >=2 present": risks silent-wrong on a
  load-bearing field; contradicts pass04 C1 collision rule. Use fail-loud.
- Gemini CUT of `validate_tolerant_data`/C5: MISREAD -- C5 is the reusable core
  (strict+clamp+post_validator) the binary lane depends on; only the SEPARATE
  alias normalizer is dropped, not the core.
- Candidate B (sole reliance on C4 repair): cut -- reintroduces the proven
  exhaustion path (all 3 + anchor).
- Candidate C (recursive except-arm normalizer): cut -- pydantic recursion +
  per-model before-validator achieves the nested fix with far less core surface
  (all 3 + anchor).

## VERIFY-AT-BUILD (carried into pass01_plan.md checklist)
- Confirm every production `RadioEditPlan.model_validate` path runs the Guard1
  post_validator (GPT MUST-FIX 6 / DeepSeek #6) -- the load-bearing backstop.
- `ClassVar` import in `_otr_radio_editor.py` (GPT SHOULD-FIX 1).
- DeepSeek MUST-FIX 3 (alias + top-level clamp don't compose on the SAME field):
  documented limitation; no such field today; do not alias a top-level capped
  string until the clamp reads the post-remap dict.

## CONVERGENCE CALL
Converged at pass01 (1 round). No R2/R3/R4 re-loop -- the fix is a strict,
grounded completion of pass04 using the project's own shipped pattern; no panel
surfaced new architecture. Build per pass01_plan.md, folded into pass04 C0-C6.
