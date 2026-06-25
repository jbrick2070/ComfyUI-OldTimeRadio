<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: yes-with-fixes. Candidate A is the viable mechanism, but the plan is not build-ready until the alias helper API, collision semantics, validation order, and test surface are made explicit against the current `_otr_structured_call.py`.

MUST-FIX BEFORE BUILD:

1. [Candidate A / C0] The shared alias mechanism is underspecified and could be implemented three incompatible ways. Pick one concrete implementation now. Smallest safe fix: add a pure helper, then keep a per-schema pydantic validator on `BeatEdit` that calls it. Do not rely on an unspecified mixin/decorator.
   Concrete shape:
   - helper accepts `data: Any` and `aliases: Mapping[str, tuple[str, ...]]`
   - returns `data` unchanged for non-dicts
   - shallow-copies only when adding a canonical key
   - never mutates the input dict in place
   - never overwrites an explicit canonical key
   - only maps when exactly one synonym is present
   Then in `BeatEdit`:
   - add `__otr_field_aliases__: ClassVar[dict[str, tuple[str, ...]]]`
   - keep `@model_validator(mode="before")`
   - call the shared helper from `_accept_field_aliases`.
   This preserves the current dict-only guard and “explicit `beat_index` wins” behavior shown in `_otr_radio_editor.py` lines 253-308.

2. [Candidate A / “REPLACES BeatEdit’s bespoke `_accept_field_aliases`”] Do not remove the `BeatEdit` validator unless the replacement is demonstrably collected by pydantic v2. A plain mixin with `@model_validator` is a pydantic ordering/discovery risk [ASSUMPTION]. Least risky for this one-schema v1 is: keep the validator method on `BeatEdit`, replace only its body with a call to the shared helper. Defer mixin/decorator rollout until there are tests proving validator inheritance/order across existing models.

3. [Candidate A / Q5] Define collision behavior exactly. Required rule:
   - if canonical key exists: leave data unchanged; canonical wins, even if alias also exists and disagrees
   - if canonical absent and exactly one synonym exists: copy synonym value to canonical
   - if canonical absent and multiple synonyms exist: no mapping; let pydantic fail required-field validation
   This matches the shipped baseline for `index` vs `beat_index` and `merge_with` vs `merge_with_index`. Add tests for disagreeing `index`/`beat_index` and `lever`/`action`.

4. [Candidate A / “ONE source of truth”] The statement is currently misleading for `schema=RadioEditPlan`. `_normalize_field_keys` running on `RadioEditPlan` cannot see `BeatEdit.__otr_field_aliases__` if it remains top-level-only. Fix the wording and implementation boundary:
   - nested aliases are handled only by the nested model’s `mode="before"` validator during `RadioEditPlan.model_validate`
   - `_normalize_field_keys(schema=RadioEditPlan, ...)` only uses aliases declared on `RadioEditPlan` itself
   - do not make `_otr_structured_call` import `BeatEdit` or `_otr_radio_editor.py`.
   This preserves invariant 3.

5. [Candidate A / Invariant 1 and Q6] Candidate A weakens the original “tolerance fires only in the `except ValidationError` arm” gate because `BeatEdit`’s before-validator runs during every validation. That is already true for the shipped `index` and `merge_with` aliases, but the plan must say so explicitly. Fix invariant wording to: canonical-valid inputs are returned byte-identically; whitelist aliases may be accepted during pydantic validation for annotated models only. Add a canonical-valid regression test for `RadioEditPlan` proving `model_dump()` is unchanged when only `beat_index`, `action`, and `merge_with_index` are present.

6. [Candidate A / Q3] `action` is typed as `str`, not as a pydantic enum/Literal in the excerpt. Therefore `lever -> action` can make a previously schema-invalid object schema-valid before Guard1 runs. That is acceptable only if every production validation path for `RadioEditPlan` runs the `post_validator`. The excerpt shows the structured-call invocation passes `post_validator=post_validator`, and the grounding note says both entrypoints share that path, but still verify all direct `RadioEditPlan.model_validate(...)` production uses. Concrete fix: add a regression test where `{"lever": "NOT_AN_ACTION"}` maps to `action` and is rejected by `post_validator` with Guard1.

7. [Candidate A / “KEEP all of pass04’s core unchanged”] The current `_otr_structured_call.py` excerpt does not contain `validate_tolerant_data`, `parse_validate_tolerant`, or `_normalize_field_keys`. A build plan cannot say “keep unchanged” against this source. Concrete fix: specify the actual new function signatures and where `_parse_and_validate` calls them. At minimum:
   - `validate_tolerant_data(data: object, schema: type[T]) -> T`
   - `parse_validate_tolerant(raw: str, schema: type[T]) -> T`
   - `_parse_and_validate` should call the tolerant validator before running `post_validator`.

8. [Candidate A / C2 + current `_clamp_overlong_strings`] Validation repair ordering is ambiguous. Current code handles only overlong top-level strings after a `ValidationError`; if there are both an alias error and a string length error, one-pass repair can still fail. Concrete fix: define a deterministic coerce/revalidate loop order, e.g.:
   1. parse JSON
   2. `schema.model_validate(data)` strict attempt
   3. on `ValidationError`, apply top-level alias normalization if possible
   4. revalidate
   5. if still failing, apply `_clamp_overlong_strings`
   6. revalidate once
   7. if still failing, raise the current `ValidationError`
   Or implement a bounded loop over the known coercers. Do not silently swallow remaining validation errors.

SHOULD-FIX:

1. [Candidate A / C0] Add `ClassVar` import in `_otr_radio_editor.py` if the alias map is declared there. Without `ClassVar`, pydantic may treat `__otr_field_aliases__` as model state or reject/ignore it depending on config/version [ASSUMPTION].

2. [Candidate A / Q1] If a future top-level schema has `__otr_field_aliases__`, both the before-validator and post-failure `_normalize_field_keys` could touch the same alias. This is deterministic only if both call the same helper and use the same collision rules. Do not duplicate alias logic in `_otr_structured_call.py`.

3. [Candidate A / tests] Add the exact proven-failure test:
   `RadioEditPlan.model_validate({"edits": [{"index": 14, "lever": "SHORTEN_LINE"}], "projected_word_total": 123})`
   should produce `edits[0].beat_index == 14` and `edits[0].action == "SHORTEN_LINE"`.

4. [Candidate A / tests] Add no-fabrication tests:
   - missing both `action` and `lever` still raises `ValidationError`
   - missing both `beat_index` and `index` still raises `ValidationError`
   - `lever` with invalid value reaches `post_validator` and fails loudly.

5. [Candidate A / current `_parse_and_validate`] Ensure `PostValidationError` behavior is unchanged. Tolerant validation must run before `post_validator`; post-validation failures must continue to be caught by the ladder’s existing `except (json.JSONDecodeError, ValidationError, ValueError)` arms.

6. [Candidate A / Binary lane invariant] Do not require every schema to define `__otr_field_aliases__`. `validate_tolerant_data` must treat missing alias maps as empty maps so the Lever-2 one-field `Literal["A","B"]` schema remains unaffected.

OPTIONAL / NICE-TO-HAVE:

- Add a debug-level log when an alias is accepted, but avoid warning-level noise on expected remote-model drift.
- Consider eventually changing `BeatEdit.action` from `str` to a generated `Literal[...]`, but that is not required for this fork because Guard1 already enforces `ALL_ACTIONS`.

CUT THESE:

1. [Candidate B] Cut. It does not deterministically fix the proven nested failure. The grounding shows the failure loc is nested under `('edits', N, 'action')`; top-level-only normalization will skip it, and the Opus retry ladder already exhausted.

2. [Candidate C] Cut for v1. Recursive `_normalize_field_keys` through arbitrary `dict/list` error locs is unnecessary here because pydantic recursion already invokes `BeatEdit` validators inside `RadioEditPlan.edits`. It increases blast radius in the shared core for no additional proven case.