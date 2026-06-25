<!-- requested_model: deepseek/deepseek-v4-pro | resolved_model: deepseek/deepseek-v4-pro-20260423 -->

VERDICT: yes-with-fixes — Candidate A is directionally correct, but the integration of the before-validator with the pass04 top‑level normalizer is underspecified, and several interactions (ante‑/post‑validation data flow with `_clamp_overlong_strings`, byte‑identical alias‑cleanup, exact validator wiring) are not resolved in the document.

MUST-FIX BEFORE BUILD:
1. [Candidate A / integration into _parse_and_validate] The pass04 `_normalize_field_keys` must be placed inside `_parse_and_validate`’s except `ValidationError` arm. The plan does **not** define the ordering between `_normalize_field_keys` and `_clamp_overlong_strings`, nor how re‑validation loops are composed. Wrong ordering can mask errors or break the ladder. Specify: (a) whether normalize runs before or after clamping; (b) the exact re‑validation policy (e.g., after each repair step re‑validate, and if still failing, run the other); (c) how the normalized dict is passed to clamping. Flesh out the except‑arm block so the implementer has unambiguous control flow.

2. [Candidate A / byte‑identity & alias cleanup] The before‑validator should ensure that removing the alias key or leaving it extra produces the same pydantic instance as canonical‑only input. The current `_accept_field_aliases` keeps the alias key; the shared mechanism must document its behaviour (keep or drop) and confirm byte‑identity on a canonical‑only dict. Ambiguity risks a future pass that inadvertently stores extra fields, breaking `test_audio_byte_identical`.

3. [Candidate A / before‑validator vs `_clamp_overlong_strings`] The `_clamp_overlong_strings` helper (in `_ctructured_call.py`) operates on the original `data` dict, not the dict after before‑validators have remapped keys. If a remap moves a value to a canonical key and the canonical key later overflows max‑length, `_clamp_overlong_strings` will look for the canonical key in the original `data` and find nothing → clamping is silently skipped. For the immediate schema (`BeatEdit`) the overlong fields are not top‑level strings, but for future schemas this is a real gap. The plan must either (a) use the post‑remap dict for clamping, or (b) declare that the alias feature is forbidden for top‑level string fields, and guard against it.

4. [Candidate A / validator mechanism] The plan treats “mixin base” vs “decorator” vs “per‑schema helper” as a panel question but does not select one. pydantic‑v2 validator inheritance and ordering can bite when other validators are added later. The safest route for a small core is a simple per‑schema `@model_validator(mode="before")` that calls a shared helper function, **not** a mixin base class. Decide and document the pattern before implementation.

SHOULD-FIX:
5. [Section: top‑level `_normalize_field_keys`] The plan says `_normalize_field_keys` reads the same `__otr_field_aliases__` map. The map lives on `BeatEdit`, not on `RadioEditPlan`. Clarify that `_normalize_field_keys` only consults the **top‑level** schema’s own `__otr_field_aliases__` (future extension) and does **not** descend into nested models, avoiding confusion when the implementer later adds top‑level aliases.

6. [Section: Guard1 & truncated lever value] The document mentions `'S...'` but does not confirm that Guard1 (`post_validate_plan`) will reject an aliased action that is not in `ALL_ACTIONS`. Explicitly note that after alias remapping, Guard1 still runs and will fail loud on any invalid action value, preserving the fail‑loud invariant.

OPTIONAL / NICE-TO-HAVE:
- No optional items.

CUT THESE (over-engineering):
- Candidate B (rely solely on C4 repair) is safe to cut; it reintroduces the proven exhaustion path.
- Candidate C (recursive `_normalize_field_keys`) is fragile and explicitly rejected by the document; further analysis is unnecessary.
- No other cuttable sections.

[ASSUMPTION] The pass04 `_normalize_field_keys` function exists (or will exist) and has a signature that accepts `(data: dict, schema: type[BaseModel], ve: ValidationError)` or similar; the integration proposal depends on that being defined, but the shape is not in the grounding.