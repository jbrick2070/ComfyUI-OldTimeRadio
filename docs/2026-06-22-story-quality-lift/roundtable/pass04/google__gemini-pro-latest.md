<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: yes-with-fixes. The plan is highly converged and structurally sound, but introduces a critical logic regression in the stage-direction floor that would cause it to fail its own acceptance criteria, plus two build-blocking ambiguities.

MUST-FIX BEFORE BUILD:
1. [Section 2] **Fix-introduced regression (Pronoun Logic)**: The rule for `is_third_person_action_clause` requires "no `_PRONOUN_ROOTS` token". If this includes third-person pronouns, it will falsely reject b010 ("clutches *her* wedding ring") and b012 ("taps *his* cane"), completely breaking the Tier 3 floor and failing the stated acceptance criteria.
   *Concrete fix*: Change "no `_PRONOUN_ROOTS` token" to "no FIRST-PERSON pronouns (I, me, my, mine, we, us)". Third-person pronouns (he, she, his, her, they) MUST be permitted in third-person stage directions.
2. [Section 3] **Build-blocking ambiguity (Source of `cast_ids`)**: The final pre-freeze sweep calls `coerce_speaker_role_for_char_id(line, cast_ids, source)` but does not specify how the cascade node acquires `cast_ids`.
   *Concrete fix*: Explicitly state that the sweep derives `cast_ids` dynamically from `set(ledger.get("cast", {}).keys())` (excluding the announcer/music sentinels).

SHOULD-FIX:
1. [Section 4] **Build-blocking ambiguity (Critic Context)**: The plan offers an implementation choice: "[pass cast/protagonist/central-object context into the critic prompt, or relax target to a free-form string validated in tests]". A build-ready plan cannot leave "or" choices to the coder.
   *Concrete fix*: Remove the choice. Lock it to: "relax `target` to a free-form string; do NOT wire new cast context into the critic prompt."
2. [Section 4] **Type Ambiguity**: `missing_turn_beat [id OR reason string]` implies a Union type which can cause Pydantic validation headaches if the critic report model is strict.
   *Concrete fix*: Define `missing_turn_beat` strictly as a `str` (which can comfortably hold either an ID or a descriptive reason).

OPTIONAL / NICE-TO-HAVE:
- [Section 2] In `_NARRATION_VERBS`, explicitly add "turns", "looks", "smiles", and "sighs" as these are the most common micro-actions in LLM dialogue generation alongside the mechanical ones listed.

CUT THESE:
- None — plan converged. The previous cuts (auto-repair, coherence hints) successfully stripped the unbuildable cross-run state.

VERIFY-AT-BUILD:
1. Confirm the exact pre-freeze sweep insertion phase sits *after* `cast_lock` in the cascade ordering, so it does not revert legitimate announcer re-stamps.
2. Confirm `compose_flags` has no strict format validation downstream that would crash on the new `"kind:detail"` string formats [ASSUMPTION].
3. Confirm the critic report model accepts the new `StanceIssue` field (lenient vs strict Pydantic validation).
4. Confirm `OTR_TEST_MODE` is the correct environment variable/conftest convention for gating the CI-only invariant asserts.