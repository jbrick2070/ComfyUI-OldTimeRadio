<!-- requested_model: deepseek/deepseek-v4-pro | resolved_model: deepseek/deepseek-v4-pro-20260423 -->

VERDICT: no. Multiple tasks lack critical specifications (beat_lo/beat_hi source, speech_signature generation, ending_change guarantee) that would block implementation; the plan is not build-ready as-is.

MUST-FIX BEFORE BUILD:
1. [T1.1] The plan does not define where `beat_lo` and `beat_hi` values come from, only asks to confirm they are available. If they are not present in `_build_user_prompt` scope, the task cannot be completed. Fix: specify how to compute or retrieve per-beat word count bands (e.g., derive from `beat_target_words` with a fixed ratio, or access from beat metadata). Without this, implementation is blocked.
2. [T1.3] The plan states `meta.dramatic_state.ending_change` is "always present" but provides no verification or fallback. If it is missing, the outro prompt will lack context and the post-check may fail. Fix: add a guard: if `ending_change` is missing, treat as unresolved and skip the recompose; or verify in code that it is always set.
3. [T2.2] The plan says "add speech_signature (<=5 words) to each cast card" but does not specify how the signature is generated. Without generation logic, cast cards will lack the field and the acceptance test will fail. Fix: define the generation method (e.g., LLM call during casting, or derived from character traits) and where it is invoked.
4. [T1.3] The post-check requires the composer to classify `ending_change` as RESOLVED using a "keyword rule in the script". The plan does not specify that the composer will have access to this classification logic, risking divergence from the scan script. Fix: extract the RESOLVED classification into a shared utility used by both the composer and the scan script, or hardcode the set of resolved endings in the composer and document that it must match the scan script.
5. [T1.2] The "empty case" handling: "force the contract onto the chosen last character slot" is undefined when there are no character beats (zero-eligible path). The plan mentions a unit test for zero-eligible but does not state the expected outcome (e.g., episode invalid, contract not created). Fix: clarify that if no character beats exist, the contract is not created and the episode will be invalid (acceptable under acceptance target), or define a safe fallback.

SHOULD-FIX:
1. [Measurement contract] `length_pass_fired` is described as a bool but the target is a count (<=2/12). Clarify that the scan script reports a count of legs where the pass fired, not a per-leg bool, to avoid confusion.
2. [T1.3] The fallback "final-beat summary" is not defined. Specify what constitutes the final-beat summary (e.g., the beat's description text) and ensure it is available at outro-compose time.
3. [T2.1] The plan says "pass normalized {gender}/pronouns ... or require them in the _otr_casting contract" but does not confirm that cast cards already contain gender/pronouns. Verify and, if missing, add them to the casting output.
4. [T3.1] The arc_shape selection is "SEEDED" but the plan does not specify which seed is used or how the mapping from seed to arc_shape is done. Document the seeding mechanism to ensure reproducibility across runs.
5. [T1.1] The test "no mid-sentence truncation" is vague for a unit test. Clarify that the unit test checks prompt formatting and token cap, while the smoke test measures truncation indirectly via length_pass_fired.

OPTIONAL / NICE-TO-HAVE:
- Add a maximum recompose attempts guard for F3 and F7 to prevent infinite loops in edge cases (though single recompose is specified, a safety net is prudent).
- For F4, include pronoun-consistency tests that are name-independent to catch mismatches like "Mister <female>".

CUT THESE (over-engineering):
- None identified; the plan is appropriately scoped.

[ASSUMPTION] `meta.dramatic_state.ending_change` is always present (stated in plan but unverified). [ASSUMPTION] Cast cards already contain gender/pronouns or can be derived without additional generation. [ASSUMPTION] The "final-beat summary" is accessible at outro-compose time (not defined).