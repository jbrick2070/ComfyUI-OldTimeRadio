<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.6-sol -->

VERDICT: no — the proposed defaults are plausible, but the plan conflates a Lemmy-specific listening decision with a global IndexTTS2 rollout, leaves two competing configuration knobs, and does not define reproducible qualification evidence for the actual shipped settings.

MUST-FIX BEFORE BUILD:
1. [§4, §5] The change is global to every IndexTTS2 character, but the perceptual approval and re-qualification plan are Lemmy-specific. Changing `EMO_ALPHA_DEFAULT` and `EFFECTIVE_EMOTION_MASS_CAP` changes every IndexTTS2 render; measuring 57 vectors proves saturation, not that 0.560 preserves identity or performance for other reference voices. Either scope 0.560 to Lemmy’s route/profile, or audition a representative set of IndexTTS2 characters and re-qualify every route whose qualification contract depends on the changed engine fingerprint.

2. [§6.6, `otr_lemmy_production_audition.py:ARMS`] The audition instrument does not render the proposed shipped configuration. Its `"shipped"` arm remains alpha 0.4/cap 0.4, not alpha 1.0/cap 0.56. Update it before re-qualification and rename the arm so “shipped” cannot silently mean an obsolete build.

3. [§1, §6.6] The proposed production A/B confounds the emotion change with the seed-policy change: `"shipped"` uses character-stable seed while `"prefix"` uses per-line seed. That can establish only whether the combined build sounds different, not whether the approved 0.560 blend ships correctly. Use at least an alpha 1.0/cap 0.56 arm and an alpha 1.0/cap 8 arm under the same character-stable seed. Test per-line versus character-stable seed separately.

4. [§4, §6.1, `audio_engine_profiles.yaml:char_indextts2_v1`] “Collapse onto one knob” is not the architecture being proposed. Alpha remains an environment override, a cache/receipt field, an acceptance-check input, and a profile `default_params` field, while the actual tuning knob—mass cap—does not appear in the profile defaults. Define one canonical production configuration source. Smallest fix: classify alpha explicitly as a compatibility/diagnostic override fixed at default 1.0, add `emo_mass_cap: 0.56` to the profile metadata, and verify that request building, rendering, receipts, and cache keys resolve the same values. [ASSUMPTION] Verify whether profile `emo_alpha` currently affects runtime behavior anywhere outside the shown adapter; `emotion_payload` itself resolves from the environment/constant, not the profile.

5. [§2] The claim `effective mass = min(alpha * sum(raw), cap)` is mathematically false for the shown implementation. `_apply_vendor_alpha` truncates each component independently when alpha is not 1.0, and cap rescaling floors components to three decimals, so the result can be below that idealized expression. Replace the formula with the quantized algorithm’s actual result, or state it only as an unquantized approximation. Acceptance must assert `emotion_payload()["effective_mass"]`, not the simplified formula.

6. [§6.5] Re-qualification must occur after every fingerprint-bearing edit, not merely after “the constants land.” Rewriting the docstrings in the same `eng_indextts2.py` file can change `live_engine_impl_version("indextts2")` again. Freeze all fingerprint sources—including constants, code, comments/docstrings if the whole file is hashed, profile YAML, and worker sources—then render, capture the runtime fingerprint, and write the qualification record. Verify the exact `RUNTIME_FINGERPRINT_SOURCES` before sequencing.

7. [§6.7, `otr_lemmy_production_audition.py:main`] The evidence overwrite protection is incomplete. It refuses only when `MANIFEST.json` already exists; an existing or partially rendered directory without that file can have WAVs overwritten, and the sibling `_KEY` directory is created without any refusal check. Refuse any existing nonempty output or key directory, create both with exclusive semantics, render into a temporary directory, and publish the manifest last by atomic rename.

8. [§6.4–§6.7] The audition manifest is insufficient to prove which runtime produced the qualification evidence. It records clip hashes, seeds, and effective mass, but not alpha, cap, commit, live engine fingerprint, profile hash, reference-file hash, or the blinded key’s hash. Add those fields and retain the per-line receipt/log. The qualification record should cite both the manifest hash and the key hash so arm identity cannot be rewritten later.

9. [§7] Flipping the eight stale route assertions is not an acceptance plan for the behavior being introduced. Add direct tests for default alpha 1.0, default cap 0.56, profile/runtime agreement, a production-derived vector binding at the cap, a below-cap vector remaining below it, alpha/cap environment overrides, and cache-key changes for either resolved value. Keep the synthetic stale-record tests isolated from the current shipped fingerprint so they continue exercising demotion after future re-qualification.

SHOULD-FIX:
1. [§5] “The ceiling binds on 100% of real production lines” overstates a sample of 57 character lines from six recent ledgers. Change it to “100% of the sampled 57 lines.” Future lines, zero vectors, hand-edited vectors, and other ledger populations can fall below the cap.

2. [§5] “Every real line lands exactly on the ceiling, which is exactly the rung he heard” equates total mass with perceptual equivalence. The document itself acknowledges that vector shape varies; text, shape, seed, and reference voice also affect output. State only that sampled lines match the approved rung’s total budget, then use the production audition to establish perceptual transfer.

3. [§6.1] “Permanently-inert knob” contradicts retaining alpha as an environment override. Call it inert only on the default path. Document its supported range, diagnostic purpose, cache behavior, and whether qualification records produced under non-default alpha are eligible for shipped routes.

4. [§6.2, `current_emo_alpha`, `current_emo_mass_cap`, `emotion_payload`, YAML comment] The stale narrative is broader than the two identified docstrings. The shown constant comments, `current_emo_alpha` history, profile comment, audition module description, arm labels, prediction text, and key labels all encode alpha 0.4/cap 0.4 as shipped. Update them in the same pre-qualification commit and grep for `alpha 0.4`, `ceiling 0.4`, `leaves 0.6`, and `shipped`.

5. [§7] The synthetic stale-record fixtures need an explicit contract: mismatch exactly one runtime fingerprint field while leaving the record otherwise structurally valid. Otherwise a malformed fixture could make the tests pass through an unrelated rejection path.

OPTIONAL / NICE-TO-HAVE:
- [§5] Record saturation rate and effective-mass distribution in a reusable report generated from ledger paths, commit, and delivery-table version rather than retaining only pasted aggregate numbers.
- [§6.6] Rename `"prefix"` to `"pre_fix"`; “prefix” is easy to misread and weakens evidence labeling.

CUT THESE (scope / over-engineering):
1. [§6.1] Do not delete alpha in this change. Removing it across worker payloads, cache keys, receipts, schema, and acceptance tooling is unrelated to shipping the approved default and increases rollback risk. Deprecate it as a production-facing tuning control instead.

2. [§6.6] Do not preserve the obsolete alpha 0.4/cap 0.4 arm under the name `"shipped"`. If historical comparison is still useful, label it `"previous_shipped"` and keep it outside the minimum qualification pair.

3. [§5] Do not make “all production lines must saturate” a permanent requirement. The stated goal is a maximum emotion budget, not forced uniform mass; below-cap vectors are valid behavior and need a test, not elimination.