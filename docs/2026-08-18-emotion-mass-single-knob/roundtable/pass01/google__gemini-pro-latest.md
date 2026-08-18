<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview -->

VERDICT: yes-with-fixes. The math correctly collapses the 2x2 degeneracy into the single knob the operator approved, but the plan misses a CI break and invents unnecessary test work.

MUST-FIX BEFORE BUILD:
1. **[6.3] Acceptance Checker Breakage**: You noted `otr_voice_identity_acceptance.py:185` compares alpha as a string but proposed no fix. If this script is run in CI with `--expect-alpha 0.4`, it will fail the moment you change the default to 1.0. Update the acceptance checker's expected value or its CI invocation to `"1.0"`.
2. **[6.7] KEY.json Overwrite Hazard**: The audition script checks for `MANIFEST.json` in `out_dir` but blindly calls `write_text` on `key_dir / "KEY.json"`. If a previous run's `out_dir` was moved or deleted but its `_KEY` directory was left behind, you will silently overwrite the blinding key. Add `if (key_dir / "KEY.json").exists():` to the refusal block.
3. **[7] Wasted Effort on "Synthetic" Stale Records**: You state `test_the_shipped_receipt_is_no_longer_SELECTED` needs a "synthetic stale record." It does not. In the very next bullet, you correctly plan to preserve the real 2026-08-10 record in `superseded_native_routes`. Pass that exact superseded 2026-08-10 record into the selection test to prove it gets rejected.

SHOULD-FIX:
1. **[6.1] Pin or Delete Alpha**: Pin it. Deleting it requires schema migrations for `audio_engine_profiles.yaml`, breaks the cache key schema, and removes the operator's rollback lever. Keep it as an env override defaulting to 1.0, but update the docstring to state it is now a pass-through by default.
2. **[6.2] Docstring Lies**: Rewrite `current_emo_mass_cap` (delete the 2x2 degeneracy paragraph; it is now the sole intensity knob, not a secondary safety net) and `emotion_payload` (delete "alpha is the taste knob above it").
3. **[6.5] Re-qualification Ordering Trap**: The trap is test execution. You must 1) edit the code, 2) run the audition script to generate the new fingerprint, 3) update the test ledgers with the new fingerprint, and 4) run pytest. If you run pytest immediately after editing the code, the tests asserting the *new* shipped route will fail because the test ledger still holds the `c18df292a41d3ddc` fingerprint.

OPTIONAL / NICE-TO-HAVE:
- **[5] Acknowledge the Dynamic Range Loss**: The plan correctly states "Total mass becomes uniform... every character line is 0.560." You should explicitly document in `current_emo_mass_cap` that this intentionally flattens the *intensity* variance across lines, as approved by the operator, leaving only the vector *shape* to vary.

CUT THESE:
1. **[7] Synthetic stale records**: Safe to cut because the real 2026-08-10 record serves this exact purpose perfectly.

[ASSUMPTION] Section 6.5 claims "Changing `eng_indextts2.py` moves `live_engine_impl_version("indextts2")`". I am assuming `live_engine_impl_version` dynamically hashes the Python file contents to generate the fingerprint. If it instead reads the hardcoded `engine_impl_version: "2"` from `audio_engine_profiles.yaml`, you will also need to manually bump that string to `"3"` to actually demote the route. Verify how `live_engine_impl_version` is calculated.