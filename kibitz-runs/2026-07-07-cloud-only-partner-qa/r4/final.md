# R4 Driver Synthesis: Cloud-Only Partner QA

Verdict: ship after the CastLock/profile hardening that was applied in this pass.

Grounded findings:

1. Antigravity's `delivery_profile` widget-shift concern is not a defect in this tree. Baseline `HEAD:nodes/cast_lock.py` already had only `voice_bank`, `cast_voice_policy`, and `allow_voice_reuse` surfaced, and baseline `workflows/otr_scifi_16gb_full.json` node 80 had `["default","auto_registry",true]`. The new CastLock widgets are appended after those three saved values.
2. Antigravity's cloud-only visual-safety scoping recommendation was verified by the full suite. The initial global render-driver hook mutated byte-locked local LTX/HuMo prompt contracts, so the render-driver safety hook is now restricted to `cloud_*` engines. Cloud video adapters still append safety at the Partner boundary, and image prompt safety remains in the image prompt/dispatch path.
3. Claude's preserve-ledger voice-engine concern was real. Explicit `char_voice_engine` and `announcer_voice_engine` choices must be durable CastLock metadata even when `cast_voice_policy="preserve_ledger"`. This pass now stamps and validates those choices outside the recast-only path.
4. Claude's profile-mapping gap was real enough to close. `slot_overrides.cast_voice_policy` is now a managed profile key, and the checked-in profiles explicitly set `auto_registry`.

Regression added for the accepted finding:

- `tests/test_cloud_elevenlabs_cast.py::test_preserve_ledger_stamps_explicit_elevenlabs_voice_engines`
- `tests/test_workflow_apply.py::test_apply_cloud_all_lands_cloud_only_routes` now asserts `OTR_CastLock.cast_voice_policy == "auto_registry"`.

Focused verification after these edits:

`289 passed, 3 skipped, 2 xfailed` for the touched cloud/video/image/voice/workflow focused set.
