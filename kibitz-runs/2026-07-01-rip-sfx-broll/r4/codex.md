VERDICT: yes-with-fixes — one guard-test assumption is false against the real registry defaults and would fail the build.

MUST-FIX BEFORE BUILD:
1. [§6] The new guard “default_engine_for_role(role) non-empty for all 3 roles (character -> humo)” is false today. `nodes/_otr_video_engines/eng_humo.py:104-105` has `roles = ("character_video",)` but `default_roles = ()`, and no other video engine declares `character_video` in `default_roles` [ASSUMPTION based on git grep]. Concrete fix: either promote the intended character default explicitly, e.g. set `HuMo14BLandscapeEngine.default_roles = ("character_video",)` if `humo_14B_169` is the real default, or change the guard to assert `engines_for_role("character_video")` is non-empty and that the canonical workflow pins `character_video_model` to `humo_14B_169`.

SHOULD-FIX:
1. [§1.4] “NO aliases / FIVE_ROLES -> ALL_ROLES” misses the stale function name `profile_keys_for_all_five()` in `nodes/_otr_shared/slot_matrix.py:70-72`. It appears unused, but leaving it contradicts the clean rename. Concrete fix: rename it to `profile_keys_for_all_roles()` or delete it if no callers remain.
2. [§6] Test fallout is under-specified for known non-video SFX fixtures. Add explicit coverage for `tests/fixtures/ledger_stub.py:19-119` and `tests/test_full_workflow_v2_audio_wiring.py:243-247`, because both pin `speaker_role="sfx"` / `sfx_audio_clips` behavior and are outside the named rewrite list.

OPTIONAL / NICE-TO-HAVE:
- [§4] Clean doc/comment-only `sfx` references in `nodes/_otr_audio_engines/registry.py:3-38` if the sweep becomes too noisy, but they are not build blockers.

CUT THESE:
1. [§6] Cut the `default_engine_for_role()` guard if the project does not actually want a registry-level character default. The canonical workflow already pins `character_video_model`; forcing a registry default is extra policy.

VERIFY-AT-BUILD checklist:
- [§7] Run the explicit post-edit workflow audit for node 3 link id 2 dst_slot `3 -> 2`, plus dst_slot-vs-input-name order, JSON round-trip, widget count vs live `INPUT_TYPES`, and link referential integrity.
- [§6/§7] Confirm the adjusted default-engine guard either passes after a real `default_roles` promotion or has been replaced by the canonical-workflow pin check.
- [§5/§7] Verify `scripts/otr_video_soak.py`, `scripts/otr_coverage_sweep.py`, and `scripts/run_otr_30word_smoke.py` no longer reference removed `scene_broll` / `background_abstract` roles.
- [§6/§7] Full suite must catch and retire remaining hard-coded 5-role / SFX fixtures without reintroducing fallbacks.
- [§7] Run Bug Bible and B7 sweep; then verify pushed `HEAD == origin/v2.0-alpha`, no 0-byte files, no BOM, and AST parse on touched `.py` files.
