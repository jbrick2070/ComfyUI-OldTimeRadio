VERDICT: yes-with-fixes. Main wiring is plausible, but the workflow link surgery and profile/soak producers will break build gates as written.

MUST-FIX BEFORE BUILD:
1. [2.2] Defect: “links[] untouched” is wrong for node 3. In workflows/otr_scifi_16gb_full.json, OTR_SceneSequencer currently has sfx_audio_clips at input slot 2 and script_json at slot 3; link id 2 is [2,62,1,3,3,"STRING"]. Removing sfx_audio_clips shifts script_json to dst_slot 2, so leaving links[] untouched points link 2 at the wrong input. Concrete fix: after removing node 3 inputs sfx_audio_clips and sfx_offset_ms, rewrite link id 2 dst_slot from 3 to 2 and run the referential/link-slot audit.

2. [5,7] Defect: deleting widget_mapping entries for role_overrides.scene_broll_visual/background_abstract_visual will break generated profiles that still emit those keys. cross_validate_profile rejects unmapped overrides at nodes/_otr_shared/capability_profiles.py:332-335. Stale producers include scripts/_otr_combo_soak.py:67-89 and scripts/_otr_overnight_420_soak.py:149-151. Concrete fix: update every profile-producing soak/script to the 3-role model before deleting the mapping keys; add a grep gate for scene_broll_visual/background_abstract_visual outside historical docs.

3. [4,6,7] Defect: deleting SFX_DUR_MIN_S/SFX_DUR_MAX_S from nodes/_otr_ledger_freeze.py will break tests/test_fixture_dur_s_audit.py:20, not just tests/test_per_cue_sfx_dur.py. Concrete fix: delete or rewrite the fixture duration audit in the same commit, and grep for SFX_DUR_MIN_S/SFX_DUR_MAX_S imports before running the suite.

4. [5] Defect: removed pooling widgets still have a direct patch utility. scripts/_otr_patch_pool_default.py:26-27 patches other_beats_clip_mode and other_beats_n, which will hard-fail once OTR_VideoDirector INPUT_TYPES drops them. Concrete fix: delete the script or convert it to a loud tombstone that exits nonzero with “pooling removed”; do not leave a script that claims to patch the canonical workflow.

SHOULD-FIX:
1. [1.4,5] slot_matrix rename needs all harness consumers, not just the three named in the plan. scripts/_otr_combo_soak.py:87 still calls build_all_five_role_profile. Concrete fix: rename call sites to build_all_role_profile and make ALL_ROLES the only exported role tuple unless a tested compatibility alias is intentionally kept.

2. [5] scripts/otr_coverage_sweep.py:86-91 still defines only announcer/music/other_beats slots and maps other_beats_visual to character_video. After the rip, that hides the character_video-specific lane behind the legacy slot. Concrete fix: make the sweep explicit over announcer_visual, music_visual, character_visual, and keep other_beats_visual only as a migration-slot test if still required.

3. [7] Add a workflow audit that checks link dst_slot indexes against the post-edit inputs[] order, not only link id existence. The node 3 issue above passes simple referential integrity but fails slot semantics.

OPTIONAL / NICE-TO-HAVE:
- Update comments/docstrings mentioning “five roles” in registry/render-driver/image-engine docs during the same pass; stale comments are not runtime defects but will mislead future wiring work.

CUT THESE (over-engineering):
1. [5] Cut scripts/_otr_patch_pool_default.py. It exists only to force the pooling mode being removed, so preserving it creates a false integration surface.
2. [1.4] Cut compatibility aliases named FIVE_ROLES/build_all_five_role_profile unless a current script needs a transition period. The plan is an atomic rip, so aliases would keep the old mental model alive.
