# CROSS-SPRINT COORDINATION -- token-cleanup sprint (branch v2.0-alpha)

Read this BEFORE planning your sprint. Another coder just finished a repo-wide naming/cleanup pass;
plan around these INVARIANTS so you don't reintroduce retired tokens or delete a live defense.

## What shipped (commits on v2.0-alpha, all green: full suite + Bug Bible)
- `40fd4473`, `e854f4e6`, `e5c495ba` -- unify the third image role on **`character`**
  (`character_image` role key, `character_granularity` widget). Superseded an interim `char_beats`.
- `c7198acd` -- neutralize retired **`scene_broll` / `background_abstract`** tokens repo-wide.
- `3eef8e1c` -- neutralize the isolated retired **`sfx` speaker-role** guards.
- `220cd4d5` -- retire the last SFX GENERATOR (removed `[SFX:]` from the dormant period writer prompt;
  added a pinning guard). Defensive SFX handling was intentionally KEPT.

## HARD INVARIANTS -- do not violate in your sprint
1. **Image role token = `character`.** Role key `character_image`; granularity widget
   `character_granularity` (INPUT_TYPES key + `otr_image_director.direct()` param + workflow node 88
   input[4] all agree). The MODEL widget is `character_image_model`. NEVER `other_beats*` or `char_beats*`.
2. **Video slot = `character_visual` / `character_video_model`.** The retired `other_beats_visual` /
   `other_beats_video_model` slots are GONE. Do not resurrect them.
3. **Retired tokens are ELIMINATED from tracked code** (case-insensitive grep = 0): `other_beats`,
   `scene_broll`, `background_abstract`. Tests that must reference "a retired/unknown role" use NEUTRAL
   placeholders: `retired_role_a` / `retired_role_b` / `retired_role_c`, `retired_video_slot`,
   `legacy_image_key`. Follow that convention; do not reintroduce the real dead names.
4. **Role/slot guards are POSITIVE closed-set assertions**, e.g.
   `assert set(RS.VIDEO_SLOT_ROLES) == {"announcer_video_model","music_video_model","character_video_model"}`
   -- not `assert "dead" not in X`. If you add role guards, use the positive closed-set form.
5. **SFX defenses MUST STAY** (they guard LLM hallucination + stale on-disk ledgers -- BUG-LOCAL-090/097):
   - the `[ENV|SFX|MUSIC:]` TTS strippers (`scene_sequencer.py:455`, `_otr_bark_lib.py:318`);
   - the cast-name SFX blocklist (`_otr_casting.py`) + editor `FORMAT_FAILURE` needle (`_otr_editor_constraints.py`);
   - the `speaker_role=="sfx"` rejection sites (`_otr_speaker_role.py`, `scene_sequencer.py`,
     `otr_meta_brief_image_prompt.py`, `otr_shot_lock.py`);
   - the forbidden-socket / filename tombstones (`_workflow_validation.py`: `sfx_plan_json`,
     `sfx_audio_clips`, `sfx_offset_ms`; `test_filename_pattern_audit.py`).
   No writer prompt may instruct `[SFX:]` (pinned by `test_period_prompts.py::test_system_prompt_never_solicits_sfx`).
6. **Workflow JSON contract** (`workflows/otr_scifi_16gb_full.json`): node 87 = OTR_VideoDirector (12
   widgets), node 88 = OTR_ImageDirector. Any node/widget change goes IN that JSON in the SAME commit
   as the code, then re-validate (`OTR_WorkflowValidator` + JSON round-trip + link/widget audit).

## Files this sprint touched (grep here first if you're near them)
- nodes: `_otr_shared/slot_matrix.py`, `_otr_shared/role_slots.py`, `otr_image_director.py`,
  `otr_image_gen_dispatcher.py`, `otr_video_director.py`, `_otr_workflow_apply.py`, `otr_shot_lock.py`,
  `_otr_period_prompts.py`, several `_otr_video_engines/*` (comments only).
- config: `profiles/16gb_full.json`, `profiles/8gb_lite.json`, `profiles/widget_mapping.json`.
- workflow: `workflows/otr_scifi_16gb_full.json` (node 88 granularity widget rename).
- tests: `test_slot_matrix_soak.py`, `test_rip_sfx_broll_guard.py`, `test_route_a_14b_promotion.py`,
  `test_video_platform_aseam.py`, `test_still_spine_helpers.py`, `test_image_platform_c1.py`,
  `test_period_prompts.py`, + ~30 more (role-token neutralization).
- docs: `docs/2026-07-04-other-beats-rename/`, `docs/2026-07-04-sfx-rip/`.

## Ground rules for merging with this branch
- Rebase on latest `v2.0-alpha` (HEAD `220cd4d5` or later) before you start.
- Run the FULL suite + Bug Bible after your change (this repo's standard); commit+push per green chunk.
- If your sprint NEEDS a retired token back (it shouldn't), stop and coordinate -- don't silently re-add it.
