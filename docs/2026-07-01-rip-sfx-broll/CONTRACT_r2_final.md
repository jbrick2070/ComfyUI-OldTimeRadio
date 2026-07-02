# Rip the WHOLE sfx subsystem + scene_broll/background_abstract -- r2 CODING PLAN (converged)

Panel r2: Codex (grounded) + Claude anchor (+ Antigravity pending). CONVERGENCE: Codex agrees the
entire sfx subsystem is dead and should go -- its r2 "no" was only about the r1 plan still saying
BOTH rip AND keep `sfx_cue`. Resolved: FULL RIP. Below folds Codex r2's grounded sharpenings.

## Decision locks (settled)
- RIP the whole sfx subsystem: sfx speaker-role + sfx_cue + `[SFX:]` token + scene_broll (its
  video route) + the SceneSequencer sfx audio overlay (node 3, unwired) + per-cue sfx dur/procsfx.
  Proven to generate ZERO script/audio/video content (writer nudge never fires; TTS never sees it;
  sfx_audio_clips input link=None with no producer node; scene_broll never gets a beat).
- RIP scene_broll + background_abstract video-roles + the dead other-beats pooling.
- NO FALLBACKS: unmapped speaker-role / unknown video-role FAILS LOUD.

## r2 MUST-FIX (Codex, grounded -- fold into the build)
1. ONE ATOMIC CHANGE, not 4 phases (Codex CUT #1): the speaker-role, the video-role map, the
   director widgets, the workflow JSON, and the policy schema are one coupled data model.
   Splitting creates intermediate RED states (e.g. sfx still valid in _otr_speaker_role but
   unroutable in shot_lock). Build + land it as ONE buildable commit (still: suite+BugBible+B7
   green before push).
2. NO-FALLBACKS is MULTIPLE functions, not just `_DEFAULT_VIDEO_ROLE`. Enumerate + make loud:
   - `_otr_speaker_role.py:104-127` (silent unknown->map), `:174-185` (mutates invalid->character)
   - `_otr_shared/role_slots.py:91-98` (unknown video-role -> LEGACY_OTHER_BEATS_SLOT)
   - `otr_shot_lock.py:81-83` (`_DEFAULT_VIDEO_ROLE`) + `otr_meta_brief_image_prompt.py:483-485`
   Touched fns: `resolve_speaker_role`, `stamp_default_role`, `_video_role_for_line`,
   `slot_for_role`, `engine_id_for_role`, `derive_scene_still_targets` -> raise at a NAMED
   boundary (VALID_SPEAKER_ROLES already gates input; reject old `sfx` ledgers LOUD).
3. DEFINE the exact remaining role model after removal (don't just delete tokens): the 3-value
   `Role` enum, `ROLE_AVAILABLE_INPUTS`, `ROLE_TO_VIDEO_SLOT`, `VIDEO_SLOT_ROLES`, policy
   `aspects` keys (`otr_video_director.py:423-429`), profile keys.
4. WIDEN the engine-registry scope (Codex #5): role tuples also in
   `eng_ltx_video.py:274-275`, `eng_still_parallax.py:178`, `eng_viz_mandala.py:54`,
   `eng_viz_rainbow.py:43`, `eng_wan_i2v.py:85`, `render_driver.py:92-100 (ENGINE_FAMILY/_PROFILES)`,
   `cheap_families.py:180-181 (StillMotionFamily.default_roles=("scene_broll",) -> new default)`.
5. WORKFLOW JSON node 87 has TWO removals of DIFFERENT risk (GROUNDED, widgets_values shown):
   - TAIL (drift-safe): idx 17 `scene_broll_video_model`, idx 18 `background_abstract_video_model`.
   - MID-LIST (DRIFT): idx 6 `other_beats_clip_mode` ('unique_per_beat'), idx 7 `other_beats_n` (4)
     -- fps/width/height/seed/models are at 8..18 and SHIFT down by 2. Must drop the INPUT_TYPES
     widgets AND widgets_values indices 6,7 TOGETHER and re-audit every later widget BY NAME
     (widget-count vs live INPUT_TYPES + OTR_WorkflowValidator + round-trip + link audit).
6. Remove `policy["other_beats"]` from EVERY producer/consumer/report (Codex #7):
   `otr_video_director.py:382`, `otr_image_director.py:381-386`,
   `otr_meta_brief_image_prompt.py:309-317 & 1301-1307`, `otr_shot_lock.py:956-967` -- or keep the
   key as an explicit documented no-op. (Recommend remove everywhere in the same commit.)
7. CONFIG (Codex SHOULD-FIX #2): `other_beats_image_model` is the dispatcher default for
   non-announcer/music image roles (`otr_image_gen_dispatcher.py:151-173`). DECIDE: keep it as the
   character/other image slot (recommended -- character stills still need an image model) and just
   drop the scene_broll/background_abstract role_overrides from `widget_mapping.json` + profiles;
   do NOT delete other_beats_image itself.

## r2 test guidance
- sfx_cue tests to update/delete (Codex): `test_phase1_composer_prompt.py:367`,
  `test_phase2b_progressive_ledger.py:135`, `test_s1_music_suppression.py:138`, plus
  `test_per_cue_sfx_dur.py` (whole file -- sfx-BEAT duration).
- Broad role tests: `test_route_a_14b_promotion.py`, `test_slot_matrix_soak.py:45-54`,
  `test_video_role_compat_additive.py:48-69`, `test_still_spine_helpers.py:330-338`,
  `test_speaker_role.py` (count 6->5? no -- sfx removed AND role model changes: assert the new set).
- Add a GUARD test: no `sfx` in VALID_SPEAKER_ROLES; Role enum == {announcer_visual, music_visual,
  character_video}; an old `speaker_role:"sfx"` ledger is rejected LOUD.
- Audio regression (Codex CUT #2): NOT "byte-identical fresh gen" (the outline prompt/schema
  changes) -- use a FIXED no-sfx ledger -> audio determinism regression instead.

## Coder-window handoff
This is a plan-window artifact. Package for the separate coder window as ONE atomic commit with
the above must-fixes, green-gated (full suite + Bug Bible + B7 + workflow re-validate) before push.
