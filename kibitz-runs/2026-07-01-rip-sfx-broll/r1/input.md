# Rip out the dead SFX + scene_broll + background_abstract paths -- PLAN (for kibitz)

## Why
The writer never emits `sfx` lines, so three roles are dormant "code that doesn't work
and confuses people": the `sfx` speaker-role, the `scene_broll` video-role (only reachable
via sfx), and the `background_abstract` video-role (only the unmapped-role fallback -- no
valid speaker-role ever lands on it). Ledgers from real renders show only
`character` / `announcer` / `music_*` lines. Remove the dead paths cleanly.

Grounded blast radius (3-subagent traversal of the real repo, 2026-07-01):
~16-18 production files (8 STRUCTURAL), ~61 test files / 358 refs (18+ targeted test fns),
workflow node 87 widgets 17-18.

## Scope decision (RECOMMENDED, panel please confirm)
Remove ALL THREE dead concepts in ONE campaign (they are coupled -- sfx is the only tie to
scene_broll; scene_broll+background_abstract are the "other beats" pair):
1. `sfx` speaker-role -- INCLUDING its dormant procedural-audio subsystem (sfx cues, per-cue
   `dur_s` bounds / `procsfx`, `[SFX: ...]` transcript token, `NON_VOICED_ROLES` sfx member,
   the writer's non-voiced sfx compose branch). It has never fired in a real episode.
2. `scene_broll` video-role (+ its `scene_broll_video_model` slot/widget).
3. `background_abstract` video-role (+ its `background_abstract_video_model` slot/widget).

OPEN QUESTION 1 (panel): is the procedural-SFX AUDIO subsystem truly dead too, or should we
remove only the ROLE plumbing and leave the audio-cue machinery (in case the writer is later
taught to emit sfx)? Recommendation: remove it -- it is unreachable and is the biggest source
of "confusing dead code" (11 whole test fns in test_per_cue_sfx_dur.py alone).

OPEN QUESTION 2 (panel): what becomes `_DEFAULT_VIDEO_ROLE` (currently `background_abstract`)?
An unmapped speaker-role must still resolve to SOMETHING. Recommendation: default to
`character_video` (the dominant real role) and/or make an unknown speaker-role a LOUD hard
error at the writer boundary (VALID_SPEAKER_ROLES already gates input), so the fallback is
never silently hit. Panel: pick fail-loud vs default-to-character.

## The STRUCTURAL touch points (grounded, file:line)
- `_otr_shared/role_compat.py`: the `Role` enum (`SCENE_BROLL`, `BACKGROUND_ABSTRACT` members
  :~40), `ROLE_AVAILABLE_INPUTS` entries. Core enum -- every role consumer imports it.
- `_otr_speaker_role.py`: `SPEAKER_ROLE_SFX` (:59), `VALID_SPEAKER_ROLES` (:63-70),
  `_NEVER_HUMO_ROLES` (:96-102, sfx member).
- `otr_shot_lock.py`: `SPEAKER_TO_VIDEO_ROLE["sfx"]` (:70), `_DEFAULT_VIDEO_ROLE` (:72).
- `_otr_shared/role_slots.py`: `_OTHER_BEATS_ROLES`, `ROLE_TO_VIDEO_SLOT`, `VIDEO_SLOT_ROLES`,
  `NEW_ROUTE_A_VIDEO_SLOTS` (scene_broll_video_model / background_abstract_video_model).
- `_otr_shared/slot_matrix.py`: `scene_broll`/`background_abstract` -> slot rows.
- `_otr_video_engines/cheap_families.py`: still_motion/still_pan/still_flat `roles` +
  still_motion `default_roles=("scene_broll",)`.
- `otr_video_director.py`: `scene_broll_video_model` + `background_abstract_video_model`
  INPUT_TYPES widgets (:291-303), execute() signature params (:328), the per-role loop (:349),
  role->aspect map (:409/:427).
- `_otr_workflow_apply.py`: `_VIDEO_DIRECTOR_WIDGETS` set (drop the two slots).
- `otr_image_director.py`: `other_beats_image_model` roles tuple (drop scene_broll), the 3D
  granularity filter ref.
- `otr_meta_brief_image_prompt.py`: `_OTHER_BEATS_ROLES` (:399) + `derive_scene_still_targets`
  pooling (becomes a no-op once both roles are gone -- delete the pool path).
- `OTR_LedgerScriptWriter.py`: `NON_VOICED_ROLES` (:147 sfx), sfx compose branch (:3843/4085/
  4362/4806/4810), `beat.sfx_cue`.
- `production_ledger.py`: `_NON_CHARACTER_CHAR_ID_SENTINELS` (:93 sfx), `[SFX: ...]` assembly
  (:1338).
- `_otr_outline.py`: `SpeakerRole` Literal (:69-76 sfx member).
- `video_engine.py`: HUD sfx branch (:1287) -- delete.
- (procedural-sfx audio: `procsfx` / per-cue dur_s validator -- locate + remove if Q1=yes.)

## The WORKFLOW-JSON change (hard, same commit as the code -- CLAUDE.md S0)
Node 87 `OTR_VideoDirector`: `widgets_values` index 17 = `scene_broll_video_model`, index 18 =
`background_abstract_video_model`, both `"(use Other Beats default)"`; index 18 is terminal.
`widgets_values` is POSITIONAL (BUG-LOCAL-097): removing a widget shifts every later value.
Since these two are the LAST two widgets (17,18 of a 0..18 list), dropping BOTH truncates the
tail cleanly with NO drift to earlier widgets -- do them TOGETHER, last-first, and re-validate
(`OTR_WorkflowValidator` + JSON round-trip + widget-count vs live INPUT_TYPES + link audit).
If only one were removed the other would shift -- so remove BOTH or NEITHER.

## Phased build (suite green + commit/push per phase)
- P1 -- video ROLES: drop `SCENE_BROLL` + `BACKGROUND_ABSTRACT` from the `Role` enum and every
  role_compat / role_slots / slot_matrix / cheap_families / director consumer; delete the two
  director widgets + the workflow-JSON widgets (SAME commit); fix `_DEFAULT_VIDEO_ROLE`
  (Q2). Update the video-role tests.
- P2 -- sfx SPEAKER-role: drop `SPEAKER_ROLE_SFX` from VALID_SPEAKER_ROLES / _NEVER_HUMO_ROLES /
  SPEAKER_TO_VIDEO_ROLE / _outline Literal / production_ledger sentinels + [SFX] token /
  NON_VOICED_ROLES; delete the writer sfx compose branch. Update the speaker-role/writer tests.
- P3 -- procedural-SFX AUDIO (if Q1=yes): delete sfx_cue / per-cue dur_s / procsfx machinery
  and its tests (test_per_cue_sfx_dur.py etc.).
- P4 -- sweep: grep the repo for any residual `scene_broll|background_abstract|"sfx"|SFX`;
  update remaining incidental test fixtures; add a GUARD test asserting the reduced role set
  (VALID_SPEAKER_ROLES has no sfx; Role enum has no scene_broll/background_abstract) so it
  can't creep back. Full suite + Bug Bible + B7 green; workflow re-validated.

## Invariants
- Audio byte-identical for episodes that never had sfx (all of them) -- `test_audio_byte_identical`
  green. The removal must not change a single real episode's output.
- Workflow JSON edited in the SAME change as the code (node 87), positional-drift-safe, re-validated.
- No silent fallback: an unknown speaker-role fails LOUD (or defaults to character -- Q2), never
  a dead role.
- One coder window; suite + Bug Bible + B7 green per phase; commit AND push per green phase.
- UTF-8 no BOM; SFW.

## Ask for the panel
1. Confirm background_abstract is dead and safe to remove (subagent verdict: yes -- nothing
   routes to it; engines list it but never default to it).
2. Q1: remove the procedural-SFX AUDIO subsystem too, or leave it dormant?
3. Q2: `_DEFAULT_VIDEO_ROLE` replacement -- fail-loud on unknown vs default-to-character?
4. The workflow-JSON positional-drift plan (remove BOTH trailing widgets together) -- sound?
5. Any load-bearing consumer the phased order would break mid-phase (e.g. an engine whose
   `default_roles` becomes empty, or role_compat filter that assumes >=1 other-beats role)?
