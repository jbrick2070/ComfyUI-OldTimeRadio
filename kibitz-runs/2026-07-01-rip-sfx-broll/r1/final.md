# Rip out dead SFX-beat + scene_broll + background_abstract -- HARDENED PLAN (kibitz r1)

Panel: Codex + Antigravity (grounded) + Claude anchor. Claude Code lane ran slow; folded what
landed. Every claim below was verified against the real files.

## Verdict
Proceed -- but with a SHARPER scope than the draft. The draft conflated the dead `sfx`
speaker-role/beat with the LIVE `sfx_cue` ambient field, and picked a dangerous fallback.

## Codex's "keep sfx_cue" -- OVERRULED by grounding (operator trace 2026-07-01)
Codex flagged `sfx_cue` as live "SOUND IN THE ROOM" dialogue atmosphere and said KEEP it. The
code PATH exists (`_otr_line_composer.py:1414-1416` renders `SOUND IN THE ROOM: <cue>` into the
WRITER prompt, gated `if req.sfx_cue:`), but the DATA disproves "live":
- The voice engines NEVER ingest it: 0 references to `sfx_cue` in batch_character_voices / the
  voice resolvers / any engine incl. indextts2/bark/kokoro. It does not concatenate into TTS
  input and does not signal "in response to a sound." TTS only ever gets the final line text.
- It is an OPTIONAL outline field (`_otr_outline.py:120`, prompt `:555` "sfx_cue: str max 80 or
  null", constructed `None` at :1633/1651/1675/1715) that comes back NULL in every real episode
  (0 populated across 12 ledgers / 60 rows). So `if req.sfx_cue:` never fires -> the writer nudge
  never renders. Dead-in-practice, same bucket as sfx lines.
=> RIP `sfx_cue` OUT TOO. It is Optional/None everywhere (no required-field parse break). Remove:
the outline Beat field (:120) + the prompt line (:555) + the `None` constructors (:1633/1651/1675
/1715); the line-composer render (`_otr_line_composer.py:848` attr + `:1279`/`:1412-1416`); the
ledger carry (`production_ledger.py:806/852-860`, `OTR_LedgerScriptWriter.py:4362/4806-4811`,
`_otr_ledger_freeze.py:27-28`, `_otr_legacy_to_stage1_adapter.py:527`).

## SceneSequencer sfx overlay ALSO dead -- PROVEN (operator: rip the WHOLE subsystem)
Workflow JSON node 3 `OTR_SceneSequencer`: `sfx_audio_clips` link = **None** (UNWIRED),
`sfx_offset_ms` link = **None**, and NO node in the workflow produces sfx audio. So `sfx_clips`
is always empty and the overlay loop (`scene_sequencer.py:842-846`) never runs -- zero audio.
=> IN SCOPE: remove `scene_sequencer.py` `sfx_audio_clips`/`sfx_offset_ms` INPUT_TYPES (:625/:648)
+ params + extraction + overlay loop (:694/:698/:739/:742/:762-764/:842-846), and node 3's two
dead (unwired) inputs in the workflow JSON. Plus the per-cue sfx `dur_s`/`procsfx` validator +
`test_per_cue_sfx_dur.py`. NET: the ENTIRE sfx subsystem (speaker-role, cue, [SFX] token, video
route, audio overlay, duration validator) is retired -- it produces no script/audio/video.

## Final scope (what goes)
REMOVE the allowed-but-unused capability (Codex: the writer CONTRACT currently ALLOWS sfx, so
the contract is the first removal target):
1. `sfx` SPEAKER-role: `_otr_speaker_role.py` (`SPEAKER_ROLE_SFX`, `VALID_SPEAKER_ROLES`,
   `_NEVER_HUMO_ROLES`); the outline CONTRACT (`_otr_outline.py` `SpeakerRole` Literal + the
   prompt line that tells the model sfx is valid, ~:551); `OTR_LedgerScriptWriter.py`
   `NON_VOICED_ROLES` sfx member + the sfx non-voiced compose branch (keep the branch's use of
   `sfx_cue`? No -- that branch is for sfx BEATS; the sfx_cue AMBIENT field on voiced beats is a
   different code path in _otr_line_composer and stays); `production_ledger.py`
   `_NON_CHARACTER_CHAR_ID_SENTINELS` sfx + `[SFX: ...]` assembly (:1338); `otr_shot_lock.py`
   `SPEAKER_TO_VIDEO_ROLE["sfx"]`; `video_engine.py` HUD sfx branch (:1287).
2. `scene_broll` + `background_abstract` VIDEO-roles: the `Role` enum members
   (`_otr_shared/role_compat.py`) + `ROLE_AVAILABLE_INPUTS`; `role_slots.py`
   (`_OTHER_BEATS_ROLES`, `ROLE_TO_VIDEO_SLOT`, `VIDEO_SLOT_ROLES`, `NEW_ROUTE_A_VIDEO_SLOTS`);
   `slot_matrix.py`; `cheap_families.py` still_motion/still_pan/still_flat `roles` +
   still_motion `default_roles`; `otr_video_director.py` the two slot widgets + execute() params
   + per-role loop + role->aspect map; `otr_image_director.py` other_beats_image roles tuple;
   `_otr_workflow_apply.py` `_VIDEO_DIRECTOR_WIDGETS`.
3. The now-dead OTHER-BEATS POOLING (Antigravity, GROUNDED): with both pooled roles gone,
   `other_beats_clip_mode` / `other_beats_n` widgets (`otr_video_director.py:241-247`) + the
   pooling logic (`otr_shot_lock.py:364-387`, `otr_meta_brief_image_prompt.py`
   `_OTHER_BEATS_ROLES` + `derive_scene_still_targets` pool path :478-513) are dead -> remove.
4. CONFIG (Antigravity, GROUNDED -- the draft missed these): `config/profiles/widget_mapping.json`
   drop `role_overrides.scene_broll_visual` + `role_overrides.background_abstract_visual`; the
   committed `config/profiles/*.json` that set `other_beats_visual`/`other_beats_image` overrides
   -- rename to a character fallback or drop (they now only hit character).
5. WORKFLOW JSON (hard, same commit): node 87 `OTR_VideoDirector` widgets index 17
   (`scene_broll_video_model`) + 18 (`background_abstract_video_model`) -- the LAST two, so drop
   BOTH together (tail truncation, no positional drift to earlier widgets); re-validate
   (`OTR_WorkflowValidator` + round-trip + widget-count vs live INPUT_TYPES + link audit).

## Resolved decisions (were open questions)
- Q1 (procedural-SFX audio): DO NOT remove `sfx_cue`; leave SceneSequencer `sfx_audio_clips` /
  node-3 alone. CUT the P3 "audio subsystem" phase from the draft (Codex MUST-FIX #1).
- Q2 (`_DEFAULT_VIDEO_ROLE`): FAIL-LOUD. Both agents: defaulting an unknown speaker-role to
  `character_video` routes silence to HuMo -> BUG-LOCAL-129 crash. Make an unmapped speaker-role
  a LOUD `ValueError` at the writer/freeze boundary (VALID_SPEAKER_ROLES already gates input);
  remove the normal-path `_DEFAULT_VIDEO_ROLE` fallback from shot/still routing. (No
  default-to-character; no silent fallback.)
- background_abstract IS dead (subagent + both agents) -- not re-litigated; the fallback is the
  real thing to fix (done via fail-loud).

## Phase order (Codex: avoid dangling refs; atomic where coupled)
- P1 (ATOMIC role removal): delete `SPEAKER_TO_VIDEO_ROLE["sfx"]` FIRST, THEN the `Role`
  enum members + all role_compat/role_slots/slot_matrix/cheap_families/director consumers +
  the two director widgets + widget_mapping.json + config profiles + the workflow-JSON widgets
  (node 87) -- ONE commit (removing the enum before the mapping ref would leave a live ref to a
  deleted member). Fail-loud fallback in the same commit.
- P2: sfx SPEAKER-role contract: outline Literal + prompt, VALID_SPEAKER_ROLES / _NEVER_HUMO /
  NON_VOICED, production_ledger sentinel + `[SFX:]`, video_engine HUD branch.
- P3: dead OTHER-BEATS pooling (widgets + logic).
- P4: tests -- delete the TARGETED sfx/scene_broll/background_abstract test fns (incl. all of
  test_per_cue_sfx_dur.py IF it tests sfx BEATS not sfx_cue -- VERIFY first), update ENUM/count
  tests (test_speaker_role count 6->5), mechanical fixture updates; add a GUARD test (no sfx in
  VALID_SPEAKER_ROLES; no scene_broll/background_abstract in Role) so it can't creep back.
- Each phase: full suite + Bug Bible + B7 green; workflow re-validated; commit AND push.

## Invariants / verification (folded)
- KEEP sfx_cue (dialogue atmosphere) -- prove with a line-composer test that SOUND IN THE ROOM
  still renders from sfx_cue.
- Audio byte-identical: define as ledger->audio determinism for FIXED existing no-sfx ledgers,
  NOT fresh LLM generation after the prompt/schema change (Codex SHOULD-FIX #2).
- Two-sweep P4 grep (Codex): forbidden = `speaker_role=="sfx"` / `[SFX:]` / the removed video
  roles; ALLOWED = `sfx_cue` references (must remain). A blanket `"sfx"` grep is wrong.
- Old saved ledgers with `speaker_role:"sfx"`: reject LOUD at load (Codex optional -> adopt as a
  small guard).
- One coder window; UTF-8 no BOM; SFW.

## Residual verify-at-build
- Confirm test_per_cue_sfx_dur.py targets sfx BEATS (dur_s bounds on sfx lines), not sfx_cue --
  if it's sfx-beat duration, delete; if it touches sfx_cue, keep/adapt.
- Confirm no engine's `roles`/`default_roles` becomes EMPTY after dropping the two video roles
  (still_motion default_roles was `("scene_broll",)` -> must get a new default or the family is
  unreachable; pick announcer_visual/music_visual).
