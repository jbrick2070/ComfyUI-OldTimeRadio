# Rip sfx + scene_broll/background_abstract -- CODER BUILD PLAN (r3 input)

Contract: kibitz-runs/2026-07-01-rip-sfx-broll/r2/final.md (r1+r2 converged; scope LOCKED).
This doc is the coder-window build plan: exact final role model, exact workflow-JSON widget
rewrite, the loud-failure list, and the file-by-file change map. ONE ATOMIC COMMIT.
All line numbers grounded against v2.0-alpha HEAD on 2026-07-01.

## 0. Decision locks (from the contract -- do not re-litigate)

- RIP the whole sfx subsystem + scene_broll + background_abstract video roles + other-beats
  pooling. Proven ZERO script/audio/video content.
- NO FALLBACKS: unmapped speaker-role / unknown video-role FAILS LOUD. Old ledgers carrying
  speaker_role:"sfx" are rejected LOUD.
- ONE atomic commit (code + workflow JSON + config + tests together), green-gated
  (full suite + Bug Bible + B7 + workflow re-validate) before push to v2.0-alpha.
- KEEP: other_beats_image_model (character stills still need an image slot), the legacy
  other_beats_video_model slot (migration input, now character-only).

## 1. Final role model (exact)

### 1.1 Speaker roles -- nodes/_otr_speaker_role.py
- VALID_SPEAKER_ROLES = ("character", "announcer", "music_open", "music_close", "music_inter")  # 5
- DELETE SPEAKER_ROLE_SFX constant (+ __all__ entry).
- _NEVER_HUMO_ROLES = {announcer, music_open, music_close, music_inter}.
- resolve_speaker_role(line): RAISES ValueError on non-Mapping input, missing field,
  non-string, or unknown value (incl. "sfx" -> the old-ledger rejection). Keeps
  strip/lower normalization. (Grounded: zero production callers; tests only.)
- stamp_default_role(line): keeps TypeError on non-dict; RAISES ValueError on a
  missing/invalid speaker_role instead of silently overwriting with "character".
- is_never_humo_role / is_music_role / is_dialogue_role / is_radio_role survive (sfx
  membership removed).

### 1.2 Video roles -- nodes/_otr_shared/role_compat.py
- Role enum = {ANNOUNCER_VISUAL="announcer_visual", MUSIC_VISUAL="music_visual",
  CHARACTER_VIDEO="character_video"}  # exactly 3
- ROLE_AVAILABLE_INPUTS: the 3 keys, each frozenset({"text_prompt","init_image",
  "audio_ref","base_clip_ref"}).
- ROLES = 3-tuple. RoleCompatError behavior unchanged (unknown role raises).

### 1.3 Slots -- nodes/_otr_shared/role_slots.py
- ROLE_TO_VIDEO_SLOT = {announcer_visual: "announcer_video_model",
  music_visual: "music_video_model", character_video: "character_video_model"}.
- LEGACY_OTHER_BEATS_SLOT = "other_beats_video_model" KEPT (migration INPUT).
- _OTHER_BEATS_ROLES = (Role.CHARACTER_VIDEO.value,)   # the only survivor of the 3
- VIDEO_SLOT_ROLES = {announcer_video_model: (announcer_visual,), music_video_model:
  (music_visual,), character_video_model: (character_video,),
  other_beats_video_model: (character_video,)}.
- PER_ROLE_VIDEO_SLOTS = (announcer_video_model, music_video_model, character_video_model).
- NEW_ROUTE_A_VIDEO_SLOTS = (character_video_model,).
- slot_for_role(role): unknown role RAISES ValueError (was .get(role, LEGACY) silent).
- engine_id_for_role(vm, role): validates role via slot_for_role (raises on unknown);
  the character_video -> legacy other_beats slot fallback KEPT (documented Route-A
  migration lane for the kept slot, not a dead role).

### 1.4 Profile-key matrix -- nodes/_otr_shared/slot_matrix.py
- ROLE_TO_PROFILE_KEY = {announcer_visual: announcer_visual, music_visual: music_visual,
  character_video: character_visual}  # 3 entries
- FIVE_ROLES renamed ALL_ROLES (3-tuple); fix consumers (scripts/otr_video_soak.py,
  tests/test_slot_matrix_soak.py, tests/test_video_soak_fixture.py -- verify at build).
- IMAGE_KEYS unchanged ("announcer_image","music_image","other_beats_image").
- build_all_five_role_profile renamed build_all_role_profile (same semantics over 3).

### 1.5 Speaker->video map -- nodes/otr_shot_lock.py:55-83
- SPEAKER_TO_VIDEO_ROLE = {announcer: ANNOUNCER_VISUAL, music: MUSIC_VISUAL,
  music_open/close/inter: MUSIC_VISUAL, character/char_voice/dialogue: CHARACTER_VIDEO}.
  ("sfx" entry deleted.)
- _DEFAULT_VIDEO_ROLE DELETED. _video_role_for_line: unmapped role RAISES ValueError
  naming the line_id + role (this rejects old sfx ledgers in the video path).
- otr_meta_brief_image_prompt.py:443-450 stops importing _DEFAULT_VIDEO_ROLE; its role
  lookup (483-485) raises on unmapped via the same map.

### 1.6 Aspects + policy + config keys
- otr_video_director._role_aspects -> 3 keys (announcer_visual/music_visual/character_video).
- policy["aspects"] consumers tolerate missing keys (dict .get) -- verify at build.
- config/profiles/widget_mapping.json: DELETE role_overrides.scene_broll_visual +
  role_overrides.background_abstract_visual. All other entries KEPT. The three profile
  JSONs (16gb_full/8gb_lite/cpu_floor) set NEITHER key (grounded) -> untouched.

## 2. Workflow JSON -- workflows/otr_scifi_16gb_full.json (SAME commit)

### 2.1 Node 87 OTR_VideoDirector (grounded current: 19 widgets_values)
Current:
["viz_green","viz_green","viz_green","flux_gen1","flux_gen1","flux_gen1",
 "unique_per_beat",4,25,832,480,"request_hash",42,false,"auto","{}",
 "humo_14B_169","(use Other Beats default)","(use Other Beats default)"]
New (15):
["viz_green","viz_green","viz_green","flux_gen1","flux_gen1","flux_gen1",
 25,832,480,"request_hash",42,false,"auto","{}","humo_14B_169"]
- Drop idx 6 other_beats_clip_mode + idx 7 other_beats_n (MID-LIST: fps/canvas/seed/
  models at 8..18 shift down 2 -- re-audit every later value BY NAME after the edit).
- Drop idx 17 scene_broll_video_model + idx 18 background_abstract_video_model (TAIL).
- inputs[]: remove the 4 matching converted-widget entries (other_beats_clip_mode,
  other_beats_n, scene_broll_video_model, background_abstract_video_model) -- ALL have
  link:null (grounded), so no link surgery; links[] untouched.
- INPUT_TYPES in otr_video_director.py loses the same 4 widgets in the same commit.

### 2.2 Node 3 OTR_SceneSequencer (forced by INPUT_TYPES change; grounded)
Current widgets_values: ["[]", 0, 999, "", "bark", 0, 0]
New (6):                ["[]", 0, 999, "", "bark", 0]
- sfx_offset_ms is the TAIL widget value -- drift-safe drop.
- inputs[]: remove sfx_audio_clips (link:null -- the contract-proven unwired overlay
  input) + sfx_offset_ms (link:null). No other node consumes/produces them.
- Re-validate after BOTH node edits: OTR_WorkflowValidator + JSON round-trip +
  widget-count vs live INPUT_TYPES (every node type) + link referential integrity.

## 3. Loud-failure conversions (6 sites)

1. _otr_speaker_role.resolve_speaker_role:116-138  silent ->character  => ValueError
2. _otr_speaker_role.stamp_default_role:201-219    silent overwrite    => ValueError
3. _otr_shared/role_slots.slot_for_role:91-98 (+engine_id_for_role)    => ValueError on unknown role
4. otr_shot_lock._video_role_for_line:81-83        ->background_abstract => ValueError
5. otr_meta_brief_image_prompt.derive_scene_still_targets:483-485      => ValueError (same map)
6. scene_sequencer role dispatch :809-810 else->dialogue               => ValueError
   (kills the silent legacy default; combined with 1 + the freeze gate this rejects an
   old sfx ledger LOUD in every path: writer/video/audio/validation.)
Freeze gate: _otr_ledger_freeze.ALLOWED_SPEAKER_ROLES drops "sfx" -> per-line invariant
(:307-315) reports any sfx row as a hard ERROR (existing loud machinery).

## 4. sfx subsystem rip -- file-by-file

- nodes/_otr_outline.py: SpeakerRole Literal -> 5 (:69-76); Beat.sfx_cue field DELETED
  (:120-124); system prompt: role list line :551 loses sfx, :555 sfx_cue line deleted,
  :550 "NARRATOR for music/sfx" -> "NARRATOR for music beats"; combiner ctors drop
  sfx_cue=None (:1633,:1651,:1675,:1715); docstrings (:96,:155,:1751).
  _check_speaker_role_alignment auto-follows VALID_SPEAKER_ROLES.
- nodes/OTR_LedgerScriptWriter.py: NON_VOICED_ROLES = {music_open,music_close,music_inter}
  (:147); compose_line call drops sfx_cue= (:4362); NON_VOICED branch (:4795-4813):
  the role-membership test for last_lines.clear() simplifies (all members are music now),
  cleaned = "" (no sfx_cue), token = "" and empty tokens are NOT appended to
  script_text_parts (:4855 guard) -- slot-0 authority is assemble_script_text_from_ledger
  anyway; module self-test sample (:5942) drops the [SFX: ...] line; doc comments
  (:22,:145,:816,:4085).
- nodes/_otr_line_composer.py: LineRequest.sfx_cue field deleted (:848 + :835-836 doc);
  SOUND IN THE ROOM block deleted (:1412-1416 + :1279 doc); banned-token list :2912
  KEPT (it scrubs LLM output text, vocab-agnostic).
- nodes/production_ledger.py: NON_CAST_CHAR_ID sentinels drop "sfx" (:93);
  init_lines_from_outline drops the sfx_cue read (:806) -- non-spoken rows stamp text=""
  unconditionally (:849-860 simplified); assemble_script_text_from_ledger: the
  music_*/sfx [SFX:] branch (:1338-1339) becomes a SKIP for music_* roles (render
  contracts carry no transcript text; rows are text=="" post-S1 so the branch is
  dead-by-construction -- deleting the [SFX:] token entirely); docstrings.
- nodes/_otr_ledger_freeze.py: ALLOWED_SPEAKER_ROLES -> 5 (:91-98); G7
  _check_g7_sfx_dur_invariant DELETED (:658-660 call + :667-750 body) with
  SFX_DUR_MIN_S/SFX_DUR_MAX_S + their __all__ entries (:63-64); comments :126-128, G8
  docstring ProcSFX mentions trimmed.
- nodes/_otr_ledger_consumers.py: _OPTIONAL_STRING_FIELDS drops sfx_wav_path/sfx_engine/
  sfx_type/sfx_render_status (:207-209,:216); ALLOWED_SFX_RENDER_STATUS deleted
  (:250-259) + the walker enum branch (:323-330) + __all__ (:355). Music parity fields
  + ALLOWED_MUSIC_RENDER_STATUS KEPT. (The producing AudioGen/ProcSFX nodes were deleted
  long ago; fields are sfx-only. Contract: KEEP nothing sfx.)
- nodes/scene_sequencer.py: INPUT_TYPES drops sfx_audio_clips (:625-628) +
  sfx_offset_ms (:648-652); sequence() signature drops both params (:694,:698);
  DELETE sfx_clips/sfx_clip_idx extraction (:739,:742), sfx_timeline +
  sfx_offset_samples (:752,:762-764), the sfx dispatch branch (:805-806,:840-855),
  the SFX ducking/overlay block (:970-988), sfx_line_positions + its write-back loop
  (:791,:929-940,:1050-1062) + log lines (:745-746,:1084-1087). Dispatch else -> RAISE
  (section 3.6). Text-scrub regex :455 KEPT (cleans ENV/SFX/MUSIC tokens out of TTS
  text -- defensive, vocab-agnostic).
- nodes/video_engine.py: HUD sfx item branch (:1287-1288), [SFX] renderer (:1575-1578),
  treatment sfx branches (:1902-1903 s_count, :1917-1918) deleted; comments.
- nodes/_otr_video_engines/render_driver.py: "sfx" motion prompt entry (:561-563);
  _ltx_motion_role_key sfx branch (:589-590); the wav-prefix guard :515 drops "sfx"
  (keeps "music"); comments :826-833.
- nodes/_otr_radio_editor.py: NON_VOICED_ROLES -> 3 music roles (:225-227); module
  self-test fixture text strings "[SFX] ..." on music rows renamed "[MUSIC] ..."
  (inert text, renamed for vocabulary hygiene).
- nodes/_otr_ledger_reviewer.py: _ALLOWED_SPEAKER_ROLES drops "sfx" (:859-863).
- nodes/_otr_story_brief.py: non_dialogue_roles -> {"music","env"} (:609). The :349
  stop-word list KEPT (text vocabulary, not a role).
- nodes/_otr_creative_qa.py: module self-test fixture row speaker_role "sfx" (:590)
  -> "music_inter" (same non-voiced semantics).
- nodes/_otr_reroll.py: comments/docstrings (:116-117,:292).
- nodes/_otr_render_plan.py: comment (:148-150).
- nodes/_otr_legacy_to_stage1_adapter.py: sfx_cue mention (:527) -- verify at build
  whether the body reads beat.sfx_cue; if so, drop the read.
- nodes/_workflow_validation.py: ADD "sfx_audio_clips" + "sfx_offset_ms" to
  FORBIDDEN_INPUT_SOCKETS (tombstones; existing sfx_plan_json tombstone stays).
- Comment-only touches: _otr_freeze_cascade.py:1176, _otr_news_wiring.py:29/:62,
  _otr_ledger_scrub.py:202/:212/:859, _otr_speaker_role.py header, _otr_ledger.py:55
  (schema history NOTE: keep historical changelog lines).
- story_orchestrator.py [SFX:] regex/token handling KEPT: legacy TEXT-script parser
  vocabulary (also covers ENV/MUSIC), outside the ledger sfx subsystem.

## 5. scene_broll/background_abstract + pooling rip -- file-by-file

- nodes/otr_video_director.py: INPUT_TYPES drops other_beats_clip_mode/other_beats_n
  (:232-247) + scene_broll_video_model/background_abstract_video_model (:291-304);
  CLIP_MODES deleted (:144); direct() signature + video_models map + the per-role loop
  lose both slots (:321-355); the other_beats_n clamp block deleted (:363-367);
  policy drops "other_beats" (:382); _role_aspects -> 3 keys (:405-429);
  USE_OTHER_BEATS sentinel + character_video_model widget KEPT.
- nodes/otr_image_director.py: IMAGE_SLOT_ROLES other_beats_image_model ->
  ("character_video",) (:58-64); three_d_locked_slots img_slot_roles likewise
  (:146-154); policy passthrough "other_beats" deleted (:381-387).
- nodes/_otr_workflow_apply.py: _VIDEO_DIRECTOR_WIDGETS drops scene_broll_video_model +
  background_abstract_video_model (:139-144).
- nodes/otr_shot_lock.py: compute_clip_budget -> {per_beat, total_frames, warnings}
  (:328-387; other/clip_mode/pool_n/render_count deleted); build_execution_plan drops
  _pool_N/_OTHER_BEATS/_ob_i/still_pool_key stamping (:738-754,:774); shot strategy
  mode -> fixed "unique_per_beat"?  NO: strategy key becomes {"mode":"per_beat"}?  --
  DECISION: keep the shot "strategy" field but stamp the constant "unique_per_beat"
  (schema-stable for downstream readers; verify render_driver reads of
  strategy/still_pool_key at build and delete the still_pool_key consumer);
  ledger video.clip_budget -> {"total_frames": ...} (:954-958); report lines (:965-969).
- nodes/otr_meta_brief_image_prompt.py: _OTHER_BEATS_ROLES deleted (:395-399);
  derive_scene_still_targets loses other_beats param + the pooling branch
  (:402-513 simplified: character -> per-beat scene_character, announcer/music ->
  per-beat scene_beat, open unchanged); _other_beats_from_policy deleted (:309-320);
  execute() stops passing other_beats (:1306); derive_image_prompts signature drops
  other_beats (:873).
- Engine role tuples: eng_ltx_video.py:274-275 -> ("music_visual","announcer_visual");
  eng_still_parallax.py:177-178 -> ("announcer_visual","music_visual","character_video");
  eng_viz_mandala.py:53-54 + eng_viz_rainbow.py:42-43 + still_pan/still_flat
  (cheap_families :211-212,:232-233) -> the 3-tuple; eng_wan_i2v.py:85 ->
  ("music_visual","character_video"); eng_wan_ti2v.py:103 roles=ROLES auto-shrinks;
  StillMotionFamily (cheap_families:180-181): roles -> the 3-tuple, default_roles=()
  (universal floor; no role auto-defaults to it now -- capability keeps it eligible
  everywhere incl. the character OOM chain). Verify default_engine_for_role consumers
  at build (no role may resolve to an empty default).
- nodes/_otr_video_engines/render_driver.py: _PROFILES drops the scene_broll +
  background_abstract legs (:96,:99) -> 4 legs; comments (:826-833,:1428-1429,:1505,
  :1920); ENGINE_FAMILY map: verify at build (grounded :85-87 tail shows engine->family
  map -- drop any scene_broll/background_abstract-only rows if present).
- scripts/otr_video_soak.py + scripts/otr_coverage_sweep.py +
  scripts/run_otr_30word_smoke.py: update role enumerations (FIVE_ROLES->ALL_ROLES,
  role lists) -- verify exact usage at build.

## 6. Tests

- DELETE tests/test_per_cue_sfx_dur.py.
- Rewrite/trim: test_speaker_role.py (assert the NEW 5-set + raising semantics),
  test_route_a_14b_promotion.py, test_slot_matrix_soak.py,
  test_video_role_compat_additive.py, test_still_spine_helpers.py (pooling cases ->
  per-beat), test_phase1_composer_prompt.py:367 (SOUND IN THE ROOM case),
  test_phase2b_progressive_ledger.py:135, test_s1_music_suppression.py:138,
  test_post_freeze_writeback_audit.py (sfx fields gone; music parity stays), plus
  whatever the suite surfaces (fixtures across ~20 video tests use the dead roles --
  fix each to the 3-role model, never by re-adding fallbacks).
- NEW tests/test_rip_sfx_broll_guard.py:
  (a) "sfx" not in VALID_SPEAKER_ROLES and len==5;
  (b) Role enum values == exactly {announcer_visual, music_visual, character_video};
  (c) resolve_speaker_role({"speaker_role":"sfx"}) raises ValueError;
  (d) ledger_freeze gap audit ERRORs on a line with speaker_role "sfx";
  (e) _video_role_for_line raises on "sfx"; slot_for_role raises on "scene_broll" /
      "background_abstract";
  (f) node-87 widgets_values length/name audit vs live INPUT_TYPES (or covered by the
      existing workflow-validator test -- verify at build).
- Audio regression: FIXED no-sfx ledger -> audio determinism (existing fixture-ledger
  determinism tests should already cover; adapt if any regenerate from the outline
  prompt, which changed).

## 7. Gates + wrap-up

- OTR_WorkflowValidator + JSON round-trip + widget-count-vs-INPUT_TYPES + link audit.
- Full suite (Windows venv, PYTHONUTF8=1, pytest -q -p no:cacheprovider) + Bug Bible
  (survival-guide repo root, relative path) + B7 forbidden sweep -- ALL green BEFORE push.
- ONE commit -> push v2.0-alpha -> verify HEAD==origin, no 0-byte, no BOM, AST parse.
- docs/HANDOFF_LOG.md append + docs/GO_FORWARD_PLAN.md refresh (lean, forward-only).
