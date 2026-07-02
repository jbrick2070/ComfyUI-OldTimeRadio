# Rip sfx + scene_broll/background_abstract -- CODER BUILD PLAN v2 (post-r3)

Contract: kibitz-runs/2026-07-01-rip-sfx-broll/r2/final.md (scope LOCKED).
r3 wiring pass done (anchor + codex + claude-code; agy credit-timeout, dropped):
all panel must-fixes grounded + folded -- see r3/judgment.md. ONE ATOMIC COMMIT.
All line numbers grounded against v2.0-alpha HEAD on 2026-07-01.

## 0. Decision locks (contract -- do not re-litigate)

- RIP whole sfx subsystem + scene_broll + background_abstract video roles + other-beats
  pooling (proven ZERO content). NO FALLBACKS: unmapped speaker-role / unknown video-role
  FAILS LOUD; old speaker_role:"sfx" ledgers rejected LOUD.
- ONE atomic commit (code + workflow JSON + config + scripts + tests), green-gated
  (full suite + Bug Bible + B7 + workflow re-validate) before push to v2.0-alpha.
- KEEP: other_beats_image_model (character image slot); legacy other_beats_video_model
  slot (migration INPUT, now character-only). NO renamed-symbol compat aliases (r3 codex).

## 1. Final role model (exact)

### 1.1 Speaker roles -- nodes/_otr_speaker_role.py
- VALID_SPEAKER_ROLES = ("character","announcer","music_open","music_close","music_inter").
- DELETE SPEAKER_ROLE_SFX (+ __all__). _NEVER_HUMO_ROLES = the 4 non-character roles.
- resolve_speaker_role: RAISES ValueError on non-Mapping/missing/non-string/unknown
  (incl. "sfx"). stamp_default_role: keeps TypeError on non-dict; RAISES ValueError on
  missing OR invalid role (note: raise-on-MISSING is deliberate -- no backfill survives).
  Caller audit DONE (driver grep): zero non-test callers of either fn; cite in commit msg
  + docstrings.

### 1.2 Video roles -- nodes/_otr_shared/role_compat.py (explicit change-map entry)
- Role enum = {ANNOUNCER_VISUAL, MUSIC_VISUAL, CHARACTER_VIDEO}; docstring "five"->"three".
- ROLE_AVAILABLE_INPUTS: DELETE the "scene_broll" + "background_abstract" entries (string
  keys materialized at import -- enum shrink alone does NOT remove them); the 3 survivors
  each frozenset({"text_prompt","init_image","audio_ref","base_clip_ref"}).
- ROLES auto-shrinks to 3. RoleCompatError on unknown role unchanged.

### 1.3 Slots -- nodes/_otr_shared/role_slots.py
- ROLE_TO_VIDEO_SLOT = {announcer_visual: announcer_video_model, music_visual:
  music_video_model, character_video: character_video_model}.
- LEGACY_OTHER_BEATS_SLOT kept; _OTHER_BEATS_ROLES = (character_video,).
- VIDEO_SLOT_ROLES = the 3 inverses + other_beats_video_model: (character_video,).
- PER_ROLE_VIDEO_SLOTS = 3-tuple; NEW_ROUTE_A_VIDEO_SLOTS = (character_video_model,).
- slot_for_role: unknown role RAISES ValueError (no silent legacy mapping).
- engine_id_for_role: role validated via slot_for_role FIRST (unknown raises).
  CLARIFICATION (r3): the kept "legacy fallback" is ONLY the empty-slot migration read
  (character_video_model unset in the policy -> read other_beats_video_model); it NEVER
  catches unknown roles. Do not soften the raise.

### 1.4 Profile-key matrix -- nodes/_otr_shared/slot_matrix.py (NO aliases)
- ROLE_TO_PROFILE_KEY = {announcer_visual: announcer_visual, music_visual: music_visual,
  character_video: character_visual}.
- RENAME FIVE_ROLES -> ALL_ROLES and build_all_five_role_profile -> build_all_role_profile
  outright (atomic rip, no compat aliases). Consumers fixed same commit:
  scripts/_otr_combo_soak.py:87, tests/test_slot_matrix_soak.py:68/:92-:104.
- build_all_role_profile also defensively pops "scene_broll_visual" +
  "background_abstract_visual" (alongside the existing other_beats_visual pop) so a
  user-customized profile carrying dead keys cannot reach the applier.
- IMAGE_KEYS unchanged.

### 1.5 Speaker->video map -- nodes/otr_shot_lock.py:55-83
- SPEAKER_TO_VIDEO_ROLE: drop the "sfx" entry; keep announcer/music*/character aliases.
- _DEFAULT_VIDEO_ROLE DELETED. _video_role_for_line: EXPLICIT membership check ->
  ValueError naming line_id + role (never dict.get with default).
- otr_meta_brief_image_prompt.py: the lazy import (:443-450) drops _DEFAULT_VIDEO_ROLE;
  the lookup (:483-485) becomes EXPLICIT: normalize key; `if key not in
  SPEAKER_TO_VIDEO_ROLE: raise ValueError(f"unmapped speaker_role {key!r} on line
  {line_id!r} ...")`; NEVER a bare .get() without default (that returns None silently --
  r3 claude MF4). The pooling block :478-512 deletes in the same hunk.

### 1.6 Aspects + config
- otr_video_director._role_aspects -> 3 keys. policy["aspects"] consumers use .get
  (verified tolerant).
- config/profiles/widget_mapping.json: DELETE role_overrides.scene_broll_visual +
  role_overrides.background_abstract_visual. Profile JSONs untouched (set neither).

## 2. Workflow JSON -- workflows/otr_scifi_16gb_full.json (SAME commit)

### 2.1 Node 87 OTR_VideoDirector
Replace widgets_values WHOLESALE with the 15-entry array (never sequential deletions --
BUG-LOCAL-097 discipline):
["viz_green","viz_green","viz_green","flux_gen1","flux_gen1","flux_gen1",
 25,832,480,"request_hash",42,false,"auto","{}","humo_14B_169"]
(drops old idx 6 other_beats_clip_mode, 7 other_beats_n, 17 scene_broll_video_model,
18 background_abstract_video_model; fps/canvas_w/canvas_h/seed_mode/request_seed/... land
2 earlier; re-audit every value BY NAME vs live INPUT_TYPES).
- inputs[]: remove the 4 matching entries (all link:null -- grounded). gate_in is slot 0
  (before every removal) -> NO node-87 link surgery.
- otr_video_director.py INPUT_TYPES loses the same 4 widgets, same commit.

### 2.2 Node 3 OTR_SceneSequencer (+ LINK SURGERY -- r3 codex MF1, grounded)
- widgets_values: ["[]",0,999,"","bark",0,0] -> ["[]",0,999,"","bark",0]
  (sfx_offset_ms is the tail value).
- inputs[]: remove sfx_audio_clips (slot 2, link:null) + sfx_offset_ms (slot 9, link:null).
- REWRITE link id 2: [2,62,1,3,3,"STRING"] -> dst_slot 3->2 (script_json shifts up after
  the slot-2 removal). Links 239/240 target slots 0/1 (before the removal) -- unchanged.
- POST-EDIT AUDIT (both nodes, r3 codex SF3): for EVERY link, dst_slot must index the
  input whose name matches the link's declared wiring -- link dst_slot vs post-edit
  inputs[] ORDER (slot semantics), plus OTR_WorkflowValidator + JSON round-trip +
  widget-count vs live INPUT_TYPES + referential id audit.

## 3. Loud-failure conversions (6 sites)

1. _otr_speaker_role.resolve_speaker_role -> ValueError (was silent ->character)
2. _otr_speaker_role.stamp_default_role -> ValueError (was silent overwrite; missing incl.)
3. role_slots.slot_for_role (+engine_id_for_role head) -> ValueError on unknown role
4. otr_shot_lock._video_role_for_line -> ValueError (explicit membership check)
5. otr_meta_brief_image_prompt derive_scene_still_targets lookup -> ValueError (explicit)
6. scene_sequencer dispatch else-branch (:809-810) -> ValueError (was silent ->dialogue)
Freeze gate: ledger_freeze.ALLOWED_SPEAKER_ROLES drops "sfx"; per-line invariant collects
the error and phase_10_gap_audit_post_and_freeze RAISES (grounded: Phase 0 collect /
Phase 10 raise). Old sfx ledgers now fail LOUD at freeze, writer, video, and audio paths.

## 4. sfx subsystem rip -- file-by-file

- nodes/_otr_outline.py: SpeakerRole Literal -> 5 (:69-76); Beat.sfx_cue DELETED
  (:120-124); prompt :551 role list, :555 sfx_cue line deleted, :550 "music/sfx" ->
  "music beats"; ctors drop sfx_cue=None (:1633,:1651,:1675,:1715); docstrings
  (:96,:155,:1751). _check_speaker_role_alignment auto-follows.
- nodes/OTR_LedgerScriptWriter.py: NON_VOICED_ROLES = 3 music roles (:147); compose_line
  call drops sfx_cue= (:4362); NON_VOICED branch (:4795-4813): last_lines.clear()
  unconditional (all members music), cleaned="", token="" -- empty tokens NOT appended
  (:4855 guard; slot-0 authority = assemble_script_text_from_ledger); self-test sample
  :5942 drops the [SFX:] line; docs :22,:145,:816,:4085.
- nodes/_otr_line_composer.py: LineRequest.sfx_cue deleted (:848, :835-836); SOUND IN THE
  ROOM block deleted (:1412-1416, :1279). :2912 scrub token list KEPT (LLM-output vocab).
- nodes/production_ledger.py: sentinels drop "sfx" (:93); init_lines drops the sfx_cue
  read (:806) -> non-spoken rows stamp text="" unconditionally (:849-860); assemble:
  music_*/sfx [SFX:] branch (:1338-1339) -> SKIP music_* rows (dead-by-construction:
  post-S1 those rows are text=="" and already `continue` at :1322); docstrings.
- nodes/_otr_ledger_freeze.py: ALLOWED_SPEAKER_ROLES -> 5 (:91-98); DELETE G7 call
  (:658-660) + body (:667-750) + SFX_DUR_MIN_S/MAX_S (:63-64,:682-683) + __all__ entries;
  trim :126-128 + G8 ProcSFX comments.
- nodes/_otr_ledger_consumers.py: _OPTIONAL_STRING_FIELDS drops the 4 sfx_* fields
  (:207-209,:216); DELETE ALLOWED_SFX_RENDER_STATUS (:250-259) + walker branch (:323-330)
  + __all__ (:355). Music parity fields + enum KEPT.
- nodes/scene_sequencer.py: INPUT_TYPES drops sfx_audio_clips (:625-628) + sfx_offset_ms
  (:648-652); sequence() drops both params (:694,:698); DELETE sfx clip extraction
  (:739,:742), sfx_timeline + offset (:752,:762-764), sfx dispatch branch (:805-806,
  :840-855), ducking/overlay (:970-988), sfx_line_positions + write-back (:791,:929-940,
  :1050-1062), log strings (:745-746,:1084-1087). Dispatch else -> RAISE. :455 text-scrub
  regex KEPT.
- nodes/video_engine.py: HUD sfx branches (:1287-1288,:1575-1578) + treatment
  (:1902-1903,:1917-1918) deleted; comments.
- nodes/_otr_video_engines/render_driver.py: "sfx" motion prompt (:561-563);
  _ltx_motion_role_key sfx branch (:589-590); :515 prefix guard keeps only "music";
  comments :826-833.
- nodes/_otr_radio_editor.py: NON_VOICED_ROLES -> 3 (:225-227); self-test "[SFX]" fixture
  text -> "[MUSIC]" (inert, hygiene).
- nodes/_otr_ledger_reviewer.py: _ALLOWED_SPEAKER_ROLES drops "sfx" (:859-863).
- nodes/_otr_story_brief.py: non_dialogue_roles -> {"music","env"} (:609); :349 KEPT.
- nodes/_otr_creative_qa.py: self-test fixture "sfx" row (:590) -> "music_inter".
- nodes/_otr_reroll.py: comments (:116-117,:292).
- nodes/_otr_render_plan.py: comment (:148-150).
- nodes/_otr_legacy_to_stage1_adapter.py: :527 is DOCSTRING-ONLY (grounded; no body read).
- nodes/_workflow_validation.py: ADD "sfx_audio_clips" + "sfx_offset_ms" to
  FORBIDDEN_INPUT_SOCKETS (sfx_plan_json tombstone stays).
- Comment-only: _otr_freeze_cascade.py:1176, _otr_news_wiring.py:29/:62,
  _otr_ledger_scrub.py:202/:212/:859, _otr_speaker_role.py header. _otr_ledger.py:55
  schema-history line KEPT (changelog).
- story_orchestrator.py [SFX:] text-token handling KEPT (legacy TEXT-script parser vocab).

## 5. scene_broll/background_abstract + pooling rip -- file-by-file

- nodes/otr_video_director.py: INPUT_TYPES drops other_beats_clip_mode/other_beats_n
  (:232-247) + scene_broll_video_model/background_abstract_video_model (:291-304);
  CLIP_MODES deleted (:144); direct() signature/map/loop lose both slots (:321-355);
  other_beats_n clamp deleted (:363-367); policy drops "other_beats" (:382);
  _role_aspects -> 3 keys (:405-429). USE_OTHER_BEATS + character_video_model KEPT.
- nodes/otr_image_director.py: IMAGE_SLOT_ROLES other_beats_image_model ->
  ("character_video",) (:58-64); three_d_locked_slots likewise (:146-154); policy
  passthrough "other_beats" deleted (:381-387).
- nodes/_otr_workflow_apply.py: _VIDEO_DIRECTOR_WIDGETS drops the 2 dead slots (:139-144).
- nodes/otr_shot_lock.py: compute_clip_budget -> {per_beat, total_frames, warnings}
  (:328-387); build_execution_plan drops _pool_N/_OTHER_BEATS/_ob_i/still_pool_key
  (:738-754,:774); shot "strategy" stamps the CONSTANT {"mode":"unique_per_beat"}
  (schema-stable); ledger video.clip_budget -> {"total_frames": ...} (:954-958); report
  lines (:965-969).
- nodes/_otr_video_engines/render_driver.py: DELETE the still_pool_key reads -- use _bid
  directly (:1015,:1046,:1130; the `or _bid` chain is the natural fallback but the dead
  key read goes, root cause); _PROFILES drops the scene_broll + background_abstract legs
  (:96,:99) -> 4 legs; ENGINE_FAMILY (:70-87) needs NO role edits (engine_id->family map,
  grounded); comments :1428-1429,:1505,:1920.
- nodes/otr_meta_brief_image_prompt.py: _OTHER_BEATS_ROLES deleted (:395-399);
  derive_scene_still_targets loses other_beats param + pooling branch (:402-513;
  character -> per-beat scene_character, announcer/music -> per-beat scene_beat, open
  unchanged); _other_beats_from_policy deleted (:309-320); execute() stops passing it
  (:1306); derive_image_prompts signature drops other_beats (:873).
- Engine role tuples: eng_ltx_video.py:274-275 -> ("music_visual","announcer_visual");
  eng_still_parallax.py:177-178 -> ("announcer_visual","music_visual","character_video");
  eng_viz_mandala.py:53-54 + eng_viz_rainbow.py:42-43 + still_pan/still_flat
  (cheap_families:211-212,:232-233) -> the 3-tuple; eng_wan_i2v.py:85 ->
  ("music_visual","character_video"); eng_wan_ti2v.py:103 roles=ROLES auto-shrinks;
  StillMotionFamily (cheap_families:180-181) roles -> 3-tuple, default_roles=()
  (universal floor; capability keeps it eligible everywhere incl. character OOM chain).
- SCRIPTS (r3 codex, all grounded):
  * scripts/_otr_combo_soak.py:87 -> build_all_role_profile / ALL_ROLES.
  * scripts/_otr_overnight_420_soak.py:149-151 -> 3-role model (drop the
    scene_broll_visual/background_abstract_visual writes).
  * scripts/otr_coverage_sweep.py:87-91 SLOTS -> explicit
    ("character_visual","character_video") lane (stop hiding character behind
    other_beats_visual).
  * DELETE scripts/_otr_patch_pool_default.py (patches removed widgets; no tombstone).
  * scripts/otr_video_soak.py + run_otr_30word_smoke.py: verify role enumerations at
    build (rename fallout).

## 6. Tests

- DELETE tests/test_per_cue_sfx_dur.py AND tests/test_fixture_dur_s_audit.py (both pin
  the G7 sfx dur_s contract -- r3 codex MF3).
- Rewrite/trim: test_speaker_role.py (new 5-set + raising semantics),
  test_route_a_14b_promotion.py, test_slot_matrix_soak.py (ALL_ROLES /
  build_all_role_profile), test_video_role_compat_additive.py, test_still_spine_helpers.py
  (pooling -> per-beat), test_phase1_composer_prompt.py:367,
  test_phase2b_progressive_ledger.py:135, test_s1_music_suppression.py:138,
  test_post_freeze_writeback_audit.py (sfx fields gone; music parity stays), plus suite
  fallout across the video tests (fix fixtures to the 3-role model; never re-add
  fallbacks).
- NEW tests/test_rip_sfx_broll_guard.py:
  (a) "sfx" not in VALID_SPEAKER_ROLES; len == 5.
  (b) Role enum values == {announcer_visual, music_visual, character_video} exactly;
      ROLE_AVAILABLE_INPUTS keys == the same 3.
  (c) resolve_speaker_role({"speaker_role":"sfx"}) raises ValueError.
  (d) ledger_freeze gap audit ERRORs a speaker_role:"sfx" line (old-ledger rejection).
  (e) _video_role_for_line raises on "sfx"; slot_for_role raises on "scene_broll" +
      "background_abstract".
  (f) other_beats_image_model STILL in OTR_VideoDirector.INPUT_TYPES()["required"]
      (kept-widget assertion, r3 claude SF10).
  (g) default_engine_for_role(role) non-empty for all 3 roles (pre-commit gate,
      r3 claude SF9).
- Audio regression: FIXED no-sfx ledger -> audio determinism (adapt any test that
  regenerates from the changed outline prompt).

## 7. Gates + wrap-up

- Workflow re-validation (both nodes): OTR_WorkflowValidator + JSON round-trip +
  widget-count vs live INPUT_TYPES + link referential audit + link dst_slot vs post-edit
  inputs[] ORDER (slot semantics).
- Full suite (Windows venv, PYTHONUTF8=1, pytest -q -p no:cacheprovider) + Bug Bible
  (survival-guide repo root, RELATIVE path) + B7 forbidden sweep -- ALL green BEFORE push.
- ONE commit -> push v2.0-alpha -> verify HEAD==origin, no 0-byte, no BOM, AST parse.
- docs/HANDOFF_LOG.md append + docs/GO_FORWARD_PLAN.md refresh (lean, forward-only).
