# LTX audio-in consolidation + ROLE-DRIVEN still/audio/prompt routing -- BUILD PLAN (pass01)

Synthesis of R1 (GPT-5.5 + Gemini-3.1-pro + DeepSeek-v4-pro, grounded). The R1
panel converged, independently, on a defect pass00 missed: `ltx_audio_in` is ONE
engine serving BOTH character-face beats AND scene/bookend beats, so a routing
rule keyed on the engine's FAMILY (`audio_conditioned_video`) mis-routes character
beats three ways. The fix is to route on the BEAT ROLE, not the engine family.
That is the genuinely robust, model-agnostic generalization the operator asked for.

## Goal / end state (clarified)

ONE LTX audio-in engine (`ltx_audio_in`): regular `ltx_video` = no audio;
`ltx_audio_in` = audio-in (I2V on WHATEVER still the DRIVER hands it + the shot
audio). The engine is still-AGNOSTIC. The render_driver decides WHICH still and
WHICH audio per beat, by ROLE:
- CHARACTER-FACE beat -> the character PORTRAIT + the character's OWN clean audio +
  the character prompt (gear-scrub / M4 / char-fallback). [old ltx_av_talk behavior]
- SCENE / BOOKEND beat (announcer / music / scene_broll / background) -> the WIDE
  SCENE still + the ambient master slice (when no per-line timing) + the scene/brief
  prompt. [old ltx_av_music + ltx_video behavior]
DELETE `ltx_av_talk` and `ltx_av_music`. The talk-vs-music split was never about
the engine -- it encoded these two ROLE routings; move the routing to the role.

## Invariants (a fix that breaks one is rejected)

1. `test_audio_byte_identical` green; frozen master-audio spine untouched; video
   lane always silent.
2. CLAUDE.md S0: canonical workflow JSON change ships in the SAME commit; re-run
   OTR_WorkflowValidator + round-trip + widget/link audit; widgets_values positional.
3. NO FALLBACKS: failed render fails LOUD. Remove the legacy SYNTH_FALLBACKS
   entries; add NO ltx_audio_in fallback entry.
4. 14.5 GB NVML ceiling preserved for the heavy LTX-AV lane (incl. the 512x288 AV
   canvas clamp -- must follow ltx_audio_in after the rename).
5. CHARACTER-FACE beats keep the 832x1216 PORTRAIT + clean own-voice audio + char
   prompt. SCENE/BOOKEND beats get the 16:9 scene still; the portrait is CLEARED on
   wide beats so it can never leak into a wide frame (BUG-LOCAL-403 / 2026-06-20).
6. Mesh-fodder path unchanged (`requires_mesh_fodder` -> fodder, not scene still).
7. `station_card` stays EXCLUDED from scene-still conditioning (it inherits
   accepts_still=True from the base but is a title card -- it must NOT pull a scene
   still). The capability gate must not re-include it.
8. UTF-8 no BOM, ASCII, SFW. Regression suite + Bug Bible green; commit+push
   v2.0-alpha same session; HEAD==origin.

## Grounded current state (verified)

The render_driver routes THREE axes, today all family/name-keyed:
- STILL: family branch `_SCENE_INIT_FAMILIES={image_to_video,static_motion}`
  (479/842, guarded `not _requires_fodder`); name branch flux_still/flat_still
  (869, wide-only, clears portrait, LOUD-missing); name branch ltx_video +
  OTR_ENABLE_LTX_I2V (906, optional, _i2v_still_missing stamp); audio_driven_face
  keeps the portrait (not in any scene branch). `ltx_audio_in`
  (audio_conditioned_video, accepts_still=True) matches NONE -> init_image='' ->
  render_clip raises (it requires init_image) -> THE BUG.
- AUDIO: `_uses_ambient_master_audio(engine_id,family)` (730) returns True for
  `family=="audio_conditioned_video" or engine_id=="visualizer"` -- FAMILY-keyed,
  so ltx_audio_in character beats would get the ambient master slice (wrong voice).
- PROMPT: scene-prompt branch (1163) `in ("ltx_video","wan_i2v","ltx_av_music")
  and not text_prompt`, gated `_is_open` (announcer/music) -- adding ltx_audio_in
  naively would scene-prompt character beats.

Verified capabilities: visualizer accepts_still=False (eng_visualizer:49);
flat_still accepts_still=True (cheap_families:235); base MotionEngineBase
accepts_still=True default (motion_common:447) -> station_card inherits True (so an
accepts_still-only gate WOULD wrongly include it). VALIDATED_ENGINES
(registry:289) lists ltx_video + ltx_av_music + ltx_av_talk + ... ;
validated_engine_names() intersects it with the live registry for the per-role
dropdown (test_tested_only_dropdown_gate). 16gb_full.json pins announcer_visual +
music_visual = ltx_av_music. node-87 widgets (workflow) = ltx_av_music.

## Part A -- collapse to one engine

DELETE `LtxAvTalkEngine` + `LtxAvMusicEngine`. Keep `LtxAudioInEngine`
(name=`ltx_audio_in`; KEEP the id -- unanimous panel: rename is pure churn). Set
`default_roles=("music_visual","announcer_visual")` (inherits the music engine's
per-role default). accepts_still=True, _is_talk=True, roles=(announcer_visual,
music_visual, character_video), required=(text_prompt,audio_ref,init_image),
fallback None. Rewrite the module/class docstrings (drop the "two adapters /
back-compat / dark" narrative so no one rebuilds the split). Update __all__.

VALIDATED_ENGINES (registry): remove `ltx_av_music` + `ltx_av_talk`; ADD
`ltx_audio_in`. Provenance: it is the SAME GPU-proven _LtxAvBase + weights the
validated pair used, on the I2V branch ltx_av_talk already exercised (the only
delta is "always I2V"); the Part E smoke confirms the live render before the soak.
(If the operator prefers strict gating, hold the VALIDATED add until the smoke is
green and land it as a one-line follow-up in the same session.)

## Part B -- ROLE-DRIVEN routing (the robust generalization)

Add ONE beat classifier + a still-route helper; rewrite the three axes to consult
them. All derive from the ROLE + the engine's DECLARED still capability, read from
the REGISTERED engine (fail LOUD on an unknown selected engine -- never a silent
"does not consume still" default; GPT #6).

B0. Classifier (pure, role-first):
    `_is_character_face_beat(shot)` = role=="character_video" OR (family=="
    audio_driven_face" AND role not in (announcer_visual,music_visual)) OR
    (char_id present AND role not in (announcer_visual,music_visual)). This is the
    SINGLE source of "is this a talking head" used by all three axes.

B1. STILL route helper `still_route(engine, shot) -> {portrait|scene_still|
    mesh_fodder|none}`:
    - `requires_mesh_fodder` -> mesh_fodder (unchanged path).
    - `_is_character_face_beat` AND engine consumes a still -> portrait (keep
      asset_refs.init_image; do NOT overwrite with the scene still).
    - scene/bookend beat AND engine consumes a still -> scene_still (wide; clear
      portrait so it can't leak into a wide frame).
    - else -> none.
    "consumes a still" = `init_image in required_inputs` (REQUIRED still) OR
    `accepts_still and engine in the optional-I2V set {ltx_video}` (OPTIONAL still).
    Missing-still terminality: REQUIRED-still engine (ltx_audio_in: init_image in
    required_inputs) with no still -> FAIL LOUD before GPU (input-gap, clear
    message + trace stamp). OPTIONAL-still engine (ltx_video) with no still ->
    degrade to text-only, LOUD, init_source=missing_scene_still (the existing
    _i2v_still_missing semantics). This single helper SUBSUMES the
    _SCENE_INIT_FAMILIES + flux_still/flat_still + ltx_video branches; station_card
    has no init_image in required_inputs and is NOT in the optional set -> route
    none (exclusion preserved). VERIFY-AT-BUILD: enumerate every registered
    engine's (name, family, render_aspect, accepts_still, required_inputs, intended
    route) and assert the helper reproduces today's behavior for each.

B2. AUDIO: `_uses_ambient_master_audio` must return False for a character-face
    beat regardless of family. Change signature to take the shot (or pass the
    classifier result): ambient master slice only for scene/bookend
    audio_conditioned_video + visualizer; character-face beats keep the clean
    own-voice / no-audio degrade (the audio_driven_face behavior).

B3. PROMPT: the scene-prompt branch (1163) applies to scene/bookend beats only.
    Add `ltx_audio_in` to the engine set BUT gate the whole branch on `not
    _is_character_face_beat` so a character beat falls through to the
    char-fallback / gear-scrub / M4 path (as audio_driven_face does today).

B4. OTR_ENABLE_LTX_I2V stays scoped to `eng=="ltx_video"` ONLY (it governs
    ltx_video's OPTIONAL I2V). It must NOT gate ltx_audio_in's REQUIRED still
    (that would disable the new default engine by design). No new flag.

## Part C -- wiring (same commit; JSON re-validated)

1. registry.py: delete the two CAPABILITIES rows + the engine-list entries;
   VALIDATED_ENGINES per Part A. Keep ltx_audio_in's row.
2. otr_video_dep_pilot.py: delete the two OPT_IN_ENGINES entries; keep ltx_audio_in.
3. render_driver name-maps: drop the two from SYNTH_FALLBACKS, ENGINE_FAMILY,
   _LTX_OPEN_ENGINES; ENGINE_FAMILY["ltx_audio_in"]="audio_conditioned_video";
   _LTX_OPEN_ENGINES add ltx_audio_in. Canvas clamp (1082): replace the name-set
   with `("ltx_audio_in",)` (simplest correct -- Gemini: the driver has only the
   id string here; a requires_flag lookup needs a guarded registry fetch, not worth
   it). engine_family() registry-first where cheap.
4. 16gb_full.json: announcer_visual + music_visual -> ltx_audio_in.
5. workflows/otr_scifi_16gb_full.json: node-87 announcer/music widgets
   ltx_av_music -> ltx_audio_in (positional value-in-place). Re-validate.
6. otr_scifi_16gb_full_api.json: confirm NO runtime/validator/launcher consumes it
   (grep the loaders); if a consumer exists, regenerate from canonical; else leave
   + note it generated. Do NOT hand-edit as canonical.

## Part D -- tests

- A NEW table-driven `build_request_from_shot` routing-matrix test (the high-risk
  surface, GPT #8): ltx_audio_in announcer/music -> scene_still + ambient slice +
  scene prompt; ltx_audio_in character -> portrait + clean audio + char prompt;
  ltx_audio_in scene-beat MISSING still -> fail LOUD (required); audio_driven_face
  -> portrait; ltx_video missing still -> text-only LOUD (optional);
  flux_still/flat_still -> scene_still wide, portrait cleared; mesh_stage -> fodder;
  station_card -> none; visualizer -> none.
- test_ltx_audio_in_engine.py: drop test_two_legacy_variants_unchanged; assert the
  two are GONE from registry + CAPABILITIES + VALIDATED_ENGINES, ltx_audio_in is
  the default for music+announcer.
- Rewrite to ltx_audio_in: test_video_ltx_av, test_ltx_av_driver_wiring,
  test_capability_profiles, test_workflow_live_passes_validator (wv87),
  test_image_platform_c1 (accepts_still opt-out case -> visualizer; add ltx_audio_in
  required-still case), test_video_ledger, test_video_motion,
  test_video_render_driver_perbeat_audio, test_still_aspect_and_labels,
  test_tested_only_dropdown_gate (validated set), test_ltx_open_health,
  tests/debug_prompt.json.

## Part E -- validate + ship (pre-merge vs post-merge split, GPT #5)

PRE-MERGE (gates the commit): full pytest -q + Bug Bible regression + workflow
validator + AST/no-BOM on touched .py. Commit+push v2.0-alpha; HEAD==origin.
POST-MERGE (gates the soak, not the code): reset box (selective CIM kill), boot
UTF-8 launcher, SMOKE one short episode (ltx_audio_in bookends + still_parallax
char beats + indextts2) -> OBS final in output/otr/obs; THEN relaunch the 420 soak.

## Resolved forks

- F1 narrow-vs-robust: ROBUST (role-driven). The "narrow safe-mirror" (add
  ltx_audio_in to the ltx_video branch) would STILL clobber the character portrait
  -- it does not escape the defect. Role-driven is the only correct option AND it
  is the operator's "most robust".
- F2 capability: gate on `init_image in required_inputs` (required) + an explicit
  optional set {ltx_video} (accepts_still), NOT accepts_still alone (station_card
  inherits True). Panel-unanimous.
- F3 rename: CUT. Keep ltx_audio_in.
- F4 kill-switch: scope OTR_ENABLE_LTX_I2V to ltx_video only.

## Verify-at-build checklist

1. Enumerate every registered engine: name, family, render_aspect, accepts_still,
   required_inputs -> assert still_route reproduces today's behavior per engine.
2. station_card.required_inputs has NO init_image (route none) -- confirm.
3. ltx_video.required_inputs -- confirm init_image NOT required (optional set).
4. wan_i2v / flux_still / flat_still / mesh_stage routes unchanged.
5. default-role resolution actually consults default_roles for ltx_audio_in.
6. no other workflow widget (beyond node-87) names the two engines.
7. no runtime/validator consumer of otr_scifi_16gb_full_api.json.
