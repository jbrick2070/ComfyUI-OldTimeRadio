# Still-count must follow the video clip_mode (pass00)

## Operator ask (verbatim intent)
After the coverage-arch fix (every video lane consumes the SELECTED image via
`accepts_still`), the image phase now emits a scene-still target for EVERY beat.
Operator refines: "only generate the images if it's a per-beat model; if it's
visualizer, NO images; if it's `pool_n_loop`, only generate the NUMBER of stills
indicated (N); and make `pool_n_loop` + N=4 the full-JSON default." Screenshot:
OTR_VideoDirector other_beats_video_model=humo_1.7B, other_beats_clip_mode=
pool_n_loop, other_beats_n=4.

So the still COUNT for the other-beats role must follow the clip mode:
- procedural floor (visualizer/abstract, `accepts_still=False`) -> 0 stills.
- `unique_per_beat` + a scene-consuming engine -> 1 still per beat.
- `pool_n_loop` -> exactly N stills, SHARED/looped across the M other-beats.

## Grounded facts (verified vs HEAD 0c16e85)
- `accepts_still` (MotionEngineBase default True; visualizer/ltx_av_music False) is
  read centrally by `engine_consumes_still` in otr_image_gen_dispatcher; the
  dispatcher already SKIPS minting a still whose role's video engine has
  `accepts_still=False`. So "visualizer -> 0 stills" is ALREADY handled.
- `derive_scene_still_targets(lines, fps)` (otr_meta_brief_image_prompt.py:177) now
  emits a `scene_beat` target for EVERY beat (open + announcer + music + character +
  background_abstract), role-mapped via SPEAKER_TO_VIDEO_ROLE (default
  background_abstract). It does NOT see the clip plan today.
- The VIDEO clip budget already pools: otr_shot_lock `_audio_derived_clip_budget`
  reads `policy["other_beats"]={clip_mode, pool_n}`; for pool_n_loop
  `render_count = min(pool_n, len(other))` (the other-beats = roles
  BACKGROUND_ABSTRACT + SCENE_BROLL). This runs AFTER audio_done / during ShotLock;
  the still phase runs BEFORE ShotLock (graph order).
- render_driver resolves a beat's init image by FAMILY (render_driver.py ~695-769):
  `audio_driven_face` (HuMo) keeps the character PORTRAIT (`_portrait_index` by
  char_id); `ltx_video`/static_motion/image_to_video/flux_still use the beat's SCENE
  still (`_still_index` by beat_id); a missing scene still on ltx_video -> the LOUD
  `LTX-I2V MISSING-STILL` text-only degrade. So in the screenshot's HuMo config the
  other-beats use PORTRAITS, not scene stills at all.
- Threading path for the clip plan into the still phase:
  OTR_VideoDirector (owns other_beats {clip_mode, pool_n}) -> OTR_ImageDirector
  (parses video_policy_json, emits image_policy_json + still 'aspects') ->
  OTRMetaBriefImagePromptGen.generate (gets image_policy_json) -> derive_image_prompts
  -> derive_scene_still_targets. (ImageDirector other_beats_image_model role set =
  {character_video, scene_broll, background_abstract}.)
- OTR_VideoDirector widget defaults TODAY: other_beats_clip_mode=unique_per_beat,
  other_beats_n=8. Operator wants pool_n_loop + 4 as the otr_scifi_16gb_full.json
  default.

## The crux (the real architecture question)
pool_n_loop is NOT just "cap the still count to N" -- the M other-beats must SHARE
the N pool stills, or beats N+1..M hit MISSING-STILL in render_driver (it looks up a
still per beat_id). So the still phase must either (a) emit N pool stills + a
beat->pool mapping the render_driver/still_index honors (beat i -> pool[i mod N]),
or (b) emit N stills keyed so every other-beat resolves to one of them. The VIDEO
budget already loops the CLIPS; the STILL index must loop consistently so the same
beat that reuses pool-clip k also reuses pool-still k (determinism).

## The decisions to harden
1. WHERE to enforce the still count: at emission (derive_scene_still_targets, needs
   the clip plan threaded) vs at dispatch (the dispatcher dedups) vs let the still
   index/loop mapping handle it. Which keeps "one place" + deterministic?
2. HOW the M other-beats share the N pool stills so NO beat misses (the loop mapping
   beat->pool, and where it lives so still-gen, the dispatcher, AND render_driver
   agree -- ideally the SAME pool/loop math otr_shot_lock uses for clips).
3. FACE-engine interaction: when other_beats=HuMo (portraits, not scene stills),
   should the still phase emit ANY other-beats scene stills at all? (Today's
   accepts_still=True for HuMo would mint per-beat scene stills HuMo never uses ->
   waste.) Should "consumes SCENE stills" be distinct from "accepts_still"?
4. The JSON/default change (pool_n_loop + N=4) -- widget default vs only the saved
   otr_scifi_16gb_full.json node; keep the workflow source-of-truth + re-validate.
5. Name the correctness traps: graph-order (still phase before ShotLock has no audio
   budget), determinism of the loop mapping, the existing per-beat tests, no silent
   fallback, do NOT over-generate, keep announcer/music per-beat (not pooled).

## Invariants
accepts_still stays the central usability gate; role_compat unchanged; no silent
fallback (LOUD on any miss); single-resident unchanged (fewer renders, not more);
workflow JSON is source of truth; determinism (seed-keyed, no RNG); UTF-8 no BOM.
