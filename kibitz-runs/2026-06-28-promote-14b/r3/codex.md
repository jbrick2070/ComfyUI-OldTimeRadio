VERDICT: no. Route-A/Route-B are not build-ready until the image-policy joins, profile/applier wiring, and HuMo frame contract are made explicit.

MUST-FIX BEFORE BUILD:
1. [CENTRAL FINDING / MUST-FIX 1] Route-A omits the image-side role-to-video-slot joins. `nodes\otr_image_director.py:156-196` and `nodes\otr_image_gen_dispatcher.py:280-341` still map `character_video`, `scene_broll`, and `background_abstract` to `other_beats_video_model`. If `other_beats_video_model` stays `visualizer`, dispatcher sees `visualizer.accepts_still=False` (`nodes\_otr_video_engines\eng_visualizer.py:45-49`) and can skip the character still that HuMo requires (`nodes\_otr_video_engines\eng_humo.py:331-334`). Fix: update both maps, preferably via one shared helper, so `character_video -> character_video_model`, `scene_broll -> scene_broll_video_model`, `background_abstract -> background_abstract_video_model`.

2. [CENTRAL FINDING / MUST-FIX 1] The plan’s “scene/background -> current compatible engine (visualizer/ltx)” is not compatible under the current validator. `background_abstract` supplies only `text_prompt` (`nodes\_otr_shared\role_compat.py:69-72`); `visualizer` requires `audio_ref` (`nodes\_otr_video_engines\eng_visualizer.py:45`) and `ltx_audio_in` requires `text_prompt`, `audio_ref`, and `init_image` (`nodes\_otr_video_engines\eng_ltx_av.py:829-832`). Fix: choose per-role engines that actually pass `engine_fits_role` (`nodes\_otr_shared\role_compat.py:107-131`), or deliberately update `ROLE_AVAILABLE_INPUTS` if background beats truly receive bounded master audio.

3. [MUST-FIX 3] `extend_frames_to_target` does not guarantee `frame_count == target_frame_count`. HuMo currently quantizes upward with min 33 / max 177 (`nodes\_otr_video_engines\eng_humo.py:339-341`; `nodes\_otr_video_engines\wrapper_bridge.py:386-400`), while `extend_frames_to_target` returns unchanged when `target <= n` (`nodes\_otr_video_engines\wrapper_bridge.py:457-459`). Short targets or 4n+1 overshoot will still encode too many frames. Fix: add an exact-fit step that trims when rendered frames exceed target and mirror-extends only when short, before `encode_frames_to_silent_mp4`.

4. [MUST-FIX 1 / MUST-FIX 4] New profile keys are not wired through the applier contract. `apply_profile` flattens only `role_overrides`, `slot_overrides`, `features`, and `seed_policy` (`nodes\_otr_workflow_apply.py:428-476`), and the trusted video-widget allowlist only names the old three Director widgets (`nodes\_otr_workflow_apply.py:139-141`, `226-234`). Fix: put new keys under an existing flattened section or extend the schema intentionally; update `config\profiles\widget_mapping.json`, `_VIDEO_DIRECTOR_WIDGETS`, and the mapping/profile tests together.

5. [Build order / MUST-FIX 4] Route-A widget edits can silently drift saved values if inserted mid-list. `OTR_VideoDirector.INPUT_TYPES` serializes widget order from `announcer_video_model` through `custom_models_json` (`nodes\otr_video_director.py:148-224`), and the real workflow has positional `widgets_values` for node 87 (`workflows\otr_scifi_16gb_full.json:1`). Fix: append new optional widgets at the end or preserve old slots and add new slots after existing serialized widgets; update the workflow JSON in the same change and run the widget-count audit.

6. [Route-B / CENTRAL FINDING] Route-B has no clean configuration source for “character = `humo_14B_169`, scene/background = configured engine.” Today ShotLock maps all three roles to `other_beats_video_model` (`nodes\otr_shot_lock.py:708-780`). `OTR_VideoRenderBatch.engine` is not a substitute because episode mode ignores it and renders from the ShotLock ledger (`nodes\otr_video_render_batch.py:127-139`, `198-206`). Fix: either use Route-A, or add an explicit character override input/config path instead of hardcoding 14B in ShotLock.

SHOULD-FIX:
1. [SHOULD-FIX acceptance] Update the soak/acceptance checks, not only the live manifest assertion. `_episode_facts` counts only exact `"humo"` (`nodes\_otr_video_engines\render_driver.py:2030-2047`), so `humo_14B_169` will not satisfy it. `assert_soak_ok` also still expects fallback decisions even though `render_shot` disables fallbacks (`nodes\_otr_video_engines\render_driver.py:1468-1495`, `2069-2107`). Fix if `run_gpu_soak` remains a gate.

2. [Build order] Replace “director rejects an aggregate engine” with an end-to-end routing test: Director policy -> ShotLock shots -> every shot’s `engine_id` fits its `role`, and ImageGenDispatcher keeps required HuMo stills. The current failure crosses nodes, so a Director-only test is too narrow.

3. [MUST-FIX 3] Make the safe-frame cap tier-specific and observable. Use an `OTR_`-prefixed env/config name or a class override on `HuMo14BLandscapeEngine`; do not cap `humo_1.7B` accidentally through the base class.

OPTIONAL / NICE-TO-HAVE:
- Centralize role-to-video-slot mapping in one shared module so Director, ImageDirector, ImageGenDispatcher, and ShotLock cannot diverge again.

CUT THESE (over-engineering):
1. [Route-B] Cut Route-B if Route-A is accepted. A ShotLock special case creates a second routing language and makes profile/workflow truth harder to audit.

2. [Build order] Cut or rewrite the “aggregate engine” CPU test. After Route-A there should be no aggregate other-beats engine for character/scene/background; test the stamped per-role contract instead.