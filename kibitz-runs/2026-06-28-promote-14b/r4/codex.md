VERDICT: no. The plan still leaves the scene/background engine choices and the obsolete soak gate unresolved, so a builder can produce incompatible outputs or a failing suite.

MUST-FIX BEFORE BUILD:
1. [OPERATOR product call / The change #1] Build-blocking unresolved defaults: `scene_broll_video_model` and `background_abstract_video_model` are still “which role-valid engines / current ones.” Current code has no single valid “current” other-beats answer: `workflows/otr_scifi_16gb_full.json` node 87 currently stores `other_beats_video_model = "visualizer (16:9)"`, while `visualizer` requires `audio_ref` and declares only announcer/music/character roles (`nodes/_otr_video_engines/eng_visualizer.py:40-49`), and `background_abstract` supplies only `text_prompt` (`nodes/_otr_shared/role_compat.py:69-72`). Concrete fix: replace the operator gate with exact committed values, e.g. `character_video_model = humo_14B_169`, `scene_broll_video_model = <exact id>`, `background_abstract_video_model = <exact id>`, and require each value to pass `engine_fits_role` for its specific role.

2. [Workflow widgets / Build order] Widget append location is underspecified and can silently corrupt saved values. `serialized_slot_names` orders all required widget-backed fields before optional widget-backed fields (`nodes/_otr_workflow_apply.py:172-204`). `OTR_VideoDirector` currently has widget-backed optional `episode_duration_target` and `custom_models_json` after required widgets (`nodes/otr_video_director.py:203-219`). If the new role widgets are added as required fields, they will insert before those existing optional slots, not append to the saved vector. Concrete fix: state exactly where to add them: add new widget-backed fields after existing serialized widgets, e.g. optional after `custom_models_json` and before forceInput `gate_in`, then append the same values to node 87 `widgets_values`.

3. [Acceptance] `assert_soak_ok` is still conditional: “fix if run_gpu_soak stays a gate.” That is not build-ready because `render_shot` explicitly disables fallbacks (`nodes/_otr_video_engines/render_driver.py:1468-1500`) while `assert_soak_ok` still requires OOM fallback to `still_kenburns`, an exact degradation trail, and two fallback decisions (`nodes/_otr_video_engines/render_driver.py:2082-2097`). Concrete fix: make the plan unconditional: either retire `run_gpu_soak` as a gate, or update `assert_soak_ok` and `tests/test_video_render_driver.py:74-121` to the no-fallback/14B routing invariant.

SHOULD-FIX:
1. [The change #2] Name the shared role-to-video-slot helper and its complete consumer list. Current duplicate maps exist in `OTR_ShotLock` (`nodes/otr_shot_lock.py:708-715`), `OTR_ImageDirector` (`nodes/otr_image_director.py:156-165`), `OTR_ImageGenDispatcher` (`nodes/otr_image_gen_dispatcher.py:280-289`), and `OTR_VideoDirector._role_aspects` (`nodes/otr_video_director.py:306-325`). Concrete fix: specify one helper module/function and require all four call sites, including `_role_aspects`, to use it.

2. [Acceptance] Add an explicit test assertion that `video_policy["aspects"]` contains per-role entries for `character_video`, `scene_broll`, and `background_abstract`. MetaBrief consumes that map for still sizing (`nodes/otr_meta_brief_image_prompt.py:150-164`), so routing can be correct while still dimensions remain stale.

OPTIONAL / NICE-TO-HAVE:
- Add a one-line migration note for old profiles using `role_overrides.other_beats_visual`, since current profiles still use that key (`config/profiles/16gb_full.json:12-14`, `config/profiles/8gb_lite.json:12-14`, `config/profiles/cpu_floor.json:12-14`).

CUT THESE:
1. None. The remaining checks are tied to real historical failure modes: workflow widget drift, unwired JSON, exact frame count, and no-fallback acceptance.

VERIFY-AT-BUILD checklist:
- Verify final scene/background engine ids against `engine_fits_role` for their exact roles; do not rely on current `other_beats_video_model`.
- Verify node 87 widget order: saved widget names remain an ordered prefix/subsequence of live `INPUT_TYPES`; no mid-list insertion.
- Run `OTR_WorkflowValidator`, JSON round-trip, link integrity, wired input-name audit, and widget-count/order audit on `workflows/otr_scifi_16gb_full.json`.
- Verify `HuMo14BLandscapeEngine` alone gets the 14B frame cap; base `humo` and `humo_1.7B` remain uncapped.
- Verify rendered HuMo14B clips have `frame_count == target_frame_count` after trim/extend.
- Verify live episode histogram has `humo_14B_169 > 0` only on `role == character_video`.
- Verify HuMo14B install/preflight: `OTR_ENABLE_HUMO`, checkpoint path, wrapper node classes, and no OOM at representative and max-cap beats.