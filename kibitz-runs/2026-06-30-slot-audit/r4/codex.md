VERDICT: yes-with-fixes. Core direction has converged, but C1/C5 still leave registered still engines in an undefined state and C5’s oracle/baselines are not build-specific enough.

MUST-FIX BEFORE BUILD:
1. [C1/C5] Registered still engines contradict “all_engine_names() x role_compat” coverage. `StillMotionFamily` is registered, `uses_still=True`, `required_inputs=("text_prompt",)`, but has no `accepts_still` (`nodes/_otr_video_engines/cheap_families.py:174-181`); `engine_consumes_still()` will return False unless `accepts_still` is set or `init_image` is required (`nodes/otr_image_gen_dispatcher.py:287-304`). `render_driver` would bind a scene still for `static_motion` if one existed (`nodes/_otr_video_engines/render_driver.py:895-903`), so C1 currently leaves still_motion able to black-floor under the all-engine soak. Concrete fix: add `accepts_still=True` to `StillMotionFamily` and test `init_source=="scene_still"`, or retire/deregister it before C5 enumeration.

2. [C1/C5] `station_card` is called “out of scope” but remains registered (`nodes/_otr_video_engines/cheap_families.py:184-191`) and C5 says enumerate `all_engine_names()`. It is also explicitly excluded from the scene-still binding branch (`nodes/_otr_video_engines/render_driver.py:910-918`). Concrete fix: make the retirement concrete before build: remove/deregister `station_card` and its capability row, or explicitly exclude it from C4/C5 with a named non-goal. Do not leave it as a registered engine that the “all engines” soak must exercise.

3. [C5] Non-under-test baselines are underspecified. “KNOWN-COMPATIBLE baselines” allows different builders to choose different fillers, and a bad filler can hide or create failures through `other_beats_video_model` fallback. Concrete fix: name the baseline per role in the plan, e.g. `still_flat` for all non-under-test video roles plus `flux_gen1` for image roles, unless the tested role itself requires a different carrier; require the profile to set all five role keys, never rely on the legacy other-beats fallback.

4. [C5] Oracle interface is still ambiguous: “per-beat clip / obs final” gives two incompatible sources, and `YAVG > floor threshold` has no numeric threshold. The clip manifest already carries per-beat paths, engine_id, family, role, frame_count, and init_source (`nodes/_otr_video_engines/render_driver.py:2000-2055`; `nodes/otr_video_render_batch.py:85-87`). Concrete fix: use per-beat manifest `clips[].path` for per-beat luma/motion checks; reserve obs final for publish smoke only. Specify the ffmpeg/ffprobe command, numeric luma floor, frame-diff/freezedetect window, and exact exemptions.

SHOULD-FIX:
1. [C1] “still-consuming bind set” is imprecise because `engine_consumes_still()` is true for portrait consumers such as HuMo. Concrete fix: rename this test target to “explicit scene-still bind set” and list `still_pan`, `still_flat`, `ltx_audio_in` plus `ltx_video` through its separate branch.

2. [C2] Unknown-role handling is internally muddy: C2 says fall back to legacy `roles` when the role is unknown, but also says wrap `RoleCompatError` to `EngineUnusable`. Concrete fix: specify exact behavior separately for `engines_for_role()` and `assert_usable()`.

3. [C3] Keep the “verify actual speaker_role token” step, but add the source to inspect: the writer/schema prompt that constrains `speaker_role` values. [ASSUMPTION] “scene” may be the intended token, but the plan correctly does not assert it yet.

OPTIONAL / NICE-TO-HAVE:
- Keep the widget_mapping audit as optional, but it is useful: current mapping already includes all five video role keys in `config/profiles/widget_mapping.json`.

CUT THESE:
1. [C5] Cut per-beat checks against the obs final. The final mux/composite can obscure which beat failed; per-beat manifest paths are the right oracle surface.
2. [C1] Cut the vague phrase “retirement candidate” unless the build includes actual deregistration. It is not actionable.

VERIFY-AT-BUILD checklist:
1. [C3] Verify the actual b-roll `speaker_role` token and add a test that it resolves to `scene_broll` through `SPEAKER_TO_VIDEO_ROLE` (`nodes/otr_shot_lock.py:55-76`) and the image prompt import path (`nodes/otr_meta_brief_image_prompt.py:290-297`).
2. [C2] Verify `required_inputs=()` does not fall back to legacy `roles`; `AbstractFamily.required_inputs=()` must fit all five roles by capability (`nodes/_otr_video_engines/cheap_families.py:165-172`; `nodes/_otr_shared/role_compat.py:107-131`).
3. [C5] Verify all three stale soak entry points are retired/delegated: current stale 3-slot code exists in `scripts/_otr_cov_runner.py:50-55`, `scripts/otr_coverage_sweep.py:86-91`, and `_otr_combo_soak.py:67-89`.
4. [C5] Verify the canonical JSON path is loaded and patched through `apply_profile_to_workflow`, using `widget_mapping.json` node-type targets, not node ids.
5. [C4] Verify the matrix is generated from registry descriptors and `role_compat.engine_fits_role`, not a hand-coded expected table.
6. [C5] Verify the oracle reads `node_episode_manifest.json` / `clip_manifest_json` rows and checks actual per-beat clip files before final publish.