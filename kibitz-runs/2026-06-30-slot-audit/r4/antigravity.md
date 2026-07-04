VERDICT: yes-with-fixes. The soak scripts contain stale 3-slot model and role references that will block successful execution of the coverage sweep and combo soaks under the new Route-A 5-slot architecture.

MUST-FIX BEFORE BUILD:
1. [C5] Stale slots and roles in soak scripts block running coverage sweeps or combo soaks on the new 5-slot model.
   - Defect: [scripts/_otr_cov_runner.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/scripts/_otr_cov_runner.py) (lines 50-52, 95-98), [scripts/otr_coverage_sweep.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/scripts/otr_coverage_sweep.py) (lines 87-91), and [scripts/_otr_combo_soak.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/scripts/_otr_combo_soak.py) (lines 71-76) use the legacy 3-slot role overrides (`other_beats_visual`, `other_beats_image`, `character_visual`) instead of the 5 Route-A roles (`announcer_visual`, `music_visual`, `character_video`, `scene_broll`, `background_abstract`).
   - Concrete fix: Update the `VIDEO_SLOTS`, `SLOTS`, `build_profile`, and `_combo_profile` overrides in all three scripts to use the five Route-A video slots (`announcer_video_model`, `music_video_model`, `character_video_model`, `scene_broll_video_model`, `background_abstract_video_model`) and their respective roles.
2. [C1] `StillPanFamily` is missing `accepts_still` declaration, causing black frames/dark floor fallbacks when selected.
   - Defect: [nodes/_otr_video_engines/cheap_families.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_video_engines/cheap_families.py#L204-L222) does not declare `accepts_still = True`. This causes `engine_consumes_still` to evaluate to `False` for `still_pan`, leading the dispatcher to skip generating the scene still.
   - Concrete fix: Add `accepts_still = True` to the `StillPanFamily` class in `cheap_families.py`.
3. [C2] Capability filtering is not implemented in the video engine registry.
   - Defect: [nodes/_otr_video_engines/registry.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_video_engines/registry.py) directly instantiates the base `EngineRegistry` without overriding `engines_for_role` and `assert_usable` to check `role_compat.engine_fits_role`.
   - Concrete fix: Define a subclass `VideoEngineRegistry(EngineRegistry)` in `nodes/_otr_video_engines/registry.py` that overrides `engines_for_role` and `assert_usable`. Have it call `role_compat.engine_fits_role` using descriptors derived from registered engines, and implement the fail-soft fallback to the legacy `roles` whitelist when `required_inputs` is `None` or the role is unknown to `role_compat`.
4. [C3] Stale `speaker_role` mapping leaves `scene_broll_video_model` unreachable.
   - Defect: [nodes/otr_shot_lock.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/otr_shot_lock.py#L55-L64) does not map `"scene"` or `"sfx"` to `Role.SCENE_BROLL.value` in `SPEAKER_TO_VIDEO_ROLE`. As a result, all scene-descriptor/sfx beats fall back to `_DEFAULT_VIDEO_ROLE` (`"background_abstract"`), routing to `background_abstract_video_model` rather than `scene_broll_video_model`.
   - Concrete fix: Add `"scene": Role.SCENE_BROLL.value` and `"sfx": Role.SCENE_BROLL.value` to the `SPEAKER_TO_VIDEO_ROLE` dictionary.

SHOULD-FIX:
1. [C5] Oracle test harness lacks automated metric validation of non-dark frames.
   - Defect: The soak sweep requires programmatic verification of frame content.
   - Concrete fix: Define and implement the Oracle test interface checking that output clips have an ffmpeg `signalstats` YAVG value greater than a minimum floor threshold for non-darkness.

OPTIONAL / NICE-TO-HAVE:
- [C5] Exempting `still_flat` and `visualizer` from temporal variance validation in the Oracle test sweep to avoid false positives. [ASSUMPTION: We assume some test framework handles filtering static vs motion engines based on `uses_still` or `_still_motion` attributes].

CUT THESE:
- None. All proposed items are necessary to support the 5-slot architecture alignment.

VERIFY-AT-BUILD checklist:
1. Verify `test_engine_consumes_still_capability_vs_dual_read` in [tests/test_image_platform_c1.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/tests/test_image_platform_c1.py) passes.
2. Verify `test_video_motion.py` capability assertions pass and reject based on missing capabilities rather than the flat role whitelist.
3. Verify that running `python scripts/_otr_cov_runner.py` executes without missing-still warnings (proving `StillPanFamily` successfully triggers dispatcher image generation).
4. Verify that a beat with `speaker_role="scene"` yields `role="scene_broll"` (asserted via a unit test in `tests/test_still_spine_helpers.py`).
