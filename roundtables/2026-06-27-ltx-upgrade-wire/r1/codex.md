VERDICT: yes-with-fixes. The plan is close, but it overclaims recipe-agnostic safety and makes a global composite-look change for an LTX-specific bakeoff win.

MUST-FIX BEFORE BUILD:
1. [LOCKED decision / Code changes / Pre-ship VERIFY] The evidence is for distilled_native, but the production default still auto-resolves to sharp_lora. `scripts/build_ltx_av_q_bakeoff_workflow.py:57-65` pins distilled Q3 plus DEV companions; `nodes/_otr_video_engines/eng_ltx_av.py:240-294` defaults to dev Q3 and maps dev to `RECIPE_SHARP_LORA`. Concrete fix: make the canonical smoke explicitly run the current no-env default and record `recipe=sharp_lora` plus peak, or explicitly defer the code change until distilled_native default is also decided.

2. [SCALER (`otr_silent_composite._seg_vf`)] The unsharp change is global, not LTX-only. `_seg_vf` is used by `_encode_segment` for both clip and floor segments (`nodes/otr_silent_composite.py:319-349`, `:579-585`), so it will sharpen procgen floor and any non-LTX clip too. Concrete fix: either gate the sharpen filter to `engine_id == "ltx_audio_in"` or state it as an intentional whole-episode look change and add a smoke covering floor/non-LTX segments.

3. [Pre-ship VERIFY #5] The “peak VRAM < 14500” acceptance needs a real render-window peak sampler. Current `eng_ltx_av` calls `run_graph` first, then reclaim, then an instantaneous ceiling assert (`nodes/_otr_video_engines/eng_ltx_av.py:621-633`); `motion_common.VramPeakProbe` exists specifically because post-render reads miss peaks (`nodes/_otr_video_engines/motion_common.py:242-295`). Concrete fix: require the smoke harness to sample NVML across `run_graph` and fail if peak is missing/zero or >14500.

4. [Pre-ship VERIFY #1] `wrapper_bridge.py:37` is the wrong ceiling sentinel for this engine’s runtime guard. `eng_ltx_av` imports `motion_common` as `_MC` (`nodes/_otr_video_engines/eng_ltx_av.py:46-48`) and uses `_MC.dynamic_vram_ceiling_mb()` / `_MC.assert_vram_within_ceiling()` (`nodes/_otr_video_engines/eng_ltx_av.py:350-356`, `:632-633`), while `wrapper_bridge.py:37` is just a static constant. Concrete fix: replace this verify item with `motion_common.dynamic_vram_ceiling_mb()` under the intended env/profile plus the measured peak assertion.

SHOULD-FIX:
1. [No canonical-workflow-JSON edit] Add an explicit workflow-route audit, not just “no temporal widgets.” The canonical workflow currently routes `OTR_VideoDirector` announcer/music to `ltx_audio_in` and wires manifest into `OTR_SilentComposite` (`workflows/otr_scifi_16gb_full.json`, node 87 widgets and links 261/271), while node 92 still has an `engine` widget value of `humo_1.7B`. Concrete fix: verify the actual canonical API prompt contains at least one `ltx_audio_in` beat and that it reaches node 84’s manifest input.

2. [Code changes / Tests] The scaler test should assert both the intended LTX path and the non-LTX policy. Current tests mention `OTR_SilentComposite` broadly but no `_seg_vf` filter contract (`tests/test_video_render_path_cw4.py`, `tests/test_video_directory_clip.py`). Concrete fix: add the `_seg_vf` assertion and, if sharpening is gated, a test that floor/non-LTX filters remain unchanged.

3. [Pre-ship VERIFY #5] Add the reset prerequisite from repo operating rules before the headless smoke. The plan says run a real canonical-workflow smoke but does not state selective process/port/VRAM reset; `CLAUDE.md:73-82` makes that mandatory. Concrete fix: prepend the reset checklist to the verify section.

OPTIONAL / NICE-TO-HAVE:
- Add a tiny visual-metric check after the composite filter, because the planned unit test only proves the ffmpeg string, not that the delivered clip has the intended sharpness.

CUT THESE (scope / over-engineering):
1. [Pre-ship VERIFY #1] Cut `wrapper_bridge.py:37 still VRAM_CEILING_MB = 14500`; it is not the active runtime ceiling for `ltx_audio_in` and can give false confidence.

2. [Optional] Cut companion-drift hardening from this build. `scripts/build_ltx_av_q_bakeoff_workflow.py:62-65` already names the bakeoff companions; hardening the bakeoff manifest does not change the production path.

3. [LOCKED decision] Cut “whole-clip documented manual max-quality option” from this commit unless a real doc or operator-facing switch is being changed. The code change list only wires 128/32 and `_seg_vf`; documenting 4096/8 here is extra surface.