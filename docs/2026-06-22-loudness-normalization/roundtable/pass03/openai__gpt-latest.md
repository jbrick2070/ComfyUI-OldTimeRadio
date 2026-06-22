<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. The dark-default code path may be buildable, but the rms flip/re-baseline procedure is internally inconsistent and can generate fixtures that fail the runtime regression depending on environment.

MUST-FIX BEFORE BUILD:
1. [3][5] The “flip to rms + commit BOTH fixtures” plan contradicts “OTR_SEGMENT_* is env-only” and “default = peak.” If fixtures are regenerated under `OTR_SEGMENT_LOUDNORM=rms` but the test/runtime default remains peak, `test_audio_byte_identical_to_baseline` will generate peak-mode bytes and compare them to an rms-mode golden whenever `OTR_REGRESSION_RUNTIME=1` is set. Concrete fix: define the flip artifact explicitly:
   - Option A: change the code default from `peak` to `rms` in the same commit as the regenerated fixtures, and document calibrated `OTR_SEGMENT_TARGET_RMS_DBFS`.
   - Option B: keep default `peak`, do not commit rms fixtures, and treat rms as deployment-only; if you need an rms regression, add a separate rms-specific test that forces the env.
   - Option C: make the byte-identical test set the intended mode explicitly before generation, so fixture mode and generated mode cannot diverge.

2. [3][6] The re-baseline procedure says “Set env” before running `python tests/test_audio_byte_identical.py --capture-baseline`, but the env must exist in the process that executes `nodes/scene_sequencer.py`, where `_master_loudness` reads `os.environ`. If `tests/_run_baseline.py` talks to an already-running ComfyUI process, setting env only in the capture shell will not affect the node execution. Concrete fix: update the procedure to start/restart the ComfyUI/node process with `OTR_SEGMENT_LOUDNORM`, `OTR_SEGMENT_TARGET_RMS_DBFS`, `OTR_MASTER_MAKEUP_DB`, `OTR_CAST_SEED`, and `OTR_STYLE_SEED`; add a capture preflight/log assertion that the effective loudnorm mode and target are `rms` and the expected value. verify: whether `_run_baseline.py` executes in-process or through a running ComfyUI server.

3. [3][4] `OTR_MASTER_MAKEUP_DB` is not pinned during calibration/re-baseline. Grounding says `_master_loudness` currently reads `OTR_MASTER_MAKEUP_DB` and the plan only defaults it to `0.0` in rms mode when env is unset. A stale operator env value, especially the old `4.0`, will silently change the captured baseline and calibration result. Concrete fix: explicitly set `OTR_MASTER_MAKEUP_DB=0` for rms calibration and rms baseline capture, or explicitly unset it and assert the effective value is `0.0`.

4. [4] Calibration mixes incompatible measurement points: “master WAV” versus “per-clip dialogue segments.” A master WAV includes silence, SFX/music if present, and master makeup; per-clip dialogue segments do not. Those RMS values are not interchangeable targets for per-segment normalization. Concrete fix: specify one reference point. For segment normalization, measure dialogue-only clips after the current peak normalization path and before master makeup; if the goal is matching final heard level, compensate for the old master makeup or measure final dialogue-only output consistently.

5. [4] The statement “set target to current peak-mode dialogue RMS so, with master makeup -> 0, program loudness roughly matches today” is not necessarily true with the grounded current default `OTR_MASTER_MAKEUP_DB=4.0`. If today’s audible output includes +4 dB master makeup, then rms mode with master makeup 0 and the same pre-master segment RMS will be about 4 dB lower. Concrete fix: decide whether the intended release changes final program loudness. If not, calibrate the rms target against current final output level, or retain/explicitly account for the 4 dB makeup.

6. [2][5] The plan treats the byte-identical regression as safe even though it normally skips unless `OTR_REGRESSION_RUNTIME=1` and GPU are available. That skip does not prove the default peak path is byte-identical after replacing the three dialogue call sites. Concrete fix: either run the runtime byte-identical test once with `OTR_REGRESSION_RUNTIME=1` before landing Chunk 1, or make `tests/test_segment_loudnorm.py` include an exact no-op test proving unset/`peak` mode produces byte-identical arrays to the previous `_normalize_clip(...)` behavior for the dialogue helper/wrapper used at :747/:753/:775.

7. [6] “Confirm module-level `import os` / `import math`” is left as open verification, but missing imports would be a direct build/runtime failure if the R2 implementation uses them. Concrete fix: make import verification part of Chunk 1, run the relevant unit tests/import checks, and do not leave this as an operator/build-time discovery. [ASSUMPTION] This applies if the new loudnorm code references `os`/`math` as implied by the plan.

SHOULD-FIX:
1. [3] Step 4 says “A/B the old vs new WAV” after Step 3 rewrites the baseline WAV. Git can recover the old file, but the procedure should not rely on that. Concrete fix: copy the old baseline to a temp path or name the capture output separately before overwriting `tests/fixtures/baseline_v1.5.wav`.

2. [4] The calibration script is underspecified for PCM scaling and channels. RMS dBFS depends on whether samples are int16, float `[-1, 1]`, stereo summed, or channel-averaged. Concrete fix: specify conversion to float full-scale, channel handling, silence trimming/windowing, and clipping/NaN guards.

3. [4] “Mean RMS dBFS” over an entire episode will be biased by silence and pauses even if using dialogue-only rendered clips. Concrete fix: compute RMS on active dialogue windows or clips above a silence threshold, then aggregate per clip/line, not over all samples including gaps.

4. [5] Documentation is split into a later “Chunk 2,” but Chunk 1 introduces live env knobs. Concrete fix: include the README/audio-level env table in the same PR/merge as the code, even if it is a separate commit.

5. [3] The fixture name `baseline_v1.5.wav` becomes misleading if it is regenerated for an rms-mode v2 behavior. Concrete fix: either rename/update the test constants in the same change, or explicitly document that the fixture filename is historical and not a semantic version.

6. [3] The manual acceptance criteria are vague: “listen + loudness read” and “2-3 episodes” do not define pass/fail. Concrete fix: state minimum checks: no clipping, no obvious pumping/noise-floor lift, target RMS tolerance, peak ceiling, and sample episode selection.

7. [1] The no-JSON claim is grounded for the searched normalization terms, but the plan should avoid depending on that grep as the only guard. Concrete fix: document the exact intended runtime boundary: workflow remains unchanged because all knobs are read inside `OTR_EpisodeAssembler`/`_master_loudness`.

OPTIONAL / NICE-TO-HAVE:
- Add effective audio-level config to the render log: mode, target RMS, peak ceiling, master makeup effective value.
- Add a small reusable calibration script under `tools/` instead of a “throwaway numpy script,” so future fixture updates are reproducible.
- Record the calibrated value and sample episodes used in the release notes.

CUT THESE (over-engineering):
1. [5] Cut the separate docs chunk as a separate phase. Fold docs into Chunk 1; this is safer and simpler because the env knobs become live as soon as the code lands.
2. [3] If the flip remains purely env/deployment-only, cut committing rms-regenerated golden fixtures. It is safe to cut because the default regression baseline should match the default repo behavior; use a separate env-forced rms regression only if needed.