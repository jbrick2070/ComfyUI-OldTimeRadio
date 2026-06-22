<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: no. The re-baseline procedure fundamentally misunderstands how environment variables interact with the ComfyUI server process, which will result in capturing a false (peak) baseline.

MUST-FIX BEFORE BUILD:
1. [Section 3 & 6] Env vars do not propagate from the test client to the ComfyUI server. The grounding explicitly states "start ComfyUI with OTR_CAST_SEED". Setting `OTR_SEGMENT_LOUDNORM=rms` in the terminal running `tests/test_audio_byte_identical.py` does nothing to the nodes executing inside the server. Fix: Add a strict step to Step 2 to *restart the ComfyUI server process* with `OTR_SEGMENT_LOUDNORM`, `OTR_SEGMENT_TARGET_RMS_DBFS`, and the seeds in its environment before running the capture script.
2. [Section 5] False assumption regarding test skipping. The plan states "Default = peak -> ... byte-compare skips". Grounding shows the test skips based on `_HAS_BASELINE` and `OTR_REGRESSION_RUNTIME`, not the normalization mode. Fix: Correct the expectation. If the regression runtime is active, the test will *run and pass* (not skip) because the default peak output is byte-identical to the v1.5 baseline.
3. [Section 3 & 5] Guaranteed CI breakage on re-baseline. Step 6 says "Commit BOTH fixtures with the flip". If Chunk 1 leaves the codebase default as `peak`, and the operator commits RMS fixtures without updating the CI runner's environment, CI will generate peak audio, compare it to the RMS baseline, and fail. Fix: Explicitly define "the flip" to include either changing the code's fallback default to `rms` or injecting `OTR_SEGMENT_LOUDNORM=rms` into the CI pipeline's ComfyUI startup configuration in the exact same commit as the fixtures.

SHOULD-FIX:
1. [Section 4] Calibration math ignores the master makeup drop. The grounding notes `_master_loudness` defaults to 0.0 under RMS, dropping the previous 4.0 dB makeup. If you measure the raw segment RMS and set the target to exactly that, the final episode will be 4.0 dB quieter than before. Fix: The calibration target must account for the lost master gain (e.g., `target = measured_raw_rms + 4.0`).
2. [Callsites :726] [ASSUMPTION] The plan assumes SFX stays peak because line 726 passes `target_peak=0.85`. This assumes `_normalize_clip` prioritizes explicit kwargs over the global `OTR_SEGMENT_LOUDNORM=rms` env var. Fix: Verify/enforce that `_normalize_clip` is implemented to force peak mode and ignore the RMS env var whenever `target_peak` is explicitly provided.

OPTIONAL / NICE-TO-HAVE:
- [Section 3] Step 4 (Eyeball) should explicitly check for audio clipping (values exceeding [-1.0, 1.0] before the master bus), as RMS normalization on highly dynamic TTS lines can easily push peaks past 0 dBFS if the target RMS is set too high. 

CUT THESE (over-engineering):
- None. The plan is relatively minimal, but the procedural gaps in execution are fatal.