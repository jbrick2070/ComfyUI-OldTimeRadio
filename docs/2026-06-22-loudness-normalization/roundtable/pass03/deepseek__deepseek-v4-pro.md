<!-- requested_model: deepseek/deepseek-v4-pro | resolved_model: deepseek/deepseek-v4-pro-20260423 -->

VERDICT: build-ready as-is? **yes-with-fixes** – the plan is conceptually sound but has concrete gaps that will cause problems at build or operator time unless addressed.

MUST-FIX BEFORE BUILD:
1. [2. Byte-identical gate mechanics] The plan assumes the new peak-mode path is byte-identical to the current baseline, but it does not require a verification step. The byte-compare test skips by default, so a code change could silently break the peak path. **Fix:** Add a requirement in Chunk 1 to *temporarily* set `OTR_REGRESSION_RUNTIME=1` and run the byte-identical test against the existing baseline with `OTR_SEGMENT_LOUDNORM=peak` before landing. Confirm it passes. Without this, the “green” commit is not actually proven green.

2. [4. Calibration procedure] The procedure is underspecified—it does not explain how to measure the RMS dBFS of *dialogue-only* segments from an episode. The “master WAV” contains mixed SFX/music; “per-clip dialogue segments” are not described as being saved externally. The operator cannot reliably compute the target without concrete instructions. **Fix:** Provide a specific script that loads the individual segment audio (e.g., from the pipeline’s intermediate clip store) or a method to extract dialogue segments using the render log timestamps. If the clips are not readily accessible, the plan must address that.

3. [3. Re-baseline procedure] Step 2 says to set `OTR_CAST_SEED=42` and `OTR_STYLE_SEED=42`, but the baseline capture uses `FIXED_SEEDS` from `tests/_run_baseline.py`. It is unclear whether those env vars are respected or overridden. If the function hardcodes different seeds, the re-baseline will produce a different output. **Fix:** Verify the actual behaviour of `_run_baseline` with respect to those env vars. Either adjust the procedure to use the seeds that are actually effective, or confirm that `FIXED_SEEDS` already equals 42 and the env vars are redundant. Document the outcome.

4. [6. Open / verify-at-build] The plan states “verify when capturing” that `OTR_SEGMENT_LOUDNORM` flows through, but this is a manual dependency that could be missed. The re-baseline procedure completely depends on the capture script reading the env var. **Fix:** Either add a small pre-flight check in the capture script (e.g., a log line) or an explicit step in the re-baseline procedure to inspect the process output to confirm the env was used. Without this, the operator could capture a peak-mode baseline unknowingly.

5. [5. Commit chunking] The `_master_loudness` tweak description is ambiguous: “default it to 0.0 when OTR_SEGMENT_LOUDNORM=='rms' and the env is unset”. It does not clarify whether an explicitly set `OTR_MASTER_MAKEUP_DB` should be honoured in rms mode, or if the default is always zero regardless of the env. **Fix:** Define the rule precisely: if `OTR_MASTER_MAKEUP_DB` is explicitly set, use it; otherwise, if `OTR_SEGMENT_LOUDNORM=='rms'`, default to 0.0; else default to 4.0. This prevents a future operator from inadvertently breaking the gain staging.

SHOULD-FIX:
6. [4. Calibration procedure] The plan mentions a “-20 placeholder” for `OTR_SEGMENT_TARGET_RMS_DBFS` but does not state the actual default value anywhere. The calibration procedure should note the placeholder default so the operator knows what they are overriding. Minor.

7. [3. Re-baseline procedure] The “eyeball” step is vague; it could be improved by specifying a concrete loudness measurement tool (e.g., `ffmpeg` loudnorm filter) so the A/B check is not purely subjective.

OPTIONAL / NICE-TO-HAVE:
- Provide a small helper script that measures the current dialogue RMS dBFS from an episode’s intermediate clips, to be used in the calibration procedure.
- Add a dedicated test that exercises the `_normalize_clip` function with all env combinations (peak, rms, with/without target) to catch regressions early.

CUT THESE (over-engineering):
- None. The plan is appropriately scoped; the commit chunking and operator-gated steps are necessary for safe rollout.

[ASSUMPTION] The re-baseline capture script in `tests/_run_baseline` respects environment variables for segment loudnorm; the grounding excerpt does not show the internals. The plan’s calibration/rms flip depends on this. Verify against the actual `_run_baseline.py` source.