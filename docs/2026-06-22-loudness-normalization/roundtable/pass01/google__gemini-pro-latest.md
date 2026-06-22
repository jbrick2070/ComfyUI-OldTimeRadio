<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: no. The plan will crash on short clips, introduces hard clipping vulnerabilities, and proposes an algorithm (LUFS) whose required inputs are missing from the function signature.

MUST-FIX BEFORE BUILD:
1. [Section 4.1 / 4.4] **LUFS Sample Rate & Short Clip Crash**. ITU-R BS.1770 (LUFS) requires K-weighting filters (which require a sample rate) and relies on 400ms gating blocks. `_normalize_clip` (line 93) has no `sample_rate` parameter, and clips under 400ms (e.g., "No.") will throw a `ValueError` in standard LUFS libraries.
   *Fix*: Either change the signature to `def _normalize_clip(clip_np, sr, target_lufs)` and add an explicit fallback (`if len(clip_np) < 0.4 * sr: return peak_normalize(...)`), OR use simple RMS which requires neither sample rate nor minimum lengths.
2. [Section 2] **Unbounded Peak Clipping**. The plan states "apply gain only (do NOT compress)". If you apply a static gain to hit a loudness target on a highly dynamic clip, the new peak can easily exceed 1.0 (0 dBFS). If this `float32` array is cast to `int16` or processed by downstream nodes before `_master_loudness`, it will hard clip.
   *Fix*: Enforce a peak-safe ceiling inside `_normalize_clip` after applying the loudness gain. (e.g., `if new_peak > 0.99: clip_np *= (0.99 / new_peak)`).
3. [Section 2 / Grounding line 109] **Tanh Limiter Discrepancy**. `_master_loudness` peak-normalizes the entire episode to `ceiling` *before* the `tanh` limiter. If clips are loudness-normalized, they will have wildly different peaks. The `tanh` curve will disproportionately squash clips with high crest factors (high peaks) while leaving low-peak clips linear, destroying the relative loudness match you just built.
   *Fix*: If using per-clip loudness normalization, `OTR_MASTER_MAKEUP_DB` must default to `0.0` (disabling the tanh limiter) to act *only* as a transparent peak safety ceiling.

SHOULD-FIX:
1. [Section 4.3] **Noise-floor Gate Threshold**. The current `1e-6` check is for digital silence, not room tone. A genuine quiet take and room hiss will both easily exceed `1e-6`.
   *Fix*: Raise the gate threshold to `-50 dBFS` (approx `0.00316` linear) to safely ignore room tone segments.
2. [Section 4.5] **Target Scope**. SFX and music beds have entirely different crest factors and perceived loudness curves than dialogue. Applying spoken-word LUFS/RMS targets to them will result in massive mix imbalances.
   *Fix*: Apply this *only* to dialogue clips. Music and SFX must retain their authored levels or use a separate, much lower target.

OPTIONAL / NICE-TO-HAVE:
- Use A-weighted RMS instead of flat RMS if you want perceptual accuracy without the full BS.1770 dependency overhead.

CUT THESE (over-engineering):
1. **LUFS / pyloudnorm (Section 4.1)**: Cut it. Spoken-word TTS outputs do not contain sub-bass that needs K-weighting discounting. Simple RMS (`np.sqrt(np.mean(clip_np**2))`) is zero-dependency, sample-rate agnostic, handles 100ms clips flawlessly, and solves the "thin/low" problem 95% as well as LUFS.

[ASSUMPTION]: Assuming `clip_np` in `_normalize_clip` is strictly dialogue and has not yet been mixed with SFX/Music.
[ASSUMPTION]: Assuming `pyloudnorm` or similar was the intended LUFS library, which relies on `scipy.signal` (a massive dependency for a frozen CPU pipeline).