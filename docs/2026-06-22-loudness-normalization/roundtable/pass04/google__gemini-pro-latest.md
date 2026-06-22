<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: no. The gain calculation fatally mixes dB and linear scales (which will invert phase and blow up amplitude), and the master loudness stage unconditionally peak-normalizes the episode, destroying the cross-episode consistency the RMS feature is trying to achieve.

MUST-FIX BEFORE BUILD:
1. [New code] Gain calculation mixes dB and linear scales. `g=clamp(target-rms_dbfs, max_cut, max_boost)` computes a gain in dB. `g=min(g, peak_ceiling/peak)` compares this dB value to a linear ratio. Finally, `return (clip*g)` multiplies the audio by the dB value (e.g., a -6dB cut multiplies the signal by -6.0, inverting phase and boosting by ~15.5dB).
   Fix: Convert `g` to linear before the peak ceiling check: `g_db = clamp(...)`, `g_linear = 10.0 ** (g_db / 20.0)`, `g_final = min(g_linear, peak_ceiling / peak)`, then return `(clip * g_final).astype(np.float32)`.
2. [Edits] `_master_loudness` unconditionally peak-normalizes the entire episode to `ceiling_dbfs` before applying makeup gain. In RMS mode with `makeup_db=0.0`, an episode with no SFX will be boosted massively (e.g., +10dB) to hit the ceiling, while an episode with loud SFX will barely be boosted. This completely destroys the cross-episode dialogue consistency RMS was supposed to solve.
   Fix: In `_master_loudness`, conditionally bypass the auto-gain. If `_segment_loudnorm_mode() == 'rms'` and `makeup_db == 0.0`, skip `waveform = waveform * (ceiling / peak)` and instead just clamp/limit the waveform to `ceiling` to prevent clipping.
3. [New code] Caching `_segment_params()` at server boot breaks test isolation. `monkeypatch.setenv` in `tests/test_segment_loudnorm.py` will have no effect because the module evaluates the env vars at import time, causing tests to fail.
   Fix: Remove the boot-time cache. Read `os.environ` dynamically inside `_segment_params()` on every call.
4. [Edits] `_master_loudness` empty-string parse will crash. `float(os.environ.get("OTR_MASTER_MAKEUP_DB", "4.0"))` raises `ValueError` if the env var is set to `""`.
   Fix: Use `val = os.environ.get("OTR_MASTER_MAKEUP_DB", "").strip()`; `makeup_db = float(val) if val else (0.0 if _segment_loudnorm_mode() == 'rms' else 4.0)`.
5. [Fixture/CI] The byte-compare test does not actually skip for RMS mode. The spec states it skips, but `tests/test_audio_byte_identical.