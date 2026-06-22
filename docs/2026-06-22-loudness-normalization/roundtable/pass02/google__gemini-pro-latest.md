<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: yes-with-fixes. The DSP logic is sound and respects the frozen spine, but contains a math domain trap and inefficient/unsafe environment parsing in the render loop.

MUST-FIX BEFORE BUILD:
1. [New functions] Math domain error in `_loudness_normalize_clip`. The plan specifies `rms_dbfs = 20*log10(rms)`. If `rms` evaluates to exactly `0.0` (which can happen despite the `peak < 1e-6` check due to `float32` precision limits on squaring tiny arrays), `np.log10(0)` raises a `RuntimeWarning: divide by zero` and returns `-inf`, breaking downstream math.
   - **Fix**: Clamp RMS before the log: `rms_dbfs = 20 * np.log10(max(rms, 1e-10))`.
2. [Edits] Fragile environment parsing in `_master_loudness`. The plan dictates defaulting to 0.0 when `OTR_MASTER_MAKEUP_DB` is "UNSET". `os.environ.get` returns `None` if unset, but users frequently pass empty strings (`""`) in CI/CD pipelines. Calling `float("")` raises a `ValueError` and will crash the final `assemble()` step.
   - **Fix**: Implement safe parsing:
     ```python
     raw_makeup = os.environ.get("OTR_MASTER_MAKEUP_DB", "").strip()
     if raw_makeup:
         makeup_db = float(raw_makeup)
     else:
         mode = os.environ.get("OTR_SEGMENT_LOUDNORM", "peak").lower()
         makeup_db = 0.0 if mode == "rms" else 4.0
     ```

SHOULD-FIX:
1. [New functions] Per-clip environment parsing overhead. `_level_dialogue_clip` is slated to read and parse 6 environment variables (mode + 5 float parameters) *every time* a dialogue clip is processed. This is inefficient and risks a mid-render `ValueError` if a user provides a malformed float string.
   - **Fix**: Parse, validate, clamp, and cache these environment variables at the module level or on first use, rather than inside the `_level_dialogue_clip` execution path.
2. [New functions] Array math functions. The plan specifies `sqrt(mean(clip**2))`. 
   - **Fix**: Ensure this is implemented strictly as `np.sqrt(np.mean(clip_np**2))`. Python's native `math.sqrt` will raise a `TypeError` if passed a numpy scalar/array.

OPTIONAL / NICE-TO-HAVE:
- [Tests] Add a test case specifically for `OTR_MASTER_MAKEUP_DB=""` (empty string) to ensure it correctly falls back to the mode-based defaults (0.0 or 4.0) without throwing a `ValueError`.
- [Open] Calibrating the `-20.0` dBFS placeholder against the current engine output is highly recommended. Bark's default output often sits closer to -16 dBFS to -18 dBFS; a -20 dBFS target might result in an unexpectedly aggressive cut.

CUT THESE (over-engineering):
- None. The plan is strictly scoped and avoids unnecessary CUDA/PyTorch dependencies for the per-segment DSP.