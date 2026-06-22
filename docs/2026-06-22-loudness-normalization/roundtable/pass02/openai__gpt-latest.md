<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: build-ready as-is? yes-with-fixes. The core change is plausible, but the plan has test/behavior contradictions and underspecified env/master behavior that can produce build-time test failures or accidental legacy-output drift.

MUST-FIX BEFORE BUILD:
1. [Tests / New functions 2] Contradiction: “NaN/Inf/empty -> returned unchanged; always float32” conflicts with “returned unchanged” and with current legacy behavior. Grounding shows `_normalize_clip([])` would call `.max()` and raise, and non-finite legacy output is not “unchanged” in all cases. Concrete fix: state that NaN/Inf/empty handling is for `_loudness_normalize_clip` / `_level_dialogue_clip` only when `OTR_SEGMENT_LOUDNORM=rms`; peak mode must preserve `_normalize_clip` behavior for byte parity. Define “unchanged” as “sample values not gain-adjusted, returned as `np.float32` ndarray via `astype(np.float32, copy=False)`,” not object identity.

2. [New functions 2 / 3] Return annotations are wrong: `_loudness_normalize_clip(...) -> np.float32` and `_level_dialogue_clip(...) -> np.float32` describe a scalar dtype, but the functions return an audio array. Concrete fix: use `-> np.ndarray` or omit annotations. Tests should assert `out.dtype == np.float32`, not scalar return type.

3. [Edits / `_master_loudness`] The conditional default for makeup requires distinguishing “env var unset” from “set explicitly.” Current grounded code uses `os.environ.get("OTR_MASTER_MAKEUP_DB", "4.0")`, which cannot implement the planned unset-only RMS default if copied naïvely. Concrete fix:
   ```
   if makeup_db is None:
       mode = os.environ.get("OTR_SEGMENT_LOUDNORM", "peak").strip().lower()
       if "OTR_MASTER_MAKEUP_DB" in os.environ:
           makeup_db = float(os.environ["OTR_MASTER_MAKEUP_DB"])
       elif mode == "rms":
           makeup_db = 0.0
       else:
           makeup_db = 4.0
   ```
   Then apply the existing `0..12` clamp. Explicit `makeup_db` function argument must continue to win over all env behavior.

4. [New functions 3] Env override parsing is underspecified. “Parsed safe/clamped” does not say what happens for invalid strings, NaN, reversed signs, or out-of-range values. Concrete fix: define one helper and ranges before coding, e.g. catch `ValueError`/`TypeError`, reject non-finite, then clamp. Also force `MAX_BOOST_DB >= 0`, `MAX_CUT_DB <= 0`, `PEAK_CEILING > 0`. Without this, tests for env behavior will be implementation-dependent.

5. [New functions 3 / Edits] Hidden import dependency: `_level_dialogue_clip` reads `os.environ`, but the grounding only shows `os` imported locally inside `_master_loudness`; it does not prove a module-level `os` import exists. Concrete fix: add `import os` inside `_level_dialogue_clip` or verify and rely on an existing module-level import. [ASSUMPTION] based on grounding excerpt not showing file imports.

6. [Open] “Reconfirm no `_normalize_clip` callers beyond the four found” is listed as verify-at-build, but it affects whether the edit is complete. Concrete fix: before build, run a source grep for `_normalize_clip(` and either update every intended dialogue call or explicitly document remaining callers as intentionally unchanged.

SHOULD-FIX:
1. [New functions 2] RMS over the whole clip will include leading/trailing TTS silence. [ASSUMPTION] If generated clips contain variable padding, quiet/silent padding will cause over-boosting of otherwise normal speech. Concrete fix: either explicitly accept whole-clip RMS as intentional, or compute RMS over active samples above a small gate/percentile mask while still leaving fully gated room tone unchanged.

2. [New functions 2] State that peak safety can override `max_cut_db`. With `g = min(g, peak_ceiling / peak)`, very high-peak inputs may be attenuated by more than `max_cut_db`. Concrete fix: add one sentence and test this edge case, or rename `max_cut_db` to “RMS-desired max cut before peak safety.”

3. [New functions 2] Use float64 for RMS math, then cast output to float32. Concrete fix:
   ```
   x = np.asarray(clip_np)
   rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
   ```
   or equivalent. This avoids overflow/precision surprises while preserving float32 output.

4. [New functions 3 / Edits] Use identical mode parsing everywhere. Concrete fix: centralize:
   ```
   def _segment_loudnorm_mode():
       return os.environ.get("OTR_SEGMENT_LOUDNORM", "peak").strip().lower()
   ```
   Then `_level_dialogue_clip` and `_master_loudness` cannot diverge on whitespace/case handling.

5. [Tests] Add a regression test or static check that the SFX call path still uses `_normalize_clip(..., target_peak=0.85)`. The plan says SFX/music/themes untouched, but the proposed tests only cover dialogue/master behavior.

6. [Edits] Update `_master_loudness` docstring too. Grounding docstring currently says default env else `4.0`; after this change it will be false in RMS mode when `OTR_MASTER_MAKEUP_DB` is unset.

7. [Tests] Ensure tests isolate env mutations with `monkeypatch` or equivalent. Otherwise `OTR_SEGMENT_LOUDNORM=rms` can leak into legacy byte-identical tests and break default-output assumptions.

OPTIONAL / NICE-TO-HAVE:
- Add a tiny test for invalid `OTR_SEGMENT_LOUDNORM` values proving they fall back to legacy peak behavior.
- Add a test for `peak_ceiling` already exceeded before gain, proving the output peak is reduced to the ceiling.
- Log the selected segment loudnorm mode and master makeup used once per render, if there is already a render log path. [ASSUMPTION] Existing logging structure beyond excerpt not verified.

CUT THESE (over-engineering):
1. [Tests] Cut “repeat run byte-identical” for the pure helper functions. The proposed math is deterministic numpy with no RNG; the more valuable byte-identical coverage is peak-mode parity and golden fixture preservation.

2. [Tests] Cut or narrow “assert no CUDA.” The new segment functions are numpy-only by construction, and grounded `_master_loudness` already uses torch CPU tensors without any CUDA call shown. A brittle CUDA assertion is less useful than simply avoiding any new torch/CUDA code in the implementation.

3. [Mode + re-baseline] Do not put the “validate perceived loudness across 2-3 episodes” requirement into build-blocking automated tests. Keep it as release validation/manual acceptance; the build should cover deterministic numeric invariants only.