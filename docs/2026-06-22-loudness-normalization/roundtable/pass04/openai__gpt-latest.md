<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. The RMS path as specified has a dB-vs-linear gain bug that will produce wrong audio, and the test/env/cache details are not build-ready.

MUST-FIX BEFORE BUILD:
1. [New code / `_loudness_normalize_clip`] Gain is computed in dB but applied as a linear multiplier: `g=clamp(target-rms_dbfs, ...)`, then `return clip*g`. This is wrong and can invert phase for cuts because `g` may be negative. Concrete fix: compute `gain_db = clamp(...)`; then `gain_lin = 10.0 ** (gain_db / 20.0)`; then apply peak safety with `gain_lin = min(gain_lin, peak_ceiling / peak)`; return `(clip * gain_lin).astype(np.float32)`.

2. [New code / `_loudness_normalize_clip`] The active-sample gate is underspecified. “mask above gate” uses a dBFS value, but the clip is linear full-scale. Concrete fix: define the mask explicitly as something like `gate_amp = 10.0 ** (gate_dbfs / 20.0)` and `active = np.abs(x) >= gate_amp`; use active samples if any, otherwise whole clip.

3. [Tests / peak-parity vs malformed input] The test requirements conflict with the legacy `_normalize_clip` behavior shown in grounding. `_normalize_clip(empty)` will raise on `.max()`, and `_normalize_clip([inf])` can produce `nan`, not “unchanged float32”. But the spec also requires `_level_dialogue_clip` in peak mode to equal `_normalize_clip` byte-for-byte. Concrete fix: scope “empty/NaN/Inf unchanged float32” tests to `_loudness_normalize_clip` / RMS mode only, or deliberately change `_normalize_clip` and narrow peak-parity to finite non-empty clips. Do not require both for the same inputs.

4. [New code / `_segment_params()` + Tests] “Cache the 5 params once” conflicts with “deterministic, monkeypatch env isolation” tests. Once cached, later monkeypatched env values will not be seen. Concrete fix: either remove the cache, or implement it with an explicit reset such as `_segment_params.cache_clear()` and require every env-mutating test fixture to clear it before/after.

5. [Edits / `_master_loudness`] The current grounded code does `float(os.environ.get("OTR_MASTER_MAKEUP_DB", "4.0"))`, so empty or junk env values crash before clamping. The spec only says “empty-string-safe parse” but tests also depend on mode-sensitive defaults. Concrete fix: replace parsing with explicit logic: if env var is unset or `strip()==""`, default to `0.0` when `_segment_loudnorm_mode()=="rms"` else `4.0`; if set but junk/non-finite, use the same default or document a different safe fallback; clamp finite explicit values to `0.0..12.0`.

6. [New code / `_segment_params()`] The spec says “Cache the 5 params” but names only `OTR_SEGMENT_TARGET_RMS_DBFS` in the re-baseline section. The env names and bounds for max boost, max cut, gate, and peak ceiling are missing, so tests and implementation cannot be made unambiguous. Concrete fix: define all five names and exact bounds in the spec, e.g. target/gate dBFS ranges, boost range `>=0`, cut range `<=0`, ceiling linear range `0<ceiling<=1`.

7. [Wiring / Fixture/CI] The existing grounded `test_audio_byte_identical_to_baseline` skips only on missing baseline or missing `OTR_REGRESSION_RUNTIME`; it does not skip when `OTR_SEGMENT_LOUDNORM=rms`. The spec says byte-compare skips in RMS mode, but does not list an edit to `tests/test_audio_byte_identical.py`. Concrete fix: add an actual skip condition there, e.g. skip when `os.environ.get("OTR_SEGMENT_LOUDNORM", "peak").strip().lower()` is not `""`/`"peak"`.

SHOULD-FIX:
1. [Decision / `_segment_loudnorm_mode()`] Unknown mode values silently fall back to peak because `_level_dialogue_clip` uses RMS only for exact `"rms"`. That is safe for bytes but bad for operator diagnosis. Concrete fix: treat unknown values as peak but log/warn once, or normalize through a helper that returns only `"peak"` or `"rms"`.

2. [Edits / `_master_loudness`] Add explicit tests for `OTR_MASTER_MAKEUP_DB` junk and non-finite values, not just empty/unset. The grounded current implementation is fragile here, and operators commonly pass env values as strings.

3. [New code / `_rms`] Specify whether non-finite detection happens before squaring. If `x` contains `inf`, `x*x` remains `inf`; if `nan`, mean becomes `nan`. Concrete fix: first check `np.all(np.isfinite(x))`; if false, return `0.0` or trigger the caller’s unchanged path before RMS math.

4. [New code / dtype handling] “unchanged float32” is ambiguous for non-float32 inputs: casting changes dtype and may copy. Concrete fix: specify `return np.asarray(clip, dtype=np.float32)` for malformed/empty RMS inputs, and adjust tests to compare values rather than object identity.

5. [Calibration] “target = measured + 4.0 dB” depends on the measurement script using exactly the same active-window/gate semantics as `_loudness_normalize_clip`. Concrete fix: share the RMS/gate helper between production and `tools/measure_dialogue_rms.py`, or document that both use the same gate threshold conversion.

6. [Commit chunking] “preflight log asserts effective values” is mentioned only under operator-gated re-baseline, not in Chunk 1. Concrete fix: include the logging code in Chunk 1 if operators are expected to verify mode/target/master makeup at runtime.

OPTIONAL / NICE-TO-HAVE:
- [README env note] Include one copy-paste server-start example for peak/default and RMS calibration mode.
- [Tests] Add a test proving RMS mode does not touch the SFX call path by monkeypatching `_level_dialogue_clip` and verifying the SFX branch still calls `_normalize_clip`, if assembler-level testing is practical. [ASSUMPTION] This depends on testability of the assembler branch not shown in grounding.
- [Runtime logging] Log the effective segment loudnorm mode once per episode or once at module init to avoid silent peak fallback.

CUT THESE (over-engineering):
1. [New code / `_segment_params()`] Cut the parameter cache. Five env reads and float parses per dialogue clip are negligible compared with resampling/TTS/audio assembly, and removing the cache eliminates test isolation hazards. If server-restart semantics are desired, document them operationally rather than enforcing them through process-global cache state.

2. [Wiring / “one runtime peak run vs `baseline_v1.5` before landing”] Keep as a release checklist item, not a code/build requirement. Default peak parity is already covered by the unit parity test plus the existing runtime byte-identical test when `OTR_REGRESSION_RUNTIME=1`; requiring an additional manual run blocks builds on environment availability.