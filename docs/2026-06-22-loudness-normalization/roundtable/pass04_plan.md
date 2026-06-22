# FINAL build-ready spec -- per-segment RMS loudness normalization (R4 convergence check)

Consolidates R1 (approach) + R2 (coding) + R3 (wiring), all panel-hardened + code-grounded. R4 question:
any NEW must-fix? All in `nodes/scene_sequencer.py` unless noted.

## Decision
Even out PERCEIVED dialogue loudness shot-to-shot via per-segment RMS leveling (not LUFS: the seam has no
sample-rate, LUFS crashes short lines + pulls scipy). Dialogue-only. Gated behind
`OTR_SEGMENT_LOUDNORM=peak (DEFAULT) | rms`; default keeps `test_audio_byte_identical` green (ships dark).

## New code (numpy-only, CPU, deterministic, no new dep; os+math already imported at module scope)
- `_segment_loudnorm_mode()` -> `os.environ.get("OTR_SEGMENT_LOUDNORM","peak").strip().lower()`.
- `_parse_env_float(name, default, lo, hi, sign)` -> empty/junk/non-finite -> default; sign-enforce
  (boost>=0, cut<=0, ceiling>0); clamp. Cache the 5 params once (`_segment_params()`; env read once,
  set at server boot).
- `_rms(x)` -> float64 `sqrt(mean(x^2))`; empty/non-finite -> 0.0.
- `_loudness_normalize_clip(clip, target_rms_dbfs=-20.0, max_boost_db=9.0, max_cut_db=-12.0,
  gate_dbfs=-50.0, peak_ceiling=0.95) -> np.ndarray`: empty/non-finite -> unchanged float32;
  `peak<1e-6` -> unchanged; active-sample RMS (mask above gate, fallback whole-clip); `rms<=0` ->
  unchanged; `rms_dbfs=20*log10(max(rms,1e-10))` under `np.errstate`; `rms_dbfs<=gate` -> unchanged;
  `g=clamp(target-rms_dbfs, max_cut, max_boost)`; `g=min(g, peak_ceiling/peak)` (peak-safety MAY exceed
  max_cut -- intentional); return `(clip*g).astype(float32)`.
- `_level_dialogue_clip(clip) -> np.ndarray`: rms mode -> `_loudness_normalize_clip(**_segment_params())`;
  else `_normalize_clip(clip)` (legacy peak).

## Edits
- Replace `_normalize_clip(segment_np)` at :747/:753/:775 with `_level_dialogue_clip(segment_np)`.
  Leave :726 SFX (`_normalize_clip(...,0.85)`) untouched -- it never calls the router, and
  `_normalize_clip` never reads the env -> peak by construction.
- `_master_loudness` makeup default: explicit `OTR_MASTER_MAKEUP_DB` wins (empty-string-safe parse);
  else `0.0` if mode==rms; else `4.0`. Update docstring.
- `_normalize_clip` docstring: add "sample-peak" qualifier.

## Wiring (R3)
- NO workflow-JSON change (env-gated; edits inside the already-wired `OTR_EpisodeAssembler`).
- Peak parity PROVEN: in-suite test `_level_dialogue_clip`(peak) == `_normalize_clip` byte-for-byte; plus
  one runtime `OTR_REGRESSION_RUNTIME=1` peak run vs `baseline_v1.5` before landing.
- Re-baseline (rms flip, operator-gated): env on the SERVER process (restart) -- `OTR_SEGMENT_LOUDNORM=rms`,
  `OTR_SEGMENT_TARGET_RMS_DBFS=<measured+4>`, `OTR_MASTER_MAKEUP_DB=0`, `OTR_CAST_SEED/STYLE_SEED=42`;
  preflight log asserts effective values; `--capture-baseline`; A/B (ffmpeg loudnorm); 2-3 episode listen.
- Fixture/CI: keep `baseline_v1.5` as the PEAK golden; do NOT commit rms fixtures while default=peak;
  byte-compare skips when `OTR_SEGMENT_LOUDNORM` not in {unset, peak}. rms-as-default = a future separate
  commit (flip default + commit rms golden together).
- Calibration: commit `tools/measure_dialogue_rms.py` (dialogue-only, pre-master, active-window, float
  full-scale, mono); target = measured + 4.0 dB (compensate the dropped master makeup).

## Tests (`tests/test_segment_loudnorm.py`; deterministic, monkeypatch env isolation)
peak-parity (== `_normalize_clip`); SFX-stays-peak; silence (<1e-6) + room tone (<=gate) unchanged;
empty/NaN/Inf unchanged float32; quiet boosted <=max_boost AND peak<=ceiling; loud attenuated peak<=ceiling;
env parse (empty/junk/reversed/out-of-range) safe; determinism with non-default clamped overrides;
`_master_loudness` rms-unset->0.0 / rms-empty->0.0 / explicit-honored / peak->4.0.

## Commit chunking
Chunk 1 (one green commit + push to v2.0-alpha): code + call-site edits + master tweak + tests + README
env note. Default peak -> suite + Bug Bible green. Operator-gated (separate): calibrate -> server rms
restart -> re-baseline -> multi-episode listen.

## Invariants
Frozen spine held by default-peak gate; CPU/numpy-only (I-11, no CUDA, no new dep); SFX/music/themes
untouched; peak values LINEAR (0..1); deterministic; UTF-8 no BOM; SFW; prod/main gated.
