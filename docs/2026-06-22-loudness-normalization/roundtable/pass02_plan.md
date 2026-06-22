# R2 HARDENED -- coding spec: per-segment RMS loudness normalization

Folds the R2 panel must-fixes (see pass02_judgment.md). All in `nodes/scene_sequencer.py` unless noted.
Approach is R1-converged (RMS, dialogue-only, peak-safe, env-mode-gated, default = legacy peak).

## Grounded call sites (verified)
- `_normalize_clip` def :93. Callers: **:726** SFX (`target_peak=0.85`) -> UNCHANGED; **:747** announcer,
  **:753** pre-rendered TTS, **:775** inline-Bark -> the three DIALOGUE sites get the new router.
- Clips mono float32 (`.squeeze()` :567). Master `_master_loudness` :109 applied once in `assemble()`
  :1122; makeup default env `OTR_MASTER_MAKEUP_DB` (line 131, `get(...,"4.0")` -- can't see "unset").
- VERIFY-AT-BUILD: a module-level `import os` (grounding only showed a local import in `_master_loudness`).

## New code (numpy-only, CPU, deterministic, no new dep)

```
_SEG_PARAMS_CACHE = None   # parsed once; env is set at server boot

def _segment_loudnorm_mode():
    return os.environ.get("OTR_SEGMENT_LOUDNORM", "peak").strip().lower()

def _parse_env_float(name, default, lo=None, hi=None, sign=None):
    raw = os.environ.get(name, "").strip()
    try:
        v = float(raw) if raw else float(default)
    except (TypeError, ValueError):
        v = float(default)
    if not math.isfinite(v): v = float(default)
    if sign == "pos": v = max(v, 0.0)          # boost >= 0
    if sign == "neg": v = min(v, 0.0)          # cut <= 0
    if lo is not None: v = max(v, lo)
    if hi is not None: v = min(v, hi)
    return v

def _segment_params():            # lazy cache (env read once)
    global _SEG_PARAMS_CACHE
    if _SEG_PARAMS_CACHE is None:
        _SEG_PARAMS_CACHE = dict(
            target_rms_dbfs=_parse_env_float("OTR_SEGMENT_TARGET_RMS_DBFS", -20.0, -60.0, 0.0),
            max_boost_db   =_parse_env_float("OTR_SEGMENT_MAX_BOOST_DB",      9.0, sign="pos"),
            max_cut_db     =_parse_env_float("OTR_SEGMENT_MAX_CUT_DB",      -12.0, sign="neg"),
            gate_dbfs      =_parse_env_float("OTR_SEGMENT_GATE_DBFS",       -50.0),
            peak_ceiling   =_parse_env_float("OTR_SEGMENT_PEAK_CEILING",     0.95, 0.0, 1.0),
        )
    return _SEG_PARAMS_CACHE

def _rms(clip_np):                 # float64 math; empty/non-finite -> 0.0
    x = np.asarray(clip_np, dtype=np.float64)
    if x.size == 0 or not np.all(np.isfinite(x)): return 0.0
    return float(np.sqrt(np.mean(np.square(x))))

def _loudness_normalize_clip(clip_np, target_rms_dbfs=-20.0, max_boost_db=9.0,
                             max_cut_db=-12.0, gate_dbfs=-50.0, peak_ceiling=0.95) -> np.ndarray:
    x = np.asarray(clip_np)
    if x.size == 0 or not np.all(np.isfinite(x)):
        return x.astype(np.float32)                          # unchanged (values), float32
    peak = float(np.abs(x).max())
    if peak < 1e-6: return x.astype(np.float32)              # digital-silence guard (kept)
    # active-sample RMS so leading/trailing TTS padding doesn't over-boost speech;
    # fallback to whole-clip if too few active samples. (_trim_trailing_silence already trims the tail.)
    active = x[np.abs(x) > 10 ** (gate_dbfs / 20.0)]
    rms = _rms(active if active.size >= max(1, x.size // 50) else x)
    if rms <= 0.0: return x.astype(np.float32)               # log10 guard
    with np.errstate(divide="ignore", invalid="ignore"):
        rms_dbfs = 20.0 * math.log10(max(rms, 1e-10))
    if rms_dbfs <= gate_dbfs: return x.astype(np.float32)    # room tone -> no gain (<= treats gate as silence)
    desired_db = min(max(target_rms_dbfs - rms_dbfs, max_cut_db), max_boost_db)
    g = 10.0 ** (desired_db / 20.0)
    g = min(g, peak_ceiling / peak)                          # peak-safety (MAY exceed max_cut -- intentional)
    return (x * g).astype(np.float32)

def _level_dialogue_clip(clip_np) -> np.ndarray:
    if _segment_loudnorm_mode() == "rms":
        return _loudness_normalize_clip(clip_np, **_segment_params())
    return _normalize_clip(clip_np)                          # legacy peak (byte-identical)
```

## Edits
- Replace the THREE dialogue calls at :747/:753/:775 `_normalize_clip(segment_np)` ->
  `_level_dialogue_clip(segment_np)` (one-line comment at each). Leave :726 (SFX) UNCHANGED.
- `_master_loudness` makeup default block (was line ~129-133) -> empty-safe + mode-aware:
  ```
  if makeup_db is None:
      raw = os.environ.get("OTR_MASTER_MAKEUP_DB", "").strip()
      if raw:
          try: makeup_db = float(raw)
          except ValueError: makeup_db = 4.0
      else:
          makeup_db = 0.0 if _segment_loudnorm_mode() == "rms" else 4.0
  ```
  (explicit `makeup_db` arg still wins; keep the 0..12 clamp). Update its docstring.
- Calibration GATE: keep -20 dBFS placeholder; LOUD one-time warn if `rms` mode + uncalibrated default;
  do NOT set `OTR_SEGMENT_LOUDNORM=rms` in production until measured against a real dialogue mix.
- `_normalize_clip` docstring: add "sample-peak" qualifier (trivial; panel says near no-op).
- Ensure module-level `import os` + `import math` exist (verify-at-build).

## Mode + re-baseline
`OTR_SEGMENT_LOUDNORM = peak (DEFAULT) | rms`. Default keeps `test_audio_byte_identical` GREEN (ships
dark). Flip to `rms` = deliberate, operator-gated golden re-baseline (headless render regenerates the
fixture; eyeball diff; validate perceived loudness across 2-3 episodes -- manual acceptance, not a unit test).

## Tests (`tests/test_segment_loudnorm.py`; deterministic, monkeypatch env isolation)
- peak mode: `_level_dialogue_clip` == `_normalize_clip` byte-for-byte (legacy parity).
- SFX regression: :726 path still `_normalize_clip(...,0.85)` (assert unchanged).
- silence (<1e-6) + room tone (<=gate) unchanged; empty/NaN/Inf -> unchanged values, float32.
- quiet clip boosted but <= max_boost AND final peak <= peak_ceiling; loud clip attenuated, peak <= ceiling.
- env overrides parsed/clamped (empty string, junk, reversed sign, out-of-range) -> safe; rms path
  deterministic with non-default clamped overrides set.
- `_master_loudness`: rms+unset -> 0.0; rms+`""` -> 0.0; explicit arg/env honored; peak mode -> 4.0.

## Invariants / cuts
Frozen spine held via default-peak gate; CPU/numpy-only (I-11, no new dep, no CUDA assertion needed by
construction); SFX/music/themes untouched; peak values LINEAR (0..1); UTF-8 no BOM; SFW. NO workflow-JSON
node/widget change (env-gated, like `OTR_MASTER_MAKEUP_DB`) -- R3 confirms this + the re-baseline procedure.
