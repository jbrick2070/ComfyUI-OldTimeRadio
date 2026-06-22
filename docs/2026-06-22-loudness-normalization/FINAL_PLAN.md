# FINAL build-ready spec -- per-segment RMS loudness normalization

4-round roundtable CONVERGED (GPT-5.5 + Gemini-3.1-pro + DeepSeek-v4-pro + Grok-4.3 + Claude grounded
judge; ~$0.52). Goal: even out PERCEIVED dialogue loudness shot-to-shot (the operator's "consistent audio
levels"). All in `nodes/scene_sequencer.py` unless noted. Content/code-only; NO workflow-JSON change.

## Decision
Per-segment RMS leveling (NOT LUFS: the seam has no sample-rate; LUFS crashes short lines + pulls scipy).
Dialogue-only. Gated `OTR_SEGMENT_LOUDNORM = peak (DEFAULT) | rms`; default keeps `test_audio_byte_identical`
green -> ships dark. (`os`+`math` already imported module-level, lines 28-29.)

## New code (numpy-only, CPU, deterministic, no new dep)
```
def _segment_loudnorm_mode():
    return os.environ.get("OTR_SEGMENT_LOUDNORM", "peak").strip().lower()

def _env_float(name, default, lo=None, hi=None):     # read per call (NO cache -> test-safe)
    raw = os.environ.get(name, "").strip()
    try: v = float(raw) if raw else float(default)
    except ValueError: v = float(default)
    if not math.isfinite(v): v = float(default)
    if lo is not None: v = max(v, lo)
    if hi is not None: v = min(v, hi)
    return v

def _rms(x):                                          # float64; NaN/Inf/empty -> 0.0
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0 or not np.all(np.isfinite(x)): return 0.0
    return float(np.sqrt(np.mean(np.square(x))))

def _loudness_normalize_clip(clip, target_rms_dbfs, max_boost_db, max_cut_db, gate_dbfs, peak_ceiling):
    x = np.asarray(clip)
    if x.size == 0 or not np.all(np.isfinite(x)): return np.asarray(clip, dtype=np.float32)
    peak = float(np.abs(x).max())
    if peak < 1e-6: return np.asarray(clip, dtype=np.float32)        # digital silence
    gate_amp = 10.0 ** (gate_dbfs / 20.0)
    active = x[np.abs(x) >= gate_amp]
    rms = _rms(active if active.size else x)
    if rms <= 0.0: return np.asarray(clip, dtype=np.float32)
    rms_dbfs = 20.0 * math.log10(max(rms, 1e-10))
    if rms_dbfs <= gate_dbfs: return np.asarray(clip, dtype=np.float32)   # room tone -> no gain
    gain_db  = min(max(target_rms_dbfs - rms_dbfs, max_cut_db), max_boost_db)
    gain_lin = 10.0 ** (gain_db / 20.0)                               # <-- dB -> LINEAR (R4 must-fix)
    gain_lin = min(gain_lin, peak_ceiling / peak)                     # peak-safety: MAY go below max_cut
    return (x * gain_lin).astype(np.float32)

def _level_dialogue_clip(clip):
    if _segment_loudnorm_mode() == "rms":
        return _loudness_normalize_clip(
            clip,
            target_rms_dbfs=_env_float("OTR_SEGMENT_TARGET_RMS_DBFS", -20.0, -60.0, 0.0),
            max_boost_db   =max(_env_float("OTR_SEGMENT_MAX_BOOST_DB", 9.0), 0.0),
            max_cut_db     =min(_env_float("OTR_SEGMENT_MAX_CUT_DB", -12.0), 0.0),
            gate_dbfs      =_env_float("OTR_SEGMENT_GATE_DBFS", -50.0),
            peak_ceiling   =_env_float("OTR_SEGMENT_PEAK_CEILING", 0.95, 0.0, 1.0),
        )
    return _normalize_clip(clip)                                      # legacy peak (byte-identical)
```

## Edits
- :747 / :753 / :775 `_normalize_clip(segment_np)` -> `_level_dialogue_clip(segment_np)`. :726 SFX
  UNCHANGED (calls `_normalize_clip` directly; it never reads the env -> peak by construction).
- `_master_loudness` makeup default (INLINE, the func reads os.environ directly):
  ```
  if makeup_db is None:
      raw = os.environ.get("OTR_MASTER_MAKEUP_DB", "").strip()
      if raw:
          try: makeup_db = float(raw)
          except ValueError: makeup_db = 0.0 if _segment_loudnorm_mode()=="rms" else 4.0
      else:
          makeup_db = 0.0 if _segment_loudnorm_mode()=="rms" else 4.0
  ```
  (explicit value wins; keep 0..12 clamp; update docstring to name both env vars.)
- `tests/test_audio_byte_identical.py`: add a skip to the byte-compare test --
  `@pytest.mark.skipif(_segment_loudnorm_mode() not in ("","peak"), reason="rms mode has its own golden")`.
- One preflight log line (Chunk 1) at assemble/episode start: effective mode + target + master makeup.
- `_normalize_clip` docstring: add "sample-peak" qualifier (keep the existing text).

## Tests (`tests/test_segment_loudnorm.py`; deterministic, monkeypatch env isolation)
peak-parity (`_level_dialogue_clip` peak == `_normalize_clip` byte-for-byte, finite non-empty input);
SFX-stays-peak; silence (<1e-6) + room tone (<=gate) unchanged; empty/NaN/Inf -> unchanged float32;
quiet boosted <= max_boost AND final peak <= ceiling; loud attenuated, final peak <= ceiling;
env parse (empty/junk/out-of-range) safe; determinism with non-default clamped overrides;
`_master_loudness` rms-unset->0.0 / rms-empty->0.0 / explicit-honored / peak->4.0.

## Operator-gated rollout (NOT in the code chunk)
1. `tools/measure_dialogue_rms.py` (commit; shares the gate/RMS helper): dialogue-only, pre-master,
   active-window, float full-scale, mono -> mean RMS dBFS. Set `OTR_SEGMENT_TARGET_RMS_DBFS = measured + 4.0`
   (compensate the dropped +4 dB master makeup; STARTING POINT -- the listen test is the validator).
2. RESTART ComfyUI (server-side env): `OTR_SEGMENT_LOUDNORM=rms`, `OTR_SEGMENT_TARGET_RMS_DBFS=<+4>`,
   `OTR_MASTER_MAKEUP_DB=0`, `OTR_CAST_SEED=42`, `OTR_STYLE_SEED=42`. (`_run_baseline` drives the SERVER;
   the capture shell does NOT propagate. `FIXED_SEEDS={}` -> these env seeds ARE the determinism.)
3. `python tests/test_audio_byte_identical.py --capture-baseline` (copy the old WAV aside first); A/B with
   ffmpeg `loudnorm`; listen across 2-3 episodes (no clipping, no pumping, even shot-to-shot).
4. Promote rms to DEFAULT later = a SEPARATE commit (flip the code default + commit the rms golden together).
   Until then keep `baseline_v1.5` as the PEAK golden; do NOT commit rms fixtures.

## Known design note (recorded)
`_master_loudness` peak-normalizes the whole episode to the ceiling (one GLOBAL gain) -> preserves
within-episode shot-to-shot leveling (the goal). Cross-EPISODE absolute matching = a future
integrated-LUFS-master follow-up, out of v1 scope.

## Commit chunking
Chunk 1 (one green commit + push to v2.0-alpha): code + 3 call-site edits + master tweak + byte-compare
skip-guard + preflight log + `tests/test_segment_loudnorm.py` + README env note. Default peak -> full
suite + Bug Bible green; `test_audio_byte_identical` structural green / byte-compare skips.

## Invariants
Frozen spine held by the default-peak gate; CPU/numpy-only (I-11, no CUDA, no new dep); SFX/music/themes
untouched; peak values LINEAR (0..1); deterministic; UTF-8 no BOM; SFW; prod/main gated.
