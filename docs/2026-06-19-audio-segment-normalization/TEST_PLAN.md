# Audio Segment Normalization -- Test Plan

Date: 2026-06-19
Branch: `v2.0-alpha`
Grounded against HEAD `9935161`.

The canary is the disabled-path byte-identity. Every test below exists to protect
it or to prove the enabled path does what the spec asks **without** touching the
frozen master. Parity test is written and green FIRST, before any feature code.

---

## 0. Test inventory at a glance

| # | Test | Layer | Gate | Runs in CI? | Proves |
|---|------|-------|------|-------------|--------|
| 1 | `test_segnorm_disabled_is_identity` | pure | OFF | yes | helper is a strict no-op when OFF |
| 2 | `test_segnorm_disabled_concat_bitwise_equal` | pure | OFF | yes | concat input unchanged when OFF |
| 3 | `test_audio_byte_identical_to_baseline` (existing) | runtime | OFF | no (GPU) | full master WAV SHA matches frozen baseline |
| 4 | `test_segnorm_enabled_no_clipping` | pure | ON | yes | peak stays < 1.0, no new clipping |
| 5 | `test_segnorm_enabled_reduces_spread` | pure | ON | yes | segment loudness spread shrinks |
| 6 | `test_segnorm_measures_each_segment` | pure | ON | yes | per-segment measurement vector present |
| 7 | `test_segnorm_meta_stamp_present` | pure | OFF+ON | yes | stamp written in both states |
| 8 | `test_segnorm_bark_arrays_untouched` | pure | ON | yes | Bark/source arrays not re-rendered/mutated in place |
| 9 | `test_episode_loudness_report_shape` | pure | ON | yes | report carries rms+peak (lufs if avail) |
| 10 | Bark no-regression (existing suite) | suite | n/a | yes | `test_bark_*` stays green |

"pure" = no GPU/torch-CUDA, runs under the standard `pytest -q -p no:cacheprovider`
conftest (`OTR_TEST_MODE=1`, `CUDA_VISIBLE_DEVICES=''`). "runtime" = operator/GPU,
`OTR_REGRESSION_RUNTIME=1`.

---

## 1. Parity tests -- the canary (written + green BEFORE feature code)

### 1a. `test_segnorm_disabled_is_identity` (pure, CI)

With `OTR_ENABLE_SEGMENT_NORM` unset, `_apply_segment_normalization(segs, sr, log)`
must return the **same list object** carrying the **same array objects**:

```python
out, meta = _apply_segment_normalization(segs, 48000, [])
assert out is segs                         # no new list
assert all(a is b for a, b in zip(out, segs))   # no new arrays
assert meta["enabled"] is False
```

Rationale: identity at the object level is the strongest possible proof that
`np.concatenate(all_segments)` at `scene_sequencer.py:829` sees unchanged input.

### 1b. `test_segnorm_disabled_concat_bitwise_equal` (pure, CI)

Belt-and-suspenders on the *bytes*, not just object identity: build a fixed
synthetic `all_segments` (seeded), concatenate it directly, then run it through the
OFF helper and concatenate the result; assert `np.array_equal` AND identical
`.tobytes()`:

```python
ref = np.concatenate(segs)
out, _ = _apply_segment_normalization(segs, 48000, [])
got = np.concatenate(out)
assert np.array_equal(ref, got)
assert ref.tobytes() == got.tobytes()
assert ref.dtype == got.dtype           # no float32->float64 drift
```

### 1c. `test_audio_byte_identical_to_baseline` (runtime, GPU, EXISTING)

`tests/test_audio_byte_identical.py:190`. Run with `OTR_REGRESSION_RUNTIME=1`,
`OTR_CAST_SEED=42`, `OTR_STYLE_SEED=42`, and `OTR_ENABLE_SEGMENT_NORM` **unset**.
The SHA-256 of the produced master must equal the stored baseline hash
(`tests/fixtures/baseline_v1.5.sha256`). This is the final, full-pipeline proof.
Run UNPROMPTED after every Phase 1 chunk.

> Baseline note: the existing baseline fixture predates this feature. The disabled
> path must match it with **no re-capture** -- if it doesn't, the disabled path is
> not byte-identical and the change is reverted (canary stop condition). We do NOT
> re-capture the baseline to make this pass.

---

## 2. Enabled-path tests (no clipping, real normalization, measurement)

### 2a. `test_segnorm_enabled_no_clipping` (pure, CI) -- success criterion

Construct segments with deliberately divergent loudness (one near-silent, one hot
at ~0.95 peak). With the gate ON, assert every output segment's peak is strictly
`< 1.0` and the post-concat `combined` peak is `< 1.0`:

```python
out, meta = _apply_segment_normalization(loud_and_quiet_segs, 48000, [],
                                         enabled=True)
for seg in out:
    assert np.abs(seg).max() < 1.0       # no sample at/over full scale
assert np.abs(np.concatenate(out)).max() < 1.0
```

Also assert no NaN/Inf introduced.

### 2b. `test_segnorm_enabled_reduces_spread` (pure, CI) -- success criterion

The spec's headline goal ("audible reduction in segment-to-segment volume swings").
Measure each segment's loudness (RMS dBFS) before and after; assert the standard
deviation (or max-min spread) across segments **decreases** with the gate ON:

```python
spread_before = stdev([rms_db(s) for s in segs])
out, _ = _apply_segment_normalization(segs, 48000, [], enabled=True)
spread_after = stdev([rms_db(s) for s in out])
assert spread_after < spread_before
```

### 2c. `test_segnorm_measures_each_segment` (pure, CI) -- first-target step 1

Assert the meta carries one measurement per input segment and one gain per segment:

```python
out, meta = _apply_segment_normalization(segs, 48000, [], enabled=True)
assert len(meta["per_segment_db"]) == len(segs)
assert len(meta["before_after_delta_db"]) == len(segs)
```

### 2d. `test_segnorm_bark_arrays_untouched` (pure, CI) -- Bark no-regression at unit level

Prove the stage does not mutate its inputs in place (Bark/source arrays are
reused elsewhere -- e.g. ledger position math). Snapshot input bytes, run ON,
assert inputs unchanged:

```python
before = [s.copy() for s in segs]
_apply_segment_normalization(segs, 48000, [], enabled=True)   # ON
for s, b in zip(segs, before):
    assert s.tobytes() == b.tobytes()    # inputs not mutated; new arrays returned
```

---

## 3. Metadata stamp tests

### 3a. `test_segnorm_meta_stamp_present` (pure, CI) -- both states

The spec requires a stamp "indicating normalization enabled/disabled" in **both**
states. Assert the meta dict has the documented shape (ARCHITECTURE.md §4) when OFF
(`enabled: False, method: "off"`) and when ON (`enabled: True, method: <metric>`,
non-empty vectors). If the stamp is wired through the ledger in Phase 1, add a
ledger-round-trip assertion that `meta["segment_normalization"]` survives
load/save.

---

## 4. Final episode loudness report test

### 4a. `test_episode_loudness_report_shape` (pure, CI) -- first-target step 4

Build a known episode waveform; assert the report carries `rms_dbfs` and
`peak_dbfs` always, and `lufs` **iff** a loudness lib is importable (skip-marked
otherwise, never a hard failure -- "if available"):

```python
rep = episode_loudness_report(episode_waveform, sample_rate)
assert "rms_dbfs" in rep and "peak_dbfs" in rep
assert rep["peak_dbfs"] <= 0.0           # dBFS, full-scale is 0
# LUFS only asserted when the lib is present
```

---

## 5. Bark-no-regression at suite level

### 5a. Existing Bark suite stays green

Run the existing Bark tests UNPROMPTED every chunk -- e.g.
`tests/test_bark_trim.py`, `tests/test_bark_voice_stage_temps.py`. Segment
normalization is downstream of Bark and must not perturb them. Any red here =
the design crossed into the Bark pipeline (stop condition: log + ping).

---

## 6. Regression discipline per chunk (non-negotiable, unprompted)

After EVERY Phase 1 code chunk, in this order:

1. `pytest -q -p no:cacheprovider` -- full suite green.
2. Bug Bible regression (`comfyui-custom-node-survival-guide`, relative
   `tests\bug_bible_regression.py`).
3. `tests/test_audio_byte_identical.py` -- structural always; runtime gate when the
   GPU is free (Phase 2). Disabled-path identity tests (1a/1b) green on CPU every
   chunk.
4. Widget/INPUT_TYPES audit only if a widget is ever added (v1 is env-var only, so
   no `widgets_values` change is expected -- if that changes, re-validate the
   workflow JSON per CLAUDE.md §0).

If the audio byte-identity (disabled) breaks: `git reset --hard` uncommitted, ping.

---

## 7. Where the new tests live

- New pure tests: `tests/test_audio_segment_normalization.py` (new file alongside
  the existing audio tests). Keeps the parity-specific cases together and importable
  without GPU.
- The runtime byte-identical gate stays in the existing
  `tests/test_audio_byte_identical.py` -- we extend usage (run it with the gate
  unset) rather than fork it.
