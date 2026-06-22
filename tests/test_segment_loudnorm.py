"""Per-segment RMS loudness normalization (scene_sequencer).

RMS leveling is the BAKED-IN DEFAULT; OTR_SEGMENT_LOUDNORM=peak is the legacy
escape hatch. These tests are deterministic + CPU/numpy-only (no CUDA).
"""

import importlib

import numpy as np
import pytest

ss = importlib.import_module("nodes.scene_sequencer")


def _sine(amp, n=2000, freq=20):
    t = np.arange(n, dtype=np.float64)
    return (amp * np.sin(2 * np.pi * freq * t / n)).astype(np.float32)


def _dbfs(x):
    r = float(np.sqrt(np.mean(np.square(np.asarray(x, dtype=np.float64)))))
    return 20.0 * np.log10(max(r, 1e-12))


# --- mode + default ---------------------------------------------------------

def test_default_mode_is_rms(monkeypatch):
    monkeypatch.delenv("OTR_SEGMENT_LOUDNORM", raising=False)
    assert ss._segment_loudnorm_mode() == "rms"


def test_peak_override(monkeypatch):
    monkeypatch.setenv("OTR_SEGMENT_LOUDNORM", "PEAK")  # case/space tolerant
    assert ss._segment_loudnorm_mode() == "peak"


def test_peak_mode_byte_identical_to_legacy(monkeypatch):
    monkeypatch.setenv("OTR_SEGMENT_LOUDNORM", "peak")
    x = _sine(0.3)
    out = ss._level_dialogue_clip(x)
    legacy = ss._normalize_clip(x)
    assert np.array_equal(out, legacy)
    assert out.dtype == np.float32


def test_sfx_path_ignores_env(monkeypatch):
    # SFX calls _normalize_clip directly; it never reads the env -> always peak.
    monkeypatch.setenv("OTR_SEGMENT_LOUDNORM", "rms")
    x = _sine(0.2)
    out = ss._normalize_clip(x, target_peak=0.85)
    assert abs(float(np.abs(out).max()) - 0.85) < 1e-4


# --- _rms / _env_float ------------------------------------------------------

def test_rms_guards():
    assert ss._rms(np.array([], dtype=np.float32)) == 0.0
    assert ss._rms(np.array([np.nan, 1.0], dtype=np.float32)) == 0.0
    assert ss._rms(np.array([np.inf], dtype=np.float32)) == 0.0
    assert abs(ss._rms(np.array([1.0, -1.0], dtype=np.float32)) - 1.0) < 1e-9


def test_env_float_parsing(monkeypatch):
    monkeypatch.setenv("X", "")
    assert ss._env_float("X", -16.0) == -16.0          # empty -> default
    monkeypatch.setenv("X", "junk")
    assert ss._env_float("X", -16.0) == -16.0          # junk -> default
    monkeypatch.setenv("X", "999")
    assert ss._env_float("X", -16.0, lo=-60.0, hi=0.0) == 0.0   # clamp hi
    monkeypatch.setenv("X", "-999")
    assert ss._env_float("X", -16.0, lo=-60.0, hi=0.0) == -60.0  # clamp lo
    monkeypatch.setenv("X", "nan")
    assert ss._env_float("X", -16.0) == -16.0          # non-finite -> default


# --- _loudness_normalize_clip behavior --------------------------------------

P = dict(target_rms_dbfs=-16.0, max_boost_db=9.0, max_cut_db=-12.0,
         gate_dbfs=-50.0, peak_ceiling=0.95)


def test_digital_silence_unchanged():
    x = np.zeros(1000, dtype=np.float32)
    out = ss._loudness_normalize_clip(x, **P)
    assert np.array_equal(out, x) and out.dtype == np.float32


def test_room_tone_below_gate_unchanged():
    x = _sine(0.001)  # ~ -57 dBFS, below the -50 gate
    out = ss._loudness_normalize_clip(x, **P)
    assert np.array_equal(out, x.astype(np.float32))


def test_nan_inf_empty_unchanged_float32():
    for x in (np.array([], dtype=np.float32),
              np.array([np.nan, 0.1], dtype=np.float32),
              np.array([np.inf, 0.1], dtype=np.float32)):
        out = ss._loudness_normalize_clip(x, **P)
        assert out.dtype == np.float32 and out.shape == x.shape


def test_quiet_clip_boosted_capped_and_peak_safe():
    x = _sine(0.02)  # ~ -37 dBFS rms, wants > +9 dB -> capped at +9
    out = ss._loudness_normalize_clip(x, **P)
    gain_db = _dbfs(out) - _dbfs(x)
    assert 8.9 < gain_db < 9.1                       # boost clamp held
    assert float(np.abs(out).max()) <= P["peak_ceiling"] + 1e-6


def test_loud_clip_attenuated_toward_target():
    x = _sine(0.5)  # ~ -9 dBFS rms, above target -> attenuate
    out = ss._loudness_normalize_clip(x, **P)
    assert _dbfs(out) < _dbfs(x)
    assert abs(_dbfs(out) - (-16.0)) < 1.0           # lands near target
    assert float(np.abs(out).max()) <= P["peak_ceiling"] + 1e-6


def test_peak_safety_overrides_boost():
    # low active-RMS body + a few high peaks: boost wanted, but peak-safety caps.
    x = np.full(1000, 0.1, dtype=np.float32)
    x[:5] = 0.9
    out = ss._loudness_normalize_clip(x, **P)
    assert float(np.abs(out).max()) <= P["peak_ceiling"] + 1e-6


def test_deterministic():
    x = _sine(0.05)
    a = ss._loudness_normalize_clip(x, **P)
    b = ss._loudness_normalize_clip(x, **P)
    assert np.array_equal(a, b)
