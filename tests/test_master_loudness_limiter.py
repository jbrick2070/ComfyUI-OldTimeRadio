"""The delivery master's peak rail became a limiter (2026-09-11).

Measured on the published finals: moonlit_deception_20260911_185439 shipped at
-29.4 LUFS against the -14 target, laughter_in_the_shadows_20260911_214126 at
-15.0. Same code, same target; the difference was one clipped music burst at
9.6 s. `_master_loudness` gains to target and then guarded the ceiling with a
WHOLE-FILE scale, so that one transient set the level of the entire 113 s
episode -- dialogue included (operator: "our volume is too low ... may not be
normalizing right"). `_limit_peaks` now attenuates only the samples around a
peak, with a look-ahead attack and a 60 dB/s release, and leaves gain 1.0
everywhere else.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest
import torch

from nodes.scene_sequencer import (_LIMITER_LOOKAHEAD_BLOCKS, _limit_peaks,
                                   _master_loudness)

SR = 48000
pyloudnorm = pytest.importorskip("pyloudnorm")


def _speechlike(seconds=4.0, amp=0.25, sr=SR):
    """The same deterministic speech-ish signal the LUFS tests use."""
    n = int(seconds * sr)
    t = np.arange(n, dtype=np.float64) / sr
    tone = 0.6 * np.sin(2 * np.pi * 180 * t) + 0.4 * np.sin(2 * np.pi * 320 * t)
    env = (np.sin(2 * np.pi * 1.7 * t) > -0.2).astype(np.float64)
    sig = (tone * env * amp).astype(np.float32)
    return torch.from_numpy(np.stack([sig, sig])).unsqueeze(0)   # (1, ch, n)


def _with_burst(wave, at_s=2.0, ms=20, level=0.99, sr=SR):
    """A 1 kHz burst ``ms`` long at ``level``: the moonlit squelch in miniature."""
    wave = wave.clone()
    start, length = int(at_s * sr), int(sr * ms / 1000.0)
    t = np.arange(length, dtype=np.float64) / sr
    burst = torch.from_numpy((level * np.sin(2 * np.pi * 1000 * t)).astype(np.float32))
    wave[:, :, start:start + length] = burst
    return wave


def _lufs(wave, sr=SR):
    a = wave.detach().cpu().numpy()
    while a.ndim > 2:
        a = a[0]
    return float(pyloudnorm.Meter(sr).integrated_loudness(
        np.ascontiguousarray(a.T, dtype=np.float64)))


def _peak_db(wave):
    return 20.0 * np.log10(float(wave.abs().max()) + 1e-12)


# --------------------------------------------------------------------------- #
# the defect, in miniature, and its fix
# --------------------------------------------------------------------------- #
def test_a_20_ms_burst_no_longer_sets_the_level_of_the_whole_episode():
    quiet_with_burst = _with_burst(_speechlike(amp=0.08))
    out, info = _master_loudness(quiet_with_burst, ceiling_dbfs=-1.0, sample_rate=SR)
    assert info["mode"] == "lufs" and info["gain_db"] > 6.0, info
    assert abs(_lufs(out) - (-14.0)) < 1.0, (
        "the whole-file rail used to leave this ~12 dB low: %.2f LUFS" % _lufs(out))
    assert _peak_db(out) <= -1.0 + 0.01, info
    assert info["peak_limited"] is True
    assert info["limiter_max_reduction_db"] > 6.0
    assert info["limiter_engaged_fraction"] < 0.05, "a 20 ms burst is a moment"
    assert info["delivered_lufs"] is not None
    assert abs(info["delivered_lufs"] - (-14.0)) < 1.0


def test_far_from_the_burst_the_gain_is_exactly_the_lufs_gain():
    """The clip-to-clip balance survives wherever the limiter is idle."""
    wave = _with_burst(_speechlike(amp=0.08), at_s=2.0)
    out, info = _master_loudness(wave, ceiling_dbfs=-1.0, sample_rate=SR)
    # the receipt's gain_db is rounded; read the gain off an untouched region
    # and require the SAME gain before the burst and after the release
    # (a 20 ms burst plus a 60 dB/s release is over well inside 0.4 s)
    ratios = []
    for lo, hi in ((0.2, 1.5), (2.6, 3.9)):
        before = wave[:, :, int(lo * SR):int(hi * SR)]
        after = out[:, :, int(lo * SR):int(hi * SR)]
        mask = before.abs() > 1e-3
        ratios.append(after[mask] / before[mask])
    expected = 10.0 ** (info["gain_db"] / 20.0)
    for ratio in ratios:
        assert float(ratio.min()) == pytest.approx(float(ratio.max()), rel=1e-5)
        assert float(ratio.mean()) == pytest.approx(expected, rel=2e-3)
    assert float(ratios[0].mean()) == pytest.approx(float(ratios[1].mean()), rel=1e-5)


def test_on_ordinary_material_the_limiter_is_idle_and_receipts_zero():
    _out, info = _master_loudness(_speechlike(amp=0.25), ceiling_dbfs=-1.0, sample_rate=SR)
    assert info["peak_limited"] is False
    assert info["limiter_max_reduction_db"] == 0.0
    assert info["limiter_engaged_fraction"] == 0.0
    assert info["rail_clipped_samples"] == 0
    assert info["delivered_lufs"] is None, "nothing was taken, nothing to re-measure"


def test_the_limiter_is_deterministic():
    wave = _with_burst(_speechlike(amp=0.08))
    a, _ = _master_loudness(wave, ceiling_dbfs=-1.0, sample_rate=SR)
    b, _ = _master_loudness(wave.clone(), ceiling_dbfs=-1.0, sample_rate=SR)
    assert torch.equal(a, b)


@pytest.mark.parametrize("channels", [1, 2])
def test_mono_and_stereo_shapes_survive_and_stay_under_the_ceiling(channels):
    wave = _with_burst(_speechlike(amp=0.08))[:, :channels, :]
    out, info = _master_loudness(wave, ceiling_dbfs=-1.0, sample_rate=SR)
    assert out.shape == wave.shape and out.dtype == wave.dtype
    assert _peak_db(out) <= -1.0 + 0.01, info


def test_a_peak_in_the_very_first_block_is_caught():
    wave = _speechlike(amp=0.08)
    wave[:, :, 3] = 0.99
    out, info = _master_loudness(wave, ceiling_dbfs=-1.0, sample_rate=SR)
    assert _peak_db(out) <= -1.0 + 0.01, info
    assert info["peak_limited"] is True


def test_the_attack_precedes_the_peak_and_the_release_is_finite():
    """Straight at the limiter: a lone full-scale spike against a low ceiling."""
    n = 2 * SR
    x = np.full((1, 1, n), 0.05, dtype=np.float32)
    spike = SR
    x[0, 0, spike] = 1.0
    ceiling = 0.1                                   # a 20 dB reduction is needed
    out, stats = _limit_peaks(torch.from_numpy(x), ceiling, SR)
    out = out.numpy()[0, 0]
    assert stats["engaged"] and abs(stats["max_reduction_db"] - 20.0) < 0.05
    assert out[spike] <= ceiling * (1 + 1e-6)
    block = int(SR * 0.001)
    ahead = _LIMITER_LOOKAHEAD_BLOCKS * block
    assert out[spike - ahead] < 0.05, "the gain is already down before the peak"
    assert out[spike - 3 * ahead] == pytest.approx(0.05, rel=1e-6), "...but not long before"
    # 20 dB at 60 dB/s recovers in 0.33 s; well before 0.5 s the gain is unity
    assert out[spike + int(0.5 * SR)] == pytest.approx(0.05, rel=1e-4)
    # inside the ramp the gain is monotone: no clicks from a step
    ramp = out[spike - ahead - block:spike - ahead + block]
    assert np.all(np.diff(ramp) <= 1e-9)


def test_a_peak_at_the_end_of_a_block_gets_the_full_reduction_not_the_rail():
    """codex r1: an envelope that releases inside the peak block would let a
    late maximum escape to the hard clip. The block value is held under the
    ramp, so the limiter itself handles it and the rail counts nothing."""
    n = 2 * SR
    block = int(SR * 0.001)
    x = np.full((1, 1, n), 0.05, dtype=np.float32)
    spike = SR + block - 1                          # the last sample of a block
    x[0, 0, spike] = 1.0
    out, stats = _limit_peaks(torch.from_numpy(x), 0.1, SR)
    assert stats["rail_clipped_samples"] == 0, stats
    # float32 output: the sample is the float32 image of 0.1, one ulp over
    assert float(out[0, 0, spike]) <= 0.1 * (1 + 1e-6)


def test_a_non_finite_sample_stays_local_and_never_zeroes_the_tail():
    """Sonnet QA: the cumulative-max release carried one NaN to the end of the
    file and one inf silenced everything after it. The old rail left a bad
    sample alone; so does the limiter now -- NaN asks for no reduction, inf
    for a finite bounded one, and the tail recovers."""
    n = 4 * SR
    x = np.full((1, 1, n), 0.05, dtype=np.float32)
    x[0, 0, SR // 2] = np.nan
    x[0, 0, SR] = np.inf
    out, stats = _limit_peaks(torch.from_numpy(x), 0.1, SR)
    out = out.numpy()[0, 0]
    assert np.isnan(out[SR // 2]), "the bad sample is still the bad sample"
    assert np.isfinite(out[:SR // 2]).all()
    assert np.isfinite(out[SR // 2 + 1:]).all(), "nothing after a NaN is NaN"
    assert out[SR] <= 0.1, "the inf sample is clipped to the ceiling"
    assert out[SR - 1] > 0.0 and out[SR + 1] > 0.0, "neighbours are not zeroed"
    assert out[-1] == pytest.approx(0.05, rel=1e-3), "the tail recovers to unity"
    assert np.isfinite(stats["max_reduction_db"]) and stats["max_reduction_db"] <= 120.0


def test_a_hard_working_limiter_warns_and_never_fails(caplog):
    """A quiet body under a full-scale 20 ms burst needs well over 12 dB of
    reduction at the burst: a MIX problem. The master says so and still
    delivers a file under the ceiling at the target. (A body much quieter
    than this falls under the meter's relative gate, the burst alone then
    defines the measurement, and the gain goes NEGATIVE -- no limiting.)"""
    wave = _with_burst(_speechlike(amp=0.08), at_s=2.0, ms=20, level=0.99)
    with caplog.at_level(logging.WARNING, logger="OTR"):
        out, info = _master_loudness(wave, ceiling_dbfs=-1.0, sample_rate=SR)
    assert _peak_db(out) <= -1.0 + 0.01
    assert info["peak_limited"] is True
    assert info["limiter_max_reduction_db"] > 12.0, info
    assert any("worked HARD" in r.getMessage() for r in caplog.records), info


def test_the_rail_residue_is_receipted_not_hidden():
    wave = _with_burst(_speechlike(amp=0.08))
    _out, info = _master_loudness(wave, ceiling_dbfs=-1.0, sample_rate=SR)
    assert isinstance(info["rail_clipped_samples"], int)
    assert info["rail_clipped_samples"] < 200, (
        "the interpolation residue is a handful of samples, not the burst")
