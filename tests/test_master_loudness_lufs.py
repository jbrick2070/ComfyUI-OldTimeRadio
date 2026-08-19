"""The episode master targets LOUDNESS (-14 LUFS), not peak.

WHY THIS CHANGED (2026-08-19, PBUG-20260819-01). The old master applied a fixed
+4 dB makeup into a tanh soft knee and trimmed the PEAK to -1.0 dBFS. Peak is
the wrong control for a platform that normalises by loudness: two files can
share a -1.0 dBFS peak and differ by 10 dB in LUFS. Measured over 8 real
masters spanning two months, that stage delivered a mean of -9.87 LUFS
(std 0.41) -- about 4 dB hotter than YouTube's -14 LUFS target. YouTube
attenuates louder content and NEVER boosts quieter content, so the extra
loudness was discarded at playback while the limiting used to buy it stayed in
the audio.

THE OPERATOR ASKED FOR THIS CHECK BEFORE AGREEING TO ANYTHING: *"if you can
confirm our normalization path is accurate w/ fabel and sonet to agree best
practice for youtiube i agree."* Both lanes agreed independently: target -14.0
LUFS integrated, keep a -1.0 dBTP true-peak safety rail, leave dynamics alone.

THE TRAP THESE TESTS EXIST TO PREVENT. Peak ceiling and delivered loudness are
NOT linearly related in the old algorithm, because it renormalised to the
ceiling BEFORE the tanh saturated. Measured on the real function, an 8 dB
ceiling move produced a 10.3 dB loudness move (-13.26 -> -23.58 LUFS). A blind
fader A/B could not predict its own delivered level -- which is exactly how a
by-ear pick of -9.0 dBFS would have shipped roughly 10 dB under target. Nobody
should tune this stage by ear again.

AND THE PROPERTY THE OPERATOR SPECIFICALLY WORRIED ABOUT: he asked whether
normalisation should be per-clip or at the end, reasoning "i think we need to
start with clip level so the clips balance out". He is right, and that already
exists -- `_level_dialogue_clip` levels every spoken line to -16 dBFS active
RMS on both the announcer and character buses. This stage runs AFTER it and
applies a SINGLE LINEAR GAIN, which cannot change any clip-to-clip ratio.
`test_a_single_linear_gain_preserves_the_clip_balance` pins that.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
import torch

from nodes.scene_sequencer import _MASTER_TARGET_LUFS, _master_loudness


REPO_ROOT = Path(__file__).resolve().parent.parent
SEQ = REPO_ROOT / "nodes" / "scene_sequencer.py"
SR = 48000

pyloudnorm = pytest.importorskip("pyloudnorm")


def _speechlike(seconds=4.0, amp=0.25, sr=SR):
    """A deterministic, speech-ish signal: bursts with gaps, no RNG seeding
    surprises. Peaks well above its own average, like dialogue."""
    n = int(seconds * sr)
    t = np.arange(n, dtype=np.float64) / sr
    tone = 0.6 * np.sin(2 * np.pi * 180 * t) + 0.4 * np.sin(2 * np.pi * 320 * t)
    env = (np.sin(2 * np.pi * 1.7 * t) > -0.2).astype(np.float64)
    sig = (tone * env * amp).astype(np.float32)
    stereo = np.stack([sig, sig])              # (ch, n)
    return torch.from_numpy(stereo).unsqueeze(0)   # (1, ch, n)


def _lufs(wave, sr=SR):
    a = wave.detach().cpu().numpy()
    while a.ndim > 2:
        a = a[0]
    return float(pyloudnorm.Meter(sr).integrated_loudness(
        np.ascontiguousarray(a.T, dtype=np.float64)))


def _peak_db(wave):
    return 20.0 * np.log10(float(wave.abs().max()) + 1e-12)


# --------------------------------------------------------------------------
# It hits the target
# --------------------------------------------------------------------------

def test_the_default_target_is_the_youtube_figure():
    assert _MASTER_TARGET_LUFS == -14.0


@pytest.mark.parametrize("amp", [0.05, 0.15, 0.35, 0.7])
def test_any_input_level_lands_on_the_target(amp):
    """The whole point: delivery level stops depending on what came in."""
    out, info = _master_loudness(_speechlike(amp=amp), ceiling_dbfs=-1.0,
                                 sample_rate=SR)
    assert info["mode"] == "lufs"
    assert abs(_lufs(out) - (-14.0)) < 0.5, info


def test_a_quiet_input_is_brought_UP_not_only_down():
    """Down-only would leave quiet episodes quiet, which is the failure mode
    YouTube will not rescue."""
    out, info = _master_loudness(_speechlike(amp=0.02), ceiling_dbfs=-1.0,
                                 sample_rate=SR)
    assert info["gain_db"] > 0, "a quiet master must be gained UP to target"
    assert abs(_lufs(out) - (-14.0)) < 0.5


def test_the_target_is_overridable_by_env(monkeypatch):
    monkeypatch.setenv("OTR_MASTER_TARGET_LUFS", "-16.0")
    out, info = _master_loudness(_speechlike(), ceiling_dbfs=-1.0, sample_rate=SR)
    assert info["target_lufs"] == -16.0
    assert abs(_lufs(out) - (-16.0)) < 0.5


# --------------------------------------------------------------------------
# The safety rail
# --------------------------------------------------------------------------

def test_the_true_peak_ceiling_is_never_exceeded():
    """A very peaky mix must still leave under the ceiling."""
    wave = _speechlike(amp=0.3)
    wave[:, :, 1000] = 0.98          # a lone spike far above the body
    out, info = _master_loudness(wave, ceiling_dbfs=-1.0, sample_rate=SR)
    assert _peak_db(out) <= -1.0 + 0.01, info


def test_on_ordinary_material_the_limiter_does_not_fire():
    """We now master QUIETER than before, so the rail should be inert -- if it
    starts firing on normal episodes, something upstream got much hotter."""
    _out, info = _master_loudness(_speechlike(amp=0.25), ceiling_dbfs=-1.0,
                                  sample_rate=SR)
    assert info["peak_limited"] is False, info


# --------------------------------------------------------------------------
# The property the operator asked about: clip balance survives
# --------------------------------------------------------------------------

def test_a_single_linear_gain_preserves_the_clip_balance():
    """Per-clip levelling happens upstream; this stage must not disturb it.

    Two segments at deliberately different levels must keep EXACTLY their
    ratio through the master. This is what makes it safe to change delivery
    level without re-balancing the mix.
    """
    a = _speechlike(seconds=2.0, amp=0.30)
    b = _speechlike(seconds=2.0, amp=0.10)
    joined = torch.cat([a, b], dim=-1)

    n = a.shape[-1]
    before = float(joined[:, :, :n].abs().max()) / float(joined[:, :, n:].abs().max())
    out, _info = _master_loudness(joined, ceiling_dbfs=-1.0, sample_rate=SR)
    after = float(out[:, :, :n].abs().max()) / float(out[:, :, n:].abs().max())

    assert abs(before - after) < 1e-5, (
        "the master changed the ratio between two clips (%.6f -> %.6f); it must "
        "apply ONE gain to everything" % (before, after)
    )


# --------------------------------------------------------------------------
# Degrade, never fail
# --------------------------------------------------------------------------

def test_silence_returns_untouched_and_does_not_divide_by_zero():
    quiet = torch.zeros(1, 2, SR)
    out, info = _master_loudness(quiet, ceiling_dbfs=-1.0, sample_rate=SR)
    assert info["mode"] == "silent"
    assert torch.equal(out, quiet)


def test_audio_too_short_to_measure_falls_back_instead_of_raising():
    """pyloudnorm needs a 400 ms block. Shorter input must still render."""
    tiny = _speechlike(seconds=0.2)
    out, info = _master_loudness(tiny, ceiling_dbfs=-1.0, sample_rate=SR)
    assert info["mode"] == "legacy_peak", info
    assert out.shape == tiny.shape


# --------------------------------------------------------------------------
# The call site -- the old one relied on silent defaults, which is how the
# ceiling became invisible in the first place
# --------------------------------------------------------------------------

def _call_node():
    tree = ast.parse(SEQ.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "_master_loudness"):
            return node
    raise AssertionError("no _master_loudness call site found")


def test_the_call_site_passes_ceiling_and_sample_rate_EXPLICITLY():
    kwargs = {kw.arg for kw in _call_node().keywords}
    assert "sample_rate" in kwargs, "the meter cannot work without the rate"
    assert "ceiling_dbfs" in kwargs, (
        "the old call site passed NOTHING and relied on the default, which is "
        "how a -1.0 ceiling became invisible to every reader"
    )
