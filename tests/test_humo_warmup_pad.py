"""Regression test for BUG-LOCAL-102: HuMo motion-onset audio pre-pad.

HuMo (audio-to-video lip-sync) holds the reference image static for
its first ~3-6 output frames before motion onset. Without intervention
the first audible phoneme lands during that freeze and the listener
perceives a constant ~150-200 ms audio-leads-lips lag.

Fix: pad each per-line audio with leading silence at HuMo's input,
then trim symmetrically before the on-disk save so timeline math
stays in terms of the original ``dur_s``.

This test exercises the two new helpers
(``_pad_audio_silence_lead`` and ``_trim_audio_lead``) directly.
The module is loaded via ``importlib.util.spec_from_file_location``
to mirror the loading pattern in ``test_batch_humo_render.py``.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
NODE_PATH = REPO_ROOT / "nodes" / "batch_humo_render.py"


def _load():
    spec = importlib.util.spec_from_file_location(
        "batch_humo_render_warmup_pad_test", str(NODE_PATH),
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def m():
    return _load()


def _make_audio(samples: int, channels: int = 1, sr: int = 48000,
                value: float = 0.5) -> dict:
    """Build a ComfyUI AUDIO dict with deterministic non-zero content."""
    waveform = torch.full((1, channels, samples), value, dtype=torch.float32)
    return {"waveform": waveform, "sample_rate": sr}


# ---------------------------------------------------------------------------
# _pad_audio_silence_lead
# ---------------------------------------------------------------------------

class TestPadAudioSilenceLead:
    """``_pad_audio_silence_lead(audio_dict, pad_samples)`` prepends
    ``pad_samples`` zero-valued samples to the waveform's time axis."""

    def test_pad_adds_correct_sample_count(self, m):
        audio = _make_audio(samples=48000, sr=48000)
        padded = m._pad_audio_silence_lead(audio, pad_samples=9600)
        assert padded["waveform"].shape[-1] == 48000 + 9600

    def test_pad_preserves_sample_rate(self, m):
        audio = _make_audio(samples=24000, sr=24000)
        padded = m._pad_audio_silence_lead(audio, pad_samples=4800)
        assert padded["sample_rate"] == 24000

    def test_pad_preserves_channel_count(self, m):
        audio = _make_audio(samples=48000, channels=2, sr=48000)
        padded = m._pad_audio_silence_lead(audio, pad_samples=9600)
        # Shape should be [B=1, C=2, samples]
        assert padded["waveform"].shape[0] == 1
        assert padded["waveform"].shape[1] == 2
        assert padded["waveform"].shape[2] == 48000 + 9600

    def test_pad_leading_samples_are_zero(self, m):
        audio = _make_audio(samples=1000, value=0.5)
        padded = m._pad_audio_silence_lead(audio, pad_samples=200)
        # First 200 samples must be zero (silence)
        leading = padded["waveform"][..., :200]
        assert torch.all(leading == 0).item()

    def test_pad_trailing_samples_unchanged(self, m):
        audio = _make_audio(samples=1000, value=0.5)
        padded = m._pad_audio_silence_lead(audio, pad_samples=200)
        # The original signal sits AFTER the silence
        original = padded["waveform"][..., 200:]
        assert torch.all(original == 0.5).item()

    def test_pad_zero_returns_unchanged_dict(self, m):
        audio = _make_audio(samples=1000)
        padded = m._pad_audio_silence_lead(audio, pad_samples=0)
        # pad_samples=0 should be a no-op
        assert padded["waveform"].shape == audio["waveform"].shape
        assert padded["sample_rate"] == audio["sample_rate"]

    def test_pad_negative_returns_unchanged(self, m):
        # Defensive: negative pad should not throw, just no-op
        audio = _make_audio(samples=1000)
        padded = m._pad_audio_silence_lead(audio, pad_samples=-50)
        assert padded["waveform"].shape == audio["waveform"].shape

    def test_pad_returns_contiguous_tensor(self, m):
        # Downstream torch ops (concat, ffmpeg conversion) assume
        # contiguous memory. A non-contiguous result would silently
        # mis-render frames.
        audio = _make_audio(samples=2000)
        padded = m._pad_audio_silence_lead(audio, pad_samples=400)
        assert padded["waveform"].is_contiguous()


# ---------------------------------------------------------------------------
# _trim_audio_lead
# ---------------------------------------------------------------------------

class TestTrimAudioLead:
    """``_trim_audio_lead(audio_dict, pad_samples)`` is the inverse of
    ``_pad_audio_silence_lead``: drops the leading ``pad_samples`` from
    the waveform's time axis. Used at HuMo clip save time so the
    on-disk mp4 has dialogue starting at sample 0."""

    def test_trim_removes_correct_sample_count(self, m):
        audio = _make_audio(samples=57600, sr=48000)
        trimmed = m._trim_audio_lead(audio, pad_samples=9600)
        assert trimmed["waveform"].shape[-1] == 57600 - 9600

    def test_trim_preserves_sample_rate(self, m):
        audio = _make_audio(samples=10000, sr=16000)
        trimmed = m._trim_audio_lead(audio, pad_samples=1600)
        assert trimmed["sample_rate"] == 16000

    def test_trim_zero_returns_unchanged(self, m):
        audio = _make_audio(samples=1000)
        trimmed = m._trim_audio_lead(audio, pad_samples=0)
        assert trimmed["waveform"].shape == audio["waveform"].shape

    def test_trim_negative_returns_unchanged(self, m):
        audio = _make_audio(samples=1000)
        trimmed = m._trim_audio_lead(audio, pad_samples=-10)
        assert trimmed["waveform"].shape == audio["waveform"].shape

    def test_trim_when_pad_exceeds_length_degrades_safely(self, m):
        # Edge case: pad_samples > waveform length. Caller should never
        # trigger this, but the helper must not crash.
        audio = _make_audio(samples=100)
        trimmed = m._trim_audio_lead(audio, pad_samples=200)
        # Helper falls back to a 1-sample slice rather than empty / negative
        assert trimmed["waveform"].shape[-1] >= 1

    def test_trim_returns_contiguous_tensor(self, m):
        audio = _make_audio(samples=2000)
        trimmed = m._trim_audio_lead(audio, pad_samples=400)
        assert trimmed["waveform"].is_contiguous()


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------

class TestPadTrimRoundTrip:
    """The pad-then-trim cycle is what BatchHumoRender does at the
    save boundary. The original waveform must come back identically."""

    @pytest.mark.parametrize("samples, pad_samples, sr", [
        (48000, 9600, 48000),    # 1.0 s original + 200 ms pad @ 48 kHz
        (24000, 4800, 24000),    # 1.0 s original + 200 ms pad @ 24 kHz
        (16000, 1600, 16000),    # 1.0 s original + 100 ms pad @ 16 kHz
        (170000, 9600, 48000),   # ~3.5 s line + 200 ms pad
        (336000, 9600, 48000),   # 7.0 s line (humo cap) + 200 ms pad
    ])
    def test_pad_then_trim_returns_original_length(
        self, m, samples, pad_samples, sr,
    ):
        audio = _make_audio(samples=samples, sr=sr, value=0.3)
        padded = m._pad_audio_silence_lead(audio, pad_samples=pad_samples)
        trimmed = m._trim_audio_lead(padded, pad_samples=pad_samples)
        assert trimmed["waveform"].shape[-1] == samples

    def test_pad_then_trim_recovers_original_values(self, m):
        # Make sure the trimmed payload is bit-identical to the input
        # (audio integrity is a CLAUDE.md C7 contract).
        audio = _make_audio(samples=1000, value=0.42)
        padded = m._pad_audio_silence_lead(audio, pad_samples=200)
        trimmed = m._trim_audio_lead(padded, pad_samples=200)
        assert torch.equal(trimmed["waveform"], audio["waveform"])

    def test_round_trip_preserves_multichannel_layout(self, m):
        audio = _make_audio(samples=5000, channels=2, value=0.7)
        padded = m._pad_audio_silence_lead(audio, pad_samples=1000)
        trimmed = m._trim_audio_lead(padded, pad_samples=1000)
        assert trimmed["waveform"].shape == audio["waveform"].shape
        assert torch.equal(trimmed["waveform"], audio["waveform"])
