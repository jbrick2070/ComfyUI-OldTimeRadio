"""Tests for the [B, C, T] canonical audio helpers."""
import torch

from nodes._otr_audio_utils import canonical_audio, mono_safe


def test_canonical_promotes_1d():
    out = canonical_audio(torch.zeros(100))
    assert out["waveform"].shape == (1, 1, 100)


def test_canonical_promotes_2d_stereo():
    out = canonical_audio(torch.zeros(2, 100))
    assert out["waveform"].shape == (1, 2, 100)


def test_canonical_passthrough_3d_and_sr():
    out = canonical_audio({"waveform": torch.zeros(1, 1, 50), "sample_rate": 48000})
    assert out["waveform"].shape == (1, 1, 50)
    assert out["sample_rate"] == 48000


def test_mono_safe_downmixes_stereo():
    out = mono_safe({"waveform": torch.ones(1, 2, 100), "sample_rate": 24000})
    assert out["waveform"].shape == (1, 1, 100)


def test_mono_safe_keeps_mono_byte_path():
    wf = torch.arange(100, dtype=torch.float32).reshape(1, 1, 100)
    out = mono_safe({"waveform": wf, "sample_rate": 24000})
    assert out["waveform"].shape == (1, 1, 100)
    assert torch.allclose(out["waveform"], wf)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
