"""Tests for the audio-engine adapter base + the AUDIO-batch packer (piece 4)."""
import pytest
import torch

from nodes._otr_audio_engines import (
    AudioEngineAdapter,
    engine_supports_external_generator,
    pack_audio_batch,
)
from nodes._otr_resolved_request import assert_audio_batch_contract


def _audio(vals, sr=24000):
    wf = torch.tensor(vals, dtype=torch.float32).view(1, 1, -1)
    return {"waveform": wf, "sample_rate": sr}


def test_empty_pack_is_canonical_empty_batch():
    out = pack_audio_batch([])
    assert tuple(out["waveform"].shape) == (1, 1, 0)
    assert out["sample_rate"] == 24000
    assert_audio_batch_contract(out, where="test")


def test_empty_pack_respects_requested_sample_rate():
    out = pack_audio_batch([], sample_rate=16000)
    assert out["sample_rate"] == 16000
    assert tuple(out["waveform"].shape) == (1, 1, 0)


def test_single_line_pack_preserves_values():
    out = pack_audio_batch([_audio([1.0, 2.0, 3.0])])
    assert tuple(out["waveform"].shape) == (1, 1, 3)
    assert out["waveform"][0, 0].tolist() == [1.0, 2.0, 3.0]


def test_multi_line_right_pads_to_max_and_keeps_values():
    out = pack_audio_batch([_audio([1.0, 2.0]), _audio([3.0, 4.0, 5.0])])
    wf = out["waveform"]
    assert tuple(wf.shape) == (2, 1, 3)
    assert wf[0, 0].tolist() == [1.0, 2.0, 0.0]  # shorter line zero-padded right
    assert wf[1, 0].tolist() == [3.0, 4.0, 5.0]
    assert_audio_batch_contract(out, where="test")


def test_mixed_sample_rate_raises():
    with pytest.raises(ValueError):
        pack_audio_batch([_audio([1.0], sr=24000), _audio([2.0], sr=16000)])


def test_stereo_input_downmixed_to_mono_by_default():
    stereo = {
        "waveform": torch.tensor([[[1.0, 1.0], [3.0, 3.0]]], dtype=torch.float32),
        "sample_rate": 24000,
    }  # [B=1, C=2, T=2]
    out = pack_audio_batch([stereo])
    assert tuple(out["waveform"].shape) == (1, 1, 2)
    assert out["waveform"][0, 0].tolist() == [2.0, 2.0]  # mean across channels


def test_mono_false_preserves_channels():
    stereo = {"waveform": torch.zeros(1, 2, 4), "sample_rate": 24000}
    out = pack_audio_batch([stereo], mono=False)
    assert tuple(out["waveform"].shape) == (1, 2, 4)


def test_base_adapter_defaults_and_noop_lifecycle():
    class _E(AudioEngineAdapter):
        name = "x_stub"

    e = _E()
    assert e.supports_external_generator is False
    assert engine_supports_external_generator(e) is False
    # load/unload defaults are cheap no-ops and must not raise.
    e.load()
    e.unload()


def test_external_generator_helper_reads_flag():
    class _E(AudioEngineAdapter):
        supports_external_generator = True

    assert engine_supports_external_generator(_E()) is True

    class _Duck:  # never sets the flag -> reads False
        pass

    assert engine_supports_external_generator(_Duck()) is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
