"""Regression: the indextts2(22050) + bark-fallback(24000) mixed-rate crash.

Reproduces the live failure where a cast that mixes reference-clip'd (indextts2,
22050 Hz) and ref-less (bark fallback, 24000 Hz) characters made
``pack_audio_batch`` abort with "mixed sample rates [22050, 24000]", and pins the
fix: ``resample_audio`` downsamples the fallback clip to the primary rate before
packing. Pure synthetic AUDIO tensors -- no model loads, no GPU, headless.

See docs/2026-06-05-voice-mixed-rate/roundtable/pass01_plan.md.
"""
import os
import sys

import pytest
import torch

# scipy is a test dep for resample_audio; absent in base Python 3.11 (lives in
# the ComfyUI venv). Skip the whole module cleanly rather than failing on import.
pytest.importorskip("scipy")

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from nodes._otr_audio_utils import resample_audio, canonical_audio  # noqa: E402
from nodes._otr_audio_engines import pack_audio_batch  # noqa: E402

_INDEX_SR = 22050  # eng_indextts2.sample_rate
_BARK_SR = 24000   # eng_bark.sample_rate


def _tone(sr, dur_s=0.25, freq=220.0):
    """A deterministic mono AUDIO dict {"waveform":[1,1,T], "sample_rate"}."""
    t = torch.arange(int(sr * dur_s), dtype=torch.float32) / float(sr)
    wave = torch.sin(2 * torch.pi * freq * t)
    return {"waveform": wave.reshape(1, 1, -1), "sample_rate": int(sr)}


# --------------------------------------------------------------------------- #
# resample_audio helper
# --------------------------------------------------------------------------- #
def test_resample_downsample_rate_and_shape():
    out = resample_audio(_tone(_BARK_SR), _INDEX_SR)
    assert out["sample_rate"] == _INDEX_SR
    assert out["waveform"].dim() == 3 and out["waveform"].shape[:2] == (1, 1)
    # 0.25 s at 22050 ~= 5512 samples; allow a couple for the polyphase edge.
    assert abs(int(out["waveform"].shape[-1]) - int(0.25 * _INDEX_SR)) <= 2
    assert out["waveform"].dtype == torch.float32


def test_resample_same_rate_is_noop():
    src = _tone(_INDEX_SR)
    out = resample_audio(src, _INDEX_SR)
    assert out["sample_rate"] == _INDEX_SR
    assert torch.equal(canonical_audio(src)["waveform"], out["waveform"])


def test_resample_is_deterministic():
    a = resample_audio(_tone(_BARK_SR), _INDEX_SR)
    b = resample_audio(_tone(_BARK_SR), _INDEX_SR)
    assert torch.equal(a["waveform"], b["waveform"])


def test_resample_empty_buffer_restamps_rate():
    empty = {"waveform": torch.zeros(1, 1, 0, dtype=torch.float32), "sample_rate": _BARK_SR}
    out = resample_audio(empty, _INDEX_SR)
    assert out["sample_rate"] == _INDEX_SR
    assert int(out["waveform"].shape[-1]) == 0


# --------------------------------------------------------------------------- #
# the bug, and the fix, through pack_audio_batch
# --------------------------------------------------------------------------- #
def test_pack_raises_on_mixed_rates_reproduces_bug():
    clips = [_tone(_INDEX_SR), _tone(_BARK_SR)]  # one index line + one bark fallback
    with pytest.raises(ValueError, match="mixed sample rates"):
        pack_audio_batch(clips, sample_rate=_INDEX_SR, mono=True)


def test_resample_fallback_then_pack_succeeds():
    index_clip = _tone(_INDEX_SR)
    bark_clip = _tone(_BARK_SR)
    # The fix: the fallback clip is resampled to the primary rate before packing.
    bark_fixed = resample_audio(bark_clip, _INDEX_SR)
    packed = pack_audio_batch([index_clip, bark_fixed], sample_rate=_INDEX_SR, mono=True)
    assert packed["sample_rate"] == _INDEX_SR
    assert int(packed["waveform"].shape[0]) == 2   # B == number of lines
    assert int(packed["waveform"].shape[1]) == 1   # mono


def test_all_fallback_episode_packs_at_primary_rate():
    # indextts2 selected but every line fell back to bark: still packs at 22050.
    clips = [resample_audio(_tone(_BARK_SR), _INDEX_SR) for _ in range(3)]
    packed = pack_audio_batch(clips, sample_rate=_INDEX_SR, mono=True)
    assert packed["sample_rate"] == _INDEX_SR
    assert int(packed["waveform"].shape[0]) == 3


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
