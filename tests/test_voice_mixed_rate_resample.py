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

from nodes._otr_audio_utils import canonical_audio
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


# --------------------------------------------------------------------------- #
# the bug, and the fix, through pack_audio_batch
# --------------------------------------------------------------------------- #
def test_pack_RESAMPLES_the_mixed_rate_cast_instead_of_aborting():
    """The live bug, and the fix this file's own docstring already described.

    RESTORED 2026-09-11. The `resample_audio` helper named in the docstring was
    retired along with the ref-less bark fallback that called it, and this test
    was reduced to pinning the ValueError -- so the file has been contradicting
    its own opening paragraph ever since. `pack_audio_batch` now owns the
    conversion, which is the one place every caller passes through."""
    clips = [_tone(_INDEX_SR), _tone(_BARK_SR)]  # one index line + one bark fallback
    packed = pack_audio_batch(clips, sample_rate=_INDEX_SR, mono=True)

    assert int(packed["sample_rate"]) == _INDEX_SR
    assert int(packed["waveform"].shape[0]) == 2, (
        "a cast line was dropped; losing recorded dialogue is a worse render "
        "than converting it")


def test_the_DOWNSAMPLED_clip_keeps_its_duration():
    """24k -> 22.05k is a real engine pairing here, and it is the direction
    where a non-anti-aliased fallback would show. Duration is the property every
    downstream consumer reads."""
    packed = pack_audio_batch(
        [_tone(_BARK_SR, dur_s=0.25)], sample_rate=_INDEX_SR, mono=True)
    got = int(packed["waveform"].shape[-1])
    expected = int(_INDEX_SR * 0.25)
    assert abs(got - expected) <= 64, (
        "resampled to %d samples, expected ~%d at %d Hz" % (got, expected, _INDEX_SR))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
