"""Cloud stereo music is levelled without clipping a channel (audit B15).

canonicalize_audio measured loudness on the mono mean and applied the gain
the leveler chose to both original channels. A wide stereo mix has channel
peaks well above its mean's peak, so the 0.95 ceiling did not hold for L or
R and the pcm_s16le write clipped them. UTF-8, no BOM, SFW.
"""
from __future__ import annotations

import numpy as np
import pytest

sf = pytest.importorskip("soundfile")

from nodes._otr_shared import cloud_media_canonical as cmc  # noqa: E402

SR = 48000


def _canon(tmp_path, left, right):
    t = np.arange(SR * 2, dtype=np.float32) / SR
    tone = np.sin(2 * np.pi * 220 * t).astype(np.float32)
    src = tmp_path / "provider.wav"
    sf.write(str(src), np.stack([left * tone, right * tone], axis=1), SR,
             subtype="FLOAT")
    if not cmc.resolve_ffmpeg():
        pytest.skip("ffmpeg not on this box")
    asset = cmc.canonicalize_audio(
        {"path": str(src), "content_type": "audio/wav"},
        {"sample_rate": SR, "stereo_policy": "stereo",
         "out_path": str(tmp_path / "out.canon.wav")})
    data, _ = sf.read(str(asset.path), dtype="float32", always_2d=True)
    return data


def test_a_wide_stereo_pair_stays_under_the_ceiling(tmp_path):
    data = _canon(tmp_path, 0.9, -0.8)
    assert float(np.abs(data).max()) <= 0.951
    assert int((np.abs(data) >= 0.9999).sum()) == 0


def test_a_centred_mix_still_reaches_its_loudness_target(tmp_path):
    """The cap only bites when a channel would cross the ceiling; a quiet
    centred mix is still boosted toward the RMS target as before."""
    data = _canon(tmp_path, 0.05, 0.05)
    assert float(np.abs(data).max()) > 0.1
