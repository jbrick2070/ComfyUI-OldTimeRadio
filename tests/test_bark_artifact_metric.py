"""Bark artifact QA -- deterministic high-band edge gate metric.

A simple, deterministic regression catch: high-band (>4 kHz) energy fraction
at a clip's head/tail. Speech edges are low-band; the squeal/whine artifact
B1 prevents is almost all high-band. Synthetic tones make this fully
deterministic and CPU-testable (no model, no GPU).

UTF-8, no BOM, SFW.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes._otr_bark_lib import (  # noqa: E402
    flag_high_band_artifact,
    high_band_edge_ratio,
)

SR = 24000


def _tone(freq, dur_s=0.5, sr=SR):
    t = np.arange(int(dur_s * sr)) / sr
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class TestHighBandMetric:
    def test_low_band_speechlike_tone_not_flagged(self):
        clip = _tone(220.0)  # well below 4 kHz, like voiced speech
        flagged, ratio = flag_high_band_artifact(clip, SR)
        assert ratio < 0.1
        assert flagged is False

    def test_high_band_squeal_flagged(self):
        clip = _tone(8000.0)  # > 4 kHz throughout, like the squeal
        flagged, ratio = flag_high_band_artifact(clip, SR)
        assert ratio > 0.9
        assert flagged is True

    def test_squeal_only_at_head_is_detected(self):
        body = _tone(220.0, dur_s=0.5)
        head = _tone(9000.0, dur_s=0.15)
        clip = np.concatenate([head, body]).astype(np.float32)
        flagged, ratio = flag_high_band_artifact(clip, SR)
        assert flagged is True, f"head squeal missed (ratio={ratio})"

    def test_squeal_only_at_tail_is_detected(self):
        body = _tone(220.0, dur_s=0.5)
        tail = _tone(9000.0, dur_s=0.15)
        clip = np.concatenate([body, tail]).astype(np.float32)
        flagged, _ = flag_high_band_artifact(clip, SR)
        assert flagged is True

    def test_silence_not_flagged(self):
        clip = np.zeros(SR, dtype=np.float32)
        flagged, ratio = flag_high_band_artifact(clip, SR)
        assert ratio == 0.0
        assert flagged is False

    def test_empty_and_degenerate_safe(self):
        assert high_band_edge_ratio(np.zeros(0, dtype=np.float32), SR) == 0.0
        assert high_band_edge_ratio(_tone(220.0), 0) == 0.0

    def test_threshold_is_respected(self):
        clip = _tone(8000.0)
        assert flag_high_band_artifact(clip, SR, threshold=0.99)[0] is True
        # an impossibly high threshold never flags
        assert flag_high_band_artifact(clip, SR, threshold=1.01)[0] is False

    def test_deterministic(self):
        clip = _tone(8000.0)
        assert high_band_edge_ratio(clip, SR) == high_band_edge_ratio(clip, SR)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
