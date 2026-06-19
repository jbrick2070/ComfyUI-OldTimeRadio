"""Bark runaway-length / trailing-pad fix (caption-fit, 2026-06-18).

CPU coverage for the two pure pieces (no model load):
  - _resolve_min_eos_p: default 0.1, env override, '0' disables, bad -> default;
  - _trim_trailing_silence: trims trailing near-silence, keeps speech + a short
    decay, no-ops on empty / all-silence / speech-only.
"""
from __future__ import annotations

import numpy as np

from nodes import _otr_bark_lib as bl


def test_resolve_min_eos_p_default(monkeypatch):
    monkeypatch.delenv("OTR_BARK_MIN_EOS_P", raising=False)
    assert bl._resolve_min_eos_p() == 0.1


def test_resolve_min_eos_p_env_override(monkeypatch):
    monkeypatch.setenv("OTR_BARK_MIN_EOS_P", "0.05")
    assert bl._resolve_min_eos_p() == 0.05


def test_resolve_min_eos_p_zero_disables(monkeypatch):
    monkeypatch.setenv("OTR_BARK_MIN_EOS_P", "0")
    assert bl._resolve_min_eos_p() == 0.0


def test_resolve_min_eos_p_bad_falls_back(monkeypatch):
    monkeypatch.setenv("OTR_BARK_MIN_EOS_P", "not-a-number")
    assert bl._resolve_min_eos_p() == 0.1


def _speech(sr, secs, amp=0.5, freq=180.0):
    t = np.arange(int(sr * secs)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_trim_removes_trailing_silence():
    sr = 24000
    speech = _speech(sr, 1.0)
    clip = np.concatenate([speech, np.zeros(sr, dtype=np.float32)])  # +1s silence
    out = bl._trim_trailing_silence(clip, sr)
    # speech (1.0s) + min_keep (0.15s) kept; the ~1s trailing silence is gone.
    assert len(speech) <= len(out) < len(clip)
    assert len(out) / sr < 1.4


def test_trim_keeps_speech_only_clip():
    sr = 24000
    speech = _speech(sr, 1.0)
    out = bl._trim_trailing_silence(speech, sr)
    assert abs(len(out) - len(speech)) <= int(0.2 * sr)  # essentially unchanged


def test_trim_allsilence_unchanged():
    sr = 24000
    sil = np.zeros(sr, dtype=np.float32)
    out = bl._trim_trailing_silence(sil, sr)
    assert len(out) == len(sil)               # never zero a silent placeholder


def test_trim_empty_unchanged():
    out = bl._trim_trailing_silence(np.zeros(0, dtype=np.float32), 24000)
    assert len(out) == 0
