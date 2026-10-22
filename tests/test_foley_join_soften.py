"""Soft segment joins in the beat foley assembler (2026-10-22).

Operator: "this repetitive cymbal crash". A chained beat's segments each make
their own audio and are joined with a hard cut; a segment that opens on a crash
puts one at every join. ``OTR_FOLEY_JOIN_SOFTEN_MS`` (off by default) blends
the shared frame and ducks the next segment's opening. Off must stay
byte-identical to the shipped hard cut.
"""

from __future__ import annotations

import numpy as np
import pytest

from nodes._otr_video_engines import foley_stems as fs

FPS = 25
RATE = 48000
STEP = RATE // FPS          # 1920 samples a frame


def _stem(tmp_path, name, frames, value, crash=False):
    arr = np.full((2, frames * STEP), value, dtype=np.float32)
    if crash:
        arr[:, STEP:STEP + 2400] = 0.9      # a loud opening just past frame 0
    path = tmp_path / name
    fs.write_pcm16_wav(str(path), arr, RATE)
    return str(path)


def _assemble(tmp_path, monkeypatch, soften):
    if soften is None:
        monkeypatch.delenv(fs.FOLEY_JOIN_SOFTEN_ENV, raising=False)
    else:
        monkeypatch.setenv(fs.FOLEY_JOIN_SOFTEN_ENV, str(soften))
    a = _stem(tmp_path, "a.wav", 97, 0.1)
    b = _stem(tmp_path, "b.wav", 97, 0.1, crash=True)
    out = tmp_path / ("beat_%s.wav" % soften)
    fs.assemble_beat_foley_segments(
        [(a, 0, 97), (b, 1, 60)], str(out), expect_frames=157, fps=FPS)
    return fs.read_pcm16_wav(str(out))[0]


def test_off_is_the_shipped_hard_cut(tmp_path, monkeypatch):
    plain = _assemble(tmp_path, monkeypatch, None)
    zero = _assemble(tmp_path, monkeypatch, 0)
    assert np.array_equal(plain, zero)
    # the crash arrives at full level right at the join
    assert plain[0, 97 * STEP:97 * STEP + 2000].max() > 0.85


def test_on_keeps_the_length_and_takes_the_edge_off_the_join(
        tmp_path, monkeypatch):
    plain = _assemble(tmp_path, monkeypatch, None)
    soft = _assemble(tmp_path, monkeypatch, 300)
    assert soft.shape == plain.shape
    join = 97 * STEP
    assert soft[0, join:join + 2000].max() < 0.5 * plain[0, join:join + 2000].max()
    # well past the duck, the audio is untouched
    assert np.allclose(soft[:, join + RATE:], plain[:, join + RATE:], atol=1e-4)
    # the first segment's body is untouched; only its last frame is blended
    assert np.allclose(soft[:, :join - STEP], plain[:, :join - STEP], atol=1e-4)


@pytest.mark.parametrize("raw,expected", [("", 0), ("abc", 0), ("-5", 0),
                                          ("250", 250)])
def test_the_knob_never_raises(monkeypatch, raw, expected):
    monkeypatch.setenv(fs.FOLEY_JOIN_SOFTEN_ENV, raw)
    assert fs.foley_join_soften_ms() == expected


def test_a_sustained_tone_passes_the_join_without_a_step(tmp_path, monkeypatch):
    """Sonnet QA 2026-10-22: a crossfade ending at unity into a duck starting
    at 0.25 put a 0.75x gain jump -- a click -- at every join."""
    monkeypatch.setenv(fs.FOLEY_JOIN_SOFTEN_ENV, "300")
    n = 97 * STEP
    tone = (0.5 * np.sin(2 * np.pi * 220 * np.arange(n) / RATE)).astype(np.float32)
    paths = []
    for name in ("t1.wav", "t2.wav"):
        p = tmp_path / name
        fs.write_pcm16_wav(str(p), np.vstack([tone, tone]), RATE)
        paths.append(str(p))
    out = tmp_path / "beat.wav"
    receipt = fs.assemble_beat_foley_segments(
        [(paths[0], 0, 97), (paths[1], 1, 60)], str(out), expect_frames=157, fps=FPS)
    y = fs.read_pcm16_wav(str(out))[0][0]
    join = 97 * STEP
    jump = np.abs(np.diff(y[join - STEP - 4:join + 4])).max()
    natural = np.abs(np.diff(tone)).max()
    assert jump < 1.5 * natural
    assert receipt["foley_join_soften_ms"] == 300
