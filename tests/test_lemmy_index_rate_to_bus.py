"""Branch A acceptance gate: Lemmy renders at IndexTTS2's NATIVE rate and the
SceneSequencer resampling to the mixed bus is verified.

WHY THIS GATE EXISTS AND WHY IT IS NEW TODAY. Until 2026-08-10 the qualified
Lemmy route did not exist, so nothing in this pack routed a character to a
22,050 Hz engine on purpose. Every other char-voice engine runs at the bus rate:

    indextts2   22050   <- Lemmy's qualified route, and the ONLY outlier
    chatterbox  24000
    kokoro      24000
    bark        24000

An off-rate clip that reaches the mix WITHOUT being resampled does not crash. It
plays at the wrong speed and the wrong pitch -- roughly 9% slow and about a
semitone and a half flat -- which is the kind of defect that ships, because
nothing red ever appears. So the plan made "succeeds at native Index rate and
SceneSequencer resampling is verified" an explicit Branch A gate, and this file
is that gate as executable coverage rather than a sentence in a document.

Headless and pure: it exercises the real `_resample_audio` and the real profile
resolver. No engine, no model, no GPU.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pytest

os.environ.setdefault("OTR_TEST_MODE", "1")

BUS_SR = 24000
INDEX_SR = 22050


def _tone(freq_hz, seconds, sample_rate):
    """A pure sine. Its FREQUENCY is the thing under test: resampling must move
    the sample count without moving the pitch."""
    t = np.arange(int(seconds * sample_rate), dtype=np.float32) / float(sample_rate)
    return np.sin(2.0 * np.pi * freq_hz * t).astype(np.float32)


def _dominant_freq(clip, sample_rate):
    spectrum = np.abs(np.fft.rfft(clip.astype(np.float64)))
    return float(np.fft.rfftfreq(len(clip), 1.0 / sample_rate)[int(np.argmax(spectrum))])


# ---------------------------------------------------------------------------
# The rates themselves -- if these drift, the gate below is measuring nothing.
# ---------------------------------------------------------------------------
def test_lemmys_route_really_is_the_off_rate_lane():
    from nodes._otr_engine_profiles import require_resolver

    resolver = require_resolver()
    index = resolver.resolve_casting_plan(role="char_voice", engine="indextts2")
    assert int(index.sample_rate) == INDEX_SR, (
        "IndexTTS2 is no longer 22050 Hz; this whole gate needs re-deriving")
    for same_as_bus in ("kokoro", "chatterbox"):
        other = resolver.resolve_casting_plan(role="char_voice", engine=same_as_bus)
        assert int(other.sample_rate) == BUS_SR, same_as_bus


def test_the_qualified_lemmy_route_targets_that_engine():
    """Ties the rate question to the ACTUAL shipped route rather than to a
    hard-coded engine name."""
    from config.cast_pools import LEMMY_VOICE_POLICY as P

    route = P["approved_native_routes"]["indextts2"]
    assert route["qualification_record"]["engine"] == "indextts2"


# ---------------------------------------------------------------------------
# The gate: 22050 -> 24000 preserves pitch and duration.
# ---------------------------------------------------------------------------
def test_resampling_to_the_bus_preserves_DURATION():
    from nodes.scene_sequencer import _resample_audio

    seconds = 1.5
    clip = _tone(220.0, seconds, INDEX_SR)
    out = _resample_audio(clip, INDEX_SR, BUS_SR)

    expected = int(round(len(clip) * BUS_SR / INDEX_SR))
    assert abs(len(out) - expected) <= 2, (len(out), expected)
    # The audible consequence, stated as the thing a listener would notice:
    assert abs(len(out) / BUS_SR - seconds) < 0.01, (
        "Lemmy's line would play at the wrong SPEED in the mix")


def test_resampling_to_the_bus_preserves_PITCH():
    """The defect this catches is silent. An unresampled 22050 clip played at
    24000 is ~8.8% fast and about a semitone and a half sharp -- nothing errors,
    it just is not the voice that was auditioned."""
    from nodes.scene_sequencer import _resample_audio

    freq = 440.0
    clip = _tone(freq, 1.0, INDEX_SR)
    out = _resample_audio(clip, INDEX_SR, BUS_SR)

    assert abs(_dominant_freq(out, BUS_SR) - freq) < 5.0

    # And prove the failure mode is real: reinterpreting the SAME samples at the
    # bus rate without resampling shifts the pitch audibly.
    naive = _dominant_freq(clip, BUS_SR)
    assert naive > freq + 25.0, (
        "the no-resample failure mode is not detectable by this test, so the "
        "test above proves less than it claims")
    cents = 1200.0 * math.log2(naive / freq)
    assert cents > 100.0, cents          # more than a semitone -- clearly audible


def test_a_same_rate_clip_is_returned_untouched():
    """Every non-Lemmy character stays byte-identical: the resampler must be a
    no-op at the bus rate, or this change would perturb the whole cast."""
    from nodes.scene_sequencer import _resample_audio

    clip = _tone(300.0, 0.25, BUS_SR)
    out = _resample_audio(clip, BUS_SR, BUS_SR)
    assert out.dtype == np.float32
    assert np.array_equal(out, clip)


def test_the_resampler_stays_on_CPU_and_returns_float32():
    """Invariant I-11: post-engine DSP is CPU-only so the audio baseline is
    determinism-stable. A float64 or CUDA return would break the mix contract."""
    from nodes.scene_sequencer import _resample_audio

    out = _resample_audio(_tone(200.0, 0.2, INDEX_SR), INDEX_SR, BUS_SR)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float32


@pytest.mark.parametrize("src", [22050, 44100, 16000, 48000])
def test_every_shipped_engine_rate_lands_on_the_bus(src):
    """Not just Lemmy's: dia is 44100 and any future engine may differ again."""
    from nodes.scene_sequencer import _resample_audio

    out = _resample_audio(_tone(330.0, 0.4, src), src, BUS_SR)
    assert abs(len(out) / BUS_SR - 0.4) < 0.01
    assert abs(_dominant_freq(out, BUS_SR) - 330.0) < 6.0
