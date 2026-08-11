"""Branch A acceptance gate: Lemmy renders at IndexTTS2's NATIVE rate and the
SceneSequencer resampling to the mixed bus is verified.

CORRECTED 2026-08-10, SAME DAY IT WAS WRITTEN. The first version of this file
asserted a 24,000 Hz bus and framed IndexTTS2 as "the ONLY outlier" because every
other char-voice engine is 24000. Both claims were wrong, and a Sonnet 5 final
review caught them. The real mixed bus is **48,000 Hz** -- `scene_sequencer.py`
sets `sample_rate = 48000  # standardize output` unconditionally, and the publish
encode is `-ar 48000` (`otr_master_audio_mux.py`). So:

    indextts2   22050  -> resampled 22050 -> 48000     (Lemmy's qualified route)
    chatterbox  24000  -> resampled 24000 -> 48000
    kokoro      24000  -> resampled 24000 -> 48000
    bark        24000  -> resampled 24000 -> 48000
    dia         44100  -> resampled 44100 -> 48000

NOBODY is already at the bus rate. Every clip is resampled, per clip, using its
own true native rate, by one ratio-agnostic gcd/polyphase helper. That is why
Lemmy works: he is not a special case the code had to learn, he is a new ratio
through a path that was already general.

The original file was wrong about the mechanism's SHAPE while being right that it
holds. Left uncorrected it would have taught the next engineer that non-Lemmy
characters bypass resampling, which is false and would make any future change to
that path look safer than it is.

WHAT IS ACTUALLY AT RISK, and why the gate is still worth having: an off-rate
clip that reaches the mix unresampled does not crash. It plays at the wrong speed
and the wrong pitch -- 22050 read as 48000 is more than twice as fast -- and
nothing goes red. That is the failure this file makes impossible to ship quietly.

Headless and pure: the real `_resample_audio` and the real profile resolver. No
engine, no model, no GPU.
"""
from __future__ import annotations

import math
import os
import re

import numpy as np
import pytest

os.environ.setdefault("OTR_TEST_MODE", "1")

#: The mixed bus. Pinned here AND re-derived from the source below, so this
#: constant cannot silently drift away from what the sequencer actually does --
#: which is exactly how the first version of this file went wrong.
BUS_SR = 48000
INDEX_SR = 22050

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _tone(freq_hz, seconds, sample_rate):
    """A pure sine. Its FREQUENCY is the thing under test: resampling must move
    the sample count without moving the pitch."""
    t = np.arange(int(seconds * sample_rate), dtype=np.float32) / float(sample_rate)
    return np.sin(2.0 * np.pi * freq_hz * t).astype(np.float32)


def _dominant_freq(clip, sample_rate):
    spectrum = np.abs(np.fft.rfft(clip.astype(np.float64)))
    return float(np.fft.rfftfreq(len(clip), 1.0 / sample_rate)[int(np.argmax(spectrum))])


# ---------------------------------------------------------------------------
# The rates themselves. If these drift, every assertion below is measuring the
# wrong thing -- which is not hypothetical, it already happened once.
# ---------------------------------------------------------------------------
def test_the_bus_rate_in_this_file_matches_the_sequencers_own_literal():
    """THE GUARD ON THE GUARD. The first version of this file hardcoded 24000
    against a 48000 bus and every test still passed, because they all agreed
    with each other. Read the real number out of the source instead."""
    with open(os.path.join(_REPO, "nodes", "scene_sequencer.py"),
              encoding="utf-8") as fh:
        source = fh.read()
    found = re.search(r"sample_rate\s*=\s*(\d+)\s*#\s*standardize output", source)
    assert found, ("the sequencer no longer declares a standardized output rate "
                   "the way this test recognises -- re-derive the bus")
    assert int(found.group(1)) == BUS_SR, (
        "sequencer standardizes to %s Hz but this file assumes %s"
        % (found.group(1), BUS_SR))


def test_the_publish_encode_agrees_with_the_bus():
    """A mix at one rate and an encode at another is the same defect one step
    later."""
    with open(os.path.join(_REPO, "nodes", "otr_master_audio_mux.py"),
              encoding="utf-8") as fh:
        source = fh.read()
    assert '"-ar", "%d"' % BUS_SR in source or "'-ar', '%d'" % BUS_SR in source


def test_no_char_voice_engine_is_already_at_the_bus_rate():
    """The correction, pinned. Every engine is resampled; Lemmy is a different
    RATIO, not a different kind of thing."""
    from nodes._otr_engine_profiles import require_resolver

    resolver = require_resolver()
    rates = {}
    for engine in ("indextts2", "kokoro", "chatterbox"):
        plan = resolver.resolve_casting_plan(role="char_voice", engine=engine)
        rates[engine] = int(plan.sample_rate)

    assert rates["indextts2"] == INDEX_SR, (
        "IndexTTS2 is no longer 22050 Hz; this gate needs re-deriving")
    for engine, rate in rates.items():
        assert rate != BUS_SR, (
            "%s is at the bus rate, which contradicts this file's premise that "
            "every clip gets resampled" % engine)


def test_the_qualified_lemmy_route_targets_that_engine():
    """Ties the rate question to the ACTUAL shipped route rather than to a
    hard-coded engine name."""
    from config.cast_pools import LEMMY_VOICE_POLICY as P

    route = P["approved_native_routes"]["indextts2"]
    assert route["qualification_record"]["engine"] == "indextts2"


# ---------------------------------------------------------------------------
# The gate: 22050 -> 48000 preserves pitch and duration.
# ---------------------------------------------------------------------------
def test_resampling_to_the_bus_preserves_DURATION():
    from nodes.scene_sequencer import _resample_audio

    seconds = 1.5
    clip = _tone(220.0, seconds, INDEX_SR)
    out = _resample_audio(clip, INDEX_SR, BUS_SR)

    expected = int(round(len(clip) * BUS_SR / INDEX_SR))
    assert abs(len(out) - expected) <= 2, (len(out), expected)
    assert abs(len(out) / BUS_SR - seconds) < 0.01, (
        "Lemmy's line would play at the wrong SPEED in the mix")


def test_resampling_to_the_bus_preserves_PITCH():
    """The defect this catches is silent. 22050 samples read as 48000 run more
    than twice as fast -- nothing errors, it simply is not the voice that was
    auditioned."""
    from nodes.scene_sequencer import _resample_audio

    freq = 440.0
    clip = _tone(freq, 1.0, INDEX_SR)
    out = _resample_audio(clip, INDEX_SR, BUS_SR)

    assert abs(_dominant_freq(out, BUS_SR) - freq) < 5.0

    # Prove the failure mode is detectable, or the assertion above proves less
    # than it claims: reinterpreting the SAME samples at the bus rate shifts the
    # pitch by well over an octave.
    naive = _dominant_freq(clip, BUS_SR)
    cents = 1200.0 * math.log2(naive / freq)
    assert cents > 1200.0, cents


def test_the_resampler_is_a_no_op_only_when_the_rates_MATCH():
    """A property of the helper, NOT a claim that any shipped engine skips
    resampling -- none does. The first version of this file said otherwise."""
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


@pytest.mark.parametrize("src", [22050, 24000, 44100, 16000])
def test_every_shipped_engine_rate_lands_on_the_bus(src):
    """22050 indextts2, 24000 kokoro/chatterbox/bark, 44100 dia. One
    ratio-agnostic path serves all of them, which is why Lemmy needed no special
    case."""
    from nodes.scene_sequencer import _resample_audio

    out = _resample_audio(_tone(330.0, 0.4, src), src, BUS_SR)
    assert abs(len(out) / BUS_SR - 0.4) < 0.01
    assert abs(_dominant_freq(out, BUS_SR) - 330.0) < 6.0
