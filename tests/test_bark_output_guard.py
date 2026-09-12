"""The bark output guard: a take that is not shaped like speech is re-rolled,
a bounded number of times, and the best take always ships.

PBUG-20260902-03: bark's semantic stage can derail on any roll into seven
seconds of noise floor and two of a steady 2.5 kHz tone, with nothing in the
log. The record calls the missing guard "a silent wrong render". The scorer
was calibrated on 32 real bark takes (docs/2026-09-12-bark-output-guard/):
pitch in 70-400 Hz frame by frame, because the whole-second dominant bin the
record first proposed reads a formant-heavy voice as "not speech".

The Bible verify condition: a stub engine that returns a pure tone on the
first call and speech on the second must produce speech from the guarded
path in one retry.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

import nodes._otr_bark_lib as lib
from nodes._otr_audio_engines.eng_bark import BarkEngine, BarkSilentOutputError

SR = 24000


def _t(seconds=4.0):
    return np.arange(int(SR * seconds)) / SR


def buzz(seconds=4.0):
    """A pitched signal at 180 Hz with harmonics: shaped like speech."""
    t = _t(seconds)
    return sum(0.4 / k * np.sin(2 * np.pi * 180.0 * k * t) for k in range(1, 8)).astype(np.float32)


def tone(seconds=4.0, hz=2600.0):
    return (0.5 * np.sin(2 * np.pi * hz * _t(seconds))).astype(np.float32)


def noise(seconds=4.0):
    return np.random.default_rng(7).normal(0.0, 0.3, int(SR * seconds)).astype(np.float32)


# --- the scorer, pure ---------------------------------------------------------

def test_the_scorer_separates_speech_from_the_two_documented_artifacts():
    assert lib.speech_shape_score(buzz(), SR) >= 0.9
    assert lib.speech_shape_score(tone(), SR) == 0.0          # the 2.5 kHz tone
    assert lib.speech_shape_score(noise(), SR) == 0.0         # the noise floor
    seven_then_two = np.concatenate([noise(7.0), tone(2.0)])   # the exact record
    assert lib.speech_shape_score(seven_then_two, SR) == 0.0


def test_a_pause_is_not_a_verdict():
    b = buzz(6.0)
    b[SR * 2:SR * 4] = 0.0
    assert lib.speech_shape_score(b, SR) >= 0.9
    assert lib.speech_shape_score(np.zeros(SR * 3, dtype=np.float32), SR) == 0.0


def test_a_voice_with_a_strong_second_harmonic_is_still_speech():
    """The first-peak rule, adversarially (codex, 2026-09-12). A voice whose
    second harmonic is LOUDER than its fundamental correlates better at twice
    the pitch period, so a global-maximum search reads it an octave down --
    and at the range edge that demotes real speech. The FIRST local peak is
    the pitch period either way."""
    t = _t(4.0)
    for f0 in (95.0, 120.0, 140.0, 180.0, 200.0, 250.0, 350.0):
        equal = (0.5 * np.sin(2 * np.pi * f0 * t)
                 + 0.5 * np.sin(2 * np.pi * 2 * f0 * t)
                 + 0.25 * np.sin(2 * np.pi * 3 * f0 * t)).astype(np.float32)
        led = (0.9 * np.sin(2 * np.pi * f0 * t)
               + 0.3 * np.sin(2 * np.pi * 2 * f0 * t)).astype(np.float32)
        assert lib.speech_shape_score(equal, SR) >= 0.9, ("equal", f0)
        assert lib.speech_shape_score(led, SR) >= 0.9, ("led", f0)


def test_the_range_edge_is_measured_and_deliberate():
    """THE ONE SHAPE THIS DETECTOR TURNS AWAY, measured exactly: a voice
    ABOVE 200 Hz whose second harmonic is several times louder than its
    fundamental peaks at half its pitch period and reads an octave high.
    Below 200 Hz the half-period is still inside the range, so every
    measured bark preset (95-250 Hz) passes on any harmonic balance.

    It is left here on purpose. Accepting integer multiples of the first
    peak would cover this case and would also admit the 2.6 kHz tone -- its
    9-sample period times seven lands squarely in the speaking range -- which
    is the defect the guard exists to catch. Such a line costs up to three
    takes and never costs the take."""
    t = _t(4.0)

    def two_f0_dominant(f0):
        return (0.25 * np.sin(2 * np.pi * f0 * t)
                + 0.9 * np.sin(2 * np.pi * 2 * f0 * t)
                + 0.4 * np.sin(2 * np.pi * 3 * f0 * t)).astype(np.float32)

    for f0 in (95.0, 140.0, 180.0, 200.0):
        assert lib.speech_shape_score(two_f0_dominant(f0), SR) >= 0.9, f0
    for f0 in (220.0, 250.0, 350.0):
        assert lib.speech_shape_score(two_f0_dominant(f0), SR) == 0.0, f0


def test_a_quiet_but_usable_take_is_judged_on_shape_not_level():
    """A take can clear the engine's 1e-4 peak gate and still be soft. Level
    is the silent gate's business; scoring it 0 would spend two bark renders
    to ship the same audio (codex, 2026-09-12)."""
    loud = buzz()
    quiet = (loud * 0.002).astype(np.float32)     # peak ~6e-3, far above 1e-4
    assert float(np.abs(quiet).max()) > 1e-4
    assert lib.speech_shape_score(quiet, SR) == lib.speech_shape_score(loud, SR)
    assert lib.speech_shape_score(quiet, SR) >= 0.9
    # and a quiet TONE is still not speech
    assert lib.speech_shape_score((tone() * 0.002).astype(np.float32), SR) == 0.0


def test_the_pass_line_sits_between_the_artifacts_and_normal_speech():
    """Calibration (docs/2026-09-12-bark-output-guard/): normal presets scored
    0.50-1.00, every artifact 0.00. The line must stay in that gap."""
    assert 0.0 < lib.SPEECH_SHAPE_PASS < 0.5
    assert lib.BARK_REROLLS_MAX == 2, "bounded: at most three takes a line"


def test_the_reroll_seed_ladder_is_deterministic_and_far_from_the_line_seed():
    s = 123456789
    ladder = [lib.bark_reroll_seed(s, k) for k in range(3)]
    assert ladder[0] == s
    assert ladder == [lib.bark_reroll_seed(s, k) for k in range(3)]
    assert len(set(ladder)) == 3
    assert all(0 <= v < 2 ** 63 for v in ladder)
    assert all(abs(v - s) > 1000 for v in ladder[1:]), "not seed + 1"


# --- the adapter, stubbed ----------------------------------------------------

@pytest.fixture()
def bark_engine(monkeypatch):
    """`_generate_single_line` returns whatever `eng._takes` holds for the
    seed it was called with (falling back to `eng._default_take`), and
    records every seed, so the ladder and the call count are observable."""
    eng = BarkEngine()
    eng._takes = {}
    eng._default_take = buzz()
    eng._seeds = []

    def _gen(text, preset, model, processor, **kw):
        seed = kw["seed"]
        eng._seeds.append(seed)
        return eng._takes.get(seed, eng._default_take), SR

    monkeypatch.setattr(lib, "_load_bark", lambda _repo, device=None: (object(), object()))
    monkeypatch.setattr(lib, "_resolve_bark_speech_only", lambda: True)
    monkeypatch.setattr(lib, "_resolve_bark_inject_anchor", lambda: False)
    monkeypatch.setattr(lib, "_generate_single_line", _gen)
    return eng


def _call(eng, seed=7):
    return eng.generate_voice("That's an order, CHALLENGER.", "v2/en_speaker_7", None, seed)


def _ladder(seed=7):
    return [lib.bark_reroll_seed(seed, k) for k in range(3)]


def test_a_take_shaped_like_speech_ships_on_the_first_call(bark_engine, caplog):
    with caplog.at_level(logging.WARNING, logger="OTR"):
        out = _call(bark_engine)
    assert bark_engine._seeds == [7]
    assert out["sample_rate"] == SR
    assert not [r for r in caplog.records if "speech-shape" in r.getMessage()]


def test_the_bible_condition_tone_then_speech_takes_one_retry(bark_engine, caplog):
    s0, s1, _ = _ladder()
    bark_engine._takes = {s0: tone(), s1: buzz()}
    with caplog.at_level(logging.WARNING, logger="OTR"):
        out = _call(bark_engine)
    assert bark_engine._seeds == [s0, s1], "exactly one re-roll, on the ladder"
    got = out["waveform"].reshape(-1).numpy()
    assert lib.speech_shape_score(got, SR) >= 0.9, "the speech take shipped"
    msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("re-rolling" in m for m in msgs)


def test_every_take_failing_ships_the_best_and_never_raises(bark_engine, caplog):
    s0, s1, s2 = _ladder()
    weaker = noise()
    # One speech-shaped second in four: score 0.25, under the 0.30 line, so it
    # FAILS the guard -- and still outscores the noise and the tone.
    stronger = np.concatenate([tone(3.0), buzz(1.0)])
    assert lib.speech_shape_score(stronger, SR) < lib.SPEECH_SHAPE_PASS
    bark_engine._takes = {s0: weaker, s1: tone(), s2: stronger}
    with caplog.at_level(logging.WARNING, logger="OTR"):
        out = _call(bark_engine)
    assert bark_engine._seeds == [s0, s1, s2], "three takes, then stop"
    got = out["waveform"].reshape(-1).numpy()
    assert np.allclose(got, stronger), "the best-scoring take ships"
    assert any("best of 3" in r.getMessage() for r in caplog.records)


def test_a_silent_take_is_not_a_candidate_but_does_not_end_the_line(bark_engine):
    s0, s1, _ = _ladder()
    bark_engine._takes = {s0: np.zeros(SR * 2, dtype=np.float32), s1: buzz()}
    out = _call(bark_engine)
    assert bark_engine._seeds == [s0, s1]
    assert float(out["waveform"].abs().max()) > 1e-4


def test_silence_on_every_take_still_raises_the_silent_output_error(bark_engine):
    bark_engine._default_take = np.zeros(SR * 2, dtype=np.float32)
    with pytest.raises(BarkSilentOutputError, match="never remap"):
        _call(bark_engine)
    assert len(bark_engine._seeds) == 3, "the ladder was walked before giving up"


def test_the_guard_is_bounded_to_the_ladder_and_never_loops(bark_engine):
    bark_engine._default_take = tone()
    _call(bark_engine)
    assert bark_engine._seeds == _ladder(), "three takes and not one more"


def test_a_thrown_attempt_does_not_discard_a_banked_take(bark_engine, caplog):
    """"The ledger field is always filled" has to survive a transient error.
    Before the guard, one failed generation meant one failed line and there
    was nothing banked to lose; now attempt 2 raising must not throw away a
    usable take from attempt 1 (Sonnet QA, 2026-09-12)."""
    s0, s1, s2 = _ladder()
    banked = np.concatenate([tone(3.0), buzz(1.0)])     # usable, fails the guard
    assert lib.speech_shape_score(banked, SR) < lib.SPEECH_SHAPE_PASS

    def _gen(text, preset, model, processor, **kw):
        seed = kw["seed"]
        bark_engine._seeds.append(seed)
        if seed == s0:
            return banked, SR
        raise RuntimeError("transient CUDA hiccup")

    bark_engine._takes = {}
    import nodes._otr_bark_lib as _lib
    caplog.set_level(logging.WARNING, logger="OTR")
    _lib._generate_single_line = _gen                    # monkeypatched fixture
    out = _call(bark_engine)
    assert bark_engine._seeds == [s0, s1, s2]
    assert np.allclose(out["waveform"].reshape(-1).numpy(), banked)
    assert any("raised" in r.getMessage() for r in caplog.records)


def test_when_every_attempt_throws_the_real_error_reaches_the_caller(bark_engine):
    """A genuinely broken engine must still fail loudly on the first line --
    the guard may not convert it into a silence complaint."""
    def _boom(text, preset, model, processor, **kw):
        bark_engine._seeds.append(kw["seed"])
        raise RuntimeError("bark is not installed correctly")

    import nodes._otr_bark_lib as _lib
    _lib._generate_single_line = _boom
    with pytest.raises(RuntimeError, match="not installed correctly"):
        _call(bark_engine)
    assert len(bark_engine._seeds) == 3


def test_the_shipped_sample_rate_belongs_to_the_take_that_shipped(bark_engine):
    """`sr` from the last loop iteration would mislabel the audio the day
    anything reloads the model mid-ladder (Sonnet QA, 2026-09-12)."""
    s0, s1, s2 = _ladder()
    banked = np.concatenate([tone(3.0), buzz(1.0)])
    rates = {s0: 24000, s1: 48000, s2: 16000}

    def _gen(text, preset, model, processor, **kw):
        seed = kw["seed"]
        bark_engine._seeds.append(seed)
        return (banked if seed == s0 else tone()), rates[seed]

    import nodes._otr_bark_lib as _lib
    _lib._generate_single_line = _gen
    out = _call(bark_engine)
    assert np.allclose(out["waveform"].reshape(-1).numpy(), banked)
    assert out["sample_rate"] == 24000, "the winning take's rate, not the last"


def test_the_adapter_still_threads_the_line_seed_by_name():
    """The determinism contract other tests grep for: `seed=seed` reaches
    `_generate_single_line`, and on attempt 0 it IS the line seed."""
    import inspect
    src = inspect.getsource(BarkEngine.generate_voice)
    assert "seed=seed" in src
    assert "bark_reroll_seed(base_seed, attempt)" in src
