"""The Bark output gate: silent-but-nonempty audio must not leave the engine.

WHY THIS EXISTS (C2 verdict, 2026-08-28, adversarial review confirmed): every
downstream contract -- packing, sequencing, enhance, mastering, mux,
obs_publish -- checks shape, duration, rate and hash, and NOT ONE checks that
the audio is audible. A finite, nonempty, silent tensor walks the whole path
and publishes as a structurally valid clip: missing dialogue disguised as
delivered dialogue. `BarkEngine.generate_voice` is the one seam every live
production Bark render passes through, so the gate lives there.

The threshold (1e-4, ~-80 dBFS) is deliberately the SAME constant
`scene_sequencer._trim_trailing_silence` uses to call a sample silent --
aligned with existing semantics, not invented. A test pins that alignment so
the two cannot drift apart silently.
"""
from __future__ import annotations

import numpy as np
import pytest

from nodes._otr_audio_engines.eng_bark import BarkEngine, BarkSilentOutputError


@pytest.fixture()
def bark_engine(monkeypatch):
    """A BarkEngine whose model call is stubbed -- the audio each test wants
    is injected via `engine._test_audio` without loading Bark."""
    import nodes._otr_bark_lib as lib

    eng = BarkEngine()
    monkeypatch.setattr(lib, "_load_bark",
                        lambda _repo: (object(), object()))
    monkeypatch.setattr(lib, "_resolve_bark_speech_only", lambda: True)
    monkeypatch.setattr(lib, "_resolve_bark_inject_anchor", lambda: False)
    monkeypatch.setattr(
        lib, "_generate_single_line",
        lambda *a, **k: (eng._test_audio, 24000))
    return eng


def _call(eng):
    return eng.generate_voice("Hello there.", "v2/en_speaker_3", None, 7)


def test_a_healthy_line_passes_untouched(bark_engine):
    bark_engine._test_audio = (np.sin(np.linspace(0, 60, 24000))
                               .astype(np.float32) * 0.35)
    out = _call(bark_engine)
    assert out["sample_rate"] == 24000
    assert out["waveform"].shape[0:2] == (1, 1)
    assert float(out["waveform"].abs().max()) > 1e-4


def test_SILENT_but_nonempty_output_raises(bark_engine):
    """The exact shape the adversarial review proved reaches obs unchecked."""
    bark_engine._test_audio = np.zeros(24000, dtype=np.float32)
    with pytest.raises(BarkSilentOutputError, match="peak"):
        _call(bark_engine)


def test_below_threshold_whisper_of_nothing_raises(bark_engine):
    bark_engine._test_audio = (np.ones(24000, dtype=np.float32) * 5e-5)
    with pytest.raises(BarkSilentOutputError):
        _call(bark_engine)


def test_nonfinite_output_raises(bark_engine):
    audio = np.zeros(24000, dtype=np.float32)
    audio[100] = np.nan
    audio[200] = 0.5            # audible AND poisoned: still rejected
    bark_engine._test_audio = audio
    with pytest.raises(BarkSilentOutputError, match="finite"):
        _call(bark_engine)


def test_empty_output_raises(bark_engine):
    bark_engine._test_audio = np.zeros(0, dtype=np.float32)
    with pytest.raises(BarkSilentOutputError):
        _call(bark_engine)


def test_the_error_never_names_a_replacement_preset(bark_engine):
    """The verdict forbids remapping. The message must instruct a re-run and
    must not suggest another preset or engine -- wording is the contract."""
    bark_engine._test_audio = np.zeros(24000, dtype=np.float32)
    with pytest.raises(BarkSilentOutputError) as exc:
        _call(bark_engine)
    msg = str(exc.value)
    assert "never remap" in msg
    assert "v2/en_speaker_3" in msg          # names the FAILING preset only


def test_threshold_matches_the_sequencers_silence_semantics():
    """1e-4 is the sequencer's own silence constant. If either side changes,
    this fails and the two definitions of 'silent' must be reconciled
    deliberately rather than drifting apart."""
    import inspect

    from nodes import scene_sequencer as seq
    from nodes._otr_audio_engines import eng_bark

    sig = inspect.signature(seq._trim_trailing_silence)
    assert sig.parameters["threshold"].default == 1e-4
    src = inspect.getsource(eng_bark.BarkEngine.generate_voice)
    assert "1e-4" in src
