"""Character voices inherit CastLock's ledger stamp (2026-09-16).

`OTR_BatchCharacterVoices` has no engine widget. A leftover `engine=` from an
old canvas or a direct Python test must not override a concrete
`meta.char_voice_engine` stamp. `auto` is not a concrete stamp.
"""
from __future__ import annotations

import json
import os
import sys
import types

import numpy as np
import pytest

os.environ.setdefault("OTR_TEST_MODE", "1")

from nodes._otr_audio_engines import eng_kokoro
from nodes.batch_character_voices import BatchCharacterVoices


def _script(stamped_engine):
    return json.dumps({
        "lines": [{"line_id": "b001", "speaker_role": "character", "char_id": "c01",
                   "speaker": "X", "text": "A line the guard must see."}],
        "cast": [{"char_id": "c01", "name": "X", "gender": "male",
                  "voice_engine": "kokoro", "voice_ref_id": "bm_george"}],
        "meta": {"char_voice_engine": stamped_engine, "episode_seed": 7},
    })


def _stub_kokoro(monkeypatch):
    """A kokoro that renders 0.1 s of a flat tone, so the agreeing cases run the
    whole per-line path without a model, a GPU, or a voice file."""
    class _Pipe:
        def __init__(self, **kw):
            pass

        def __call__(self, text, **kw):
            yield ("g", "p", np.full(2400, 0.25, dtype=np.float32))

    monkeypatch.setitem(sys.modules, "kokoro", types.SimpleNamespace(KPipeline=_Pipe))
    monkeypatch.delenv("OTR_KOKORO_BACKEND", raising=False)
    monkeypatch.setattr(eng_kokoro, "_kokoro_voice_path", lambda v: __file__)


def test_leftover_engine_arg_cannot_override_a_concrete_stamp(monkeypatch):
    _stub_kokoro(monkeypatch)
    _audio, log_str, done = BatchCharacterVoices().generate(
        script_json=_script("kokoro"), engine="chatterbox")
    assert "controls disagree" not in log_str
    assert done.startswith("char_voice:done")


@pytest.mark.parametrize("stamped", ["auto", "kokoro", ""])
def test_auto_and_an_agreeing_stamp_never_trip_the_guard(monkeypatch, stamped):
    _stub_kokoro(monkeypatch)
    _audio, log_str, done = BatchCharacterVoices().generate(
        script_json=_script(stamped), engine="kokoro")
    assert "controls disagree" not in log_str, (stamped, log_str)
    assert done.startswith("char_voice:done"), (stamped, done, log_str)
