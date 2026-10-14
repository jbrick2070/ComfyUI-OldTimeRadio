"""A ledger cast for Bark, rendered on Kokoro under preserve_ledger, must not
silently collapse every character into the announcer's voice.

Voice review V2 (Muse Wonder Full, Codex source QA; reproduced 2026-10-14 by a
Sonnet 5.5 fixture driving the real CastLock -> BatchCharacterVoices). Kokoro
speaks from the bank's ``voice_ref_id``; a character row carrying only a Bark
``voice_preset`` passed CastLock admission (preset OR ref) and then fell
through to Kokoro's EPISODE voice -- the announcer's -- so three distinct
characters rendered with one voice, with no warning, while the ledger still
named their Bark identities.

preserve_ledger re-casts nothing (tests/test_voice_casting_identity.py pins
that CastLock adds no voice_ref_id), so the honest outcome is a named failure
at line dispatch, telling the operator how to cast for this engine. The
announcer's own episode pick is its legitimate default and stays untouched
(tests/test_announcer_voice.py).

Kokoro is stubbed: no model is loaded.
"""
from __future__ import annotations

import json
import sys
import types

import numpy as np
import pytest

_CAST = [
    {"char_id": "ann", "name": "ANNOUNCER", "gender": "male",
     "speaker_role": "announcer", "voice_preset": "v2/en_speaker_6"},
    {"char_id": "c01", "name": "MONTY", "gender": "male", "voice_preset": "v2/en_speaker_1"},
    {"char_id": "c02", "name": "VERA", "gender": "female", "voice_preset": "v2/en_speaker_9"},
    {"char_id": "c03", "name": "DR HALE", "gender": "male", "voice_preset": "v2/en_speaker_3"},
]
_LINES = [
    {"line_id": "l1", "char_id": "ann", "speaker_role": "announcer", "text": "Tonight, on the air."},
    {"line_id": "l2", "char_id": "c01", "speaker_role": "character", "speaker": "MONTY",
     "text": "Monty speaking here."},
    {"line_id": "l3", "char_id": "c02", "speaker_role": "character", "speaker": "VERA",
     "text": "Vera speaking here."},
    {"line_id": "l4", "char_id": "c03", "speaker_role": "character", "speaker": "DR HALE",
     "text": "Hale speaking here."},
]
_KOKORO_REFS = {"c01": "bm_george", "c02": "af_bella", "c03": "bm_fable"}


def _ledger(cast):
    return json.dumps({"schema_version": "l3-2026-05-14",
                       "meta": {"episode_seed": "v2-seed-1", "language": "en"},
                       "cast": json.loads(json.dumps(cast)),
                       "lines": json.loads(json.dumps(_LINES))})


@pytest.fixture
def kokoro_stub(monkeypatch):
    """Stub the Kokoro pipeline and record which voice each line used."""
    from nodes._otr_audio_engines import eng_kokoro

    rendered = []

    class _Pipe:
        def __init__(self, **_kw):
            pass

        def __call__(self, text, voice=None, **_kw):
            rendered.append(voice)
            yield ("g", "p", np.full(2400, 0.25, dtype=np.float32))

    monkeypatch.setitem(sys.modules, "kokoro", types.SimpleNamespace(KPipeline=_Pipe))
    monkeypatch.delenv("OTR_KOKORO_BACKEND", raising=False)
    monkeypatch.delenv("OTR_VOICE_REFERENCE_BANK", raising=False)
    monkeypatch.setattr(eng_kokoro, "_kokoro_voice_path", lambda _v: __file__)
    return rendered


def _lock(cast, policy="preserve_ledger"):
    from nodes.cast_lock import CastLock

    return CastLock().lock(script_json=_ledger(cast), char_voice_engine="kokoro",
                           announcer_voice_engine="kokoro", cast_voice_policy=policy)[0]


def _characters(cast, ledger_json):
    from nodes.batch_character_voices import BatchCharacterVoices

    return BatchCharacterVoices().generate(script_json=_ledger(cast), ledger_json=ledger_json)


def test_bark_only_characters_on_kokoro_fail_loud_and_name_the_fix(kokoro_stub):
    with pytest.raises(Exception) as excinfo:
        _characters(_CAST, _lock(_CAST))
    msg = str(excinfo.value)
    assert type(excinfo.value).__name__ == "EngineUnusable", type(excinfo.value)
    assert "char=c01" in msg and "kokoro" in msg
    assert "Bark identity" in msg and "auto_registry" in msg
    assert kokoro_stub == [], "nothing may render before the failure"


def test_a_previously_bark_locked_ledger_fails_loud_too(kokoro_stub):
    cast = [dict(r, voice_engine="bark", tts_model="bark", voice_ref_id="") for r in _CAST]
    with pytest.raises(Exception, match="no voice_ref_id for engine 'kokoro'"):
        _characters(cast, _lock(cast))


def test_control_kokoro_refs_render_distinct_voices(kokoro_stub):
    cast = [dict(r) for r in _CAST]
    for row in cast:
        if row["char_id"] in _KOKORO_REFS:
            row.pop("voice_preset")
            row["voice_ref_id"] = _KOKORO_REFS[row["char_id"]]
    _characters(cast, _lock(cast))
    assert len(kokoro_stub) == 3 and len(set(kokoro_stub)) == 3, kokoro_stub


def test_control_auto_registry_casts_for_kokoro_and_renders(kokoro_stub):
    _characters(_CAST, _lock(_CAST, policy="auto_registry"))
    assert len(kokoro_stub) == 3 and len(set(kokoro_stub)) == 3, kokoro_stub


def test_control_the_announcer_keeps_its_episode_pick(kokoro_stub):
    from nodes.announcer_voice import AnnouncerVoice

    AnnouncerVoice().generate(script_json=_ledger(_CAST), ledger_json=_lock(_CAST))
    assert len(kokoro_stub) == 1 and kokoro_stub[0], kokoro_stub
