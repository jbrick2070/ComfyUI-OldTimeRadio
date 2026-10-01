# -*- coding: utf-8 -*-
"""ElevenLabs is admitted on every episode language Kokoro is.

WHY (operator 2026-09-30). A Spanish My Story run on otr_cloud_low_1act
stopped at the queue-time language gate: "the cloud_elevenlabs voice does not
speak Spanish". It does. The engine only ever sends eleven_multilingual_v2 or
eleven_v3 (eng_cloud_elevenlabs._SUPPORTED_MODELS), both of which speak every
language OTR ships, from the text alone -- like Google TTS, admission needs no
per-language config, only the entry. Every cloud workflow pins
cloud_elevenlabs for both voice slots, so the gap made every cloud workflow
English-only.

Chatterbox (ChatterboxTTS, English-only) and IndexTTS2 (Chinese/English) are
deliberately NOT admitted: their models would read a Spanish line with an
English voice and no error.

Headless. No engine, no model, no GPU, no network.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from nodes import cast_lock
from nodes import _otr_episode_languages as eplang
from nodes._otr_audio_engines import eng_cloud_elevenlabs

REPO = pathlib.Path(__file__).resolve().parents[1]
ROWS = json.loads((REPO / "config" / "episode_languages.json")
                  .read_text(encoding="utf-8"))["rows"]


def test_the_engine_only_sends_multilingual_models():
    """The admission below rests on this; a new English-only model id here
    would make it false."""
    assert set(eng_cloud_elevenlabs._SUPPORTED_MODELS) == {
        "eleven_multilingual_v2", "eleven_v3"}


@pytest.mark.parametrize("row", ROWS, ids=lambda r: r["iso"])
def test_elevenlabs_is_admitted_wherever_kokoro_is(row):
    if "kokoro" in row["engines"]:
        assert "cloud_elevenlabs" in row["engines"], row["iso"]


@pytest.mark.parametrize("iso", [r["iso"] for r in ROWS if r["iso"] != "en"])
def test_castlock_admits_elevenlabs_on_a_non_english_episode(iso, monkeypatch):
    monkeypatch.setattr(eplang, "readiness_extra_ok", lambda extra: True)
    meta = {"episode_language": iso}
    assert cast_lock._require_language_engines(
        meta, "cloud_elevenlabs", "cloud_elevenlabs") == iso


def test_every_elevenlabs_voice_speaks_every_admitting_language():
    """Admission alone is not enough: the bank reads an absent `languages` as
    English-only, so a Spanish ElevenLabs character would pass the gate and
    then find no voice. The bank tags and the rows must agree."""
    from nodes._otr_voice_bank import entry_languages, load_voice_bank
    admitting = {r["iso"] for r in ROWS if "cloud_elevenlabs" in r["engines"]}
    voices = [e for e in load_voice_bank()[0] if e.engine == "cloud_elevenlabs"]
    assert voices
    for entry in voices:
        assert set(entry_languages(entry)) == admitting, entry.voice_ref_id


@pytest.mark.parametrize("engine", ["chatterbox", "indextts2"])
def test_english_only_local_engines_stay_refused_off_english(engine, monkeypatch):
    monkeypatch.setattr(eplang, "readiness_extra_ok", lambda extra: True)
    with pytest.raises(ValueError, match="not admitted"):
        cast_lock._require_language_engines(
            {"episode_language": "es"}, engine, engine)
