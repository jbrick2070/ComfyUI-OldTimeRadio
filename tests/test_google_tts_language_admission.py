# -*- coding: utf-8 -*-
"""Google TTS is admitted on every episode language Kokoro is.

WHY (0i, operator 2026-09-25: "all supported as Kokoro, just shove it
Kokoro's stuff"). Non-English rows in config/episode_languages.json admitted
only `kokoro`, so a non-English episode on the Google lane stopped at
CastLock's language gate before a word was spoken. Gemini TTS prebuilt
voices are not tied to a language -- the request carries the text and a
voice name, and the model speaks the text's language -- so admission needs no
per-language config, only the entry.

The operator's ear on any one language stays the operator's; admission is now
a config fact, not a per-language wait.

Headless. No engine, no model, no GPU, no network.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from nodes import cast_lock
from nodes import _otr_episode_languages as eplang

REPO = pathlib.Path(__file__).resolve().parents[1]
ROWS = json.loads((REPO / "config" / "episode_languages.json")
                  .read_text(encoding="utf-8"))["rows"]


@pytest.mark.parametrize("row", ROWS, ids=lambda r: r["iso"])
def test_google_tts_is_admitted_wherever_kokoro_is(row):
    if "kokoro" in row["engines"]:
        assert "google_tts" in row["engines"], row["iso"]


@pytest.mark.parametrize("iso", [r["iso"] for r in ROWS if r["iso"] != "en"])
def test_castlock_admits_google_tts_on_a_non_english_episode(iso, monkeypatch):
    """Drive the real gate. Readiness extras (e.g. a Japanese tokenizer) are a
    separate, later check about THIS box's installs; they are stubbed so this
    test proves admission only."""
    monkeypatch.setattr(eplang, "readiness_extra_ok", lambda extra: True)
    meta = {"episode_language": iso}
    assert cast_lock._require_language_engines(meta, "google_tts", "google_tts") == iso
    assert cast_lock._require_language_engines(meta, "kokoro", "google_tts") == iso


def test_every_google_voice_speaks_every_admitting_language():
    """Admission alone is not enough: the bank reads an absent `languages` as
    English-only, so a French Google character would pass the gate and then
    find no voice. The bank tags and the rows must agree."""
    from nodes._otr_voice_bank import entry_languages, load_voice_bank
    admitting = {r["iso"] for r in ROWS if "google_tts" in r["engines"]}
    google = [e for e in load_voice_bank()[0] if e.engine == "google_tts"]
    assert google
    for entry in google:
        assert set(entry_languages(entry)) == admitting, entry.voice_ref_id


def test_a_french_google_character_is_actually_cast(monkeypatch):
    """End to end through CastLock: gate, then a real French Google voice."""
    import json as _json
    from nodes.cast_lock import CastLock
    monkeypatch.setattr(eplang, "readiness_extra_ok", lambda extra: True)
    ledger = _json.dumps({
        "meta": {"episode_seed": 42, "episode_language": "fr"},
        "cast": [{"char_id": "c01", "name": "HORATIO", "gender": "male",
                  "voice_preset": "v2/en_speaker_1"}],
        "lines": []})
    out = CastLock().lock(script_json=ledger, voice_bank="google_tts",
                          char_voice_engine="google_tts",
                          cast_voice_policy="auto_registry")[0]
    row = _json.loads(out)["cast"][0]
    assert str(row.get("voice_ref_id", "")).startswith("gt_"), row


def test_an_unlisted_engine_is_still_refused_off_english(monkeypatch):
    """The gate still means something: bark is not on the French row."""
    monkeypatch.setattr(eplang, "readiness_extra_ok", lambda extra: True)
    with pytest.raises(ValueError, match="not admitted"):
        cast_lock._require_language_engines({"episode_language": "fr"}, "bark", "kokoro")
