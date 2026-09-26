# -*- coding: utf-8 -*-
"""A Google TTS character with no stated gender is cast by seed, not refused.

WHY (0j, operator 2026-09-25: "random genders and leave it nebulous").
CastLock raised `VoiceCastingError ... NO FALLBACK` for `google_tts` when a
cast row carried no gender, so a My Story character whose author never said
one stopped the render on the Google lane -- while the same row on Kokoro
took the seeded gender-agnostic draw. `_otr_my_story` leaves an unstated
gender empty on purpose and still does. Now google_tts takes that same draw;
the voice is a coin the episode seed flips, so it is deterministic.

A STATED gender is honoured unchanged: no cross-gender fallback for a
provider voice.

BOTH CASTING POLICIES (the two-policies rule): `auto_registry` casts ordinary
rows at CastLock, which is where the refusal lived. `preserve_ledger` does
not cast ordinary rows at all (it leaves an unassigned row with nothing
added), and the render-time resolver deliberately refuses to invent a Google
voice -- so under that policy an ordinary Google row is unstamped whether or
not a gender is stated. That behaviour is proven unchanged here, not fixed.

Headless. No engine, no model, no GPU, no network.
"""
from __future__ import annotations

import json

import pytest

from nodes._otr_voice_bank import load_voice_bank
from nodes.cast_lock import CastLock

UNSTATED = {"char_id": "c01", "name": "NELL", "voice_preset": "v2/en_speaker_1"}
STATED = {"char_id": "c02", "name": "TOM", "gender": "male",
          "voice_preset": "v2/en_speaker_2"}


def _lock(cast, *, seed=42, policy="auto_registry"):
    ledger = json.dumps({"meta": {"episode_seed": seed}, "cast": cast, "lines": []})
    out = CastLock().lock(script_json=ledger, voice_bank="google_tts",
                          char_voice_engine="google_tts",
                          cast_voice_policy=policy)[0]
    return {e["char_id"]: e for e in json.loads(out)["cast"]}


@pytest.fixture(scope="module")
def google_bank():
    return {e.voice_ref_id: e for e in load_voice_bank()[0]
            if e.engine == "google_tts"}


def test_an_unstated_gender_is_cast_from_the_google_pool(google_bank):
    rows = _lock([dict(UNSTATED)])
    vrid = rows["c01"].get("voice_ref_id")
    assert vrid in google_bank, vrid


def test_the_same_seed_picks_the_same_voice_twice():
    first = _lock([dict(UNSTATED)], seed=7)["c01"].get("voice_ref_id")
    second = _lock([dict(UNSTATED)], seed=7)["c01"].get("voice_ref_id")
    assert first and first == second


def test_a_stated_gender_is_still_honoured(google_bank):
    rows = _lock([dict(UNSTATED), dict(STATED)])
    tom = google_bank[rows["c02"]["voice_ref_id"]]
    assert tom.gender == "male"


def test_preserve_ledger_is_unchanged_for_both_rows():
    rows = _lock([dict(UNSTATED), dict(STATED)], policy="preserve_ledger")
    assert "voice_ref_id" not in rows["c01"]
    assert "voice_ref_id" not in rows["c02"]
