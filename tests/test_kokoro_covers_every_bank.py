"""Every source bank casts fully on kokoro, with the right genders (operator
requirement 2026-09-02, queue item 2: "be sure all the voices and genders are
mapped for all media / source bank usage").

kokoro becomes the shipped default for BOTH voice slots on every lane, so the
kokoro pool must dress the largest cast any bank can ask for -- 10 speaking
characters on scifi_news_pro, 6 on the legacy banks -- with a distinct voice of
the requested gender per character, plus an announcer of either gender, and
every voice the bank names must be one the boot prefetch places on disk (and
the reverse), so a bank/prefetch drift fails here by name rather than as a
mid-episode "voice file missing" refusal.
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("OTR_TEST_MODE", "1")

import pytest

from nodes._otr_casting import _LEGACY_MAX_SPEAKING_CAST, _VALID_GENDERS
from nodes._otr_kokoro_voice_prefetch import ENGLISH_VOICES
from nodes._otr_scifi_news_pro import MAX_SPEAKING_CAST as _PRO_MAX_CAST
from nodes._otr_voice_bank import (
    VoiceCastingError, announcer_voice_ref, assign_voice_for_slot,
    gender_agnostic_fallback_ref, load_voice_bank,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_BANKS = os.path.join(_HERE, "..", "nodes", "story_packs", "banks.json")


def _bank_ids() -> list:
    with open(_BANKS, encoding="utf-8") as fh:
        rows = json.load(fh)["banks"]
    ids = [r["source_bank_id"] for r in rows if r.get("runnable", True)]
    assert ids, "banks.json lists no runnable bank"
    return ids


def _cast_ceiling(bank_id: str) -> int:
    return _PRO_MAX_CAST if bank_id == "scifi_news_pro" else _LEGACY_MAX_SPEAKING_CAST


def _kokoro_entries():
    bank = load_voice_bank()[0]
    return bank, [e for e in bank if e.engine == "kokoro"]


def test_bank_and_prefetch_name_the_same_kokoro_voices():
    _, ko = _kokoro_entries()
    bank_ids = {e.voice_ref_id for e in ko}
    assert bank_ids == set(ENGLISH_VOICES), (
        "voice bank / prefetch drift -- only in bank: %s; only in prefetch: %s"
        % (sorted(bank_ids - set(ENGLISH_VOICES)),
           sorted(set(ENGLISH_VOICES) - bank_ids)))
    for e in ko:
        assert e.gender in ("male", "female"), (e.voice_ref_id, e.gender)
        assert "char_voice" in e.roles, "%s cannot serve characters" % e.voice_ref_id
        assert e.ref_path.replace("\\", "/").endswith(
            "voices/%s.pt" % e.voice_ref_id), (e.voice_ref_id, e.ref_path)


@pytest.mark.parametrize("bank_id", _bank_ids())
def test_every_bank_casts_its_ceiling_on_kokoro(bank_id):
    bank, _ = _kokoro_entries()
    ceiling = _cast_ceiling(bank_id)
    genders = sorted(_VALID_GENDERS)          # female, male, other -- all three
    used: set = set()
    assigned = []
    seed = hash(bank_id) & 0xFFFF
    for i in range(ceiling):
        gender = genders[i % len(genders)]
        char_id = "c%02d" % (i + 1)
        try:
            ref = assign_voice_for_slot(
                char_id=char_id, role="char_voice", gender=gender,
                engine="kokoro", episode_seed=seed,
                used_voice_ref_ids=used, allow_voice_reuse=False, bank=bank,
            )
        except VoiceCastingError:
            # Production (cast_lock.py) takes this branch for exactly one
            # reason: no bank carries 'other' rows for ANY engine, so an 'other'
            # row renders through the gender-agnostic draw. A male or female
            # row must never land here on kokoro.
            assert gender == "other", (
                "%s: kokoro could not cast a %s character %s" % (bank_id, gender, char_id))
            ref = gender_agnostic_fallback_ref(
                bank, engine="kokoro", char_id=char_id, episode_seed=seed,
                role="char_voice", used=used)
            assert ref is not None, "%s: no gender-agnostic kokoro voice" % bank_id
        assert ref.voice_ref_id in ENGLISH_VOICES, (bank_id, gender, ref.voice_ref_id)
        entry = next(e for e in bank if e.voice_ref_id == ref.voice_ref_id
                     and e.engine == "kokoro")
        if gender in ("male", "female"):
            assert entry.gender == gender, (
                "%s: character %d asked for %s and got %s (%s)"
                % (bank_id, i + 1, gender, entry.gender, ref.voice_ref_id))
        used.add(ref.voice_ref_id)
        assigned.append(ref.voice_ref_id)
    assert len(set(assigned)) == ceiling, (
        "%s: kokoro reused a voice inside one cast: %s" % (bank_id, assigned))


def test_kokoro_announcer_covers_both_genders():
    bank, _ = _kokoro_entries()
    by_id = {e.voice_ref_id: e for e in bank if e.engine == "kokoro"}
    seen = set()
    for seed in range(40):
        ref = announcer_voice_ref("kokoro", episode_seed=seed)
        assert ref.voice_ref_id in ENGLISH_VOICES, ref.voice_ref_id
        seen.add(by_id[ref.voice_ref_id].gender)
    assert seen == {"male", "female"}, (
        "the seeded kokoro announcer pick never produced both genders: %s" % seen)
