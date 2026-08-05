"""A candidate tier of ONE is a pin, not a choice.

`_ladder_pick` used to accept the first non-empty tier at any size, so an
(engine, gender, timbre) combination whose top tier held exactly one entry drew
that same voice on 100% of seeds. Against the shipped bank three combos did
exactly that, which is a large part of why a 40-voice bank sounded like three.

Threshold-free by design. An earlier version of this plan proposed acceptance
criteria like "top-3 concentration under 25%", and two independent
reconstructions of that simulation disagreed on the baseline -- a number tuned to
one arbitrary gender shape and seed range, which any bank edit invalidates. The
invariant here is structural instead: no first-accepted tier may have size 1
unless it is the always-accepted gender-only last tier. It is measured directly
against the committed bank, needs no simulation, and cannot drift.
"""
from __future__ import annotations

import pytest

from nodes import _otr_casting as _CASTING
from nodes._otr_voice_bank import (
    _LADDER, _matches, _min_tier_pool, assign_voice_for_slot, load_voice_bank,
)

ENGINES = ("indextts2", "kokoro")
GENDERS = ("male", "female")

# The three combos that were 100% pinned before the floor landed, with the tier
# sizes measured against the shipped bank. Named so a bank edit that changes them
# is visible in the diff rather than silently loosening the test.
KNOWN_THIN = (
    ("indextts2", "female", "warm"),
    ("kokoro", "male", "warm"),
    ("kokoro", "male", "deep"),
)


@pytest.fixture(autouse=True)
def _pinned_env(monkeypatch):
    """Never read whatever the operator last exported into their shell."""
    monkeypatch.delenv("OTR_CAST_MIN_TIER_POOL", raising=False)
    monkeypatch.delenv("OTR_CAST_WEIGHTED", raising=False)


def _tier_sizes(entries, engine, gender, word):
    return [
        len([e for e in entries
             if e.engine == engine and e.quality_tier != "reject"
             and _matches(e, dims, gender=gender, timbre=(word,),
                          role="char_voice", age_band="adult")])
        for dims in _LADDER
    ]


def _first_accepted(sizes, floor):
    for i, n in enumerate(sizes):
        if n and (n >= floor or i == len(_LADDER) - 1):
            return i, n
    return None, 0


def test_no_accepted_tier_is_a_pin_of_one():
    """The invariant. Fails today on exactly the three KNOWN_THIN combos."""
    entries, _ = load_voice_bank()
    floor = _min_tier_pool()
    offenders = []
    for engine in ENGINES:
        for gender in GENDERS:
            for word in _CASTING._TIMBRE_VOCAB:
                sizes = _tier_sizes(entries, engine, gender, word)
                idx, n = _first_accepted(sizes, floor)
                if idx is not None and n == 1 and idx != len(_LADDER) - 1:
                    offenders.append((engine, gender, word, sizes))
    assert not offenders, offenders


def test_the_three_known_pins_now_draw_more_than_one_voice():
    """The audible half. Each of these returned ONE voice on 100% of seeds."""
    for engine, gender, word in KNOWN_THIN:
        drawn = {
            assign_voice_for_slot(
                role="char_voice", engine=engine, char_id="c02",
                gender=gender, timbre=(word,), age_band="adult",
                episode_seed=seed,
            ).voice_ref_id
            for seed in range(200)
        }
        assert len(drawn) > 1, (engine, gender, word, drawn)


def test_the_known_thin_tiers_really_are_thin():
    """Guards the premise. If a bank edit fattens these, the test above stops
    proving anything and this one says so out loud.
    """
    entries, _ = load_voice_bank()
    for engine, gender, word in KNOWN_THIN:
        sizes = _tier_sizes(entries, engine, gender, word)
        assert sizes[0] == 1, (engine, gender, word, sizes)
        assert sizes[-1] > 1, (engine, gender, word, sizes)


def test_the_gender_only_last_tier_is_always_accepted():
    """The floor must not make a new VoiceCastingError reachable: the last tier
    is accepted at any size, so fail-closed behaviour is unchanged.
    """
    entries, _ = load_voice_bank()
    floor = _min_tier_pool()
    for engine in ENGINES:
        for gender in GENDERS:
            for word in _CASTING._TIMBRE_VOCAB:
                sizes = _tier_sizes(entries, engine, gender, word)
                if any(sizes):
                    idx, n = _first_accepted(sizes, floor)
                    assert idx is not None, (engine, gender, word, sizes)


def test_floor_is_folded_into_the_cast_seed():
    """Two floors must never both claim to be policy 3 -- otherwise one policy
    string describes two different byte outputs.
    """
    from nodes._otr_voice_bank import CASTING_POLICY_VERSION, stable_cast_seed

    a = stable_cast_seed(
        episode_seed=7, casting_policy_version=f"{CASTING_POLICY_VERSION}+mtp2",
        char_id="c02", gender="male", timbre=("warm",), role="char_voice",
        age_band="adult")
    b = stable_cast_seed(
        episode_seed=7, casting_policy_version=f"{CASTING_POLICY_VERSION}+mtp3",
        char_id="c02", gender="male", timbre=("warm",), role="char_voice",
        age_band="adult")
    assert a != b


def test_the_floor_is_env_overridable_for_a_live_ab(monkeypatch):
    monkeypatch.setenv("OTR_CAST_MIN_TIER_POOL", "3")
    assert _min_tier_pool() == 3
    monkeypatch.setenv("OTR_CAST_MIN_TIER_POOL", "0")
    assert _min_tier_pool() == 1, "a floor below 1 would accept an empty tier"
    monkeypatch.setenv("OTR_CAST_MIN_TIER_POOL", "not-a-number")
    assert _min_tier_pool() == 2, "junk falls back to the shipped default"


def test_same_human_two_recordings_cannot_both_be_cast():
    """speaker_id: LibriVox's Mark F. Smith has a plain and a grandfatherly take
    in two DIFFERENT files, so ref_path collision cannot catch them.
    """
    from nodes._otr_voice_bank import voice_ref_usage_keys

    entries, _ = load_voice_bank()
    by_id = {e.voice_ref_id: e for e in entries}
    for a_id, b_id in (
        ("vz_pd_librivox_mark_f_smith", "vz_pd_librivox_mark_f_smith_elder"),
        ("cb_pd_librivox_mark_f_smith", "cb_pd_librivox_mark_f_smith_elder"),
        ("vz_ljspeech", "vz_pd_ljspeech_linda_johnson"),
        ("cb_ljspeech", "cb_pd_ljspeech_linda_johnson"),
    ):
        a, b = by_id[a_id], by_id[b_id]
        assert a.ref_path != b.ref_path, (a_id, b_id, "premise: different files")
        assert voice_ref_usage_keys(a) & voice_ref_usage_keys(b), (a_id, b_id)


def test_a_ref_without_a_speaker_id_is_its_own_speaker():
    from nodes._otr_voice_bank import voice_ref_usage_keys

    entries, _ = load_voice_bank()
    plain = next(e for e in entries if not e.speaker_id)
    assert not any(k.startswith("speaker:") for k in voice_ref_usage_keys(plain))
