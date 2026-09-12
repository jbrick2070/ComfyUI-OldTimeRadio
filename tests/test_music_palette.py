"""The story -> ensemble step behind every music prompt (operator, 2026-09-11:
"make them more musical ... ideally it is relevant to the story").

`_otr_music_palette.story_palette` names the instruments and idiom for the
story's period and bank; `mood_devices` turns the brief's mood words into
musical devices. Both are total over any meta shape (a dead render over a
malformed year is not an OOM) and keyed only on ledger-stable facts -- the
visual style roll, drawn from OS entropy, is deliberately not an input.
"""
from __future__ import annotations

import inspect

import pytest

from nodes import _otr_music_palette as P

_NOISE_WORDS = ("hiss", "static", "noise", "crackle", "squelch", "tape",
                "vintage", "lo-fi", "lofi", "degraded", "radio texture")


@pytest.mark.parametrize("raw,expected", [
    (1606, 1606), ("1606", 1606), ("c. 1595", 1595), ("1895-1897", 1895),
    ("first performed 1611, printed 1623", 1611), ("unknown", None),
    ("", None), (None, None), (12, None), ("year 999", None),
])
def test_the_year_is_read_leniently_and_never_raises(raw, expected):
    assert P.year_of({"year": raw}) == expected


def test_the_year_falls_back_to_other_date_fields_and_junk_shapes():
    assert P.year_of({"date": "1897"}) == 1897
    assert P.year_of({"published": "London, 1818"}) == 1818
    assert P.year_of(None) is None and P.year_of("1606") is None
    assert P.year_of({"year": {"nested": 1606}}) is None


@pytest.mark.parametrize("meta,key", [
    ({"source_bank": "shakespeare", "source_meta": {"year": "c. 1595"}}, "early_consort"),
    ({"source_bank": "shakespeare"}, "early_consort"),
    ({"source_bank": "public_domain", "source_meta": {"year": 1750}}, "baroque_chamber"),
    ({"source_bank": "public_domain", "source_meta": {"year": 1897}}, "romantic_chamber"),
    ({"source_bank": "public_domain", "source_meta": {"year": 1920}}, "radio_orchestra"),
    ({"source_bank": "public_domain", "source_meta": {"year": 1975}}, "electric_combo"),
    ({"source_bank": "scifi_news_pro"}, "scifi_orchestra"),
    ({"source_bank": "original"}, "radio_orchestra"),
    ({"source_bank": "media_archive"}, "radio_orchestra"),
    ({"source_bank": "my_story"}, "radio_orchestra"),
    ({}, "radio_orchestra"),
    ({"source_bank": "public_domain", "source_meta": {"year": "unknown"}}, "radio_orchestra"),
])
def test_the_palette_follows_the_year_then_the_bank_then_the_house(meta, key):
    palette = P.story_palette(meta)
    assert palette.key == key
    assert palette.instruments and palette.idiom


@pytest.mark.parametrize("junk", [None, "shakespeare", 42, [], {"source_meta": "1595"},
                                  {"source_bank": None, "source_meta": None}])
def test_junk_meta_is_the_house_orchestra_never_a_dead_render(junk):
    assert P.story_palette(junk) is P.HOUSE_PALETTE


def test_the_bank_is_read_from_every_shape_the_ledger_has_used():
    assert P.bank_of({"source_bank": "Shakespeare"}) == "shakespeare"
    assert P.bank_of({"bank": "original"}) == "original"
    assert P.bank_of({"story_routing": {"bank": "my_story"}}) == "my_story"
    assert P.bank_of({"source_meta": {"bank": "public_domain"}}) == "public_domain"
    assert P.bank_of({}) == "" and P.bank_of(None) == ""


def test_the_style_roll_is_not_an_input():
    """`style_roll.seed_source` is OS entropy on every ledger: keying the
    ensemble on it would make the same seed play different instruments."""
    assert "style_roll" not in inspect.getsource(P.story_palette)
    assert "style" not in inspect.getsource(P.bank_of)


def test_mood_words_become_musical_devices_in_mood_order():
    devices = P.mood_devices(["tense", "moonlit", "secretive"])
    assert devices == ["minor key, tremolo strings, low brass swells",
                       "celesta, harp glissandi, shimmering strings"]
    assert P.mood_devices(["sombre", "uneasy", "menacing"])[0].startswith("slow cello")


def test_short_stems_are_bounded_so_warm_is_not_war_and_moonlight_is_not_light():
    assert P.mood_devices(["warm"]) == ["major key, legato strings, soft woodwinds"]
    assert P.mood_devices(["war"]) == ["driving rhythm, staccato strings, snare accents"]
    assert P.mood_devices(["moonlight"]) == ["celesta, harp glissandi, shimmering strings"]
    assert P.mood_devices(["lighthearted"]) == ["pizzicato strings, bright woodwinds, brushed drums"]
    # despair is not warmth: the stems that used to swallow their opposites
    sad = "slow cello line, muted piano, held minor chords"
    warm = "major key, legato strings, soft woodwinds"
    assert P.mood_devices(["hopeless"]) == [sad]
    assert P.mood_devices(["loveless"]) == [sad]
    assert P.mood_devices(["hopeful"]) == [warm]
    assert P.mood_devices(["lovely"]) == [warm]
    assert P.mood_devices(["love"]) == [warm]


def test_no_mood_or_junk_still_yields_a_musical_instruction():
    assert P.mood_devices([]) == [P.DEFAULT_DEVICE]
    assert P.mood_devices(None) == [P.DEFAULT_DEVICE]
    assert P.mood_devices(["zzz", 42, None, ""]) == [P.DEFAULT_DEVICE]
    assert P.mood_devices(["tense", "tense", "dread"]) == [
        "minor key, tremolo strings, low brass swells"], "one device per idea"
    assert len(P.mood_devices(["tense", "warm", "grand"], limit=3)) == 3


def test_nothing_in_the_palette_names_the_withdrawn_texture():
    text = " ".join(
        p.instruments + " " + p.idiom
        for p in (P.HOUSE_PALETTE, P.EARLY_CONSORT, P.BAROQUE_CHAMBER,
                  P.ROMANTIC_CHAMBER, P.ELECTRIC_COMBO, P.SCIFI_ORCHESTRA)
    ) + " " + " ".join(d for _, d in P._MOOD_DEVICES) + " " + P.DEFAULT_DEVICE
    low = text.lower()
    assert not [w for w in _NOISE_WORDS if w in low], low
