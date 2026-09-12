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
import itertools

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
    # A DECLARED BANK GENRE BEATS THE PERIOD BAND (operator, 2026-09-12: "sci-fi
    # news is Detroit techno, media archive will be jazz quartet, original will
    # be salsa, public domain Chicago house"). Read year-first, every one of
    # those banks would have come back a period ensemble and the instruction
    # would have had no audible effect -- most public-domain sources are
    # Victorian, so "public domain is Chicago house" has to outrank 1897.
    ({"source_bank": "scifi_news_pro"}, "detroit_techno"),
    ({"source_bank": "media_archive"}, "jazz_quartet"),
    ({"source_bank": "original"}, "salsa_conjunto"),
    ({"source_bank": "public_domain"}, "chicago_house"),
    ({"source_bank": "public_domain", "source_meta": {"year": 1750}}, "chicago_house"),
    ({"source_bank": "public_domain", "source_meta": {"year": 1897}}, "chicago_house"),
    ({"source_bank": "media_archive", "source_meta": {"year": 1948}}, "jazz_quartet"),
    # Shakespeare keeps its consort: it was never given a genre, so it still
    # routes by year and then by bank, exactly as before.
    ({"source_bank": "shakespeare", "source_meta": {"year": "c. 1595"}}, "early_consort"),
    ({"source_bank": "shakespeare"}, "early_consort"),
    # And so does every bank without a declared genre.
    ({"source_bank": "my_story"}, "radio_orchestra"),
    ({"source_bank": "my_story", "source_meta": {"year": 1750}}, "baroque_chamber"),
    ({"source_bank": "my_story", "source_meta": {"year": 1897}}, "romantic_chamber"),
    ({"source_bank": "my_story", "source_meta": {"year": 1975}}, "electric_combo"),
    ({}, "radio_orchestra"),
    ({"source_bank": "my_story", "source_meta": {"year": "unknown"}}, "radio_orchestra"),
])
def test_the_palette_follows_the_bank_genre_then_the_year_then_the_house(meta, key):
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
    assert devices == ["minor key, low brass swells, slow rising strings",
                       "shimmering sustained strings, soft celesta colour"]
    assert P.mood_devices(["sombre", "uneasy", "menacing"])[0].startswith("a slow cello")


def test_one_pace_per_cue_a_contradictory_second_device_is_dropped():
    """The defect the operator heard on 2026-09-12: "melancholic, playful"
    asked one 12-second cue to be slow/held AND pizzicato/brushed-drums, and
    the model answered with a loop. A second device now joins only when its
    pace agrees."""
    both = P.mood_devices(["folk", "melancholic", "playful"])
    assert both == ["a slow cello line over held minor chords"], both
    assert P.mood_pace(["folk", "melancholic", "playful"]) == "slow"
    # two SLOW moods still stack -- the rule drops contradictions, not depth
    agree = P.mood_devices(["melancholic", "mysterious"])
    assert len(agree) == 2, agree
    # a neutral device joins either pace
    assert len(P.mood_devices(["heroic", "playful"])) == 2


def test_every_cue_is_told_how_fast_to_play():
    """Duration never reaches the text, so without this the model picks its
    own pace -- and at 4-12 seconds it picks a repeating figure."""
    assert P.tempo_phrase(["melancholic"]) == "slow tempo, unhurried, expressive rubato"
    assert P.tempo_phrase(["frantic"]) == "moving tempo, flowing line"
    assert P.tempo_phrase(["heroic"]) == "unhurried tempo, broad phrasing"
    assert P.tempo_phrase([]) == "slow tempo, unhurried, expressive rubato"
    assert P.tempo_phrase(None) and P.tempo_phrase(["zzz"])


def test_dread_is_slow_and_RIGID_where_grief_is_slow_and_free():
    """Fable, 2026-09-12: "unhurried, expressive rubato" is right for grief
    and wrong for dread. Suspense is slow AND rigid -- rubato removes the
    pulse, and the pulse is the thing a listener feels. Both remain "slow"
    for the contradiction rule; only the tempo words differ."""
    assert P.tempo_phrase(["tense"]) == "slow tempo, sustained and taut, no rubato"
    assert P.tempo_phrase(["foreboding"]) == "slow tempo, sustained and taut, no rubato"
    assert P.tempo_phrase(["grief"]) == "slow tempo, unhurried, expressive rubato"
    assert P.mood_pace(["tense"]) == P.mood_pace(["grief"]) == "slow"
    assert len(P.mood_devices(["tense", "grief"])) == 2, "they do not contradict"


def test_the_majority_mood_sets_the_pace_not_whichever_word_came_first():
    """Fable, 2026-09-12: the brief lists its mood words in no particular
    order, so letting the first decide gave a comedy a rubato lullaby when
    "pastoral" happened to precede "playful"."""
    comedy = ["playful", "merry", "pastoral"]
    assert P.mood_pace(comedy) == "fast", "two of three are playful"
    assert P.tempo_phrase(comedy) == "moving tempo, flowing line"
    # the same words in a different order reach the same verdict
    assert P.mood_pace(["pastoral", "playful", "merry"]) == "fast"
    # A GENUINE TIE RESOLVES THE SAME WAY BOTH WAYS ROUND (codex,
    # 2026-09-12). It used to fall to whichever word the writer typed
    # first, which is the very thing this test is named after; it now
    # falls to `_PACE_TIE_ORDER`, where the slower reading wins: at four to
    # twelve seconds the fast one is what comes back as a repeating figure.
    assert P.mood_pace(["melancholic", "playful"]) == "slow"
    assert P.mood_pace(["playful", "melancholic"]) == "slow"


def test_the_same_moods_give_the_same_pace_and_tempo_in_any_order():
    """codex r1, 2026-09-12: the majority-pace rule removed the brief's
    word order from the PACE and left it in the TEMPO, where it could also
    contradict the pace -- ["heroic","playful"] resolved to a fast cue and
    then asked for heroic's unhurried phrasing.

    The device LIST still follows the brief on purpose (see `mood_devices`),
    so this asserts the two whole-cue facts, not the list."""
    for brief in (["heroic", "playful"], ["grief", "tense"],
                  ["tense", "warm", "grand"], ["playful", "merry", "pastoral"],
                  ["melancholic", "playful"], ["moonlit", "dread", "secretive"],
                  ["urgent", "warm"], ["tense", "tense", "grief"],
                  ["grief", "tense", "playful", "grand"]):
        verdicts = {(P.mood_pace(list(order)), P.tempo_phrase(list(order)))
                    for order in itertools.permutations(brief)}
        assert len(verdicts) == 1, (brief, verdicts)


def test_the_tempo_may_never_contradict_the_pace():
    """The defect codex found, stated as the invariant it breaks: a FAST
    cue asking for unhurried broad phrasing because a neutral device
    happened to be listed first."""
    assert P.mood_pace(["heroic", "playful"]) == "fast"
    assert P.tempo_phrase(["heroic", "playful"]) == "moving tempo, flowing line"
    assert P.tempo_phrase(["playful", "heroic"]) == "moving tempo, flowing line"
    # a slow cue never takes the one moving phrase
    assert P.tempo_phrase(["grief", "tense"]) == "slow tempo, sustained and taut, no rubato"
    # ... and the tempo is the brief's, not the surviving devices': the
    # limit drops a device from the PROMPT and must not change the tempo
    assert P.tempo_phrase(["tense", "warm", "grand"]) == P.tempo_phrase(["grand", "warm", "tense"])


def test_a_limit_below_one_still_yields_a_device():
    """A cue with no device has no prompt, so the floor is one however the
    caller asks -- and junk cannot raise inside a render."""
    assert len(P.mood_devices(["tense", "grief"], limit=0)) == 1
    assert len(P.mood_devices(["tense", "grief"], limit=-5)) == 1
    assert len(P.mood_devices(["tense", "grief"], limit=None)) == 2
    assert len(P.mood_devices(["tense", "grief"], limit="two")) == 2
    # the coercions, stated rather than discovered (codex r2)
    assert len(P.mood_devices(["tense", "grief"], limit=1.9)) == 1, "int(), not round()"
    assert len(P.mood_devices(["tense", "grief"], limit=True)) == 1, "True is 1"
    assert len(P.mood_devices(["tense", "grief"], limit=99)) == 2, "no more than matched"


def test_no_tempo_phrase_asks_for_a_pulse_a_beat_or_a_steady_anything():
    """MEASURED 2026-09-12, and it cost a leg: the tension tempo phrase first
    read "a held and unwavering pulse" and produced the worst loopiness of the
    campaign (0.864 on shadows_in_the_mist, against 0.485 for the episode that
    started the complaint). A word that names a repeating pulse gets a
    repeating pulse; suspense has to be carried by sustain instead."""
    banned = ("pulse", "beat", "steady", "metronome", "driving", "throb",
              "ostinato", "loop", "rhythmic", "on the downbeat")
    phrases = [t for _p, _d, _pace, t in P._MOOD_DEVICES] + list(P._PACE_TEMPO.values())
    offenders = [(w, t) for t in phrases for w in banned if w in t.lower()]
    assert not offenders, offenders


def test_no_device_and_no_palette_names_a_drum():
    """A drum in a 4-to-12-second background cue can only be a loop. Every
    percussion phrase that used to live here was in a cue the operator heard
    as a tape deck."""
    percussion = ("drum", "snare", "timpani", "pizzicato", "brushed",
                  "percussion", "beat", "tremolo")
    haystack = " ".join(d for _p, d, _pace, _t in P._MOOD_DEVICES).lower()
    haystack += " " + " ".join(
        pal.instruments.lower() for pal in (
            P.HOUSE_PALETTE, P.EARLY_CONSORT, P.BAROQUE_CHAMBER,
            P.ROMANTIC_CHAMBER, P.ELECTRIC_COMBO, P.SCIFI_ORCHESTRA))
    hits = [w for w in percussion if w in haystack]
    assert not hits, hits


def test_short_stems_are_bounded_so_warm_is_not_war_and_moonlight_is_not_light():
    assert P.mood_devices(["warm"]) == ["major key, legato strings, soft woodwinds"]
    assert P.mood_devices(["war"]) == ["urgent strings climbing over a restless bass line"]
    assert P.mood_devices(["moonlight"]) == ["shimmering sustained strings, soft celesta colour"]
    assert P.mood_devices(["lighthearted"]) == ["a light dancing woodwind melody, bright major colour"]
    # despair is not warmth: the stems that used to swallow their opposites
    sad = "a slow cello line over held minor chords"
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
        "minor key, low brass swells, slow rising strings"], "one device per idea"
    # tense and warm are both SLOW, so both land; grand is neutral and would
    # land too, but the limit stops at two unless it is raised
    assert len(P.mood_devices(["tense", "warm", "grand"], limit=3)) == 3


def test_nothing_in_the_palette_names_the_withdrawn_texture():
    text = " ".join(
        p.instruments + " " + p.idiom
        for p in (P.HOUSE_PALETTE, P.EARLY_CONSORT, P.BAROQUE_CHAMBER,
                  P.ROMANTIC_CHAMBER, P.ELECTRIC_COMBO, P.SCIFI_ORCHESTRA)
    ) + " " + " ".join(d for _p, d, _pace, _t in P._MOOD_DEVICES) + " " + P.DEFAULT_DEVICE
    low = text.lower()
    assert not [w for w in _NOISE_WORDS if w in low], low


def test_only_my_story_may_name_its_own_music():
    """Operator, 2026-09-12: "only 'my story' allows an original music prompt."

    Every other bank has a fixed musical identity, and that is the point of
    having one -- the sci-fi news lane being Detroit techno every week is what
    makes it recognisable, and a Shakespeare episode scored as surf rock is not
    a feature. My Story is the bring-your-own lane, so it takes a bring-your-own
    score exactly as it already takes an authored prompt."""
    for bank in ("shakespeare", "scifi_news_pro", "media_archive", "original",
                 "public_domain"):
        meta = {"source_bank": bank, "music_style": "surf rock"}
        assert P.story_palette(meta).key != "custom", bank
    mine = {"source_bank": "my_story", "music_style": "surf rock"}
    assert P.story_palette(mine).key == "custom"
    assert P.story_palette(mine).instruments == "surf rock"
    # blank, whitespace and junk all mean "use the bank default"
    for blank in ("", "   ", None):
        assert P.story_palette({"source_bank": "my_story",
                                "music_style": blank}).key == "radio_orchestra"


def test_a_typed_style_that_names_a_groove_is_not_told_to_avoid_one():
    """The self-cancelling request, guarded. A user who types "drum and bass"
    and receives the underscore's negative -- which bans loop, ostinato, drum
    machine, metronome and beat -- has asked for a genre and forbidden it in the
    same breath. That is what tore the cues on 2026-09-12, and it is worse than
    the opposite error, so anything that sounds like a groove is treated as
    one."""
    for text in ("drum and bass", "Detroit techno", "surf rock", "salsa",
                 "gamelan orchestra", "bagpipes and a marching drum", "909 workout"):
        assert P.custom_palette(text).rhythmic is True, text
    for text in ("solo cello", "Gregorian chant", "string quartet", "ambient drone"):
        assert P.custom_palette(text).rhythmic is False, text


def test_custom_palette_never_raises_and_never_runs_away():
    """It runs inside a render and takes whatever a person typed."""
    assert P.custom_palette(None) is None and P.custom_palette("") is None
    assert P.custom_palette("   	 ") is None
    assert P.custom_palette(42).instruments == "42"
    long = P.custom_palette("cello " * 400)
    assert len(long.instruments) <= 200
    assert P.custom_palette("  spaced   out   words ").instruments == "spaced out words"


def test_every_bank_genre_leads_with_its_rhythm_section_and_is_marked_rhythmic():
    """A genre bank inverts this module's own leading rule on purpose: for
    techno the drum machine IS the subject, not the backing. The `rhythmic`
    flag is what makes that inversion explicit rather than accidental, and it
    is the same flag that picks the negative prompt."""
    for palette, first in ((P.DETROIT_TECHNO, "Roland TR-909"),
                           (P.JAZZ_QUARTET, "brushed drums"),
                           (P.SALSA_CONJUNTO, "congas"),
                           (P.CHICAGO_HOUSE, "Roland TR-707")):
        assert palette.rhythmic is True, palette.key
        assert palette.instruments.startswith(first), palette.key
    for palette in (P.HOUSE_PALETTE, P.EARLY_CONSORT, P.BAROQUE_CHAMBER,
                    P.ROMANTIC_CHAMBER, P.ELECTRIC_COMBO, P.SCIFI_ORCHESTRA):
        assert palette.rhythmic is False, palette.key
