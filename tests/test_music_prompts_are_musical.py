"""Every music prompt the pack composes is MUSICAL and story-relevant
(operator, 2026-09-11 evening, after `moonlit_deception`: "the 'music' needs to
be improved, doesn't sound like music ... sounds like radio hiss, which is what
we asked. So I am asking to go over ALL musical prompts and make them more
musical ... ideally it is relevant to the story").

Two products from one composer (`_otr_music_prompt`): the ROW text the ledger
stores and hashes, and the ENGINE prompt every adapter hears -- the story
palette's instruments and a clean production anchor in front of the row text,
plus the one negative prompt. No engine prepends its own texture any more.
"""
from __future__ import annotations

import inspect
import itertools

import pytest

from nodes import _otr_music_palette as P
from nodes import _otr_music_prompt as MP
from nodes._otr_audio_engines import eng_stable_audio_3 as SA3

_NOISE_WORDS = ("hiss", "static", "noise", "crackle", "squelch", "tape",
                "vintage", "lo-fi", "lofi", "degraded", "pristine")
_BANKS = ("shakespeare", "public_domain", "original", "media_archive",
          "my_story", "scifi_news_pro")
_YEARS = (None, "c. 1595", 1750, 1897, 1935, 1975)


def _meta(bank, year, moods=("tense", "moonlit", "secretive"),
          setting=("forest", "moonlit night")):
    meta = {"story_brief_status": "ok",
            "story_brief_terms": {"setting": list(setting)},
            "music_mood_terms": list(moods),
            "source_bank": bank}
    if year is not None:
        meta["source_meta"] = {"year": year}
    return meta


def _instrument_count(text):
    low = text.lower()
    return sum(1 for name in (
        "lute", "viol", "recorder", "harpsichord", "strings", "brass", "clarinet",
        "vibraphone", "bass", "piano", "quartet", "horn", "oboe", "bassoon",
        "guitar", "organ", "drum", "theremin", "timpani", "cello", "harp",
        "celesta", "flute", "woodwinds", "snare") if name in low)


@pytest.mark.parametrize("bank,year,cue", list(itertools.product(_BANKS, _YEARS, MP.CUE_DURATIONS)))
def test_every_bank_period_and_cue_composes_music_not_texture(bank, year, cue):
    row, duration = MP.compose_music_prompt(_meta(bank, year), cue)
    engine = MP.compose_engine_prompt(_meta(bank, year), row)
    for text in (row, engine.text):
        low = text.lower()
        assert not [w for w in _NOISE_WORDS if w in low], text
    assert _instrument_count(engine.text) >= 2, engine.text
    assert row.endswith("instrumental only, no dialogue, no vocals")
    assert engine.text.endswith("instrumental only, no dialogue, no vocals")
    assert engine.text.startswith(P.story_palette(_meta(bank, year)).instruments)
    assert MP.PRODUCTION_ANCHOR in engine.text
    assert duration == MP.CUE_DURATIONS[cue]
    assert len(engine.text) <= MP.ENGINE_PROMPT_MAX_CHARS


def test_the_row_text_reads_as_a_musical_instruction_about_this_story():
    row, _ = MP.compose_music_prompt(_meta("shakespeare", "c. 1595",
                                           moods=("enchanted", "playful", "tension-building")),
                                     "opening")
    assert row.startswith("enchanted, playful, tension-building"), "the brief's words lead"
    assert "celesta, harp glissandi, shimmering strings" in row
    assert "pizzicato strings, bright woodwinds, brushed drums" in row
    assert "Elizabethan consort music" in row
    assert "evokes forest, moonlit night" in row
    assert "a rising overture that settles into a steady theme" in row


def test_the_engine_prompt_leads_with_the_ensemble_and_a_clean_recording():
    meta = _meta("shakespeare", "c. 1595")
    row, _ = MP.compose_music_prompt(meta, "closing")
    engine = MP.compose_engine_prompt(meta, row)
    assert engine.text.startswith("lute, viol consort, recorder, harpsichord, "
                                  "clearly recorded, clean balanced studio mix, natural room. ")
    assert engine.text.endswith(row)
    assert engine.palette_key == "early_consort"
    assert engine.negative == MP.NEGATIVE_PROMPT_DEFAULT


def test_the_negative_prompt_names_the_withdrawn_texture_and_speech():
    low = MP.NEGATIVE_PROMPT_DEFAULT.lower()
    for word in ("hiss", "static", "noise", "vocals", "speech", "clipping"):
        assert word in low


def test_an_authored_prompt_survives_verbatim_inside_the_engine_prompt():
    authored = "slow open on a lonely trumpet over a city at night"
    engine = MP.compose_engine_prompt(_meta("my_story", None), authored)
    assert engine.text.endswith(". " + authored)
    assert engine.text.startswith(P.HOUSE_PALETTE.instruments)


def test_the_engine_prompt_is_capped_below_the_smallest_engine_budget_without_raising():
    long_row = ", ".join("a long clause number %d" % i for i in range(120))
    assert len(long_row) > MP.ENGINE_PROMPT_MAX_CHARS
    engine = MP.compose_engine_prompt(_meta("original", None), long_row)
    assert len(engine.text) <= MP.ENGINE_PROMPT_MAX_CHARS
    assert engine.text.startswith(P.HOUSE_PALETTE.instruments)
    assert engine.text.endswith("clause number %d" % (
        int(engine.text.rsplit("number ", 1)[-1])))  # cut at a clause boundary
    assert not engine.text.endswith(",")


def test_an_overflowing_composed_row_keeps_its_instrumental_tail():
    meta = _meta("original", None,
                 moods=tuple("mood word number %d" % i for i in range(3)),
                 setting=tuple("a setting phrase that runs long %d" % i for i in range(2)))
    row, _ = MP.compose_music_prompt(meta, "opening")
    long_row = row.replace("evokes", "evokes " + ", ".join(
        "an extra scenic clause %d" % i for i in range(60)) + ",")
    assert len(long_row) > MP.ENGINE_PROMPT_MAX_CHARS
    assert long_row.endswith("instrumental only, no dialogue, no vocals")
    engine = MP.compose_engine_prompt(meta, long_row)
    assert len(engine.text) <= MP.ENGINE_PROMPT_MAX_CHARS
    assert engine.text.endswith("instrumental only, no dialogue, no vocals")
    assert engine.text.startswith(P.HOUSE_PALETTE.instruments)


def test_junk_meta_still_composes_both_products():
    for junk in (None, {}, [], "x", {"source_meta": "c. 1595", "music_mood_terms": 3}):
        row, _ = MP.compose_music_prompt(junk if isinstance(junk, dict) else {}, "opening")
        engine = MP.compose_engine_prompt(junk, row)
        assert engine.text and engine.negative and engine.palette_key


def test_the_dead_period_voice_overlay_is_gone():
    """`gen_params_initial.period_voice` had zero producers (grep, 2026-09-11)
    and the palette now owns the period. Half-removed is the defect."""
    source = inspect.getsource(MP)
    assert "period_voice" not in source


# --------------------------------------------------------------------------- #
# the SA3 engine no longer prepends its own era anchor
# --------------------------------------------------------------------------- #
def test_sa3_sends_the_composed_prompt_verbatim_and_uses_the_composer_negative():
    source = inspect.getsource(SA3)
    for gone in ("_sa3_augment_prompt", "_SA3_PERIOD_GENRE", "_SA3_DEFAULT_GENRE",
                 "_SA3_NEG_DEFAULT", "analog tape warmth", "modern pristine mix"):
        assert gone not in source, gone
    body = inspect.getsource(SA3.StableAudio3Engine.generate_clip)
    assert "pos_text = str(prompt or \"\").strip()" in body
    assert "NEGATIVE_PROMPT_DEFAULT" in body
    assert "OTR_SA3_NEG_PROMPT" in body, "the operator's one override stays"
    assert "placement or prompt" in body, "the cue's placement names the window"


def test_sa3_window_follows_the_placement_not_a_word_in_the_prompt():
    ctx = 12.0
    assert SA3._sa3_clip_window("opening", 12.0, ctx) == (0.0, 12.0)
    start, total = SA3._sa3_clip_window("closing", 8.0, ctx)
    assert abs(start - 4.0) < 1e-6 and total == 12.0
    start, total = SA3._sa3_clip_window("interstitial", 4.0, ctx)
    assert abs(start - 4.0) < 1e-6 and total == 12.0


# --------------------------------------------------------------------------- #
# every adapter accepts the two keyword arguments the theme node now passes
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("module,cls", [
    ("eng_stable_audio_3", "StableAudio3Engine"),
    ("eng_musicgen", None), ("eng_google_lyria", None),
    ("eng_cloud_sonilo", None), ("eng_stable_audio", None),
])
def test_every_music_adapter_accepts_placement_and_negative_prompt(module, cls):
    import importlib
    mod = importlib.import_module("nodes._otr_audio_engines." + module)
    classes = [getattr(mod, cls)] if cls else [
        obj for name, obj in vars(mod).items()
        if inspect.isclass(obj) and hasattr(obj, "generate_clip")
        and obj.__module__ == mod.__name__]
    assert classes, module
    for klass in classes:
        params = inspect.signature(klass.generate_clip).parameters
        assert params["placement"].kind is inspect.Parameter.KEYWORD_ONLY, klass
        assert params["negative_prompt"].kind is inspect.Parameter.KEYWORD_ONLY, klass
