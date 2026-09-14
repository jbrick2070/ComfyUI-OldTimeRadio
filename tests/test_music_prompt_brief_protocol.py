"""Music cue prompts route through the Meta brief protocol (clean-break 1c).

nodes/_otr_music_prompt.compose_music_prompt is the single source of truth for
theme-music cue prompts. It reads the brief via the brief-reader protocol
(_otr_brief_reader._read_brief_field) -- the same propagating creative brief the
visual slots consume -- and layers cue-specific shaping on top. These pins lock
that the composer pulls mood + setting + period from the brief (not a local
template) and degrades gracefully when the brief is absent.
"""
from __future__ import annotations

from nodes._otr_music_prompt import CUE_DURATIONS, compose_music_prompt


def _meta_full() -> dict:
    return {
        "story_brief_status": "ok",
        "story_brief_terms": {
            "setting": ["interrogation room", "steel table"],
            "lighting": ["bare bulb"],
            "atmosphere": ["sweat", "smoke", "tense"],
        },
        "music_mood_terms": ["sombre", "uneasy", "menacing"],
    }


def test_mood_from_brief_v2_field():
    prompt, _ = compose_music_prompt(_meta_full(), "opening")
    assert "sombre" in prompt and "uneasy" in prompt and "menacing" in prompt


def test_mood_sliced_to_top_three():
    meta = _meta_full()
    meta["music_mood_terms"] = ["sombre", "uneasy", "menacing", "frantic"]
    prompt, _ = compose_music_prompt(meta, "opening")
    assert "frantic" not in prompt


def test_setting_clause_from_brief():
    prompt, _ = compose_music_prompt(_meta_full(), "opening")
    assert "evokes" in prompt
    assert "interrogation room" in prompt or "steel table" in prompt


def test_period_idiom_from_the_story_palette():
    """The period used to be a `gen_params_initial.period_voice` overlay that
    nothing ever produced; since 2026-09-11 the story palette owns it."""
    prompt, _ = compose_music_prompt(_meta_full(), "opening")
    assert "1940s radio drama orchestra" in prompt
    meta = dict(_meta_full(), source_bank="shakespeare", source_meta={"year": "c. 1595"})
    prompt, _ = compose_music_prompt(meta, "opening")
    assert "Elizabethan consort music" in prompt


def test_atmosphere_fallback_when_no_music_mood():
    meta = _meta_full()
    meta["music_mood_terms"] = []
    prompt, _ = compose_music_prompt(meta, "opening")
    # Falls through to story_brief_terms.atmosphere.
    assert "sweat" in prompt or "smoke" in prompt or "tense" in prompt


def test_neutral_default_and_tail_when_brief_empty():
    prompt, dur = compose_music_prompt({}, "opening")
    assert "atmospheric" in prompt
    assert prompt.endswith("instrumental only, no dialogue, no vocals")
    assert dur == CUE_DURATIONS["opening"]


def test_cue_character_and_duration_per_slot():
    for slot, marker in (
        ("opening", "rising overture"),
        ("closing", "resolving to a warm held chord"),
        ("interstitial", "melodic bridge"),
    ):
        prompt, dur = compose_music_prompt(_meta_full(), slot)
        assert marker in prompt
        assert dur == CUE_DURATIONS[slot]


def test_reads_via_brief_protocol_not_direct_meta():
    """The composer must accept a parent dict carrying `meta` (the protocol
    reader normalizes both shapes) -- proving it routes through the brief
    reader rather than poking a fixed top-level key."""
    parent = {"meta": _meta_full()}
    # music_mood_terms lives under parent["meta"]; the protocol reader resolves it.
    prompt, _ = compose_music_prompt(parent.get("meta") or {}, "opening")
    assert "sombre" in prompt


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))


def test_the_three_cues_do_not_collapse_into_one_prompt():
    """Opening, interstitial and closing must not ask the engine for the same
    thing.

    THE REGRESSION THIS PINS, 2026-09-13. `compose_brief_engine_prompt` read no
    cue at all, so on a DERIVED row all three cues composed a byte-identical
    string and only the seed and the duration differed -- the ledger recorded
    three different asks and the model heard one. It went unnoticed because the
    change that shipped the brief form to every engine was verified along the
    ENGINE axis (SA3 against MusicGen) and never along the CUE axis. Caught by
    a contrarian pass, not by the suite.

    An AUTHORED row was never affected: it is per-cue already.
    """
    import nodes._otr_music_prompt as MP

    for bank in ("scifi_news_pro", "public_domain", "media_archive",
                 "original", "shakespeare"):
        meta = {"source_bank": bank,
                "story_brief_terms": {"atmosphere": ["tense"],
                                      "setting": ["a dead orbital city"]}}
        texts = {cue: MP.compose_brief_engine_prompt(meta, "", cue_id=cue).text
                 for cue in ("opening", "interstitial", "closing")}
        assert len(set(texts.values())) == 3, (bank, texts)


def test_a_groove_palette_is_never_told_to_resolve():
    """The cue clause is palette-aware for the reason the row text is: telling
    a 128 BPM drum machine to "resolve" is the contradiction that tore the
    techno and house cues on 2026-09-12."""
    import nodes._otr_music_palette as P
    import nodes._otr_music_prompt as MP

    for bank in ("scifi_news_pro", "public_domain"):
        meta = {"source_bank": bank}
        assert P.story_palette(meta).groove_arc is True, bank
        closing = MP.compose_brief_engine_prompt(meta, "", cue_id="closing").text
        assert "resolv" not in closing.lower(), (bank, closing)
        assert "downbeat" in closing.lower(), (bank, closing)


def test_an_unknown_cue_contributes_nothing_and_never_raises():
    """The short form promises never to raise. `inter_NN` is the real case --
    an episode can carry several interstitials -- and anything else simply
    contributes no clause rather than a KeyError, which is what
    `compose_music_prompt` does."""
    import nodes._otr_music_prompt as MP

    meta = {"source_bank": "original"}
    assert "short bridge" in MP.compose_brief_engine_prompt(
        meta, "", cue_id="inter_07").text
    for junk in (None, "", "bogus", 7, [], {}):
        text = MP.compose_brief_engine_prompt(meta, "", cue_id=junk).text
        assert text and text.endswith("instrumental, no vocals"), junk


def _led(scenes=(), shots=(), meta=None):
    return {"meta": meta or {"source_bank": "original"},
            "scenes": [{"description": d} for d in scenes],
            "shots": list(shots)}


def test_the_cue_carries_the_scene_the_story_opens_and_closes_on():
    """Operator, 2026-09-13: "a story aware cinematic flavour based on the
    opening and closing, or beat of a story if interstitial"."""
    import nodes._otr_music_prompt as MP

    led = _led(scenes=("a sunlit toy playground", "a toy kitchen at dusk"))
    assert MP.cue_story_flavour(led, {"placement": "opening"}) == "a sunlit toy playground"
    assert MP.cue_story_flavour(led, {"placement": "closing"}) == "a toy kitchen at dusk"


def test_an_interstitial_takes_the_shot_it_is_anchored_to():
    """An interstitial sits at a BEAT, not at an end, and the music row names
    that beat through `anchor_line_id` ("shot_NNN_music")."""
    import nodes._otr_music_prompt as MP

    led = _led(scenes=("first scene", "last scene"),
               shots=({"shot_id": "shot_001", "description": "the grass beyond the fence"},
                      {"shot_id": "shot_002", "description": "a humming green box"}))
    got = MP.cue_story_flavour(
        led, {"placement": "interstitial", "anchor_line_id": "shot_002_music"})
    assert got == "a humming green box"


def test_a_stage_note_is_not_a_scene_and_is_dropped():
    """Roughly a third of ledgers on disk describe a scene as what HAPPENS --
    "the team discusses the sensor trial results". Music cannot play plot, so
    those contribute nothing rather than being handed over."""
    import nodes._otr_music_prompt as MP

    for note in ("The team discusses the sensor trial results.",
                 "Introduces the tension of the debate.",
                 "Provides resolution and finality to the drama."):
        assert MP.cue_story_flavour(_led(scenes=(note,)), {"placement": "opening"}) == ""


def test_a_trimmed_flavour_never_ends_on_a_dangling_word():
    """A hard word-count cut produced "the area behind Whiskers by the", which
    asks the model to finish a preposition instead of scoring a scene."""
    import nodes._otr_music_prompt as MP

    long_scene = ("a toy kitchen with a painted flame stove and the grass where "
                  "the boxes settled behind the")
    got = MP.cue_story_flavour(_led(scenes=(long_scene,)), {"placement": "opening"})
    assert got, "a long scene should still contribute something"
    assert got.split()[-1].lower() not in MP._FLAVOUR_DANGLING, got
    assert len(got.split()) <= MP._FLAVOUR_MAX_WORDS


def test_the_flavour_is_optional_and_never_raises():
    """Two thirds of the ledgers on disk carry no scene description at all, so
    this can never be what makes the cues differ -- the arc clause is."""
    import nodes._otr_music_prompt as MP

    for junk in (None, {}, [], "x", {"scenes": None}, {"scenes": [{"description": ""}]}):
        assert MP.cue_story_flavour(junk, {"placement": "opening"}) == ""
        assert MP.cue_story_flavour(_led(scenes=("a room",)), junk) == ""
    meta = {"source_bank": "original"}
    assert MP.compose_brief_engine_prompt(meta, "", cue_id="opening",
                                          story_flavour="").text
