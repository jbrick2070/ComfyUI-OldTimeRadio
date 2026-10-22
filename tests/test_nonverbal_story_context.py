"""The nonverbal director's PRIVATE story context (2026-08-29), pinned.

The builder used to ``del meta`` and hand the model nothing but the bare line,
so a demand under threat and a weather report earned the same fidget. It now
reads the episode logline and key objects from meta, and each beat's
beat_intent/traits off the beat row -- all INPUT-ONLY: the response schema,
the quote filter, and every ban are unchanged, and the literal line still
appears nowhere except under ``line_context_do_not_quote``.

CPU-safe and pure: no LLM call is made -- these tests read the PROMPT STRING.
"""

from __future__ import annotations

import json

from nodes import otr_shot_lock as sl

_LINE = "Play it again, and keep the gain low."


def _ledger():
    return {
        "cast": [{"char_id": "c01", "name": "Vera",
                  "appearance": "a wiry radio operator in a wool coat"}],
        "lines": [{
            "line_id": "b001", "char_id": "c01", "speaker": "VERA",
            "speaker_role": "character",
            "text": _LINE, "dur_s": 3.0,
            "shot_id": "shot_001", "beat_intent":
                "Demands the recording under threat of exposure.",
            "traits": "clipped, exacting",
        }],
    }


def _meta():
    return {
        "produced_story": {"logline": "A mountain that answers back."},
        "key_objects": ["Reel deck", "Gain dial"],
    }


def test_extract_beats_carries_the_story_fields():
    beats = sl.extract_beats(_ledger())
    assert len(beats) == 1
    beat = beats[0]
    assert beat["shot_id"] == "shot_001"
    assert beat["beat_intent"].startswith("Demands the recording")
    assert beat["traits"] == "clipped, exacting"


def test_the_nonverbal_prompt_carries_the_private_context():
    ledger = _ledger()
    beats = sl.extract_beats(ledger)
    prompt = sl._build_nonverbal_batch_prompt(
        beats, _meta(), ledger, "a cliff-top listening station")
    assert "Story (private context): A mountain that answers back." in prompt
    assert "Objects on hand: Reel deck, Gain dial" in prompt
    assert "Demands the recording under threat of exposure." in prompt
    assert "clipped, exacting" in prompt


def test_the_line_reaches_the_model_ONLY_as_do_not_quote_context():
    ledger = _ledger()
    beats = sl.extract_beats(ledger)
    prompt = sl._build_nonverbal_batch_prompt(
        beats, _meta(), ledger, "a cliff-top listening station")
    payload_rows = [json.loads(ln) for ln in prompt.splitlines()
                    if ln.startswith("{")]
    assert len(payload_rows) == 1
    row = payload_rows[0]
    assert row["line_context_do_not_quote"].startswith("Play it again")
    # the line text appears nowhere else in the whole prompt
    assert prompt.count("Play it again") == 1


def test_the_response_schema_and_the_bans_are_unchanged():
    """The context is INPUT-ONLY: the model still returns exactly
    {beat_id, expression, motion, camera} and every prohibition stands."""
    ledger = _ledger()
    beats = sl.extract_beats(ledger)
    prompt = sl._build_nonverbal_batch_prompt(
        beats, _meta(), ledger, "a station")
    assert '{"beat_id","expression","motion","camera"}' in prompt
    assert "Do NOT write a text_prompt field." in prompt
    assert "NEVER quote or paraphrase the line itself." in prompt


def test_a_directive_quoting_the_beat_intent_is_dropped():
    """An instruction is not an enforcement: the private context gets the same
    contiguous-run backstop as the line. A writer that echoes the beat intent
    verbatim into a directive loses that field to the deterministic fallback,
    loudly -- only a whole verbatim run fires it, never shared vocabulary."""
    ledger = _ledger()
    beats = sl.extract_beats(ledger)
    intent = ledger["lines"][0]["beat_intent"]

    def llm(_prompt):
        return json.dumps([{"beat_id": "b001",
                            "expression": "hard certainty",
                            "motion": intent,
                            "camera": "slow push-in"}])

    policy = {"policy_version": 2, "video_models": {
        "announcer_video_model": {"engine_id": "ltx25_video"},
        "music_video_model": {"engine_id": "ltx25_video"},
        "character_video_model": {"engine_id": "ltx25_foley_16gb"}}}
    creative, warnings = sl.derive_creative_directives(
        beats, _meta(), ledger, llm_fn=llm, video_policy=policy)
    row = creative["b001"]
    assert intent not in row["text_prompt"]
    assert any("private context" in w for w in warnings)
    # the honest fields survive untouched
    assert "hard certainty" in row["text_prompt"]


def test_a_sparse_beat_acquires_no_null_context_keys():
    """A synthetic music beat or an old ledger has no beat_intent/traits --
    present-key only, so the payload does not grow nulls."""
    ledger = _ledger()
    del ledger["lines"][0]["beat_intent"]
    del ledger["lines"][0]["traits"]
    beats = sl.extract_beats(ledger)
    prompt = sl._build_nonverbal_batch_prompt(
        beats, {}, ledger, "a station")
    row = [json.loads(ln) for ln in prompt.splitlines()
           if ln.startswith("{")][0]
    assert "beat_intent" not in row
    assert "traits" not in row
    assert "Story (private context)" not in prompt
    assert "Objects on hand" not in prompt


def test_the_director_no_longer_mandates_hitting_a_prop():
    """2026-10-22 (operator: "always banging something on a desk"). The foley
    lanes decode their sound from the director's words, and the old ask made
    every beat touch an object, worked example a chart slapped onto a desk."""
    ledger = _ledger()
    prompt = sl._build_nonverbal_batch_prompt(
        sl.extract_beats(ledger), _meta(), ledger, "a station")
    assert "REAL CONTACT" not in prompt
    assert "wooden desk" not in prompt
    assert "slams" not in prompt
    assert "a blow only when the line is a blow" in prompt
    assert "take the ground, the air and the objects from THIS" in prompt


def test_the_directors_sound_examples_are_words_the_lexicon_maps():
    """An example the lexicon cannot hear would teach the director to write
    sounds that never reach the prompt."""
    from nodes._otr_video_engines import eng_ltx25 as ltx25
    for example, sound in (
            ("boots crunching on gravel", "gravel crunching underfoot"),
            ("a coat snapping in the wind", "wind moaning"),
            ("feet splashing through a puddle", "feet splashing through water"),
            ("pushing through the undergrowth", "leaves and branches rustling"),
            ("a ledger's pages riffled", "pages riffling"),
            ("a brass dial turned", "close dial ticks and clicks"),
            ("a curtain pulled aside", "cloth shifting")):
        assert any(sound in s for s in ltx25.named_sounds_for(example)), example
