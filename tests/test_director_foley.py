"""The director writes each beat's foley (2026-10-22).

Operator: "have them write foley prompts for a variety of non-verbal sound
effects that may co-occur." On a lane that decodes its audio from the prompt
(the local LTX 2.5 foley and mime lanes), the nonverbal director answers a
``sound`` key; ShotLock vets it and the joint-AV composer seats it where the
lexicon's sounds go. The lexicon stays the fallback for every beat the
director did not fill.

CPU-safe and pure: no model is loaded -- a stub writer returns the JSON.
"""

from __future__ import annotations

import json

import pytest

from nodes import otr_shot_lock as sl
from nodes._otr_video_engines import eng_ltx25 as ltx25

_LINE = "Play it again, and keep the gain low."


def _ledger():
    return {
        "meta": {"story_brief_terms": {"setting": ["a cliff-top station"]}},
        "cast": [{"char_id": "c01", "name": "Vera",
                  "appearance": "a wiry radio operator in a wool coat"}],
        "lines": [{
            "line_id": "b1", "char_id": "c01", "speaker": "VERA",
            "speaker_role": "character", "text": _LINE, "dur_s": 3.0,
            "shot_id": "shot_001",
        }],
    }


def _policy(engine):
    return {"policy_version": 2, "video_models": {
        "announcer_video_model": {"engine_id": "ltx25_video"},
        "music_video_model": {"engine_id": "ltx25_video"},
        "character_video_model": {"engine_id": engine}}}


def _writer(sink, sound):
    def _call(prompt):
        sink.append(prompt)
        return json.dumps([{
            "beat_id": "b1", "expression": "jaw set",
            "motion": "she strides across the wet deck and wrenches a hatch",
            "camera": "track with her toward the hatch", "sound": sound}])
    return _call


def _derive(engine, sound, sink=None):
    sink = [] if sink is None else sink
    led = _ledger()
    beats = sl.extract_beats(led)
    creative, warnings = sl.derive_creative_directives(
        beats, led["meta"], led, llm_fn=_writer(sink, sound),
        video_policy=_policy(engine))
    return creative.get("b1") or {}, warnings, sink


# --------------------------------------------------------------------------- #
# which lanes are asked
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("engine,expected", [
    ("ltx25_foley_16gb", True),
    ("ltx25_mime_16gb", True),
    ("ltx25_foley_8gb", True),
    ("cloud_ltx25_foley_plus", False),   # no composer seats it; lexicon tail
    ("ltx25_video", False),              # silent
    ("humo", False),                     # audio-IN
])
def test_only_lanes_that_build_their_sound_from_the_prompt_are_asked(
        engine, expected):
    assert sl._lane_decodes_audio(engine) is expected


def test_a_foley_lane_is_asked_for_sound_and_keeps_it():
    sound = "boots slapping hard on wet planks, a hatch wheel grinding"
    c, warnings, sink = _derive("ltx25_foley_16gb", sound)
    assert '{"beat_id","expression","motion","camera","sound"}' in sink[0]
    assert "a name or a weapon" in sink[0]
    assert c["sound"] == sound
    assert not [w for w in warnings if "director sound dropped" in w]


def test_a_silent_lane_is_never_asked_and_stores_no_sound():
    c, _w, sink = _derive("ltx25_video", "boots slapping on planks")
    assert '{"beat_id","expression","motion","camera"}' in sink[0]
    assert '"sound"' not in sink[0]
    assert c["sound"] == ""


# --------------------------------------------------------------------------- #
# the guard
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("sound", [
    "a crystal glass ringing, a single bell struck",
    "cockpit switches snapping, a human body thudding into a seat",
    "rain hammering the roof, gravel crunching underfoot",
])
def test_ordinary_sounds_are_kept(sound):
    kept, reason = sl._clean_director_sound(sound, ["Vera"])
    assert kept == sound and reason == ""


@pytest.mark.parametrize("sound,why", [
    ("a crowd shouting, boots on stone", "voice"),
    ("her breathing quickens, a door creaks", "voice"),
    ("soft music drifting, wind in the rigging", "voice"),
    ("someone laughing far off", "voice"),
    ("a revolver hammer cocking", "weapon"),
    ("a hammer clicking back, a chair scraping", "weapon"),
    ("gunshots echoing", "weapon"),
    ("Vera's coat snapping in the wind", "cast"),
    ("the gown of the Queen of the Fairies rustling", "proper noun"),
    ("one, two, three, four", "longer"),
])
def test_a_sound_the_lane_would_speak_or_mismatch_is_dropped(sound, why):
    kept, reason = sl._clean_director_sound(sound, ["Vera"])
    assert kept == ""
    assert reason


def test_a_dropped_sound_falls_back_to_the_lexicon_and_says_so():
    c, warnings, _ = _derive("ltx25_foley_16gb", "a crowd shouting her name")
    assert c["sound"] == ""
    assert [w for w in warnings if "director sound dropped" in w]


def test_a_sound_that_quotes_the_line_is_dropped():
    c, warnings, _ = _derive("ltx25_foley_16gb", _LINE.rstrip("."))
    assert c["sound"] == ""


# --------------------------------------------------------------------------- #
# the composer
# --------------------------------------------------------------------------- #

def _inputs(sound=""):
    return {"setting": "a cliff-top station", "expression": "jaw set",
            "motion": "she slams the hatch shut",
            "camera": "track with her", "sound": sound}


def test_the_composer_seats_the_directors_sound_in_the_lexicons_slot():
    sound = "a hatch clanging shut, spray hissing across the deck"
    core = ltx25._ltx25_joint_av_core(_inputs(sound))
    assert core.index(sound) < core.index(ltx25._SOUND_FRAME) \
        < core.index("track with her")
    assert "a loud heavy wooden thud echoing" not in core


def test_no_director_sound_keeps_the_lexicon():
    core = ltx25._ltx25_joint_av_core(_inputs(""))
    assert "a loud heavy wooden thud echoing" in core


def test_the_finished_prompt_still_ends_on_the_no_voice_clause():
    core = ltx25._ltx25_joint_av_core(_inputs("spray hissing across the deck"))
    out = ltx25.finish_joint_av_positive("ltx25_foley_16gb", core)
    assert ltx25.joint_av_prompt_is_finished(out)


def test_the_render_driver_hands_the_sound_to_the_composer_and_receipts_it():
    """A helper nothing calls is this repo's most repeated defect: assert the
    wiring at its real site."""
    import inspect

    from nodes._otr_video_engines import render_driver
    src = inspect.getsource(render_driver.run_episode)
    assert '"sound": str(creative.get("sound") or "")' in inspect.getsource(
        render_driver)
    assert '"joint_av_sound_source"' in src
