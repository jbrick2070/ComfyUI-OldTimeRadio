"""My Story: a role's gender is the listener's to state, and every role has one
before any voice is chosen (PBUG-20261017-03).

THE DEFECT (4060 leg r4, 2026-10-17). `story_characters` = "Ada, the keeper.
Tom." The treatment cast Ada with gender "" while its own description of her
read "the sound she hears ... Gender unstated", and CastLock's open-pool draw
gave her Kokoro am_liam, a man's voice.

THE RULING (operator): the listener defines gender per role, and any role the
story leaves open is given a gender and description BEFORE a voice is picked,
as the other lanes do. So: a "(she)" / "(he)" / "(they)" marker after a name is
read by Python and is binding; every other role is settled to male or female
before `_assign_voices`.

Headless: canned slot functions, no model, no GPU.
"""
from __future__ import annotations

import json

import pytest

from nodes import _otr_my_story as MS
from nodes import _otr_story_input as SI

from tests.test_my_story_runner import Slots, _run, _treatment  # noqa: F401


# --------------------------------------------------------------------------- #
# the marker grammar
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text,expected", [
    ("Ada (she), the keeper. Tom (he).", [("Ada", "female"), ("Tom", "male")]),
    ("Ada (She/Her)\nTom (m)", [("Ada", "female"), ("Tom", "male")]),
    ("Dr. Ada Vance (she) and her brother Tom (he)",
     [("Dr. Ada Vance", "female"), ("and her brother Tom", "male")]),
    ("Mabel (they); Ruth (woman)", [("Mabel", "other"), ("Ruth", "female")]),
])
def test_a_marker_after_a_name_states_that_role_s_gender(text, expected):
    assert [(g.name, g.gender) for g in SI.stated_genders(text)] == expected


@pytest.mark.parametrize("text", [
    "Ada, the keeper. Tom.",            # the live input: nothing stated
    "Ada (the keeper). Tom (her brother).",
    "(she) with no name before it",
    "",
])
def test_ordinary_notes_state_nothing(text):
    assert SI.stated_genders(text) == ()


# --------------------------------------------------------------------------- #
# matching a marker to the treatment's people
# --------------------------------------------------------------------------- #
def _marks(text):
    return SI.stated_genders(text)


def test_an_exact_name_wins_over_a_longer_one():
    bound, notes = MS._bind_stated_genders(_marks("Ada (she)"),
                                           ["Ada", "Ada Vance"])
    assert bound == {"Ada": "female"} and notes == []


def test_a_name_inside_a_phrase_and_a_first_name_both_match_on_word_boundaries():
    bound, _ = MS._bind_stated_genders(
        _marks("Her brother Tom (he). Ada (she)"), ["Tom Reyes", "Ada Vance"])
    assert bound == {"Tom Reyes": "male", "Ada Vance": "female"}


def test_a_prefix_that_is_not_a_whole_word_never_matches():
    bound, notes = MS._bind_stated_genders(_marks("Ann (she)"), ["Anna"])
    assert bound == {}
    assert notes[0]["kind"] == "stated_gender_unmatched"


def test_an_honorific_does_not_hide_the_name():
    bound, _ = MS._bind_stated_genders(_marks("Dr. Ada (she)"), ["Ada"])
    assert bound == {"Ada": "female"}


def test_a_name_marked_both_ways_pins_nothing_whatever_the_order():
    for text in ("Ada (she); Ada (he)", "Ada (he); Ada (she)"):
        bound, notes = MS._bind_stated_genders(_marks(text), ["Ada"])
        assert bound == {}
        assert notes[0]["kind"] == "stated_gender_conflict"


def test_a_marker_that_fits_two_people_pins_neither():
    bound, notes = MS._bind_stated_genders(_marks("Ada (she)"),
                                           ["Ada Vance", "Ada Pell"])
    assert bound == {}
    assert notes[0]["kind"] == "stated_gender_ambiguous"


# --------------------------------------------------------------------------- #
# settling every role before a voice is chosen
# --------------------------------------------------------------------------- #
def _member(name, gender="", role="", description=""):
    return MS.CastMember(name=name, gender=gender, role=role,
                         character_description=description)


# The exact rows the live leg's treatment returned.
_LIVE_ADA = dict(
    name="Ada", gender="",
    description=("Deeply isolated, haunted by solitude; initially skeptical, "
                 "then deeply unnerved by the sound she hears. Speaks with a "
                 "strained, weary tone. Heard the voice first. Gender unstated. "
                 "Age band 40s. Register anxious, then resigned. Timbre low, "
                 "strained."))
_LIVE_TOM = dict(
    name="Tom", gender="",
    description=("A mysterious, calm voice that seems to emanate from the storm "
                 "itself. Appears briefly, speaking cryptically. Gender "
                 "unstated. Age band 30s. Register smooth, unsettlingly "
                 "articulate. Timbre clear, resonant."))


def _live_treatment():
    t = MS.StoryTreatment.model_validate(_treatment())
    t.cast = [_member(**_LIVE_ADA), _member(**_LIVE_TOM)]
    return t


def test_the_live_ada_row_is_settled_female_from_her_own_description():
    t = _live_treatment()
    receipts = MS._settle_cast_genders(t, {})
    by = {m.name: m for m in t.cast}
    assert by["Ada"].gender == "female"
    # Tom: no pronoun and not in the curated pool, so the stable coin.
    assert by["Tom"].gender in ("male", "female")
    assert {r["name"]: r["source"] for r in receipts} == {
        "Ada": "description_pronouns", "Tom": "seeded_coin"}


def test_a_name_the_pool_knows_is_settled_from_the_pool():
    from config import cast_pools

    name = next(n for n in cast_pools.FIRST_NAMES_BY_GENDER["female"]
                if cast_pools.gender_of_first_name(n) == "female")
    assert MS._settle_gender(_member(name)) == ("female", "name_pool")


def test_a_listener_marker_beats_what_the_model_wrote():
    t = _live_treatment()
    t.cast[0].gender = "male"
    MS._settle_cast_genders(t, {"Ada": "female"})
    assert t.cast[0].gender == "female"


def test_a_listener_they_is_kept_as_written():
    t = _live_treatment()
    MS._settle_cast_genders(t, {"Tom": "other"})
    assert t.cast[1].gender == "other"


def test_a_model_other_without_a_marker_is_still_settled():
    t = _live_treatment()
    t.cast[1].gender = "other"
    MS._settle_cast_genders(t, {})
    assert t.cast[1].gender in ("male", "female")


def test_a_committed_gender_is_left_alone():
    t = _live_treatment()
    t.cast[0].gender = "female"
    t.cast[1].gender = "female"
    assert MS._settle_cast_genders(t, {}) == []
    assert [m.gender for m in t.cast] == ["female", "female"]


def test_the_role_outranks_pronouns_about_someone_else():
    """'her brother' is a man: the possessive belongs to his sister."""
    m = _member("Zed", role="her brother",
                description="Arrived last night with a map he will not show her.")
    assert MS._settle_gender(m) == ("male", "role_relation")


def test_mixed_pronouns_are_no_evidence():
    m = _member("Qx1", description="She waits by the lamp. He leaves at dawn.")
    gender, source = MS._settle_gender(m)
    assert source == "seeded_coin" and gender in ("male", "female")


def test_the_coin_is_stable_for_a_name():
    m = _member("Qx1")
    assert MS._settle_gender(m) == MS._settle_gender(_member("Qx1"))


# --------------------------------------------------------------------------- #
# end to end through the real runner, to the ledger cast rows
# --------------------------------------------------------------------------- #
class _WrongGenders(Slots):
    """The treatment model calls Ada male and leaves Tom open."""

    def _answer(self, messages):
        system = messages[0]["content"] if messages else ""
        if "radio dramatist" in system:
            t = _treatment(self._acts, self._cast)
            for c in t["cast"]:
                c["gender"] = "male" if c["name"] == "Ada" else ""
            return json.dumps(t)
        return super()._answer(messages)


def test_the_marked_ada_case_reaches_the_ledger_cast_female():
    slots = _WrongGenders()
    led, _ = _run(slots, characters="Ada (she), the keeper. Tom (he).")
    cast = {r["name"]: r["gender"] for r in led.data["cast"]}
    assert cast["Ada"] == "female"
    assert cast["Tom"] == "male"
    story = led.data["meta"]["my_story"]
    assert story["stated_genders"] == [
        {"name": "Ada", "gender": "female", "marker": "she"},
        {"name": "Tom", "gender": "male", "marker": "he"}]
    assert {r["name"]: r["source"] for r in story["gender_settlement"]} == {
        "Ada": "listener", "Tom": "listener"}
    # the model's own answer is kept as evidence
    assert story["treatment_proposal"]["cast"][0]["gender"] == "male"


def test_the_interpretation_receipt_carries_the_binding_marker():
    led, _ = _run(_WrongGenders(), characters="Ada, the keeper. Tom (he).")
    named = {p["name"]: p["stated_gender"]
             for p in led.data["meta"]["my_story"]["interpretation"]["named_cast"]}
    assert named["Tom"] == "male"


def test_an_unmarked_live_input_still_casts_every_role_with_a_gender():
    led, _ = _run(_WrongGenders(), characters="Ada, the keeper. Tom.")
    genders = [r["gender"] for r in led.data["cast"] if r["name"] != "ANNOUNCER"]
    assert all(g in ("male", "female") for g in genders)


def test_the_settle_runs_before_voices_are_assigned():
    import inspect

    src = inspect.getsource(MS.run_my_story_episode)
    assert src.index("_settle_cast_genders(") < src.index("_assign_voices(")


# --------------------------------------------------------------------------- #
# ONE SOURCE: the stamped cast row is what the voice, still and video read
# --------------------------------------------------------------------------- #
class _LiveAda(Slots):
    """The live treatment: Ada and Tom with no gender, Ada called "she"."""

    def _answer(self, messages):
        system = messages[0]["content"] if messages else ""
        if "radio dramatist" in system:
            t = _treatment(self._acts, self._cast)
            live = {"Ada": _LIVE_ADA, "Tom": _LIVE_TOM}
            for c in t["cast"]:
                c["gender"] = ""
                c["character_description"] = live[c["name"]]["description"]
            return json.dumps(t)
        return super()._answer(messages)


def test_the_live_ada_case_stamps_a_female_row_that_stills_and_video_read():
    from nodes import otr_meta_brief_image_prompt as IMG
    from nodes import otr_shot_lock as SL

    led, _ = _run(_LiveAda(), characters="Ada, the keeper. Tom.")
    ada = next(r for r in led.data["cast"] if r["name"] == "Ada")
    assert ada["gender"] == "female"
    assert "unstated" not in ada["character_description"].lower()
    assert "the sound she hears" in ada["character_description"]
    # the treatment receipt the still prompts fall back to agrees with the row
    planned = {c["name"]: c["gender"]
               for c in led.data["meta"]["my_story"]["treatment"]["cast"]}
    assert planned["Ada"] == "female"
    # video: shot lock's appearance lookup anchors on the row's gender
    assert SL._appearance_for_char(led.data, ada["char_id"]).startswith(
        "adult woman, ")
    # stills: the portrait prompt anchors on the same row
    assert IMG._ensure_gender_anchor(ada["character_description"], ada) \
        .startswith("adult woman, ")


# --------------------------------------------------------------------------- #
# Sonnet QA on 4ccbcf10: phrasings and wrong guesses it reproduced
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text,names,expected", [
    ("Ada, the keeper (she). Tom (he).", ["Ada", "Tom"],
     {"Ada": "female", "Tom": "male"}),
    ("Ada the keeper (she).", ["Ada"], {"Ada": "female"}),
    ("Ada, (she)", ["Ada"], {"Ada": "female"}),
    ("Don (he)", ["Don Quixote"], {"Don Quixote": "male"}),
    ("Major (he)", ["Major Tom"], {"Major Tom": "male"}),
    ("Ada (ella), Tom (\u00e9l)", ["Ada", "Tom"], {"Ada": "female", "Tom": "male"}),
])
def test_natural_phrasings_bind_the_person_they_follow(text, names, expected):
    bound, _ = MS._bind_stated_genders(SI.stated_genders(text), names)
    assert bound == expected


@pytest.mark.parametrize("role,description,others", [
    ("skeptic", "Doubts that she saw lights.", ()),
    ("", "Tells Ada she is wrong.", ("Ada",)),
    ("keeps count of ships", "", ()),
    ("a fool for love", "", ()),
    ("mother's helper", "", ()),
])
def test_evidence_about_someone_else_never_decides(role, description, others):
    gender, source = MS._settle_gender(
        _member("Qx1", role=role, description=description), others)
    assert source == "seeded_coin"


def test_a_description_without_the_stale_note_is_left_byte_identical():
    t = _live_treatment()
    t.cast[1].character_description = "A calm   voice."
    MS._settle_cast_genders(t, {})
    assert t.cast[1].character_description == "A calm   voice."


# --------------------------------------------------------------------------- #
# Sonnet QA on 7abcea1e: true self-references keep their evidence
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("description,others,expected", [
    ("Runs the diner and she never sleeps.", ("The Doctor", "The Old Man"), "female"),
    ("A doctor who loves the sea, and she shows it.", ("The Doctor",), "female"),
    ("Hears voices; she is afraid.", (), "female"),
    ("A scholar who says he has seen the lights.", (), "male"),
    ("Runs the diner. When she laughs the room goes quiet.", (), "female"),
    ("the sound she hears", ("Tom",), "female"),
    ("Doubts that she saw lights.", (), ""),
    ("Tells Ada she is wrong.", ("Ada",), ""),
])
def test_description_pronouns_read_the_character_not_others(description, others, expected):
    assert MS._description_gender(_member("Qx1", description=description),
                                  others) == expected


def test_a_relative_clause_marker_binds_the_name_its_record_leads_with():
    bound, _ = MS._bind_stated_genders(
        SI.stated_genders("Tom, who loves Ada (he). Ada (she)."), ["Ada", "Tom"])
    assert bound == {"Tom": "male", "Ada": "female"}
