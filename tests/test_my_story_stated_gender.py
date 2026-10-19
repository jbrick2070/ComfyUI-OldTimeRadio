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
    # "(they)" is ordinary notes: every character ends male or female
    ("Mabel (they); Ruth (woman)", [("Ruth", "female")]),
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
                         appearance=description)


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


def test_a_non_binary_stated_value_is_never_kept():
    """Operator 2026-10-17: every character ends male or female."""
    t = _live_treatment()
    MS._settle_cast_genders(t, {"Tom": "other"})
    assert t.cast[1].gender in ("male", "female")


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
                c["appearance"] = live[c["name"]]["description"]
            return json.dumps(t)
        return super()._answer(messages)


def test_the_live_ada_case_stamps_a_female_row_that_stills_and_video_read():
    from nodes import otr_meta_brief_image_prompt as IMG
    from nodes import otr_shot_lock as SL

    led, _ = _run(_LiveAda(), characters="Ada, the keeper. Tom.")
    ada = next(r for r in led.data["cast"] if r["name"] == "Ada")
    assert ada["gender"] == "female"
    assert "unstated" not in ada["appearance"].lower()
    assert "the sound she hears" in ada["appearance"]
    # the treatment receipt the still prompts fall back to agrees with the row
    planned = {c["name"]: c["gender"]
               for c in led.data["meta"]["my_story"]["treatment"]["cast"]}
    assert planned["Ada"] == "female"
    # video: shot lock's appearance lookup anchors on the row's gender
    assert SL._appearance_for_char(led.data, ada["char_id"]).startswith(
        "adult woman, ")
    # stills: the portrait prompt anchors on the same row
    assert IMG._ensure_gender_anchor(ada["appearance"], ada) \
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
    t.cast[1].appearance = "A calm   voice."
    MS._settle_cast_genders(t, {})
    assert t.cast[1].appearance == "A calm   voice."


# --------------------------------------------------------------------------- #
# Sonnet QA on 7abcea1e: true self-references keep their evidence
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("description,others,expected", [
    ("Runs the diner and she never sleeps.", ("The Doctor", "The Old Man"), "female"),
    # names the other cast member, so the fallback abstains (the pass decides)
    ("A doctor who loves the sea, and she shows it.", ("The Doctor",), ""),
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


# --------------------------------------------------------------------------- #
# Two-strikes panel (Codex): one typed model pass reads the open roles; the
# deterministic ladder is only its fallback
# --------------------------------------------------------------------------- #
class _CastGenderAnswers(_LiveAda):
    """The live treatment, plus a cast-gender pass that answers ``reply``."""

    def __init__(self, reply, **kw):
        if "interpretation" not in kw:
            # nothing stated anywhere, so the open roles reach the pass
            from tests.test_my_story_runner import _interpretation
            kw["interpretation"] = _interpretation()
            for p in kw["interpretation"]["named_cast"]:
                p["stated_gender"] = ""
        super().__init__(**kw)
        self.reply = reply
        self.gender_prompts = []

    def _answer(self, messages):
        system = messages[0]["content"] if messages else ""
        if "You cast voices for a radio drama" in system:
            self.gender_prompts.append(messages)
            return json.dumps(self.reply)
        return super()._answer(messages)


def test_the_cast_gender_pass_decides_the_open_roles():
    slots = _CastGenderAnswers({"cast": [{"name": "Ada", "gender": "female"},
                                         {"name": "Tom", "gender": "male"}]})
    led, _ = _run(slots, characters="Ada, the keeper. Tom.")
    cast = {r["name"]: r["gender"] for r in led.data["cast"]}
    assert cast["Ada"] == "female" and cast["Tom"] == "male"
    sources = {r["name"]: r["source"]
               for r in led.data["meta"]["my_story"]["gender_settlement"]}
    assert sources == {"Ada": "cast_gender_pass", "Tom": "cast_gender_pass"}
    user = slots.gender_prompts[0][-1]["content"]
    assert "DECIDE: Ada, Tom" in user and "the sound she hears" in user


def test_the_pass_is_skipped_when_every_role_is_already_settled():
    slots = _CastGenderAnswers({"cast": []})
    _run(slots, characters="Ada (she), the keeper. Tom (he).")
    assert slots.gender_prompts == []


def test_a_pass_answer_naming_someone_else_falls_back_to_the_ladder():
    slots = _CastGenderAnswers({"cast": [{"name": "Mabel", "gender": "male"}]})
    led, _ = _run(slots, characters="Ada, the keeper. Tom.")
    story = led.data["meta"]["my_story"]
    sources = {r["name"]: r["source"] for r in story["gender_settlement"]}
    assert sources["Ada"] == "description_pronouns"
    assert next(r for r in led.data["cast"] if r["name"] == "Ada")["gender"] == "female"


def test_a_pass_that_raises_never_fails_the_episode():
    # The plain _LiveAda slots raise on the unknown cast-gender prompt.
    led, _ = _run(_LiveAda(), characters="Ada, the keeper. Tom.")
    genders = [r["gender"] for r in led.data["cast"] if r["name"] != "ANNOUNCER"]
    assert all(g in ("male", "female") for g in genders)


@pytest.mark.parametrize("description,others", [
    ("Tom loves her more than he loves life.", ("Tom",)),
    ("Tom hires a pilot because he needs one.", ("Tom",)),
    ("The Doctor thinks he is mad.", ("The Doctor",)),
    ("Tom, the grizzled old sheriff, hopes he wins.", ("Tom",)),
])
def test_the_fallback_abstains_when_another_cast_member_is_named(description, others):
    assert MS._description_gender(_member("Qx1", description=description),
                                  others) == ""


@pytest.mark.parametrize("text,names,expected", [
    ("My boss, who is Tom (he).", ["Ada", "Tom"], {"Tom": "male"}),
    ("That Tom (he).", ["Tom"], {"Tom": "male"}),
    ("Ada, who runs it, and Tom, who loves Ada (he).", ["Ada", "Tom"], {"Tom": "male"}),
    ("Ada, that old keeper (she).", ["Ada"], {"Ada": "female"}),
])
def test_relative_clause_markers_bind_the_person_before_the_clause(text, names, expected):
    bound, _ = MS._bind_stated_genders(SI.stated_genders(text), names)
    assert bound == expected


def test_the_cast_gender_seam_is_declared_and_shipped():
    import json as _json
    from pathlib import Path

    root = Path(MS.__file__).resolve().parent / "story_packs"
    pack = _json.loads((root / "my_story" / "my_story.json").read_text(encoding="utf-8"))
    assert pack["prompt_stages"]["my_story_cast_gender_system"].startswith(
        "You cast voices for a radio drama")
    pipes = _json.loads((root / "pipelines.json").read_text(encoding="utf-8"))
    row = next(p for p in pipes["pipelines"] if p["story_pipeline_id"] == "my_story_multipass")
    assert "my_story_cast_gender_system" in row["declared_seams"]


def test_the_interpretation_s_reading_is_context_for_the_pass_not_a_bypass():
    """QA on 660a46c5: the interpretation's prompt does not resolve pronoun
    referents, so its reading goes TO the pass, and the pass decides."""
    from tests.test_my_story_runner import _interpretation

    interp = _interpretation()
    for p in interp["named_cast"]:
        p["stated_gender"] = "female" if p["name"] == "Ada" else "male"
    slots = _CastGenderAnswers({"cast": [{"name": "Ada", "gender": "female"},
                                         {"name": "Tom", "gender": "male"}]},
                               interpretation=interp)
    led, _ = _run(slots, characters="Ada, the keeper. Tom.")
    sources = {r["name"]: r["source"]
               for r in led.data["meta"]["my_story"]["gender_settlement"]}
    assert sources == {"Ada": "cast_gender_pass", "Tom": "cast_gender_pass"}
    user = slots.gender_prompts[0][-1]["content"]
    assert '"listener_text_suggests": "female"' in user


def test_the_interpretation_s_reading_answers_when_the_pass_fails():
    from tests.test_my_story_runner import _interpretation

    interp = _interpretation()
    for p in interp["named_cast"]:
        p["stated_gender"] = "female" if p["name"] == "Ada" else "male"
    slots = _CastGenderAnswers({"cast": [{"name": "Nobody", "gender": "male"}]},
                               interpretation=interp)
    led, _ = _run(slots, characters="Ada, the keeper. Tom.")
    sources = {r["name"]: r["source"]
               for r in led.data["meta"]["my_story"]["gender_settlement"]}
    assert sources == {"Ada": "interpretation", "Tom": "interpretation"}


def test_two_interpretation_entries_that_disagree_say_nothing():
    from tests.test_my_story_runner import _interpretation

    interp = _interpretation(named=("Ada", "Ada", "Tom"))
    interp["named_cast"][0]["stated_gender"] = "female"
    interp["named_cast"][1]["stated_gender"] = "male"
    interp["named_cast"][2]["stated_gender"] = ""
    slots = _CastGenderAnswers({"cast": [{"name": "Nobody", "gender": "male"}]},
                               interpretation=interp)
    led, _ = _run(slots, characters="Ada, the keeper. Tom.")
    sources = {r["name"]: r["source"]
               for r in led.data["meta"]["my_story"]["gender_settlement"]}
    assert sources["Ada"] != "interpretation"
