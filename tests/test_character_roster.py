"""Character roster parsing and gender, the input that stops casting rolling dice.

Every case here comes from a real Folger cast list. The defect this guards is
concrete: `gender_of_first_name` returns "unknown" for every Shakespeare name, so
without the roster a female lead draws a male voice on roughly half of seeds.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from nodes._otr_character_roster import (
    CharacterRecord,
    infer_gender,
    parse_character_roster,
)

SOURCES = (
    pathlib.Path(__file__).resolve().parent.parent
    / "config" / "source_banks" / "shakespeare" / "sources"
)

AS_YOU_LIKE_IT = """Characters in the Play
======================
ORLANDO, youngest son of Sir Rowland de Boys
OLIVER, his elder brother
ADAM, servant to Oliver and friend to Orlando
ROSALIND, daughter to Duke Senior
CELIA, Rosalind's cousin, daughter to Duke Frederick
DUKE FREDERICK, the usurping duke
Lords attending Duke Senior in exile:
  JAQUES
  AMIENS
PHOEBE, a disdainful shepherdess

ACT 1
"""

ROMEO_AND_JULIET = """Characters in the Play
======================
ROMEO
MONTAGUE, his father
LADY MONTAGUE, his mother
JULIET
CAPULET, her father
NURSE to Juliet

ACT 1
"""

MUCH_ADO = """Characters in the Play
======================
LEONATO, Governor of Messina
HERO, his daughter
BEATRICE, his niece
Waiting gentlewomen to Hero:
  MARGARET
  URSULA
COUNT CLAUDIO, a young lord from Florence
SIGNIOR BENEDICK, a gentleman from Padua

ACT 1
"""


def _one(text, name):
    return next(r for r in parse_character_roster(text) if r.matches(name))


class TestGenderFromDescription:
    def test_relation_words_gender_this_character(self):
        assert _one(AS_YOU_LIKE_IT, "ORLANDO").gender == "male"
        assert _one(AS_YOU_LIKE_IT, "ROSALIND").gender == "female"

    def test_a_relation_outranks_a_title_belonging_to_someone_else(self):
        # "daughter to Duke Senior" is a woman despite naming a Duke.
        rosalind = _one(AS_YOU_LIKE_IT, "ROSALIND")
        assert (rosalind.gender, rosalind.gender_source) == ("female", "relation")

    def test_title_in_the_name_genders_when_the_description_does_not(self):
        duke = _one(AS_YOU_LIKE_IT, "DUKE FREDERICK")
        assert (duke.gender, duke.gender_source) == ("male", "title")

    def test_an_ungendered_description_stays_unknown_rather_than_guessing(self):
        adam = _one(AS_YOU_LIKE_IT, "ADAM")
        assert adam.gender == "unknown"
        assert adam.is_known_gender is False


class TestGroupHeaders:
    def test_indented_members_inherit_their_group_description(self):
        jaques = _one(AS_YOU_LIKE_IT, "JAQUES")
        assert (jaques.gender, jaques.gender_source) == ("male", "group")

    def test_plural_group_words_still_gender(self):
        # "Waiting gentlewomen to Hero:" is plural; the lists are singular.
        for name in ("MARGARET", "URSULA"):
            record = _one(MUCH_ADO, name)
            assert (record.gender, record.gender_source) == ("female", "group"), name

    def test_a_group_header_is_not_itself_a_character(self):
        names = {r.name for r in parse_character_roster(AS_YOU_LIKE_IT)}
        assert not any("attending" in n.lower() for n in names)


class TestBackReference:
    def test_a_bare_name_is_gendered_by_the_next_entry_possessive(self):
        # ROMEO and JULIET have NO descriptions of their own. The following
        # entries -- "MONTAGUE, his father" and "CAPULET, her father" -- do.
        romeo = _one(ROMEO_AND_JULIET, "ROMEO")
        juliet = _one(ROMEO_AND_JULIET, "JULIET")
        assert (romeo.gender, romeo.gender_source) == ("male", "back_reference")
        assert (juliet.gender, juliet.gender_source) == ("female", "back_reference")

    def test_the_referring_entry_keeps_its_own_gender(self):
        assert _one(ROMEO_AND_JULIET, "MONTAGUE").gender == "male"
        assert _one(ROMEO_AND_JULIET, "LADY MONTAGUE").gender == "female"

    def test_an_entry_with_its_own_description_is_not_overwritten(self):
        hero = _one(MUCH_ADO, "HERO")
        assert hero.gender_source == "relation"


class TestEntryShapes:
    def test_an_entry_without_a_comma_is_still_parsed(self):
        # "NURSE to Juliet" carries no comma and was being dropped entirely.
        nurse = _one(ROMEO_AND_JULIET, "NURSE")
        assert nurse.gender == "female"

    def test_formal_roster_names_match_short_speech_prefixes(self):
        # The cast list says SIGNIOR BENEDICK / COUNT CLAUDIO; the play speaks
        # as BENEDICK / CLAUDIO.
        assert _one(MUCH_ADO, "BENEDICK").name == "SIGNIOR BENEDICK"
        assert _one(MUCH_ADO, "CLAUDIO").name == "COUNT CLAUDIO"

    def test_the_honorific_still_supplies_the_gender_it_was_stripped_for(self):
        assert _one(MUCH_ADO, "BENEDICK").gender == "male"

    def test_a_scene_without_a_cast_block_yields_nothing(self):
        assert parse_character_roster("ROMEO\nBut soft, what light\n") == ()

    def test_empty_input_is_safe(self):
        assert parse_character_roster("") == ()
        assert parse_character_roster(None) == ()


class TestInferGenderDirectly:
    @pytest.mark.parametrize("description,expected", [
        ("his elder brother", "male"),
        ("daughter to Duke Senior", "female"),
        ("a disdainful shepherdess", "female"),
        ("the usurping duke", "male"),
        ("servant to Oliver", "unknown"),
    ])
    def test_descriptions(self, description, expected):
        assert infer_gender("SOMEONE", description)[0] == expected


@pytest.mark.skipif(not SOURCES.exists(), reason="vendored corpus absent")
class TestVendoredSidecars:
    def test_every_sidecar_carries_scene_characters_with_a_gender_field(self):
        sidecars = sorted(SOURCES.glob("*.provenance.json"))
        assert sidecars, "no provenance sidecars vendored"
        for path in sidecars:
            data = json.loads(path.read_text(encoding="utf-8"))
            if "speakers" not in data:
                continue  # a prose unit, not a play scene
            assert "characters" in data, path.name
            assert len(data["characters"]) == len(data["speakers"]), path.name
            for entry in data["characters"]:
                assert entry["gender"] in ("male", "female", "unknown"), path.name
                assert entry["gender_source"], path.name

    def test_the_leads_of_the_balcony_scene_are_gendered_correctly(self):
        path = SOURCES / "romeo_juliet__act2_scene2.provenance.json"
        if not path.exists():
            pytest.skip("balcony scene not vendored")
        by_name = {
            c["name"]: c
            for c in json.loads(path.read_text(encoding="utf-8"))["characters"]
        }
        assert by_name["ROMEO"]["gender"] == "male"
        assert by_name["JULIET"]["gender"] == "female"
        assert by_name["NURSE"]["gender"] == "female"


def test_record_matching_is_case_insensitive_and_safe():
    record = CharacterRecord(
        name="SIGNIOR BENEDICK", description="a gentleman", gender="male",
        gender_source="title", aliases=("BENEDICK",),
    )
    assert record.matches("benedick")
    assert record.matches("SIGNIOR BENEDICK")
    assert not record.matches("")
    assert not record.matches("CLAUDIO")
