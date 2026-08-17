"""Chunk B step 3 -- the cameo on the surviving content-owned lane.

WHAT THIS LANE MAKES HARD. It derives its cast FROM the finished script and then
gates on it, so the cameo cannot be injected afterwards: it has to be decided
before the prompts, carried through the treatment, and reconciled with a casting
pass that was never told about him. Each test below pins one seam of that.
"""
import random

import pytest

from config import cast_pools as POOLS
from nodes import _otr_scifi_news_pro as PRO
from nodes._otr_casting import resolve_lemmy_cameo

_LEMMY_PRESET = POOLS.LEMMY_PROFILE["voice_preset"]
_SIGNATURE = POOLS.LEMMY_PROFILE["speech_signature"]


def _hit():
    return resolve_lemmy_cameo("scifi_news_pro", True)


def _miss():
    return resolve_lemmy_cameo("scifi_news_pro", False)


def _shape(name, register="clipped", want="w", pressure="p", role="r"):
    return PRO.CastShape(name=name, role=role, want=want, pressure=pressure,
                         register=register)


def _treatment(*shapes):
    return PRO.Treatment(
        title="T", dramatic_question="Q", setting="S",
        cast_shapes=list(shapes), turn="turn",
        priced_ending={"choice": "c", "cost_paid": "p"},
        news_thread="thread",
    )


# --- identity -------------------------------------------------------------


@pytest.mark.parametrize("spelling", ["LEMMY", "lemmy", "  Lemmy  ", "LeMmY"])
def test_the_cameo_is_recognized_however_the_model_spells_him(spelling):
    """Always the parser's identity rule, never == "LEMMY". The model writes
    names in whatever case it likes; a raw comparison would miss him and the
    lane would deal a second voice to a second Lemmy."""
    assert PRO._is_lemmy(spelling)


def test_somebody_else_is_not_the_cameo():
    assert not PRO._is_lemmy("LEMUEL")
    assert not PRO._is_lemmy("ANNOUNCER")


# --- the prompt contract --------------------------------------------------


def test_the_prompt_names_him_on_a_hit_and_says_nothing_on_a_miss():
    clause = PRO._cameo_prompt_clause(_hit())

    assert PRO.LEMMY_NAME in clause
    assert POOLS.LEMMY_PROFILE["character_description"] in clause
    assert PRO._cameo_prompt_clause(_miss()) == ""
    assert PRO._cameo_prompt_clause(None) == ""


def test_the_prompt_asks_for_his_stake_but_does_not_supply_one():
    """A Python-canned want is the stapled-on cameo the exemplar bar rejects."""
    clause = PRO._cameo_prompt_clause(_hit())

    assert "want" in clause.lower()
    for canned in POOLS.LEMMY_PROFILE.get("notes", "").split():
        assert canned not in clause or len(canned) < 5


# --- the treatment validator ----------------------------------------------


def test_on_a_hit_the_cast_must_name_him_exactly_once():
    check = PRO._make_treatment_validator(None, 3, {}, "", _hit())

    assert check(_treatment(_shape("LEMMY"), _shape("SARAH"))) is None
    assert check(_treatment(_shape("SARAH"), _shape("ELIAS"))) is not None


def test_on_a_hit_he_needs_somebody_to_talk_to():
    """A cast of Lemmy alone is a different show, not a cameo."""
    check = PRO._make_treatment_validator(None, 3, {}, "", _hit())

    assert check(_treatment(_shape("LEMMY"))) is not None


def test_on_a_miss_the_name_is_reserved_show_wide():
    """An accidental second Lemmy would be dealt a random voice from the open
    pool and arrive as a stranger wearing his name."""
    check = PRO._make_treatment_validator(None, 3, {}, "", _miss())

    assert check(_treatment(_shape("SARAH"), _shape("ELIAS"))) is None
    assert check(_treatment(_shape("LEMMY"), _shape("SARAH"))) is not None


def test_without_a_decision_the_validator_behaves_exactly_as_before():
    """The pre-chunk-B call shape keeps working, so the API lands green ahead
    of its callers."""
    check = PRO._make_treatment_validator(None, 3, {}, "")

    assert check(_treatment(_shape("SARAH"))) is None


# --- the deterministic normalization --------------------------------------


def test_normalization_pins_identity_and_leaves_the_stake_alone():
    treatment = _treatment(
        _shape("lemmy", register="whatever the model wrote",
               want="to fix the relay", pressure="the storm", role="comms"),
        _shape("SARAH"),
    )

    out = PRO._normalize_lemmy_shape(treatment, _hit())
    lemmy = next(s for s in out.cast_shapes if PRO._is_lemmy(s.name))

    assert lemmy.name == PRO.LEMMY_NAME          # the show's spelling
    assert lemmy.register == _SIGNATURE          # how the Cockney reaches dialogue
    assert lemmy.want == "to fix the relay"      # the model's, untouched
    assert lemmy.pressure == "the storm"
    assert lemmy.role == "comms"
    assert [s.name for s in out.cast_shapes][1] == "SARAH"


def test_normalization_is_a_no_op_on_a_miss():
    treatment = _treatment(_shape("SARAH"))

    assert PRO._normalize_lemmy_shape(treatment, _miss()) is treatment
    assert PRO._normalize_lemmy_shape(treatment, None) is treatment


# --- the voice deal -------------------------------------------------------


def test_the_menu_withholds_his_preset_so_nobody_else_draws_it():
    """His preset is IN the open pool -- its timbre row reads
    gravelly/engineer/mechanic, exactly what a casting LLM reaches for."""
    open_menu = PRO._deal_voice_menu(2)
    reserved = PRO._deal_voice_menu(2, taken={_LEMMY_PRESET})

    assert _LEMMY_PRESET in {e.preset for e in open_menu.entries}
    assert _LEMMY_PRESET not in {e.preset for e in reserved.entries}


# --- row assembly ---------------------------------------------------------


def _casting(*names):
    return PRO.CastingVoices(cast=[
        PRO.CastVoice(name=n, role="r", character_description="d",
                      gender="female", timbre=f"m{i + 1:02d}")
        for i, n in enumerate(names)
    ])


def test_the_cameo_row_is_synthesized_from_the_profile_not_the_casting_llm():
    """The casting artifact never covered him -- the menu he would have been
    cast from never offered his voice."""
    menu = PRO._deal_voice_menu(1, taken={_LEMMY_PRESET})
    speaker_order = ["SARAH", "LEMMY"]

    rows = PRO._assign_voices(_casting("SARAH"), menu, random.Random(7),
                              speaker_order, _hit())

    lemmy = next(r for r in rows if PRO._is_lemmy(r["name"]))
    assert lemmy["voice_preset"] == _LEMMY_PRESET
    assert lemmy["gender"] == POOLS.LEMMY_PROFILE["gender"]
    assert lemmy["character_description"] == \
        POOLS.LEMMY_PROFILE["character_description"]
    # He keeps his real position in the script, which is what makes the
    # assembly gate (speaker set == cast rows) hold by construction.
    assert [r["name"] for r in rows] == ["ANNOUNCER", "SARAH", PRO.LEMMY_NAME]


def test_no_two_characters_share_a_voice_on_a_cameo_episode():
    from nodes._otr_casting import _assert_unique_bark_voices

    menu = PRO._deal_voice_menu(2, taken={_LEMMY_PRESET})
    rows = PRO._assign_voices(_casting("SARAH", "ELIAS"), menu,
                              random.Random(7),
                              ["SARAH", "LEMMY", "ELIAS"], _hit())

    _assert_unique_bark_voices(rows)      # raises if it ever double-allocates
    presets = [r["voice_preset"] for r in rows if r["name"] != "ANNOUNCER"]
    assert len(presets) == len(set(presets))


def test_a_miss_assigns_voices_exactly_as_before():
    menu = PRO._deal_voice_menu(1)
    rows = PRO._assign_voices(_casting("SARAH"), menu, random.Random(7),
                              ["SARAH"], _miss())

    assert [r["name"] for r in rows] == ["ANNOUNCER", "SARAH"]
