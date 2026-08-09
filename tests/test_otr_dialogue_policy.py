"""Unit tests for _otr_dialogue_policy."""

import pytest
from nodes._otr_dialogue_policy import (
    roster_has_lemmy,
    append_dialogue_policy,
    _COCKNEY_ORTHOGRAPHY_RULE,
)
from config.cast_pools import LEMMY_PROFILE, lemmy_row, LEMMY_VOICE_POLICY


def test_lemmy_profile_and_row_schema():
    assert LEMMY_PROFILE["accent"] == "cockney"
    assert LEMMY_PROFILE["dialogue_orthography"] == "standard_english"
    assert "Cockney" in LEMMY_PROFILE["speech_signature"]

    row = lemmy_row()
    assert row["name"] == "LEMMY"
    assert row["accent"] == "cockney"
    assert row["dialogue_orthography"] == "standard_english"
    assert row["tts_model"] == "bark"
    assert row["voice_preset"] == "v2/en_speaker_8"


def test_lemmy_voice_policy_structure():
    assert LEMMY_VOICE_POLICY["policy_version"] == "lemmy-cockney-v1"
    assert LEMMY_VOICE_POLICY["required_accent"] == "cockney"
    assert LEMMY_VOICE_POLICY["canonical_route"]["engine"] == "bark"


def test_roster_has_lemmy():
    assert roster_has_lemmy(["MARLOW", "LEMMY"]) is True
    assert roster_has_lemmy([{"name": "LEMMY"}]) is True
    assert roster_has_lemmy([{"char_id": "LEMMY"}]) is True
    assert roster_has_lemmy(["MARLOW", "HAYES"]) is False
    assert roster_has_lemmy([]) is False


def test_append_dialogue_policy():
    base_prompt = "You are an AI radio script writer."
    
    # Non-LEMMY roster -> prompt unchanged
    no_lemmy_result = append_dialogue_policy(base_prompt, ["MARLOW", "HAYES"])
    assert no_lemmy_result == base_prompt
    assert _COCKNEY_ORTHOGRAPHY_RULE not in no_lemmy_result

    # LEMMY roster -> orthography rule appended
    lemmy_result = append_dialogue_policy(base_prompt, ["MARLOW", "LEMMY"])
    assert _COCKNEY_ORTHOGRAPHY_RULE in lemmy_result
    assert lemmy_result.startswith(base_prompt)
