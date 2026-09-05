"""Unit tests for _otr_dialogue_policy."""

import pytest
from nodes._otr_dialogue_policy import (
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


_BASE_PROMPT = "You are an AI radio script writer."


def test_a_non_lemmy_active_speaker_leaves_the_prompt_byte_identical():
    result = append_dialogue_policy(_BASE_PROMPT, active_speakers=("MARLOW",))
    assert result == _BASE_PROMPT
    assert _COCKNEY_ORTHOGRAPHY_RULE not in result


def test_lemmy_is_matched_past_case_and_surrounding_whitespace():
    result = append_dialogue_policy(_BASE_PROMPT, active_speakers=("  lemmy  ",))
    assert result.startswith(_BASE_PROMPT)
    assert _COCKNEY_ORTHOGRAPHY_RULE in result


def test_a_mixed_exchange_names_lemmy_and_protects_everyone_else():
    """One rule, subject LEMMY, with the other characters explicitly fenced."""
    result = append_dialogue_policy(
        _BASE_PROMPT, active_speakers=("MARLOW", "LEMMY", "REESE"),
    )
    assert result.count(_COCKNEY_ORTHOGRAPHY_RULE) == 1
    assert "For LEMMY's spoken lines only" in result
    assert "Every other character must retain" in result


def test_no_active_speakers_leaves_the_prompt_byte_identical():
    result = append_dialogue_policy(_BASE_PROMPT, active_speakers=())
    assert result == _BASE_PROMPT


@pytest.mark.parametrize("scalar", ["LEMMY", b"LEMMY"])
def test_a_scalar_speaker_name_raises_rather_than_iterating_characters(scalar):
    """"LEMMY" iterates to 'L','E',... -- a silent no-match, not a speaker."""
    with pytest.raises(TypeError):
        append_dialogue_policy(_BASE_PROMPT, active_speakers=scalar)


class _CastShimLike:
    """Shaped like the production `_CastShim` the exchange path normalizes to."""

    def __init__(self, name):
        self.name = name
        self.persona = "Cockney fixer."


def _generator_of_speakers():
    yield "LEMMY"


@pytest.mark.parametrize(
    "roster_shaped",
    [
        {"LEMMY": {"persona": "Cockney fixer."}},
        {"LEMMY", "MARLOW"},
        _generator_of_speakers(),
        _CastShimLike("LEMMY"),
        [{"name": "LEMMY"}],
        [{"char_id": "LEMMY"}],
        [_CastShimLike("LEMMY")],
        ("LEMMY", _CastShimLike("MARLOW")),
    ],
)
def test_roster_shaped_values_are_rejected_as_active_speakers(roster_shaped):
    """The whole defect was a roster arriving where speakers were expected.

    Every element is validated before any name is tested, so a good first
    entry cannot smuggle a wrong category in behind it.
    """
    with pytest.raises(TypeError):
        append_dialogue_policy(_BASE_PROMPT, active_speakers=roster_shaped)


def test_the_speaker_category_is_keyword_only():
    with pytest.raises(TypeError):
        append_dialogue_policy(_BASE_PROMPT, ("LEMMY",))


# ---------------------------------------------------------------------------
# Qualification receipts -- a route is approved only if it can PROVE it.
#
# `approved_native_routes` used to list bark with
# `qualification_receipt: "canonical_bark_preset_v1"` -- a bare string asserting
# an audition that never happened, inside the very policy meant to keep Lemmy
# consistent. That is BUG-12.86: a field that reads as evidence and is not.
# These pin the honest shape so it cannot quietly come back.
# ---------------------------------------------------------------------------
from config.cast_pools import QUALIFICATION_RECEIPT_REQUIRED_FIELDS


def _full_receipt(**over):
    r = {f: "x" for f in QUALIFICATION_RECEIPT_REQUIRED_FIELDS}
    r.update(over)
    return {"engine": "bark", "identity_kind": "preset",
            "identity_id": "v2/en_speaker_8", "qualification_receipt": r}


