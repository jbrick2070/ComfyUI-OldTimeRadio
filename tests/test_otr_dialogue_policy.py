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


# ---------------------------------------------------------------------------
# Qualification receipts -- a route is approved only if it can PROVE it.
#
# `approved_native_routes` used to list bark with
# `qualification_receipt: "canonical_bark_preset_v1"` -- a bare string asserting
# an audition that never happened, inside the very policy meant to keep Lemmy
# consistent. That is BUG-12.86: a field that reads as evidence and is not.
# These pin the honest shape so it cannot quietly come back.
# ---------------------------------------------------------------------------
from config.cast_pools import (  # noqa: E402
    QUALIFICATION_RECEIPT_REQUIRED_FIELDS,
    is_qualified_route,
)


def _full_receipt(**over):
    r = {f: "x" for f in QUALIFICATION_RECEIPT_REQUIRED_FIELDS}
    r.update(over)
    return {"engine": "bark", "identity_kind": "preset",
            "identity_id": "v2/en_speaker_8", "qualification_receipt": r}


def test_no_route_is_currently_approved():
    """The honest state as of 2026-08-10. If this fails, either a real audition
    happened (update the test with its receipt) or a label crept back in."""
    assert LEMMY_VOICE_POLICY["approved_native_routes"] == {}


def test_the_canonical_route_is_routing_not_a_qualification_claim():
    """bark stays the DEFAULT destination -- that is where lemmy_row() pins him
    today -- but the policy must not imply anyone auditioned it."""
    route = LEMMY_VOICE_POLICY["canonical_route"]
    assert route["engine"] == "bark"
    assert route["qualification_receipt"] is None
    assert not is_qualified_route(route)


def test_a_bare_string_receipt_is_not_a_qualification():
    """The exact shape that shipped. A label is not evidence."""
    assert not is_qualified_route(
        {"engine": "bark", "qualification_receipt": "canonical_bark_preset_v1"})


@pytest.mark.parametrize("missing", sorted(QUALIFICATION_RECEIPT_REQUIRED_FIELDS))
def test_a_receipt_missing_any_required_field_is_not_qualified(missing):
    """Fail-closed per field, so no single omission can be waved through."""
    route = _full_receipt()
    del route["qualification_receipt"][missing]
    assert not is_qualified_route(route), f"omitting {missing!r} still passed"


@pytest.mark.parametrize("blanked", sorted(QUALIFICATION_RECEIPT_REQUIRED_FIELDS))
def test_a_present_but_empty_field_is_not_qualified(blanked):
    """Present-but-empty is the same lie in a longer form."""
    assert not is_qualified_route(_full_receipt(**{blanked: "   "}))


def test_a_complete_receipt_IS_qualified():
    """The control -- without this the guard could be fail-always, which proves
    nothing and would block a real audition from ever registering."""
    assert is_qualified_route(_full_receipt())


@pytest.mark.parametrize("bad", [None, "string", 123, [], {"no": "receipt"}])
def test_non_route_shapes_are_rejected_rather_than_crashing(bad):
    assert not is_qualified_route(bad)


def test_operator_verdict_is_required_because_only_a_human_can_supply_it():
    """Every other field could in principle be machine-filled. This one cannot,
    and it is the field that makes the receipt mean 'someone listened'."""
    assert "operator_verdict" in QUALIFICATION_RECEIPT_REQUIRED_FIELDS
    assert not is_qualified_route(_full_receipt(operator_verdict=""))
