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
from config.cast_pools import (  # noqa: E402
    QUALIFICATION_RECEIPT_REQUIRED_FIELDS,
    is_qualified_route,
)


def _full_receipt(**over):
    r = {f: "x" for f in QUALIFICATION_RECEIPT_REQUIRED_FIELDS}
    r.update(over)
    return {"engine": "bark", "identity_kind": "preset",
            "identity_id": "v2/en_speaker_8", "qualification_receipt": r}


def test_the_indextts2_route_is_approved_and_carries_its_receipt():
    """A REAL AUDITION HAPPENED, which is the branch the previous version of this
    test named in its own docstring: *"If this fails, either a real audition
    happened (update the test with its receipt) or a label crept back in."*

    G1 Test A ran blinded on 2026-08-10. The operator ranked the candidate first
    on Cockney, and -- without seeing any label -- heard the historic incumbent
    as Indian rather than Cockney, which is the floor-evidence failure evidenced
    properly instead of inferred from its filename.

    RE-AUDITIONED 2026-08-18 (`prod-audition-2026-08-18`). The voice-identity
    fix and then the emotion-ceiling change both moved the code that renders
    him, so the August record was withdrawn on its runtime fingerprint and a
    new blinded audition was run through the PRODUCTION dispatch. The old
    record is preserved verbatim under `superseded_native_routes`; this test
    reads the CURRENT one.

    So the assertion flips, but the thing being protected does not: a route may
    only be here WITH the receipt that earned it. `is_qualified_route` is the
    cheap legacy helper and may never authorize a selected route on its own --
    but it must AGREE with the real validator rather than contradict it, so the
    shipped route is checked against it here.
    """
    routes = LEMMY_VOICE_POLICY["approved_native_routes"]
    assert set(routes) == {"indextts2"}
    route = routes["indextts2"]
    assert is_qualified_route(route), (
        "the shipped route does not satisfy the legacy receipt helper -- a "
        "compatibility check that says 'no' where the authority says 'yes' is "
        "its own support ticket")
    receipt = route["qualification_receipt"]
    assert receipt["operator_verdict"].startswith("PASS")
    assert receipt["audited_on"] == "2026-08-18"
    assert receipt["identity_id"] == "idx_lemmy_algenib_cockney_v1"


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
