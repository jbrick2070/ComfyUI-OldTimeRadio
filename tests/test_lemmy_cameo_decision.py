"""The cameo decision is made ONCE, and every consumer reads that one answer.

WHY A DEDICATED FILE. The content-owned lanes derive their cast from the
finished script and then gate on it, so the cameo has to be decided BEFORE any
authoring -- and once it is decided, the prompt contract, the cast contract, the
roll receipt and the voice deal must all agree. The failure this pins is a lane
stamping `lemmy_hit: False` beside a Lemmy who is standing in the script.
"""
import json

import pytest

from nodes import _otr_casting as CASTING
from nodes._otr_casting import (
    LEMMY_CAMEO_DECISION_SCHEMA,
    LEMMY_KNOB_FORCED_EXCLUDE,
    LEMMY_KNOB_FORCED_INCLUDE,
    LEMMY_KNOB_NATURAL_ROLL,
    LEMMY_POLICY_OPERATOR_CAMEO,
    LEMMY_POLICY_SOURCE_FIDELITY_EXCLUSION,
    LemmyCameoDecision,
    content_owned_cast_contract,
    resolve_lemmy_cameo,
)

_EXCLUDED_BANK = "public_domain"
_CAMEO_BANK = "scifi_news_pro"


@pytest.fixture
def counted_roll(monkeypatch):
    """Count roll_lemmy calls so 'exactly once, and only here' is provable."""
    calls = []

    def _fake_roll():
        calls.append(True)
        return True

    monkeypatch.setattr(CASTING._POOLS, "roll_lemmy", _fake_roll)
    return calls


# --- the normative truth table -------------------------------------------


def test_an_excluded_bank_refuses_the_cameo_and_never_rolls(counted_roll):
    """Fidelity outranks everything: the source's cast is the point of the lane."""
    for knob in (None, True, False):
        decision = resolve_lemmy_cameo(_EXCLUDED_BANK, knob)

        assert decision.lemmy_hit is False
        assert decision.lemmy_policy == LEMMY_POLICY_SOURCE_FIDELITY_EXCLUSION
        assert decision.roll_executed is False

    assert counted_roll == [], "an excluded bank must not spend the roll"


def test_exclusion_outranks_an_operator_who_forced_the_cameo(counted_roll):
    """The knob still records what was ASKED, even though the bank refused --
    that is how a reader tells a refusal from a decline."""
    decision = resolve_lemmy_cameo(_EXCLUDED_BANK, True)

    assert decision.knob_state == LEMMY_KNOB_FORCED_INCLUDE
    assert decision.lemmy_hit is False


def test_forced_include_does_not_roll(counted_roll):
    decision = resolve_lemmy_cameo(_CAMEO_BANK, True)

    assert decision.lemmy_hit is True
    assert decision.lemmy_policy == LEMMY_POLICY_OPERATOR_CAMEO
    assert decision.knob_state == LEMMY_KNOB_FORCED_INCLUDE
    assert decision.roll_executed is False
    assert counted_roll == [], "a forced decision must not consult chance"


def test_forced_exclude_does_not_roll(counted_roll):
    decision = resolve_lemmy_cameo(_CAMEO_BANK, False)

    assert decision.lemmy_hit is False
    assert decision.lemmy_policy == LEMMY_POLICY_OPERATOR_CAMEO
    assert decision.knob_state == LEMMY_KNOB_FORCED_EXCLUDE
    assert decision.roll_executed is False
    assert counted_roll == []


def test_the_natural_roll_spends_exactly_one_roll(counted_roll):
    decision = resolve_lemmy_cameo(_CAMEO_BANK, None)

    assert decision.lemmy_hit is True          # the fake roll returns True
    assert decision.knob_state == LEMMY_KNOB_NATURAL_ROLL
    assert decision.roll_executed is True
    assert counted_roll == [True], "the roll runs once, not zero or twice"


def test_a_bank_variant_inherits_the_family_exclusion(counted_roll):
    """Fidelity is a family behaviour, so a bake-off variant cannot opt in."""
    decision = resolve_lemmy_cameo("public_domain_v3", None)

    assert decision.lemmy_policy == LEMMY_POLICY_SOURCE_FIDELITY_EXCLUSION
    assert counted_roll == []


# --- the receipt ----------------------------------------------------------


def test_to_meta_is_json_serializable_with_a_pinned_key_set():
    """`Ledger.save()` NEVER RAISES -- a non-serializable object in meta logs a
    warning, returns None, and a dozen call sites never check. So the episode
    would lose its receipt in silence. Everything here must be a primitive."""
    meta = resolve_lemmy_cameo(_CAMEO_BANK, True).to_meta()

    assert set(meta) == {
        "schema_version", "lemmy_hit", "lemmy_policy",
        "knob_state", "source_bank_id", "roll_executed",
    }
    assert meta["schema_version"] == LEMMY_CAMEO_DECISION_SCHEMA
    assert json.loads(json.dumps(meta)) == meta
    for value in meta.values():
        assert isinstance(value, (bool, int, str)), value


def test_the_decision_is_immutable():
    """Every consumer reads the same answer, so nobody may edit it in flight."""
    decision = resolve_lemmy_cameo(_CAMEO_BANK, True)

    with pytest.raises(Exception):
        decision.lemmy_hit = False


def test_the_receipt_records_the_bank_it_was_handed():
    assert resolve_lemmy_cameo("  media_archive  ", False).source_bank_id == (
        "media_archive")


# --- the cast contract ----------------------------------------------------


def test_no_decision_preserves_the_pre_chunk_b_contract_exactly():
    """The API lands green BEFORE its callers migrate, so an unmigrated lane
    must stamp byte-for-byte what it stamped before."""
    contract = content_owned_cast_contract(
        source_bank_id=_CAMEO_BANK,
        num_characters_request=3,
        num_characters_locked=4,
    )

    assert contract == {
        "lemmy_hit": False,
        "lemmy_policy": CASTING.CONTENT_OWNED_NO_CAMEO_ROLL,
        "casting_attempts": [],
        "num_characters_request": 3,
        "num_characters_locked": 4,
    }


def test_no_decision_still_reports_the_fidelity_exclusion():
    contract = content_owned_cast_contract(
        source_bank_id=_EXCLUDED_BANK,
        num_characters_request=2,
        num_characters_locked=2,
    )

    assert contract["lemmy_policy"] == LEMMY_POLICY_SOURCE_FIDELITY_EXCLUSION


def test_a_decision_drives_the_contract_so_the_two_cannot_disagree():
    decision = resolve_lemmy_cameo(_CAMEO_BANK, True)
    contract = content_owned_cast_contract(
        source_bank_id=_CAMEO_BANK,
        num_characters_request=3,
        num_characters_locked=4,
        decision=decision,
    )

    assert contract["lemmy_hit"] is True
    assert contract["lemmy_policy"] == LEMMY_POLICY_OPERATOR_CAMEO
    assert contract["lemmy_hit"] == decision.to_meta()["lemmy_hit"]


def test_the_contract_never_grows_a_cast_seed():
    """`cast_seed` is a CLAIM that the writer's seeded picker produced this
    cast and can be replayed from it. A lane-owned cast has no such seed, and
    CastLock detonates when it tries to replay one."""
    contract = content_owned_cast_contract(
        source_bank_id=_CAMEO_BANK,
        num_characters_request=1,
        num_characters_locked=1,
        decision=resolve_lemmy_cameo(_CAMEO_BANK, False),
    )

    for forbidden in CASTING.CONTENT_OWNED_CONTRACT_FORBIDDEN_KEYS:
        assert forbidden not in contract


def test_the_decision_type_is_exported_for_the_runners():
    assert isinstance(resolve_lemmy_cameo(_CAMEO_BANK, False), LemmyCameoDecision)
