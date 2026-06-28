"""Story-quality 3.1 -- dignity / safety guard for the DramaticState fallback.

The deterministic fallback templates must NEVER put PEOPLE or a protected-identity
/ harm-population group up for ownership / credit / control (frostbite_facility
shipped "take sole credit for transgender people" / "Control of transgender people
passes to whoever is willing to pay the higher price."). Ownership over an OBJECT
("the findings", "the decryption machine") stays fine. CPU; deterministic.
UTF-8, no BOM, ASCII-only source.
"""
from __future__ import annotations

from nodes._otr_dramatic_state_llm import (
    is_nonownable_story_object,
    derive_safe_fallback_term,
    _fallback_state,
    _pick_templates,
)
from nodes._otr_specificity import derive_central_object


def _meta(key_terms):
    return {"news": {"key_terms": list(key_terms), "script_brief": ""}}


# --- is_nonownable_story_object ---------------------------------------------
def test_people_class_terms_are_nonownable():
    for t in ("transgender people", "the residents", "violence victims",
              "suicide thoughts", "children", "the population", "the workers"):
        assert is_nonownable_story_object(t) is True, t


def test_object_heads_with_people_modifier_are_ownable():
    for t in ("patient records", "children's health study", "survey data",
              "the decryption machine", "the findings", "rail death record"):
        assert is_nonownable_story_object(t) is False, t


def test_nonownable_empty_or_none_is_false():
    assert is_nonownable_story_object("") is False
    assert is_nonownable_story_object(None) is False


# --- derive_safe_fallback_term ----------------------------------------------
def test_derive_skips_people_picks_first_ownable():
    assert derive_safe_fallback_term(
        ["transgender people", "patient records"]) == "patient records"


def test_derive_all_nonownable_returns_default():
    assert derive_safe_fallback_term(
        ["transgender people", "the residents"]) == "the findings"
    assert derive_safe_fallback_term(
        ["the victims"], default="the event") == "the event"


def test_derive_first_ownable_kept_byte_for_byte():
    assert derive_safe_fallback_term(
        ["the ledger", "patient records"]) == "the ledger"


def test_derive_skips_cast_names():
    assert derive_safe_fallback_term(
        ["Watson", "the ledger"], cast=["Watson"]) == "the ledger"


# --- _pick_templates ownership belt -----------------------------------------
def test_pick_templates_allow_ownership_false_drops_ownership_tuple():
    a, b, q, e = _pick_templates("the residents", "", allow_ownership=False)
    blob = " ".join([a, b, q, e]).lower()
    assert "take sole credit" not in blob
    assert "passes to whoever is willing to pay" not in blob


# --- _fallback_state dignity (the frostbite defect) -------------------------
def test_fallback_state_no_people_in_ownership_template():
    meta = _meta(["transgender people"])
    ds = _fallback_state(meta, ("d001", "d002"), "d001",
                         arc_shape="setup_complication_resolution")
    blob = " ".join([ds.character_a_wants, ds.character_b_wants,
                     ds.dramatic_question, ds.ending_change]).lower()
    # the people term never appears -- the safe default object stands in
    assert "transgender people" not in blob
    assert meta["dramatic_state_fallback_term"] == "the findings"
    assert meta["dramatic_state_fallback_term_replaced"] is True


def test_fallback_state_ownable_term_unchanged():
    meta = _meta(["the decryption machine"])
    ds = _fallback_state(meta, ("d001", "d002"), "d001",
                         arc_shape="setup_complication_resolution")
    assert meta["dramatic_state_fallback_term"] == "the decryption machine"
    assert meta["dramatic_state_fallback_term_replaced"] is False
    blob = " ".join([ds.character_a_wants, ds.character_b_wants]).lower()
    assert "the decryption machine" in blob


# --- derive_central_object secondary skip -----------------------------------
def test_central_object_skips_people_class():
    assert derive_central_object(["transgender people"], cast=[]) == ""


def test_central_object_keeps_real_object():
    assert derive_central_object(
        ["the decryption machine"], cast=[]) == "the decryption machine"
