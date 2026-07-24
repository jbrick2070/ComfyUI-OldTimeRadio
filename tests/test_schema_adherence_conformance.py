"""Conformance suite for the shared model-agnostic schema-adherence core.

The retired radio-editor models are intentionally absent. These tests cover
the reusable alias resolver, tolerant parsing, and structural retry behavior.
"""
from __future__ import annotations

from typing import Literal

import pytest
from pydantic import BaseModel, ValidationError

from nodes._otr_structured_call import (
    PostValidationError,
    StructuredCallFailedError,
    apply_field_aliases,
    parse_validate_tolerant,
    structured_call,
    validate_tolerant_data,
)


# --------------------------------------------------------------------------- #
# The shared helper: pure, deterministic, collision-safe
# --------------------------------------------------------------------------- #


def test_apply_field_aliases_does_not_mutate_input():
    aliases = {"action": ("lever",)}
    data = {"beat_index": 0, "lever": "KEEP"}
    out = apply_field_aliases(aliases, data)
    assert data == {"beat_index": 0, "lever": "KEEP"}  # input untouched
    assert out == {"beat_index": 0, "action": "KEEP"}  # synonym moved + dropped


def test_apply_field_aliases_non_dict_noop():
    sentinel = ["not", "a", "dict"]
    assert apply_field_aliases({"a": ("b",)}, sentinel) is sentinel
    assert apply_field_aliases({"a": ("b",)}, None) is None


def test_apply_field_aliases_two_synonyms_is_ambiguous():
    """>= 2 declared synonyms present + canonical absent -> leave the field
    failing (no silent pick)."""
    aliases = {"action": ("lever", "act")}
    data = {"beat_index": 0, "lever": "KEEP", "act": "SHORTEN_LINE"}
    out = apply_field_aliases(aliases, data)
    assert "action" not in out  # ambiguous -> not mapped


def test_apply_field_aliases_empty_map_noop():
    data = {"x": 1}
    assert apply_field_aliases({}, data) is data


# --------------------------------------------------------------------------- #
# The shared tolerant core is reusable + a no-op for un-annotated schemas
# (the binary decision lane: a 1-field Literal["A","B"] schema)
# --------------------------------------------------------------------------- #


class _BinaryDecision(BaseModel):
    choice: Literal["A", "B"]


def test_validate_tolerant_data_binary_lane_noop():
    assert validate_tolerant_data({"choice": "A"}, _BinaryDecision).choice == "A"


def test_parse_validate_tolerant_binary_lane():
    assert parse_validate_tolerant('{"choice": "B"}', _BinaryDecision).choice == "B"


def test_validate_tolerant_data_still_fails_loud_on_bad_value():
    with pytest.raises(ValidationError):
        validate_tolerant_data({"choice": "C"}, _BinaryDecision)


def test_validate_tolerant_data_post_validator_rejection():
    def reject(_inst):
        return "too_long: rejected by content check"

    with pytest.raises(PostValidationError):
        validate_tolerant_data({"choice": "A"}, _BinaryDecision, post_validator=reject)


# --------------------------------------------------------------------------- #
# C3 ladder: the structural retry is spent ONLY on a JSON SYNTAX failure; a
# ValidationError skips it (the Opus token-burn fix); a plain ValueError
# propagates (only the three recoverable types are caught).
# --------------------------------------------------------------------------- #


class _Req(BaseModel):
    x: int


def _counting_slot(raw_out):
    """A slot fn that records its call count and always returns ``raw_out``."""
    state = {"n": 0}

    def slot(_messages, *, temperature, max_new_tokens, **_kw):
        state["n"] += 1
        return raw_out

    slot.calls = state
    return slot


def test_ladder_spends_structural_retry_on_json_syntax_error():
    """Unparseable output -> json.JSONDecodeError -> the structural retry RUNS.
    base + structural + typed repair = 3 slot calls (default max_attempts=3)."""
    slot = _counting_slot("this is not json at all")
    with pytest.raises(StructuredCallFailedError):
        structured_call(
            prompt="p", schema=_Req, slot_fn=slot,
            base_temperature=0.6, structural_retry_temperature=0.3,
            helper_name="ladder_json",
        )
    assert slot.calls["n"] == 3


def test_ladder_skips_structural_retry_on_validation_error():
    """Parseable JSON but the wrong shape -> ValidationError -> the structural
    retry is SKIPPED; only base + typed repair run = 2 slot calls. This is the
    token-burn fix: a re-prompt would just re-emit the same bad shape."""
    slot = _counting_slot('{"y": 1}')  # parses, but `x` is missing
    with pytest.raises(StructuredCallFailedError):
        structured_call(
            prompt="p", schema=_Req, slot_fn=slot,
            base_temperature=0.6, structural_retry_temperature=0.3,
            helper_name="ladder_validation",
        )
    assert slot.calls["n"] == 2


def test_ladder_plain_valueerror_propagates():
    """A plain ValueError from the slot fn is NOT recoverable -> it propagates
    unwrapped (not swallowed into StructuredCallFailedError)."""
    def slot(_messages, *, temperature, max_new_tokens, **_kw):
        raise ValueError("transport boom")

    with pytest.raises(ValueError) as ei:
        structured_call(
            prompt="p", schema=_Req, slot_fn=slot,
            base_temperature=0.6, structural_retry_temperature=0.3,
            helper_name="ladder_valueerror",
        )
    assert "transport boom" in str(ei.value)
    assert not isinstance(ei.value, StructuredCallFailedError)
