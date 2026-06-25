"""Conformance suite for the model-agnostic schema-adherence Lever 1.

Offline, no GPU / no network. Proves the converged nested-alias resolution
(docs/2026-06-25-schema-adherence/nested-fork/roundtable/pass01_plan.md):

  * the proven Opus failure (a NESTED BeatEdit whose action arrived under the
    key `lever`) now validates deterministically;
  * canonical input is byte-identical (the local happy path is untouched);
  * a genuinely missing required field still fails LOUD (no fabrication);
  * a mis-aliased load-bearing `action` value is fail-closed (not a real tier);
  * the shared `apply_field_aliases` helper is pure (no input mutation) and a
    no-op for non-dicts and for schemas with no alias map (the binary lane).

UTF-8 no BOM. 4-space indentation.
"""
from __future__ import annotations

from typing import Literal

import pytest
from pydantic import BaseModel, ValidationError

from nodes._otr_radio_editor import BeatEdit, RadioEditPlan
from nodes._otr_structured_call import (
    PostValidationError,
    apply_field_aliases,
    parse_validate_tolerant,
    validate_tolerant_data,
)


# --------------------------------------------------------------------------- #
# The proven nested failure: RadioEditPlan -> List[BeatEdit], action under `lever`
# --------------------------------------------------------------------------- #


def test_nested_proven_opus_failure_validates():
    """The exact 2026-06-25 Opus shape: a nested BeatEdit with `index` +
    `beat_index` + the action under `lever`. pydantic recursion runs BeatEdit's
    before-validator, so the nested edit normalizes without any core change."""
    plan = RadioEditPlan.model_validate(
        {
            "edits": [{"index": 14, "lever": "SHORTEN_LINE", "beat_index": 14}],
            "projected_word_total": 120,
        }
    )
    assert len(plan.edits) == 1
    assert plan.edits[0].beat_index == 14
    assert plan.edits[0].action == "SHORTEN_LINE"


def test_beat_edit_lever_alias_direct():
    """BeatEdit on its own: `lever` -> `action`, `index` -> `beat_index`."""
    edit = BeatEdit.model_validate({"index": 7, "lever": "SPLIT_LINE"})
    assert edit.beat_index == 7
    assert edit.action == "SPLIT_LINE"


def test_merge_with_alias_preserved():
    """The shipped BUG-LOCAL-303 `merge_with` -> `merge_with_index` remap still
    works through the shared helper."""
    edit = BeatEdit.model_validate(
        {"beat_index": 3, "action": "MERGE_SHORT_LINES", "merge_with": 4}
    )
    assert edit.merge_with_index == 4


# --------------------------------------------------------------------------- #
# Byte-identity on canonical input (the local happy path is untouched)
# --------------------------------------------------------------------------- #


def test_canonical_input_byte_identical():
    """Canonical keys -> the validated instance is identical to a directly
    constructed one; no alias path fires."""
    canonical = {
        "edits": [
            {"beat_index": 0, "action": "KEEP"},
            {"beat_index": 1, "action": "MERGE_SHORT_LINES", "merge_with_index": 2},
        ],
        "projected_word_total": 200,
    }
    from_dict = RadioEditPlan.model_validate(canonical)
    built = RadioEditPlan(
        edits=[
            BeatEdit(beat_index=0, action="KEEP"),
            BeatEdit(beat_index=1, action="MERGE_SHORT_LINES", merge_with_index=2),
        ],
        projected_word_total=200,
    )
    assert from_dict.model_dump() == built.model_dump()


def test_canonical_key_wins_over_synonym():
    """An explicit canonical key always wins, even if a disagreeing synonym is
    also present (matches the shipped `index` vs `beat_index` precedence)."""
    edit = BeatEdit.model_validate({"index": 1, "beat_index": 9, "action": "KEEP"})
    assert edit.beat_index == 9


# --------------------------------------------------------------------------- #
# Fail-loud: an alias renames a present value, it never fabricates a missing one
# --------------------------------------------------------------------------- #


def test_missing_action_and_lever_still_fails():
    with pytest.raises(ValidationError):
        BeatEdit.model_validate({"beat_index": 2})


def test_missing_beat_index_and_index_still_fails():
    with pytest.raises(ValidationError):
        BeatEdit.model_validate({"action": "KEEP"})


def test_mis_aliased_action_value_is_fail_closed():
    """`lever` -> `action` can make a pydantic-valid-but-wrong object (action is
    a free `str`), but a bogus value is NOT a real tier action -- Guard1
    (post_validate_plan) rejects it downstream; here we prove it never
    masquerades as a valid Tier-1/Tier-2 action."""
    edit = BeatEdit.model_validate({"beat_index": 0, "lever": "NOT_AN_ACTION"})
    assert edit.action == "NOT_AN_ACTION"
    assert not edit.is_tier1()
    assert not edit.is_tier2()


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
