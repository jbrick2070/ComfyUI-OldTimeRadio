"""BUG-LOCAL-303 regression: BeatEdit accepts `index` as an alias for
`beat_index` (and `merge_with` for `merge_with_index`).

On the live all-Comfy run, claude-opus returned its RadioEditPlan edit items
with `index` instead of the schema's `beat_index`, so the length pass failed
pydantic validation and burned 2-3 credit-billed structured-call retries before
giving up. The alias lets the plan validate on attempt 1 -- no wasted tokens.
Pure-Python: pydantic only, no LLM / GPU / network.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes._otr_radio_editor import BeatEdit, RadioEditPlan  # noqa: E402


def test_index_alias_maps_to_beat_index():
    e = BeatEdit(**{"index": 2, "action": "KEEP"})
    assert e.beat_index == 2


def test_explicit_beat_index_still_works_and_wins():
    e = BeatEdit(**{"beat_index": 3, "action": "KEEP"})
    assert e.beat_index == 3
    # An explicit beat_index always wins over a stray index.
    e2 = BeatEdit(**{"beat_index": 5, "index": 9, "action": "KEEP"})
    assert e2.beat_index == 5


def test_merge_with_alias_maps_to_merge_with_index():
    e = BeatEdit(**{"index": 1, "action": "MERGE_SHORT_LINES", "merge_with": 2})
    assert e.beat_index == 1
    assert e.merge_with_index == 2


def test_radioeditplan_parses_edits_with_index_alias():
    plan = RadioEditPlan(**{
        "edits": [{"index": 0, "action": "KEEP"},
                  {"index": 1, "action": "KEEP"}],
        "projected_word_total": 120,
    })
    assert [x.beat_index for x in plan.edits] == [0, 1]


# ---------------------------------------------------------------------------
# S-D: gemma wrapper-key drift -- {"RadioEditPlan": {...}} unwrap
# ---------------------------------------------------------------------------
import pytest  # noqa: E402
from pydantic import ValidationError  # noqa: E402


def test_unwrap_schema_name_wrapper():
    # gemma nests the plan one level deep under its schema name -> made
    # projected_word_total read as 'missing' -> retries exhausted -> length
    # normalization silently skipped. The unwrap peels EXACTLY this wrapper.
    plan = RadioEditPlan.model_validate({
        "RadioEditPlan": {"edits": [{"index": 0, "action": "KEEP"}],
                          "projected_word_total": 90}})
    assert plan.projected_word_total == 90
    assert [x.beat_index for x in plan.edits] == [0]


def test_unwrapped_plan_unchanged():
    plan = RadioEditPlan.model_validate(
        {"edits": [{"index": 1, "action": "KEEP"}], "projected_word_total": 50})
    assert plan.projected_word_total == 50
    assert [x.beat_index for x in plan.edits] == [1]


def test_single_key_non_wrapper_plan_ok():
    # a single-key plan whose key is NOT the schema name is left alone.
    plan = RadioEditPlan.model_validate({"projected_word_total": 30})
    assert plan.projected_word_total == 30 and plan.edits == []


def test_multi_key_wrapper_rejected():
    # an AMBIGUOUS multi-key wrapper is NOT unwrapped -> fails LOUD.
    with pytest.raises(ValidationError):
        RadioEditPlan.model_validate({
            "RadioEditPlan": {"edits": [], "projected_word_total": 10},
            "stray": 1})


def test_non_dict_inner_wrapper_not_unwrapped():
    # a wrapper whose inner is not a dict is left alone -> fails LOUD.
    with pytest.raises(ValidationError):
        RadioEditPlan.model_validate({"RadioEditPlan": [1, 2, 3]})
