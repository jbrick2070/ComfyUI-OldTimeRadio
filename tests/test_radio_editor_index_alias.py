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
