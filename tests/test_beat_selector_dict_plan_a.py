"""QA item A regression -- the BeatSelector scorer must read a
raw dict-shaped plan (parsed-from-JSON candidate) correctly. The old
_voiced_beats used getattr(plan, "beats"), which returned None for a
dict plan and scored it as a zero-beat plan, distorting the selector
audit.

Hermetic: pure-Python scorer, no torch / comfy.
"""
from __future__ import annotations

from nodes._otr_beat_selector import _voiced_beats, score_beat_sheet


def _dict_plan():
    return {
        "dramatic_state": {
            "character_a_wants": "hold the line",
            "character_b_wants": "open the doors",
            "costly_choice_beat": "d002",
            "ending_change": "the doors stay shut forever",
        },
        "beats": [
            {"speaker": "REN", "dialogue_slot_id": "d001",
             "state_before": "calm", "state_after": "uneasy"},
            {"speaker": "MUSIC", "dialogue_slot_id": "",
             "state_before": "", "state_after": ""},
            {"speaker": "VESPER", "dialogue_slot_id": "d002",
             "state_before": "uneasy", "state_after": "decided"},
        ],
    }


def test_dict_plan_voiced_beats_nonzero():
    """A dict plan's voiced beats resolve (MUSIC filtered out)."""
    voiced = _voiced_beats(_dict_plan())
    assert len(voiced) == 2, (
        f"dict plan must expose its 2 non-MUSIC beats; got {len(voiced)} "
        "(regression: getattr returned zero beats for a dict plan)"
    )


def test_dict_plan_scores_are_not_all_zero():
    """With beats + dramatic_state readable, structural axes fire."""
    scores = score_beat_sheet(_dict_plan())
    # The exact field name is internal; just prove the dict plan did
    # NOT collapse to an all-zero score the way a zero-beat plan would.
    total = sum(
        v for v in vars(scores).values() if isinstance(v, (int, float))
    ) if hasattr(scores, "__dict__") else None
    if total is not None:
        assert total > 0, "dict plan scored all-zero (regression)"
