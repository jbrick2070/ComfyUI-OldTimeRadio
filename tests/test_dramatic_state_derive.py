"""Structural costly-choice slot selection coverage."""
from __future__ import annotations

from nodes._otr_dramatic_state import pick_costly_choice_slot


def test_long_list_picks_second_to_last_voiced_slot():
    assert pick_costly_choice_slot(
        ["d001", "d002", "d003", "d004", "d005"]
    ) == "d004"


def test_short_list_picks_last_voiced_slot():
    assert pick_costly_choice_slot(["d001", "d002"]) == "d002"


def test_empty_input_uses_structural_default():
    assert pick_costly_choice_slot([]) == "d001"
    assert pick_costly_choice_slot(None) == "d001"
