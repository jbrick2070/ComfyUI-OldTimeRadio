"""Positive structural coverage for DramaticState."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from nodes._otr_dramatic_state import DramaticState


def _payload() -> dict:
    return {
        "dramatic_question": "Will Maeve sign the confession?",
        "character_a_wants": "hide the ledger",
        "character_b_wants": "publish the ledger",
        "costly_choice_beat": "d003",
        "ending_change": "the ledger becomes public",
    }


def test_dramatic_state_preserves_authored_wording_and_length():
    payload = _payload()
    payload["dramatic_question"] = "Q"
    payload["character_a_wants"] = "A"
    payload["character_b_wants"] = "A"
    payload["ending_change"] = "This ending is deliberately long. " * 100

    state = DramaticState.model_validate(payload)

    assert state.dramatic_question == "Q"
    assert state.character_a_wants == state.character_b_wants == "A"
    assert state.ending_change == payload["ending_change"]


@pytest.mark.parametrize(
    "field",
    [
        "dramatic_question",
        "character_a_wants",
        "character_b_wants",
        "ending_change",
    ],
)
def test_dramatic_state_requires_nonblank_structural_text(field):
    payload = _payload()
    payload[field] = "   "
    with pytest.raises(ValidationError):
        DramaticState.model_validate(payload)


def test_dramatic_state_rejects_invalid_slot_id():
    payload = _payload()
    payload["costly_choice_beat"] = "b003"
    with pytest.raises(ValidationError):
        DramaticState.model_validate(payload)
