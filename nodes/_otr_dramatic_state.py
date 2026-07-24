"""Typed episode-level dramatic state.

The model preserves authored prose exactly. Python enforces only non-empty
fields and the structural dialogue-slot identifier used by the ledger graph;
wording, length, style, and dramatic quality are prompt-owned.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


__all__ = ["DramaticState", "pick_costly_choice_slot"]


class DramaticState(BaseModel):
    """Prompt context for the episode's conflict and ending."""

    dramatic_question: str
    character_a_wants: str
    character_b_wants: str
    costly_choice_beat: str = Field(pattern=r"^d\d{3}$")
    ending_change: str

    @field_validator(
        "dramatic_question",
        "character_a_wants",
        "character_b_wants",
        "ending_change",
    )
    @classmethod
    def _authored_text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("dramatic-state text must not be blank")
        return value


def pick_costly_choice_slot(voice_slot_ids) -> str:
    """Choose a structurally valid voiced slot for the costly choice."""
    ids = [
        str(slot).strip()
        for slot in (voice_slot_ids or [])
        if str(slot).strip()
    ]
    if not ids:
        return "d001"
    if len(ids) >= 4:
        return ids[-2]
    return ids[-1]
