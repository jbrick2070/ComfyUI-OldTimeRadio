"""nodes/_otr_dramatic_state.py -- Sprint 2 (2026-05-28).

The DramaticState object replaces the 350-char `script_brief`
postage-stamp arc as the episode's reproducibility anchor. The
news_interpreter and Director write it; the Writer reads it; the
structural validators in `_otr_beat_validators` judge whether a beat
sheet honors it.

Why the postage-stamp arc had to go (round-robin consensus):
  - 350 chars is too small to specify the dramatic question, both
    characters' opposed desires, and the costly-choice beat the
    final third must turn on.
  - Without a typed costly_choice_beat reference, the structural
    validators have nothing to point at -- "make it better" / "more
    drama" feedback (Sprint 5's editor downgrade target) became the
    only available signal.

The state object is intentionally narrow. It does NOT carry tone,
mood, style, or genre -- those ride on the existing Style picker +
period flags. The state object's job is plot architecture only:
  * What is the dramatic question?
  * What do A and B each want, and why must those wants oppose?
  * Where (which dialogue_slot_id) is the costly choice that pivots
    the third act?
  * What is the ending change from the opening state?

Storage decision (plan open-question resolved): attached to
Stage1Plan as a top-level optional `dramatic_state` field. Optional
because legacy plans (and the early-stage tests) pre-date the
object; the structural validators skip cleanly when it is None.

PURE module: pydantic only, no LLM call, no I/O. UTF-8 no BOM.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, model_validator


__all__ = [
    "DramaticState",
]


class DramaticState(BaseModel):
    """Episode-level dramatic state. Sprint 2 keystone.

    Field-by-field rationale:
      * dramatic_question -- the one question the audience holds in
        their head from beat 1 to the ending. Long enough to be a
        full question (10..240 chars), short enough that the Writer
        and Editor can hold it in one breath.
      * character_a_wants / character_b_wants -- the two opposed
        desires that drive the conflict. The post-init validator
        asserts the strings are non-trivially distinct so the LLM
        cannot satisfy the schema by emitting the same desire twice.
      * costly_choice_beat -- a `dialogue_slot_id` (d\\d{3}) pointing
        at the beat where the irreversible choice lands. Sprint 4's
        best-of-N selector scores presence of this beat as one of its
        five structural axes; Sprint 5's editor constraints check
        that the beat's state_before != state_after.
      * ending_change -- one sentence naming what is different about
        the world (or the characters) at the close vs at the open.
        The structural validator asserts this is NOT equal to the
        opening state (`Stage1Plan.beats[0]` state_before).
    """

    dramatic_question: str = Field(
        ...,
        min_length=10,
        max_length=240,
        description=(
            "The single question the audience holds across the whole "
            "episode (e.g. 'Will Maeve sign the confession before the "
            "audit closes?'). Sprint 5 editor checks the close "
            "answers this question."
        ),
    )
    character_a_wants: str = Field(
        ...,
        min_length=4,
        max_length=120,
        description=(
            "Concrete want of the first principal. Verb phrase; "
            "specific stake (not 'happiness', not 'closure'). "
            "Sprint 4 selector scores opposition strength against "
            "character_b_wants."
        ),
    )
    character_b_wants: str = Field(
        ...,
        min_length=4,
        max_length=120,
        description=(
            "Concrete want of the second principal. MUST oppose "
            "character_a_wants -- post-init validator rejects strings "
            "that normalize-equal (case + whitespace), giving the "
            "operator a fast feedback signal when the LLM emits "
            "twin desires."
        ),
    )
    costly_choice_beat: str = Field(
        ...,
        pattern=r"^d\d{3}$",
        description=(
            "dialogue_slot_id of the beat where the irreversible "
            "choice lands. Sprint 4 selector scores presence; "
            "Sprint 5 editor confirms state_before != state_after "
            "on the named slot."
        ),
    )
    ending_change: str = Field(
        ...,
        min_length=4,
        max_length=200,
        description=(
            "One-sentence statement of what is different at the "
            "close vs the open. Structural validator asserts this "
            "is NOT equal to the opening state."
        ),
    )

    @model_validator(mode="after")
    def _wants_must_oppose(self) -> "DramaticState":
        """The two principals' wants must not be the same string.

        Cheap structural floor: the LLM occasionally satisfies the
        opposed-desires field by emitting the same string twice
        (Mistral-Nemo NF4 has a tendency to coast on already-rendered
        tokens under constrained decode). The expensive check is the
        Sprint 4 selector's `clear_opposed_desires` axis; this is
        the floor that catches the trivial case at parse time.
        """
        def _norm(s: str) -> str:
            return " ".join((s or "").lower().split())

        a = _norm(self.character_a_wants)
        b = _norm(self.character_b_wants)
        if a == b:
            raise ValueError(
                "character_a_wants and character_b_wants must "
                "differ; got identical strings after normalization. "
                "The conflict needs opposed desires, not echoed "
                f"ones: {self.character_a_wants!r}."
            )
        return self
