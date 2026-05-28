"""nodes/_otr_dramatic_state.py -- Sprint 2 (2026-05-28).

The DramaticState object replaces the 350-char `script_brief`
postage-stamp arc as the episode's reproducibility anchor. The
news_interpreter and the brief-builder write it; the Writer reads it;
the structural validators in `_otr_beat_validators` judge whether a
beat sheet honors it.

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
    "derive_dramatic_state_from_meta",
    "pick_costly_choice_slot",
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


# ---------------------------------------------------------------------------
# Sprint 2.1 wire-up helpers (2026-05-28)
# ---------------------------------------------------------------------------


_DEFAULT_QUESTION = (
    "Will the principals make the costly choice before the episode ends?"
)
_DEFAULT_A_WANTS = "honor the established commitment, whatever the cost"
_DEFAULT_B_WANTS = "force a compromise that protects the status quo"
_DEFAULT_ENDING = (
    "the situation closes meaningfully different from where it opened"
)


def _first_two_cast_names(cast_rows) -> tuple[str, str]:
    """Return (A, B) names from the cast roster. Falls back to
    PROTAGONIST / ANTAGONIST when fewer than two cast rows exist."""
    names: list[str] = []
    for row in (cast_rows or []):
        nm = ""
        if isinstance(row, dict):
            nm = str(row.get("name", "") or "").strip()
        else:
            nm = str(getattr(row, "name", "") or "").strip()
        if nm:
            names.append(nm)
        if len(names) >= 2:
            break
    if not names:
        return ("PROTAGONIST", "ANTAGONIST")
    if len(names) == 1:
        return (names[0], "OPPOSITION")
    return (names[0], names[1])


def pick_costly_choice_slot(voice_slot_ids) -> str:
    """Pick a costly_choice_beat slot id from a list of voiced slot
    ids. Heuristic: the LAST voiced beat before the close (i.e.
    second-to-last if there are >= 4 slots, otherwise the last). The
    Sprint 4 selector / Sprint 5 constraint checker can override.

    Returns "d001" as a defensive last resort when the list is empty
    (the DramaticState pattern validator demands a non-empty d-id).
    """
    ids = [str(s).strip() for s in (voice_slot_ids or []) if str(s).strip()]
    if not ids:
        return "d001"
    if len(ids) >= 4:
        return ids[-2]
    return ids[-1]


def _truncate(s: str, n: int) -> str:
    s = " ".join((s or "").split())
    if len(s) <= n:
        return s
    # Cut at a word boundary near n if possible so the pydantic
    # constraint passes cleanly without mid-word truncation.
    cut = s[: n - 1]
    sp = cut.rfind(" ")
    if sp > n // 2:
        return cut[:sp].rstrip()
    return cut.rstrip()


def derive_dramatic_state_from_meta(
    *,
    script_brief: str,
    cast_rows,
    voice_slot_ids,
    ending_change: str = "",
) -> "DramaticState":
    """Build a best-effort DramaticState from existing producer
    outputs (news_interpreter brief + locked cast + Stage 1 voice
    slot ids).

    The Sprint 2.1 wire-up calls this AFTER:
      * news_interpreter has run (or graceful-degraded -- the helper
        defaults when script_brief is empty),
      * the cast has been locked (cast_rows = ledger.data["cast"]),
      * Stage 1 has stamped slot ids on ledger lines
        (voice_slot_ids derived from ledger.data["lines"]).

    All fields satisfy the DramaticState schema (the pydantic
    constructor runs at the end; helper guarantees field lengths +
    the wants-must-oppose floor). On any pydantic failure the
    caller should log + skip stamping rather than crashing the
    writer.

    Tone: this is a STARTING POINT for the structural validators
    to score against. Sprint 2.1's job is to make the surface
    populated; later sprints (or operator edits) replace the
    defaults with LLM-derived strings.
    """
    a_name, b_name = _first_two_cast_names(cast_rows)
    # Question: derive from script_brief when it carries a question
    # mark; else default.
    sb = " ".join((script_brief or "").split())
    if "?" in sb:
        # Take the first sentence with a question mark.
        for chunk in sb.split("?"):
            candidate = (chunk.strip() + "?").strip()
            if 10 <= len(candidate) <= 240:
                question = candidate
                break
        else:
            question = _truncate(_DEFAULT_QUESTION, 240)
    elif sb:
        # No question mark; convert the brief into a question template.
        snip = _truncate(sb, 200)
        question = f"How does this resolve: {snip}?"
        question = _truncate(question, 240)
    else:
        question = _DEFAULT_QUESTION

    # Wants: defaults that name the cast so they are non-trivially
    # distinct (satisfies the wants-must-oppose validator).
    a_wants = _truncate(f"{a_name}: {_DEFAULT_A_WANTS}", 120)
    b_wants = _truncate(f"{b_name}: {_DEFAULT_B_WANTS}", 120)

    # Costly choice slot: heuristic on voice slot ids.
    cc_slot = pick_costly_choice_slot(voice_slot_ids)

    # Ending change: use caller-supplied when present; else default.
    ec = _truncate(ending_change.strip(), 200) if ending_change else ""
    if not ec:
        ec = _truncate(_DEFAULT_ENDING, 200)

    return DramaticState(
        dramatic_question=question,
        character_a_wants=a_wants,
        character_b_wants=b_wants,
        costly_choice_beat=cc_slot,
        ending_change=ec,
    )
