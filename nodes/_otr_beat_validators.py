"""nodes/_otr_beat_validators.py -- Sprint 2 structural validators.

Per the build plan, validators judge VISIBLE STRUCTURE, never taste.
Small local models like Mistral-Nemo-Instruct-2407 NF4 cannot score
taste reliably; they can however reliably check for:
  * dead beats (state_before == state_after after normalization),
  * costly choice resolution (the named slot's state changes), and
  * ending change (the close differs from the opening state).

The output of `validate_beat_sheet(plan)` is a list of human-readable
defect strings. An empty list means "structurally fine -- escalate
to taste judgement via the human A/B listen test." A non-empty list
is the signal to trigger a structural re-roll of the BEAT SHEET (not
a line re-roll) per the round-robin's §5b principle: take more shots
at a good story, do not repair a bad one.

The validators are pure functions; the caller (the Stage 1 re-roll
loop in OTR_LedgerScriptWriter, and Sprint 4's best-of-N selector)
decides what to do with the defect list.

UTF-8 no BOM. Lazy import of Stage1Plan-shape kept duck-typed so this
module's imports stay stdlib + pydantic-free for cheap loading from
the structural test suite.
"""
from __future__ import annotations

from typing import Any, List, Optional


__all__ = [
    "validate_beat_sheet",
    "is_dead_beat",
    "DefectKind",
]


# Defect kind enum -- string-keyed so the structural-retry logger can
# bucket what kind of structural failure triggered the re-roll.
class DefectKind:
    DEAD_BEAT = "dead_beat"
    NO_COSTLY_CHOICE = "no_costly_choice"
    UNRESOLVED_COSTLY_CHOICE = "unresolved_costly_choice"
    UNCHANGED_ENDING = "unchanged_ending"
    NO_DRAMATIC_STATE = "no_dramatic_state"


def _norm(s: Any) -> str:
    """Normalize a state string for equality comparison.

    Strips outer whitespace, lowercases, collapses internal whitespace.
    Deliberately NOT semantic-similar (no embeddings, no edit
    distance) -- the validator is structural; taste judgements live
    in the Sprint 5 editor + the human listen test.
    """
    return " ".join((str(s or "").strip()).lower().split())


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    """Read `name` off `obj` whether it's a pydantic model, a
    dataclass, or a plain dict (Sprint 4.1: Stage1Plan.dramatic_state
    is Optional[Any] so the round-trip through JSON yields a dict,
    not a DramaticState. Both shapes need to work in the validator
    + scorer)."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def is_dead_beat(beat: Any) -> bool:
    """A beat is dead if it carries typed state and state_before
    normalizes-equal to state_after. Untyped beats (legacy Sprint 1
    or earlier) are NOT dead by definition -- they have no claim to
    test."""
    sb = getattr(beat, "state_before", None)
    sa = getattr(beat, "state_after", None)
    if not sb or not sa:
        return False
    return _norm(sb) == _norm(sa)


def _voiced_beats(plan: Any) -> List[Any]:
    """Return voiced beats (speaker != 'MUSIC')."""
    return [
        b for b in (getattr(plan, "beats", None) or [])
        if str(getattr(b, "speaker", "") or "").strip() != "MUSIC"
    ]


def validate_beat_sheet(plan: Any) -> List[str]:
    """Return a list of structural defect strings for the plan.

    Empty list means structurally clean. Each defect string starts
    with one of the DefectKind constants so the caller can bucket.

    Validators (Sprint 2 set; Sprint 4's selector adds more):
      1. dead-beat: any voiced beat where state_before ==
         state_after.
      2. costly-choice presence: dramatic_state.costly_choice_beat
         resolves to an existing dialogue_slot_id on the plan.
      3. costly-choice resolution: that beat's state_after differs
         from state_before (it actually pivots).
      4. ending-change: dramatic_state.ending_change does not
         normalize-equal the opening voiced beat's state_before.

    Sprint 2 design: when plan.dramatic_state is None, validators
    2-4 short-circuit silently (legacy plan; nothing to claim
    against). Validator 1 still runs on whatever beats are typed.
    """
    defects: List[str] = []
    voiced = _voiced_beats(plan)

    # ---- Validator 1: dead beat -----------------------------------
    for beat in voiced:
        if is_dead_beat(beat):
            beat_id = getattr(beat, "beat_id", "?")
            sb = getattr(beat, "state_before", "")
            defects.append(
                f"{DefectKind.DEAD_BEAT}: beat {beat_id} has "
                f"state_before == state_after ({sb!r}). Every voiced "
                "beat must change the situation."
            )

    ds = getattr(plan, "dramatic_state", None)
    if ds is None:
        return defects  # legacy plan; rest of validators have nothing to claim

    # ---- Validator 2: costly-choice beat exists ------------------
    target_slot = str(_attr(ds,"costly_choice_beat", "") or "").strip()
    if not target_slot:
        defects.append(
            f"{DefectKind.NO_COSTLY_CHOICE}: dramatic_state.costly_"
            "choice_beat is empty. The third act needs a typed pivot."
        )
        return defects

    by_slot = {
        str(getattr(b, "dialogue_slot_id", "") or "").strip(): b
        for b in voiced
    }
    by_slot.pop("", None)
    if target_slot not in by_slot:
        defects.append(
            f"{DefectKind.NO_COSTLY_CHOICE}: dramatic_state.costly_"
            f"choice_beat={target_slot!r} does not name any voiced "
            f"beat's dialogue_slot_id (known: {sorted(by_slot)[:8]}...)."
        )
        return defects

    # ---- Validator 3: costly-choice beat actually pivots ---------
    pivot = by_slot[target_slot]
    sb = getattr(pivot, "state_before", None)
    sa = getattr(pivot, "state_after", None)
    if sb is None or sa is None:
        # Typed only when both sides are populated; an untyped pivot
        # beat is a separate (sprint 2.1) gripe.
        pass
    elif _norm(sb) == _norm(sa):
        defects.append(
            f"{DefectKind.UNRESOLVED_COSTLY_CHOICE}: beat "
            f"{getattr(pivot, 'beat_id', '?')} (slot "
            f"{target_slot}) is the costly_choice_beat but "
            "state_before == state_after. The choice must change "
            "the situation."
        )

    # ---- Validator 4: ending change --------------------------------
    ending_change = _norm(_attr(ds,"ending_change", ""))
    opening_state = ""
    if voiced:
        opening_state = _norm(
            getattr(voiced[0], "state_before", "") or ""
        )
    if (
        ending_change
        and opening_state
        and ending_change == opening_state
    ):
        defects.append(
            f"{DefectKind.UNCHANGED_ENDING}: ending_change "
            "normalize-equals the opening state. The close must "
            "differ from the open."
        )

    return defects
