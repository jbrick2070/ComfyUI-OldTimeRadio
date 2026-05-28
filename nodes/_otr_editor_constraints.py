"""nodes/_otr_editor_constraints.py -- Sprint 5 (2026-05-28).

Editor downgrade to constraint checker. Strips taste verbs from the
editor prompt, narrows the editor's job to five concrete constraints
that small local models can check reliably, and provides a pure-
Python rule-based check_constraints() that the wire-up sprint plugs
into the Story Room loop in place of the taste-based EditorVerdict.

Why downgrade: the round-robin agreed that revision cannot
ORIGINATE quality. The pre-Sprint-5 editor ran up to three cycles
of free-form "make it better / more drama / improve pacing"
revision; under live observation those cycles either rubber-stamped
or chased their tails. Sprint 4's best-of-N selection produces the
ceiling; Sprint 5's editor only enforces the floor.

Five constraints (all checkable by a small model OR by Python):
  WRONG_SPEAKER         -- a line is attributed to a name not in cast.
  PHANTOM_CHARACTER     -- a non-cast proper noun speaks or is named.
  MISSING_COSTLY_CHOICE -- dramatic_state.costly_choice_beat does
                           not resolve (the named slot is missing,
                           or its state does not change).
  NO_FINAL_THIRD_TURN   -- the final third of the voiced beats has
                           no state change (state_before ==
                           state_after across the run).
  FORMAT_FAILURE        -- the draft is malformed / off-schema.

This module ships PURE -- no LLM call, no I/O. The wire-up sprint
threads it into `run_story_room` as the single repair loop after
the Sprint 4 selector picks a winner; revision cap drops from 3 to
1 (one targeted repair, then ship-or-fail).

UTF-8 no BOM.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Tuple

from ._otr_beat_validators import is_dead_beat


__all__ = [
    "EditorConstraint",
    "EditorConstraintVerdict",
    "EDITOR_CONSTRAINTS_SYSTEM_PROMPT",
    "DEFAULT_MAX_EDITOR_CONSTRAINT_CYCLES",
    "check_constraints",
]


# Single-source-of-truth enum of constraint codes. Strings (not
# IntEnum) so the constrained-decode schema can pin them as a
# Literal[...] union when the LLM-based variant lands.
class EditorConstraint:
    WRONG_SPEAKER = "WRONG_SPEAKER"
    PHANTOM_CHARACTER = "PHANTOM_CHARACTER"
    MISSING_COSTLY_CHOICE = "MISSING_COSTLY_CHOICE"
    NO_FINAL_THIRD_TURN = "NO_FINAL_THIRD_TURN"
    FORMAT_FAILURE = "FORMAT_FAILURE"

    ALL = (
        WRONG_SPEAKER,
        PHANTOM_CHARACTER,
        MISSING_COSTLY_CHOICE,
        NO_FINAL_THIRD_TURN,
        FORMAT_FAILURE,
    )


# Sprint 5 (2026-05-28) -- hard cap on serial revision cycles. The
# pre-Sprint-5 default was 3; under the new constraint-checker
# regime quality comes from the Sprint 4 selector, not from editor
# cycles. One targeted repair turn is the maximum; if the repair
# does not clear the failing constraints the loop ships or fails
# loud (caller's choice).
DEFAULT_MAX_EDITOR_CONSTRAINT_CYCLES: int = 1


# Sprint 5 system prompt -- strictly no taste verbs. The wire-up
# sprint feeds this into a constrained-decode call against an enum
# Literal schema so the LLM cannot wander.
EDITOR_CONSTRAINTS_SYSTEM_PROMPT = """\
You are the EDITOR. You are NOT a taste editor. You check five
constraints only and return a verdict listing the constraints
that FAIL on the current draft.

Do NOT touch pacing. Do NOT ask for additional dramatic intensity.
Do NOT redraft the line. Your job is to list failing constraints
and stop.

The five constraints are:
  WRONG_SPEAKER         A line is attributed to a name not in the
                        locked cast.
  PHANTOM_CHARACTER     A proper noun that is not in the cast (and
                        not on the canonical journalistic-terms
                        roster) is named or speaks.
  MISSING_COSTLY_CHOICE The dramatic_state.costly_choice_beat does
                        not resolve in the draft -- either the
                        named slot is missing, or its state does
                        not change between before and after.
  NO_FINAL_THIRD_TURN   The final third of the voiced beats has no
                        state change (state_before ==
                        state_after across the run).
  FORMAT_FAILURE        The draft is malformed -- bracket stage
                        directions inside dialogue, JSON inside
                        dialogue, markdown headers, etc.

Output a single JSON object: {"pass_decision": <bool>,
"failing_constraints": [<one or more of the five codes above>],
"repair_note": "<one-sentence concrete repair instruction, only
when pass_decision is False>"}. JSON only. No prose preamble.

pass_decision MUST be True iff failing_constraints is empty.
"""


# ---------------------------------------------------------------------------
# Verdict dataclass
# ---------------------------------------------------------------------------


@dataclass
class EditorConstraintVerdict:
    """Verdict for the constraint-only editor.

    Shape intentionally narrow vs the pre-Sprint-5 EditorVerdict
    (no per_axis_notes, no overall_note, no failing_axes -- those
    were taste-rubric artifacts). `repair_note` is a single
    actionable sentence the Writer receives for the one targeted
    repair turn; populated only when pass_decision is False.
    """

    pass_decision: bool
    failing_constraints: List[str] = field(default_factory=list)
    repair_note: str = ""
    cycle: int = 0

    def to_dict(self) -> dict:
        return {
            "pass_decision": bool(self.pass_decision),
            "failing_constraints": list(self.failing_constraints),
            "repair_note": str(self.repair_note),
            "cycle": int(self.cycle),
        }


# ---------------------------------------------------------------------------
# Pure-Python rule-based check_constraints
# ---------------------------------------------------------------------------


def _norm(s: Any) -> str:
    return " ".join((str(s or "").strip()).lower().split())


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    """Dict-or-attribute read (Sprint 4.1 wire-up)."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _voiced_beats(plan: Any) -> List[Any]:
    return [
        b for b in (getattr(plan, "beats", None) or [])
        if str(getattr(b, "speaker", "") or "").strip() != "MUSIC"
    ]


def _cast_names(plan: Any) -> set[str]:
    names: set[str] = set()
    for c in (getattr(plan, "cast", None) or []):
        nm = str(getattr(c, "name", "") or "").strip().upper()
        if nm:
            names.add(nm)
    return names


_FORMAT_FAILURE_NEEDLES = (
    "[sfx",        # bracket stage direction
    "[music",
    "[sound",
    "```",          # markdown code fence
    "**",           # markdown bold/italic
)


def check_constraints(
    plan: Any,
    *,
    cast_names_extra: Optional[Iterable[str]] = None,
    allowed_proper_nouns: Optional[Iterable[str]] = None,
) -> EditorConstraintVerdict:
    """Run all five constraints against a Stage1Plan-shaped object
    and return a single EditorConstraintVerdict.

    Pure function. The wire-up sprint calls this from
    `run_story_room` after the writer drafts; on a failed verdict
    the repair_note is threaded into a single Writer revision turn.

    cast_names_extra        Optional iterable of additional cast
                            names to allow (e.g. ANNOUNCER, MUSIC).
                            Defaulted to add ANNOUNCER + MUSIC.
    allowed_proper_nouns    Optional iterable of journalistic terms
                            (places, agencies, things) that PHANTOM_
                            CHARACTER must not flag. Defaulted to
                            empty; caller threads from
                            news_interpreter.key_terms.
    """
    failing: list[str] = []
    notes: list[str] = []

    voiced = _voiced_beats(plan)
    cast = _cast_names(plan)
    extra = {str(n).strip().upper() for n in (cast_names_extra or [])}
    extra.update({"ANNOUNCER", "MUSIC"})
    valid_speakers = cast | extra
    allowed_pns = {
        str(n).strip().upper()
        for n in (allowed_proper_nouns or [])
    }

    # ---- WRONG_SPEAKER -------------------------------------------------
    for b in voiced:
        sp = str(getattr(b, "speaker", "") or "").strip().upper()
        if sp and sp not in valid_speakers:
            failing.append(EditorConstraint.WRONG_SPEAKER)
            notes.append(
                f"beat {getattr(b, 'beat_id', '?')} attributed to "
                f"{sp!r}, which is not in the locked cast "
                f"({sorted(cast)})."
            )
            break  # one report is enough for this constraint

    # ---- PHANTOM_CHARACTER --------------------------------------------
    # Scan voiced beat intent + objective + state_* for ALL-CAPS
    # words that are not in cast/valid/allowed sets. This is a
    # conservative phantom check -- the line-composer Phase 0 gate
    # already does the rigorous version; this constraint is a
    # backup at the BEAT-SHEET level (the writer may have planned
    # an off-cast character into the structure).
    #
    # Key signal: the SOURCE token must already be ALL CAPS (proper
    # name convention). Lowercase / titlecase tokens are common
    # English words and never count as phantoms.
    phantom_candidates: set[str] = set()
    for b in voiced:
        for attr in ("intent", "objective", "obstacle", "turn",
                     "state_before", "state_after"):
            v = str(getattr(b, attr, "") or "")
            for tok in v.split():
                bare_src = tok.strip(".,;:!?\"'()[]{}")
                if (
                    len(bare_src) >= 3
                    and bare_src.isalpha()
                    and bare_src == bare_src.upper()  # was ALL CAPS in source
                ):
                    bare = bare_src.upper()
                    if (
                        bare not in valid_speakers
                        and bare not in allowed_pns
                    ):
                        phantom_candidates.add(bare)
    # Filter out obviously generic ALL-CAPS words that crop up in
    # English (SFX, OK, TV, AM, PM, USA, NASA, FBI, CIA, NIST). The
    # caller's allowed_proper_nouns is the better signal; this is
    # just a noise floor.
    generic_caps = {"OK", "TV", "AM", "PM", "USA", "NASA", "FBI",
                    "CIA", "NIST", "MIT", "SFX", "DC", "LA", "NYC",
                    "GMT", "UTC", "AI", "ML", "CPU", "GPU"}
    phantom_candidates -= generic_caps
    if phantom_candidates:
        failing.append(EditorConstraint.PHANTOM_CHARACTER)
        notes.append(
            "non-cast proper noun(s) in the beat sheet: "
            f"{sorted(phantom_candidates)}. Either add them to "
            "the cast or remove them from the beats."
        )

    # ---- MISSING_COSTLY_CHOICE ----------------------------------------
    ds = getattr(plan, "dramatic_state", None)
    if ds is not None:
        slot = str(_attr(ds,"costly_choice_beat", "") or "").strip()
        if not slot:
            failing.append(EditorConstraint.MISSING_COSTLY_CHOICE)
            notes.append(
                "dramatic_state.costly_choice_beat is empty -- the "
                "third act needs a typed pivot."
            )
        else:
            by_slot = {
                str(getattr(b, "dialogue_slot_id", "") or "").strip(): b
                for b in voiced
            }
            by_slot.pop("", None)
            pivot = by_slot.get(slot)
            if pivot is None:
                failing.append(EditorConstraint.MISSING_COSTLY_CHOICE)
                notes.append(
                    f"costly_choice_beat={slot!r} does not name any "
                    "voiced beat's dialogue_slot_id."
                )
            else:
                sb = _norm(getattr(pivot, "state_before", ""))
                sa = _norm(getattr(pivot, "state_after", ""))
                if sb and sa and sb == sa:
                    failing.append(EditorConstraint.MISSING_COSTLY_CHOICE)
                    notes.append(
                        f"beat {getattr(pivot, 'beat_id', '?')} "
                        f"(slot {slot}) is the costly_choice_beat "
                        "but its state does not change. The choice "
                        "must move the situation."
                    )

    # ---- NO_FINAL_THIRD_TURN ------------------------------------------
    # Take the last 1/3 of voiced beats (at least 1). Every one of
    # them must NOT be a dead beat. Untyped beats are ignored.
    if voiced:
        n = len(voiced)
        cut = max(1, n // 3)
        final_third = voiced[-cut:]
        typed_dead = [
            b for b in final_third
            if (
                getattr(b, "state_before", None)
                and getattr(b, "state_after", None)
                and is_dead_beat(b)
            )
        ]
        if typed_dead and len(typed_dead) == len(final_third):
            # Every typed beat in the final third is dead -> no turn.
            failing.append(EditorConstraint.NO_FINAL_THIRD_TURN)
            notes.append(
                "the final third of the voiced beats carries no "
                "state change. The episode must turn before close."
            )

    # ---- FORMAT_FAILURE (rule-based; scans typed beat strings) -------
    fmt_offenders: list[str] = []
    for b in voiced:
        for attr in ("intent", "objective", "obstacle", "turn"):
            v = (str(getattr(b, attr, "") or "")).lower()
            if any(n in v for n in _FORMAT_FAILURE_NEEDLES):
                fmt_offenders.append(
                    f"beat {getattr(b, 'beat_id', '?')}.{attr}"
                )
                break
    if fmt_offenders:
        failing.append(EditorConstraint.FORMAT_FAILURE)
        notes.append(
            "bracket stage directions / markdown fragments in the "
            f"beat sheet: {fmt_offenders[:4]}."
        )

    # ---- Build the verdict --------------------------------------------
    if failing:
        repair_note = " | ".join(notes)
        return EditorConstraintVerdict(
            pass_decision=False,
            failing_constraints=failing,
            repair_note=repair_note,
        )
    return EditorConstraintVerdict(
        pass_decision=True,
        failing_constraints=[],
        repair_note="",
    )
