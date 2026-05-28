"""nodes/_otr_beat_selector.py -- Sprint 4 (2026-05-28).

Best-of-N at the structure (not at line revision). Operationalises
the round-robin's §5b principle: take more shots at a good story,
do not repair a bad one.

The selector is PURE PYTHON -- no LLM call, no constrained decode.
It scores each candidate Stage1Plan on five visible-structure axes
(never taste), filters out plans that fail the Sprint 2 structural
validators, and picks the winner by summed score with a deterministic
tie break (lowest-index wins to keep behaviour reproducible).

Diversity knobs A / B / C live on the caller side (the Stage 1 call
site fans out three plan-generation calls with distinct system
prompts):
  A: moral-dilemma          -- two characters facing equally bad choices
  B: bureaucratic-absurd    -- ordinary process becomes the antagonist
  C: intimate personal-cost -- the cost lands on a body in the room

The selector does NOT name the knob a candidate came from; it only
scores the plans. The caller stamps the knob label on the audit
dict so a postmortem can tell which knob shipped.

Scoring axes (each 0 or 1; total in [0..5]):
  * clear_opposed_desires
  * costly_choice_present
  * each_beat_changes_situation
  * ending_changed_from_beginning
  * no_alarm_countdown_rescue       (penalty axis: 1 = absent = good)

UTF-8 no BOM. Module imports are stdlib + the project's own modules.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Tuple

from ._otr_beat_validators import validate_beat_sheet, is_dead_beat


__all__ = [
    "BeatSelectorScores",
    "BeatSelectorAudit",
    "NoValidBeatSheetError",
    "score_beat_sheet",
    "select_winning_beat_sheet",
]


# Phrases that trigger the no_alarm_countdown_rescue penalty axis.
# These are stock "cheap stakes" structural patterns small models
# default to under pressure. Lowercase compared against the
# concatenated intent + objective + turn text of every beat.
_ALARM_PATTERNS = (
    "alarm sounds",
    "alarm goes off",
    "countdown begins",
    "countdown starts",
    "countdown to detonation",
    "rescue team arrives",
    "rescue arrives",
    "cavalry arrives",
    "deus ex",
)


# ---------------------------------------------------------------------------
# Scores + audit dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BeatSelectorScores:
    """Per-candidate structural score. Each field is 0 or 1.

    `no_alarm_countdown_rescue` is the inverse axis -- 1 means the
    candidate is FREE of the cheap-stakes pattern. So total = sum of
    all five fields, max 5, higher is better.
    """

    clear_opposed_desires: int
    costly_choice_present: int
    each_beat_changes_situation: int
    ending_changed_from_beginning: int
    no_alarm_countdown_rescue: int

    @property
    def total(self) -> int:
        return (
            self.clear_opposed_desires
            + self.costly_choice_present
            + self.each_beat_changes_situation
            + self.ending_changed_from_beginning
            + self.no_alarm_countdown_rescue
        )


@dataclass
class BeatSelectorAudit:
    """Structured audit for the selector decision.

    The caller stamps this on `meta.beat_selector` so the postmortem
    can replay the choice. `reason` is a short human-readable
    paragraph; `scores_per_candidate` is the full score matrix.
    """

    winner_index: int
    reason: str
    scores_per_candidate: List[BeatSelectorScores] = field(default_factory=list)
    defects_per_candidate: List[List[str]] = field(default_factory=list)
    eligible_indices: List[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "winner_index": self.winner_index,
            "reason": self.reason,
            "scores_per_candidate": [
                {
                    "clear_opposed_desires": s.clear_opposed_desires,
                    "costly_choice_present": s.costly_choice_present,
                    "each_beat_changes_situation": s.each_beat_changes_situation,
                    "ending_changed_from_beginning": s.ending_changed_from_beginning,
                    "no_alarm_countdown_rescue": s.no_alarm_countdown_rescue,
                    "total": s.total,
                }
                for s in self.scores_per_candidate
            ],
            "defects_per_candidate": [list(d) for d in self.defects_per_candidate],
            "eligible_indices": list(self.eligible_indices),
        }


class NoValidBeatSheetError(RuntimeError):
    """Raised when every candidate fails the Sprint 2 structural
    validators. The selector refuses to ship a dead sheet; the
    caller's job is to re-roll the candidate fan-out (or surface a
    hard workflow failure)."""


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _norm(s: Any) -> str:
    return " ".join((str(s or "").strip()).lower().split())


def _voiced_beats(plan: Any) -> List[Any]:
    return [
        b for b in (getattr(plan, "beats", None) or [])
        if str(getattr(b, "speaker", "") or "").strip() != "MUSIC"
    ]


def _carries_alarm_pattern(plan: Any) -> bool:
    """Penalty-axis check: True when any beat intent / objective / turn
    mentions one of the cheap-stakes patterns."""
    haystack_parts: list[str] = []
    for b in _voiced_beats(plan):
        for attr in ("intent", "objective", "turn"):
            v = getattr(b, attr, None)
            if v:
                haystack_parts.append(str(v).lower())
    haystack = "\n".join(haystack_parts)
    return any(pat in haystack for pat in _ALARM_PATTERNS)


def score_beat_sheet(plan: Any) -> BeatSelectorScores:
    """Compute the 5-axis structural score for one plan.

    Pure function. The five axes are intentionally crude -- the
    selector is judging visible structure, never taste. Small local
    models can score this; humans handle the listen test.
    """
    ds = getattr(plan, "dramatic_state", None)
    voiced = _voiced_beats(plan)

    # 1. clear_opposed_desires -- DramaticState present and the two
    #    wants are non-trivially distinct (the DramaticState
    #    constructor's own validator already rejected echo strings,
    #    so we only need to confirm both fields are populated).
    clear_opposed_desires = int(bool(
        ds is not None
        and getattr(ds, "character_a_wants", "")
        and getattr(ds, "character_b_wants", "")
    ))

    # 2. costly_choice_present -- the named slot exists AND its
    #    state pivots (state_before != state_after).
    costly_choice_present = 0
    if ds is not None:
        slot = str(getattr(ds, "costly_choice_beat", "") or "").strip()
        if slot:
            by_slot = {
                str(getattr(b, "dialogue_slot_id", "") or "").strip(): b
                for b in voiced
            }
            by_slot.pop("", None)
            pivot = by_slot.get(slot)
            if pivot is not None:
                sb = _norm(getattr(pivot, "state_before", ""))
                sa = _norm(getattr(pivot, "state_after", ""))
                if sb and sa and sb != sa:
                    costly_choice_present = 1

    # 3. each_beat_changes_situation -- zero dead beats across the
    #    voiced beats. Untyped beats (no state_before / state_after)
    #    don't count against this axis (Sprint 2 design).
    each_beat_changes_situation = 1
    for b in voiced:
        if is_dead_beat(b):
            each_beat_changes_situation = 0
            break

    # 4. ending_changed_from_beginning -- DramaticState.ending_change
    #    differs from the opening beat's state_before. When either
    #    side is empty the axis fails (cannot prove change).
    ending_changed_from_beginning = 0
    if ds is not None and voiced:
        ec = _norm(getattr(ds, "ending_change", ""))
        opening = _norm(getattr(voiced[0], "state_before", ""))
        if ec and opening and ec != opening:
            ending_changed_from_beginning = 1

    # 5. no_alarm_countdown_rescue -- inverse axis, 1 = clean.
    no_alarm_countdown_rescue = 0 if _carries_alarm_pattern(plan) else 1

    return BeatSelectorScores(
        clear_opposed_desires=clear_opposed_desires,
        costly_choice_present=costly_choice_present,
        each_beat_changes_situation=each_beat_changes_situation,
        ending_changed_from_beginning=ending_changed_from_beginning,
        no_alarm_countdown_rescue=no_alarm_countdown_rescue,
    )


# ---------------------------------------------------------------------------
# Selector
# ---------------------------------------------------------------------------


def select_winning_beat_sheet(
    candidates: Iterable[Any],
) -> Tuple[Any, BeatSelectorAudit]:
    """Return (winning_plan, audit) from a list of candidate
    Stage1Plan-shaped objects.

    Algorithm:
      1. Score every candidate (always, even invalid ones -- the
         audit captures the full picture).
      2. Collect structural defects via validate_beat_sheet for
         every candidate.
      3. Filter to candidates with zero validate_beat_sheet defects.
      4. From the eligible pool, pick by highest total score; break
         ties on lowest index (deterministic).
      5. If the eligible pool is empty: raise NoValidBeatSheetError
         (audit attached as exception attribute for the caller).

    The caller decides what to do with NoValidBeatSheetError -- the
    plan's "no silent shipping of a dead sheet" rule means the
    standard recovery is a single re-roll of the fan-out, then a
    hard workflow failure.
    """
    cand_list = list(candidates or [])
    if not cand_list:
        audit = BeatSelectorAudit(
            winner_index=-1,
            reason="empty candidate list",
        )
        raise _attach_audit(NoValidBeatSheetError("no candidates"), audit)

    scores: List[BeatSelectorScores] = []
    defects: List[List[str]] = []
    for plan in cand_list:
        scores.append(score_beat_sheet(plan))
        defects.append(list(validate_beat_sheet(plan)))

    eligible = [i for i, d in enumerate(defects) if not d]

    if not eligible:
        audit = BeatSelectorAudit(
            winner_index=-1,
            reason=(
                "every candidate failed the Sprint 2 structural "
                "validators -- caller should re-roll the fan-out "
                "and not ship a dead sheet."
            ),
            scores_per_candidate=scores,
            defects_per_candidate=defects,
            eligible_indices=[],
        )
        raise _attach_audit(
            NoValidBeatSheetError(
                "all candidates failed validate_beat_sheet"
            ),
            audit,
        )

    # Pick by highest total; ties broken on lowest index (so the
    # caller's diversity-knob ordering produces reproducible
    # winners). Iterating only the eligible indices guarantees we
    # never accidentally ship a candidate that failed validation.
    winner_idx = eligible[0]
    for i in eligible[1:]:
        if scores[i].total > scores[winner_idx].total:
            winner_idx = i

    audit = BeatSelectorAudit(
        winner_index=winner_idx,
        reason=(
            f"winner=candidate[{winner_idx}] "
            f"total={scores[winner_idx].total}/5 "
            f"eligible={eligible} "
            "(tie break: lowest index of the max-score eligible set)."
        ),
        scores_per_candidate=scores,
        defects_per_candidate=defects,
        eligible_indices=eligible,
    )
    return cand_list[winner_idx], audit


def _attach_audit(
    exc: NoValidBeatSheetError, audit: BeatSelectorAudit,
) -> NoValidBeatSheetError:
    """Attach the audit to the raised exception so callers can stamp
    it on meta.beat_selector even on failure."""
    setattr(exc, "audit", audit)
    return exc
