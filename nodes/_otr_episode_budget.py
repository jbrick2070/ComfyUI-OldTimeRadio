"""nodes/_otr_episode_budget.py — episode act topology.

The operator selects `act_count` (1..8). That is the only knob, and
everything else — arc-phase labels, per-phase beat counts, music_inter
count — is derived from it alone.

WORDS ARE AN OBSERVATION, NEVER AN INSTRUCTION (operator directive
2026-08-14). `target_words` and every word-derived control were removed
from this module in the same change: the per-phase word allocation, the
per-beat word range widening, the word->act breakpoint table, the
default/max act gates that could REFUSE an operator's act choice, and
the word-feasibility guard. A story's length is what the story turns out
to be. Nothing here may reintroduce a length authority, in any form.

`BEAT_WORD_HARD_MAX` deliberately SURVIVES that removal and is not a
length authority: it is the Stage-3 Beat schema's structural cap, and
`_otr_passage_selector.beats_for_words` needs it to split one long
source speech across consecutive beats. Without it the Shakespeare lane
silently loses its best material — Banquo's 91-word speech, Lear's love
test, Prospero's history, Juliet's balcony. It is a fixed property of
the schema and does not move when anything else moves.

The test that separates a legal number from an illegal one: **could this
number change if a word target changed?** If yes it is a length
authority and does not belong here. If it is a fixed property of the
job or the schema, it is capacity.

Cowork-side this module:

  * builds an `EpisodeBudget` from (act_count, include_act_breaks,
    num_characters) via `compute_episode_budget`
  * holds the `ARC_PHASE_GUIDANCE` table the composer's per-beat
    prompt consumes

Pure stdlib + dataclasses. No ComfyUI / torch / pydantic imports
at module load.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger("OTR")


__all__ = [
    "InvalidEpisodeBudgetError",
    "ACT_COUNT_CONFIG",
    "BEATS_PER_ACT",
    "BEAT_WORD_HARD_MAX",
    "ARC_PHASE_GUIDANCE",
    "MIN_ACT_COUNT",
    "MAX_ACT_COUNT",
    "EpisodeBudget",
    "compute_episode_budget",
    "voiced_beat_count",
]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class InvalidEpisodeBudgetError(ValueError):
    """Raised when act_count / num_characters cannot produce an outline.

    Inherits from ValueError so existing widget-validation `except
    ValueError` clauses in the writer surface the structured error.
    """


# ---------------------------------------------------------------------------
# Act-count range
# ---------------------------------------------------------------------------

#: The operator picks any act count in this inclusive range and the pick is
#: always honoured. There is deliberately NO derived default and NO derived
#: ceiling: the previous `default_act_count` / `max_act_count` pair computed
#: both from `target_words` and RAISED when the operator's choice fell
#: outside, which made a word total able to refuse an act choice. That was a
#: word-count veto in a project whose law says word targets are advisory.
#:
#: MAX_ACT_COUNT LOWERED 8 -> 6 (PBUG-20260825-01, operator decision). The
#: outline topology is `5*act_count + 1` total beats (4 voiced beats/act +
#: 2 announcer + `act_count-1` music-interstitial, with `include_act_breaks`
#: defaulting True end to end) against `Outline.beats`'s own hard
#: `max_length=32` in `_otr_outline.py` -- which only 1..6 ever satisfy. 7
#: and 8 were reachable through this range check but GUARANTEED to fail
#: three call-frames later at `Outline` construction (36 and 41 beats
#: respectively); reproduced live on two banks the same night this was
#: found. Confirmed before lowering: the canonical workflow and every
#: variant have `act_count` saved as `'3'` (the default), never `'7'` or
#: `'8'`, so no saved graph goes invalid.
MIN_ACT_COUNT: int = 1
MAX_ACT_COUNT: int = 6


# ---------------------------------------------------------------------------
# ACT_COUNT_CONFIG -- per-act-count outline shape
# ---------------------------------------------------------------------------
#
# Two keys only. `act_word_fractions` and `words_per_beat_range` were removed
# 2026-08-14: the first split a word total across phases, the second was
# widened from that split, and both fed word counts into the outline prompt.
#
# `voiced_beats_per_act` is NOT word-derived and never was -- it is the act
# topology, which is why the beat count moves in steps rather than sliding
# with a length request.

#: Beats in one act. OPERATOR RULING 2026-08-15: *"ideally we say each act is
#: 4 beats and we have a separate LLM pass per beat"*, and an act count is a
#: number of ACT PATHS -- *"if I say 7 acts it goes through 7 different act
#: paths, that should ensure there are more with 7"*.
#:
#: What that replaced: a hand-tuned table where every row had a different
#: shape and two rows broke the operator's own model. Three acts and four
#: acts both bought 14 beats, and SEVEN acts bought 19 while six bought 20 --
#: asking for a longer story made a shorter one. The 7-act row `(2,3,3,3,3,3,2)`
#: predates the word rip and was almost certainly a word-fitting artifact:
#: act counts above 3 were only reachable when `target_words // 50` allowed
#: them, so the rows were tuned to fit a word budget rather than to describe
#: a dramatic shape. The budget is gone; the shapes it left behind are too.
#:
#: A uniform act is also the honest one. Every act runs the same arc machinery
#: and gets the same per-beat authoring passes, so there is no mechanism that
#: would make a "setup" act intrinsically thinner than a "climax" act -- the
#: old varying rows encoded a length guess, not a dramatic fact.
BEATS_PER_ACT: int = 4

ACT_COUNT_CONFIG: dict[int, dict] = {
    1: {
        "arc_phases":            ("scene",),
    },
    2: {
        "arc_phases":            ("setup", "resolution"),
    },
    3: {
        "arc_phases":            ("setup", "complication", "resolution"),
    },
    4: {
        "arc_phases":            ("setup", "rising_action", "complication",
                                  "resolution"),
    },
    5: {
        "arc_phases":            ("exposition", "rising_action",
                                  "complication", "climax", "resolution"),
    },
    6: {
        "arc_phases":            ("setup", "catalyst", "rising_action",
                                  "complication", "climax", "resolution"),
    },
}
# 7 and 8 REMOVED (PBUG-20260825-01, same change that lowered MAX_ACT_COUNT):
# both were unreachable dead entries the moment the range check above stops
# admitting them, and leaving them in this table would read as "still a
# valid choice" to the next person scanning it.

# `voiced_beats_per_act` is DERIVED, not authored: one entry per arc phase,
# every entry BEATS_PER_ACT. Derived rather than typed out so the two can
# never drift apart, and so "how many beats is an act" has exactly one owner.
for _cfg in ACT_COUNT_CONFIG.values():
    _cfg["voiced_beats_per_act"] = (BEATS_PER_ACT,) * len(_cfg["arc_phases"])
del _cfg


# The Stage-3 Beat pydantic schema hard-caps one voiced beat at 80 words.
# STRUCTURAL, NOT A LENGTH AUTHORITY -- see the module docstring. Read by
# `_otr_passage_selector.beats_for_words` to carry a long source speech
# across consecutive beats instead of dropping it.
BEAT_WORD_HARD_MAX: int = 80


# Per-arc-phase composer guidance. The composer prompt appends
# `ARC PHASE: <phase>\n  <guidance>` to every beat so the writer gets a
# clear directional cue beyond `mood`. Every phase named in
# ACT_COUNT_CONFIG must have an entry here.
ARC_PHASE_GUIDANCE: dict[str, str] = {
    "scene":         "A single moment or exchange. Self-contained.",
    "setup":         "Establish the situation. Introduce characters and stakes. Do not resolve.",
    "exposition":    "Establish the world and characters. Hint at the inciting incident.",
    "catalyst":      "The inciting incident lands. Characters react. Trajectory shifts.",
    "rising_action": "Stakes escalate. New information or pressure enters the scene.",
    "complication":  "Escalate or introduce conflict. Make resolution harder, not easier.",
    "midpoint":      "Reversal or revelation. The shape of the problem changes.",
    "crisis":        "The lowest point. What the character wanted looks lost, and no way forward is visible.",
    "climax":        "The decisive confrontation. Highest tension. Outcome becomes inevitable.",
    "resolution":    "Close out the arc. Show consequence. Do not introduce new conflict.",
}


# ---------------------------------------------------------------------------
# EpisodeBudget dataclass + compute_episode_budget
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EpisodeBudget:
    """Authoritative act topology for one episode.

    Built once by `compute_episode_budget` from
    (act_count, include_act_breaks, num_characters).

    Consumers:
      * outline LLM prompt          arc_phases / per_phase_beats /
                                     music_inter_count / announcer_beats
      * outline validators           all fields
      * composer prompt (arc_phase)  arc_phases (looked up by beat index)

    `per_phase_words`, `words_per_beat_range` and `target_words` were
    removed 2026-08-14. Nothing here carries a length instruction.
    """

    act_count: int
    arc_phases: tuple[str, ...]
    per_phase_beats: tuple[int, ...]
    music_inter_count: int
    announcer_beats: int            # always 2 (open + close)
    cast_size: int


def compute_episode_budget(
    act_count: int,
    include_act_breaks: bool,
    num_characters: int,
) -> EpisodeBudget:
    """Derive the episode's act topology from the operator's act count.

    The operator's `act_count` is ALWAYS honoured inside
    [MIN_ACT_COUNT, MAX_ACT_COUNT]. There is no derived floor, no derived
    ceiling and no feasibility guard, because all three used to be
    computed from a word total.

    Raises InvalidEpisodeBudgetError on:
      * act_count outside [MIN_ACT_COUNT, MAX_ACT_COUNT]
      * num_characters < 1
    """
    if not (MIN_ACT_COUNT <= act_count <= MAX_ACT_COUNT):
        raise InvalidEpisodeBudgetError(
            f"act_count={act_count} out of range "
            f"[{MIN_ACT_COUNT}, {MAX_ACT_COUNT}]."
        )
    if num_characters < 1:
        raise InvalidEpisodeBudgetError(
            "num_characters must be >= 1"
        )

    cfg = ACT_COUNT_CONFIG[act_count]

    return EpisodeBudget(
        act_count=act_count,
        arc_phases=cfg["arc_phases"],
        per_phase_beats=tuple(cfg["voiced_beats_per_act"]),
        music_inter_count=(act_count - 1) if include_act_breaks else 0,
        announcer_beats=2,
        cast_size=num_characters,
    )


def voiced_beat_count(act_count: int) -> int:
    """Voiced CHARACTER beats at this act count (announcer rows excluded).

    One source of truth for a number that is easy to assume wrongly: the
    beat count follows the ACT TOPOLOGY, so it moves in steps.

    This matters wherever content has to be performed one unit per beat --
    a VERBATIM passage from a play carries one speech per voiced beat, so a
    seven-speech exchange cannot be performed inside three beats. It is also
    the predicate a cast-capacity guard needs: every locked character must
    have at least one beat to speak in, and at 3 beats a four-person cast is
    a mathematical guarantee of a coverage failure, not a risk.
    """
    acts = int(act_count)
    if acts not in ACT_COUNT_CONFIG:
        raise InvalidEpisodeBudgetError(
            f"act_count {acts} is not a configured topology "
            f"(have {sorted(ACT_COUNT_CONFIG)})"
        )
    return sum(ACT_COUNT_CONFIG[acts]["voiced_beats_per_act"])


# ---------------------------------------------------------------------------
# Self-test (run as `python nodes/_otr_episode_budget.py`)
# ---------------------------------------------------------------------------

