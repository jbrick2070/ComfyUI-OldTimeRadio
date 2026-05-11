"""nodes/_otr_episode_budget.py — Phase 2A episode budget machinery.

The user selects `target_words` (canonical length unit) and
`act_count` (number of acts). Everything else — per-phase word
targets, per-phase beat counts, arc-phase labels, music_inter count
— is derived from the (target_words, act_count) pair via
`ACT_COUNT_CONFIG`.

Cowork-side this module:

  * computes default + max act_count from target_words
    (`default_act_count` / `max_act_count`)
  * builds an `EpisodeBudget` from (target_words, act_count,
    include_act_breaks, num_characters) via `compute_episode_budget`
  * holds the `ARC_PHASE_GUIDANCE` table the composer's per-beat
    prompt consumes

Pure stdlib + dataclasses. No ComfyUI / torch / pydantic imports
at module load.

See script-writing-architecture synthesis §3 Phase 2A + §6 for the
authoritative design.

Status: Phase 2A of v2.0 sprint (2026-05-11). New module.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger("OTR")


__all__ = [
    "InvalidEpisodeBudgetError",
    "ACT_COUNT_CONFIG",
    "ARC_PHASE_GUIDANCE",
    "EpisodeBudget",
    "default_act_count",
    "max_act_count",
    "compute_episode_budget",
]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class InvalidEpisodeBudgetError(ValueError):
    """Raised when target_words / act_count / num_characters combination
    cannot produce a valid outline.

    Inherits from ValueError so existing widget-validation `except
    ValueError` clauses in the writer surface the structured error.
    """


# ---------------------------------------------------------------------------
# Act-count defaults / caps (per synthesis §3 Phase 2A Step 0)
# ---------------------------------------------------------------------------

# (target_words floor, default act_count) tuples. Walked in order so
# the LAST matching floor wins. Keep this in sync with the JS mirror
# in web/js/otr_act_count_widget.js — Python is authoritative.
_DEFAULT_ACT_BREAKPOINTS: tuple[tuple[int, int], ...] = (
    (30,  1),   # 30-149   → default 1 act
    (150, 2),   # 150-299  → default 2 acts
    (300, 3),   # 300+     → default 3 acts (anything longer also stays 3)
)


def default_act_count(target_words: int) -> int:
    """The conservative default act_count for `target_words`.

    User can override upward via the act_count widget (up to
    `max_act_count`). Below this default is not selectable.

    Raises InvalidEpisodeBudgetError if target_words < 30
    (single-scene minimum).
    """
    if target_words < 30:
        raise InvalidEpisodeBudgetError(
            f"target_words={target_words} below minimum of 30. "
            f"Even a single-scene piece needs ~30 words for one exchange."
        )
    result = 1
    for floor, acts in _DEFAULT_ACT_BREAKPOINTS:
        if target_words >= floor:
            result = acts
    return result


def max_act_count(target_words: int) -> int:
    """Maximum act_count the user can pick for `target_words`.

    Each act needs ~50 words for two real beats. Capped at 7 absolute
    (the ACT_COUNT_CONFIG table only goes to 7). Minimum 1.
    """
    return min(7, max(1, target_words // 50))


# ---------------------------------------------------------------------------
# ACT_COUNT_CONFIG -- per-act-count outline shape (per synthesis §3 Step 1)
# ---------------------------------------------------------------------------

ACT_COUNT_CONFIG: dict[int, dict] = {
    1: {
        "arc_phases":            ("scene",),
        "act_word_fractions":    (1.0,),
        "voiced_beats_per_act":  (3,),
        "words_per_beat_range":  (10, 50),
    },
    2: {
        "arc_phases":            ("setup", "resolution"),
        "act_word_fractions":    (0.45, 0.55),
        "voiced_beats_per_act":  (3, 3),
        "words_per_beat_range":  (18, 40),
    },
    3: {
        "arc_phases":            ("setup", "complication", "resolution"),
        "act_word_fractions":    (0.28, 0.44, 0.28),
        "voiced_beats_per_act":  (4, 6, 4),
        "words_per_beat_range":  (20, 35),
    },
    4: {
        "arc_phases":            ("setup", "rising_action", "complication",
                                  "resolution"),
        "act_word_fractions":    (0.20, 0.30, 0.30, 0.20),
        "voiced_beats_per_act":  (3, 4, 4, 3),
        "words_per_beat_range":  (22, 40),
    },
    5: {
        "arc_phases":            ("exposition", "rising_action",
                                  "complication", "climax", "resolution"),
        "act_word_fractions":    (0.15, 0.25, 0.25, 0.20, 0.15),
        "voiced_beats_per_act":  (3, 4, 4, 3, 3),
        "words_per_beat_range":  (25, 42),
    },
    6: {
        "arc_phases":            ("setup", "catalyst", "rising_action",
                                  "complication", "climax", "resolution"),
        "act_word_fractions":    (0.13, 0.15, 0.22, 0.22, 0.15, 0.13),
        "voiced_beats_per_act":  (3, 3, 4, 4, 3, 3),
        "words_per_beat_range":  (25, 45),
    },
    7: {
        "arc_phases":            ("setup", "catalyst", "rising_action",
                                  "midpoint", "complication", "climax",
                                  "resolution"),
        "act_word_fractions":    (0.10, 0.12, 0.18, 0.20, 0.18, 0.12, 0.10),
        "voiced_beats_per_act":  (2, 3, 3, 3, 3, 3, 2),
        "words_per_beat_range":  (28, 50),
    },
}


# Per-arc-phase composer guidance. The composer prompt (Phase 2A
# Step 5) appends `ARC PHASE: <phase>\n  <guidance>` to every beat
# so Mistral-Nemo gets a clear directional cue beyond `mood`.
ARC_PHASE_GUIDANCE: dict[str, str] = {
    "scene":         "A single moment or exchange. Self-contained.",
    "setup":         "Establish the situation. Introduce characters and stakes. Do not resolve.",
    "exposition":    "Establish the world and characters. Hint at the inciting incident.",
    "catalyst":      "The inciting incident lands. Characters react. Trajectory shifts.",
    "rising_action": "Stakes escalate. New information or pressure enters the scene.",
    "complication":  "Escalate or introduce conflict. Make resolution harder, not easier.",
    "midpoint":      "Reversal or revelation. The shape of the problem changes.",
    "climax":        "The decisive confrontation. Highest tension. Outcome becomes inevitable.",
    "resolution":    "Close out the arc. Show consequence. Do not introduce new conflict.",
}


# ---------------------------------------------------------------------------
# EpisodeBudget dataclass + compute_episode_budget
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EpisodeBudget:
    """Authoritative budget for one episode.

    Built once by `compute_episode_budget` from
    (target_words, act_count, include_act_breaks, num_characters).

    Consumers:
      * outline LLM prompt          per_phase_words / per_phase_beats /
                                     words_per_beat_range / arc_phases /
                                     music_inter_count / announcer_beats
      * 8 outline validators         all fields
      * composer prompt (arc_phase)  arc_phases (looked up by beat index)
    """

    act_count: int
    arc_phases: tuple[str, ...]
    per_phase_words: tuple[int, ...]
    per_phase_beats: tuple[int, ...]
    words_per_beat_range: tuple[int, int]
    music_inter_count: int
    announcer_beats: int            # always 2 (open + close)
    cast_size: int
    target_words: int               # echoed for downstream convenience


def compute_episode_budget(
    target_words: int,
    act_count: int,
    include_act_breaks: bool,
    num_characters: int,
) -> EpisodeBudget:
    """Validate widget combo and derive the EpisodeBudget.

    Raises InvalidEpisodeBudgetError on any of:
      * target_words < 30
      * act_count out of range [1, 7]
      * act_count < default_act_count(target_words)
        (user can override UPWARD but not below the default)
      * act_count > max_act_count(target_words)
      * num_characters < 1
    """
    if target_words < 30:
        raise InvalidEpisodeBudgetError(
            f"target_words={target_words} below minimum of 30."
        )
    if not (1 <= act_count <= 7):
        raise InvalidEpisodeBudgetError(
            f"act_count={act_count} out of range [1, 7]."
        )
    if num_characters < 1:
        raise InvalidEpisodeBudgetError(
            "num_characters must be >= 1"
        )

    dflt = default_act_count(target_words)
    max_allowed = max_act_count(target_words)
    if act_count < dflt:
        raise InvalidEpisodeBudgetError(
            f"act_count={act_count} below default {dflt} for "
            f"target_words={target_words}. Override upward only -- "
            f"pick {dflt} or higher."
        )
    if act_count > max_allowed:
        raise InvalidEpisodeBudgetError(
            f"act_count={act_count} exceeds max {max_allowed} for "
            f"target_words={target_words}. Each act needs ~50 words minimum."
        )

    cfg = ACT_COUNT_CONFIG[act_count]
    return EpisodeBudget(
        act_count=act_count,
        arc_phases=cfg["arc_phases"],
        per_phase_words=tuple(
            round(target_words * f) for f in cfg["act_word_fractions"]
        ),
        per_phase_beats=tuple(cfg["voiced_beats_per_act"]),
        words_per_beat_range=tuple(cfg["words_per_beat_range"]),
        music_inter_count=(act_count - 1) if include_act_breaks else 0,
        announcer_beats=2,
        cast_size=num_characters,
        target_words=target_words,
    )


# ---------------------------------------------------------------------------
# Self-test (run as `python nodes/_otr_episode_budget.py`)
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("=== _otr_episode_budget.py self-test ===")

    # Test 1: default_act_count + max_act_count sanity table.
    print("\n[Test 1] default_act_count + max_act_count sanity")
    cases = [
        # (target_words, expected_default, expected_max)
        (30,    1, 1),
        (100,   1, 2),
        (150,   2, 3),
        (200,   2, 4),
        (300,   3, 6),
        (350,   3, 7),
        (1000,  3, 7),
        (5000,  3, 7),
    ]
    for tw, dft, mx in cases:
        got_dft = default_act_count(tw)
        got_mx = max_act_count(tw)
        marker = "PASS" if (got_dft, got_mx) == (dft, mx) else "FAIL"
        print(f"  {marker}: tw={tw:5} -> default={got_dft} (expected {dft}), "
              f"max={got_mx} (expected {mx})")

    # Test 2: default_act_count rejects below-minimum.
    print("\n[Test 2] default_act_count rejects target_words < 30")
    try:
        default_act_count(29)
        print("  FAIL: 29 accepted")
    except InvalidEpisodeBudgetError as exc:
        assert "below minimum of 30" in str(exc)
        print("  PASS")

    # Test 3: compute_episode_budget for the screenshot case
    #         (target_words=350, act_count=3, include_act_breaks=True,
    #          num_characters=2). Per synthesis §3 Phase 2A.
    print("\n[Test 3] compute_episode_budget(350, 3, True, 2)")
    eb = compute_episode_budget(350, 3, True, 2)
    assert eb.act_count == 3
    assert eb.arc_phases == ("setup", "complication", "resolution")
    assert eb.per_phase_words == (98, 154, 98), \
        f"got {eb.per_phase_words}"
    assert eb.per_phase_beats == (4, 6, 4)
    assert eb.words_per_beat_range == (20, 35)
    assert eb.music_inter_count == 2
    assert eb.announcer_beats == 2
    assert eb.cast_size == 2
    print("  PASS")

    # Test 4: include_act_breaks=False zeros music_inter_count.
    print("\n[Test 4] include_act_breaks=False -> music_inter_count=0")
    eb_no_breaks = compute_episode_budget(350, 3, False, 2)
    assert eb_no_breaks.music_inter_count == 0
    print("  PASS")

    # Test 5: act_count below default rejected.
    print("\n[Test 5] act_count below default rejected")
    try:
        # tw=350 -> default 3, user picks 2 -- invalid.
        compute_episode_budget(350, 2, True, 2)
        print("  FAIL: act_count=2 below default=3 accepted")
    except InvalidEpisodeBudgetError as exc:
        assert "below default" in str(exc)
        print("  PASS")

    # Test 6: act_count above max rejected.
    print("\n[Test 6] act_count above max rejected")
    try:
        # tw=100 -> max 2, user picks 5 -- invalid.
        compute_episode_budget(100, 5, True, 2)
        print("  FAIL: act_count=5 above max=2 accepted")
    except InvalidEpisodeBudgetError as exc:
        assert "exceeds max" in str(exc)
        print("  PASS")

    # Test 7: ACT_COUNT_CONFIG fractions sum to ~1.0 for every act count.
    print("\n[Test 7] act_word_fractions sum to 1.0 per act count")
    for ac, cfg in ACT_COUNT_CONFIG.items():
        s = sum(cfg["act_word_fractions"])
        ok = abs(s - 1.0) < 1e-6
        marker = "PASS" if ok else "FAIL"
        print(f"  {marker}: act_count={ac} sum={s:.4f}")

    # Test 8: arc_phases length matches act_count for every config.
    print("\n[Test 8] arc_phases length matches act_count")
    for ac, cfg in ACT_COUNT_CONFIG.items():
        assert len(cfg["arc_phases"]) == ac, (
            f"act_count={ac} has {len(cfg['arc_phases'])} arc_phases"
        )
        assert len(cfg["voiced_beats_per_act"]) == ac
        assert len(cfg["act_word_fractions"]) == ac
        # Every arc_phase must have ARC_PHASE_GUIDANCE entry.
        for ph in cfg["arc_phases"]:
            assert ph in ARC_PHASE_GUIDANCE, \
                f"missing ARC_PHASE_GUIDANCE for {ph!r}"
    print("  PASS")

    print("\n=== _otr_episode_budget self-tests passed ===")
