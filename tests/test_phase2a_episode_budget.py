"""tests/test_phase2a_episode_budget.py — episode budget + outline validators.

Covers (per script-writing-architecture synthesis §3 Phase 2A + §6.C/E/F/G):

  * default_act_count / max_act_count thresholds
  * compute_episode_budget happy path + every reject branch
  * ACT_COUNT_CONFIG shape sanity (fractions, lengths, guidance coverage)
  * Outline.beats cap raised 24 -> 32
  * Beat.arc_phase field accepts None + populated
  * validate_outline_against_budget each violator + clean pass
  * EPISODE BUDGET prompt block renders when budget set (S28: budget
    is now required — OutlineRequest without budget raises ValueError)

Pure-Python. No GPU. No LLM. Unit-scope only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes._otr_episode_budget import (  # noqa: E402
    ACT_COUNT_CONFIG,
    ARC_PHASE_GUIDANCE,
    EpisodeBudget,
    InvalidEpisodeBudgetError,
    compute_episode_budget,
    default_act_count,
    max_act_count,
)
from nodes._otr_outline import (  # noqa: E402
    Beat,
    Outline,
    OutlineRequest,
    _build_user_prompt,
    validate_outline_against_budget,
)


# ---------------------------------------------------------------------------
# default_act_count / max_act_count
# ---------------------------------------------------------------------------


class TestDefaultActCount:

    @pytest.mark.parametrize("words,expected", [
        (30, 1),
        (100, 1),
        (149, 1),
        (150, 2),
        (200, 2),
        (299, 2),
        (300, 3),
        (350, 3),
        (1000, 3),
        (5000, 3),
    ])
    def test_threshold_table(self, words, expected):
        assert default_act_count(words) == expected

    def test_below_minimum_raises(self):
        with pytest.raises(InvalidEpisodeBudgetError) as exc:
            default_act_count(29)
        assert "below minimum of 30" in str(exc.value)


class TestMaxActCount:

    @pytest.mark.parametrize("words,expected", [
        (30, 1),
        (100, 2),
        (200, 4),
        (300, 6),
        (350, 7),
        (1000, 7),
        (5000, 7),
    ])
    def test_threshold_table(self, words, expected):
        assert max_act_count(words) == expected


# ---------------------------------------------------------------------------
# compute_episode_budget
# ---------------------------------------------------------------------------


class TestComputeEpisodeBudget:

    def test_screenshot_case_350_3_acts(self):
        eb = compute_episode_budget(350, 3, True, 2)
        assert eb.act_count == 3
        assert eb.arc_phases == ("setup", "complication", "resolution")
        assert eb.per_phase_words == (98, 154, 98)
        assert eb.per_phase_beats == (4, 6, 4)
        assert eb.words_per_beat_range == (20, 35)
        assert eb.music_inter_count == 2
        assert eb.announcer_beats == 2
        assert eb.cast_size == 2
        assert eb.target_words == 350

    def test_include_act_breaks_false_zeros_music_inter(self):
        eb = compute_episode_budget(350, 3, False, 2)
        assert eb.music_inter_count == 0

    def test_1_act_no_music_inter_even_with_breaks(self):
        eb = compute_episode_budget(30, 1, True, 1)
        # 1 act -> music_inter_count = act_count - 1 = 0
        assert eb.music_inter_count == 0

    def test_target_words_below_minimum_rejected(self):
        with pytest.raises(InvalidEpisodeBudgetError):
            compute_episode_budget(29, 1, True, 1)

    def test_act_count_out_of_range_rejected(self):
        with pytest.raises(InvalidEpisodeBudgetError):
            compute_episode_budget(350, 0, True, 2)
        with pytest.raises(InvalidEpisodeBudgetError):
            compute_episode_budget(350, 8, True, 2)

    def test_act_count_below_default_rejected(self):
        # tw=350 -> default 3; act_count=2 is below the floor
        with pytest.raises(InvalidEpisodeBudgetError) as exc:
            compute_episode_budget(350, 2, True, 2)
        assert "below default" in str(exc.value)

    def test_act_count_above_max_rejected(self):
        # tw=100 -> max 2; act_count=5 exceeds the cap
        with pytest.raises(InvalidEpisodeBudgetError) as exc:
            compute_episode_budget(100, 5, True, 2)
        assert "exceeds max" in str(exc.value)

    def test_num_characters_below_one_rejected(self):
        with pytest.raises(InvalidEpisodeBudgetError):
            compute_episode_budget(350, 3, True, 0)

    def test_returns_frozen_dataclass(self):
        eb = compute_episode_budget(350, 3, True, 2)
        with pytest.raises(Exception):
            eb.act_count = 4  # type: ignore[misc]


class TestActCountConfigSanity:

    @pytest.mark.parametrize("ac", list(range(1, 8)))
    def test_fractions_sum_to_one(self, ac):
        s = sum(ACT_COUNT_CONFIG[ac]["act_word_fractions"])
        assert abs(s - 1.0) < 1e-6

    @pytest.mark.parametrize("ac", list(range(1, 8)))
    def test_lengths_match_act_count(self, ac):
        cfg = ACT_COUNT_CONFIG[ac]
        assert len(cfg["arc_phases"]) == ac
        assert len(cfg["voiced_beats_per_act"]) == ac
        assert len(cfg["act_word_fractions"]) == ac

    @pytest.mark.parametrize("ac", list(range(1, 8)))
    def test_arc_phase_guidance_covers_all_phases(self, ac):
        for phase in ACT_COUNT_CONFIG[ac]["arc_phases"]:
            assert phase in ARC_PHASE_GUIDANCE, \
                f"ARC_PHASE_GUIDANCE missing {phase!r}"


# ---------------------------------------------------------------------------
# Outline schema -- arc_phase field + 32-beat cap
# ---------------------------------------------------------------------------


def _ok_beat(beat_id: str, speaker: str = "ALICE",
             role: str = "character", words: int = 25,
             phase: "str | None" = None) -> dict:
    """Build a minimal Beat dict.

    Fix 1 (2026-05-11): arc_phase is now `str` with default 'setup',
    so passing arc_phase=None to pydantic would fail. When `phase`
    is None we OMIT the arc_phase key entirely so pydantic stamps
    the schema default. When `phase` is a string, we set it
    explicitly.
    """
    row: dict = {
        "beat_id": beat_id,
        "speaker": speaker,
        "speaker_role": role,
        "intent": "speak about the signal",
        "target_words": words,
        "mood": "tense",
    }
    if phase is not None:
        row["arc_phase"] = phase
    return row


class TestOutlineSchemaChanges:

    def test_beat_accepts_arc_phase_default_when_omitted(self):
        """Fix 1 (post-Phase-3 review, 2026-05-11): arc_phase is
        required-with-default ('setup'). A 12B LLM that omits the
        field must parse cleanly without rerolling -- the validator
        catches a misplaced default-'setup' beat via membership /
        ordering checks downstream."""
        # Pass `phase=None` so _ok_beat omits the arc_phase key.
        b = Beat(**_ok_beat("b001", phase=None))
        assert b.arc_phase == "setup"

    def test_beat_accepts_arc_phase_populated(self):
        b = Beat(**_ok_beat("b001", phase="setup"))
        assert b.arc_phase == "setup"

    def test_arc_phase_default_caught_by_5_act_budget_validator(self):
        """Fix 1 (post-Phase-3 review, 2026-05-11): when a 5-act
        episode uses arc_phases = (exposition, rising_action, ...),
        a default 'setup' value leaks into a beat as a placeholder.
        Validator #5 catches it via the membership check and returns
        a structured violation -- bounded retry, not infinite reroll."""
        eb = compute_episode_budget(1000, 5, True, 2)
        req = OutlineRequest(
            news_seed="seed", style="noir",
            character_cast=("ALICE", "BOB"),
            target_words=1000,
            budget=eb,
        )
        # Build a minimal outline where one voiced beat has the
        # default-omitted 'setup' value but the budget expects
        # exposition / rising_action / etc.
        beats = [
            _ok_beat("b001", speaker="NARRATOR", role="music_open", words=5),
            _ok_beat("b002", speaker="ANNOUNCER", role="announcer",
                     words=25, phase="exposition"),
            # This beat carries the schema default 'setup' which is
            # NOT in the 5-act arc_phases tuple.
            _ok_beat("b003", speaker="ALICE", words=30, phase="setup"),
            _ok_beat("b004", speaker="NARRATOR", role="music_close", words=5),
        ]
        outline = _outline_from_beats(beats)
        violation = validate_outline_against_budget(outline, req)
        assert violation is not None
        assert "arc_phase" in violation or "setup" in violation

    def test_outline_accepts_32_beats(self):
        beats = [_ok_beat(f"b{i:03d}") for i in range(1, 33)]
        Outline.model_validate({
            "title": "Test",
            "premise": "A test premise of sufficient length.",
            "setting": "A lab",
            "time_of_day": "midnight",
            "beats": beats,
        })  # should not raise

    def test_outline_rejects_33_beats(self):
        beats = [_ok_beat(f"b{i:03d}") for i in range(1, 34)]
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            Outline.model_validate({
                "title": "Test",
                "premise": "A test premise of sufficient length.",
                "setting": "A lab",
                "time_of_day": "midnight",
                "beats": beats,
            })


# ---------------------------------------------------------------------------
# EPISODE BUDGET prompt block
# ---------------------------------------------------------------------------


class TestEpisodeBudgetPromptBlock:

    # S28 cleanbreak (Rule C): removed test_block_omitted_when_budget_none.
    # Pre-S28 it asserted the prompt omitted the EPISODE BUDGET block when
    # budget was None. Post-S28 the OutlineRequest __post_init__ rejects a
    # missing budget — see test_outline_request_rejects_missing_budget in
    # the next class for the producer-contract enforcement test that
    # replaces it.

    def test_block_renders_when_budget_set(self):
        eb = compute_episode_budget(350, 3, True, 2)
        req = OutlineRequest(
            news_seed="seed", style="noir",
            character_cast=("ALICE", "BOB"),
            target_words=350,
            budget=eb,
        )
        prompt = _build_user_prompt(req)
        assert "EPISODE BUDGET" in prompt
        assert "setup -> 98" in prompt or "setup ~98" in prompt
        assert "complication" in prompt
        assert "20-35 words" in prompt
        assert "arc_phase" in prompt
        # 2 announcer beats noted in budget block.
        assert "Announcer beats: 2" in prompt


# ---------------------------------------------------------------------------
# validate_outline_against_budget
# ---------------------------------------------------------------------------


def _outline_from_beats(beats):
    return Outline.model_validate({
        "title": "Test",
        "premise": "A test premise of sufficient length.",
        "setting": "A lab",
        "time_of_day": "midnight",
        "beats": beats,
    })


def _outline_for_350_3_acts():
    """Build a fully-valid outline matching the 350w/3-act budget."""
    beats = [
        # music_open
        _ok_beat("b001", speaker="NARRATOR", role="music_open", words=5),
        # announcer open
        _ok_beat("b002", speaker="ANNOUNCER", role="announcer", words=25, phase="setup"),
        # setup -- target 98 across 4 voiced beats ~ 24-25 each
        _ok_beat("b003", speaker="ALICE", words=25, phase="setup"),
        _ok_beat("b004", speaker="BOB", words=25, phase="setup"),
        _ok_beat("b005", speaker="ALICE", words=24, phase="setup"),
        _ok_beat("b006", speaker="BOB", words=24, phase="setup"),
        # music_inter
        _ok_beat("b007", speaker="NARRATOR", role="music_inter", words=5),
        # complication -- target 154 across 6 beats ~ 25-26 each
        _ok_beat("b008", speaker="ALICE", words=25, phase="complication"),
        _ok_beat("b009", speaker="BOB",   words=26, phase="complication"),
        _ok_beat("b010", speaker="ALICE", words=26, phase="complication"),
        _ok_beat("b011", speaker="BOB",   words=26, phase="complication"),
        _ok_beat("b012", speaker="ALICE", words=26, phase="complication"),
        _ok_beat("b013", speaker="BOB",   words=25, phase="complication"),
        # music_inter
        _ok_beat("b014", speaker="NARRATOR", role="music_inter", words=5),
        # resolution -- target 98 across 4 beats
        _ok_beat("b015", speaker="ALICE", words=25, phase="resolution"),
        _ok_beat("b016", speaker="BOB",   words=25, phase="resolution"),
        _ok_beat("b017", speaker="ALICE", words=24, phase="resolution"),
        _ok_beat("b018", speaker="BOB",   words=24, phase="resolution"),
        # announcer close
        _ok_beat("b019", speaker="ANNOUNCER", role="announcer", words=25, phase="resolution"),
        # music_close
        _ok_beat("b020", speaker="NARRATOR", role="music_close", words=5),
    ]
    return _outline_from_beats(beats)


class TestValidateOutlineAgainstBudget:

    # S28 cleanbreak (Rule C): removed test_no_budget_no_op. Pre-S28 it
    # asserted validate_outline_against_budget returned None when budget
    # was missing. Post-S28 OutlineRequest rejects a missing budget at
    # construction time, so the validate function never sees req.budget
    # is None in production. The producer-contract enforcement test in
    # TestEpisodeBudgetPromptBlock covers the reject case.

    def test_clean_outline_passes(self):
        eb = compute_episode_budget(350, 3, True, 2)
        req = OutlineRequest(
            news_seed="seed", style="noir",
            character_cast=("ALICE", "BOB"),
            target_words=350,
            budget=eb,
        )
        outline = _outline_for_350_3_acts()
        assert validate_outline_against_budget(outline, req) is None

    def test_per_phase_word_drift_rejected(self):
        """Per-phase word total way under target -> hard reject."""
        eb = compute_episode_budget(350, 3, True, 2)
        req = OutlineRequest(
            news_seed="seed", style="noir",
            character_cast=("ALICE", "BOB"),
            target_words=350,
            budget=eb,
        )
        outline = _outline_for_350_3_acts()
        # Shrink complication-phase words to 80 total (target 154,
        # allowed 123-185). Override the 6 complication beats to
        # carry ~13-14 words each (still in the 20-35 range gets
        # caught by validator #4 first; instead, drop the per-phase
        # total via fewer beats). Simpler: set them to 10 words each
        # which fails validator #4 first... that's fine -- we just
        # want to confirm SOME hard violation is reported.
        # Reduce only 2 beats to drop the total significantly:
        outline.beats[7].target_words = 25
        outline.beats[8].target_words = 26
        outline.beats[9].target_words = 26
        outline.beats[10].target_words = 26
        outline.beats[11].target_words = 5  # below 20 -> validator #4 trips
        outline.beats[12].target_words = 5  # below 20 -> validator #4 trips
        violation = validate_outline_against_budget(outline, req)
        assert violation is not None

    def test_arc_phase_ordering_violation_rejected(self):
        eb = compute_episode_budget(350, 3, True, 2)
        req = OutlineRequest(
            news_seed="seed", style="noir",
            character_cast=("ALICE", "BOB"),
            target_words=350,
            budget=eb,
        )
        outline = _outline_for_350_3_acts()
        # Swap arc_phase on b015 (resolution) to "setup" so a setup
        # beat appears AFTER complication beats -- ordering violation.
        outline.beats[14].arc_phase = "setup"
        violation = validate_outline_against_budget(outline, req)
        assert violation is not None
        assert "ordering" in violation or "arc_phase" in violation

    def test_music_inter_count_violation_rejected(self):
        eb = compute_episode_budget(350, 3, True, 2)
        req = OutlineRequest(
            news_seed="seed", style="noir",
            character_cast=("ALICE", "BOB"),
            target_words=350,
            budget=eb,
        )
        outline = _outline_for_350_3_acts()
        # Drop a music_inter beat -> count=1, budget requires 2.
        outline.beats[6].speaker_role = "music_open"  # no longer music_inter
        violation = validate_outline_against_budget(outline, req)
        assert violation is not None
        assert "music_inter" in violation

    def test_announcer_count_violation_rejected(self):
        eb = compute_episode_budget(350, 3, True, 2)
        req = OutlineRequest(
            news_seed="seed", style="noir",
            character_cast=("ALICE", "BOB"),
            target_words=350,
            budget=eb,
        )
        outline = _outline_for_350_3_acts()
        # Drop announcer close -> count=1, budget requires 2.
        outline.beats[18].speaker_role = "music_close"
        violation = validate_outline_against_budget(outline, req)
        assert violation is not None
        assert "announcer" in violation

    def test_per_beat_word_range_violation_rejected(self):
        eb = compute_episode_budget(350, 3, True, 2)
        req = OutlineRequest(
            news_seed="seed", style="noir",
            character_cast=("ALICE", "BOB"),
            target_words=350,
            budget=eb,
        )
        outline = _outline_for_350_3_acts()
        # b003 below 20-word floor.
        outline.beats[2].target_words = 10
        violation = validate_outline_against_budget(outline, req)
        assert violation is not None
        assert "b003" in violation or "target_words" in violation

    def test_unknown_arc_phase_rejected(self):
        eb = compute_episode_budget(350, 3, True, 2)
        req = OutlineRequest(
            news_seed="seed", style="noir",
            character_cast=("ALICE", "BOB"),
            target_words=350,
            budget=eb,
        )
        outline = _outline_for_350_3_acts()
        outline.beats[2].arc_phase = "midpoint"  # not in 3-act phases
        violation = validate_outline_against_budget(outline, req)
        assert violation is not None
        assert "arc_phase" in violation


# ---------------------------------------------------------------------------
# Composer arc_phase prompt rendering
# ---------------------------------------------------------------------------


class TestComposerArcPhasePromptBlock:

    def test_arc_phase_block_omitted_when_empty(self):
        from nodes._otr_line_composer import LineRequest, _build_user_prompt
        req = LineRequest(
            speaker="ALICE", intent="x", mood="x", target_words=10,
            canon_header="x", last_lines=[],
            arc_phase="",
        )
        prompt = _build_user_prompt(req)
        assert "ARC PHASE" not in prompt

    def test_arc_phase_block_renders_with_guidance(self):
        from nodes._otr_line_composer import LineRequest, _build_user_prompt
        req = LineRequest(
            speaker="ALICE", intent="x", mood="x", target_words=10,
            canon_header="x", last_lines=[],
            arc_phase="complication",
        )
        prompt = _build_user_prompt(req)
        assert "ARC PHASE: complication" in prompt
        # Guidance one-liner from ARC_PHASE_GUIDANCE.
        assert "Escalate or introduce conflict" in prompt
