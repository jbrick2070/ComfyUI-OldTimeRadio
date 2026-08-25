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
    MAX_ACT_COUNT,
    MIN_ACT_COUNT,
    EpisodeBudget,
    InvalidEpisodeBudgetError,
    compute_episode_budget,
)
from nodes._otr_outline import (  # noqa: E402
    Beat,
    Outline,
    OutlineRequest,
    _build_user_prompt,
    validate_outline_against_budget,
)


# ---------------------------------------------------------------------------
# compute_episode_budget -- act topology only
#
# REWRITTEN 2026-08-14. Three whole classes were deleted rather than repaired:
# TestDefaultActCount, TestMaxActCount and TestAutoActCount each pinned a
# word-derived act count, and all three helpers were removed with the word
# authority. Repairing them would have meant inventing a new meaning for
# tests whose entire premise was "how many acts does this WORD TOTAL buy".
# ---------------------------------------------------------------------------


class TestComputeEpisodeBudget:

    def test_three_act_shape(self):
        eb = compute_episode_budget(3, True, 2)
        assert eb.act_count == 3
        assert eb.arc_phases == ("setup", "complication", "resolution")
        # ONE ENTRY PER ACT PATH. The SIZE is deliberately not asserted --
        # "no chasing beats" (operator 2026-08-15), so a beat count is a
        # request and no test pins it. The old hand-tuned (4, 6, 4) went with
        # the word budget it had been fitted to.
        assert len(eb.per_phase_beats) == 3
        assert eb.music_inter_count == 2
        assert eb.announcer_beats == 2
        assert eb.cast_size == 2

    def test_include_act_breaks_false_zeros_music_inter(self):
        assert compute_episode_budget(3, False, 2).music_inter_count == 0

    def test_1_act_no_music_inter_even_with_breaks(self):
        # music_inter_count = act_count - 1 = 0
        assert compute_episode_budget(1, True, 1).music_inter_count == 0

    def test_act_count_out_of_range_rejected(self):
        with pytest.raises(InvalidEpisodeBudgetError):
            compute_episode_budget(MIN_ACT_COUNT - 1, True, 2)
        with pytest.raises(InvalidEpisodeBudgetError):
            compute_episode_budget(MAX_ACT_COUNT + 1, True, 2)

    def test_seven_and_eight_acts_are_rejected_again(self):
        # The ceiling moved 7 -> 8 on 2026-08-14, making this test assert
        # 8 was IN range; it moved back to 6 on 2026-08-25
        # (PBUG-20260825-01) -- 7 and 8 were reachable through this check
        # but guaranteed to fail three frames later at Outline construction
        # (Outline.beats' own max_length=32 only fits act_count<=6). Both
        # values are OUT of range again, same as the original pre-08-14
        # ceiling, for an unrelated reason this time.
        with pytest.raises(InvalidEpisodeBudgetError):
            compute_episode_budget(7, True, 2)
        with pytest.raises(InvalidEpisodeBudgetError):
            compute_episode_budget(8, True, 2)

    def test_num_characters_below_one_rejected(self):
        with pytest.raises(InvalidEpisodeBudgetError):
            compute_episode_budget(3, True, 0)

    def test_returns_frozen_dataclass(self):
        eb = compute_episode_budget(3, True, 2)
        with pytest.raises(Exception):
            eb.act_count = 4  # type: ignore[misc]

    # --- the word authority is gone and must stay gone -------------------

    def test_every_operator_act_choice_is_honoured(self):
        """No derived floor, no derived ceiling, no refusal.

        The removed `default_act_count` / `max_act_count` pair could REFUSE
        an act count because of a word total -- a word-count veto in a
        project whose law says word targets are advisory.
        """
        for acts in range(MIN_ACT_COUNT, MAX_ACT_COUNT + 1):
            eb = compute_episode_budget(acts, True, 2)
            assert eb.act_count == acts

    def test_no_word_field_survives_on_the_budget(self):
        eb = compute_episode_budget(3, True, 2)
        for banned in (
            "target_words", "per_phase_words", "words_per_beat_range",
        ):
            assert not hasattr(eb, banned), banned

    def test_the_retired_word_helpers_are_really_gone(self):
        import nodes._otr_episode_budget as budget

        for retired in (
            "auto_act_count", "default_act_count", "max_act_count",
            "_DEFAULT_ACT_BREAKPOINTS", "_max_target_words_for_act_count",
        ):
            assert not hasattr(budget, retired), retired


class TestActCountConfigSanity:

    @pytest.mark.parametrize("ac", list(range(MIN_ACT_COUNT, MAX_ACT_COUNT + 1)))
    def test_lengths_match_act_count(self, ac):
        cfg = ACT_COUNT_CONFIG[ac]
        assert len(cfg["arc_phases"]) == ac
        assert len(cfg["voiced_beats_per_act"]) == ac

    @pytest.mark.parametrize("ac", list(range(MIN_ACT_COUNT, MAX_ACT_COUNT + 1)))
    def test_arc_phase_guidance_covers_all_phases(self, ac):
        for phase in ACT_COUNT_CONFIG[ac]["arc_phases"]:
            assert phase in ARC_PHASE_GUIDANCE,                 f"ARC_PHASE_GUIDANCE missing {phase!r}"

    @pytest.mark.parametrize("ac", list(range(MIN_ACT_COUNT, MAX_ACT_COUNT + 1)))
    def test_no_word_keys_left_in_the_topology(self, ac):
        cfg = ACT_COUNT_CONFIG[ac]
        assert "act_word_fractions" not in cfg
        assert "words_per_beat_range" not in cfg

    def test_the_table_covers_exactly_the_offered_range(self):
        assert sorted(ACT_COUNT_CONFIG) == list(
            range(MIN_ACT_COUNT, MAX_ACT_COUNT + 1)
        )



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

    def test_block_renders_as_nonbinding_plan(self):
        eb = compute_episode_budget(3, True, 2)
        req = OutlineRequest(
            news_seed="seed", style="noir",
            character_cast=("ALICE", "BOB"),
            budget=eb,
        )
        prompt = _build_user_prompt(req)
        assert "EPISODE PLAN:" in prompt
        # THE WORD LINE IS GONE (2026-08-14). This assertion used to require
        # "Requested spoken length: about 350 words" -- the word count
        # physically reaching the model. Its absence is now the contract.
        assert "Requested spoken length" not in prompt
        assert "Target total dialogue length" not in prompt
        assert "350" not in prompt
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
        eb = compute_episode_budget(3, True, 2)
        req = OutlineRequest(
            news_seed="seed", style="noir",
            character_cast=("ALICE", "BOB"),
            budget=eb,
        )
        outline = _outline_for_350_3_acts()
        assert validate_outline_against_budget(outline, req) is None

    def test_per_phase_word_drift_is_metadata_only(self):
        eb = compute_episode_budget(3, True, 2)
        req = OutlineRequest(
            news_seed="seed", style="noir",
            character_cast=("ALICE", "BOB"),
            budget=eb,
        )
        outline = _outline_for_350_3_acts()
        assert validate_outline_against_budget(outline, req) is None

    def test_arc_phase_ordering_is_metadata_only(self):
        eb = compute_episode_budget(3, True, 2)
        req = OutlineRequest(
            news_seed="seed", style="noir",
            character_cast=("ALICE", "BOB"),
            budget=eb,
        )
        outline = _outline_for_350_3_acts()
        outline.beats[14].arc_phase = "setup"
        assert validate_outline_against_budget(outline, req) is None

    def test_music_inter_count_violation_rejected(self):
        eb = compute_episode_budget(3, True, 2)
        req = OutlineRequest(
            news_seed="seed", style="noir",
            character_cast=("ALICE", "BOB"),
            budget=eb,
        )
        outline = _outline_for_350_3_acts()
        # Drop a music_inter beat -> count=1, budget requires 2.
        outline.beats[6].speaker_role = "music_open"  # no longer music_inter
        violation = validate_outline_against_budget(outline, req)
        assert violation is not None
        assert "music_inter" in violation

    def test_announcer_count_violation_rejected(self):
        eb = compute_episode_budget(3, True, 2)
        req = OutlineRequest(
            news_seed="seed", style="noir",
            character_cast=("ALICE", "BOB"),
            budget=eb,
        )
        outline = _outline_for_350_3_acts()
        # Drop announcer close -> count=1, budget requires 2.
        outline.beats[18].speaker_role = "music_close"
        violation = validate_outline_against_budget(outline, req)
        assert violation is not None
        assert "announcer" in violation

    def test_per_beat_word_range_is_metadata_only(self):
        eb = compute_episode_budget(3, True, 2)
        req = OutlineRequest(
            news_seed="seed", style="noir",
            character_cast=("ALICE", "BOB"),
            budget=eb,
        )
        outline = _outline_for_350_3_acts()
        assert validate_outline_against_budget(outline, req) is None

    def test_unknown_arc_phase_is_metadata_only(self):
        eb = compute_episode_budget(3, True, 2)
        req = OutlineRequest(
            news_seed="seed", style="noir",
            character_cast=("ALICE", "BOB"),
            budget=eb,
        )
        outline = _outline_for_350_3_acts()
        outline.beats[2].arc_phase = "midpoint"
        assert validate_outline_against_budget(outline, req) is None


# ---------------------------------------------------------------------------
# Composer arc_phase prompt rendering
# ---------------------------------------------------------------------------


class TestComposerArcPhasePromptBlock:

    def test_arc_phase_block_omitted_when_empty(self):
        from nodes._otr_line_composer import LineRequest, _build_user_prompt
        req = LineRequest(
            speaker="ALICE", intent="x", mood="x",
            canon_header="x", last_lines=[],
            arc_phase="",
        )
        prompt = _build_user_prompt(req)
        assert "ARC PHASE" not in prompt

    def test_arc_phase_block_renders_with_guidance(self):
        from nodes._otr_line_composer import LineRequest, _build_user_prompt
        req = LineRequest(
            speaker="ALICE", intent="x", mood="x",
            canon_header="x", last_lines=[],
            arc_phase="complication",
        )
        prompt = _build_user_prompt(req)
        assert "ARC PHASE: complication" in prompt
        # Guidance one-liner from ARC_PHASE_GUIDANCE.
        assert "Escalate or introduce conflict" in prompt
