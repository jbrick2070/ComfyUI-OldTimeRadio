"""One owner for "how many voiced beats does this word target buy".

The number moves in steps with the ACT TOPOLOGY, not smoothly with words, and
assuming otherwise is how a verbatim passage gets selected that cannot be
performed. These tests pin the steps against the real config.
"""

from __future__ import annotations

import pytest

from nodes._otr_episode_budget import (
    ACT_COUNT_CONFIG,
    InvalidEpisodeBudgetError,
    auto_act_count,
    voiced_beat_count,
)


class TestVoicedBeatCount:
    @pytest.mark.parametrize(
        "target_words,expected_beats",
        [(30, 3), (60, 3), (120, 3), (150, 6), (200, 6), (300, 14), (900, 14)],
    )
    def test_the_measured_steps(self, target_words, expected_beats):
        assert voiced_beat_count(target_words) == expected_beats

    def test_it_agrees_with_the_topology_it_reads(self):
        for target in (30, 150, 300, 1500):
            acts = auto_act_count(target)
            expected = sum(ACT_COUNT_CONFIG[acts]["voiced_beats_per_act"])
            assert voiced_beat_count(target) == expected

    def test_explicit_act_count_bypasses_auto(self):
        for acts, cfg in ACT_COUNT_CONFIG.items():
            assert voiced_beat_count(300, act_count=acts) == sum(
                cfg["voiced_beats_per_act"]
            )

    def test_a_120_word_episode_cannot_hold_a_four_person_cast(self):
        # Not a risk -- a pigeonhole certainty. Every locked character needs a
        # beat to speak in, and this is the guard's predicate.
        assert voiced_beat_count(120) < 4

    def test_the_recommended_budget_holds_a_real_exchange(self):
        # 300 is what every fidelity manifest already recommends.
        assert voiced_beat_count(300) >= 14

    def test_unconfigured_act_count_raises(self):
        with pytest.raises(InvalidEpisodeBudgetError, match="not a configured topology"):
            voiced_beat_count(300, act_count=99)

    def test_infeasible_target_propagates_the_budget_error(self):
        with pytest.raises(InvalidEpisodeBudgetError):
            voiced_beat_count(5)
