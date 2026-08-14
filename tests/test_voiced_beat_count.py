"""One owner for "how many voiced beats does this act count buy".

REWRITTEN 2026-08-14. This file used to ask "how many beats does a WORD TARGET
buy" -- `voiced_beat_count(target_words)`, with a table of measured steps from
30 words to 900. That authority was removed with the rest of the word
machinery, and the function now takes an act count.

The surviving intent, and it is the reason this file exists: the beat count is
a property of the ACT TOPOLOGY, and assuming otherwise is how a verbatim
passage gets selected that cannot be performed. These tests pin it against the
real config rather than against a copy of the numbers.
"""

from __future__ import annotations

import pytest

from nodes._otr_episode_budget import (
    ACT_COUNT_CONFIG,
    MAX_ACT_COUNT,
    MIN_ACT_COUNT,
    InvalidEpisodeBudgetError,
    voiced_beat_count,
)


class TestVoicedBeatCount:
    def test_it_agrees_with_the_topology_it_reads(self):
        # Not a second copy of the numbers -- the same table, summed.
        for acts, cfg in ACT_COUNT_CONFIG.items():
            assert voiced_beat_count(acts) == sum(cfg["voiced_beats_per_act"])

    def test_every_operator_choice_is_answerable(self):
        # The widget offers 1..8 and every one of them must resolve, because
        # the operator's pick is always honoured now -- there is no derived
        # floor or ceiling that can refuse it.
        for acts in range(MIN_ACT_COUNT, MAX_ACT_COUNT + 1):
            assert voiced_beat_count(acts) >= 3

    def test_the_seven_act_inversion_is_pinned_not_endorsed(self):
        """7 acts yields FEWER beats than 6. Pinned so it cannot drift silently.

        Measured 2026-08-14: 1->3, 2->6, 3->14, 4->14, 5->17, 6->20,
        **7->19**, 8->22. Six acts buys 20 voiced beats and seven buys 19.

        This is INHERITED, not introduced by the word rip -- the 7-act row
        `(2,3,3,3,3,3,2)` predates it. It was probably a word-fitting artifact:
        act counts above 3 were only reachable when `target_words // 50`
        allowed them, so the rows were tuned to fit a word budget rather than
        to describe a dramatic shape.

        It contradicts the operator's stated model that more acts means a
        story with more turns in it, so it is RAISED rather than fixed here --
        changing the topology changes every episode rendered at 7 acts. When
        the operator rules, this test changes with the table.
        """
        assert voiced_beat_count(6) == 20
        assert voiced_beat_count(7) == 19
        assert voiced_beat_count(7) < voiced_beat_count(6)

    def test_every_act_count_has_at_least_one_beat_per_act(self):
        for acts in range(MIN_ACT_COUNT, MAX_ACT_COUNT + 1):
            assert voiced_beat_count(acts) >= acts

    def test_a_single_act_cannot_hold_a_four_person_cast(self):
        # Not a risk -- a pigeonhole certainty. Every locked character needs a
        # beat to speak in, and this is the cast-capacity guard's predicate.
        assert voiced_beat_count(1) < 4

    def test_a_three_act_episode_holds_a_real_exchange(self):
        # Three acts is the default shape; it must have room for a scene with
        # actual back-and-forth in it, not two lines and a close.
        assert voiced_beat_count(3) >= 14

    def test_unconfigured_act_count_raises(self):
        with pytest.raises(
            InvalidEpisodeBudgetError, match="not a configured topology"
        ):
            voiced_beat_count(99)

    def test_the_retired_word_api_is_really_gone(self):
        # Mutation guard: this file's whole previous premise was that a WORD
        # TARGET bought beats. If a word-derived helper ever comes back, this
        # is the test that should have to be deleted first.
        import nodes._otr_episode_budget as budget

        for retired in (
            "auto_act_count",
            "default_act_count",
            "max_act_count",
            "_DEFAULT_ACT_BREAKPOINTS",
        ):
            assert not hasattr(budget, retired), retired
