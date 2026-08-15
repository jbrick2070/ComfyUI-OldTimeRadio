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
    BEATS_PER_ACT,
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

    def test_asking_for_more_acts_never_asks_for_less(self):
        """The operator's model. NOT a beat-count pin -- a direction check.

        OPERATOR DIRECTIVE 2026-08-15: *"it's like chasing words again -- no
        chasing beats."* So nothing in this file asserts that an act count
        buys a PARTICULAR number of beats, and nothing asserts what an episode
        came back with. A beat count is a request, exactly like a word total,
        and a test that pins the number turns the request back into a gate
        with pytest holding the gate instead of the writer.

        What survives is the only thing that was ever a defect: the REQUEST
        used to go DOWN when the operator asked for more. Seven acts asked for
        19 beats while six asked for 20, and three and four asked for the same
        14 -- a hand-tuned table left over from a word budget that no longer
        exists. This checks the direction and says nothing about the size.
        """
        counts = [
            voiced_beat_count(acts)
            for acts in range(MIN_ACT_COUNT, MAX_ACT_COUNT + 1)
        ]
        for acts, (fewer, more) in enumerate(zip(counts, counts[1:]), start=1):
            assert more > fewer, (
                f"asking for {acts + 1} acts requests fewer beats than asking "
                f"for {acts}; asking for a longer story must never ask for a "
                f"shorter one"
            )

    def test_every_act_gets_its_own_path(self):
        """An act count is a number of act paths. Nothing here sizes them.

        *"If I say 7 acts it goes through 7 different act paths"* -- so seven
        acts means seven arc phases and seven per-act entries. The SIZE of an
        entry is deliberately not asserted.
        """
        for acts, cfg in ACT_COUNT_CONFIG.items():
            assert len(cfg["arc_phases"]) == acts
            assert len(cfg["voiced_beats_per_act"]) == acts

    def test_every_act_count_has_at_least_one_beat_per_act(self):
        for acts in range(MIN_ACT_COUNT, MAX_ACT_COUNT + 1):
            assert voiced_beat_count(acts) >= acts

    def test_one_act_holds_exactly_one_act_of_beats(self):
        # The cast-capacity guard's predicate: every locked character needs a
        # beat to speak in, so the smallest episode still has to seat a cast.
        assert voiced_beat_count(1) == BEATS_PER_ACT

    def test_a_three_act_episode_holds_a_real_exchange(self):
        # Three acts is the default shape; it must have room for a scene with
        # actual back-and-forth in it, not two lines and a close.
        assert voiced_beat_count(3) == 3 * BEATS_PER_ACT

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
