"""Verbatim passage selection for play-form fidelity sources.

These tests run against the REAL vendored Folger corpus wherever they can, so a
regression shows up as "the actual play stopped working", not as a fixture drift.
"""

from __future__ import annotations

import pathlib

import pytest

from nodes._otr_episode_budget import BEAT_WORD_HARD_MAX
from nodes._otr_passage_selector import (
    Passage,
    PassageError,
    beats_for_words,
    eligible_windows,
    parse_speeches,
    select_passage,
    strip_stage_directions,
)

CORPUS = (
    pathlib.Path(__file__).resolve().parent.parent
    / "config" / "source_banks" / "shakespeare" / "sources"
)

# Both Folger speech layouts in one sample, plus a stage direction and an
# indented continuation line.
SAMPLE = """Scene 2
=======
[Enter Orlando, with a paper.]

ORLANDO
Hang there, my verse, in witness of my love.
   And thou, thrice-crowned queen of night, survey

TOBY  Come thy ways, Signior Fabian.

ROSALIND, [as Ganymede]
I prithee, shepherd, if that love or gold
Can in this desert place buy entertainment.

BENEDICK, [aside]  Now, divine air!
"""


def _corpus_files():
    return sorted(CORPUS.glob("*.txt"))


class TestParsing:
    def test_both_folger_layouts_and_qualifiers_parse(self):
        speeches = parse_speeches(SAMPLE)
        assert [s.speaker for s in speeches] == [
            "ORLANDO", "TOBY", "ROSALIND", "BENEDICK",
        ]

    def test_prose_layout_keeps_its_words(self):
        # The inline form must not lose the speech that follows the name --
        # a parser that only handled the verse layout dropped these entirely.
        speeches = {s.speaker: s.text for s in parse_speeches(SAMPLE)}
        assert speeches["TOBY"] == "Come thy ways, Signior Fabian."
        assert speeches["BENEDICK"] == "Now, divine air!"

    def test_stage_directions_are_never_spoken(self):
        for speech in parse_speeches(SAMPLE):
            assert "Enter Orlando" not in speech.text

    def test_indented_lines_continue_the_current_speech(self):
        orlando = parse_speeches(SAMPLE)[0]
        assert "thrice-crowned queen of night" in orlando.text

    def test_prose_without_speech_prefixes_parses_nothing(self):
        # Prose sources are a different problem; this module must not pretend
        # a narrator's account is dialogue.
        prose = (
            "I think that at that time none of us quite believed in the Time "
            "Machine. The fact is, the Time Traveller was one of those men who "
            "are too clever to be believed."
        )
        assert parse_speeches(prose) == ()


class TestSelection:
    def test_beats_bound_the_passage_not_just_words(self):
        speeches = parse_speeches(SAMPLE)
        roomy = eligible_windows(
            speeches, target_words=30, cast_ceiling=6, max_beats=4
        )
        cramped = eligible_windows(
            speeches, target_words=30, cast_ceiling=6, max_beats=2
        )
        assert all(end - start + 1 <= 2 for start, end in cramped)
        assert len(cramped) < len(roomy)

    def test_cast_ceiling_is_respected(self):
        speeches = parse_speeches(SAMPLE)
        for start, end in eligible_windows(
            speeches, target_words=30, cast_ceiling=2, max_beats=8
        ):
            assert len({s.speaker for s in speeches[start:end + 1]}) <= 2

    def test_same_seed_selects_the_same_passage(self):
        text = SAMPLE
        kwargs = dict(target_words=30, cast_ceiling=6, max_beats=4, tolerance=0.9)
        first = select_passage(text, seed="episode-a", **kwargs)
        again = select_passage(text, seed="episode-a", **kwargs)
        assert (first.first_index, first.last_index) == (again.first_index, again.last_index)

    def test_no_fitting_window_raises_rather_than_stretching(self):
        with pytest.raises(PassageError, match="no passage of"):
            select_passage(
                SAMPLE, target_words=5000, cast_ceiling=6,
                max_beats=3, seed="x",
            )

    def test_a_passage_needs_at_least_a_beat_per_speech(self):
        with pytest.raises(PassageError, match="at least one voiced beat per speech"):
            eligible_windows(
                parse_speeches(SAMPLE), target_words=30,
                cast_ceiling=6, max_beats=1,
            )

    def test_a_long_speech_costs_more_than_one_beat(self):
        # The Beat schema rejects a beat over 80 words, so a long speech spans
        # consecutive beats in the same voice. Ignoring this produced a Macbeth
        # passage carrying a 91-word Banquo speech that would have failed
        # validation downstream.
        assert beats_for_words(10) == 1
        assert beats_for_words(80) == 1
        assert beats_for_words(81) == 2
        assert beats_for_words(91) == 2
        assert beats_for_words(240) == 3

    def test_beat_budget_counts_split_speeches_not_speech_count(self):
        long_speech = "ORLANDO\n" + ("word " * 200).strip() + "\n\nCELIA\nShort reply.\n"
        speeches = parse_speeches(long_speech)
        assert len(speeches) == 2
        # 200 words = 3 beats, plus 1 = 4; a 3-beat budget cannot hold it.
        assert eligible_windows(
            speeches, target_words=205, cast_ceiling=6, max_beats=3, tolerance=0.9
        ) == ()
        assert eligible_windows(
            speeches, target_words=205, cast_ceiling=6, max_beats=4, tolerance=0.9
        )

    def test_non_dialogue_source_is_refused(self):
        with pytest.raises(PassageError, match="not play-form dialogue"):
            select_passage(
                "Just prose, no speakers here at all.",
                target_words=100, cast_ceiling=6, max_beats=3, seed="x",
            )


class TestStageDirectionsAndQualifiers:
    """Every case here was a PROVEN corruption of the real corpus, not a guess."""

    def test_qualifier_without_a_comma_is_still_a_prefix(self):
        # Folger writes "BOTTOM [sings]" with no comma. Requiring one mis-parsed
        # the line, so the song was appended to TITANIA's speech.
        text = (
            "TITANIA\nWhat angel wakes me from my flow'ry bed?\n\n"
            "BOTTOM [sings]\n\tThe finch, the sparrow, and the lark,\n"
        )
        by_speaker = {s.speaker: s.text for s in parse_speeches(text)}
        assert "finch" in by_speaker["BOTTOM"]
        assert "finch" not in by_speaker["TITANIA"]

    def test_multiline_stage_direction_never_reaches_spoken_text(self):
        text = (
            "DROMIO OF EPHESUS  thy name for an ass.\n\n"
            "[Enter Luce above, unseen by Antipholus of Ephesus\n"
            "and his company.]\n\n"
            "LUCE  What a coil is there!\n"
        )
        for speech in parse_speeches(text):
            assert "Enter Luce" not in speech.text
            assert "his company" not in speech.text

    def test_inline_and_trailing_directions_are_removed(self):
        text = (
            "ORLANDO  I am glad of your departure. Adieu, good\n"
            "Monsieur Melancholy.\t[Jaques exits.]\n\n"
            "ROSALIND, [aside to Celia]  I will speak to him like a\n"
            "saucy lackey, and under that habit play the knave\n"
            "with him. [As Ganymede.] Do you hear, forester?\n"
        )
        speeches = {s.speaker: s.text for s in parse_speeches(text)}
        assert "Jaques exits" not in speeches["ORLANDO"]
        assert "Monsieur Melancholy." in speeches["ORLANDO"]
        assert "As Ganymede" not in speeches["ROSALIND"]
        assert "Do you hear, forester?" in speeches["ROSALIND"]

    def test_collective_speakers_never_take_a_cast_slot(self):
        # "ALL" is a real Folger speaker label but cannot own a TTS voice.
        text = (
            "FIRST WITCH\nSpeak.\n\nSECOND WITCH\nDemand.\n\n"
            "ALL, [dancing in a circle]\nThe Weird Sisters, hand in hand.\n\n"
            "THIRD WITCH\nWe shall.\n"
        )
        speeches = parse_speeches(text)
        assert any(s.is_collective for s in speeches)
        for start, end in eligible_windows(
            speeches, target_words=10, cast_ceiling=6, max_beats=14, tolerance=0.9
        ):
            assert not any(s.is_collective for s in speeches[start:end + 1])


@pytest.mark.skipif(not CORPUS.exists(), reason="vendored Folger corpus absent")
class TestAgainstTheRealCorpus:
    def test_the_corpus_is_actually_present(self):
        # Without this the corpus tests pass vacuously on an empty directory.
        assert len(_corpus_files()) == 14

    def test_no_spoken_text_anywhere_carries_a_bracket(self):
        for path in _corpus_files():
            for speech in parse_speeches(path.read_text(encoding="utf-8")):
                assert "[" not in speech.text and "]" not in speech.text, (
                    path.name, speech.speaker, speech.text[:60]
                )

    def test_every_vendored_scene_parses_speakers(self):
        for path in _corpus_files():
            speeches = parse_speeches(path.read_text(encoding="utf-8"))
            assert speeches, f"{path.name} parsed no speeches"
            assert len({s.speaker for s in speeches}) >= 2, path.name

    def test_every_scene_yields_a_passage_at_the_300_word_budget(self):
        # 300 target words buys 14 voiced beats, which is the budget these
        # manifests already recommend. Every curated scene must be performable.
        for path in _corpus_files():
            passage = select_passage(
                path.read_text(encoding="utf-8"),
                target_words=300, cast_ceiling=6, max_beats=14,
                seed=path.stem,
            )
            assert isinstance(passage, Passage)
            assert passage.beat_cost <= 14
            assert len(passage.speakers) <= 6
            assert 225 <= passage.word_count <= 375

    def test_no_selected_passage_exceeds_the_beat_budget(self):
        # The defect this pins: a passage whose beat cost was counted as one per
        # speech could carry a 91-word speech that the Beat schema rejects.
        for path in _corpus_files():
            passage = select_passage(
                path.read_text(encoding="utf-8"),
                target_words=300, cast_ceiling=6, max_beats=14,
                seed=path.stem,
            )
            recomputed = sum(
                beats_for_words(s.word_count) for s in passage.speeches
            )
            assert recomputed == passage.beat_cost <= 14, path.name
            # Every speech is still whole -- splitting is pacing, not cutting.
            for speech in passage.speeches:
                assert speech.word_count > 0
                assert speech.beat_cost() >= 1
                if speech.word_count > BEAT_WORD_HARD_MAX:
                    assert speech.beat_cost() > 1

    def test_selected_text_is_verbatim_from_the_source(self):
        # The whole point: every word performed is the play's own. "Verbatim"
        # means the characters' WORDS -- bracketed stage directions are
        # performance instruction, removed before delivery, so the comparison is
        # against the direction-stripped body rather than the raw file.
        for path in _corpus_files():
            stripped = strip_stage_directions(path.read_text(encoding="utf-8"))
            passage = select_passage(
                path.read_text(encoding="utf-8"), target_words=300,
                cast_ceiling=6, max_beats=14, seed=path.stem,
            )
            for speech in passage.speeches:
                for line in speech.text.splitlines():
                    assert line.strip() in stripped, (path.name, line[:60])

    def test_a_120_word_budget_cannot_hold_a_long_exchange(self):
        # 120 target words buys only THREE voiced beats, so passages there are
        # short fragments. This is why the fidelity floor is argued up to 300.
        path = CORPUS / "macbeth__act1_scene3.txt"
        passage = select_passage(
            path.read_text(encoding="utf-8"),
            target_words=120, cast_ceiling=6, max_beats=3, seed="floor",
        )
        assert passage.speech_count <= 3
