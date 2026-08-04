"""Verbatim passage selection for play-form fidelity sources.

These tests run against the REAL vendored Folger corpus wherever they can, so a
regression shows up as "the actual play stopped working", not as a fixture drift.
"""

from __future__ import annotations

import pathlib

import pytest

from nodes._otr_passage_selector import (
    Passage,
    PassageError,
    eligible_windows,
    parse_speeches,
    select_passage,
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
            speeches, target_words=30, cast_ceiling=6, max_speeches=4
        )
        cramped = eligible_windows(
            speeches, target_words=30, cast_ceiling=6, max_speeches=2
        )
        assert all(end - start + 1 <= 2 for start, end in cramped)
        assert len(cramped) < len(roomy)

    def test_cast_ceiling_is_respected(self):
        speeches = parse_speeches(SAMPLE)
        for start, end in eligible_windows(
            speeches, target_words=30, cast_ceiling=2, max_speeches=8
        ):
            assert len({s.speaker for s in speeches[start:end + 1]}) <= 2

    def test_same_seed_selects_the_same_passage(self):
        text = SAMPLE
        kwargs = dict(target_words=30, cast_ceiling=6, max_speeches=4, tolerance=0.9)
        first = select_passage(text, seed="episode-a", **kwargs)
        again = select_passage(text, seed="episode-a", **kwargs)
        assert (first.first_index, first.last_index) == (again.first_index, again.last_index)

    def test_no_fitting_window_raises_rather_than_stretching(self):
        with pytest.raises(PassageError, match="no passage of"):
            select_passage(
                SAMPLE, target_words=5000, cast_ceiling=6,
                max_speeches=3, seed="x",
            )

    def test_a_passage_needs_a_beat_per_speech(self):
        with pytest.raises(PassageError, match="one voiced beat per speech"):
            eligible_windows(
                parse_speeches(SAMPLE), target_words=30,
                cast_ceiling=6, max_speeches=1,
            )

    def test_non_dialogue_source_is_refused(self):
        with pytest.raises(PassageError, match="not play-form dialogue"):
            select_passage(
                "Just prose, no speakers here at all.",
                target_words=100, cast_ceiling=6, max_speeches=3, seed="x",
            )


@pytest.mark.skipif(not CORPUS.exists(), reason="vendored Folger corpus absent")
class TestAgainstTheRealCorpus:
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
                target_words=300, cast_ceiling=6, max_speeches=14,
                seed=path.stem,
            )
            assert isinstance(passage, Passage)
            assert passage.speech_count <= 14
            assert len(passage.speakers) <= 6
            assert 225 <= passage.word_count <= 375

    def test_selected_text_is_verbatim_from_the_source(self):
        # The whole point: every word performed is the play's own.
        for path in _corpus_files():
            body = path.read_text(encoding="utf-8")
            passage = select_passage(
                body, target_words=300, cast_ceiling=6,
                max_speeches=14, seed=path.stem,
            )
            for speech in passage.speeches:
                for line in speech.text.splitlines():
                    assert line.strip() in body, (path.name, line[:60])

    def test_a_120_word_budget_cannot_hold_a_long_exchange(self):
        # 120 target words buys only THREE voiced beats, so passages there are
        # short fragments. This is why the fidelity floor is argued up to 300.
        path = CORPUS / "macbeth__act1_scene3.txt"
        passage = select_passage(
            path.read_text(encoding="utf-8"),
            target_words=120, cast_ceiling=6, max_speeches=3, seed="floor",
        )
        assert passage.speech_count <= 3
