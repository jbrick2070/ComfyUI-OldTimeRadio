"""Verbatim passage selection for play-form fidelity sources.

These tests run against the REAL vendored Folger corpus wherever they can, so a
regression shows up as "the actual play stopped working", not as a fixture drift.
"""

from __future__ import annotations

import pathlib

import pytest

from nodes import _otr_passage_selector as PS
from nodes._otr_episode_budget import BEAT_WORD_HARD_MAX
from nodes._otr_passage_selector import (
    Passage,
    PassageError,
    SpeakerBinding,
    chunk_speech,
    detect_layout,
    eligible_windows,
    parse_speeches,
    select_passage,
    strip_stage_directions,
)

CORPUS = (
    pathlib.Path(__file__).resolve().parent.parent
    / "config" / "source_banks" / "shakespeare" / "sources"
)
BANKS = pathlib.Path(__file__).resolve().parent.parent / "config" / "source_banks"
TRANSLATIONS = BANKS / "shakespeare" / "translations"

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

    def test_an_unreachable_word_target_still_returns_a_passage(self):
        # The word target is a REQUEST, not a gate: asking for more words than
        # the source can supply must not refuse the render. It returns the
        # closest performable passage, still verbatim, just shorter.
        passage = select_passage(
            SAMPLE, target_words=5000, cast_ceiling=6, max_beats=3, seed="x",
        )
        assert passage.speech_count >= 2
        assert passage.word_count > 0

    def test_a_tiny_word_target_still_returns_a_passage(self):
        passage = select_passage(
            SAMPLE, target_words=1, cast_ceiling=6, max_beats=3, seed="x",
        )
        assert len(passage.speakers) >= 2

    def test_a_source_with_no_performable_exchange_still_raises(self):
        # A budget disagreement is not a failure; a source that cannot field two
        # speakers within the beat budget is.
        one_voice = "ORLANDO\nHang there, my verse.\n"
        with pytest.raises(PassageError, match="no performable passage"):
            select_passage(
                one_voice, target_words=100, cast_ceiling=6,
                max_beats=14, seed="x",
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
        one = "word " * 10
        assert len(chunk_speech(one.strip())) == 1
        lines = "\n".join(["ten words in this line of verse to count it"] * 9)
        assert len(chunk_speech(lines)) == 2          # 90 words, whole lines
        assert len(chunk_speech("word " * 81)) == 2   # one over-cap line splits
        assert len(chunk_speech(lines, cap=30)) == 3

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
        # 14 -> 15 on 2026-09-19: `romeo_juliet__act1_scene1` was fetched so the
        # Japanese Tsubouchi scene had an English roster to bind its twelve
        # speakers to. A FILE COUNT, nothing about the layout rules.
        assert len(_corpus_files()) == 15

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
                len(chunk_speech(s.text)) for s in passage.speeches
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


# --------------------------------------------------------------------------- #
# the colon layout (2026-09-18): `NAME: speech`, the vendored translations
# --------------------------------------------------------------------------- #
#: Every stored translation, with the labels the EDITION writes (read off the
#: real files, not invented) and the first real speaker of each -- the line
#: the scene heading used to swallow.
VENDORED_SCENES = {
    "it/macbeth_1_3.txt": (
        ["1A STREGA", "2A STREGA", "3A STREGA",
         "TUTTE LE STREGHE CANTANDO E DANZANDO", "MACBETH", "BANQUO", "ROSSE",
         "ANGUS"], "1A STREGA", 51),
    "fr/hamlet_1_1.txt": (
        ["BERNARDO", "FRANCISCO", "HORATIO", "MARCELLUS"], "BERNARDO", 60),
    "fr/king_lear_1_1.txt": (
        ["KENT", "GLOCESTER", "EDMOND", "LEAR", "GONERIL", "CORDÉLIA",
         "RÉGANE", "ALBANY ET CORNOUAILLES", "LE DUC DE BOURGOGNE",
         "LE ROI DE FRANCE"], "KENT", 84),
    "es/as_you_like_it_3_2.txt": (
        ["ORLANDO", "CORINO", "PIEDRA", "ROSALINDA", "CELIA", "JAQUES"],
        "ORLANDO", 144),
}


def _english_source_files():
    return sorted(BANKS.glob("*/sources/*.txt"))


def _column_zero_colon_labels(text: str) -> list[str]:
    cleaned = strip_stage_directions(text)
    out = []
    for raw in cleaned.splitlines():
        line = raw.rstrip()
        if line and line == line.lstrip():
            hit = PS._colon_prefix(line)
            if hit is not None:
                out.append(hit[0])
    return out


@pytest.mark.skipif(not TRANSLATIONS.exists(), reason="translations absent")
class TestColonLayout:
    def test_the_english_corpus_is_measured_and_pinned(self):
        """THE MEASUREMENT THE GATE RESTS ON. Across every English source file
        there is exactly ONE column-0 `ALLCAPS:` line -- `STAVE I:  MARLEY'S
        GHOST`, a chapter heading in the prose Christmas Carol -- and it is the
        only label in its file, so the two-distinct-voices rule never admits
        it. The seven `“MR. SLOTE:` minutes in Cannibalism in the Cars open
        with a curly quote at column 0 and are not prefixes at all. If this
        count ever moves, the gate below needs re-arguing, not re-tuning."""
        files = _english_source_files()
        # 81 -> 82 on 2026-09-19 for the same added sidecar. THE MEASUREMENT
        # THIS GATE RESTS ON DID NOT MOVE: the column-0 label set is still
        # exactly {christmas_carol_marley: STAVE I}, checked below, so the
        # argument stands and only the tally was re-pinned.
        assert len(files) == 82, len(files)
        hits = {p: _column_zero_colon_labels(p.read_text(encoding="utf-8"))
                for p in files}
        found = {p.name: labels for p, labels in hits.items() if labels}
        assert found == {"christmas_carol_marley.txt": ["STAVE I"]}, found

    def test_no_english_file_is_read_in_the_colon_layout(self):
        """Detection, then the parse itself, both ways: `parse_speeches` must
        equal a Folger-only read on every English file, byte for byte."""
        for path in _english_source_files():
            text = path.read_text(encoding="utf-8")
            assert detect_layout(text) == PS.FOLGER_LAYOUT, path.name
            cleaned = strip_stage_directions(text)
            folger_only = []
            speeches: list[tuple[str, list[str]]] = []
            for raw in cleaned.splitlines():
                line = raw.rstrip()
                if not line.strip():
                    continue
                hit = PS._folger_prefix(line) if line == line.lstrip() else None
                if hit is not None:
                    speeches.append((hit[0], [hit[1]] if hit[1] else []))
                elif speeches:
                    speeches[-1][1].append(line.strip())
            for speaker, lines in speeches:
                body = "\n".join(lines).strip()
                if body:
                    folger_only.append(PS.Speech(index=len(folger_only),
                                                 speaker=speaker, text=body))
            assert parse_speeches(text) == tuple(folger_only), path.name

    def test_a_single_labelled_line_is_a_heading_not_a_dialogue(self):
        carol = (BANKS / "public_domain_story" / "sources"
                 / "christmas_carol_marley.txt").read_text(encoding="utf-8")
        assert detect_layout(carol) == PS.FOLGER_LAYOUT
        assert parse_speeches(carol) == ()
        # and a quoted label is never a prefix, however many there are
        minutes = "“MR. SLOTE: 'Gentlemen--I decline.'\n" \
                  "“MR. GASTON: 'Objection.'\n" \
                  "“THE CHAIR: 'Take your seat.'\n"
        assert _column_zero_colon_labels(minutes) == []
        assert parse_speeches(minutes) == ()

    @pytest.mark.parametrize("name, first, count", [
        (n, f, c) for n, (_, f, c) in VENDORED_SCENES.items()])
    def test_every_vendored_scene_parses_to_its_recorded_speech_count(
            self, name, first, count):
        """The manifest recorded `speaker_labels` at vendor time from the
        corpus module's own counter; the selector must now see the same
        speeches. Before this layout it saw 0, 0, 1 and 1."""
        text = (TRANSLATIONS / name).read_text(encoding="utf-8")
        assert detect_layout(text) == PS.COLON_LAYOUT
        speeches = parse_speeches(text)
        assert len(speeches) == count, (name, len(speeches))
        assert speeches[0].speaker == first, speeches[0]

    @pytest.mark.parametrize("name", sorted(VENDORED_SCENES))
    def test_every_label_the_edition_writes_is_a_speaker(self, name):
        text = (TRANSLATIONS / name).read_text(encoding="utf-8")
        seen: list[str] = []
        for speech in parse_speeches(text):
            if speech.speaker not in seen:
                seen.append(speech.speaker)
        assert seen == VENDORED_SCENES[name][0], seen

    @pytest.mark.parametrize("name", sorted(VENDORED_SCENES))
    def test_a_scene_heading_is_never_a_speaker(self, name):
        """`SCENA III.` / `SCÈNE I.` / `ESCENA II.` used to be the ONLY speaker
        (Macbeth, As You Like It) with the whole scene as its text."""
        text = (TRANSLATIONS / name).read_text(encoding="utf-8")
        for speech in parse_speeches(text):
            head = speech.speaker.split()[0].rstrip(".")
            assert head not in PS._HEADING_WORDS, speech.speaker
            assert "SCENA" not in speech.speaker and "SCÈNE" not in speech.speaker
        # the first line of every stored file is its heading, and it is gone
        heading = text.splitlines()[0]
        assert heading.split()[0].rstrip(".") in PS._HEADING_WORDS
        assert all(heading not in s.text for s in parse_speeches(text))

    def test_accented_and_ordinal_labels_are_speakers_and_headings_are_not(self):
        text = ("ESCENA II.\nEl bosque.\n"
                "1A STREGA: Ove sei tu stata, sorella?\n"
                "CORDÉLIA: Rien, monseigneur.\n"
                "ESCENA III: no es un personaje.\n"
                "RÉGANE: Étudiez-vous.\n")
        speakers = [s.speaker for s in parse_speeches(text)]
        assert speakers == ["1A STREGA", "CORDÉLIA", "RÉGANE"], speakers

    def test_colon_speeches_keep_their_words(self):
        text = "MACBETH: Parlate, se il potete: chi siete voi?\nBANQUO: Dio!\n"
        by = {s.speaker: s.text for s in parse_speeches(text)}
        # the colon INSIDE the speech is the speech's own
        assert by == {"MACBETH": "Parlate, se il potete: chi siete voi?",
                      "BANQUO": "Dio!"}

    def test_the_italian_scene_selects_a_passage_at_the_shipped_budget(self):
        text = (TRANSLATIONS / "it/macbeth_1_3.txt").read_text(encoding="utf-8")
        passage = select_passage(text, target_words=300, cast_ceiling=6,
                                 max_beats=14, seed="rusconi")
        assert passage.beat_cost <= 14 and len(passage.speakers) >= 2


# --------------------------------------------------------------------------- #
# speaker bindings: the vendored label -> the spoken name and the English roster
# --------------------------------------------------------------------------- #
def _italian_bindings() -> dict:
    return {
        "1A STREGA": SpeakerBinding("PRIMA STREGA", "FIRST WITCH"),
        "2A STREGA": SpeakerBinding("SECONDA STREGA", "SECOND WITCH"),
        "3A STREGA": SpeakerBinding("TERZA STREGA", "THIRD WITCH"),
        "TUTTE LE STREGHE CANTANDO E DANZANDO": SpeakerBinding("TUTTE LE STREGHE", "ALL"),
        "MACBETH": SpeakerBinding("MACBETH", "MACBETH"),
        "BANQUO": SpeakerBinding("BANQUO", "BANQUO"),
        "ROSSE": SpeakerBinding("ROSSE", "ROSS"),
        "ANGUS": SpeakerBinding("ANGUS", "ANGUS"),
    }


@pytest.mark.skipif(not TRANSLATIONS.exists(), reason="translations absent")
class TestSpeakerBindings:
    def test_bound_speeches_carry_spoken_name_label_and_roster(self):
        text = (TRANSLATIONS / "it/macbeth_1_3.txt").read_text(encoding="utf-8")
        first = parse_speeches(text, speaker_bindings=_italian_bindings())[0]
        assert (first.speaker, first.label, first.roster_name) == (
            "PRIMA STREGA", "1A STREGA", "FIRST WITCH")
        assert first.text.startswith("Ove sei tu stata")

    def test_the_three_witches_stay_three_distinct_speakers(self):
        """`eligible_windows` counts distinct `speech.speaker`; binding the
        ordinals to one name would merge three throats into one."""
        text = (TRANSLATIONS / "it/macbeth_1_3.txt").read_text(encoding="utf-8")
        speeches = parse_speeches(text, speaker_bindings=_italian_bindings())
        witches = {s.speaker for s in speeches if "STREGA" in s.speaker}
        assert witches == {"PRIMA STREGA", "SECONDA STREGA", "TERZA STREGA"}
        # the opening exchange (before the chorus) is a three-voice window
        opening = speeches[:11]
        assert {s.speaker for s in opening} == witches
        windows = eligible_windows(opening, target_words=60, cast_ceiling=6,
                                   max_beats=14, tolerance=0.9)
        assert any(len({s.speaker for s in opening[a:b + 1]}) == 3
                   for a, b in windows)

    def test_the_chorus_is_collective_and_never_inside_a_window(self):
        """Rusconi's `TUTTE LE STREGHE CANTANDO E DANZANDO` is Folger's `ALL,
        [dancing in a circle]` with the direction fused into the label. Bound
        to `ALL`, it is refused the way the English chorus is refused."""
        text = (TRANSLATIONS / "it/macbeth_1_3.txt").read_text(encoding="utf-8")
        speeches = parse_speeches(text, speaker_bindings=_italian_bindings())
        chorus = [s for s in speeches if s.speaker == "TUTTE LE STREGHE"]
        assert len(chorus) == 1 and chorus[0].is_collective
        for a, b in eligible_windows(speeches, target_words=30, cast_ceiling=8,
                                     max_beats=14, tolerance=0.9, min_words=0,
                                     max_words=None):
            assert not any(s.is_collective for s in speeches[a:b + 1])

    def test_an_unbound_label_is_carried_as_written_and_listed(self):
        text = (TRANSLATIONS / "it/macbeth_1_3.txt").read_text(encoding="utf-8")
        partial = {k: v for k, v in _italian_bindings().items() if k != "ROSSE"}
        passage = select_passage(text, target_words=300, cast_ceiling=6,
                                 max_beats=14, seed="rusconi",
                                 speaker_bindings=partial)
        assert passage.unbound_labels == ("ROSSE",)
        rosse = [s for s in parse_speeches(text, speaker_bindings=partial)
                 if s.speaker == "ROSSE"]
        assert rosse and rosse[0].roster_name == "" and rosse[0].label == ""
        assert "ROSSE" not in passage.roster_names

    def test_the_english_path_reports_nothing_unbound(self):
        text = (CORPUS / "macbeth__act1_scene3.txt").read_text(encoding="utf-8")
        passage = select_passage(text, target_words=300, cast_ceiling=6,
                                 max_beats=14, seed="folger")
        assert passage.unbound_labels == () and passage.roster_names == {}

    def test_a_bound_passage_renders_in_a_layout_that_parses_back(self):
        """CORDÉLIA is not a Folger prefix; the verse layout would glue her
        lines to the previous speaker on a re-parse."""
        text = (TRANSLATIONS / "fr/king_lear_1_1.txt").read_text(encoding="utf-8")
        bindings = {"LEAR": SpeakerBinding("LEAR", "LEAR"),
                    "CORDÉLIA": SpeakerBinding("CORDÉLIA", "CORDELIA")}
        passage = select_passage(text, target_words=120, cast_ceiling=2,
                                 max_beats=6, seed="hugo",
                                 speaker_bindings=bindings)
        back = parse_speeches(PS.render_passage_text(passage))
        assert [s.speaker for s in back] == [s.speaker for s in passage.speeches]
        assert [" ".join(s.text.split()) for s in back] == [
            " ".join(s.text.split()) for s in passage.speeches]
