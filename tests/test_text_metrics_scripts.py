# -*- coding: utf-8 -*-
"""The canonical word counter across scripts.

WHY THIS FILE EXISTS. `canonical_word_count` was `[A-Za-z][A-Za-z0-9'-]*`
until 2026-09-19: ASCII letters only. It counted Tsubouchi's whole Japanese
balcony scene (3,606 characters) as TWO words, so the verbatim planner refused
the vendored translation with "a 2-word passage cannot fill 12 beats" and
every Japanese Shakespeare episode was a machine translation wearing the
translator's credit line. The same pattern split `não` at the accent, so
every Portuguese, Spanish, Italian and French ledger row was over-counted.

The contract this file holds: pure-ASCII text counts EXACTLY as it always
did (frozen English ledgers are re-derived by the freeze auditor and may not
move); a Latin word with an accent is one word; CJK text is counted by its
characters at two per word because it has no spaces; and the real vendored
Japanese scene now plans.
"""
import io
import os
import re

from nodes import _otr_text_metrics as TM

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OLD_ASCII_RE = re.compile(r"[A-Za-z][A-Za-z0-9'‘’-]*")


def test_pure_ascii_counts_exactly_as_the_old_counter_did():
    """Letter for letter the old pattern, widened only to non-ASCII letters.

    `Nay--I'll` stays ONE word: the old class admitted a hyphen run inside a
    word, and a stricter continuation moved 15 of 27,597 pure-ASCII rows on
    disk -- every one a Folger double dash. The English count may not move.
    """
    samples = [
        "Nay--I'll not budge. O'er the hills--away!",
        "rock'n'roll goin' well-known 3D mp3 _under_score 2026",
        "'tis the wind -- and nothing more",
        "A--B--C 'quoted' ‘smart’ don’t",
        "",
        "   ",
    ]
    for text in samples:
        assert TM.canonical_word_count(text) == len(_OLD_ASCII_RE.findall(text)), text
    folger = os.path.join(REPO, "config", "source_banks", "shakespeare", "sources",
                          "romeo_juliet__act2_scene2.txt")
    text = io.open(folger, encoding="utf-8").read()
    assert text.isascii(), "the fixture must be pure ASCII for this to prove anything"
    assert TM.canonical_word_count(text) == len(_OLD_ASCII_RE.findall(text))


def test_an_accented_latin_word_is_one_word():
    assert TM.canonical_word_count("não é") == 2
    assert TM.canonical_word_count("Áriel torna á scena, invisível") == 5
    assert TM.canonical_word_count("¿Entonces, escravo?") == 2
    assert TM.canonical_word_count("Prós. Fer. D. Ped.") == 4


def test_cjk_is_counted_by_its_characters_at_two_per_word():
    """Han, kana and hangul carry the count; CJK punctuation does not."""
    assert TM.canonical_word_count("人の痛手") == 2          # 4 chars -> 2
    assert TM.canonical_word_count("あゝ！") == 1            # 2 kana -> 1; the mark is not counted
    assert TM.canonical_word_count("ロミオ: 人の痛手を嘲りをる") == 6   # 12 CJK chars -> 6, no Latin
    assert TM.canonical_word_count("한국어 문장") == 3        # 5 hangul -> 3
    assert TM.canonical_word_count("Romeo と Juliet") == 3   # 2 Latin + 1 kana
    assert TM.canonical_word_count("。、！？「」") == 0


def test_the_vendored_japanese_balcony_scene_now_plans():
    """The defect as it was hit: `plan_verbatim_passage` on Tsubouchi's text.

    Before the counter change the receipt read `a 2-word passage cannot fill
    12 beats` and the writer fell back to the model translation. The scene is
    in the repository, so this runs without a network and without a GPU.
    """
    from nodes import _otr_verbatim_corpus as VC
    from nodes import _otr_verbatim_lane as VL
    root = os.path.join(REPO, "config", "source_banks", "shakespeare", "translations")
    text, row = VC.vendored_text(root, "ja", "folger-romeo-juliet:act2-scene2-balcony")
    assert row and "Tsubouchi" in str(row.get("translator", "")), row
    assert TM.canonical_word_count(text) > 1000
    plan, receipt = VL.plan_verbatim_passage(
        source_text=text, source_meta={"recommended_word_budget": 300},
        num_characters=3, act_count=3,
        source_ref="folger-romeo-juliet:act2-scene2-balcony",
        speaker_map=VC.speaker_bindings(row))
    assert receipt.get("status") == "planned", receipt.get("reason")
    assert receipt.get("beat_count") == 12, receipt
    assert set(receipt.get("speakers") or []) <= {"ROMEO", "JULIET", "NURSE"}, receipt


def test_a_cjk_speech_is_cut_at_its_punctuation_on_every_seed():
    """The seed picks the window, and one window drew Juliet's 180-word
    speech: a single spaceless line that `_halve` could not cut, so the plan
    failed on that seed and succeeded on the next. Fifty seeds, every one
    must plan; and a cut speech's characters survive exactly, with nothing
    inserted where the edition printed nothing.
    """
    from nodes import _otr_passage_selector as PS
    from nodes import _otr_verbatim_corpus as VC
    root = os.path.join(REPO, "config", "source_banks", "shakespeare", "translations")
    text, row = VC.vendored_text(root, "ja", "folger-romeo-juliet:act2-scene2-balcony")
    bindings = {label: PS.SpeakerBinding(spoken=spec["spoken"], roster=spec["roster"])
                for label, spec in VC.speaker_bindings(row).items()}
    for seed in range(50):
        passage = PS.select_passage(text, target_words=300, cast_ceiling=3, max_beats=12,
                                    seed=seed, min_speakers=2, speaker_bindings=bindings)
        entries = PS.build_beat_plan(passage, beat_count=12)
        assert len(entries) == 12, (seed, len(entries))
    # the chunker on one long spaceless speech: cut after sentence marks,
    # every chunk within the cap, and the bytes re-join to the original
    line = ("あゝ、ロミオ、ロミオ！何故あなたはロミオぢゃ！" * 12).strip()
    chunks = PS.chunk_speech(line, cap=20)
    assert len(chunks) > 1 and all(TM.canonical_word_count(c) <= 20 for c in chunks), chunks
    assert "".join(chunks) == line
    assert all(c.endswith("！") for c in chunks[:-1]), chunks
    # and an English line still cuts at spaces, exactly as before
    english = " ".join("word%d" % i for i in range(45))
    assert PS.chunk_speech(english, cap=20) == (
        " ".join("word%d" % i for i in range(20)),
        " ".join("word%d" % i for i in range(20, 40)),
        " ".join("word%d" % i for i in range(40, 45)))
