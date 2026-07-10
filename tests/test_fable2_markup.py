"""scifi_fable2 markup parser -- every defect class + parser properties (S0).

Pure Python: no GPU, no LLM, no I/O. Grammar single source of truth is the
pack seam fable2_script_system (architecture doc sections 4 + 6).
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes._otr_fable2_markup import (  # noqa: E402
    Fable2ParseDefect as D,
    ParsedScript,
    parse_fable2_markup,
    render_defects,
)

CAST = ("MARA", "IVO")

GOLDEN = """\
TITLE: The Glass Meridian
MUSIC: low theremin, patient tape hiss
ANNOUNCER: Tonight, a story of a lens that would not lie, and the two people who asked it to.
SCENE 1: A hillside observatory, past midnight
MARA: The plate is dry. Read it to me.
IVO: You already know the number.
MARA: I know my number. I want yours.
MUSIC: single oboe, uncertain
SCENE 2: The observatory stairwell, minutes later
IVO: The committee meets at nine.
MARA: Then we have until nine.
ANNOUNCER: The plate kept its figure, and the hill kept its silence.
ANNOUNCER: Some numbers are only ever read once.
CODA: Beyond tonight's quiet hillside, a real measurement waits:
MUSIC: closing brass, resolving
END.
"""


def _codes(defects):
    return [d.code for d in defects]


def _parse(text, cast=CAST):
    return parse_fable2_markup(text, cast)


# ---------------------------------------------------------------------------
# 1. Golden happy path
# ---------------------------------------------------------------------------

class TestGolden:
    def test_clean_parse(self):
        script, defects = _parse(GOLDEN)
        assert defects == ()
        assert isinstance(script, ParsedScript)

    def test_structure(self):
        script, _ = _parse(GOLDEN)
        assert script.title == "The Glass Meridian"
        assert script.music_open == "low theremin, patient tape hiss"
        assert script.music_inter == ((1, "single oboe, uncertain"),)
        assert script.music_close == "closing brass, resolving"
        assert len(script.announcer_intro) == 1
        assert len(script.announcer_outro) == 2
        assert script.coda.endswith(":")
        assert [s.n for s in script.scenes] == [1, 2]
        assert script.scenes[0].setting == (
            "A hillside observatory, past midnight")

    def test_word_counters_are_separate(self):
        # r4/M4: character and announcer words are SEPARATE counters; the
        # coda is announcer-spoken and counts as announcer words.
        script, _ = _parse(GOLDEN)
        assert script.character_word_count == 30
        assert script.announcer_word_count == 44

    def test_lines_stay_per_constituent(self):
        # r2/M4: the parser NEVER merges same-speaker runs -- P7 merges and
        # proves verbatim per constituent.
        text = GOLDEN.replace(
            "MARA: Then we have until nine.",
            "MARA: Then we have until nine.\nMARA: Every minute of it.")
        script, defects = _parse(text)
        assert defects == ()
        lines = script.scenes[1].lines
        assert [ln.speaker for ln in lines] == ["IVO", "MARA", "MARA"]
        assert lines[2].text == "Every minute of it."


# ---------------------------------------------------------------------------
# 2. Every defect class
# ---------------------------------------------------------------------------

class TestDefectClasses:
    def test_missing_title(self):
        script, defects = _parse(GOLDEN.split("\n", 1)[1])
        assert script is None
        assert D.MISSING_TITLE in _codes(defects)

    def test_duplicate_title(self):
        text = GOLDEN.replace(
            "SCENE 1:", "TITLE: The Second Meridian\nSCENE 1:", 1)
        _, defects = _parse(text)
        assert D.DUPLICATE_TITLE in _codes(defects)

    def test_missing_end_is_the_only_defect(self):
        # Truncation retry contract: MISSING_END must not drown in its own
        # cascade -- the postamble-completeness skeleton checks only run at
        # a real END. line.
        text = GOLDEN.replace("END.\n", "")
        script, defects = _parse(text)
        assert script is None
        assert _codes(defects) == [D.MISSING_END]

    def test_hard_truncation_still_just_missing_end(self):
        # Cut mid-scene: no coda/outro/close music. Still ONE defect.
        text = GOLDEN.split("MUSIC: single oboe")[0]
        _, defects = _parse(text)
        assert _codes(defects) == [D.MISSING_END]

    def test_content_after_end(self):
        _, defects = _parse(GOLDEN + "MARA: One more thing.\n")
        assert D.CONTENT_AFTER_END in _codes(defects)

    def test_bad_line_shape(self):
        text = GOLDEN.replace(
            "MARA: The plate is dry. Read it to me.",
            "MARA: The plate is dry. Read it to me.\n"
            "she reaches for the plate.")
        _, defects = _parse(text)
        assert D.BAD_LINE_SHAPE in _codes(defects)

    def test_unknown_speaker_is_hard_no_remap(self):
        # r1/CUT2: no Levenshtein remap in S1-S3 -- even a one-letter
        # near-miss of a cast name is a hard defect.
        text = GOLDEN.replace(
            "IVO: You already know the number.",
            "IVOR: You already know the number.")
        _, defects = _parse(text)
        hits = [d for d in defects if d.code is D.UNKNOWN_SPEAKER]
        assert hits and hits[0].detail == "IVOR"

    def test_speaker_line_delivery_tag_is_stripped_not_a_defect(self):
        # S1b live-smoke hardening (2026-07-10): short parenthetical
        # delivery tags on SPEAKER lines are delete-only NORMALIZED
        # (legacy stage_dir_stripped precedent) and COLLECTED, never a
        # reroll defect. The spoken words survive untouched.
        text = GOLDEN.replace(
            "MARA: Then we have until nine.",
            "MARA: (checks the clock) Then we have until nine.")
        script, defects = _parse(text)
        assert script is not None, defects
        spoken = [ln.text for sc in script.scenes for ln in sc.lines]
        assert "Then we have until nine." in spoken
        assert not any("checks the clock" in t for t in spoken)
        assert any("stage_direction_stripped" in n
                   for n in script.normalizations)

    def test_markdown_bold_labels_are_stripped_not_a_defect(self):
        text = GOLDEN.replace("TITLE: The Glass Meridian",
                              "**TITLE: The Glass Meridian**")
        script, defects = _parse(text)
        assert script is not None, defects
        assert script.title == "The Glass Meridian"
        assert any("markdown_emphasis_stripped" in n
                   for n in script.normalizations)

    def test_clean_parse_records_no_normalizations(self):
        script, defects = _parse(GOLDEN)
        assert script is not None, defects
        assert script.normalizations == ()

    def test_paren_on_labeled_line_stays_a_defect(self):
        text = GOLDEN.replace(
            "MUSIC: closing brass, resolving",
            "MUSIC: closing brass (fade slowly), resolving")
        _, defects = _parse(text)
        assert D.PAREN_OR_BRACKET in _codes(defects)

    def test_embedded_paren_group_stays_a_defect(self):
        # kibitz r2 M1: an EMBEDDED group can be spoken/source content
        # ("The sample (A17) moved.") -- only LINE-LEADING delivery tags
        # normalize; anything embedded stays a hard defect.
        text = GOLDEN.replace(
            "MARA: Then we have until nine.",
            "MARA: The sample (A17) moved before nine.")
        _, defects = _parse(text)
        assert D.PAREN_OR_BRACKET in _codes(defects)

    def test_digit_bearing_leading_group_stays_a_defect(self):
        text = GOLDEN.replace(
            "MARA: Then we have until nine.",
            "MARA: (at 9:15) Then we have until nine.")
        _, defects = _parse(text)
        assert D.PAREN_OR_BRACKET in _codes(defects)

    def test_all_stage_direction_line_stays_a_defect(self):
        # A speaker line that is NOTHING but a delivery tag has no spoken
        # words to keep -- that is a real defect, not a normalization.
        text = GOLDEN.replace(
            "MARA: Then we have until nine.",
            "MARA: (long pause)")
        _, defects = _parse(text)
        assert D.BAD_LINE_SHAPE in _codes(defects)

    def test_coda_terminal_period_normalizes_to_pivot_colon(self):
        # 15th live smoke (2026-07-10): the pivot colon is a STRUCTURAL
        # seam marker, not a spoken word -- terminal '.' normalizes,
        # flagged; the words stay untouched.
        text = GOLDEN.replace(
            "CODA: Beyond tonight's quiet hillside, a real measurement "
            "waits:",
            "CODA: Beyond tonight's quiet hillside, a real measurement "
            "waits.")
        script, defects = _parse(text)
        assert script is not None, defects
        assert script.coda.endswith(":")
        assert any("coda_pivot_colon_normalized" in n
                   for n in script.normalizations)

    def test_normalize_helper_matches_parser_view(self):
        from nodes._otr_fable2_markup import normalize_fable2_markup_text
        text = GOLDEN.replace(
            "MARA: Then we have until nine.",
            "**MARA:** (dryly) Then we have until nine.")
        norm = normalize_fable2_markup_text(text)
        assert "MARA: Then we have until nine." in norm
        assert "**" not in norm and "(dryly)" not in norm
        script, defects = _parse(text)
        assert script is not None, defects
        # every parsed constituent is a substring of the normalized draft
        # (the proof-gate property the runner relies on)
        for sc in script.scenes:
            for ln in sc.lines:
                assert ln.text in norm

    def test_bracket_flagged_on_any_line_shape(self):
        text = GOLDEN.replace(
            "SCENE 2: The observatory stairwell, minutes later",
            "SCENE 2: The observatory stairwell [interior]")
        _, defects = _parse(text)
        assert D.PAREN_OR_BRACKET in _codes(defects)

    def test_quoted_dialogue(self):
        text = GOLDEN.replace(
            "IVO: The committee meets at nine.",
            'IVO: The committee meets at "nine sharp".')
        _, defects = _parse(text)
        assert D.QUOTED_DIALOGUE in _codes(defects)

    def test_scene_order(self):
        text = GOLDEN.replace("SCENE 2:", "SCENE 3:")
        _, defects = _parse(text)
        assert D.SCENE_ORDER in _codes(defects)

    def test_empty_scene(self):
        text = GOLDEN.replace(
            "ANNOUNCER: The plate kept its figure",
            "SCENE 3: The dome, later\n"
            "ANNOUNCER: The plate kept its figure")
        _, defects = _parse(text)
        hits = [d for d in defects if d.code is D.EMPTY_SCENE]
        assert hits and "SCENE 3" in hits[0].detail

    def test_skeleton_break_missing_open_music(self):
        text = GOLDEN.replace(
            "MUSIC: low theremin, patient tape hiss\n", "", 1)
        _, defects = _parse(text)
        hits = [d for d in defects if d.code is D.SKELETON_BREAK]
        assert any("opening MUSIC" in d.detail for d in hits)

    def test_skeleton_break_character_line_before_scene_one(self):
        text = GOLDEN.replace(
            "SCENE 1:", "MARA: Too early for this.\nSCENE 1:", 1)
        _, defects = _parse(text)
        hits = [d for d in defects if d.code is D.SKELETON_BREAK]
        assert any("before SCENE 1" in d.detail for d in hits)

    def test_cast_member_silent(self):
        _, defects = _parse(GOLDEN, cast=("MARA", "IVO", "NOOR"))
        hits = [d for d in defects if d.code is D.CAST_MEMBER_SILENT]
        assert hits and hits[0].detail == "NOOR"

    def test_coda_shape(self):
        # 15th live smoke (2026-07-10): a terminal '.' NORMALIZES to the
        # pivot colon (structural seam marker, not a spoken word); the
        # defect-worthy coda shape is an INNER sentence break.
        text = GOLDEN.replace(
            "CODA: Beyond tonight's quiet hillside, a real measurement waits:",
            "CODA: The hillside sleeps. A real measurement waits:")
        _, defects = _parse(text)
        assert D.CODA_SHAPE in _codes(defects)

    def test_multiple_coda(self):
        text = GOLDEN.replace(
            "MUSIC: closing brass, resolving",
            "CODA: And a second pivot, needlessly:\n"
            "MUSIC: closing brass, resolving")
        _, defects = _parse(text)
        assert D.MULTIPLE_CODA in _codes(defects)

    def test_every_defect_class_is_reachable(self):
        # Meta-pin: the tests above cover the WHOLE enum (a new class added
        # to the parser without a test dies here).
        covered = {
            D.MISSING_TITLE, D.DUPLICATE_TITLE, D.MISSING_END,
            D.CONTENT_AFTER_END, D.BAD_LINE_SHAPE, D.UNKNOWN_SPEAKER,
            D.PAREN_OR_BRACKET, D.QUOTED_DIALOGUE, D.SCENE_ORDER,
            D.EMPTY_SCENE, D.SKELETON_BREAK, D.CAST_MEMBER_SILENT,
            D.CODA_SHAPE, D.MULTIPLE_CODA,
        }
        assert covered == set(D)


# ---------------------------------------------------------------------------
# 3. Contract edges
# ---------------------------------------------------------------------------

class TestContract:
    def test_defective_parse_returns_none_script(self):
        script, defects = _parse(GOLDEN.replace("END.\n", ""))
        assert script is None and defects

    def test_clean_parse_returns_empty_defects(self):
        script, defects = _parse(GOLDEN)
        assert script is not None and defects == ()

    def test_empty_text(self):
        script, defects = _parse("")
        assert script is None
        assert D.MISSING_TITLE in _codes(defects)
        assert D.MISSING_END in _codes(defects)

    def test_render_defects_quotes_every_code(self):
        _, defects = _parse("")
        rendered = render_defects(defects)
        for d in defects:
            assert d.code.value in rendered

    def test_blank_lines_are_ignored(self):
        spaced = GOLDEN.replace("\n", "\n\n")
        script, defects = _parse(spaced)
        assert defects == ()
        assert script.character_word_count == 30

    def test_announcer_never_counts_as_cast(self):
        # ANNOUNCER is implicitly legal and never required to "speak" as a
        # cast member.
        script, defects = _parse(GOLDEN)
        assert defects == ()
        speakers = {ln.speaker for s in script.scenes for ln in s.lines}
        assert "ANNOUNCER" not in speakers


# ---------------------------------------------------------------------------
# 4. Property-style: seeded generator round-trip + mutation classes
# ---------------------------------------------------------------------------

_NAMES = ("MARA", "IVO", "NOOR", "VERA", "ELIAS", "DR. HALE", "JUNE")
_WORDS = ("signal", "plate", "ledger", "antenna", "quiet", "number",
          "meridian", "hill", "reads", "waits", "again", "tonight")


def _gen_script(rng: random.Random):
    """Random grammar-legal script + its expected tallies."""
    cast = list(rng.sample(_NAMES, rng.randint(1, 4)))
    lines = ["TITLE: A Generated Meridian",
             "MUSIC: opening cue, low strings"]
    ann_words = 0
    char_words = 0

    def sentence(k):
        return " ".join(rng.choice(_WORDS) for _ in range(k)) + "."

    intro = sentence(rng.randint(6, 14))
    ann_words += len(intro.split())
    lines.append(f"ANNOUNCER: {intro}")
    n_scenes = rng.randint(1, 4)
    inter_count = 0
    for n in range(1, n_scenes + 1):
        lines.append(f"SCENE {n}: setting number {n}, after dark")
        speakers_here = set()
        for _ in range(rng.randint(1, 4)):
            speaker = rng.choice(cast)
            speakers_here.add(speaker)
            text = sentence(rng.randint(3, 12))
            char_words += len(text.split())
            lines.append(f"{speaker}: {text}")
        # every cast member must speak somewhere: force stragglers into the
        # last scene
        if n == n_scenes:
            for missing in [c for c in cast if c not in _spoken(lines)]:
                text = sentence(rng.randint(3, 8))
                char_words += len(text.split())
                lines.append(f"{missing}: {text}")
        if n < n_scenes and rng.random() < 0.5:
            lines.append("MUSIC: interstitial cue, sparse")
            inter_count += 1
    outro = sentence(rng.randint(6, 14))
    ann_words += len(outro.split())
    lines.append(f"ANNOUNCER: {outro}")
    coda = "Beyond tonight's tale, a real report waits:"
    ann_words += len(coda.split())
    lines.append(f"CODA: {coda}")
    lines.append("MUSIC: closing cue, resolving")
    lines.append("END.")
    return "\n".join(lines) + "\n", cast, char_words, ann_words, \
        n_scenes, inter_count


def _spoken(lines):
    out = set()
    for ln in lines:
        head, sep, _ = ln.partition(": ")
        if sep and head.isupper() is False:
            continue
        if sep and head not in ("TITLE", "MUSIC", "CODA", "ANNOUNCER") \
                and not head.startswith("SCENE"):
            out.add(head)
    return out


class TestProperties:
    def test_generated_scripts_parse_clean_with_exact_tallies(self):
        rng = random.Random(20260710)
        for _ in range(25):
            text, cast, char_words, ann_words, n_scenes, inter = \
                _gen_script(rng)
            script, defects = parse_fable2_markup(text, cast)
            assert defects == (), render_defects(defects)
            assert script.character_word_count == char_words
            assert script.announcer_word_count == ann_words
            assert [s.n for s in script.scenes] == list(
                range(1, n_scenes + 1))
            assert len(script.music_inter) == inter
            spoken = {ln.speaker for s in script.scenes for ln in s.lines}
            assert spoken == set(cast)

    def test_random_mutations_produce_the_expected_defect(self):
        rng = random.Random(1946)
        mutations = (
            (lambda t: t.replace("END.\n", ""), D.MISSING_END),
            (lambda t: t.split("\n", 1)[1], D.MISSING_TITLE),
            (lambda t: t + "ANNOUNCER: encore.\n", D.CONTENT_AFTER_END),
            (lambda t: t.replace("SCENE 1:", "SCENE 7:", 1), D.SCENE_ORDER),
            # A terminal '.' now NORMALIZES to the pivot colon (15th live
            # smoke); the defect-worthy shape is an inner sentence break.
            (lambda t: t.replace("report waits:",
                                 "report ends. It waits:", 1),
             D.CODA_SHAPE),
        )
        for _ in range(10):
            text, cast, *_rest = _gen_script(rng)
            mutate, expected = mutations[rng.randrange(len(mutations))]
            script, defects = parse_fable2_markup(mutate(text), cast)
            assert script is None
            assert expected in [d.code for d in defects], (
                f"expected {expected} in {render_defects(defects)}")
