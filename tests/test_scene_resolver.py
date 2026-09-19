"""Extracting one act/scene out of a whole-work translation.

The corpus gate found seven vendored-translation leads that are the right
play at whole-work granularity -- a Gutenberg ebook or a Wikisource page
carrying all five acts. `_otr_scene_resolver.resolve_scene` is the step that
turns one of those into the one scene a fidelity render actually needs.
Fixtures below mirror the real shapes found in `tmp/corpus_cache/*.txt`
(French and Spanish Gutenberg editions): "ACTE PREMIER" / "SCÈNE I", "ACTO
PRIMERO" / "ESCENA PRIMERA", and the "FIN DU PREMIER ACTE." end-of-act
marker that genuinely re-matches as a second "act 1" heading. Pure, no
network. UTF-8 no BOM.
"""
from __future__ import annotations

import pytest

from nodes import _otr_scene_resolver as R
from nodes import _otr_verbatim_corpus as CORPUS


def _speeches(a: str, b: str, n: int) -> str:
    """`n` alternating exchanges between two speakers -- comfortably above
    the corpus module's own MIN_SPEAKER_LABELS/MIN_DISTINCT_SPEAKERS floors
    so a clean fixture scores full confidence.

    No trailing period on a line: `SPEAKER_LABEL_PATTERNS`' "Nom. speech"
    shape (a capitalised line ending in ". ") would otherwise match every
    line as ITS OWN one-off speaker and win the max-across-patterns vote
    with a higher, wrong, per-line-unique count -- caught by running this
    fixture through the resolver and finding `distinct_speakers` equal to
    the line count instead of 2.
    """
    lines = []
    for i in range(1, n + 1):
        lines.append("%s: line %d for %s\n" % (a, i, a))
        lines.append("%s: line %d for %s\n" % (b, i, b))
    return "".join(lines)


#: A clean three-act French play: two scenes in Act I, one in Act II, one in
#: Act III. No end-of-act marker, so heading counts stay unambiguous -- the
#: fixture for tests that are not about heading ambiguity.
FRENCH_PLAY = (
    "ACTE PREMIER\n\n"
    "SCÈNE I\n\n"
    "Un lieu désert. Tonnerre.\n\n"
    + _speeches("PREMIÈRE SORCIÈRE", "MACBETH", 4) +
    "\n"
    "SCÈNE II\n\n"
    "Un camp près de Forres.\n\n"
    + _speeches("DUNCAN", "MALCOLM", 4) +
    "\n"
    "ACTE DEUXIÈME\n\n"
    "SCÈNE I\n\n"
    "Une cour du château.\n\n"
    + _speeches("BANQUO", "FLEANCE", 4) +
    "\n"
    "ACTE TROISIÈME\n\n"
    "SCÈNE I\n\n"
    "Une salle du palais.\n\n"
    + _speeches("MACBETH", "LADY MACBETH", 4)
)

#: The real shape found in tmp/corpus_cache: an end-of-act marker that puts
#: the ordinal word BEFORE "ACTE", which is exactly the corpus grammar's
#: number-before-word heading order -- so it genuinely re-matches as a
#: second candidate "act 1" heading.
FRENCH_PLAY_WITH_END_MARKER = (
    "ACTE PREMIER\n\n"
    "SCÈNE I\n\n"
    "Un lieu désert. Tonnerre.\n\n"
    + _speeches("PREMIÈRE SORCIÈRE", "MACBETH", 4) +
    "\nFIN DU PREMIER ACTE.\n\n"
    "ACTE DEUXIÈME\n\n"
    "SCÈNE I\n\n"
    "Une cour du château.\n\n"
    + _speeches("BANQUO", "FLEANCE", 4)
)

#: A Spanish play using the exact heading shapes the real Hamlet lead uses:
#: "ACTO PRIMERO" (ordinal word) then "ESCENA PRIMERA" for scene 1, but a
#: roman numeral for scene 2. Act II is spelled "ACTO SEGUNDO" (also an
#: ordinal word) rather than the bare roman "ACTO II" -- seeing "II" would
#: make every act-1 query find it too, because a single-letter roman numeral
#: is a literal substring of every later one (see `_numberings`'s "1" -> "I"
#: entry: "i" alone matches the "I" inside "II", "III" and "IV"). That is a
#: property of the reused grammar, not a choice this fixture is dodging --
#: `test_a_query_for_scene_one_can_also_match_scene_two` exercises it
#: directly instead of letting it contaminate every other test.
SPANISH_PLAY = (
    "ACTO PRIMERO\n\n"
    "ESCENA PRIMERA\n\n"
    "Terraza delante del palacio.\n\n"
    + _speeches("HAMLET", "HORACIO", 4) +
    "\n"
    "ESCENA II\n\n"
    "Sala de audiencias en el palacio.\n\n"
    + _speeches("REY", "REINA", 4) +
    "\n"
    "ACTO SEGUNDO\n\n"
    "ESCENA PRIMERA\n\n"
    "Habitación en la casa de Polonio.\n\n"
    + _speeches("POLONIO", "REINALDO", 4)
)

#: Italian, for the ordinal-word heading test in a third language family.
#: Two acts, not one -- a single-scene fixture would BE almost entirely its
#: own "scene", which is exactly the implausibly-long-span signal and would
#: convict a minimal fixture of a fault a real five-act play never has.
ITALIAN_PLAY = (
    "ATTO PRIMO\n\n"
    "SCENA PRIMA\n\n"
    "Una brughiera.\n\n"
    + _speeches("STREGA", "MACBETH", 4) +
    "\n"
    "ATTO SECONDO\n\n"
    "SCENA PRIMA\n\n"
    "Una sala nel castello.\n\n"
    + _speeches("BANCO", "MACBETH", 6)
)

#: Real Gutenberg envelope shape, copied from tmp/corpus_cache/*.txt.
_GUTENBERG_HEADER = (
    "The Project Gutenberg eBook of Macbeth\n\n"
    "This ebook is for the use of anyone anywhere in the United States and\n"
    "most other parts of the world at no cost and with almost no restrictions\n"
    "whatsoever.\n\n"
    "*** START OF THE PROJECT GUTENBERG EBOOK MACBETH ***\n\n"
)
_GUTENBERG_FOOTER = (
    "\n\n*** END OF THE PROJECT GUTENBERG EBOOK MACBETH ***\n\n"
    "This and all associated files of various formats will be found in:\n"
    "http://www.gutenberg.org/\n"
)
GUTENBERG_ENVELOPED = _GUTENBERG_HEADER + FRENCH_PLAY + _GUTENBERG_FOOTER


# --------------------------------------------------------------------------- #
# locating by heading -- the core contract
# --------------------------------------------------------------------------- #


def test_a_scene_between_two_headings_is_extracted_whole():
    r = R.resolve_scene(FRENCH_PLAY, scene="1.2")
    assert r.confidence == 1.0
    assert r.text.startswith("SCÈNE II")
    assert "Un camp près de Forres." in r.text
    assert "MALCOLM: line 4 for MALCOLM" in r.text
    # Stops at the next heading -- neither Act II's content nor its heading.
    assert "ACTE DEUXIÈME" not in r.text
    assert "BANQUO" not in r.text


def test_the_same_scene_number_resolves_correctly_in_a_second_language():
    r = R.resolve_scene(SPANISH_PLAY, scene="2.1")
    assert r.confidence == 1.0
    assert r.text.startswith("ESCENA PRIMERA")
    assert "POLONIO: line 1 for POLONIO" in r.text
    assert "REINALDO" in r.text
    assert "HAMLET" not in r.text


def test_the_first_scene_of_an_act_stops_before_the_second_scene():
    r = R.resolve_scene(FRENCH_PLAY, scene="1.1")
    # A bare roman "I" is a literal substring of "II" (see `_numberings`), so
    # this query also candidate-matches "SCÈNE II" -- confidence is docked
    # for the ambiguity, but the located span is still the correct one: the
    # true "SCÈNE I" heading is the first, earliest match every time.
    assert r.confidence < 1.0
    assert any("candidate scene 1 headings" in reason for reason in r.reasons)
    assert r.text.startswith("SCÈNE I")
    assert "PREMIÈRE SORCIÈRE" in r.text
    assert "SCÈNE II" not in r.text
    assert "DUNCAN" not in r.text


def test_a_query_for_scene_one_can_also_match_scene_two():
    """Documents the grammar property `test_the_first_scene_of_an_act_...`
    relies on above: a single-letter roman numeral is a substring of every
    later one, so `_labelled` reports a "scene 1" candidate inside a "scene
    2" heading too. Not fixed here -- `_otr_verbatim_corpus.py` is out of
    scope -- only exercised, so a future fix to that grammar has a test that
    will visibly change instead of silently going stale."""
    r = R.resolve_scene(FRENCH_PLAY, scene="1.1")
    assert r.scene_candidates == 2


def test_the_last_scene_of_an_act_stops_at_the_next_act_heading():
    """Act II has exactly one scene, so its only boundary is Act III's own
    heading -- proves the fallback to the ACT boundary, not just the scene
    boundary."""
    r = R.resolve_scene(FRENCH_PLAY, scene="2.1")
    assert r.confidence == 1.0
    assert r.text.startswith("SCÈNE I")
    assert "BANQUO" in r.text and "FLEANCE" in r.text
    assert "ACTE TROISIÈME" not in r.text
    assert "LADY MACBETH" not in r.text


def test_the_final_scene_of_the_work_runs_to_the_end_of_text():
    r = R.resolve_scene(FRENCH_PLAY, scene="3.1")
    assert r.confidence == 1.0
    assert r.text.rstrip().endswith("LADY MACBETH: line 4 for LADY MACBETH")


# --------------------------------------------------------------------------- #
# refusing to guess
# --------------------------------------------------------------------------- #


def test_a_missing_scene_returns_empty_text_at_zero_confidence():
    r = R.resolve_scene(FRENCH_PLAY, scene="1.9")
    assert r.text == "" and r.confidence == 0.0
    assert r.start == 0 and r.end == 0
    assert any("scene 9" in reason for reason in r.reasons)


def test_a_missing_act_returns_empty_text_at_zero_confidence():
    r = R.resolve_scene(FRENCH_PLAY, scene="5.1")
    assert r.text == "" and r.confidence == 0.0
    assert any("act 5" in reason for reason in r.reasons)


def test_a_malformed_scene_argument_is_refused_not_guessed():
    for bad in ("", "1", "act one scene one", None):
        r = R.resolve_scene(FRENCH_PLAY, scene=bad)
        assert r.text == "" and r.confidence == 0.0


def test_a_missing_scene_never_falls_back_to_the_first_speeches():
    """The regression this guards: a resolver that gave up on the heading
    search and returned 'the first N speeches of the act' instead would
    still return non-empty text here. It must not."""
    r = R.resolve_scene(FRENCH_PLAY, scene="1.9")
    assert r.text == ""


# --------------------------------------------------------------------------- #
# ordinal-word headings -- the 19th-century norm the corpus module supports
# --------------------------------------------------------------------------- #


def test_an_ordinal_word_heading_is_read_in_a_third_language():
    r = R.resolve_scene(ITALIAN_PLAY, scene="1.1")
    assert r.confidence == 1.0
    assert r.text.startswith("SCENA PRIMA")
    assert "MACBETH: line 1 for MACBETH" in r.text
    assert "ATTO SECONDO" not in r.text


def test_a_roman_numeral_scene_heading_resolves_under_an_ordinal_word_act():
    """Spanish scene 1 is spelled 'PRIMERA' (ordinal word); scene 2 is the
    roman numeral 'II' -- the roman-numbered one must resolve too, and
    cleanly here because nothing later in Act I's region carries a "III"
    for its "II" to be mistaken for a piece of."""
    r = R.resolve_scene(SPANISH_PLAY, scene="1.2")
    assert r.confidence == 1.0
    assert r.text.startswith("ESCENA II")  # the leading E survives; see below
    assert "REY: line 1 for REY" in r.text
    assert "ACTO SEGUNDO" not in r.text
    assert "POLONIO" not in r.text


def test_a_shorter_heading_word_cannot_eat_a_longer_ones_prefix():
    """FIXED 2026-09-18 -- this test pinned the defect and now pins the fix.

    `_SCENE_WORDS` lists "scena" (Italian) before "escena" (Spanish), and
    "scena" is literally the last five letters of "escena". `_labelled` used
    to try words in tuple order and return on the first that matched
    anywhere, so it found "scena" INSIDE "escena" and never tried "escena"
    itself: the match started one character late and the leading "E" was
    dropped from every Spanish scene heading extracted here.

    Two changes in `_otr_verbatim_corpus._labelled` kill it -- words are
    tried LONGEST FIRST, and the EARLIEST match across all words wins rather
    than the first word that matches anywhere. Note the defect was invisible
    to `headings_present`, which only returns a boolean and was correct
    throughout; it only surfaces where the match POSITION is used, which is
    exactly what an extractor does.
    """
    r = R.resolve_scene(SPANISH_PLAY, scene="1.1")
    assert r.text.startswith("ESCENA PRIMERA")


# --------------------------------------------------------------------------- #
# Gutenberg boilerplate
# --------------------------------------------------------------------------- #


def test_gutenberg_licence_text_does_not_pollute_the_extracted_scene():
    r = R.resolve_scene(GUTENBERG_ENVELOPED, scene="2.1")
    assert r.confidence == 1.0
    assert "Gutenberg" not in r.text
    assert "gutenberg.org" not in r.text
    assert r.text.startswith("SCÈNE I")
    assert "BANQUO" in r.text


def test_offsets_are_reported_against_the_text_the_caller_passed_in():
    """The invariant a caller relies on to slice the ORIGINAL text itself,
    without knowing the resolver stripped a Gutenberg envelope first."""
    r = R.resolve_scene(GUTENBERG_ENVELOPED, scene="2.1")
    assert GUTENBERG_ENVELOPED[r.start:r.end] == r.text
    assert r.start > len(_GUTENBERG_HEADER)  # inside the stripped body


def test_offsets_are_exact_on_a_plain_non_gutenberg_source_too():
    r = R.resolve_scene(FRENCH_PLAY, scene="1.1")
    assert FRENCH_PLAY[r.start:r.end] == r.text


# --------------------------------------------------------------------------- #
# confidence must be earned
# --------------------------------------------------------------------------- #


def test_confidence_is_low_when_the_located_scene_has_too_few_speakers():
    """The heading IS found -- this is 'located but weak', not 'missing'.
    Text is returned; confidence says not to trust it without a look."""
    text = "ACTE PREMIER\n\nSCÈNE I\n\nX: Bref.\n"
    r = R.resolve_scene(text, scene="1.1")
    assert r.text != ""
    assert r.confidence < 0.5
    joined = " ".join(r.reasons)
    assert "implausibly short" in joined
    assert "speaker labels" in joined


def test_confidence_falls_for_a_monologue_even_at_full_length():
    text = "ACTE PREMIER\n\nSCÈNE I\n\n" + _speeches("MACBETH", "MACBETH", 4)
    # Both columns are the same speaker -- one distinct speaker, many labels.
    r = R.resolve_scene(text, scene="1.1")
    assert r.distinct_speakers == 1
    assert r.confidence < 1.0
    assert any("alternates" in reason for reason in r.reasons)


def test_a_clean_scene_earns_full_confidence_with_no_reasons():
    r = R.resolve_scene(FRENCH_PLAY, scene="2.1")
    assert r.confidence == 1.0
    assert r.reasons == ()


def test_a_real_end_of_act_marker_does_not_shadow_the_true_heading():
    """FIXED 2026-09-18 (PBUG-20260918-07) -- pinned the defect, now the fix.

    `_ACT_WORDS` lists the bare English "act" before "acte", and `_labelled`
    used to return on the FIRST word that matched ANYWHERE in the string
    rather than the earliest POSITION across words. "act" is a literal
    prefix of "acte", so a trailing `FIN DU PREMIER ACTE.` matched "act" via
    the number-before-word order and stole the anchor from the real
    `ACTE PREMIER` heading above it.

    Worse, and measured on the real cached Macbeth lead: the French word
    "action" contains "acti", and "act" + the bare roman "i" matched every
    one of its 24 occurrences -- ordinary prose scoring as act headings, so a
    five-act play measured as two. Three changes fix it: a Latin heading word
    now requires a SEPARATOR before its number (so "act" cannot run into
    "action"), words are tried LONGEST FIRST, and the EARLIEST match wins.
    NUMBER-then-WORD is kept only for CJK, where `第1幕` genuinely has no gap.

    The old behaviour failed CLOSED -- empty text at confidence 0.0 -- which
    was honest but useless: the whole point of this module is extracting the
    French whole-work leads, and it could not extract the one it was built
    against.
    """
    r = R.resolve_scene(FRENCH_PLAY_WITH_END_MARKER, scene="1.1")
    assert r.text.startswith("SCÈNE I")
    assert r.confidence > 0.0
    assert "PREMIÈRE SORCIÈRE" in r.text   # the real dialogue came with it
    assert r.act_candidates == 1           # the real heading, not the marker

    # KNOWN GAP, deliberately asserted rather than hidden: the span stops at
    # the next HEADING, and `FIN DU PREMIER ACTE.` is a marker, so it rides
    # along inside the extracted text. Harmless until this module is wired --
    # then it is chrome that would be SPOKEN, which is exactly the defect
    # PBUG-20260918-04 shipped once already. Trim it when the gate calls
    # resolve_scene, and delete this assertion in the same change.
    assert "FIN DU PREMIER ACTE" in r.text


def test_speaker_label_stats_on_the_resolver_match_the_corpus_module_directly():
    """The resolver must not invent its own speaker-counting rule -- it is
    required to reuse `speaker_label_stats` from the corpus module."""
    r = R.resolve_scene(FRENCH_PLAY, scene="1.1")
    labels, distinct = CORPUS.speaker_label_stats(r.text)
    assert (r.speaker_labels, r.distinct_speakers) == (labels, distinct)


# --------------------------------------------------------------------------- #
# strip_gutenberg_envelope in isolation
# --------------------------------------------------------------------------- #


def test_strip_gutenberg_envelope_removes_header_and_footer():
    body, offset = R.strip_gutenberg_envelope(GUTENBERG_ENVELOPED)
    assert body.strip() == FRENCH_PLAY.strip()
    # `offset` lands right after the "***" that closes the start marker --
    # not necessarily after the blank line following it in the fixture, so
    # the invariant is where it sits relative to the marker, not an exact
    # header length.
    assert GUTENBERG_ENVELOPED[:offset].rstrip().endswith("***")
    assert offset <= len(_GUTENBERG_HEADER)


def test_strip_gutenberg_envelope_is_a_no_op_on_a_wikisource_style_page():
    body, offset = R.strip_gutenberg_envelope(SPANISH_PLAY)
    assert body == SPANISH_PLAY and offset == 0


def test_strip_gutenberg_envelope_handles_a_missing_end_marker():
    text = _GUTENBERG_HEADER + FRENCH_PLAY  # no footer at all
    body, offset = R.strip_gutenberg_envelope(text)
    assert body.strip() == FRENCH_PLAY.strip()
    assert text[:offset].rstrip().endswith("***")
