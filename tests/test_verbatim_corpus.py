"""The vendored-translation corpus: legal tests, gate verdicts, manifest.

Operator 2026-09-18 EVENING: "I don't want to waste anything in rights I'm not
publishing these commercially." RIGHTS REFUSE NOTHING -- the publication years
land in `LeadReport.notes`, which no verdict reads. What still refuses is
FIDELITY (a translation made from an intermediary) and a source whose text does
not exist, both recorded by hand in `excluded`.

The morning's rule -- clear the US test AND life+70 -- was withdrawn the same
day; tests that pinned it now pin its absence, and say so in their docstrings.
CPU only, no network. UTF-8 no BOM.
"""
from __future__ import annotations

import json
import re
import os

import pytest

from nodes import _otr_verbatim_corpus as C


# --------------------------------------------------------------------------- #
# the legal tests
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("published, died", [
    (1865, 1873),      # Hugo
    ("1865-1872", 1873),
    (1858, 1889),      # Rusconi
    (1912, 1912),      # Menendez y Pelayo
    (1889, 1889),      # Luis I
    (1917, 1937),      # Sitaram
    (1910, 1935),      # Tsubouchi
])
def test_the_shipped_translators_clear_both_tests(published, died):
    assert _clears(published, died)
    assert C.publication_reasons(published, died) == []


@pytest.mark.parametrize("published, died, why", [
    (1947, 1944, "first published 1947"),        # Zhu: US test fails
    (1900, 1962, "translator died 1962"),        # Rangeya Raghav: life+70 fails
    (1990, 2003, "first published 1990"),        # Bachchan: both fail
])
def test_a_row_that_fails_either_test_says_which(published, died, why):
    """It is REPORTED, not refused -- see the module docstring. The wording is
    still pinned because it is what a reader is shown."""
    assert not _clears(published, died)
    assert any(why in r for r in C.publication_reasons(published, died))


def _clears(first_published, translator_died):
    """Did this row clear both publication tests?

    `clears_publication_anywhere` was DELETED 2026-09-18 -- a boolean is only
    useful for refusing, and rights refuse nothing now. The year-parsing rules
    it encoded are still worth pinning (a span reads its LATEST year, a death
    reads its FIRST), so these tests ask `publication_reasons` instead: no
    reasons means it cleared.
    """
    return not C.publication_reasons(first_published, translator_died)

def test_zhu_shenghao_is_the_case_the_old_rule_got_wrong():
    """Kept as a YEAR-ARITHMETIC test, not a policy one.

    d.1944 clears life+70 while the collected plays were published in 1947, so
    the withdrawn rule reported a failure here. It is worth remembering why
    that rule was wrong twice over: rights refuse nothing now, AND the
    arithmetic was never right for a foreign work -- zh.wikisource records that
    the translation was already public domain in China on the URAA date of
    1996-01-01 and never previously published in the US, so 1947 never blocked
    it in the first place. A false BLOCKED is the expensive direction: it
    discards a good source in silence."""
    assert _clears(1947, 1944) is False
    assert _clears(1930, 1944) is True


@pytest.mark.parametrize("value", ["", None, "unknown", "n/a", True, 0])
def test_an_unrecorded_year_never_clears(value):
    assert not _clears(value, 1873)
    assert not _clears(1865, value)


def test_the_death_reader_takes_the_first_year():
    assert C._year("1865-1872") == 1865
    assert C._year("c. 1889") == 1889
    assert C._year(1889) == 1889
    assert C._year("not a year") is None


# --------------------------------------------------------------------------- #
# the gate
# --------------------------------------------------------------------------- #


def _clean_report(**over):
    base = dict(
        iso="fr", play="macbeth", scene="1.3",
        http_status=200, byte_length=48000, encoding="utf-8",
        headings_present=True, speaker_labels=24, distinct_speakers=4,
        dialogue_ratio=0.92,
        licence="CC BY-SA 4.0", revision_id="1234567",
        translator="Hugo", translator_died=1873, first_published=1865,
    )
    base.update(over)
    return C.LeadReport(**base)


def test_a_clean_lead_is_ready_with_no_reasons():
    r = C.assess(_clean_report())
    assert r.verdict == C.READY and r.reasons == []


@pytest.mark.parametrize("over, fragment", [
    ({"headings_present": False}, "headings not found"),
    ({"speaker_labels": 2, "distinct_speakers": 1}, "speaker labels"),
    ({"dialogue_ratio": 0.10}, "dialogue-to-markup"),
    ({"revision_id": ""}, "revision id"),
    ({"pending_markers": ["a transcribir"]}, "transcription pending"),
    ({"http_status": 404, "byte_length": 0, "speaker_labels": 0,
      "distinct_speakers": 0}, "HTTP 404"),
])
def test_each_measured_fault_lands_in_the_reasons(over, fragment):
    r = C.assess(_clean_report(**over))
    assert r.verdict != C.READY
    assert any(fragment in reason for reason in r.reasons), r.reasons


def test_a_rights_failing_page_with_good_text_is_READY():
    """WITHDRAWN 2026-09-18: rights used to force BLOCKED. Operator: "I don't
    want to waste anything in rights I'm not publishing these commercially."

    ASSERTING READY IS THE POINT, and the first attempt at this change did not.
    Rights reasons were appended to `reasons`, the verdict is READY only when
    `reasons` is empty, and `select_scene` takes READY only -- so every
    rights-failing lead was pinned at PARTIAL and could never be selected. The
    refusal had moved, not gone, while the docstring claimed otherwise. Caught
    by the Sonnet post-QA; a `!= BLOCKED` assertion would not have caught it,
    which is why this one is `== READY`.
    """
    r = C.assess(_clean_report(translator_died=1962, first_published=1955))
    assert r.verdict == C.READY, (r.verdict, r.reasons)
    assert r.reasons == []
    # Said out loud, just never counted.
    assert any("1931" in note or "1956" in note for note in r.notes), r.notes


def test_a_missing_licence_is_a_note_not_a_refusal():
    r = C.assess(_clean_report(licence=""))
    assert r.verdict == C.READY
    assert any("transcription_license" in note for note in r.notes)


def test_notes_never_change_the_verdict():
    """The invariant behind both tests above: whatever lands in `notes`, the
    verdict is whatever the TEXT earned."""
    clean = C.assess(_clean_report())
    noted = C.assess(_clean_report(translator_died=1999, first_published=1990,
                                   licence=""))
    assert clean.verdict == noted.verdict == C.READY
    assert noted.notes and not clean.notes


def test_only_an_explicit_exclusion_can_block():
    """BLOCKED is now reserved for a human's recorded dead end -- a fidelity
    failure or a source that does not exist -- never an arithmetic result."""
    assert C.assess(_clean_report(excluded="translated from Schiller's German")
                    ).verdict == C.BLOCKED
    assert C.assess(_clean_report(translator_died=1999, first_published=1990)
                    ).verdict != C.BLOCKED


def test_an_empty_page_is_empty_and_a_thin_one_is_partial():
    assert C.assess(_clean_report(byte_length=0, speaker_labels=0,
                                  distinct_speakers=0,
                                  headings_present=False)).verdict == C.EMPTY
    assert C.assess(_clean_report(speaker_labels=3, distinct_speakers=2)).verdict == C.PARTIAL


def test_the_pending_markers_the_spec_names_are_all_detected():
    for marker in ("A transcribir", "作業中", "Not Proofread"):
        assert C.find_pending_markers("... %s ..." % marker)


def test_dialogue_ratio_separates_scaffold_from_text():
    scene = "MACBETH: Quand nous reverrons-nous toutes les trois ?\n" * 20
    scaffold = "{{Index|page=1}}[[Category:x]]{{|}}<noinclude>" * 20
    assert C.dialogue_ratio(scene) > C.MIN_DIALOGUE_RATIO
    assert C.dialogue_ratio(scaffold) < C.MIN_DIALOGUE_RATIO
    assert C.dialogue_ratio("") == 0.0


@pytest.mark.parametrize("text, want", [
    ("MACBETH: un\nBANQUO: deux\nMACBETH: trois\n", 3),
    ("Macb. un\nBanq. deux\n", 2),
    ("आलिवर--एक\nरोज़लिंड--दो\n", 2),
    ("サンプソン　はい\nグレゴリー　いいえ\n", 2),
    ("船長：老大\n水手長：在\n", 2),
])
def test_every_vendored_label_shape_counts(text, want):
    assert C.count_speaker_labels(text) == want


def test_label_shapes_are_maxed_not_summed():
    """A page uses ONE convention; summing would let two weak partial
    matches look like a performable scene."""
    mixed = "MACBETH: un\nBanq. deux\n"
    assert C.count_speaker_labels(mixed) == 1


@pytest.mark.parametrize("text, scene, want", [
    ("ACTE I, SCENE 3. Une bruyere.", "1.3", True),
    ("ATTO I -- SCENA 3", "1.3", True),
    ("ACT 1 SCENE 2", "1.3", False),
    ("no headings here", "1.3", False),
])
def test_headings_presence_is_plausibility_not_alignment(text, scene, want):
    assert C.headings_present(text, scene) is want


def test_tracking_parameters_never_reach_the_manifest():
    assert C.strip_tracking("https://x/y?utm_source=gemini") == "https://x/y"
    assert C.strip_tracking("https://x/y?a=1&utm_medium=x") == "https://x/y?a=1"
    assert C.strip_tracking("https://x/y") == "https://x/y"


# --------------------------------------------------------------------------- #
# the manifest
# --------------------------------------------------------------------------- #


def _manifest_row(**over):
    row = {
        "iso": "fr", "play": "macbeth", "scene": "1.3",
        "file": "macbeth_1.3.txt", "translator": "François-Victor Hugo",
        "translator_death_date": 1873, "translation_first_published": 1865,
        "transcription_license": "CC BY-SA 4.0",
        "source_url": "https://fr.wikisource.org/wiki/Macbeth",
        "revision_id": "1234567", "raw_sha256": "a" * 64,
        "verdict": C.READY, "alignment_confidence": 0.95,
    }
    row.update(over)
    return row


def _write(tmp_path, rows, version=C.SCHEMA_VERSION):
    path = tmp_path / C.MANIFEST_NAME
    path.write_text(json.dumps({"schema_version": version, "scenes": rows}),
                    encoding="utf-8")
    return str(path)


def test_a_missing_manifest_is_normal_and_means_author_with_the_model(tmp_path):
    assert C.load_manifest(str(tmp_path / "nope.json")) == []


def test_a_valid_manifest_round_trips(tmp_path):
    rows = C.load_manifest(_write(tmp_path, [_manifest_row()]))
    assert len(rows) == 1 and rows[0]["translator"].startswith("François")


@pytest.mark.parametrize("over", [
    {"translator_death_date": 1962},
    {"translation_first_published": 1947},
])
def test_a_manifest_row_is_never_refused_on_rights(tmp_path, over):
    """WITHDRAWN 2026-09-18, and this one was the dangerous half.

    `load_manifest` used to raise `CorpusError` here, and `CorpusError` is
    documented as never degraded around -- so once the writer's plan step reads
    this manifest, a copyright-date arithmetic result could KILL A RENDER. The
    standing rule is that authoring-time tools fail loud while the RENDER PATH
    degrades to the best available result with an honest receipt, and a
    copyright question is the textbook authoring-time concern. Operator, same
    day: rights are the installing user's to settle.

    The years are still REQUIRED fields -- the credit roll names them. They
    just decide nothing.
    """
    rows = C.load_manifest(_write(tmp_path, [_manifest_row(**over)]))
    assert len(rows) == 1
    for key in ("translator", "translator_death_date",
                "translation_first_published"):
        assert key in rows[0], "%s must survive as data" % key


def test_a_missing_field_is_refused_and_names_it(tmp_path):
    row = _manifest_row()
    del row["raw_sha256"]
    with pytest.raises(C.CorpusError) as caught:
        C.load_manifest(_write(tmp_path, [row]))
    assert "raw_sha256" in str(caught.value)


def test_a_wrong_schema_version_is_refused(tmp_path):
    with pytest.raises(C.CorpusError):
        C.load_manifest(_write(tmp_path, [_manifest_row()], version="v0"))


def test_an_unreadable_manifest_is_refused_never_degraded(tmp_path):
    path = tmp_path / C.MANIFEST_NAME
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(C.CorpusError):
        C.load_manifest(str(path))


# --------------------------------------------------------------------------- #
# selection -- the seam the writer's plan step reads
# --------------------------------------------------------------------------- #


def test_a_ready_confident_scene_is_selected():
    rows = [_manifest_row()]
    got = C.select_scene(rows, iso="fr", play="macbeth", scene="1.3")
    assert got and got["file"] == "macbeth_1.3.txt"


@pytest.mark.parametrize("over", [
    {"verdict": C.PARTIAL}, {"verdict": C.EMPTY}, {"verdict": C.BLOCKED},
    {"alignment_confidence": 0.5}, {"alignment_confidence": "nan-ish"},
])
def test_anything_short_of_ready_and_confident_falls_back_to_the_model(over):
    assert C.select_scene([_manifest_row(**over)], iso="fr", play="macbeth",
                          scene="1.3") is None


def test_another_language_or_scene_is_never_borrowed():
    rows = [_manifest_row()]
    assert C.select_scene(rows, iso="es", play="macbeth", scene="1.3") is None
    assert C.select_scene(rows, iso="fr", play="hamlet", scene="1.3") is None
    assert C.select_scene(rows, iso="fr", play="macbeth", scene="1.2") is None


def test_selection_is_case_and_space_tolerant_on_identity_fields():
    rows = [_manifest_row(iso="FR", play="Macbeth")]
    assert C.select_scene(rows, iso="fr", play=" macbeth ", scene="1.3")



# --------------------------------------------------------------------------- #
# defects found in review, 2026-09-18 -- each was a measured false READY
# --------------------------------------------------------------------------- #


def test_a_publication_span_reads_its_LATEST_year():
    """A collected edition does not say which volume held this play, so the
    conservative reading is the last year. Taking the first false-cleared a
    span straddling the cutoff."""
    assert C.publication_year("1865-1872") == 1872
    assert C.publication_year("1929-1932") == 1932
    assert not _clears("1929-1932", 1889)
    assert _clears("1865-1872", 1873)
    # A death is a single event; that reader still takes the first year.
    assert C._year("1935-02-28") == 1935


def test_tsubouchi_blocks_on_either_spelling_of_the_revision():
    assert not _clears(1933, 1935)
    assert not _clears("1933-1935", 1935)
    # The earlier first publication would clear, which is why the lead says
    # to confirm WHICH text before vendoring.
    assert _clears(1910, 1935)


def test_a_dramatis_personae_is_not_a_performable_scene():
    """Six names, six labels, none repeated -- this assessed READY before
    the alternation rule."""
    page = ("ACTE I, SCENE 3.\n"
            "ANTIPHOLUS: frere jumeau\nDROMIO: valet\nADRIANA: epouse\n"
            "LUCIANA: soeur\nBALTHASAR: marchand\nANGELO: orfevre\n")
    labels, distinct = C.speaker_label_stats(page)
    assert labels >= C.MIN_SPEAKER_LABELS and distinct == labels
    r = C.assess(_clean_report(speaker_labels=labels, distinct_speakers=distinct))
    assert r.verdict != C.READY
    assert any("list rather than a scene" in x for x in r.reasons)


def test_a_real_alternating_scene_still_passes():
    page = "".join("ANTIPHOLUS: une replique.\nBALTHASAR: une reponse.\n"
                   for _ in range(8))
    labels, distinct = C.speaker_label_stats(page)
    assert labels == 16 and distinct == 2
    assert C.assess(_clean_report(speaker_labels=labels,
                                  distinct_speakers=distinct)).verdict == C.READY


def test_a_monologue_is_not_a_scene():
    r = C.assess(_clean_report(speaker_labels=9, distinct_speakers=1))
    assert r.verdict != C.READY
    assert any("alternates" in x for x in r.reasons)


def test_markup_and_code_prefixes_are_never_speakers():
    page = ('<link href="https://x/y">\n"jsonKey": 1\n"other": 2\n'
            '<meta name="z">\nhttps://example.com: see\n')
    labels, distinct = C.speaker_label_stats(page)
    assert labels == 0, (labels, distinct)


def test_ordinary_prose_with_an_ideographic_space_is_not_a_scene():
    """Measured in review: the CJK shape counted 7 speeches on plain prose."""
    page = "今日は\u3000いい天気です。\n" * 7
    labels, distinct = C.speaker_label_stats(page)
    assert distinct < C.MIN_DISTINCT_SPEAKERS or labels < C.MIN_SPEAKER_LABELS \
        or C.assess(_clean_report(speaker_labels=labels,
                                  distinct_speakers=distinct)).verdict != C.READY


@pytest.mark.parametrize("text, scene, want", [
    ("ACTE I, SCENE 3. Une bruyere.", "1.3", True),
    ("ATTO I -- SCENA 3", "1.3", True),
    ("acto 1, escena 3", "1.3", True),
    ("第1幕第3場", "1.3", True),
    # The defects: a bare roman `i` is the English pronoun, and two loose
    # digits anywhere on the page proved nothing.
    ("I have 1 note. There are 1 witches.", "1.1", False),
    ("there are 2 witches and 2 lords", "2.2", False),
    ("ACT 1 elsewhere in a paragraph of prose that runs on for a while "
     "before it ever mentions SCENE 3", "1.3", False),
    ("ACT 1 SCENE 2", "1.3", False),
])
def test_headings_need_a_labelled_act_and_scene_together(text, scene, want):
    assert C.headings_present(text, scene) is want


def test_a_manifest_field_that_is_present_but_empty_is_refused(tmp_path):
    for key in ("file", "transcription_license", "revision_id", "raw_sha256"):
        with pytest.raises(C.CorpusError) as caught:
            C.load_manifest(_write(tmp_path, [_manifest_row(**{key: ""})]))
        assert key in str(caught.value)


@pytest.mark.parametrize("value", ["NaN", float("nan"), 1.5, -0.1, "high"])
def test_a_confidence_that_is_not_a_real_fraction_is_refused(tmp_path, value):
    with pytest.raises(C.CorpusError):
        C.load_manifest(_write(tmp_path, [_manifest_row(alignment_confidence=value)]))


def test_a_nan_confidence_never_selects():
    """`NaN < 0.8` is False, so a NaN row walked through the floor."""
    assert C.select_scene([_manifest_row(alignment_confidence=float("nan"))],
                          iso="fr", play="macbeth", scene="1.3") is None


# --------------------------------------------------------------------------- #
# post-QA findings, 2026-09-18
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("text, scene", [
    ("ACTE PREMIER, SCENE III.", "1.3"),          # the 19th-c French norm
    ("ATTO PRIMO, SCENA III", "1.3"),
    ("acto primero, escena tercera", "1.3"),
    ("ACTO PRIMEIRO, SCENA I", "1.1"),
    ("第一幕第三場", "1.3"),
    ("ACTE DEUXIÈME, SCÈNE DEUXIÈME", "2.2"),
])
def test_an_ordinal_word_heading_is_read(text, scene):
    """Matching only digits and single-letter romans made a real scene
    unreachable AND made 'headings not found' mean two different things."""
    assert C.headings_present(text, scene) is True


# --------------------------------------------------------------------------- #
# PBUG-20260918-07: the MATCH POSITION, not merely the match
#
# These three fault patterns used to be pinned in `tests/test_scene_resolver.py`
# THROUGH `resolve_scene`, and that file was deleted on 2026-09-19 with the
# module. The faults were never in the resolver: they are in `_labelled`, which
# is still live -- `headings_present` slices `raw[act_hit.start():act_hit.end()
# + _HEADING_WINDOW]`, so a match that lands one character late silently moves
# the window, and `scripts/otr_shakespeare_corpus_gate.py` calls it on every
# fetched lead. Deleting the extractor removed the only coverage of that.
#
# So they are ported to the LIVE function, which is a better test than the one
# that was lost: it asserts the behaviour at the site production depends on,
# instead of through a consumer that no longer exists.
#
# WHAT KEEPS THESE PASSING is the ordering logic inside `_labelled` -- words
# tried LONGEST FIRST, and the EARLIEST match across all words winning. The
# word tuples are still declared in the hazardous order (`act` before `acte`,
# `scena` before `escena`), so anything that "simplifies" that logic back to
# first-word-that-matches-anywhere brings all three defects back at once.
# --------------------------------------------------------------------------- #


def test_a_shorter_heading_word_cannot_eat_a_longer_ones_prefix():
    """`scena` is literally the last five letters of `escena`, and it is listed
    first. Matching the short word INSIDE the long one starts the match one
    character late and drops the leading E from every Spanish scene heading.

    Invisible to a boolean: `headings_present` was correct throughout while the
    position was wrong, which is why this asserts `.start()` and not truth.
    """
    raw = C._fold("ESCENA PRIMERA")
    hit = C._labelled(raw, C._SCENE_WORDS, "1")
    assert hit is not None
    assert hit.start() == 0, "matched inside 'escena', losing its leading E"


def test_prose_containing_a_heading_word_does_not_score_as_a_heading():
    """The French `action` contains `acti`, and `act` + the bare roman `i`
    matched all 24 occurrences on the real cached Macbeth lead -- ordinary
    prose scoring as act headings, so a five-act play measured as two. A Latin
    heading word now requires a separator before its number."""
    prose = "l'action se passe en France. " * 4
    assert C._labelled(C._fold(prose), C._ACT_WORDS, "1") is None
    assert C.act_headings_found(prose) == 0


def test_a_trailing_end_of_act_marker_does_not_steal_the_anchor():
    """`act` is a literal prefix of `acte`, so a closing `FIN DU PREMIER ACTE.`
    used to match before the real `ACTE PREMIER` heading above it and drag the
    heading window to the bottom of the page."""
    page = ("ACTE PREMIER\n\nSCÈNE I\n\n"
            "PREMIÈRE SORCIÈRE: Quand nous reverrons-nous ?\n\n"
            "FIN DU PREMIER ACTE.")
    hit = C._labelled(C._fold(page), C._ACT_WORDS, "1")
    assert hit is not None
    assert hit.start() == 0, "anchored on the closing marker, not the heading"
    # and the window built from that position still finds the scene
    assert C.headings_present(page, "1.1") is True


def test_a_whole_work_is_named_as_such_not_as_heading_less():
    """Seven leads are the right play at book granularity; calling them
    heading-less sent a reader looking for the wrong fault."""
    book = ("ACTO PRIMERO\n...\nACTO II\n...\nACTO III\n...\n"
            "ACTO IV\n...\nACTO V\n...\n")
    assert C.act_headings_found(book) == 5
    r = C.assess(_clean_report(headings_present=False, act_headings=5))
    assert any("whole work" in x and "resolve this lead" in x for x in r.reasons)
    assert not any("headings not found" in x for x in r.reasons)


def test_a_page_with_no_act_heading_still_says_headings_not_found():
    r = C.assess(_clean_report(headings_present=False, act_headings=0))
    assert any("headings not found" in x for x in r.reasons)


# --------------------------------------------------------------------------- #
# the vendored scenes, read back through the helper production calls
# --------------------------------------------------------------------------- #
#: The four vendored scenes, keyed by the source_ref THE SHIPPING BANK EMITS.
#: These strings are copied from `curated_scenes.sample.json` and `banks.json`,
#: and that is the whole point: the first cut of these tests asked with
#: `macbeth__act1_scene3`, a shape the corpus accepted and production never
#: sends. Every test passed against a lookup that could not fire on a real
#: render. A fixture invented to match the code proves the code matches the
#: fixture.
VENDORED = [
    ("it", "folger-macbeth:act1-scene3-witches", "Carlo Rusconi"),
    ("es", "folger-as-you-like-it:act3-scene2-rosalind-orlando",
     "Jos\u00e9 Arnaldo M\u00e1rquez"),
    ("fr", "folger-hamlet:act1-scene1-platform-watch",
     "Fran\u00e7ois-Victor Hugo"),
    ("fr", "folger-lear:act1-scene1-love-test", "Fran\u00e7ois-Victor Hugo"),
]


def _corpus_root():
    from pathlib import Path
    return str(Path(__file__).resolve().parent.parent / "config"
               / "source_banks" / "shakespeare" / "translations")


@pytest.mark.parametrize("iso, source_ref, translator", VENDORED)
def test_a_vendored_scene_resolves_and_its_bytes_verify(iso, source_ref,
                                                        translator):
    """The lookup a render performs, against the real files on disk.

    `vendored_text` re-hashes the file and refuses a mismatch, so this also
    proves the manifest signs the bytes that are STORED -- the first cut wrote
    `text + "\n"` and signed `text`, and every scene was refused.
    """
    text, row = C.vendored_text(_corpus_root(), iso, source_ref)
    assert text, "the scene resolved to no text"
    assert row is not None and row["iso"] == iso
    assert row["translator"].startswith(translator.split()[0])


def test_the_refs_the_bank_actually_emits_are_the_ones_that_resolve():
    """Guard the shape, not just the outcome.

    Every shipping ref is `folger-<play>:act<N>-scene<M>-<slug>`, and two plays
    are named differently on either side -- the bank says `lear` and
    `comedy-errors` where the corpus says `king_lear` and `comedy_of_errors`.
    A silent miss there is a vendored scene that never loads and never says why.
    """
    assert C._scene_key_from_ref(
        "folger-macbeth:act1-scene3-witches") == ("macbeth", "1.3")
    assert C._scene_key_from_ref(
        "folger-lear:act1-scene1-love-test") == ("king_lear", "1.1")
    assert C._scene_key_from_ref(
        "folger-comedy-errors:act3-scene1-locked-door") == ("comedy_of_errors",
                                                            "3.1")
    # the corpus's own short form still parses -- the vendoring script uses it
    assert C._scene_key_from_ref("macbeth__act1_scene3") == ("macbeth", "1.3")
    # and an unparseable ref is a miss, never a guess
    assert C._scene_key_from_ref("folger-macbeth") == ("", "")
    assert C._scene_key_from_ref("") == ("", "")


def test_a_language_with_no_vendored_scene_is_a_quiet_miss():
    """Ninety-odd of the cells have no vendored scene and never will today.
    A miss returns empty and the model translation ships -- it is the NORMAL
    answer, not a failure, and must never raise."""
    assert C.vendored_text(_corpus_root(), "ja", "folger-macbeth:act1-scene3-witches") == ("", None)
    assert C.vendored_text(_corpus_root(), "it", "folger-hamlet:act9-scene9-nothing") == ("", None)


# --------------------------------------------------------------------------- #
# speaker_map (2026-09-18): the edition's label -> spoken name + English roster
# --------------------------------------------------------------------------- #


def test_speaker_map_is_optional_and_absent_means_no_bindings(tmp_path):
    """A REQUIRED key would refuse every row vendored before it existed."""
    assert C.SPEAKER_MAP_FIELD not in C.REQUIRED_MANIFEST_FIELDS
    rows = C.load_manifest(_write(tmp_path, [_manifest_row()]))
    assert C.speaker_bindings(rows[0]) == {}
    assert C.speaker_bindings(None) == {}


def test_a_present_speaker_map_round_trips_as_plain_dicts(tmp_path):
    row = _manifest_row(speaker_map={
        " 1A  STREGA ": {"spoken": "PRIMA STREGA", "roster": "FIRST WITCH"}})
    rows = C.load_manifest(_write(tmp_path, [row]))
    assert C.speaker_bindings(rows[0]) == {
        "1A STREGA": {"spoken": "PRIMA STREGA", "roster": "FIRST WITCH"}}


@pytest.mark.parametrize("bad", [
    "not an object",
    {"": {"spoken": "X", "roster": "Y"}},
    {"1A STREGA": "PRIMA STREGA"},
    {"1A STREGA": {"spoken": "PRIMA STREGA"}},
    {"1A STREGA": {"spoken": "", "roster": "FIRST WITCH"}},
])
def test_a_malformed_speaker_map_is_refused_and_names_the_field(tmp_path, bad):
    """Present but half-written would bind some of a scene's people and leave
    the rest to the roll in silence; the manifest is never degraded around."""
    with pytest.raises(C.CorpusError) as caught:
        C.load_manifest(_write(tmp_path, [_manifest_row(speaker_map=bad)]))
    assert C.SPEAKER_MAP_FIELD in str(caught.value)


#: corpus play key -> the English Folger source stem whose sidecar is the roster
# KEYED BY (play, scene), NOT BY PLAY. A play now ships more than one scene --
# Midsummer 3.1 and 3.2, Much Ado 2.3 and 3.1, Tempest 1.2 and 3.1, Twelfth
# Night 1.5 and 2.5 -- so a per-play key silently picked one sidecar for both
# and would have checked the wrong cast.
_ENGLISH_STEM = {
    ("macbeth", "1.3"): "macbeth__act1_scene3",
    ("as_you_like_it", "3.2"): "as_you_like_it__act3_scene2",
    ("hamlet", "1.1"): "hamlet__act1_scene1",
    ("king_lear", "1.1"): "king_lear__act1_scene1",
    ("comedy_of_errors", "3.1"): "comedy_errors__act3_scene1",
    ("midsummer", "3.1"): "midsummer__act3_scene1",
    ("midsummer", "3.2"): "midsummer__act3_scene2",
    ("much_ado", "2.3"): "much_ado__act2_scene3",
    ("much_ado", "3.1"): "much_ado__act3_scene1",
    ("romeo_juliet", "2.2"): "romeo_juliet__act2_scene2",
    ("tempest", "1.2"): "tempest__act1_scene2",
    ("tempest", "3.1"): "tempest__act3_scene1",
    ("twelfth_night", "1.5"): "twelfth_night__act1_scene5",
    ("twelfth_night", "2.5"): "twelfth_night__act2_scene5",
}

#: Labels the edition prints that are NOT characters, and are knowingly left
#: unbound. An unbound label costs a VOICE, never the dialogue: the speech is
#: still that speaker's own words under their own name, the gender ladder just
#: rolls instead of resolving. These are the residue of two conventions the
#: extractor cannot fully separate, each measured and named rather than papered
#: over:
#:   * Hugo's transcriber marks the LETTER Malvolio reads in the speaker class,
#:     so fragments of the letter survive as one-line labels.
#:   * A bare `ANTIPHOLUS` / `DROMIO` is genuinely ambiguous -- the twins are two
#:     people and binding a bare label to either would be a guess. Abstaining is
#:     correct; see `resolve_roster_gender`, which abstains on the same case.
#: A TRANSLATOR'S OWN CHOICE IS NOT A DEFECT, and the entries below are the
#: first that are there for that reason rather than for a parsing gap.
#: Operator 2026-09-19: *"maybe some of these foreign translators decide to
#: create a new act or a new speech, and it's part of their local vernacular and
#: history and culture. Who am I to judge"*. Right -- and the mechanism above
#: already honours it, because an unbound label costs a VOICE and never the
#: dialogue. These characters speak their own lines under their own names and
#: draw from the roll, which is the correct outcome; refusing the scene would
#: have thrown away the translation to protect an English cast list.
_KNOWN_UNBOUND = {
    "IMBÉCILE DE CHEVALIER",
    "JE PUIS COMMANDER OÙ J’ADORE",
    "NUL HOMME NE LE DOIT SAVOIR",
    "QUI JAMAIS NE SE FATIGUE",
    "TRÈS-BEAU PYRAME",
    "ANTIPHOLUS", "DROMIO", "ANTÍFOLO",
    "AMIPHOLUS D’ÉPHÈSE",
    "i buen señor Angelo, es necesario que nos excuséis á todos",
    # Snug speaks in Rusconi's and Zhu Shenghao's Midsummer 3.1 and never
    # speaks in the Folger sidecar, where he appears only in the entry and
    # exit directions. Two translators, independently, gave him lines.
    "SNUG", "史",
    # Rusconi labels the ROLE where Folger labels the ACTOR: inside the
    # play-within-the-play these are Pyramus and Thisbe, not Bottom and Flute.
    # Binding a role to its actor would merge two speakers into one voice.
    "PIR", "TIS",
    # `FAT` WAS HERE AND IT WAS A PARSING BUG WEARING A TRANSLATOR'S CLOTHES.
    # The entry claimed the English scene names its fairies individually so no
    # single counterpart existed. The truth was that `_RUSCONI_HEAD` only
    # matched a `<sup>` ordinal, and this page writes a bare `1ª`, so all FOUR
    # fairies collapsed into one label -- eleven lines in one mouth, including
    # three separate greetings and the fairies naming themselves in turn. The
    # Chinese translation of the same scene bound all four individually on the
    # same day, which is what exposed it. They are now `1ª FAT` through
    # `4ª FAT` and bound by the names they speak: Cobweb, Peaseblossom,
    # Mustardseed, Moth.
    #
    # THE LESSON IS THE SHAPE OF THE MISTAKE. This list is for gaps somebody
    # CHOSE, and an entry written to excuse a defect looks identical to one
    # written to honour a translator. Before adding a label here, check whether
    # another edition of the same scene has more speakers than yours.
}


def test_every_shipped_map_binds_to_a_name_the_english_sidecar_carries():
    """The bridge is only worth having if the far end exists: each `roster`
    must be a `name` in the English sidecar (the ladder's exact tier) or a
    collective marker the selector refuses. Read off the real files."""
    from pathlib import Path
    from nodes import _otr_passage_selector as PS
    from nodes import _otr_roster_gender as RG
    root = Path(_corpus_root())
    sources = root.parent / "sources"
    rows = C.load_manifest(str(root / C.MANIFEST_NAME))
    assert len(rows) >= 4, "the corpus lost scenes"
    for row in rows:
        bindings = C.speaker_bindings(row)
        assert bindings, "%s/%s %s carries no speaker_map" % (
            row["iso"], row["play"], row["scene"])
        stem = _ENGLISH_STEM[(row["play"], row["scene"])]
        roster = {r["name"] for r in RG.load_roster_characters(
            sources / (stem + ".txt"))}
        assert roster, row["play"]
        for label, spec in bindings.items():
            assert spec["roster"] in roster or spec["roster"] in PS._COLLECTIVE_SPEAKERS, (
                row["play"], label, spec)
            # the spoken name is a cast-row name: upper-case, like a Folger prefix
            assert spec["spoken"] == spec["spoken"].upper(), spec


def test_every_label_in_every_vendored_text_is_bound():
    """Measured, not asserted: parse each stored scene and check the map names
    every label the edition writes. An unbound label is a voice that falls to
    the roll -- allowed by design, and each one still permitted is NAMED in
    `_KNOWN_UNBOUND` with the reason it cannot be bound honestly.

    Operator 2026-09-19: *"as long as it's reading the right dialogue is most
    important"*. An unbound label costs a VOICE and never the dialogue -- the
    character speaks their own lines under their own name -- so the bar here is
    that every gap is one somebody CHOSE, not one nobody noticed.
    """
    from pathlib import Path
    from nodes import _otr_passage_selector as PS
    root = Path(_corpus_root())
    for row in C.load_manifest(str(root / C.MANIFEST_NAME)):
        text = (root / row["file"]).read_text(encoding="utf-8")
        labels = {s.speaker for s in PS.parse_speeches(text)}
        assert labels, row["file"]
        missing = labels - set(C.speaker_bindings(row)) - _KNOWN_UNBOUND
        assert not missing, (row["file"], missing)


@pytest.mark.parametrize("iso, source_ref, translator", VENDORED)
def test_the_row_a_render_reads_carries_its_speaker_map(iso, source_ref, translator):
    _, row = C.vendored_text(_corpus_root(), iso, source_ref)
    assert row is not None and C.speaker_bindings(row)


def test_tampered_bytes_are_refused_rather_than_performed():
    """The integrity check is the whole reason the manifest carries a hash: a
    corpus whose text has drifted from what was reviewed must not be spoken."""
    import io
    import os
    import shutil
    import tempfile
    root = _corpus_root()
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copytree(root, tmp, dirs_exist_ok=True)
        victim = os.path.join(tmp, "it", "macbeth_1_3.txt")
        body = io.open(victim, encoding="utf-8").read()
        io.open(victim, "w", encoding="utf-8", newline="\n").write(
            body + "MACBETH: A line no translator wrote.\n")
        assert C.vendored_text(tmp, "it", "folger-macbeth:act1-scene3-witches") == ("", None)


#: Stage business the colon editions set in ROUND brackets, which Folger sets
#: in square ones and the selector already strips. Each of these was spoken
#: aloud in a planned window before `strip_direction_parentheticals`.
EDITION_STAGE_WORDS = (
    "entra Rosse", "entrano Rosse", "escono", "scompaiono", "scompariscono",
    "s\u2019ode un tamburo", "Entran Corino", "Sale.",
    # Survived the first strip and reached the vendored Macbeth, where it sat
    # inside MACBETH's own line and would have been performed in his voice.
    # A hand-list is why: every phrase above is short, so nothing here tested
    # the LENGTH CAP that let this one through. See the test below, which
    # pins the cause rather than one more symptom.
    "rimane alcuni istanti",
)


@pytest.mark.parametrize("iso, source_ref, _translator", VENDORED)
def test_no_stage_direction_is_left_to_be_spoken(iso, source_ref, _translator):
    """A direction that survives is READ ALOUD by whoever holds the line.

    Folger puts stage business in square brackets and the selector strips them
    before parsing; Rusconi and Marquez use round ones, so the vendored lane
    inherited an assumption that was false for its own sources. A blanket `()`
    strip is the wrong fix -- Hugo prints real dialogue in parentheses, exactly
    as Folger prints "(God shield us!)" inside Bottom's line -- so the strip is
    keyed on the edition's own italics and this asserts the outcome.
    """
    text, _row = C.vendored_text(_corpus_root(), iso, source_ref)
    assert text
    for phrase in EDITION_STAGE_WORDS:
        assert phrase.lower() not in text.lower(), (source_ref, phrase)


def test_a_long_stage_direction_is_stripped_like_a_short_one():
    """THE CAUSE, not another symptom. The strip was written `[^<]{1,80}`, and
    80 is a number somebody picked -- a stage direction is a sentence whenever
    the edition wants one. Rusconi has two past it, and the failure mode is the
    worst available: a direction too long for the cap is not reported, it is
    silently KEPT and then spoken by whoever holds the line.

    The hand-list above cannot catch this class, because every phrase in it is
    short. What bounds the match honestly is `[^<]`, which cannot cross a tag,
    so the body is one text run inside one `<i>` wrapped in parentheses.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_otr_vendor_shakespeare_under_test",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "scripts", "otr_vendor_shakespeare.py"))
    vendor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vendor)

    # The real one, 89 characters, from it/macbeth 1.3.
    long_direction = ("rimane alcuni istanti assorto in profonda meditazione, "
                      "quindi si volge ad Angus e a Rosse")
    assert len(long_direction) > 80, "this test is pointless if it fits the old cap"
    markup = ("<p>Thane di Glamis e di Cawdor! Poi... "
              "(<i>" + long_direction + "</i>) Grazie, signori.</p>")
    assert "rimane alcuni istanti" not in vendor.strip_direction_parentheticals(markup)

    # And the short ones still go, so the fix did not trade one bug for another.
    assert "entra Rosse" not in vendor.strip_direction_parentheticals(
        "<p>Poi... (<i>entra Rosse</i>) Grazie.</p>")

    # A parenthetical that is NOT italic is dialogue and is untouched, however
    # long it runs -- that is the whole reason the strip is keyed on markup.
    dialogue = "(car cette partie du monde connu l’estimait pour tel)"
    assert dialogue in vendor.strip_direction_parentheticals("<p>x " + dialogue + " y</p>")


def test_rusconi_headed_markup_claims_speakers_not_italic_prose():
    """The Rusconi rule reads real paragraph-head labels off fetched markup.

    The fixture is copied from the Italian Wikisource pages named by the held
    leads.  The same page uses italics for entrances and for words spoken by a
    character, so a loose ``<i>... </i>.`` match must not cast those strings.
    """
    import importlib.util
    # Load the vendor module independently; the fixture path is only the data
    # source, not an importable Python file.
    spec = importlib.util.spec_from_file_location(
        "_otr_vendor_shakespeare_rusconi_vendor",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "scripts", "otr_vendor_shakespeare.py"))
    vendor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vendor)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "fixtures", "rusconi_speaker_markup.html"),
              encoding="utf-8") as handle:
        markup = handle.read()

    out = vendor.to_text(markup)
    text = vendor.canonicalise_labels(vendor.normalise_labels(out))
    labels = [line.split(":", 1)[0]
              for line in text.splitlines() if ":" in line]
    assert labels == ["ORL", "BER", "VIL", "MAL"], labels
    for wrong in ("ESCONO GLOC. ED EDM", "CUCULLUS NON FACIT MONACHUM",
                  "M. O. A. I", "CON UN FOGLIO"):
        assert wrong not in labels


def test_personnage_is_read_as_a_speaker_class_like_sc():
    """The same fact in a different spelling, and missing it cost a whole play.

    Hugo's `Le soir des rois` was transcribed by someone who marked speakers
    `class="personnage"` -- 918 spans of it against 119 `class="sc"`. With only
    `sc` known, `to_text` marked 66 speakers on a page carrying 918 labels, so
    the scene would not have come out WRONG so much as almost entirely
    unattributed prose. Two independent reviewers reported it before it was
    measured; afterwards the same page marks 1902 (two marks per label).

    Folded into the union rather than held for a per-edition binding because
    `personnage` appears ZERO times on all four vendored pages -- it cannot
    reach them -- and the class name means exactly one thing.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_otr_vendor_shakespeare_personnage",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "scripts", "otr_vendor_shakespeare.py"))
    vendor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vendor)

    mark = vendor.SPEAKER_MARK
    for markup, name in [
        ('<span class="personnage">le duc.</span>', "le duc."),
        ('<span class="personnage" style="">maria.</span>', "maria."),
        ('<span class="sc">Macbeth</span>', "Macbeth"),
    ]:
        out = vendor.to_text(markup)
        assert mark in out, "%r produced no speaker mark" % markup
        assert name.rstrip(".") in out

    # a class that is neither is still not a speaker
    assert mark not in vendor.to_text('<span class="poem">un vers</span>')


def test_a_cjk_scene_ends_at_the_next_cjk_heading():
    """The scene-end stem was derived by splitting the label on WHITESPACE.

    `SCENE III.` splits to `SCENE`, which the following `SCENE IV.` starts
    with, so the cut closes. A CJK heading has no spaces, so the stem became the
    WHOLE label including its number, the next scene never started with it, and
    nothing ever ended the scene. Measured on Tsubouchi's Romeo and Juliet:
    act 1 scene 1 returned 322 lines where the window is about 114, running
    through the act and into the next, and the balcony scene came back holding
    Mercutio, the Nurse, Friar Laurence and the Prince.

    The headings here carry their SETTING after two ideographic spaces, exactly
    as Aozora prints them -- a bare `第一場` would not have reproduced the bug.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_otr_vendor_shakespeare_cjkcut",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "scripts", "otr_vendor_shakespeare.py"))
    vendor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vendor)

    lines = [
        "第一幕",                                    # act one
        "第一場　　ヴェローナ。街上。",
        "サン　やい、グレゴリー。",
        "グレ　そうとも。",
        "サン　またな。",
        "第二場　　同じく。街上。",   # <- must stop HERE
        "キャピ　これは別の場面。",
        "第二幕",
        "ロミオ　別の幕。",
    ]
    body, reason = vendor.extract(
        lines, None, "第一幕", "第一場")
    assert body, reason
    got = body.splitlines()
    assert got[0].startswith("第一場")
    # heading + three speeches; the span is [scene heading, next heading)
    assert len(got) == 4, "cut did not close at the next scene: %r" % got
    assert "第二場" not in body, "ran into the next scene"
    assert "第二幕" not in body, "ran into the next act"

    # an ACT boundary ends a scene too, even with no further scene heading
    lines2 = [
        "第三幕", "第一場　　森。",
        "ア　一。", "イ　二。", "ウ　三。",
        "第四幕", "エ　四。",
    ]
    body2, _ = vendor.extract(
        lines2, None, "第三幕", "第一場")
    assert body2 and "第四幕" not in body2

    # and the Latin path is untouched
    latin = ["SCENA III.", "PRIMA. a", "SECONDA. b", "TERZA. c", "SCENA IV.", "X. d"]
    body3, _ = vendor.extract(latin, None, None, "SCENA III.")
    assert body3 and "SCENA IV." not in body3


def test_a_ruby_gloss_is_a_pronunciation_guide_and_is_not_spoken():
    """Japanese editions annotate a kanji with how to READ it, and nobody says
    the annotation out loud.

        <ruby><rb>誓言</rb><rp>（</rp><rt>せいごん</rt><rp>）</rp></ruby>

    `<rb>` is the word, `<rt>` is the pronunciation, `<rp>` holds the fallback
    brackets a ruby-less browser shows. Strip tags naively and all three
    survive, so a line reads `誓言（せいごん）` -- the word followed by its own
    pronunciation -- and a voice performs both.

    Measured on Tsubouchi Shoyo's Romeo and Juliet (Aozora Bunko), where nearly
    every content word is glossed: one line came out as
    `威權（ゐけん）相如（あひし）く二名族（めいぞく）が、`. With `<rp>` and
    `<rt>` removed the same page yields 877 speaker-shaped lines across 56
    distinct characters, which is a whole language unblocked.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_otr_vendor_shakespeare_ruby",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "scripts", "otr_vendor_shakespeare.py"))
    vendor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vendor)

    glossed = ("<p><ruby><rb>誓言</rb><rp>（</rp>"
               "<rt>せいごん</rt><rp>）</rp></ruby>"
               "じゃ</p>")
    out = vendor.to_text(glossed)
    assert "誓言" in out, "the WORD was dropped"
    assert "せいごん" not in out, "the READING would be spoken"
    assert "（" not in out and "）" not in out, "ruby brackets survived"

    # a real parenthetical that is NOT ruby is untouched -- Tsubouchi writes
    # plain-language glosses in parentheses and those are his own words.
    plain = vendor.to_text("<p>あ（本文）</p>")
    assert "（" in plain


def test_a_small_type_block_that_is_a_bare_name_is_kept_as_a_speaker():
    """Some transcribers set the SPEAKER in small type too.

    Once the direction rule reads the whole 60-99% band instead of two
    hard-coded percentages, it reaches them. Measured on Menendez y Pelayo's
    Spanish editions: `font-size: 83%` wraps EVERY speaker -- 127 such blocks in
    Macbeth, 193 in Romeo and Juliet -- so stripping the band wholesale deletes
    `ROMEO.`, `BENVOLIO.` and `MERCUTIO.` and leaves the play with nobody to say
    the lines. Neither edition is vendored, so nothing shipped was hurt; this
    closes the trap before someone records a label for one.

    The signal is case and punctuation, never the percentage. An ACT or SCENE
    heading is kept too, deliberately: `extract` locates a scene BY those
    headings, and no name shape matches them (they carry no trailing period).
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_otr_vendor_shakespeare_smalltype",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "scripts", "otr_vendor_shakespeare.py"))
    vendor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vendor)

    def marked(body):
        # NOTE the doubled `%%`: this is a %-formatted string and a literal
        # `83%;` reads as a format spec, which raises rather than failing an
        # assertion -- it cost a confusing red run on the way in.
        return vendor.to_text(
            '<p><span style="font-size: 83%%;">%s</span></p>' % body)

    # speakers survive
    for name in ("ROMEO.", "BENVOLIO.", "LADY MACBETH.",
                 "BRUJA 1.ª"):          # the Spanish ordinal is not lowercase
        assert name in marked(name), "small-type speaker %r was deleted" % name

    # stage business still goes
    for direction in ("Plaza pública, cerca del jardin de Capuleto.",
                      "Tres BRUJAS.", "Fanfares.", "À Cordélia."):
        assert direction.strip(".") not in marked(direction), (
            "stage business %r survived" % direction)

    # apparatus with a colon is not a speaker
    assert "PERSONNAGES" not in marked("PERSONNAGES :")

    # and a heading is kept, because scene location needs it
    assert "ACTO PRIMERO" in marked("ACTO PRIMERO")


def test_a_tag_ends_at_the_first_unquoted_angle_bracket():
    """`<[^>]+>` stops at the first `>` ANYWHERE, including one inside a quoted
    attribute -- and MediaWiki's Parsoid puts a JSON blob in `data-mw` that
    contains escaped markup, so the rest of the attribute survives as TEXT.

    Measured on the transcribed Portuguese Hamlet before the fix: 16 lines over
    200 characters, FIFTY JSON tokens reaching the text, and both the act
    heading `ACTO PRIMEIRO` and the speaker `BERNARDO` buried at the tail of
    ~300-character lines instead of standing on their own. That breaks scene
    location outright, and anything vendored would have a character reading
    `"quality":{"wt":"4"}},"i":1}}` ALOUD.

    Not a Portuguese problem: every Parsoid-rendered Wikisource page carries
    `data-mw`. The first four editions came out clean by luck of layout.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_otr_vendor_shakespeare_tags",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "scripts", "otr_vendor_shakespeare.py"))
    vendor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vendor)

    parsoid = ('<p data-mw=\'{"parts":[{"x":"&lt;span&gt;"},"</span>"]}\'>'
               'ACTO PRIMEIRO</p>')
    out = vendor.to_text(parsoid)
    assert "ACTO PRIMEIRO" in out
    assert "wt" not in out and "parts" not in out, (
        "attribute JSON leaked into the spoken text: %r" % out)

    # a `>` inside an ordinary quoted attribute must not end the tag either
    assert "link" in vendor.to_text('<a title="a > b">link</a>')
    assert "b\"" not in vendor.to_text('<a title="a > b">link</a>')

    # and the ordinary cases still strip. NOTE a `class="sc"` span is a
    # SPEAKER, so it comes back wrapped in SPEAKER_MARK rather than bare --
    # that is `mark_speakers` doing its job, not tag residue.
    marked = vendor.to_text('<span class="sc">Macbeth</span>')
    assert marked.strip() == vendor.SPEAKER_MARK + "Macbeth" + vendor.SPEAKER_MARK
    assert vendor.to_text("<p><i>entra Rosse</i></p>").strip() == "entra Rosse"


def test_a_held_lead_carries_its_reason_and_is_not_vendored():
    """A HOLD is a located, correct source the extractor cannot read SAFELY yet.

    It is not `excluded`, which disqualifies the TRANSLATION (indirect, or no
    text exists). The distinction is the whole point: a held row is good source
    and someone will come back to it, so the reason travels with it.

    This exists because `alt_of` was carrying the meaning, and a flag whose name
    says "alternate" cannot say "do not vendor, the parser is not ready". It was
    read as a data error, cleared, and the row vendored on the spot with several
    speakers in the wrong mouths. A reader who clears a field named `hold` and
    reads the reason is making a decision; one who clears `alt_of` is tidying.
    """
    root = _corpus_root()
    with open(os.path.join(root, "leads.json"), encoding="utf-8") as handle:
        leads = json.load(handle)["leads"]
    with open(os.path.join(root, "manifest.json"), encoding="utf-8") as handle:
        vendored = {(r["iso"], r["play"], r["scene"])
                    for r in json.load(handle)["scenes"]}

    held = [r for r in leads if r.get("hold")]
    assert held, "no held rows -- delete this test rather than letting it pass vacuously"
    for row in held:
        key = (row["iso"], row["play"], row["scene"])
        assert str(row["hold"]).strip(), "%s/%s %s is held with no reason" % key
        assert len(str(row["hold"])) > 40, (
            "%s/%s %s: a hold reason has to say what is actually wrong" % key)
        assert key not in vendored, (
            "%s/%s %s is HELD but a manifest row vendors it anyway" % key)


def test_every_shipped_manifest_row_has_its_text_on_disk():
    """A `READY` row aimed at nothing, which NOTHING ELSE CATCHES.

    The vendor script appends manifest rows and never prunes them, so removing
    a vendored scene leaves its row behind. Measured 2026-09-19: a Chinese
    scene was vendored, found to carry two wrong mouths, and deleted -- and
    `load_manifest` validated the surviving row without complaint, because
    `file` being non-blank says a path was WRITTEN DOWN, not that anything is
    there.

    Downstream the failure is silent by design: `vendored_text` returns
    ("", None) for a missing file exactly as it does for a hash mismatch, so
    the lane falls back to the model translation while the manifest still
    advertises a real translator's words. That degradation is CORRECT at render
    time and wrong at authoring time, which is why this is a test over the
    shipped corpus rather than a raise inside the loader -- the loader is also
    called with fixture manifests whose files deliberately do not exist.
    """
    root = _corpus_root()
    with open(os.path.join(root, "manifest.json"), encoding="utf-8") as handle:
        manifest = json.load(handle)
    missing = []
    for row in manifest["scenes"]:
        if not os.path.isfile(os.path.join(root, row["file"])):
            missing.append("%s/%s %s -> %s"
                           % (row["iso"], row["play"], row["scene"], row["file"]))
    assert not missing, (
        "manifest rows naming text that is not on disk: %s" % "; ".join(missing))


def test_a_chinese_clause_is_not_mistaken_for_a_speaker_name():
    """A WRONG MOUTH, measured on Zhu Shenghao's Midsummer: Bottom's whole
    speech was performed by Snout.

        司: 咱担保她们一定会吓怕。 波　列位，你们得好好想一想：...

    `_LABEL_SHAPES[0]` accepts the FULLWIDTH colon as a name separator, so it
    matched everything up to one and offered `波　列位，你们得好好想一想` as a
    speaker. The guard that exists to catch exactly this -- punctuation proving
    the candidate is a sentence -- listed only `,;--!?`, and U+FF0C is not
    U+002C. The bogus name then occurred once, so the recurrence rule demoted
    the line to continuation and merged it into the previous speaker.

    Rejecting the clause lets shape[3] have the line, which reads the
    ideographic space and returns the real speaker. Note neither length guard
    can help here: Chinese does not space its words, so `split()` always counts
    one.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_otr_vendor_shakespeare_cjk",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "scripts", "otr_vendor_shakespeare.py"))
    vendor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vendor)

    line = "波　列位，你们得好好想一想：这事情可不得了"
    assert vendor._name_of(line) == "波", "a whole clause read as a speaker"

    # the ASCII half still works, and a real Latin label is untouched
    assert vendor._name_of("MACBETH: Thane di Glamis") == "MACBETH"
    assert vendor._name_of("Hold, hold, my heart; this is not a name") == ""


def test_the_translators_own_parenthetical_dialogue_survives():
    """The other half, and the reason a blanket strip was refused: Horatio's
    "(car cette partie du monde connu l'estimait pour tel)" is Hugo's French
    for Folger's spoken "(For so this side...)". Losing it would be the same
    defect in the opposite direction."""
    text, _row = C.vendored_text(
        _corpus_root(), "fr", "folger-hamlet:act1-scene1-platform-watch")
    assert "car cette partie du monde" in text


def test_a_vendored_row_with_no_speaker_map_reports_every_label_unbound():
    """`{}` IS NOT `None`. A vendored row whose manifest carries no speaker_map
    is the case that needs the unbound receipt most -- every label degrades to
    the gender roll -- and truthiness read it as the English path, so it was
    the one case that went unreceipted and unwarned. Two tests bracketed this
    and both missed it: one covers a PARTIAL map, one covers the key ABSENT.
    """
    from nodes import _otr_verbatim_lane as VL
    text, _row = C.vendored_text(
        _corpus_root(), "it", "folger-macbeth:act1-scene3-witches")
    plan, receipt = VL.plan_verbatim_passage(
        source_text=text, source_meta={"recommended_word_budget": 300},
        num_characters=3, act_count=1,
        source_ref="folger-macbeth:act1-scene3-witches", speaker_map={})
    assert plan is not None
    # every speaker the edition wrote is reported, because none was bound
    assert receipt["unbound_labels"], "an empty map reported nothing unbound"
    assert set(plan.speakers) <= set(receipt["unbound_labels"])
    assert not plan.roster_names


def test_the_english_path_still_reports_no_unbound_labels():
    """The other side of the sentinel: `None` is English and must stay
    byte-identical -- no unbound key, nothing to warn about."""
    from nodes import _otr_verbatim_lane as VL
    text, row = C.vendored_text(
        _corpus_root(), "it", "folger-macbeth:act1-scene3-witches")
    _plan, receipt = VL.plan_verbatim_passage(
        source_text=text, source_meta={"recommended_word_budget": 300},
        num_characters=3, act_count=1,
        source_ref="folger-macbeth:act1-scene3-witches", speaker_map=None)
    assert not receipt.get("unbound_labels")


def test_an_indent_marked_edition_gets_its_speakers_and_loses_its_business():
    """Tsubouchi's Aozora text marks by INDENT CLASS, not by type size.

    The Japanese rows sat held on a recorded diagnosis that said this edition
    "sets stage business inline, in the same run as the dialogue, with no
    markup of its own". It does not. The page is meticulously marked -- it
    simply uses a vocabulary neither other rule knows:

        <div class="burasage">サン　　dialogue...</div>       a speech
        <div class="jisage_8">...持って出る。</div>            stage business
        <div class="jisage_6"><h4>第一場　...</h4></div>       a scene heading

    `_DIRECTION_BLOCK` keys on font-size 60-99%; `_SPEAKER_SPANS` keys on span
    classes and small caps. Neither fires here, so 95 marked dialogue lines in
    act 1 scene 1 collapsed into unattributed prose -- which is how a one-line
    part came to "absorb" the Prince's entire first speech, in her voice.

    THREE THINGS THIS PINS, each of which was a live defect:

    * The business strip is UNCONDITIONAL, not `_strip_direction`. That helper
      keeps a block `_is_bare_label` calls a name, and that test asks "no
      lowercase and no colon?" -- a question JAPANESE CANNOT FAIL, having no
      case at all. So `と劍を拔く。` and `二人ともに入る。` were kept as speech.
    * A speaker label may contain an IMAGE. Shift_JIS cannot encode every
      character, so Aozora sets the missing ones as `<img class="gaiji">`, and
      Benvolio's label ends in one. A rule stopping at `<` marked 71 of 95
      lines and dropped Benvolio from the scene entirely.
    * The heading divs are EXEMPT, because `extract` locates a scene by them.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_otr_vendor_shakespeare_aozora",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "scripts", "otr_vendor_shakespeare.py"))
    vendor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vendor)

    page = (
        '<div class="jisage_6" style="margin-left: 6em">'
        '<h4 class="naka-midashi">第一場　　ローナ。街上。</h4></div>'
        '<div class="jisage_8" style="margin-left: 8em">'
        'サンプソンとグレゴリーとが劍と楯とを持って出る。<br /></div>'
        '<div class="burasage" style="margin-left: 4em;">'
        'サン　　やい、グレゴリー。</div>'
        '<div class="jisage_8" style="margin-left: 8em">と劍を拔く。<br /></div>'
        '<div class="burasage" style="margin-left: 4em;">'
        'ベン<img src="../../../gaiji/1-07/1-07-85.png" alt="gaiji" '
        'class="gaiji" />　待った！</div>'
        '<div class="jisage_8" style="margin-left: 8em">二人ともに入る。<br /></div>'
    )
    out = vendor.to_text(vendor.mark_speakers(page))

    # the scene heading survives -- `extract` needs it to find the scene
    assert "第一場" in out

    # both speakers are marked, and the gaiji-bearing label reads as its text
    labels = re.findall("%s([^%s]+)%s" % (vendor.SPEAKER_MARK,
                                          vendor.SPEAKER_MARK,
                                          vendor.SPEAKER_MARK), out)
    assert labels == ["サン", "ベン"], labels

    # every stage direction is gone -- including the two SHORT ones that
    # `_is_bare_label` would have kept
    for business in ("持って出る", "と劍を拔く", "二人ともに入る"):
        assert business not in out, "stage business survived: %s" % business

    # and the dialogue itself is untouched
    assert "やい、グレゴリー。" in out
    assert "待った！" in out
