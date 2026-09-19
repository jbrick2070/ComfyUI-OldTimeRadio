"""The vendored-translation corpus: legal tests, gate verdicts, manifest.

Operator 2026-09-18: "I want the best pack available" -- a scene ships a real
translator's words only when they clear BOTH the US test and life+70. The old
"died before 1944" line was a conservative bound, not a copyright test.
CPU only, no network. UTF-8 no BOM.
"""
from __future__ import annotations

import json

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
    assert C.clears_publication_anywhere(published, died)
    assert C.publication_reasons(published, died) == []


@pytest.mark.parametrize("published, died, why", [
    (1947, 1944, "first published 1947"),        # Zhu: US test fails
    (1900, 1962, "translator died 1962"),        # Rangeya Raghav: life+70 fails
    (1990, 2003, "first published 1990"),        # Bachchan: both fail
])
def test_a_row_that_fails_either_test_is_refused_and_says_which(published, died, why):
    assert not C.clears_publication_anywhere(published, died)
    assert any(why in r for r in C.publication_reasons(published, died))


def test_zhu_shenghao_is_the_case_the_old_rule_got_wrong():
    """d.1944 clears life+70, but the collected plays were published in 1947,
    so the pack-I-can-publish set excludes them until the US clock runs out.
    The 'died before 1944' bound would have waved them through."""
    assert C.clears_publication_anywhere(1947, 1944) is False
    assert C.clears_publication_anywhere(1930, 1944) is True


@pytest.mark.parametrize("value", ["", None, "unknown", "n/a", True, 0])
def test_an_unrecorded_year_never_clears(value):
    assert not C.clears_publication_anywhere(value, 1873)
    assert not C.clears_publication_anywhere(1865, value)


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


def test_rights_beat_every_quality_measure():
    """A perfect page from an in-copyright translator is BLOCKED, and the
    reasons name the rights, not the prose."""
    r = C.assess(_clean_report(translator_died=1962, first_published=1955))
    assert r.verdict == C.BLOCKED
    assert all("speaker" not in reason for reason in r.reasons)


def test_a_missing_licence_is_blocked_not_partial():
    r = C.assess(_clean_report(licence=""))
    assert r.verdict == C.BLOCKED
    assert "transcription_license" in r.reasons[0]


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


@pytest.mark.parametrize("over, fragment", [
    ({"translator_death_date": 1962}, "does not clear"),
    ({"translation_first_published": 1947}, "does not clear"),
])
def test_a_manifest_row_that_does_not_clear_is_refused_loudly(tmp_path, over, fragment):
    with pytest.raises(C.CorpusError) as caught:
        C.load_manifest(_write(tmp_path, [_manifest_row(**over)]))
    assert fragment in str(caught.value)


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
    assert not C.clears_publication_anywhere("1929-1932", 1889)
    assert C.clears_publication_anywhere("1865-1872", 1873)
    # A death is a single event; that reader still takes the first year.
    assert C._year("1935-02-28") == 1935


def test_tsubouchi_blocks_on_either_spelling_of_the_revision():
    assert not C.clears_publication_anywhere(1933, 1935)
    assert not C.clears_publication_anywhere("1933-1935", 1935)
    # The earlier first publication would clear, which is why the lead says
    # to confirm WHICH text before vendoring.
    assert C.clears_publication_anywhere(1910, 1935)


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
