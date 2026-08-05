"""Chunk 1 of the source-grounding build: uncapped document + overview.

The defect these pin: the pre-outline authors read a 12,000-character PREFIX
of a body that can run 25,200 words, while the packs told them to be faithful
to the whole work.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from nodes import _otr_public_domain_sources as pd
from nodes import _otr_source_document as osd
from nodes import _otr_source_payload as osp

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS = REPO_ROOT / "config" / "source_banks" / "public_domain_story" / "sources"


# ---------------------------------------------------------------------------
# normalization split: the owner is uncapped, the legacy projection is not
# ---------------------------------------------------------------------------

def test_normalize_public_domain_body_does_not_truncate():
    body = "word " * 8000  # ~40,000 chars, well past the legacy 12,000 cap
    out = pd.normalize_public_domain_body(body)
    assert len(out) > pd.INTERPRETER_TEXT_WINDOW
    assert out.split()[:3] == ["word", "word", "word"]


def test_canonicalize_still_caps_for_the_legacy_payload():
    body = "word " * 8000
    out = pd.canonicalize_public_domain_text(body)
    assert len(out) <= pd.INTERPRETER_TEXT_WINDOW


def test_projection_is_a_prefix_of_the_owner():
    body = "alpha beta gamma delta " * 900
    full = pd.normalize_public_domain_body(body)
    capped = pd.canonicalize_public_domain_text(body)
    assert full.startswith(capped)


def test_normalization_strips_gutenberg_boilerplate_and_collapses_space():
    raw = (
        "*** START OF THE PROJECT GUTENBERG EBOOK SOMETHING ***\n\n"
        "The   traveller\treturned.\n\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK SOMETHING ***\ntrailing junk"
    )
    assert pd.normalize_public_domain_body(raw) == "The traveller returned."


def test_empty_body_is_refused_by_name():
    with pytest.raises(pd.PublicDomainSourceRefError):
        pd.normalize_public_domain_body("   \n\t ")


def test_normalization_is_idempotent():
    raw = "  The   traveller\n\nreturned.  "
    once = pd.normalize_public_domain_body(raw)
    assert pd.normalize_public_domain_body(once) == once


# ---------------------------------------------------------------------------
# document identity
# ---------------------------------------------------------------------------

def test_body_hash_is_sha256_over_utf8():
    body = "The traveller returned."
    expected = hashlib.sha256(body.encode("utf-8")).hexdigest()
    assert osd.canonical_body_sha256(body) == expected


def test_build_source_document_carries_identity():
    doc = osd.build_source_document("The traveller returned.", source_ref="x:y")
    assert doc.body_sha256 == osd.canonical_body_sha256(doc.canonical_body)
    assert doc.normalization_version == osd.NORMALIZATION_VERSION
    assert doc.source_ref == "x:y"
    assert doc.word_count == 3


def test_build_source_document_refuses_empty():
    with pytest.raises(osd.SourceDocumentError):
        osd.build_source_document("   ")


def test_document_span_validates_its_own_text():
    doc = osd.build_source_document("The traveller returned.")
    span = doc.span(4, 13)
    assert span.text == "traveller"
    assert span.char_count == 9
    doc.verify_span(span)


def test_document_refuses_out_of_range_span():
    doc = osd.build_source_document("short body")
    with pytest.raises(osd.SourceDocumentError):
        doc.span(0, 999)
    with pytest.raises(osd.SourceDocumentError):
        doc.span(5, 5)


def test_verify_span_catches_drifted_text():
    doc = osd.build_source_document("The traveller returned.")
    drifted = osd.SourceSpan(start_char=4, end_char=13, text="voyager")
    with pytest.raises(osd.SourceDocumentError):
        doc.verify_span(drifted)


# ---------------------------------------------------------------------------
# overview: total coverage, deterministic, body-free receipts
# ---------------------------------------------------------------------------

def _doc(words: int = 4000) -> osd.SourceDocument:
    body = " ".join(f"w{i}" for i in range(words))
    return osd.build_source_document(body, source_ref="unit:one")


def test_overview_covers_the_entire_body():
    doc = _doc()
    overview = osd.build_source_overview(doc)
    assert overview.covered_char_count == doc.char_count


def test_overview_windows_are_ordered_and_gapless():
    overview = osd.build_source_overview(_doc())
    windows = overview.windows
    assert windows[0].start_char == 0
    for earlier, later in zip(windows, windows[1:]):
        assert earlier.end_char == later.start_char
    assert windows[-1].end_char == overview.covered_char_count


def test_every_overview_span_matches_the_body_it_quotes():
    doc = _doc()
    overview = osd.build_source_overview(doc)
    for span in list(overview.windows) + list(overview.evidence):
        assert span.text == doc.canonical_body[span.start_char:span.end_char]
        doc.verify_span(span)


def test_overview_is_deterministic():
    doc = _doc()
    first = osd.build_source_overview(doc)
    second = osd.build_source_overview(doc)
    assert first.receipt() == second.receipt()


def test_overview_tags_opening_closing_and_dialogue():
    doc = _doc()
    overview = osd.build_source_overview(doc)
    opening = overview.evidence_for(osd.EVIDENCE_OPENING)
    closing = overview.evidence_for(osd.EVIDENCE_CLOSING)
    dialogue = overview.evidence_for(osd.EVIDENCE_DIALOGUE)
    assert opening is not None and opening.start_char == 0
    assert closing is not None and closing.end_char == doc.char_count
    assert dialogue is not None


def test_dialogue_evidence_prefers_the_quote_dense_window():
    quiet = "narration here and there. " * 200
    talky = '"Story!" cried the Editor. "Tell us." ' * 60
    doc = osd.build_source_document(quiet + talky + quiet)
    overview = osd.build_source_overview(doc, window_count=3)
    dialogue = overview.evidence_for(osd.EVIDENCE_DIALOGUE)
    assert '"' in dialogue.text


def test_overview_receipt_is_body_free_and_json_serializable():
    doc = _doc()
    overview = osd.build_source_overview(doc)
    receipt = overview.receipt()
    encoded = json.dumps(receipt)
    # The receipt proves WHICH material grounded a build without carrying it.
    assert "w100" not in encoded
    assert receipt["body_sha256"] == doc.body_sha256
    assert receipt["covered_char_count"] == doc.char_count
    assert receipt["window_count"] == len(overview.windows)


def test_single_window_and_tiny_body_still_cover_totally():
    doc = osd.build_source_document("tiny body here")
    overview = osd.build_source_overview(doc, window_count=8)
    assert overview.covered_char_count == doc.char_count


# ---------------------------------------------------------------------------
# transport: transient, and not smuggled into serialized sidecars
# ---------------------------------------------------------------------------

def test_fetch_result_carries_the_document_without_the_sidecars_seeing_it():
    doc = osd.build_source_document("The traveller returned.")
    payload = {
        "headline": "h", "summary": "s", "full_text": "f", "source": "src",
        "date": "1895", "link": "u", "seed_text": "seed",
    }
    result = osp.SourceFetchResult(
        payload=payload, source_meta={"title": "t"}, source_rights={},
        source_document=doc,
    )
    _, meta, rights = osp.normalize_fetch_result(result, origin="test")
    assert "source_document" not in meta
    assert "source_document" not in rights
    assert "canonical_body" not in json.dumps(meta)


def test_document_aware_normalizer_returns_it_and_plain_one_drops_it():
    doc = osd.build_source_document("The traveller returned.")
    payload = {
        "headline": "h", "summary": "s", "full_text": "f", "source": "src",
        "date": "1895", "link": "u", "seed_text": "seed",
    }
    result = osp.SourceFetchResult(payload=payload, source_document=doc)

    assert len(osp.normalize_fetch_result(result, origin="test")) == 3

    with_doc = osp.normalize_fetch_result_with_document(result, origin="test")
    assert len(with_doc) == 4
    assert with_doc[3] is doc


def test_legacy_fetchers_still_normalize_with_no_document():
    payload = {
        "headline": "h", "summary": "s", "full_text": "f", "source": "src",
        "date": "1895", "link": "u", "seed_text": "seed",
    }
    out = osp.normalize_fetch_result_with_document(payload, origin="test")
    assert out[3] is None


# ---------------------------------------------------------------------------
# the real corpus: the property test the panel asked for
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not CORPUS.is_dir(), reason="public-domain corpus absent")
def test_every_corpus_source_normalizes_hashes_and_covers():
    texts = sorted(CORPUS.glob("*.txt"))
    assert texts, "corpus directory has no sources"
    for path in texts:
        raw = path.read_text(encoding="utf-8", errors="replace")
        body = pd.normalize_public_domain_body(raw)
        assert pd.normalize_public_domain_body(body) == body, path.name

        doc = osd.build_source_document(body, source_ref=path.stem)
        assert doc.body_sha256 == osd.canonical_body_sha256(body), path.name

        overview = osd.build_source_overview(doc)
        assert overview.covered_char_count == doc.char_count, path.name
        for span in list(overview.windows) + list(overview.evidence):
            assert span.text == body[span.start_char:span.end_char], path.name


@pytest.mark.skipif(not CORPUS.is_dir(), reason="public-domain corpus absent")
def test_the_corpus_really_does_outrun_the_legacy_cap():
    # The premise of the whole chunk: some works are far past the prefix, so
    # a capped body genuinely loses material the authors were told to carry.
    longest = max(
        (len(pd.normalize_public_domain_body(
            p.read_text(encoding="utf-8", errors="replace")))
         for p in CORPUS.glob("*.txt")),
        default=0,
    )
    assert longest > pd.INTERPRETER_TEXT_WINDOW
