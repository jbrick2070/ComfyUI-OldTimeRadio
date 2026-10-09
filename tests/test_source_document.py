"""Chunk 1 of the source-grounding build: the uncapped document.

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


def test_legacy_parity_at_max_chars_zero_still_raises():
    # The pre-refactor function truncated FIRST and checked empty AFTER, so
    # max_chars=0 raised. Splitting normalization out moved that check; it is
    # restored, because a silent empty body is worse than a loud refusal.
    with pytest.raises(pd.PublicDomainSourceRefError):
        pd.canonicalize_public_domain_text("hello world", max_chars=0)


def test_a_supplied_canonical_body_must_match_the_text():
    # The normalize-once optimization must not let a caller pair a payload
    # with a document built from different material.
    with pytest.raises(pd.PublicDomainSourceRefError):
        pd.source_document_from_text(
            "the real source text", canonical_body="something else entirely")


def test_a_matching_supplied_canonical_body_is_accepted():
    text = "  The traveller   returned.  "
    body = pd.normalize_public_domain_body(text)
    doc = pd.source_document_from_text(text, canonical_body=body)
    assert doc.canonical_body == body


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


# ---------------------------------------------------------------------------
# the body must not escape through repr -- logs, tracebacks, f-strings
# ---------------------------------------------------------------------------

_SENTINEL = "SECRETBODYWORD"


def _sentinel_doc() -> osd.SourceDocument:
    return osd.build_source_document(
        f"opening {_SENTINEL} middle " * 400, source_ref="u:1")


def test_document_repr_carries_identity_not_the_body():
    doc = _sentinel_doc()
    text = repr(doc)
    assert _SENTINEL not in text
    assert doc.body_sha256[:12] in text


def test_fetch_result_repr_does_not_carry_the_body():
    doc = _sentinel_doc()
    result = osp.SourceFetchResult(
        payload={
            "headline": "h", "summary": "s", "full_text": "f", "source": "x",
            "date": "1895", "link": "l", "seed_text": "seed",
        },
        source_document=doc,
    )
    assert _SENTINEL not in repr(result)


def test_logging_a_document_does_not_dump_the_body():
    import io
    import logging

    doc = _sentinel_doc()
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    log = logging.getLogger("otr.test.source_document")
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    try:
        log.info("document: %s", doc)
    finally:
        log.removeHandler(handler)
    assert _SENTINEL not in buf.getvalue()


def test_exception_formatting_does_not_dump_the_body():
    doc = _sentinel_doc()
    assert _SENTINEL not in str(ValueError(f"bad state: {doc!r}"))


# ---------------------------------------------------------------------------
# ...and structural serializers must REFUSE, not merely hide.
# repr=False stops display only; asdict/astuple/vars/pickle ignore it, and an
# overview's windows hold the whole body between them.
# ---------------------------------------------------------------------------

def test_structural_serializers_refuse_the_document():
    import dataclasses
    import pickle

    doc = _sentinel_doc()
    with pytest.raises(TypeError):
        dataclasses.asdict(doc)
    with pytest.raises(TypeError):
        dataclasses.astuple(doc)
    with pytest.raises(TypeError):
        vars(doc)
    with pytest.raises(osd.SourceDocumentError):
        pickle.dumps(doc)


def test_the_transient_artifacts_are_immutable():
    doc = _sentinel_doc()
    with pytest.raises(osd.SourceDocumentError):
        doc.canonical_body = "replaced"
    with pytest.raises(osd.SourceDocumentError):
        del doc.body_sha256


# ---------------------------------------------------------------------------
# identity cannot be skipped
# ---------------------------------------------------------------------------

def test_a_document_cannot_be_built_without_identity():
    with pytest.raises(osd.SourceDocumentError):
        osd.SourceDocument(canonical_body="x", body_sha256="",
                           normalization_version="")


def test_a_document_refuses_a_hash_that_is_not_its_own():
    with pytest.raises(osd.SourceDocumentError):
        osd.SourceDocument(
            canonical_body="the traveller returned",
            body_sha256=osd.canonical_body_sha256("a different body"),
            normalization_version=osd.NORMALIZATION_VERSION,
        )


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
