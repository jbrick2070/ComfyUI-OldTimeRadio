"""Chunk 2b: the transient source document reaches the writer, and nothing
serializes it.

Chunk 1 built the document; chunk 2a derived a sound world from it. This pins
the wiring between them -- and the invariant that matters more than the
feature, which is that a 25,000-word body rides in memory only.
"""
from __future__ import annotations

import json

import pytest

from nodes import _otr_source_document as osd
from nodes import _otr_source_payload as osp
from nodes import _otr_source_world as world


def _payload() -> dict:
    return {
        "headline": "h", "summary": "s", "full_text": "f", "source": "src",
        "date": "1895", "link": "u", "seed_text": "seed",
    }


def test_the_writer_normalizer_carries_the_document():
    doc = osd.build_source_document("The ship stood off the harbour.")
    result = osp.SourceFetchResult(
        payload=_payload(), source_meta={"title": "t"}, source_document=doc)
    payload, meta, rights, document = (
        osp.normalize_fetch_result_with_document(result, origin="test"))
    assert document is doc
    assert payload["seed_text"] == "seed"
    assert "source_document" not in meta
    assert "source_document" not in rights


def test_lanes_without_a_document_get_none_not_a_crash():
    # The original and custom-premise lanes have no source work, and the
    # snapshot envelope carries only the capped payload.
    out = osp.normalize_fetch_result_with_document(_payload(), origin="test")
    assert out[3] is None


def test_the_document_never_lands_in_serializable_sidecars():
    doc = osd.build_source_document("a secret body " * 200)
    result = osp.SourceFetchResult(
        payload=_payload(), source_meta={"title": "t"},
        source_rights={"license_status": "public_domain_us"},
        source_document=doc,
    )
    _, meta, rights, _ = osp.normalize_fetch_result_with_document(
        result, origin="test")
    # Both sidecars are copied into durable ledger metadata; they must stay
    # JSON-clean and body-free.
    assert "secret body" not in json.dumps(meta)
    assert "secret body" not in json.dumps(rights)


# ---------------------------------------------------------------------------
# the end-to-end shape the writer relies on
# ---------------------------------------------------------------------------

def test_a_document_yields_a_source_sound_world_and_a_body_free_receipt():
    body = "The ship stood off the harbour, and the tide ran under her deck."
    doc = osd.build_source_document(body, source_ref="sea:1")
    derived = world.derive_source_sound_world(doc.canonical_body)
    receipt = world.sound_world_receipt(
        doc.canonical_body, source_ref=doc.source_ref)

    assert derived and derived != world.NEUTRAL_PERIOD_SOUND_WORLD
    assert receipt["source_ref"] == "sea:1"
    # The receipt is stamped into meta.story_contract, so it must carry no
    # source text and must survive a JSON round trip.
    assert "harbour" not in json.dumps(receipt)
    assert json.loads(json.dumps(receipt)) == receipt


def test_a_source_with_no_recognisable_world_still_produces_a_stampable_receipt():
    doc = osd.build_source_document("He considered the proposition at length.")
    receipt = world.sound_world_receipt(doc.canonical_body)
    assert receipt["used_neutral_default"] is True
    assert json.loads(json.dumps(receipt)) == receipt


def test_writer_threading_is_wired_at_the_real_call_site():
    # Cheap structural pin: the writer must take the 4-value normalizer, keep
    # the document off source_meta, and pass a derived world into the
    # contract. A future refactor that drops any of these silently reverts
    # the whole chunk, and no unit test of the helpers would notice.
    import pathlib

    src = (pathlib.Path(__file__).resolve().parents[1]
           / "nodes" / "OTR_LedgerScriptWriter.py").read_text(encoding="utf-8")
    assert "normalize_fetch_result_with_document(" in src
    assert '"source_document": source_document,' in src
    assert "source_sound_world=_source_sound_world," in src
    assert 'str(meta.get("style_pool_class") or "") == "adaptation"' in src


def test_the_canon_palette_splits_a_derived_world_correctly():
    # The canon splitter took commas only. A derived world joins per-element
    # phrases with "; " and each phrase carries internal commas, so a
    # comma-only split fused one element's tail onto the next element's head
    # ("a bell over open water; night quiet" as ONE palette entry) on
    # essentially every adaptation episode.
    import re

    derived = world.derive_source_sound_world(
        "The ship stood off the harbour by night, and the tide ran. "
        "The ship returned at night on the tide.")
    palette = [p.strip() for p in re.split(r"\s*[;,]\s*", derived) if p.strip()]
    assert palette
    assert not any(";" in entry for entry in palette)


def test_the_writer_uses_the_two_delimiter_split():
    import pathlib

    src = (pathlib.Path(__file__).resolve().parents[1]
           / "nodes" / "OTR_LedgerScriptWriter.py").read_text(encoding="utf-8")
    assert 're.split(r"\\s*[;,]\\s*", str(contract.sound_world))' in src


def test_the_shakespeare_fetcher_carries_a_document_too():
    # shakespeare is a style_pool_class "adaptation" bank, so it is gated into
    # source-derived grounding -- but its fetcher passed no document, so it
    # silently kept the drawn palette while the code claimed to have fixed
    # "the adaptation lanes".
    import pathlib

    src = (pathlib.Path(__file__).resolve().parents[1]
           / "nodes" / "_otr_shakespeare_sources.py").read_text(encoding="utf-8")
    assert "source_document=source_document_from_text(" in src
    assert "def normalize_shakespeare_body(" in src


def test_shakespeare_normalization_is_uncapped():
    from nodes import _otr_shakespeare_sources as shk

    body = "SPEAKER  a line of verse here. " * 900
    full = shk.normalize_shakespeare_body(body)
    capped = shk.canonicalize_shakespeare_text(body)
    assert len(full) > shk.INTERPRETER_TEXT_WINDOW
    assert len(capped) <= shk.INTERPRETER_TEXT_WINDOW
    assert full.startswith(capped)


def test_the_document_is_not_stamped_into_meta_source_meta():
    # meta["source_meta"] is a copy of resolved["source_meta"]; the document
    # rides its own key precisely so that copy cannot pick it up.
    import pathlib

    src = (pathlib.Path(__file__).resolve().parents[1]
           / "nodes" / "OTR_LedgerScriptWriter.py").read_text(encoding="utf-8")
    assert 'meta["source_meta"] = dict(resolved["source_meta"])' in src
    assert 'meta["source_document"]' not in src
