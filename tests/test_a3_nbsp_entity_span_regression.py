"""A-3: the no-break-space ENTITY stops making a literal quote non-literal.

Built from the real articles, not from an invented fixture. Both strings below
are the production bytes read out of the 2026-07-29 campaign logs, where P0
rejected a quote that matched its source character for character except that
the feed carried `&nbsp;` and the model wrote the space a reader sees:

  otr_headless_65452 (MIT Genesis Mission)
    source: ...Department of Energy's (DOE)&nbsp;Genesis Mission...
    quote : ...Department of Energy's (DOE) Genesis Mission...
  otr_headless_65212 (open-source models)
    source: ...in a certain artistic style.&nbsp;But these mode
    quote : ...in a certain artistic style. But these models

Two of the fifteen P0 deaths in that campaign were this and nothing else. The
article's curly apostrophe is written as a plain one so this file stays
ASCII-only -- it plays no part in this failure, the entity is the whole of it --
and each fixture stops at a sentence end so its quote fits the 240-char schema
cap.

What this file pins:

  FIRES          -- the campaign's own quote now validates end to end, from
                    payload admission through the P0 projection to
                    _validate_fact_index.
  COUNTERFACTUAL -- against the UN-normalized payload the same quote is still
                    rejected as a non-literal source span, so the entity was
                    the entire cause and the fix is not decorating a pass that
                    already worked.
  DIGEST         -- normalization happens BEFORE the digest, and a payload
                    differing only by this entity now hashes identically, so
                    no accepted offset can shift under a re-admission.
  NARROWNESS     -- every other entity survives untouched. Widening the set
                    redefines the coordinate system source_digest pins, which
                    is an operator decision.

No GPU, no network. UTF-8, no BOM, ASCII-only source.
"""
from __future__ import annotations

import pytest

from nodes._otr_scifi_codex import (
    FactIndexV4,
    _digest,
    _normalize_span_source_text,
    _p0_evidence_projection,
    _SPAN_SOURCE_FIELDS,
    _validate_fact_index,
    validate_payload_envelope,
)


GENESIS_SOURCE = (
    "MIT researchers are set to contribute to the U.S. Department of "
    "Energy's (DOE)&nbsp;Genesis Mission, with 15 collaborative projects "
    "among those selected for funding under Genesis Phase I."
)
GENESIS_QUOTE = (
    "MIT researchers are set to contribute to the U.S. Department of "
    "Energy's (DOE) Genesis Mission, with 15 collaborative projects "
    "among those selected for funding under Genesis Phase I."
)
OPEN_MODELS_SOURCE = (
    "Many open-source models are now available online for anyone to adapt "
    "for their task, such as generating product renderings in a certain "
    "artistic style.&nbsp;But these models can be tampered with."
)
OPEN_MODELS_QUOTE = (
    "Many open-source models are now available online for anyone to adapt "
    "for their task, such as generating product renderings in a certain "
    "artistic style. But these models can be tampered with."
)
RESOLVED = {"seed_source": "rss_fetch", "target_words": 45}


def _payload(full_text: str) -> dict:
    return {
        "headline": "A headline with no entity in it",
        "summary": "A summary with no entity in it",
        "full_text": full_text,
        "source": "MIT News",
        "date": "2026-07-29",
        "link": "https://example.invalid/story",
        "seed_text": "A distinct premise line, so the projection keeps full_text",
    }


def _index(quote: str, digest: str) -> FactIndexV4:
    """A fresh index per call: _span_ok MUTATES the span it validates."""
    return FactIndexV4.model_validate({
        "facts": [{
            "fact_id": "F01",
            "claim": "The campaign's own P0 claim, shortened to the fixture.",
            "source_spans": [{
                "field": "full_text",
                "start": 0,
                "end": len(quote),
                "quote": quote,
            }],
        }],
        "entities": [],
        "numbers": [],
        "tone": "measured",
        "payload_sha256": digest,
    })


def _admit(full_text: str):
    envelope, _steer = validate_payload_envelope(_payload(full_text), RESOLVED)
    projected, allowed = _p0_evidence_projection(envelope)
    return envelope, projected, allowed


@pytest.mark.parametrize(
    "source, quote",
    [
        pytest.param(GENESIS_SOURCE, GENESIS_QUOTE, id="genesis_mission"),
        pytest.param(OPEN_MODELS_SOURCE, OPEN_MODELS_QUOTE, id="open_source_models"),
    ],
)
def test_the_campaigns_own_quote_validates_after_admission(source, quote):
    """FIRES: the live rejection, end to end, now accepted."""
    envelope, projected, allowed = _admit(source)

    error = _validate_fact_index(
        _index(quote, envelope.source_digest),
        projected,
        allowed_source_fields=allowed,
        expected_payload_sha256=envelope.source_digest,
    )

    assert error is None
    assert "&nbsp;" not in projected["full_text"]
    assert quote in projected["full_text"]


@pytest.mark.parametrize(
    "source, quote",
    [
        pytest.param(GENESIS_SOURCE, GENESIS_QUOTE, id="genesis_mission"),
        pytest.param(OPEN_MODELS_SOURCE, OPEN_MODELS_QUOTE, id="open_source_models"),
    ],
)
def test_the_same_quote_is_still_rejected_against_the_undecoded_source(source, quote):
    """COUNTERFACTUAL: the entity was the entire cause of the rejection."""
    error = _validate_fact_index(
        _index(quote, "unused"),
        {"full_text": source},
        allowed_source_fields=frozenset({"full_text"}),
    )

    assert error is not None
    assert "non-literal source span" in error


def test_normalization_happens_before_the_digest():
    """DIGEST: the coordinate system is hashed AFTER it is cleaned, not before."""
    payload = _payload(GENESIS_SOURCE)
    envelope, _steer = validate_payload_envelope(payload, RESOLVED)

    expected = {
        key: (_normalize_span_source_text(value)
              if key in _SPAN_SOURCE_FIELDS else value)
        for key, value in payload.items()
    }

    assert envelope.source_digest == _digest(expected)
    assert envelope.source_digest != _digest(payload)


def test_a_payload_differing_only_by_the_entity_hashes_identically():
    """DIGEST: re-admitting the same story cannot shift an accepted offset."""
    with_entity, _steer = validate_payload_envelope(
        _payload(GENESIS_SOURCE), RESOLVED,
    )
    with_space, _steer2 = validate_payload_envelope(
        _payload(GENESIS_QUOTE), RESOLVED,
    )

    assert with_entity.source_digest == with_space.source_digest
    assert with_entity.payload.full_text == with_space.payload.full_text


@pytest.mark.parametrize(
    "spelling",
    ["&nbsp;", "&#160;", "&#xA0;", "&#Xa0;", "&#xa0;"],
)
def test_every_spelling_of_the_no_break_space_becomes_one_space(spelling):
    assert _normalize_span_source_text("style." + spelling + "But") == "style. But"


@pytest.mark.parametrize(
    "entity",
    ["&amp;", "&mdash;", "&#8212;", "&quot;", "&lt;", "&gt;", "&apos;", "&#39;"],
)
def test_no_other_entity_is_decoded(entity):
    """NARROWNESS: widening this set moves every source offset in the build."""
    text = "before" + entity + "after"

    assert _normalize_span_source_text(text) == text


def test_entity_free_text_is_normalized_exactly_as_before():
    """CONTROL: behaviour-neutral for every payload with no entity in it."""
    assert _normalize_span_source_text("  a\n\t b  \r\n c ") == "a b c"
    assert _normalize_span_source_text("") == ""
    assert _normalize_span_source_text(None) == ""


def test_a_literal_no_break_space_was_never_the_problem():
    """CONTROL: U+00A0 is whitespace to `\\s` and always collapsed.

    Built with ``chr(160)`` so the character under test never appears in this
    file: an ASCII-only source cannot carry the byte it is asserting about.
    """
    assert _normalize_span_source_text("style." + chr(160) + "But") == "style. But"
