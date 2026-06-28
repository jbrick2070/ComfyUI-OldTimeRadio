"""Story-quality 3.7 -- register divergence via NEAR-duplicate signature collision.

Signatures already shipped, but the EXACT-only diversify_speech_signatures kept
near-duplicates ("measured, precise, weary" vs "measured, concise"), so the two
principals read identically. The fix tightens the collision to a token-overlap
threshold. CPU; deterministic. UTF-8, no BOM, ASCII-only source.
"""
from __future__ import annotations

from nodes._otr_casting import (
    speech_signature_overlap,
    diversify_speech_signatures,
)


# --- speech_signature_overlap -----------------------------------------------
def test_overlap_identical_is_one():
    assert speech_signature_overlap("measured, precise", "measured, precise") == 1.0


def test_overlap_shared_dominant_trait():
    # "measured" shared of the 2-token shorter set -> 0.5
    assert speech_signature_overlap(
        "measured, precise, weary", "measured, concise") == 0.5


def test_overlap_disjoint_is_zero():
    assert speech_signature_overlap("clipped, procedural", "warm, rambling") == 0.0


def test_overlap_empty_is_zero():
    assert speech_signature_overlap("", "anything here") == 0.0
    assert speech_signature_overlap("anything here", None) == 0.0


# --- diversify_speech_signatures near-dup collision -------------------------
def test_diversify_reassigns_near_duplicate():
    cast = [
        {"char_id": "c01", "speech_signature": "measured, precise, weary"},
        {"char_id": "c02", "speech_signature": "measured, concise"},
    ]
    diversify_speech_signatures(cast, seed=0)
    a, b = cast[0]["speech_signature"], cast[1]["speech_signature"]
    assert speech_signature_overlap(a, b) < 0.5, (a, b)
    assert b != "measured, concise"           # the near-dup was reassigned


def test_diversify_keeps_genuinely_distinct_registers():
    cast = [
        {"char_id": "c01", "speech_signature": "clipped, procedural"},
        {"char_id": "c02", "speech_signature": "warm, rambling"},
    ]
    diversify_speech_signatures(cast, seed=0)
    assert cast[0]["speech_signature"] == "clipped, procedural"
    assert cast[1]["speech_signature"] == "warm, rambling"


def test_diversify_is_deterministic_per_seed():
    def _mk():
        return [
            {"char_id": "c01", "speech_signature": "measured, precise, weary"},
            {"char_id": "c02", "speech_signature": "measured, concise"},
            {"char_id": "c03", "speech_signature": "measured, dry"},
        ]
    c1, c2 = _mk(), _mk()
    diversify_speech_signatures(c1, seed=7)
    diversify_speech_signatures(c2, seed=7)
    assert ([r["speech_signature"] for r in c1]
            == [r["speech_signature"] for r in c2])
    # all three end mutually distinct (no pair near-duplicate)
    sigs = [r["speech_signature"] for r in c1]
    for i in range(len(sigs)):
        for j in range(i + 1, len(sigs)):
            assert speech_signature_overlap(sigs[i], sigs[j]) < 0.5, (sigs[i], sigs[j])
