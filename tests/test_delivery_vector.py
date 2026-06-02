"""Tests for the deterministic per-line delivery (emotion) vector."""
from nodes._otr_delivery_vector import (
    DELIVERY_TABLE_VERSION,
    EMOTIONS,
    deterministic_delivery_vector,
    stamp_delivery_vectors,
)


def test_vector_deterministic():
    a = deterministic_delivery_vector("Warning! The reactor is failing!")
    b = deterministic_delivery_vector("Warning! The reactor is failing!")
    assert a == b


def test_vector_dims_and_range():
    v = deterministic_delivery_vector("Help! Run! No escape!", 0.8)
    assert set(v.keys()) == set(EMOTIONS)
    assert all(0.0 <= x <= 1.0 for x in v.values())


def test_neutral_line_is_calm_dominant():
    v = deterministic_delivery_vector("The report is on the desk.")
    others = max(v[e] for e in EMOTIONS if e != "calm")
    assert v["calm"] >= others


def test_fear_line_raises_afraid():
    v = deterministic_delivery_vector("Run! Danger! Hide!", 0.9)
    assert v["afraid"] > v["happy"]


def test_stamp_is_additive_and_versioned():
    led = {"lines": [{"speaker_role": "character", "text": "Hello there.", "char_id": "c1"}]}
    out = stamp_delivery_vectors(led)
    d = out["lines"][0]["delivery"]
    assert d["version"] == DELIVERY_TABLE_VERSION
    assert set(d["emotion_vector"].keys()) == set(EMOTIONS)
    # original keys preserved (additive)
    assert out["lines"][0]["text"] == "Hello there."


def test_stamp_deterministic():
    a = {"lines": [{"text": "It is over. Goodbye, old friend...", "char_id": "c1"}]}
    b = {"lines": [{"text": "It is over. Goodbye, old friend...", "char_id": "c1"}]}
    assert stamp_delivery_vectors(a) == stamp_delivery_vectors(b)


def test_stamp_handles_empty_ledger():
    assert stamp_delivery_vectors({}) == {}
    assert stamp_delivery_vectors({"lines": []}) == {"lines": []}


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
