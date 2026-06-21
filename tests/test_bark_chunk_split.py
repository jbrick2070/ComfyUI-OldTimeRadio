"""Bark artifact B3 -- chunk-split hardening for overlong sentences.

Before B3, _chunk_text_for_bark returned a single sentence longer than max_len
WHOLE (a runaway clause bark could over-generate on). B3 splits it on clause
punctuation (',' ';' ':') first, then whitespace -- never mid-word.

Pure / CPU. UTF-8, no BOM, SFW.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes._otr_bark_lib import (  # noqa: E402
    _chunk_text_for_bark,
    _pack_words,
    _split_long_sentence,
)


class TestChunkSplit:
    def test_short_text_unchanged(self):
        assert _chunk_text_for_bark("Hello there.") == ["Hello there."]

    def test_each_chunk_within_max_len(self):
        # 50 comma-separated clauses, no sentence stops -> ONE overlong sentence.
        text = ", ".join(f"clause number {i}" for i in range(50))
        chunks = _chunk_text_for_bark(text, max_len=60)
        assert len(chunks) > 1
        for c in chunks:
            assert len(c) <= 60, f"chunk exceeds max_len: {c!r}"

    def test_no_punctuation_runaway_is_split(self):
        # A long no-punctuation string was previously returned whole.
        text = "word " * 80  # 400 chars, no sentence/clause punctuation
        chunks = _chunk_text_for_bark(text.strip(), max_len=50)
        assert len(chunks) > 1
        for c in chunks:
            assert len(c) <= 50

    def test_never_splits_mid_word(self):
        text = "supercalifragilistic " * 20
        chunks = _chunk_text_for_bark(text.strip(), max_len=40)
        rejoined = " ".join(chunks).split()
        assert all(w == "supercalifragilistic" for w in rejoined)

    def test_single_overlong_word_kept_whole(self):
        word = "x" * 100
        chunks = _chunk_text_for_bark(word, max_len=40)
        # cannot break a word -> kept whole as its own chunk
        assert chunks == [word]

    def test_clause_split_preferred_over_word_split(self):
        text = "alpha beta gamma, delta epsilon zeta, eta theta iota"
        out = _split_long_sentence(text, max_len=20)
        # splits at the commas, each clause intact
        assert "alpha beta gamma," in out[0]
        for c in out:
            assert len(c) <= 22  # clause + trailing delimiter

    def test_pack_words_short_final_chunk(self):
        out = _pack_words("aa bb cc dd ee", max_len=5)
        assert out == ["aa bb", "cc dd", "ee"]

    def test_empty_safe(self):
        assert _split_long_sentence("", 10) == []
        assert _pack_words("", 10) == []


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
