"""Wave 0: prepare_text golden examples. Engine-neutral spoken-text prep on top
of clean_spoken_text -- strips asterisks / music notes / delivery tags, collapses
ellipsis to one pause, keeps . , ?. Pure + deterministic (C7)."""
import pytest

from nodes._otr_script_prep import PREPARE_TEXT_VERSION, prepare_text

GOLDEN = [
    ("[whispers] Hello, world...", "Hello, world..."),
    ("NARRATOR: The *end* is near?", "The end is near?"),
    ("Music ♪♪ plays", "Music plays"),
    ("Wait.... what??", "Wait... what??"),
    ("(softly) okay", "okay"),
    ("A… B", "A... B"),
    ("Hi, there. Ok?", "Hi, there. Ok?"),
]


@pytest.mark.parametrize("raw,expected", GOLDEN)
def test_prepare_text_golden(raw, expected):
    assert prepare_text(raw) == expected


@pytest.mark.parametrize("raw,_expected", GOLDEN)
def test_prepare_text_idempotent(raw, _expected):
    once = prepare_text(raw)
    assert prepare_text(once) == once


def test_prepare_text_keeps_sentence_punctuation():
    assert prepare_text("Yes, I see. Do you?") == "Yes, I see. Do you?"


def test_prepare_text_is_pure_no_side_effects():
    src = "  [beat] *whoa*  "
    a = prepare_text(src)
    b = prepare_text(src)
    assert a == b == "whoa"


def test_prepare_text_empty_safe():
    assert prepare_text("") == ""
    assert prepare_text(None) == ""  # clean_spoken_text tolerates None


def test_version_constant():
    assert PREPARE_TEXT_VERSION == "1"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
