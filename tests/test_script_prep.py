"""Tests for the engine-neutral spoken-text preparation + per-engine hooks."""
from nodes._otr_audio_engines import get_engine
from nodes._otr_script_prep import clean_spoken_text


def test_strips_parentheticals():
    assert clean_spoken_text("Hello (sighs) there.") == "Hello there."


def test_strips_bracket_tags():
    assert clean_spoken_text("Wait [SFX: door] now.") == "Wait now."


def test_strips_speaker_prefix():
    assert clean_spoken_text("HAYES VANCE: I'm worried.") == "I'm worried."


def test_keeps_plain_sentence_with_leading_capital():
    # No false-positive speaker strip on a normal sentence.
    assert clean_spoken_text("I'm worried about it.") == "I'm worried about it."


def test_collapses_whitespace():
    assert clean_spoken_text("a    b   c") == "a b c"


def test_deterministic_and_idempotent():
    s = "DOCTOR: (whispering) The reactor [pause] is failing."
    once = clean_spoken_text(s)
    assert once == clean_spoken_text(s)
    assert clean_spoken_text(once) == once


def test_chatterbox_prepare_text_returns_clean():
    eng = get_engine("chatterbox")
    assert eng.prepare_text("MIRA: (gasps) Run!", {"afraid": 0.9}) == "Run!"


def test_indextts2_prepare_text_strips_direction():
    eng = get_engine("indextts2")
    out = eng.prepare_text("XAN: (sad) Goodbye.", {})
    assert out == "Goodbye."
    assert "(" not in out


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
