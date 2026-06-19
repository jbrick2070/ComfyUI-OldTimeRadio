"""default_clean voice bank -- the commercial-clean cast selector (2026-06-18
voice-engine roundtable). Selecting voice_bank=default_clean routes the cast
char engine to chatterbox (MIT), then dia (Apache); indextts2 (non-commercial)
is excluded. voice_bank=default is unchanged (indextts2 first = best quality)."""
from __future__ import annotations

from types import SimpleNamespace

from nodes.cast_lock import CastLock, _VOICE_BANKS


def _entries(*engines):
    return [SimpleNamespace(engine=e) for e in engines]


def test_default_clean_is_a_selectable_bank():
    assert "default_clean" in _VOICE_BANKS
    assert "default" in _VOICE_BANKS                 # personal-use default kept


def test_default_clean_routes_to_chatterbox():
    # all three cloners have refs; default_clean excludes indextts2 -> chatterbox.
    eng = CastLock._resolve_char_engine(
        "default_clean", _entries("indextts2", "chatterbox", "dia"))
    assert eng == "chatterbox"


def test_default_still_routes_to_indextts2():
    eng = CastLock._resolve_char_engine(
        "default", _entries("indextts2", "chatterbox", "dia"))
    assert eng == "indextts2"


def test_default_clean_falls_to_dia_without_chatterbox():
    eng = CastLock._resolve_char_engine(
        "default_clean", _entries("indextts2", "dia"))
    assert eng == "dia"


def test_default_clean_excludes_non_commercial_indextts2():
    # indextts2 alone under default_clean -> no clean engine resolves (it must
    # NEVER be the commercial-clean cast pick).
    eng = CastLock._resolve_char_engine("default_clean", _entries("indextts2"))
    assert eng is None
