"""default_clean voice bank -- the commercial-clean cast selector (2026-06-18
voice-engine roundtable). Selecting voice_bank=default_clean routes the cast
char engine to chatterbox (MIT), then dia (Apache); indextts2 (non-commercial)
is excluded. voice_bank=default is unchanged (indextts2 first = best quality)."""
from __future__ import annotations

from types import SimpleNamespace

from nodes.cast_lock import CastLock


def _entries(*engines):
    return [SimpleNamespace(engine=e) for e in engines]


def test_default_clean_still_routes_through_the_engine_profiles():
    """CastLock has no voice_bank widget. leftover lock(voice_bank=) ids
    still have to name banks the chatterbox / indextts2 profiles allow."""
    from nodes._otr_engine_profiles import require_resolver

    resolver = require_resolver()
    chatter = resolver.profile_for("char_voice", "chatterbox")
    idx = resolver.profile_for("char_voice", "indextts2")
    assert chatter is not None and "default_clean" in chatter.allowed_voice_banks
    assert idx is not None and "default" in idx.allowed_voice_banks


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


def test_explicit_elevenlabs_routes_with_cloud_bank():
    eng = CastLock._resolve_char_engine(
        "elevenlabs_cloud", _entries("indextts2", "cloud_elevenlabs"),
        requested_engine="cloud_elevenlabs")
    assert eng == "cloud_elevenlabs"


def test_explicit_google_tts_routes_with_google_bank():
    eng = CastLock._resolve_char_engine(
        "google_tts", _entries("indextts2", "google_tts"),
        requested_engine="google_tts")
    assert eng == "google_tts"
