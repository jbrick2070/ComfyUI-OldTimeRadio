"""tests/test_voice_resolver.py — voice spec parsing tests."""
from __future__ import annotations

import pytest

from nodes._otr_voice_resolver import KNOWN_ENGINES, VoiceSpec, parse_voice_spec


def test_parse_bark_spec():
    s = parse_voice_spec("bark:v2/en_speaker_5")
    assert s == VoiceSpec(engine="bark", preset="v2/en_speaker_5")


def test_parse_kokoro_spec():
    s = parse_voice_spec("kokoro:bm_fable")
    assert s == VoiceSpec(engine="kokoro", preset="bm_fable")


def test_parse_lowercases_engine():
    s = parse_voice_spec("BARK:v2/en_speaker_5")
    assert s.engine == "bark"


def test_preset_can_contain_colons():
    s = parse_voice_spec("xtts:speaker_3:variant_a")
    assert s.engine == "xtts"
    assert s.preset == "speaker_3:variant_a"


def test_known_engines_contains_canonical_set():
    assert {"bark", "kokoro", "cosyvoice", "xtts", "piper"} <= KNOWN_ENGINES


def test_unknown_engine_passes_through_for_forward_compat():
    s = parse_voice_spec("future_engine:some_preset")
    assert s.engine == "future_engine"
    assert s.preset == "some_preset"


def test_missing_colon_raises():
    with pytest.raises(ValueError, match="must be 'engine:preset' form"):
        parse_voice_spec("bark_v2_en_speaker_5")


def test_empty_string_raises():
    with pytest.raises(ValueError):
        parse_voice_spec("")


def test_empty_engine_raises():
    with pytest.raises(ValueError, match="empty engine"):
        parse_voice_spec(":bm_fable")


def test_empty_preset_raises():
    with pytest.raises(ValueError, match="empty preset"):
        parse_voice_spec("bark:")


def test_to_str_roundtrip():
    s = parse_voice_spec("bark:v2/en_speaker_5")
    assert s.to_str() == "bark:v2/en_speaker_5"


def test_voice_spec_is_frozen():
    s = VoiceSpec(engine="bark", preset="v2/en_speaker_5")
    with pytest.raises((AttributeError, Exception)):  # FrozenInstanceError on dataclass
        s.engine = "kokoro"  # type: ignore[misc]
