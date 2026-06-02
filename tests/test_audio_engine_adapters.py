"""Tests for the registered audio-engine adapters (real registry population).

Scope: music + announcer + character voices. SFX is intentionally out of the
v2 audio-engine infrastructure.
"""
import pytest

from nodes import _otr_audio_engines as AE


def test_char_voice_default_is_bark():
    assert AE.engines_for_role("char_voice")[0] == "bark"


def test_announcer_default_is_kokoro():
    assert AE.engines_for_role("announcer_voice")[0] == "kokoro"


def test_music_default_is_musicgen():
    assert AE.engines_for_role("music")[0] == "musicgen"


def test_char_voice_pool_has_optins():
    pool = set(AE.engines_for_role("char_voice"))
    assert {"bark", "chatterbox", "indextts2"} <= pool


def test_music_role_has_stable_audio():
    assert "stable_audio_music" in AE.engines_for_role("music")


def test_no_sfx_role_registered():
    # SFX is deliberately excluded from the v2 audio-engine infrastructure.
    assert AE.engines_for_role("sfx") == []


def test_chatterbox_serves_both_voice_roles():
    assert "chatterbox" in AE.engines_for_role("char_voice")
    assert "chatterbox" in AE.engines_for_role("announcer_voice")


def test_optin_flags_and_licensing():
    cb = AE.get_engine("chatterbox")
    assert cb.requires_flag == "OTR_ENABLE_CHATTERBOX"
    assert cb.commercial_clean is True
    idx = AE.get_engine("indextts2")
    assert idx.requires_flag == "OTR_ENABLE_INDEXTTS2"
    assert idx.commercial_clean is False  # Bilibili non-commercial
    sa = AE.get_engine("stable_audio_music")
    assert sa.requires_flag == "OTR_ENABLE_STABLE_AUDIO"
    assert sa.commercial_clean is True


def test_assert_usable_optin_off_fails_closed(monkeypatch):
    # Fail closed (C-6): opt-in char-voice engines raise GATED_BY_FLAG when
    # their flag is off -- they are never silently swapped for bark.
    monkeypatch.delenv("OTR_ENABLE_CHATTERBOX", raising=False)
    monkeypatch.delenv("OTR_ENABLE_INDEXTTS2", raising=False)
    for name in ("chatterbox", "indextts2"):
        with pytest.raises(AE.EngineUnusable) as ei:
            AE.assert_usable(name, "char_voice")
        assert ei.value.reason is AE.EngineUsabilityReason.GATED_BY_FLAG


def test_assert_usable_optin_on_runs(monkeypatch):
    monkeypatch.setenv("OTR_ENABLE_CHATTERBOX", "1")
    assert AE.assert_usable("chatterbox", "char_voice") == "chatterbox"


def test_legacy_engines_are_batch_interface():
    for n in ("bark", "kokoro", "musicgen"):
        assert AE.get_engine(n).interface == "batch"


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
