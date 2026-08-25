"""Tests for the engine-profile resolver + YAML (piece 8 / D5)."""
import hashlib
import pathlib

import pytest

from nodes._otr_audio_engines import EngineUsabilityReason, EngineUnusable, engines_for_role
from nodes import _otr_engine_profiles as EP

PROFILES_YAML = (
    pathlib.Path(__file__).resolve().parent.parent / "config" / "audio_engine_profiles.yaml"
)


def test_load_resolver_and_source_hash():
    r = EP.load_resolver()
    assert r is not None
    text = PROFILES_YAML.read_text(encoding="utf-8")
    assert r.source_sha256 == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert set(r.profile_ids()) == {
        "char_bark_v1", "char_chatterbox_v1", "char_indextts2_v1", "char_dia_v1",
        "char_kokoro_v1",
        "announcer_kokoro_v1", "announcer_chatterbox_v1", "announcer_dia_v1",
        "announcer_bark_v1",
        "music_musicgen_v1", "music_stable_audio_v1", "music_stable_audio_3_v1",
        # cloud ElevenLabs voice (cloud-audio S1, 2026-07-03)
        "char_elevenlabs_cloud_v1", "announcer_elevenlabs_cloud_v1",
        # direct Google Gemini TTS (BYO API key, explicit-selection-only)
        "char_google_tts_v1", "announcer_google_tts_v1",
        # cloud Sonilo music (cloud-audio S5, 2026-07-03)
        "music_sonilo_cloud_v1",
        # direct Google Lyria music (BYO API key, explicit-selection-only)
        "music_google_lyria_v1",
    }


def test_resolver_is_cached_by_content():
    assert EP.load_resolver() is EP.load_resolver()


def test_every_registered_engine_has_a_profile():
    r = EP.load_resolver()
    for role in ("char_voice", "announcer_voice", "music"):
        for engine in engines_for_role(role):
            assert r.profile_for(role, engine) is not None, (role, engine)


def test_resolve_legacy_defaults_without_flags():
    r = EP.load_resolver()
    assert r.resolve_casting_plan(role="char_voice", engine="bark").profile_id == "char_bark_v1"
    assert r.resolve_casting_plan(role="announcer_voice", engine="kokoro").profile_id == "announcer_kokoro_v1"
    assert r.resolve_casting_plan(role="music", engine="musicgen").profile_id == "music_musicgen_v1"


def test_resolve_optin_selectable_no_flag(monkeypatch):
    # C6 -- registry IS the menu: an opt-in engine resolves with NO flag gate
    # (validation is the operator's MANUAL process, never a code gate).
    monkeypatch.delenv("OTR_ENABLE_CHATTERBOX", raising=False)
    monkeypatch.delenv("OTR_ENABLE_STABLE_AUDIO", raising=False)
    r = EP.load_resolver()
    p = r.resolve_casting_plan(role="char_voice", engine="chatterbox",
                               voice_bank="default")
    assert p.profile_id == "char_chatterbox_v1"
    p2 = r.resolve_casting_plan(role="music", engine="stable_audio_music")
    assert p2.profile_id  # resolves without a flag gate


def test_resolve_optin_succeeds_when_flag_on(monkeypatch):
    monkeypatch.setenv("OTR_ENABLE_CHATTERBOX", "1")
    r = EP.load_resolver()
    p = r.resolve_casting_plan(role="char_voice", engine="chatterbox", voice_bank="default")
    assert p.profile_id == "char_chatterbox_v1"


def test_ref_engine_rejects_legacy_bark_bank(monkeypatch):
    monkeypatch.setenv("OTR_ENABLE_CHATTERBOX", "1")
    r = EP.load_resolver()
    with pytest.raises(EngineUnusable) as ei:
        r.resolve_casting_plan(role="char_voice", engine="chatterbox", voice_bank="bark_legacy")
    assert ei.value.reason is EngineUsabilityReason.INCOMPATIBLE_PROFILE
    # bark itself accepts its legacy preset bank.
    assert r.resolve_casting_plan(role="char_voice", engine="bark", voice_bank="bark_legacy")


def test_unknown_profile_id_is_malformed_config():
    r = EP.load_resolver()
    with pytest.raises(EngineUnusable) as ei:
        r.get("no_such_profile")
    assert ei.value.reason is EngineUsabilityReason.MALFORMED_CONFIG


def test_missing_token_raises_in_generate_not_input_types(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    r = EP.load_resolver()
    sa = r.get("music_stable_audio_v1")
    with pytest.raises(EngineUnusable) as ei:
        EP.assert_token_for_profile(sa)
    assert ei.value.reason is EngineUsabilityReason.MISSING_HF_TOKEN
    # With a token present it passes.
    monkeypatch.setenv("HF_TOKEN", "x")
    EP.assert_token_for_profile(sa)
    # A non-gated profile never needs a token.
    monkeypatch.delenv("HF_TOKEN", raising=False)
    EP.assert_token_for_profile(r.get("char_bark_v1"))


def test_legacy_first_engines_pure_and_legacy_first():
    assert EP.legacy_first_engines("char_voice")[0] == "indextts2"  # PROMOTED 2026-06-04
    assert EP.legacy_first_engines("announcer_voice")[0] == "kokoro"
    assert EP.legacy_first_engines("music")[0] == "stable_audio_3"  # PROMOTED 2026-06-03
    assert EP.legacy_first_engines("char_voice")[-1] == "google_tts"
    assert "dia" in EP.legacy_first_engines("announcer_voice")
    assert EP.legacy_first_engines("announcer_voice")[-1] == "bark"
    assert EP.legacy_first_engines("nonexistent_role") == []


def test_direct_api_profiles_are_explicit_selection_only():
    r = EP.load_resolver()
    assert r.get("char_google_tts_v1").runtime == "direct_api"
    assert r.get("announcer_google_tts_v1").runtime == "direct_api"
    assert r.get("music_google_lyria_v1").runtime == "direct_api"
    assert "google_tts" not in [p.engine for p in r.rank_chain("char_voice")]
    assert "google_tts" not in [p.engine for p in r.rank_chain("announcer_voice")]
    assert "google_lyria" not in [p.engine for p in r.rank_chain("music")]
    assert "elevenlabs" not in [p.engine for p in r.rank_chain("char_voice")]


def _minimal_profile_row(**updates):
    row = {
        "profile_id": "x",
        "role": "char_voice",
        "engine": "google_tts",
        "commercial_clean": True,
        "model_path": "",
        "model_sha256": "",
        "default_params": {},
        "allowed_voice_banks": ["google_tts"],
        "engine_impl_version": "1",
        "sample_rate": 24000,
        "requires_hf_token": False,
        "rank": 50,
        "is_default": False,
        "runtime": "direct_api",
        "needs_ref_clip": False,
        "caps": {},
        "license_state": "clean",
        "warn_text": "",
        "partner_row": "",
        "provider_id": "google",
        "auth_required": True,
        "billing_category": "tts",
        "canonicalizer": "audio",
        "error_policy": "fail_loud",
    }
    row.update(updates)
    return row


def test_direct_api_validation_contract():
    assert EP.EngineProfile.model_validate(_minimal_profile_row()).runtime == "direct_api"
    with pytest.raises(ValueError, match="auth_required"):
        EP.EngineProfile.model_validate(_minimal_profile_row(auth_required=False))
    with pytest.raises(ValueError, match="partner_row"):
        EP.EngineProfile.model_validate(_minimal_profile_row(partner_row="cloud_row"))
    with pytest.raises(ValueError, match="error_policy"):
        EP.EngineProfile.model_validate(_minimal_profile_row(error_policy=""))


def test_cloud_validation_still_requires_partner_row():
    with pytest.raises(ValueError, match="runtime=cloud requires a partner_row"):
        EP.EngineProfile.model_validate(
            _minimal_profile_row(runtime="cloud", provider_id="elevenlabs")
        )


def test_bad_config_is_exception_wrapped(monkeypatch, tmp_path):
    # Malformed text raises inside _resolver_from_text ...
    with pytest.raises(Exception):
        EP._resolver_from_text("profiles: [ {profile_id: x} ]")  # missing required fields
    # ... but load_resolver swallows a missing file and returns None so an
    # INPUT_TYPES path never crashes (C-5).
    monkeypatch.setattr(EP, "_profiles_path", lambda: str(tmp_path / "nope.yaml"))
    assert EP.load_resolver() is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
