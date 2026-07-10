"""Tests for the registered audio-engine adapters (real registry population).

Scope: music + announcer + character voices. SFX is intentionally out of the
v2 audio-engine infrastructure.
"""
import pytest

from nodes import _otr_audio_engines as AE


def test_char_voice_default_is_bark():
    assert AE.engines_for_role("char_voice")[0] == "indextts2"  # PROMOTED 2026-06-04


def test_announcer_default_is_kokoro():
    assert AE.engines_for_role("announcer_voice")[0] == "kokoro"


def test_music_default_is_stable_audio_3():
    # PROMOTED 2026-06-03: SA3 is the shipped music default; musicgen still in the pool.
    assert AE.engines_for_role("music")[0] == "stable_audio_3"
    assert "musicgen" in AE.engines_for_role("music")


def test_char_voice_pool_has_optins():
    pool = set(AE.engines_for_role("char_voice"))
    assert {"bark", "chatterbox", "indextts2"} <= pool


def test_music_role_has_stable_audio():
    assert "stable_audio_music" in AE.engines_for_role("music")
    assert "google_lyria" in AE.engines_for_role("music")


def test_unknown_role_has_no_engines():
    # A role outside the v2 audio-engine set resolves to no engines.
    assert AE.engines_for_role("retired_role_c") == []


def test_chatterbox_serves_both_voice_roles():
    assert "chatterbox" in AE.engines_for_role("char_voice")
    assert "chatterbox" in AE.engines_for_role("announcer_voice")


def test_dia_serves_both_voice_roles_as_opt_in():
    assert "dia" in AE.engines_for_role("char_voice")
    assert "dia" in AE.engines_for_role("announcer_voice")
    assert AE.get_engine("dia").default_roles == ()


def test_google_tts_serves_both_voice_roles_as_opt_in():
    assert "google_tts" in AE.engines_for_role("char_voice")
    assert "google_tts" in AE.engines_for_role("announcer_voice")
    assert AE.get_engine("google_tts").default_roles == ()


def test_google_lyria_serves_music_as_opt_in():
    assert "google_lyria" in AE.engines_for_role("music")
    assert AE.get_engine("google_lyria").default_roles == ()


def test_optin_flags_and_licensing():
    cb = AE.get_engine("chatterbox")
    assert cb.requires_flag is None  # C6: registry IS the menu (no flag gate)
    assert cb.commercial_clean is True
    idx = AE.get_engine("indextts2")
    assert idx.requires_flag is None  # PROMOTED 2026-06-04: shipped char_voice default
    assert idx.commercial_clean is False  # Bilibili non-commercial (non-blocking warn)
    sa = AE.get_engine("stable_audio_music")
    assert sa.requires_flag is None  # C6: registry IS the menu (no flag gate)
    assert sa.commercial_clean is True


def test_assert_usable_no_flag_gate(monkeypatch):
    # C6 -- registry IS the menu: chatterbox is selectable with NO flag gate (the
    # deeper venv/weights checks run in load(), not in assert_usable). indextts2
    # (the shipped char_voice default) is likewise usable.
    monkeypatch.delenv("OTR_ENABLE_CHATTERBOX", raising=False)
    assert AE.assert_usable("chatterbox", "char_voice") == "chatterbox"
    monkeypatch.delenv("OTR_ENABLE_INDEXTTS2", raising=False)
    assert AE.assert_usable("indextts2", "char_voice") == "indextts2"


def test_assert_usable_optin_on_runs(monkeypatch):
    monkeypatch.setenv("OTR_ENABLE_CHATTERBOX", "1")
    assert AE.assert_usable("chatterbox", "char_voice") == "chatterbox"


def test_bark_is_per_line_after_cleanbreak_1a():
    # Audio clean-break (1a): bark flipped from batch delegation to a
    # self-contained per_line registry engine (nodes/_otr_audio_engines/eng_bark.py).
    assert AE.get_engine("bark").interface == "per_line"


def test_kokoro_is_per_line_after_cleanbreak_1b():
    # Audio clean-break (1b): kokoro flipped from batch delegation to a
    # self-contained per_line announcer engine (eng_kokoro.py).
    assert AE.get_engine("kokoro").interface == "per_line"


def test_kokoro_uses_explicit_requested_device(monkeypatch):
    """S4 platform-portability (2026-07-10): the cuda->mps->cpu auto-waterfall
    (_kokoro_device) is DELETED. The device is now EXPLICIT: the voice node
    threads the CastLock ledger stamp (meta.voice_device) onto the adapter as
    ``requested_device``; load() passes it straight to KPipeline (a device the
    host cannot provide fails LOUD inside KPipeline itself -- no silent
    downgrade path to cover here)."""
    import sys
    import types

    from nodes._otr_audio_engines import eng_kokoro

    # (i) the waterfall helper no longer exists on the module.
    assert getattr(eng_kokoro, "_kokoro_device", None) is None

    # (ii) KPipeline receives the adapter's requested_device verbatim.
    captured = {}

    class _FakeKPipeline:
        def __init__(self, lang_code, device, repo_id):
            captured["lang_code"] = lang_code
            captured["device"] = device
            captured["repo_id"] = repo_id

    fake_kokoro_module = types.SimpleNamespace(KPipeline=_FakeKPipeline)
    monkeypatch.setitem(sys.modules, "kokoro", fake_kokoro_module)

    explicit = eng_kokoro.KokoroEngine()
    explicit.requested_device = "cpu"
    explicit.load()
    assert captured["device"] == "cpu"

    # requested_device unset -> the nv50 baseline default ("cuda").
    default_device = eng_kokoro.KokoroEngine()
    default_device.load()
    assert captured["device"] == "cuda"


def test_musicgen_is_clip_after_cleanbreak_1c():
    # Audio clean-break (1c): musicgen flipped from batch delegation to a
    # self-contained clip engine (eng_musicgen.py).
    assert AE.get_engine("musicgen").interface == "clip"


def test_musicgen_uses_small_model_for_no_cuda_floor():
    from nodes._otr_audio_engines import eng_musicgen

    assert eng_musicgen._MUSICGEN_MODEL_ID == "facebook/musicgen-small"
    assert AE.get_engine("musicgen").model_id == "facebook/musicgen-small"


def test_no_batch_interface_engines_remain():
    # Clean-break complete (1a/1b/1c): every registered audio engine is
    # self-contained per_line / clip; the batch-delegation layer is retired.
    for n in ("bark", "chatterbox", "indextts2", "kokoro", "musicgen",
              "stable_audio_music", "google_tts", "google_lyria"):
        assert AE.get_engine(n).interface in ("per_line", "clip"), n


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
