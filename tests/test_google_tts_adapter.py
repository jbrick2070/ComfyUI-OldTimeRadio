"""Google Gemini TTS direct BYO API adapter tests."""
from __future__ import annotations

import ast
import base64
import pathlib
import struct
import urllib.error

import pytest

from nodes import _otr_audio_engines as AE
from nodes._otr_audio_engines import registry as R
from nodes._otr_audio_engines import eng_google_tts as G

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "nodes" / "_otr_audio_engines" / "eng_google_tts.py"


def _pcm_b64(*samples):
    return base64.b64encode(struct.pack("<" + "h" * len(samples), *samples)).decode()


def _clear_keys(monkeypatch):
    for name in ("OTR_GOOGLE_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def test_google_tts_registers_and_capabilities():
    eng = AE.get_engine("google_tts")
    assert eng.name == "google_tts"
    assert set(eng.roles) == {"char_voice", "announcer_voice"}
    assert eng.default_roles == ()
    assert eng.interface == "per_line"
    assert eng.sample_rate == 24000
    assert eng.voice_ref_field == "provider_voice_id"
    assert eng.missing_ref_fallback is None
    row = R.CAPABILITIES["google_tts"]
    assert row["cpu_ok"] is True
    assert row["requires_sidecar"] is False
    assert row["model_requirements"] == []


def test_module_has_no_heavy_or_google_imports_at_module_scope():
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    module_imports = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            module_imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            module_imports.add(node.module.split(".")[0])
    assert "google" not in module_imports
    assert "numpy" not in module_imports
    assert "torch" not in module_imports


def test_missing_key_fails_before_request(monkeypatch):
    _clear_keys(monkeypatch)
    called = False

    def _post(api_key, payload):
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(G, "_post_interaction", _post)
    with pytest.raises(G.GoogleTTSError, match="missing Google API key"):
        AE.get_engine("google_tts").generate_voice("hello", "Kore", None, 1)
    assert called is False


def test_bad_model_voice_blank_and_missing_voice_fail_before_request(monkeypatch):
    _clear_keys(monkeypatch)
    monkeypatch.setenv("OTR_GOOGLE_API_KEY", "KEY")

    def _post(api_key, payload):
        raise AssertionError("network must not be called")

    monkeypatch.setattr(G, "_post_interaction", _post)
    eng = AE.get_engine("google_tts")

    monkeypatch.setenv("OTR_GOOGLE_TTS_MODEL_ID", "not-a-model")
    with pytest.raises(ValueError, match="unsupported model"):
        eng.generate_voice("hello", "Kore", None, 1)

    monkeypatch.delenv("OTR_GOOGLE_TTS_MODEL_ID", raising=False)
    with pytest.raises(ValueError, match="unsupported voice"):
        eng.generate_voice("hello", "NotAVoice", None, 1)
    with pytest.raises(ValueError, match="missing provider_voice_id"):
        eng.generate_voice("hello", "", None, 1)
    with pytest.raises(ValueError, match="blank line"):
        eng.generate_voice("  ", "Kore", None, 1)


def test_interactions_request_shape_and_pcm_decode(monkeypatch):
    _clear_keys(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "KEY")
    seen = {}

    def _post(api_key, payload):
        seen["api_key"] = api_key
        seen["payload"] = payload
        return {"output_audio": {"data": _pcm_b64(-32768, 0, 32767)}}

    monkeypatch.setattr(G, "_post_interaction", _post)
    audio = AE.get_engine("google_tts").generate_voice("hello", "Kore", None, 1)

    assert seen["api_key"] == "KEY"
    payload = seen["payload"]
    assert payload["model"] == "gemini-2.5-flash-preview-tts"
    assert payload["input"] == "hello"
    assert payload["response_format"] == {"type": "audio"}
    assert payload["generation_config"]["speech_config"] == [{"voice": "Kore"}]
    assert "voice_config" not in str(payload)
    assert tuple(audio["waveform"].shape) == (1, 1, 3)
    assert int(audio["sample_rate"]) == 24000
    audio["waveform"].add_(0.0)


def test_response_parser_accepts_output_audio_camel_case(monkeypatch):
    _clear_keys(monkeypatch)
    monkeypatch.setenv("OTR_GOOGLE_API_KEY", "KEY")
    monkeypatch.setattr(
        G,
        "_post_interaction",
        lambda api_key, payload: {"outputAudio": {"data": _pcm_b64(1, 2)}},
    )
    audio = AE.get_engine("google_tts").generate_voice("hello", "Kore", None, 1)
    assert tuple(audio["waveform"].shape) == (1, 1, 2)


def test_key_redacted_from_sanitized_errors(monkeypatch):
    _clear_keys(monkeypatch)
    monkeypatch.setenv("OTR_GOOGLE_API_KEY", "SECRET123")

    def _post(api_key, payload):
        raise urllib.error.URLError("provider echoed SECRET123")

    monkeypatch.setattr(G, "_post_interaction", _post)
    with pytest.raises(G.GoogleTTSError) as exc:
        AE.get_engine("google_tts").generate_voice("hello", "Kore", None, 1)
    msg = str(exc.value)
    assert "SECRET123" not in msg
    assert "<REDACTED>" in msg


def test_same_provider_retry_is_disabled_by_default_and_bounded(monkeypatch):
    _clear_keys(monkeypatch)
    monkeypatch.setenv("OTR_GOOGLE_API_KEY", "KEY")
    monkeypatch.delenv("OTR_GOOGLE_TTS_ENABLE_MODEL_RETRY", raising=False)
    models = []

    def _post(api_key, payload):
        models.append(payload["model"])
        raise G.GoogleTTSError("temporary")

    monkeypatch.setattr(G, "_post_interaction", _post)
    with pytest.raises(G.GoogleTTSError):
        AE.get_engine("google_tts").generate_voice("hello", "Kore", None, 1)
    assert models == ["gemini-2.5-flash-preview-tts"]

    models.clear()
    monkeypatch.setenv("OTR_GOOGLE_TTS_ENABLE_MODEL_RETRY", "1")

    def _post_retry(api_key, payload):
        models.append(payload["model"])
        if len(models) == 1:
            raise G.GoogleTTSError("temporary")
        return {"output_audio": {"data": _pcm_b64(1)}}

    monkeypatch.setattr(G, "_post_interaction", _post_retry)
    AE.get_engine("google_tts").generate_voice("hello", "Kore", None, 1)
    assert models == [
        "gemini-2.5-flash-preview-tts",
        "gemini-3.1-flash-tts-preview",
    ]
    assert "gemini-2.5-pro-preview-tts" not in models


def test_prepare_text_uses_real_delivery_keys_and_stage_tag_allowlist():
    eng = AE.get_engine("google_tts")
    text = eng.prepare_text(
        "(whispers) I can still hear it tapping. (unknown)",
        {"afraid": 0.2, "calm": 0.8, "emotion": 1.0},
    )
    assert text.startswith("Say with audible fear")
    assert "[whispers]" in text
    assert "unknown" not in text
    calm = eng.prepare_text("Steady as she goes.", {"afraid": 0.05, "calm": 1.0})
    assert calm == "Steady as she goes."
    assert G._EMOTION_PREFIX_THRESHOLD == 0.15


def test_google_adapter_has_no_partner_or_local_engine_call_path():
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    forbidden = {
        "invoke_partner_node",
        "cloud_media_invoke",
        "eng_bark",
        "eng_kokoro",
        "eng_indextts2",
        "eng_cloud_elevenlabs",
    }
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attrs = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert forbidden.isdisjoint(names | attrs)
