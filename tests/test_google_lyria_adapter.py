"""Google Lyria direct BYO API music adapter tests."""
from __future__ import annotations

import ast
import base64
import pathlib

import pytest
import torch

from nodes import _otr_audio_engines as AE
from nodes._otr_audio_engines import eng_google_lyria as L
from nodes._otr_audio_engines import registry as R
from nodes._otr_google_api.client import GoogleAPIKeyMissingError, GoogleAPIRequestShapeError

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "nodes" / "_otr_audio_engines" / "eng_google_lyria.py"


def _clear_keys(monkeypatch):
    for name in ("OTR_GOOGLE_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def _audio(seconds=2, sr=L.SAMPLE_RATE):
    return {
        "waveform": torch.zeros(1, 2, int(seconds * sr), dtype=torch.float32),
        "sample_rate": sr,
    }


def test_google_lyria_registers_and_capabilities():
    eng = AE.get_engine("google_lyria")
    assert eng.name == "google_lyria"
    assert set(eng.roles) == {"music"}
    assert eng.default_roles == ()
    assert eng.interface == "clip"
    assert eng.native is False
    assert eng.sample_rate == 44100
    assert eng.fixed_provider_duration_s == 30
    row = R.CAPABILITIES["google_lyria"]
    assert row["cpu_ok"] is True
    assert row["requires_sidecar"] is False
    assert row["model_requirements"] == []


def test_module_has_no_heavy_or_google_sdk_imports_at_module_scope():
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

    def _create(payload, **kwargs):
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(L, "create_interaction", _create)
    with pytest.raises(GoogleAPIKeyMissingError, match="no API key"):
        AE.get_engine("google_lyria").generate_clip("soft synth cue", 8, 123)
    assert called is False


def test_blank_prompt_bad_duration_and_model_fail_before_request(monkeypatch):
    _clear_keys(monkeypatch)
    monkeypatch.setenv("OTR_GOOGLE_API_KEY", "KEY")

    def _create(payload, **kwargs):
        raise AssertionError("network must not be called")

    monkeypatch.setattr(L, "create_interaction", _create)
    eng = AE.get_engine("google_lyria")

    with pytest.raises(GoogleAPIRequestShapeError, match="blank music prompt"):
        eng.generate_clip("  ", 8, 1)
    with pytest.raises(GoogleAPIRequestShapeError, match="non-numeric duration"):
        eng.generate_clip("soft synth cue", "soon", 1)
    monkeypatch.setenv("OTR_GOOGLE_LYRIA_MODEL_ID", "not-a-lyria-model")
    with pytest.raises(GoogleAPIRequestShapeError, match="unsupported model"):
        eng.generate_clip("soft synth cue", 8, 1)


def test_request_shape_sends_only_model_and_input(monkeypatch):
    _clear_keys(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "KEY")
    seen = {}

    def _create(payload, **kwargs):
        seen["payload"] = dict(payload)
        seen["kwargs"] = dict(kwargs)
        return {"output_audio": {"data": base64.b64encode(b"mp3").decode()}}

    def _decode(mp3_bytes, *, sample_rate=L.SAMPLE_RATE):
        seen["mp3"] = mp3_bytes
        return _audio(seconds=30, sr=sample_rate)

    monkeypatch.setattr(L, "create_interaction", _create)
    monkeypatch.setattr(L, "_mp3_to_audio", _decode)

    audio = AE.get_engine("google_lyria").generate_clip(
        "melancholy analog synth, no vocals", 8, seed=999
    )

    assert seen["payload"] == {
        "model": "lyria-3-clip-preview",
        "input": "melancholy analog synth, no vocals",
    }
    assert "duration" not in seen["payload"]
    assert "seed" not in seen["payload"]
    assert seen["kwargs"]["timeout_s"] == 300
    assert seen["kwargs"]["max_retries"] == 0
    assert seen["kwargs"]["_api_key"] == "KEY"
    assert seen["mp3"] == b"mp3"
    assert tuple(audio["waveform"].shape) == (1, 2, 8 * L.SAMPLE_RATE)


def test_response_parser_accepts_camel_case_and_steps(monkeypatch):
    monkeypatch.setattr(L, "_mp3_to_audio", lambda data, **kw: _audio(seconds=1))
    b64 = base64.b64encode(b"mp3").decode()
    assert tuple(L._audio_from_response({"outputAudio": {"data": b64}})["waveform"].shape) == (
        1,
        2,
        L.SAMPLE_RATE,
    )
    body = {
        "steps": [
            {
                "type": "model_output",
                "content": [
                    {"type": "audio", "mime_type": "audio/mpeg", "data": b64},
                ],
            }
        ],
    }
    assert tuple(L._audio_from_response(body)["waveform"].shape) == (1, 2, L.SAMPLE_RATE)


def test_bad_response_shape_and_mime_fail_loud(monkeypatch):
    with pytest.raises(GoogleAPIRequestShapeError, match="did not include audio"):
        L._audio_from_response({"output_text": "lyrics only"})
    with pytest.raises(GoogleAPIRequestShapeError, match="unsupported"):
        L._audio_from_response(
            {"output_audio": {"data": base64.b64encode(b"x").decode(), "mime_type": "audio/wav"}}
        )


def test_duration_is_clamped_and_trimmed_locally(monkeypatch):
    _clear_keys(monkeypatch)
    monkeypatch.setenv("OTR_GOOGLE_API_KEY", "KEY")

    def _create(payload, **kwargs):
        return {"output_audio": {"data": base64.b64encode(b"mp3").decode()}}

    monkeypatch.setattr(L, "create_interaction", _create)
    monkeypatch.setattr(L, "_mp3_to_audio", lambda data, **kw: _audio(seconds=30))
    eng = AE.get_engine("google_lyria")
    short = eng.generate_clip("a cue", -5, 1)
    long = eng.generate_clip("a cue", 999, 1)
    assert tuple(short["waveform"].shape) == (1, 2, 1 * L.SAMPLE_RATE)
    assert tuple(long["waveform"].shape) == (1, 2, 30 * L.SAMPLE_RATE)


def test_google_lyria_has_no_partner_or_local_engine_call_path():
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    forbidden = {
        "invoke_partner_node",
        "cloud_media_invoke",
        "eng_musicgen",
        "eng_stable_audio",
        "eng_stable_audio_3",
        "eng_cloud_sonilo",
    }
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attrs = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert forbidden.isdisjoint(names | attrs)
