"""Google Gemini/Nano Banana direct BYO API image adapter tests."""
from __future__ import annotations

import ast
import base64
import io
import pathlib

import pytest

from nodes._otr_google_api.client import GoogleAPIKeyMissingError, GoogleAPIRequestShapeError
from nodes._otr_image_engines import eng_google_image as G
from nodes._otr_image_engines import registry as ireg

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "nodes" / "_otr_image_engines" / "eng_google_image.py"


def _clear_keys(monkeypatch):
    for name in ("OTR_GOOGLE_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def _req(**over):
    r = {
        "prompt": "a moody radio archive room, cinematic still",
        "seed": 7,
        "width": 832,
        "height": 448,
    }
    r.update(over)
    return r


def _png_b64(size=(64, 32)):
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", size, (20, 110, 180)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_google_image_registers_and_capabilities():
    assert ireg.is_registered("google_image")
    eng = ireg.get_engine("google_image")
    assert eng.name == "google_image"
    assert set(eng.roles) >= {"announcer_visual", "music_visual", "character_video"}
    assert eng.default_roles == ()
    assert eng.required_inputs == ("text_prompt",)
    assert eng.native is False
    row = ireg.CAPABILITIES["google_image"]
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
    assert "PIL" not in module_imports
    assert "torch" not in module_imports
    assert "numpy" not in module_imports


def test_missing_key_fails_before_request(monkeypatch):
    _clear_keys(monkeypatch)
    called = False

    def _create(payload, **kwargs):
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(G, "create_interaction", _create)
    with pytest.raises(GoogleAPIKeyMissingError):
        ireg.get_engine("google_image").render_image(_req(), {})
    assert called is False


def test_bad_shape_fails_before_request(monkeypatch):
    _clear_keys(monkeypatch)
    monkeypatch.setenv("OTR_GOOGLE_API_KEY", "KEY")

    def _create(payload, **kwargs):
        raise AssertionError("network must not be called")

    monkeypatch.setattr(G, "create_interaction", _create)
    eng = ireg.get_engine("google_image")
    with pytest.raises(GoogleAPIRequestShapeError, match="blank prompt"):
        eng.render_image(_req(prompt="  "), {})
    with pytest.raises(GoogleAPIRequestShapeError, match="width/height"):
        eng.render_image(_req(width="wide"), {})
    with pytest.raises(GoogleAPIRequestShapeError, match="init/reference images"):
        eng.render_image(_req(init_image="portrait.png"), {})
    monkeypatch.setenv("OTR_GOOGLE_IMAGE_MODEL_ID", "imagen-4.0-generate")
    with pytest.raises(GoogleAPIRequestShapeError, match="unsupported model"):
        eng.render_image(_req(), {})
    monkeypatch.delenv("OTR_GOOGLE_IMAGE_MODEL_ID", raising=False)
    monkeypatch.setenv("OTR_GOOGLE_IMAGE_MIME", "image/png")
    with pytest.raises(GoogleAPIRequestShapeError, match="unsupported MIME"):
        eng.render_image(_req(), {})


def test_request_shape_sends_interactions_image_format(monkeypatch):
    _clear_keys(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "KEY")
    monkeypatch.setenv("OTR_GOOGLE_IMAGE_SIZE", "2K")
    seen = {}

    def _create(payload, **kwargs):
        seen["payload"] = dict(payload)
        seen["kwargs"] = dict(kwargs)
        return {"output_image": {"data": _png_b64(), "mime_type": "image/png"}}

    monkeypatch.setattr(G, "create_interaction", _create)
    out = ireg.get_engine("google_image").render_image(_req(width=1920, height=1080), {})

    assert out.endswith(".canon.png")
    payload = seen["payload"]
    assert payload["model"] == "gemini-3.1-flash-image"
    assert "duration" not in payload
    assert "seed" not in payload
    assert "tools" not in payload
    assert payload["response_format"] == {
        "type": "image",
        "mime_type": "image/jpeg",
        "aspect_ratio": "16:9",
        "image_size": "2K",
    }
    lower = payload["input"].lower()
    assert "no blood" in lower
    assert "no guns" in lower
    assert "no knives" in lower
    assert "no smoking" in lower
    assert seen["kwargs"]["_api_key"] == "KEY"
    assert seen["kwargs"]["timeout_s"] == 300
    assert seen["kwargs"]["max_retries"] == 0


def test_lite_rejects_unsupported_size_and_aspect(monkeypatch):
    _clear_keys(monkeypatch)
    monkeypatch.setenv("OTR_GOOGLE_API_KEY", "KEY")
    monkeypatch.setenv("OTR_GOOGLE_IMAGE_MODEL_ID", "gemini-3.1-flash-lite-image")
    monkeypatch.setenv("OTR_GOOGLE_IMAGE_SIZE", "4K")
    with pytest.raises(GoogleAPIRequestShapeError, match="image_size"):
        ireg.get_engine("google_image").render_image(_req(), {})

    monkeypatch.setenv("OTR_GOOGLE_IMAGE_SIZE", "1K")
    monkeypatch.setenv("OTR_GOOGLE_IMAGE_ASPECT", "17:10")
    with pytest.raises(GoogleAPIRequestShapeError, match="aspect_ratio"):
        ireg.get_engine("google_image").render_image(_req(), {})


def test_response_parser_accepts_camel_case_and_steps():
    b64 = _png_b64()
    assert G._extract_image_data({"outputImage": {"data": b64}})["mime_type"] == "image/png"
    body = {
        "steps": [
            {
                "type": "model_output",
                "content": [
                    {"type": "image", "mime_type": "image/jpeg", "data": b64},
                ],
            }
        ]
    }
    assert G._extract_image_data(body)["mime_type"] == "image/jpeg"


def test_bad_response_shape_and_mime_fail_loud():
    with pytest.raises(GoogleAPIRequestShapeError, match="did not include image"):
        G._extract_image_data({"output_text": "not an image"})
    with pytest.raises(GoogleAPIRequestShapeError, match="unsupported"):
        G._extract_image_data({"output_image": {"data": _png_b64(), "mime_type": "image/webp"}})


def test_dispatcher_skips_gpu_lease_for_google_image():
    from nodes.otr_image_gen_dispatcher import _is_cloud_image_engine

    assert _is_cloud_image_engine("google_image") is True


def test_google_image_has_no_partner_or_local_engine_call_path():
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    forbidden = {
        "invoke_partner_node",
        "cloud_media_invoke",
        "flux_gen1",
        "z_image_turbo",
        "eng_cloud_image",
    }
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attrs = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert forbidden.isdisjoint(names | attrs)
