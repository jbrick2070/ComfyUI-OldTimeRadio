"""Google Gemini Omni Flash direct BYO API video adapter tests."""
from __future__ import annotations

import ast
import base64
import pathlib
import shutil
import subprocess

import pytest

from nodes._otr_google_api.client import GoogleAPIKeyMissingError, GoogleAPIRequestShapeError
from nodes._otr_video_engines import eng_google_omni_video as G
from nodes._otr_video_engines import registry as vreg

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "nodes" / "_otr_video_engines" / "eng_google_omni_video.py"
_FFMPEG = shutil.which("ffmpeg")


def _clear_keys(monkeypatch):
    for name in ("OTR_GOOGLE_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def _req(**over):
    r = {
        "shot_id": "shot_google_001",
        "text_prompt": "a calm archive reading room, cinematic slow pan",
        "seed_bundle": {"request_seed": 7},
        "canvas": {"w": 832, "h": 448, "fps": 25},
        "timing": {"target_frame_count": 125},
    }
    r.update(over)
    return r


def _mp4_fixture(tmp_path) -> pathlib.Path:
    out = tmp_path / "provider.mp4"
    subprocess.run(
        [
            _FFMPEG,
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=128x72:rate=12:duration=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=330:duration=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(out),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    return out


def test_google_omni_video_registers_and_capabilities():
    assert vreg.is_registered("google_omni_video")
    eng = vreg.get_engine("google_omni_video")
    assert eng.name == "google_omni_video"
    assert eng.family == "text_to_video"
    assert set(eng.roles) == {"announcer_visual", "music_visual", "character_video"}
    assert eng.default_roles == ()
    assert eng.required_inputs == ("text_prompt",)
    assert eng.native is False
    row = vreg.CAPABILITIES["google_omni_video"]
    assert row["cpu_ok"] is True
    assert row["requires_sidecar"] is False
    assert row["model_requirements"] == []
    assert "google_omni_video" in vreg.engines_for_role("music_visual")


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
        vreg.get_engine("google_omni_video").render_clip(_req(), {})
    assert called is False


def test_bad_shape_fails_before_request(monkeypatch):
    _clear_keys(monkeypatch)
    monkeypatch.setenv("OTR_GOOGLE_API_KEY", "KEY")

    def _create(payload, **kwargs):
        raise AssertionError("network must not be called")

    monkeypatch.setattr(G, "create_interaction", _create)
    eng = vreg.get_engine("google_omni_video")
    with pytest.raises(GoogleAPIRequestShapeError, match="blank text_prompt"):
        eng.render_clip(_req(text_prompt="  "), {})
    with pytest.raises(GoogleAPIRequestShapeError, match="text-to-video only"):
        eng.render_clip(_req(init_image="portrait.png"), {})
    with pytest.raises(GoogleAPIRequestShapeError, match="text-to-video only"):
        eng.render_clip(_req(audio_ref={"path": "voice.wav"}), {})
    monkeypatch.setenv("OTR_GOOGLE_VIDEO_MODEL_ID", "veo-3.1-generate-preview")
    with pytest.raises(GoogleAPIRequestShapeError, match="unsupported model"):
        eng.render_clip(_req(), {})


def test_request_shape_sends_interactions_video_uri_format(monkeypatch, tmp_path):
    _clear_keys(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "KEY")
    seen = {}

    def _create(payload, **kwargs):
        seen["payload"] = dict(payload)
        seen["kwargs"] = dict(kwargs)
        return {"output_video": {"uri": "https://example.test/v1beta/files/video-1:download?alt=media"}}

    def _download(uri, *, api_key, timeout_s):
        seen["download"] = {"uri": uri, "api_key": api_key, "timeout_s": timeout_s}
        return b"mp4-bytes"

    monkeypatch.setattr(G, "create_interaction", _create)
    monkeypatch.setattr(G, "_download_video_uri", _download)
    raw = vreg.get_engine("google_omni_video").render_clip(_req(), {})

    assert raw["path"].endswith(".mp4")
    assert pathlib.Path(raw["path"]).read_bytes() == b"mp4-bytes"
    assert raw["content_type"] == "video/mp4"
    payload = seen["payload"]
    assert payload["model"] == "gemini-omni-flash-preview"
    assert payload["response_format"] == {"type": "video", "delivery": "uri"}
    assert "duration" not in payload
    assert "seed" not in payload
    assert "resolution" not in payload
    assert "tools" not in payload
    lower = payload["input"].lower()
    assert "no blood" in lower
    assert "no guns" in lower
    assert "no knives" in lower
    assert "no smoking" in lower
    assert seen["kwargs"]["_api_key"] == "KEY"
    assert seen["kwargs"]["timeout_s"] == 900
    assert seen["kwargs"]["max_retries"] == 0
    assert seen["download"] == {
        "uri": "https://example.test/v1beta/files/video-1:download?alt=media",
        "api_key": "KEY",
        "timeout_s": 900,
    }


def test_response_parser_accepts_camel_case_base64_and_steps():
    b64 = base64.b64encode(b"mp4").decode("ascii")
    assert G._extract_video_output({"outputVideo": {"data": b64}})["data"] == b64
    body = {
        "steps": [
            {
                "type": "model_output",
                "content": [
                    {
                        "type": "video",
                        "mime_type": "video/mp4",
                        "uri": "https://example.test/v1beta/files/video-1:download?alt=media",
                    },
                ],
            }
        ]
    }
    assert G._extract_video_output(body)["uri"].endswith("alt=media")


def test_bad_response_shape_mime_and_base64_fail_loud():
    with pytest.raises(GoogleAPIRequestShapeError, match="did not include video"):
        G._extract_video_output({"output_text": "not video"})
    with pytest.raises(GoogleAPIRequestShapeError, match="unsupported"):
        G._extract_video_output({"output_video": {"data": "abc", "mime_type": "video/webm"}})
    with pytest.raises(GoogleAPIRequestShapeError, match="base64"):
        G._provider_bytes({"output_video": {"data": "not base64"}}, api_key="KEY", timeout_s=1)


def test_file_uri_poll_download(monkeypatch):
    states = iter([{"state": "PROCESSING"}, {"state": "ACTIVE"}])
    seen = {"sleep": []}

    def _get(path, **kwargs):
        seen["path"] = path
        seen["get_kwargs"] = kwargs
        return next(states)

    def _download(uri, **kwargs):
        seen["uri"] = uri
        seen["download_kwargs"] = kwargs
        return b"mp4"

    monkeypatch.setattr(G, "get_json", _get)
    monkeypatch.setattr(G, "download_media", _download)
    monkeypatch.setattr(G.time, "sleep", lambda s: seen["sleep"].append(s))
    data = G._download_video_uri(
        "https://generativelanguage.googleapis.com/v1beta/files/video-9:download?alt=media",
        api_key="KEY",
        timeout_s=30,
    )
    assert data == b"mp4"
    assert seen["path"] == "/v1beta/files/video-9"
    assert seen["get_kwargs"]["_api_key"] == "KEY"
    assert seen["uri"].endswith("video-9:download?alt=media")
    assert seen["sleep"] == [G._POLL_INTERVAL_S]


@pytest.mark.skipif(not _FFMPEG, reason="ffmpeg not on PATH")
def test_canonicalize_returns_silent_clip_dict(tmp_path):
    src = _mp4_fixture(tmp_path)
    raw = {
        "path": str(src),
        "content_type": "video/mp4",
        "duration_s": None,
        "provider_job_id": "google-job",
        "raw_meta": {},
    }
    clip = vreg.get_engine("google_omni_video").canonicalize(raw, _req(), {})
    assert clip["engine_id"] == "google_omni_video"
    assert clip["family"] == "text_to_video"
    assert clip["type"] == "video"
    assert clip["has_audio"] is False
    assert clip["fps"] == 25
    assert clip["frame_count"] > 0
    assert clip["provider_job_id"] == "google-job"
    assert pathlib.Path(clip["path"]).is_file()
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", clip["path"]],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert '"codec_type": "audio"' not in probe.stdout


def test_google_omni_video_has_no_partner_or_local_engine_call_path():
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    forbidden = {
        "invoke_partner_node",
        "cloud_media_invoke",
        "eng_cloud_video",
        "eng_ltx_video",
        "eng_wan_i2v",
        "eng_humo",
    }
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attrs = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert forbidden.isdisjoint(names | attrs)
