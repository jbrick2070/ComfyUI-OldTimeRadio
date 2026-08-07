"""Google Veo 3.1 direct BYO API video adapter tests."""
from __future__ import annotations

import ast
import base64
import pathlib
import shutil
import subprocess

import pytest

from nodes._otr_google_api.client import GoogleAPIKeyMissingError, GoogleAPIRequestShapeError
from nodes._otr_video_engines import eng_google_veo_video as G
from nodes._otr_video_engines import registry as vreg

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "nodes" / "_otr_video_engines" / "eng_google_veo_video.py"
_FFMPEG = shutil.which("ffmpeg")


def _clear_keys(monkeypatch):
    for name in ("OTR_GOOGLE_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def _clear_video_env(monkeypatch):
    for name in (
        "OTR_GOOGLE_VEO_MODEL_ID",
        "OTR_GOOGLE_VIDEO_MODEL_ID",
        "OTR_GOOGLE_VIDEO_MODEL",
        "OTR_GOOGLE_VEO_ASPECT",
        "OTR_GOOGLE_VIDEO_ASPECT",
        "OTR_GOOGLE_VEO_DURATION_S",
        "OTR_GOOGLE_VIDEO_DURATION_S",
        "OTR_GOOGLE_VEO_RESOLUTION",
        "OTR_GOOGLE_VIDEO_RESOLUTION",
    ):
        monkeypatch.delenv(name, raising=False)


def _req(**over):
    r = {
        "shot_id": "shot_veo_001",
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


def test_google_veo_video_registers_and_capabilities():
    assert vreg.is_registered("google_veo_video")
    eng = vreg.get_engine("google_veo_video")
    assert eng.name == "google_veo_video"
    assert eng.family == "text_to_video"
    assert set(eng.roles) == {"announcer_visual", "music_visual", "character_video"}
    assert eng.default_roles == ()
    assert eng.required_inputs == ("text_prompt",)
    assert eng.native is False
    assert eng.strict_text_only is False
    assert eng.accepts_still is True
    assert eng.accepts_init_image is True
    assert eng.accepts_audio_ref is False
    assert eng.render_aspect == "wide"
    row = vreg.CAPABILITIES["google_veo_video"]
    assert "cpu" in row["device_backends"] and row["practical_without_gpu"] is True
    assert row["requires_sidecar"] is False
    assert row["model_requirements"] == []
    assert "google_veo_video" in vreg.engines_for_role("music_visual")


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
    _clear_video_env(monkeypatch)
    called = False

    def _post(path, payload, **kwargs):
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(G, "post_json", _post)
    with pytest.raises(GoogleAPIKeyMissingError):
        vreg.get_engine("google_veo_video").render_clip(_req(), {})
    assert called is False


def test_bad_shape_fails_before_request(monkeypatch):
    _clear_keys(monkeypatch)
    _clear_video_env(monkeypatch)
    monkeypatch.setenv("OTR_GOOGLE_API_KEY", "KEY")

    def _post(path, payload, **kwargs):
        raise AssertionError("network must not be called")

    monkeypatch.setattr(G, "post_json", _post)
    eng = vreg.get_engine("google_veo_video")
    with pytest.raises(GoogleAPIRequestShapeError, match="blank text_prompt"):
        eng.render_clip(_req(text_prompt="  "), {})
    with pytest.raises(GoogleAPIRequestShapeError, match="audio/base_clip"):
        eng.render_clip(_req(audio_ref={"path": "voice.wav"}), {})
    monkeypatch.setenv("OTR_GOOGLE_VIDEO_MODEL_ID", "gemini-omni-flash-preview")
    with pytest.raises(GoogleAPIRequestShapeError, match="unsupported model"):
        eng.render_clip(_req(), {})
    monkeypatch.delenv("OTR_GOOGLE_VIDEO_MODEL_ID", raising=False)
    monkeypatch.setenv("OTR_GOOGLE_VEO_ASPECT", "1:1")
    with pytest.raises(GoogleAPIRequestShapeError, match="unsupported aspectRatio"):
        eng.render_clip(_req(), {})
    monkeypatch.delenv("OTR_GOOGLE_VEO_ASPECT", raising=False)
    monkeypatch.setenv("OTR_GOOGLE_VEO_RESOLUTION", "8k")
    with pytest.raises(GoogleAPIRequestShapeError, match="unsupported resolution"):
        eng.render_clip(_req(), {})


def test_request_shape_sends_predict_long_running_payload(monkeypatch, tmp_path):
    _clear_keys(monkeypatch)
    _clear_video_env(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "KEY")
    monkeypatch.setenv("OTR_GOOGLE_VEO_MODEL_ID", "veo-3.1-generate-preview")
    monkeypatch.setenv("OTR_GOOGLE_VEO_RESOLUTION", "1080p")
    seen = {}

    def _post(path, payload, **kwargs):
        seen["path"] = path
        seen["payload"] = payload
        seen["post_kwargs"] = kwargs
        return {"name": "operations/veo-op-1"}

    def _get(path, **kwargs):
        seen["get_path"] = path
        seen["get_kwargs"] = kwargs
        return {
            "name": "operations/veo-op-1",
            "done": True,
            "response": {
                "generateVideoResponse": {
                    "generatedSamples": [
                        {"video": {"uri": "https://example.test/video.mp4"}}
                    ]
                }
            },
        }

    def _download(uri, **kwargs):
        seen["download"] = {"uri": uri, **kwargs}
        return b"mp4-bytes"

    monkeypatch.setattr(G, "post_json", _post)
    monkeypatch.setattr(G, "get_json", _get)
    monkeypatch.setattr(G, "download_media", _download)
    raw = vreg.get_engine("google_veo_video").render_clip(_req(), {})

    assert pathlib.Path(raw["path"]).read_bytes() == b"mp4-bytes"
    assert raw["content_type"] == "video/mp4"
    assert raw["provider_job_id"] == "veo-op-1"
    assert seen["path"] == "/v1beta/models/veo-3.1-generate-preview:predictLongRunning"
    assert seen["post_kwargs"]["_api_key"] == "KEY"
    assert seen["post_kwargs"]["timeout_s"] == 900
    assert seen["get_path"] == "/v1beta/operations/veo-op-1"
    assert seen["download"]["uri"] == "https://example.test/video.mp4"
    payload = seen["payload"]
    assert payload["parameters"] == {
        "aspectRatio": "16:9",
        "durationSeconds": 8,
        "resolution": "1080p",
        "sampleCount": 1,
    }
    assert list(payload["instances"][0]) == ["prompt"]
    lower = payload["instances"][0]["prompt"].lower()
    # Content clause RETIRED 2026-08-05 (operator directive): the visual
    # path carried the same weapon/nudity policy the audio path was ripped
    # of. Inverted, not deleted, so it cannot quietly return.
    assert "no explicit guns" not in lower
    assert "family-safe" not in lower
    forbidden = {"seed", "tools", "audio", "lastFrame", "referenceImages"}
    assert forbidden.isdisjoint(payload)
    assert forbidden.isdisjoint(payload["instances"][0])


def test_request_payload_accepts_init_image_and_reference_images(monkeypatch, tmp_path):
    _clear_video_env(monkeypatch)
    first = tmp_path / "first.png"
    ref1 = tmp_path / "ref1.png"
    ref2 = tmp_path / "ref2.jpg"
    first.write_bytes(b"\x89PNG\r\n\x1a\npng-bytes")
    ref1.write_bytes(b"\x89PNG\r\n\x1a\nref-one")
    ref2.write_bytes(b"\xff\xd8\xff\xe0ref-two")

    payload = G._request_payload(
        "veo-3.1-generate-preview",
        _req(asset_refs={
            "init_image": str(first),
            "reference_images": [str(ref1), {"path": str(ref2)}],
        }),
    )
    instance = payload["instances"][0]
    assert set(instance) == {"prompt", "image", "referenceImages"}
    assert instance["image"]["mimeType"] == "image/png"
    assert "inlineData" not in instance["image"]
    assert "imageBytes" not in instance["image"]
    assert base64.b64decode(instance["image"]["bytesBase64Encoded"]) == first.read_bytes()
    assert len(instance["referenceImages"]) == 2
    assert instance["referenceImages"][0]["referenceType"] == "asset"
    assert payload["parameters"]["durationSeconds"] == 8

    with pytest.raises(GoogleAPIRequestShapeError, match="reference_images require"):
        G._request_payload(
            "veo-3.1-lite-generate-preview",
            _req(asset_refs={"reference_images": [str(ref1)]}),
        )
    with pytest.raises(GoogleAPIRequestShapeError, match="last_frame requires"):
        G._request_payload(
            "veo-3.1-generate-preview",
            _req(asset_refs={"last_frame": str(ref1)}),
        )
    with pytest.raises(GoogleAPIRequestShapeError, match="missing/absent"):
        G._request_payload(
            "veo-3.1-generate-preview",
            _req(asset_refs={"init_image": str(tmp_path / "missing.png")}),
        )


def test_duration_aspect_and_lite_resolution_clamps(monkeypatch):
    _clear_video_env(monkeypatch)
    req = _req(canvas={"w": 480, "h": 832, "fps": 24}, timing={"target_frame_count": 120})
    payload = G._request_payload("veo-3.1-lite-generate-preview", req)
    assert payload["parameters"]["aspectRatio"] == "9:16"
    assert payload["parameters"]["durationSeconds"] == 4
    assert payload["parameters"]["resolution"] == "720p"

    monkeypatch.setenv("OTR_GOOGLE_VEO_DURATION_S", "6")
    payload = G._request_payload("veo-3.1-lite-generate-preview", _req())
    assert payload["parameters"]["durationSeconds"] == 6

    monkeypatch.setenv("OTR_GOOGLE_VEO_RESOLUTION", "4k")
    with pytest.raises(GoogleAPIRequestShapeError, match="does not support"):
        G._request_payload("veo-3.1-lite-generate-preview", _req())

    monkeypatch.setenv("OTR_GOOGLE_VEO_MODEL_ID", "veo-3.1-fast-generate-preview")
    assert G._selected_model() == "veo-3.1-fast-generate-preview"


def test_operation_parser_accepts_rest_and_sdk_shapes():
    rest = {
        "response": {
            "generateVideoResponse": {
                "generatedSamples": [{"video": {"uri": "https://example.test/rest.mp4"}}]
            }
        }
    }
    assert G._extract_generated_video(rest)["uri"].endswith("rest.mp4")
    camel = {
        "response": {
            "generatedVideos": [{"video": {"uri": "https://example.test/sdk.mp4"}}]
        }
    }
    assert G._extract_generated_video(camel)["uri"].endswith("sdk.mp4")
    snake = {
        "response": {
            "generated_videos": [{"video": {"uri": "https://example.test/snake.mp4"}}]
        }
    }
    assert G._extract_generated_video(snake)["uri"].endswith("snake.mp4")


def test_bad_operation_shapes_fail_loud():
    with pytest.raises(GoogleAPIRequestShapeError, match="lacked name"):
        G._extract_operation_name({})
    with pytest.raises(GoogleAPIRequestShapeError, match="did not include"):
        G._extract_generated_video({"response": {}})
    with pytest.raises(GoogleAPIRequestShapeError, match="lacked uri"):
        G._video_uri({"mimeType": "video/mp4"})
    with pytest.raises(G.GoogleVeoVideoError, match="operation failed"):
        G._extract_generated_video({"error": {"message": "blocked"}})


def test_poll_operation(monkeypatch):
    states = iter([{"done": False}, {"done": True, "response": {"generatedVideos": []}}])
    seen = {"sleep": []}

    def _get(path, **kwargs):
        seen["path"] = path
        seen["kwargs"] = kwargs
        return next(states)

    monkeypatch.setattr(G, "get_json", _get)
    monkeypatch.setattr(G.time, "sleep", lambda s: seen["sleep"].append(s))
    status = G._poll_operation("operations/video-9", api_key="KEY", timeout_s=30)
    assert status["done"] is True
    assert seen["path"] == "/v1beta/operations/video-9"
    assert seen["kwargs"]["_api_key"] == "KEY"
    assert seen["sleep"] == [G._POLL_INTERVAL_S]


@pytest.mark.skipif(not _FFMPEG, reason="ffmpeg not on PATH")
def test_canonicalize_returns_silent_clip_dict(tmp_path):
    src = _mp4_fixture(tmp_path)
    raw = {
        "path": str(src),
        "content_type": "video/mp4",
        "duration_s": None,
        "provider_job_id": "google-veo-job",
        "raw_meta": {},
    }
    clip = vreg.get_engine("google_veo_video").canonicalize(raw, _req(), {})
    assert clip["engine_id"] == "google_veo_video"
    assert clip["family"] == "text_to_video"
    assert clip["type"] == "video"
    assert clip["has_audio"] is False
    assert clip["fps"] == 25
    assert clip["frame_count"] > 0
    assert clip["provider_job_id"] == "google-veo-job"
    assert pathlib.Path(clip["path"]).is_file()
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", clip["path"]],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert '"codec_type": "audio"' not in probe.stdout


def test_google_veo_video_has_no_partner_or_local_engine_call_path():
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    forbidden = {
        "invoke_partner_node",
        "cloud_media_invoke",
        "eng_cloud_video",
        "eng_ltx_video",
        "eng_wan_i2v",
        "eng_humo",
        "create_interaction",
    }
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attrs = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert forbidden.isdisjoint(names | attrs)


# ---------------------------------------------------------------------------
# Rescued coverage (rip-sfx bed, 2026-08-06). These four tests lived in the
# deleted SFX-only files (test_google_video_sfx_beds.py /
# test_google_video_sfx_render_driver.py) but pin SURVIVING behaviour:
# the silent Google adapters' V-1 silence proof, the base VEO payload never
# requesting provider audio, and the visual-Google prompt-routing branch.
# ---------------------------------------------------------------------------

def _audio_streams(path: pathlib.Path) -> list:
    import json as _json

    p = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams",
         str(path)],
        check=True, capture_output=True, text=True, timeout=60)
    doc = _json.loads(p.stdout or "{}")
    return [s for s in doc.get("streams") or []
            if s.get("codec_type") == "audio"]


def test_base_veo_payload_never_requests_provider_audio(monkeypatch):
    """V-1: the base VEO request must not ask the provider to generate audio.

    Rescued from the retired SFX-bed suite -- this is the ONLY assertion
    pinning that the surviving VEO payload builder omits ``generateAudio``."""
    _clear_video_env(monkeypatch)
    base_payload = G._request_payload("veo-3.1-lite-generate-preview", _req())
    assert "generateAudio" not in base_payload["parameters"]


@pytest.mark.skipif(not _FFMPEG, reason="ffmpeg not on PATH")
def test_existing_google_video_engines_stay_silent(tmp_path):
    """V-1 for the surviving Google lanes: canonical clips carry NO audio
    stream and NO sfx fields (the bed producers are retired)."""
    from nodes._otr_video_engines import eng_google_omni_video as OMNI  # noqa: F401

    src = _mp4_fixture(tmp_path)
    raw = {"path": str(src), "content_type": "video/mp4", "duration_s": None,
           "provider_job_id": "silent-job", "raw_meta": {}}
    for name in ("google_veo_video", "google_omni_video"):
        clip = vreg.get_engine(name).canonicalize(dict(raw), _req(), {})
        assert clip["engine_id"] == name
        assert clip["has_audio"] is False
        assert "sfx_stem_path" not in clip
        assert _audio_streams(pathlib.Path(clip["path"])) == []


def _music_open_ledger(engine_id="google_veo_video"):
    from nodes._otr_video_engines import render_driver as rd

    return {
        "audio": {"master_audio_sha256": rd.FROZEN_AUDIO_SHA,
                  "ledger_frozen": True},
        "meta": {
            "visual_style": "video_art",
            "style": "archive signal mystery",
            "story_brief_terms": {
                "setting": ["tape archive"],
                "atmosphere": ["electrical unease"],
            },
        },
        "lines": [
            {"line_id": "b000_music_open", "char_id": "",
             "speaker_role": "music_open", "start_s": 0.0, "dur_s": 4.0},
        ],
        "images": {"images": [
            {"object_id": "still_b000_music_open", "kind": "scene_open",
             "beat_id": "b000_music_open", "path": "X:/img/open.png"},
        ]},
        "video": {"video_revision": 1, "fps": 25, "shots": [
            {"shot_id": "shot_b000_music_open",
             "source_line_ids": ["b000_music_open"],
             "role": "music_visual",
             "engine_id": engine_id,
             "family": "text_to_video",
             "group_id": "grp_music",
             "target_frame_count": 100,
             "degradation_trail": [],
             "render_request_hash": "hash_veo_open",
             "cache_keys": {"request_hash": "hash_veo_open"},
             "creative": {}},
        ]},
    }


def test_silent_google_prompt_routing_still_uses_visual_google_branch():
    """The ONLY test exercising the surviving google_veo_video prompt-routing
    branch in the render driver (rescued from the retired SFX driver suite)."""
    from nodes._otr_video_engines import render_driver as rd

    led = _music_open_ledger(engine_id="google_veo_video")
    shot = led["video"]["shots"][0]
    req = rd.build_request_from_shot(shot, led)
    prompt = req["text_prompt"].lower()
    assert "single luminous radio receiver" in prompt
    # Content clause retired 2026-08-05 (operator directive); nothing appends
    # a speech clause on this branch either.
    assert "no explicit guns" not in prompt
    # NOT "applied", and that is correct: `_apply_visual_safety_prompt`
    # early-returns when the helper changes nothing, and the retired clause
    # changes nothing -- stamping "applied" would assert an enforcement that
    # did not happen.
    assert req["observability"].get("visual_safety_prompt") is None
