"""Google Veo 3.1 direct BYO-key video adapter.

Direct Google Gemini API lane for short text-to-video clips using Veo's
``predictLongRunning`` REST endpoint. This adapter is not a Comfy Cloud Partner
node and it never invokes a local video model. The first shipped surface is
text-to-video only; image/video/reference extension paths fail loud before
network until those request shapes are wired and tested.

Import-time stays light: no Google SDK, PIL, NumPy, Torch, network, or Comfy
runtime imports happen at module scope.
"""
from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

from .registry import register
from .._otr_google_api.client import (
    GoogleAPIError,
    GoogleAPIRequestShapeError,
    download_media,
    get_json,
    post_json,
    resolve_api_key,
)
from .._otr_story_brief_helpers import append_visual_safety_clause

DEFAULT_MODEL = "veo-3.1-lite-generate-preview"
SUPPORTED_MODELS = (
    "veo-3.1-generate-preview",
    "veo-3.1-fast-generate-preview",
    "veo-3.1-lite-generate-preview",
)
SUPPORTED_ASPECTS = ("16:9", "9:16")
SUPPORTED_DURATIONS_S = (4, 6, 8)
SUPPORTED_RESOLUTIONS = ("720p", "1080p", "4k")
LITE_UNSUPPORTED_RESOLUTIONS = ("4k",)
OUTPUT_FPS = 24
_TIMEOUT_S = 900
_POLL_INTERVAL_S = 10


class GoogleVeoVideoError(GoogleAPIError):
    """Raised for sanitized Google Veo video invoke/response failures."""


def _req_get(request, key, default=None):
    if isinstance(request, dict):
        return request.get(key, default)
    return getattr(request, key, default)


def _env_first(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _selected_model() -> str:
    model = str(
        _env_first(
            "OTR_GOOGLE_VEO_MODEL_ID",
            "OTR_GOOGLE_VIDEO_MODEL_ID",
            "OTR_GOOGLE_VIDEO_MODEL",
        )
        or DEFAULT_MODEL
    ).strip()
    if model not in SUPPORTED_MODELS:
        raise GoogleAPIRequestShapeError(
            "google_veo_video.render_clip: unsupported model %r; expected one "
            "of %r" % (model, SUPPORTED_MODELS)
        )
    return model


def _prompt(request) -> str:
    text = str(_req_get(request, "text_prompt") or _req_get(request, "prompt") or "").strip()
    if not text:
        raise GoogleAPIRequestShapeError(
            "google_veo_video.render_clip: blank text_prompt (no request sent)"
        )
    return append_visual_safety_clause(text)


def _reject_unsupported_inputs(request) -> None:
    assets = _req_get(request, "asset_refs") or {}
    a_get = assets.get if isinstance(assets, dict) else (
        lambda k, d=None: getattr(assets, k, d)
    )
    init_image = a_get("init_image") or _req_get(request, "init_image")
    reference_images = a_get("reference_images") or _req_get(request, "reference_images")
    last_frame = a_get("last_frame") or _req_get(request, "last_frame")
    audio_ref = _req_get(request, "audio_ref")
    base_clip_ref = _req_get(request, "base_clip_ref")
    reference_videos = _req_get(request, "reference_videos")
    if (
        init_image
        or reference_images
        or last_frame
        or audio_ref
        or base_clip_ref
        or reference_videos
    ):
        raise GoogleAPIRequestShapeError(
            "google_veo_video.render_clip: this adapter is text-to-video only; "
            "image/audio/base_clip/reference inputs are not supported yet "
            "(no request sent)"
        )


def _canvas_get(request, key: str, default):
    canvas = _req_get(request, "canvas") or {}
    if isinstance(canvas, dict):
        return canvas.get(key, default)
    return getattr(canvas, key, default)


def _aspect(request) -> str:
    value = _env_first("OTR_GOOGLE_VEO_ASPECT", "OTR_GOOGLE_VIDEO_ASPECT")
    if value:
        aspect = value
    else:
        w = int(_canvas_get(request, "w", 0) or 0)
        h = int(_canvas_get(request, "h", 0) or 0)
        aspect = "9:16" if h > w > 0 else "16:9"
    if aspect not in SUPPORTED_ASPECTS:
        raise GoogleAPIRequestShapeError(
            "google_veo_video.render_clip: unsupported aspectRatio %r; expected "
            "one of %r (no request sent)" % (aspect, SUPPORTED_ASPECTS)
        )
    return aspect


def _duration_target_s(request) -> float:
    explicit = _env_first("OTR_GOOGLE_VEO_DURATION_S", "OTR_GOOGLE_VIDEO_DURATION_S")
    if explicit:
        try:
            return float(explicit)
        except ValueError as exc:
            raise GoogleAPIRequestShapeError(
                "google_veo_video.render_clip: duration env must be numeric "
                "(no request sent)"
            ) from exc
    timing = _req_get(request, "timing") or {}
    t_get = timing.get if isinstance(timing, dict) else (
        lambda k, d=None: getattr(timing, k, d)
    )
    frames = float(t_get("target_frame_count", 0) or 0)
    fps = float(_canvas_get(request, "fps", OUTPUT_FPS) or OUTPUT_FPS)
    if frames > 0 and fps > 0:
        return frames / fps
    return 8.0


def _duration_s(request, *, resolution: str) -> int:
    if resolution in ("1080p", "4k"):
        return 8
    target = _duration_target_s(request)
    if target <= 5:
        return 4
    if target <= 7:
        return 6
    return 8


def _resolution(model: str) -> str:
    resolution = str(_env_first("OTR_GOOGLE_VEO_RESOLUTION", "OTR_GOOGLE_VIDEO_RESOLUTION") or "720p").strip()
    if resolution not in SUPPORTED_RESOLUTIONS:
        raise GoogleAPIRequestShapeError(
            "google_veo_video.render_clip: unsupported resolution %r; expected "
            "one of %r (no request sent)" % (resolution, SUPPORTED_RESOLUTIONS)
        )
    if model == "veo-3.1-lite-generate-preview" and resolution in LITE_UNSUPPORTED_RESOLUTIONS:
        raise GoogleAPIRequestShapeError(
            "google_veo_video.render_clip: %s does not support resolution %r "
            "(no request sent)" % (model, resolution)
        )
    return resolution


def _request_payload(model: str, request) -> dict:
    _reject_unsupported_inputs(request)
    resolution = _resolution(model)
    duration = _duration_s(request, resolution=resolution)
    return {
        "instances": [{"prompt": _prompt(request)}],
        "parameters": {
            "aspectRatio": _aspect(request),
            "durationSeconds": duration,
            "resolution": resolution,
            "sampleCount": 1,
        },
    }


def _operation_path(name: str) -> str:
    value = str(name or "").strip()
    if not value:
        raise GoogleAPIRequestShapeError("Google Veo operation response lacked name")
    if value.startswith("http://") or value.startswith("https://"):
        return value
    if value.startswith("/"):
        return value
    return "/v1beta/%s" % value


def _extract_operation_name(response: dict) -> str:
    if not isinstance(response, dict):
        raise GoogleAPIRequestShapeError("Google Veo operation was not a JSON object")
    name = response.get("name")
    if not isinstance(name, str) or not name.strip():
        raise GoogleAPIRequestShapeError("Google Veo operation response lacked name")
    return name.strip()


def _extract_generated_video(operation: dict) -> dict:
    if not isinstance(operation, dict):
        raise GoogleAPIRequestShapeError("Google Veo operation status was not JSON")
    if operation.get("error"):
        raise GoogleVeoVideoError("Google Veo operation failed: %s" % operation["error"])
    response = operation.get("response")
    if not isinstance(response, dict):
        raise GoogleAPIRequestShapeError("Google Veo done operation lacked response")

    # Documented REST shape: response.generateVideoResponse.generatedSamples[].
    generate_response = response.get("generateVideoResponse")
    if isinstance(generate_response, dict):
        samples = generate_response.get("generatedSamples") or []
        if samples and isinstance(samples[0], dict):
            video = samples[0].get("video")
            if isinstance(video, dict):
                return video

    # SDK/camel and snake variants kept as accepted response shapes.
    videos = response.get("generatedVideos") or response.get("generated_videos") or []
    if videos and isinstance(videos[0], dict):
        video = videos[0].get("video")
        if isinstance(video, dict):
            return video
    raise GoogleAPIRequestShapeError(
        "Google Veo response did not include "
        "response.generateVideoResponse.generatedSamples[0].video or generatedVideos[0].video"
    )


def _video_uri(video: dict) -> str:
    uri = video.get("uri")
    if not isinstance(uri, str) or not uri.strip():
        raise GoogleAPIRequestShapeError("Google Veo video response lacked uri")
    return uri.strip()


def _poll_operation(name: str, *, api_key: str, timeout_s: int) -> dict:
    deadline = time.monotonic() + max(1, int(timeout_s))
    path = _operation_path(name)
    last_status: dict | None = None
    while time.monotonic() < deadline:
        status = get_json(path, timeout_s=timeout_s, _api_key=api_key)
        last_status = status
        if status.get("done") is True:
            return status
        time.sleep(_POLL_INTERVAL_S)
    raise GoogleVeoVideoError(
        "Google Veo operation %s did not finish before timeout "
        "(last_status=%r)" % (name, last_status)
    )


def _download_operation_video(operation: dict, *, api_key: str, timeout_s: int) -> bytes:
    uri = _video_uri(_extract_generated_video(operation))
    data = download_media(uri, timeout_s=timeout_s, _api_key=api_key)
    if not data:
        raise GoogleVeoVideoError("Google Veo video download was empty")
    return data


def _write_provider_video(
    operation: dict,
    *,
    api_key: str,
    timeout_s: int,
    model: str,
    request: dict,
) -> dict:
    data = _download_operation_video(operation, api_key=api_key, timeout_s=timeout_s)
    digest = hashlib.sha256(data).hexdigest()
    from .._otr_paths import otr_shared_tmp_dir

    out_dir = Path(otr_shared_tmp_dir()) / "google_veo_video"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / ("google_veo_video_%s.mp4" % digest[:16])
    path.write_bytes(data)
    operation_name = str(operation.get("name") or digest[:16])
    resolution = _resolution(model)
    return {
        "path": str(path),
        "content_type": "video/mp4",
        "duration_s": None,
        "provider_job_id": operation_name.rsplit("/", 1)[-1],
        "raw_meta": {
            "model": model,
            "output_resolution": resolution,
            "output_fps": OUTPUT_FPS,
            "output_duration_s": _duration_s(request, resolution=resolution),
            "operation_name": operation_name,
        },
    }


@register
class GoogleVeoVideoEngine:
    """Registered as ``google_veo_video``. Direct Veo text-to-video, BYO key."""

    name = "google_veo_video"
    roles = ("announcer_visual", "music_visual", "character_video")
    default_roles = ()
    commercial_clean = True
    requires_flag = None
    family = "text_to_video"
    required_inputs = ("text_prompt",)
    invocable = True
    invocability_reason = ""
    native = False
    render_aspect = "wide"

    def load(self) -> None:
        return None

    def unload(self) -> None:
        return None

    def assert_usable(self, host_caps, profile, request_template=None):  # noqa: ARG002
        import shutil

        resolve_api_key()
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            from .registry import EngineUnusable, EngineUsabilityReason

            raise EngineUnusable(
                self.name,
                self.family,
                EngineUsabilityReason.MALFORMED_CONFIG,
                "ffmpeg/ffprobe not on PATH -- Google Veo video canonicalizer "
                "strips provider audio via ffmpeg",
                kind="video",
            )
        return self.name

    def prepare(self, host_caps, profile, session_ctx):  # noqa: ARG002
        return {}

    def render_clip(self, request, prepared):  # noqa: ARG002
        model = _selected_model()
        api_key = resolve_api_key()
        payload = _request_payload(model, request)
        operation = post_json(
            f"/v1beta/models/{model}:predictLongRunning",
            payload,
            timeout_s=_TIMEOUT_S,
            _api_key=api_key,
        )
        operation_name = _extract_operation_name(operation)
        done = _poll_operation(operation_name, api_key=api_key, timeout_s=_TIMEOUT_S)
        if "name" not in done:
            done = dict(done)
            done["name"] = operation_name
        return _write_provider_video(
            done,
            api_key=api_key,
            timeout_s=_TIMEOUT_S,
            model=model,
            request=request,
        )

    def canonicalize(self, raw, request, profile):  # noqa: ARG002
        from .._otr_shared.cloud_media_canonical import (
            canonicalize_video,
            cloud_delivery_wh,
        )

        rw = int(_canvas_get(request, "w", 0) or 0)
        rh = int(_canvas_get(request, "h", 0) or 0)
        tw, th = cloud_delivery_wh(
            rw,
            rh,
            land_env="OTR_GOOGLE_VIDEO_CANVAS",
            port_env="OTR_GOOGLE_VIDEO_CANVAS_PORTRAIT",
            land_default="1280x720",
            port_default="720x1280",
        )
        asset = canonicalize_video(raw, {
            "w": tw,
            "h": th,
            "fps": int(_canvas_get(request, "fps", 25) or 25),
        })
        frame_count = int(round((asset.duration_s or 0.0) * (asset.fps or 0.0)))
        return {
            "clip_id": _req_get(request, "shot_id") or f"{self.name}_clip",
            "type": "video",
            "path": str(asset.path),
            "container": "mp4",
            "codec": "h264",
            "pixel_format": "yuv420p",
            "fps": int(asset.fps or 25),
            "frame_count": frame_count,
            "has_audio": False,
            "color_primaries": "bt709",
            "transfer": "bt709",
            "matrix": "bt709",
            "engine_id": self.name,
            "family": self.family,
            "provider_job_id": asset.provider_job_id,
            "content_sha256": asset.sha256,
            "actual_duration_s": asset.duration_s,
        }

    def teardown(self, prepared) -> None:  # noqa: ARG002
        return None


GoogleVeoVideo = GoogleVeoVideoEngine()

__all__ = [
    "DEFAULT_MODEL",
    "GoogleVeoVideo",
    "GoogleVeoVideoEngine",
    "LITE_UNSUPPORTED_RESOLUTIONS",
    "OUTPUT_FPS",
    "SUPPORTED_ASPECTS",
    "SUPPORTED_DURATIONS_S",
    "SUPPORTED_MODELS",
    "SUPPORTED_RESOLUTIONS",
    "_extract_generated_video",
    "_operation_path",
    "_request_payload",
]
