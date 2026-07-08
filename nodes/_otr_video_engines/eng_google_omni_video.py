"""Google Gemini Omni Flash direct BYO-key video adapter.

Direct Google API lane for short text-to-video clips. This is not a Comfy
Cloud Partner node and it never invokes a local video model. The first shipped
surface is text-to-video only through the Interactions API; init/reference
image, audio, and video-editing inputs fail loud before network until those
paths are explicitly wired and tested.

Import-time stays light: no Google SDK, PIL, NumPy, Torch, network, or Comfy
runtime imports happen at module scope.
"""
from __future__ import annotations

import base64
import hashlib
import os
import time
from pathlib import Path

from .registry import register
from .._otr_google_api.client import (
    GoogleAPIError,
    GoogleAPIRequestShapeError,
    create_interaction,
    download_media,
    get_json,
    resolve_api_key,
)
from .._otr_story_brief_helpers import append_visual_safety_clause

DEFAULT_MODEL = "gemini-omni-flash-preview"
SUPPORTED_MODELS = (DEFAULT_MODEL,)
OUTPUT_DURATION_RANGE_S = (3, 10)
OUTPUT_RESOLUTION = "720p"
OUTPUT_FPS = 24
_TIMEOUT_S = 900
_POLL_INTERVAL_S = 5


class GoogleOmniVideoError(GoogleAPIError):
    """Raised for sanitized Google Omni video invoke/response failures."""


def _req_get(request, key, default=None):
    if isinstance(request, dict):
        return request.get(key, default)
    return getattr(request, key, default)


def _selected_model() -> str:
    model = str(
        os.environ.get("OTR_GOOGLE_OMNI_MODEL_ID")
        or os.environ.get("OTR_GOOGLE_VIDEO_MODEL_ID")
        or os.environ.get("OTR_GOOGLE_VIDEO_MODEL")
        or DEFAULT_MODEL
    ).strip()
    if model not in SUPPORTED_MODELS:
        raise GoogleAPIRequestShapeError(
            "google_omni_video.render_clip: unsupported model %r; expected one "
            "of %r" % (model, SUPPORTED_MODELS)
        )
    return model


def _prompt(request) -> str:
    text = str(_req_get(request, "text_prompt") or _req_get(request, "prompt") or "").strip()
    if not text:
        raise GoogleAPIRequestShapeError(
            "google_omni_video.render_clip: blank text_prompt (no request sent)"
        )
    return append_visual_safety_clause(text)


def _reject_unsupported_inputs(request) -> None:
    assets = _req_get(request, "asset_refs") or {}
    a_get = assets.get if isinstance(assets, dict) else (
        lambda k, d=None: getattr(assets, k, d)
    )
    init_image = a_get("init_image") or _req_get(request, "init_image")
    audio_ref = _req_get(request, "audio_ref")
    base_clip_ref = _req_get(request, "base_clip_ref")
    reference_videos = _req_get(request, "reference_videos")
    if init_image or audio_ref or base_clip_ref or reference_videos:
        raise GoogleAPIRequestShapeError(
            "google_omni_video.render_clip: this adapter is text-to-video only; "
            "init_image/audio/base_clip/reference video inputs are not supported "
            "yet (no request sent)"
        )


def _interaction_payload(model: str, request) -> dict:
    _reject_unsupported_inputs(request)
    return {
        "model": model,
        "input": _prompt(request),
        "response_format": {"type": "video", "delivery": "uri"},
    }


def _video_block(block: dict) -> dict:
    data = block.get("data")
    uri = block.get("uri")
    mime = str(block.get("mime_type") or block.get("mimeType") or "video/mp4").lower()
    if mime != "video/mp4":
        raise GoogleAPIRequestShapeError(
            "Google Omni video response MIME %r is unsupported; expected video/mp4"
            % mime
        )
    if isinstance(data, str) and data.strip():
        return {"data": data, "mime_type": mime}
    if isinstance(uri, str) and uri.strip():
        return {"uri": uri.strip(), "mime_type": mime}
    raise GoogleAPIRequestShapeError("Google Omni video block lacked data or uri")


def _extract_video_output(response: dict) -> dict:
    if not isinstance(response, dict):
        raise GoogleAPIRequestShapeError("Google Omni response was not a JSON object")
    for key in ("output_video", "outputVideo"):
        block = response.get(key)
        if isinstance(block, dict):
            return _video_block(block)

    for step in response.get("steps") or []:
        if not isinstance(step, dict):
            continue
        for block in step.get("content") or []:
            if not isinstance(block, dict):
                continue
            kind = str(block.get("type") or "").lower()
            mime = str(block.get("mime_type") or block.get("mimeType") or "").lower()
            if kind == "video" or mime.startswith("video/"):
                return _video_block(block)
    raise GoogleAPIRequestShapeError(
        "Google Omni response did not include video data or URI in output_video, "
        "outputVideo, or steps[].content[]"
    )


def _file_name_from_uri(uri: str) -> str:
    clean = str(uri or "").split("?", 1)[0].rstrip("/")
    if not clean:
        raise GoogleAPIRequestShapeError("Google Omni video URI was blank")
    name = clean.rsplit("/", 1)[-1]
    if name.endswith(":download"):
        name = name[: -len(":download")]
    if not name:
        raise GoogleAPIRequestShapeError("Google Omni video URI had no file id")
    return name


def _download_video_uri(uri: str, *, api_key: str, timeout_s: int) -> bytes:
    """Poll Gemini Files until ACTIVE, then download the mp4 bytes."""
    deadline = time.monotonic() + max(1, int(timeout_s))
    file_name = _file_name_from_uri(uri)
    file_path = f"/v1beta/files/{file_name}"
    last_state = ""
    while time.monotonic() < deadline:
        status = get_json(file_path, timeout_s=timeout_s, _api_key=api_key)
        state = str(status.get("state") or "").upper()
        last_state = state or last_state
        if state == "ACTIVE":
            return download_media(uri, timeout_s=timeout_s, _api_key=api_key)
        if state == "FAILED":
            raise GoogleOmniVideoError(
                "Google Omni video file processing failed for %s" % file_name
            )
        time.sleep(_POLL_INTERVAL_S)
    raise GoogleOmniVideoError(
        "Google Omni video file %s did not become ACTIVE before timeout "
        "(last_state=%r)" % (file_name, last_state)
    )


def _provider_bytes(response: dict, *, api_key: str, timeout_s: int) -> bytes:
    block = _extract_video_output(response)
    if "data" in block:
        try:
            data = base64.b64decode(block["data"], validate=True)
        except Exception as exc:  # noqa: BLE001
            raise GoogleAPIRequestShapeError(
                "Google Omni video data was not valid base64: %s" % exc
            ) from exc
    else:
        data = _download_video_uri(block["uri"], api_key=api_key, timeout_s=timeout_s)
    if not data:
        raise GoogleOmniVideoError("Google Omni video response bytes were empty")
    return data


def _write_provider_video(response: dict, *, api_key: str, timeout_s: int) -> dict:
    data = _provider_bytes(response, api_key=api_key, timeout_s=timeout_s)
    digest = hashlib.sha256(data).hexdigest()
    from .._otr_paths import otr_shared_tmp_dir

    out_dir = Path(otr_shared_tmp_dir()) / "google_omni_video"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / ("google_omni_video_%s.mp4" % digest[:16])
    path.write_bytes(data)
    return {
        "path": str(path),
        "content_type": "video/mp4",
        "duration_s": None,
        "provider_job_id": digest[:16],
        "raw_meta": {
            "model": _selected_model(),
            "output_resolution": OUTPUT_RESOLUTION,
            "output_fps": OUTPUT_FPS,
            "output_duration_range_s": OUTPUT_DURATION_RANGE_S,
        },
    }


def _canvas_get(request, key: str, default):
    canvas = _req_get(request, "canvas") or {}
    if isinstance(canvas, dict):
        return canvas.get(key, default)
    return getattr(canvas, key, default)


@register
class GoogleOmniVideoEngine:
    """Registered as ``google_omni_video``. Direct Omni text-to-video, BYO key."""

    name = "google_omni_video"
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
                "ffmpeg/ffprobe not on PATH -- Google Omni video canonicalizer "
                "strips provider audio via ffmpeg",
                kind="video",
            )
        return self.name

    def prepare(self, host_caps, profile, session_ctx):  # noqa: ARG002
        return {}

    def render_clip(self, request, prepared):  # noqa: ARG002
        model = _selected_model()
        api_key = resolve_api_key()
        payload = _interaction_payload(model, request)
        response = create_interaction(
            payload,
            timeout_s=_TIMEOUT_S,
            max_retries=0,
            _api_key=api_key,
        )
        return _write_provider_video(response, api_key=api_key, timeout_s=_TIMEOUT_S)

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


GoogleOmniVideo = GoogleOmniVideoEngine()

__all__ = [
    "DEFAULT_MODEL",
    "GoogleOmniVideo",
    "GoogleOmniVideoEngine",
    "OUTPUT_DURATION_RANGE_S",
    "OUTPUT_FPS",
    "OUTPUT_RESOLUTION",
    "SUPPORTED_MODELS",
    "_extract_video_output",
    "_interaction_payload",
]
