"""Google BYO video adapters that preserve provider audio as SFX stems.

These engines are explicit opt-in siblings of the silent Google video adapters.
They never consume OTR dialogue/audio. Provider audio is extracted from the raw
provider video into a separate ``.sfx.wav`` stem, while the canonical video clip
is still conformed through ``canonicalize_video()`` and remains silent.

Cold-import clean: stdlib plus existing cold Google adapter helpers only. No
Google SDK, PIL, NumPy, Torch, network, or Comfy runtime imports at module scope.
"""
from __future__ import annotations

import base64
import hashlib
import logging
import os
import threading
import time
from pathlib import Path

from . import eng_google_omni_video as _omni
from . import eng_google_veo_video as _veo
from .registry import register
from .._otr_google_api.client import GoogleAPIRequestShapeError
from .._otr_story_brief_helpers import append_sfx_audio_safety_clause

SFX_ENGINE_MODEL_IDS = {
    "google_vid_sfx_omni": "gemini-omni-flash-preview",
    "google_vid_sfx_veo_lite": "veo-3.1-lite-generate-preview",
    "google_vid_sfx_veo_fast": "veo-3.1-fast-generate-preview",
    "google_vid_sfx_veo_pro": "veo-3.1-generate-preview",
}
SFX_ENGINE_IDS = frozenset(SFX_ENGINE_MODEL_IDS)
_VEO_SFX_ENGINE_IDS = frozenset({
    "google_vid_sfx_veo_lite",
    "google_vid_sfx_veo_fast",
    "google_vid_sfx_veo_pro",
})

# Monkeypatch-friendly seams for tests; these are the same stdlib client
# functions the silent adapters use.
resolve_api_key = _veo.resolve_api_key
post_json = _veo.post_json
get_json = _veo.get_json
download_media = _veo.download_media
create_interaction = _omni.create_interaction

log = logging.getLogger("OTR.google_vid_sfx")
_VEO_SFX_SUBMIT_LOCK = threading.Lock()
_VEO_SFX_LAST_SUBMIT_AT = 0.0


def _veo_sfx_submit_min_interval_s() -> float:
    """Minimum spacing between Veo SFX predictLongRunning submissions.

    The API Studio paid tier 1 quota for Veo long-running generation is tight
    enough that a sequential six-shot episode can still trip a rolling-minute
    429. This is pacing, not retry/fallback: each request is still attempted
    exactly once, just not bunched.
    """
    raw = os.environ.get("OTR_GOOGLE_VEO_SFX_SUBMIT_MIN_INTERVAL_S", "").strip()
    if not raw:
        return 0.0 if os.environ.get("OTR_TEST_MODE") == "1" else 65.0
    try:
        value = float(raw)
    except ValueError as exc:
        raise GoogleAPIRequestShapeError(
            "OTR_GOOGLE_VEO_SFX_SUBMIT_MIN_INTERVAL_S must be numeric"
        ) from exc
    if value < 0:
        raise GoogleAPIRequestShapeError(
            "OTR_GOOGLE_VEO_SFX_SUBMIT_MIN_INTERVAL_S must be >= 0"
        )
    return min(value, 600.0)


def _pace_veo_sfx_submit(*, engine_name: str, model: str) -> None:
    interval = _veo_sfx_submit_min_interval_s()
    if interval <= 0:
        return
    global _VEO_SFX_LAST_SUBMIT_AT
    with _VEO_SFX_SUBMIT_LOCK:
        now = time.monotonic()
        wait_s = interval - (now - _VEO_SFX_LAST_SUBMIT_AT)
        if wait_s > 0:
            log.warning(
                "[Google SFX] pacing %s/%s submit for %.1fs to respect "
                "Veo predictLongRunning quota",
                engine_name,
                model,
                wait_s,
            )
            time.sleep(wait_s)
            now = time.monotonic()
        _VEO_SFX_LAST_SUBMIT_AT = now


def _req_get(request, key, default=None):
    if isinstance(request, dict):
        return request.get(key, default)
    return getattr(request, key, default)


def _prompt(request, *, engine_name: str) -> str:
    text = str(_req_get(request, "text_prompt") or _req_get(request, "prompt") or "").strip()
    if not text:
        raise GoogleAPIRequestShapeError(
            f"{engine_name}.render_clip: blank text_prompt (no request sent)"
        )
    return append_sfx_audio_safety_clause(text)


def _assets(request) -> dict:
    assets = _req_get(request, "asset_refs") or {}
    return assets if isinstance(assets, dict) else {}


def _reject_sfx_unsupported_inputs(request, *, engine_name: str) -> None:
    audio_ref = _req_get(request, "audio_ref")
    base_clip_ref = _req_get(request, "base_clip_ref")
    reference_videos = _req_get(request, "reference_videos")
    assets = _assets(request)
    if assets.get("audio_ref") or assets.get("audio") or assets.get("sound_file"):
        audio_ref = audio_ref or assets.get("audio_ref") or assets.get("audio")
    if assets.get("reference_videos"):
        reference_videos = assets.get("reference_videos")
    if audio_ref or base_clip_ref or reference_videos:
        raise GoogleAPIRequestShapeError(
            f"{engine_name}.render_clip: OTR audio/base_clip/reference video "
            "inputs are not supported by SFX engines (no request sent)"
        )


def _veo_sfx_payload(model: str, request, *, engine_name: str) -> dict:
    _reject_sfx_unsupported_inputs(request, engine_name=engine_name)
    resolution = _veo._resolution(model)
    init_image = _veo._init_image_ref(request)
    last_frame = _veo._last_frame_ref(request)
    refs = _veo._reference_image_refs(request)
    if last_frame and not init_image:
        raise GoogleAPIRequestShapeError(
            f"{engine_name}.render_clip: last_frame requires init_image "
            "(no request sent)"
        )
    if refs and model not in _veo.REFERENCE_IMAGE_MODELS:
        raise GoogleAPIRequestShapeError(
            f"{engine_name}.render_clip: reference_images require a Veo 3.1 "
            f"or Veo 3.1 Fast model, got {model!r} (no request sent)"
        )
    duration = _veo._duration_s(
        request, resolution=resolution, has_reference_images=bool(refs))
    instance = {"prompt": _prompt(request, engine_name=engine_name)}
    if init_image:
        instance["image"] = _veo._video_image(init_image, field="init_image")
    if last_frame:
        instance["lastFrame"] = _veo._video_image(last_frame, field="last_frame")
    if refs:
        instance["referenceImages"] = [
            {
                "image": _veo._video_image(ref, field="reference_images"),
                "referenceType": "asset",
            }
            for ref in refs
        ]
    return {
        "instances": [instance],
        "parameters": {
            "aspectRatio": _veo._aspect(request),
            "durationSeconds": duration,
            "generateAudio": True,
            "resolution": resolution,
            "sampleCount": 1,
        },
    }


def _operation_path(name: str) -> str:
    return _veo._operation_path(name)


def _poll_operation(name: str, *, api_key: str, timeout_s: int) -> dict:
    deadline = time.monotonic() + max(1, int(timeout_s))
    path = _operation_path(name)
    last_status: dict | None = None
    while time.monotonic() < deadline:
        status = get_json(path, timeout_s=timeout_s, _api_key=api_key)
        last_status = status
        if status.get("done") is True:
            return status
        time.sleep(_veo._POLL_INTERVAL_S)
    raise _veo.GoogleVeoVideoError(
        "Google Veo SFX operation %s did not finish before timeout "
        "(last_status=%r)" % (name, last_status)
    )


def _download_operation_video(operation: dict, *, api_key: str, timeout_s: int) -> bytes:
    uri = _veo._video_uri(_veo._extract_generated_video(operation))
    data = download_media(uri, timeout_s=timeout_s, _api_key=api_key)
    if not data:
        raise _veo.GoogleVeoVideoError("Google Veo SFX video download was empty")
    return data


def _write_veo_provider_video(
    operation: dict,
    *,
    api_key: str,
    timeout_s: int,
    model: str,
    request: dict,
    engine_name: str,
) -> dict:
    data = _download_operation_video(operation, api_key=api_key, timeout_s=timeout_s)
    digest = hashlib.sha256(data).hexdigest()
    from .._otr_paths import otr_shared_tmp_dir

    out_dir = Path(otr_shared_tmp_dir()) / engine_name
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / ("%s_%s.mp4" % (engine_name, digest[:16]))
    path.write_bytes(data)
    operation_name = str(operation.get("name") or digest[:16])
    resolution = _veo._resolution(model)
    refs = _veo._reference_image_refs(request)
    init_image = _veo._init_image_ref(request)
    last_frame = _veo._last_frame_ref(request)
    if last_frame:
        input_mode = "interpolation"
    elif refs:
        input_mode = "reference_images"
    elif init_image:
        input_mode = "image_to_video"
    else:
        input_mode = "text_to_video"
    return {
        "path": str(path),
        "content_type": "video/mp4",
        "duration_s": None,
        "provider_job_id": operation_name.rsplit("/", 1)[-1],
        "raw_meta": {
            "model": model,
            "input_mode": input_mode,
            "input_image_count": (1 if init_image else 0) + len(refs)
            + (1 if last_frame else 0),
            "output_resolution": resolution,
            "output_fps": _veo.OUTPUT_FPS,
            "output_duration_s": _veo._duration_s(
                request, resolution=resolution,
                has_reference_images=bool(refs)),
            "operation_name": operation_name,
            "sfx_audio_requested": True,
        },
    }


def _omni_sfx_payload(model: str, request, *, engine_name: str) -> dict:
    _reject_sfx_unsupported_inputs(request, engine_name=engine_name)
    refs = _omni._image_refs(request)
    text = _prompt(request, engine_name=engine_name)
    input_value = (
        [*[_omni._image_part(ref, field="image") for ref in refs],
         {"type": "text", "text": text}]
        if refs
        else text
    )
    return {
        "model": model,
        "input": input_value,
        "response_format": {"type": "video", "delivery": "uri"},
    }


def _download_omni_video_uri(uri: str, *, api_key: str, timeout_s: int) -> bytes:
    deadline = time.monotonic() + max(1, int(timeout_s))
    file_name = _omni._file_name_from_uri(uri)
    file_path = f"/v1beta/files/{file_name}"
    last_state = ""
    while time.monotonic() < deadline:
        status = get_json(file_path, timeout_s=timeout_s, _api_key=api_key)
        state = str(status.get("state") or "").upper()
        last_state = state or last_state
        if state == "ACTIVE":
            return download_media(uri, timeout_s=timeout_s, _api_key=api_key)
        if state == "FAILED":
            raise _omni.GoogleOmniVideoError(
                "Google Omni SFX video file processing failed for %s" % file_name
            )
        time.sleep(_omni._POLL_INTERVAL_S)
    raise _omni.GoogleOmniVideoError(
        "Google Omni SFX video file %s did not become ACTIVE before timeout "
        "(last_state=%r)" % (file_name, last_state)
    )


def _omni_provider_bytes(response: dict, *, api_key: str, timeout_s: int) -> bytes:
    block = _omni._extract_video_output(response)
    if "data" in block:
        try:
            data = base64.b64decode(block["data"], validate=True)
        except Exception as exc:  # noqa: BLE001
            raise GoogleAPIRequestShapeError(
                "Google Omni SFX video data was not valid base64: %s" % exc
            ) from exc
    else:
        data = _download_omni_video_uri(
            block["uri"], api_key=api_key, timeout_s=timeout_s)
    if not data:
        raise _omni.GoogleOmniVideoError("Google Omni SFX video response bytes were empty")
    return data


def _write_omni_provider_video(
    response: dict,
    *,
    api_key: str,
    timeout_s: int,
    model: str,
    engine_name: str,
) -> dict:
    data = _omni_provider_bytes(response, api_key=api_key, timeout_s=timeout_s)
    digest = hashlib.sha256(data).hexdigest()
    from .._otr_paths import otr_shared_tmp_dir

    out_dir = Path(otr_shared_tmp_dir()) / engine_name
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / ("%s_%s.mp4" % (engine_name, digest[:16]))
    path.write_bytes(data)
    return {
        "path": str(path),
        "content_type": "video/mp4",
        "duration_s": None,
        "provider_job_id": digest[:16],
        "raw_meta": {
            "model": model,
            "output_resolution": _omni.OUTPUT_RESOLUTION,
            "output_fps": _omni.OUTPUT_FPS,
            "output_duration_range_s": _omni.OUTPUT_DURATION_RANGE_S,
            "sfx_audio_requested": True,
        },
    }


class _GoogleVidSfxBase:
    roles = ("announcer_visual", "music_visual", "character_video")
    default_roles = ()
    commercial_clean = True
    requires_flag = None
    family = "text_to_video"
    required_inputs = ("text_prompt",)
    invocable = True
    invocability_reason = ""
    native = False
    provider_side = True
    strict_text_only = False
    accepts_audio_ref = False
    accepts_base_clip_ref = False
    accepts_still = True
    accepts_init_image = True
    render_aspect = "wide"

    name = ""
    model = ""

    def load(self) -> None:
        return None

    def unload(self) -> None:
        return None

    def assert_usable(self, host_caps, profile, request_template=None):  # noqa: ARG002
        import shutil
        from .registry import EngineUnusable, EngineUsabilityReason

        resolve_api_key()
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            raise EngineUnusable(
                self.name,
                self.family,
                EngineUsabilityReason.MALFORMED_CONFIG,
                "ffmpeg/ffprobe not on PATH -- Google SFX video needs ffprobe "
                "for provider-audio detection and ffmpeg for SFX extraction",
                kind="video",
            )
        return self.name

    def prepare(self, host_caps, profile, session_ctx):  # noqa: ARG002
        return {}

    def _canvas_get(self, request, key: str, default):
        canvas = _req_get(request, "canvas") or {}
        if isinstance(canvas, dict):
            return canvas.get(key, default)
        return getattr(canvas, key, default)

    def _canonical_video_and_sfx(self, raw, request):
        from .._otr_shared.cloud_media_canonical import (
            canonicalize_video,
            cloud_delivery_wh,
            extract_sfx_bed_from_provider_video,
        )

        sfx = extract_sfx_bed_from_provider_video(raw)
        rw = int(self._canvas_get(request, "w", 0) or 0)
        rh = int(self._canvas_get(request, "h", 0) or 0)
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
            "fps": int(self._canvas_get(request, "fps", 25) or 25),
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
            "sfx_stem_path": str(sfx.path),
            "sfx_duration_s": sfx.duration_s,
            "sfx_sha256": sfx.sha256,
        }

    def teardown(self, prepared) -> None:  # noqa: ARG002
        return None


class _GoogleVeoSfxEngine(_GoogleVidSfxBase):
    accepts_reference_images = True
    accepts_last_frame = True

    def render_clip(self, request, prepared):  # noqa: ARG002
        api_key = resolve_api_key()
        payload = _veo_sfx_payload(self.model, request, engine_name=self.name)
        _pace_veo_sfx_submit(engine_name=self.name, model=self.model)
        operation = post_json(
            f"/v1beta/models/{self.model}:predictLongRunning",
            payload,
            timeout_s=_veo._TIMEOUT_S,
            _api_key=api_key,
        )
        operation_name = _veo._extract_operation_name(operation)
        done = _poll_operation(
            operation_name, api_key=api_key, timeout_s=_veo._TIMEOUT_S)
        if "name" not in done:
            done = dict(done)
            done["name"] = operation_name
        return _write_veo_provider_video(
            done,
            api_key=api_key,
            timeout_s=_veo._TIMEOUT_S,
            model=self.model,
            request=request,
            engine_name=self.name,
        )

    def canonicalize(self, raw, request, profile):  # noqa: ARG002
        return self._canonical_video_and_sfx(raw, request)


class _GoogleOmniSfxEngine(_GoogleVidSfxBase):
    accepts_reference_images = True

    def render_clip(self, request, prepared):  # noqa: ARG002
        api_key = resolve_api_key()
        payload = _omni_sfx_payload(self.model, request, engine_name=self.name)
        response = create_interaction(
            payload,
            timeout_s=_omni._TIMEOUT_S,
            max_retries=0,
            _api_key=api_key,
        )
        return _write_omni_provider_video(
            response,
            api_key=api_key,
            timeout_s=_omni._TIMEOUT_S,
            model=self.model,
            engine_name=self.name,
        )

    def canonicalize(self, raw, request, profile):  # noqa: ARG002
        return self._canonical_video_and_sfx(raw, request)


class GoogleVidSfxOmniEngine(_GoogleOmniSfxEngine):
    name = "google_vid_sfx_omni"
    model = SFX_ENGINE_MODEL_IDS[name]


class GoogleVidSfxVeoLiteEngine(_GoogleVeoSfxEngine):
    name = "google_vid_sfx_veo_lite"
    model = SFX_ENGINE_MODEL_IDS[name]


class GoogleVidSfxVeoFastEngine(_GoogleVeoSfxEngine):
    name = "google_vid_sfx_veo_fast"
    model = SFX_ENGINE_MODEL_IDS[name]


class GoogleVidSfxVeoProEngine(_GoogleVeoSfxEngine):
    name = "google_vid_sfx_veo_pro"
    model = SFX_ENGINE_MODEL_IDS[name]


GoogleVidSfxOmni = GoogleVidSfxOmniEngine()
GoogleVidSfxVeoLite = GoogleVidSfxVeoLiteEngine()
GoogleVidSfxVeoFast = GoogleVidSfxVeoFastEngine()
GoogleVidSfxVeoPro = GoogleVidSfxVeoProEngine()

for _engine in (
    GoogleVidSfxOmni,
    GoogleVidSfxVeoLite,
    GoogleVidSfxVeoFast,
    GoogleVidSfxVeoPro,
):
    register(_engine)


__all__ = [
    "GoogleVidSfxOmni",
    "GoogleVidSfxOmniEngine",
    "GoogleVidSfxVeoFast",
    "GoogleVidSfxVeoFastEngine",
    "GoogleVidSfxVeoLite",
    "GoogleVidSfxVeoLiteEngine",
    "GoogleVidSfxVeoPro",
    "GoogleVidSfxVeoProEngine",
    "SFX_ENGINE_IDS",
    "SFX_ENGINE_MODEL_IDS",
    "_omni_sfx_payload",
    "_veo_sfx_payload",
]
