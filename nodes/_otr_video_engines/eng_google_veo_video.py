"""Google Veo 3.1 direct BYO-key video adapter.

Direct Google Gemini API lane for short video clips using Veo's
``predictLongRunning`` REST endpoint. This adapter is not a Comfy Cloud Partner
node and it never invokes a local video model. It supports text-to-video plus
Veo's raw REST start-image/reference-image request shapes. The image payload
uses the Veo/Vertex-style ``bytesBase64Encoded`` field, not Gemini
``inlineData`` or SDK-internal ``imageBytes``.

Import-time stays light: no Google SDK, PIL, NumPy, Torch, network, or Comfy
runtime imports happen at module scope.
"""
from __future__ import annotations

import hashlib
import base64
import mimetypes
import os
import time
from pathlib import Path

from .registry import register
from .frame_contract import CONTINUITY_SOFT_REFERENCE, FrameContract
from .._otr_shared.still_plan_helpers import StillPlanRow
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
#: THE CANVAS FRAME RATE, not Veo's. ``OUTPUT_FPS`` above is 24, which is what
#: Veo GENERATES at -- and it is the wrong number for a FrameContract.
#:
#: Corrected 2026-07-26 by the chunk-7a QA panel, which caught this declared as
#: (96, 144, 192) = 4/6/8 s * 24. The reasoning behind that was "Veo runs at 24
#: fps, so its frames are 24ths of a second" -- true, and irrelevant, because
#: nothing in this build ever counts a Veo frame at 24 fps. ``canonicalize()``
#: below passes ``canvas.fps`` (25) into ``canonicalize_video``, whose ffmpeg
#: ``fps=`` filter is a real duration-preserving resample, and then computes
#: ``frame_count = duration_s * asset.fps`` with ``asset.fps`` fixed at that
#: same 25. So a delivered 8-second Veo clip is 200 frames by the time anything
#: in the coverage path can see it, and 192 is not merely unusual -- it is
#: UNREACHABLE. A contract listing it would refuse every real Veo beat.
#:
#: The provider's rate matters for what Veo renders. The CANVAS rate is what
#: the partitioner counts. Those are different questions and this is the one
#: the contract answers.
CANVAS_FPS = 25
SUPPORTED_FRAMES = tuple(s * CANVAS_FPS for s in SUPPORTED_DURATIONS_S)
SUPPORTED_RESOLUTIONS = ("720p", "1080p", "4k")
LITE_UNSUPPORTED_RESOLUTIONS = ("4k",)
REFERENCE_IMAGE_MODELS = (
    "veo-3.1-generate-preview",
    "veo-3.1-fast-generate-preview",
)
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


def _ref_path(ref) -> str:
    if isinstance(ref, dict):
        for key in ("path", "file", "filename", "uri"):
            value = ref.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    if isinstance(ref, str):
        return ref.strip()
    return ""


def _assets(request) -> dict:
    assets = _req_get(request, "asset_refs") or {}
    return assets if isinstance(assets, dict) else {}


def _asset_ref(request, *keys: str):
    assets = _assets(request)
    for key in keys:
        value = assets.get(key)
        if value:
            return value
    for key in keys:
        value = _req_get(request, key)
        if value:
            return value
    return None


def _init_image_ref(request):
    return _asset_ref(request, "init_image", "image", "still")


def _last_frame_ref(request):
    return _asset_ref(request, "last_frame")


def _reference_image_refs(request) -> list:
    raw = _asset_ref(request, "reference_images")
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        refs = list(raw)
    else:
        refs = [raw]
    refs = [r for r in refs if _ref_path(r)]
    if len(refs) > 3:
        raise GoogleAPIRequestShapeError(
            "google_veo_video.render_clip: reference_images supports at most "
            "3 images (no request sent)"
        )
    return refs


def _mime_for_image(path: str) -> str:
    mime = mimetypes.guess_type(path)[0] or ""
    if mime not in ("image/png", "image/jpeg", "image/webp"):
        raise GoogleAPIRequestShapeError(
            "google_veo_video.render_clip: unsupported image MIME for %r; "
            "expected png, jpeg, or webp (no request sent)" % path
        )
    return mime


def _reject_unsupported_inputs(request) -> None:
    audio_ref = _req_get(request, "audio_ref")
    base_clip_ref = _req_get(request, "base_clip_ref")
    reference_videos = _req_get(request, "reference_videos")
    assets = _assets(request)
    if assets.get("reference_videos"):
        reference_videos = assets.get("reference_videos")
    if audio_ref or base_clip_ref or reference_videos:
        raise GoogleAPIRequestShapeError(
            "google_veo_video.render_clip: audio/base_clip/reference video "
            "inputs are not supported by this adapter yet "
            "(no request sent)"
        )


def _video_image(ref, *, field: str) -> dict:
    path = _ref_path(ref)
    if not path:
        raise GoogleAPIRequestShapeError(
            "google_veo_video.render_clip: %s image ref was blank "
            "(no request sent)" % field
        )
    p = Path(path)
    if not p.is_file():
        raise GoogleAPIRequestShapeError(
            "google_veo_video.render_clip: %s image missing/absent on disk "
            "%r (no request sent)" % (field, path)
        )
    data = p.read_bytes()
    if not data:
        raise GoogleAPIRequestShapeError(
            "google_veo_video.render_clip: %s image %r was empty "
            "(no request sent)" % (field, path)
        )
    return {
        "mimeType": _mime_for_image(path),
        "bytesBase64Encoded": base64.b64encode(data).decode("ascii"),
    }


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


def _duration_s(request, *, resolution: str, has_reference_images: bool = False) -> int:
    if resolution in ("1080p", "4k") or has_reference_images:
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
    init_image = _init_image_ref(request)
    last_frame = _last_frame_ref(request)
    refs = _reference_image_refs(request)
    if last_frame and not init_image:
        raise GoogleAPIRequestShapeError(
            "google_veo_video.render_clip: last_frame requires init_image "
            "(no request sent)"
        )
    if refs and model not in REFERENCE_IMAGE_MODELS:
        raise GoogleAPIRequestShapeError(
            "google_veo_video.render_clip: reference_images require a Veo 3.1 "
            "or Veo 3.1 Fast model, got %r (no request sent)" % model
        )
    duration = _duration_s(request, resolution=resolution,
                           has_reference_images=bool(refs))
    instance = {"prompt": _prompt(request)}
    if init_image:
        instance["image"] = _video_image(init_image, field="init_image")
    if last_frame:
        instance["lastFrame"] = _video_image(last_frame, field="last_frame")
    if refs:
        instance["referenceImages"] = [
            {
                "image": _video_image(ref, field="reference_images"),
                "referenceType": "asset",
            }
            for ref in refs
        ]
    return {
        "instances": [instance],
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
    refs = _reference_image_refs(request)
    init_image = _init_image_ref(request)
    last_frame = _last_frame_ref(request)
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
            "output_fps": OUTPUT_FPS,
            "output_duration_s": _duration_s(
                request, resolution=resolution,
                has_reference_images=bool(refs)),
            "operation_name": operation_name,
        },
    }


#: S1 (2026-07-25) per-model still plan for google_veo_video (spec
#: section 3, Shape A -- scene spine). FILE-LOCAL, fully declared.
_GOOGLE_VEO_STILL_PLAN = (
    StillPlanRow(kind="scene_open", cardinality="per_beat",
                 target_class="scene", aspect="wide", required="always",
                 framing_geometry=(
                     "full-frame macro, centered subject"),
                 style_tail_policy="full"),
    StillPlanRow(kind="scene_beat", cardinality="per_beat",
                 target_class="scene", aspect="wide", required="always",
                 framing_geometry=(
                     ("cinematic three-quarter framing, the subject shown "
                      "whole with clear space around it inside frame, "
                      "balanced composition")),
                 style_tail_policy="full"),
    StillPlanRow(kind="scene_character", cardinality="per_beat",
                 target_class="scene", aspect="wide", required="always",
                 framing_geometry=(
                     ("cinematic medium shot, the character framed within a "
                      "wide 16:9 environment, full head and shoulders with "
                      "clear headroom inside frame, face unobstructed, "
                      "balanced landscape composition")),
                 style_tail_policy="full"),
    StillPlanRow(kind="portrait", cardinality="per_subject",
                 target_class="portrait", aspect="inherit_engine",
                 required="never",
                 framing_geometry=("in-character cinematic medium shot, head "
                                   "and shoulders, face clearly visible, "
                                   "subject centred with natural headroom "
                                   "above the head (never crop the top of the "
                                   "head)"),
                 style_tail_policy="full"),
)


@register
class GoogleVeoVideoEngine:
    """Registered as ``google_veo_video``. Direct Veo video, BYO key."""

    name = "google_veo_video"
    #: THE FRAME LADDER (chunk 7a, 2026-07-26). A fixed menu, not a range:
    #: SUPPORTED_FRAMES = 100/150/200, which is Veo's published 4/6/8 SECOND
    #: menu counted at the CANVAS rate the clip is resampled to before anything
    #: reads its length. See the SUPPORTED_FRAMES comment above for why this is
    #: 25 and not Veo's own OUTPUT_FPS of 24.
    #:
    #: CONTINUITY soft_reference. Veo does accept a first frame
    #: (``instance["image"]``) and a chain would be plausible -- but plausible
    #: is not proven: the returned frame 0 is GENERATED from that image rather
    #: than being it, and this lane does not even require an init image
    #: (``required_inputs = ("text_prompt",)``). A wrong strict claim buys a
    #: visible jump at a join the plan promised was seamless. A jump cut is
    #: honest. Revisit only with a live leg that measures the seam.
    frame_contract = FrameContract(
        discrete_frames=SUPPORTED_FRAMES,
        native_fps=CANVAS_FPS,
        allow_tail_trim=True,
        continuity=CONTINUITY_SOFT_REFERENCE,
    )
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
    accepts_init_image = True
    accepts_reference_images = True
    accepts_last_frame = True
    accepts_still = True
    render_aspect = "wide"
    #: S1 per-model still plan (see ``_GOOGLE_VEO_STILL_PLAN`` above).
    still_plan = _GOOGLE_VEO_STILL_PLAN

    def load(self) -> None:
        return None

    def unload(self) -> None:
        return None

    def assert_usable(self, host_caps, profile, request_template=None):  # noqa: ARG002
        import shutil

        from .._otr_shared.ffprobe import resolve_ffprobe

        resolve_api_key()
        # ffmpeg is still PATH-only (the canonicalizer runs a literal
        # "ffmpeg"); ffprobe is found the way the probe itself finds it.
        if not shutil.which("ffmpeg") or not resolve_ffprobe():
            from .registry import EngineUnusable, EngineUsabilityReason

            raise EngineUnusable(
                self.name,
                self.family,
                EngineUsabilityReason.MALFORMED_CONFIG,
                "ffmpeg not on PATH, or no ffprobe (OTR_FFPROBE / PATH / "
                "ffmpeg sibling) -- Google Veo video canonicalizer strips "
                "provider audio via ffmpeg",
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
            # The honesty receipts (2026-08-06). Native by construction, like
            # every provider lane: the asset is downloaded whole and
            # ``frame_count`` is derived from its own measured duration and fps
            # directly above, so both counts are one derivation rather than two
            # that could drift.
            "native_frame_count": frame_count,
            "extension_mode": "none",
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


# =========================================================================== #
# PER-LANE MOTION PROMPT -- EDIT HERE (Option B, operator ruling 2026-08-27)
# =========================================================================== #
# This lane takes the ORDINARY request path (provider_side, but its prompt
# arrives as request text_prompt), so it binds a formatter like a local lane
# rather than editing a partner conditioner. Bound to the class itself:
# dispatch reads type(engine).__dict__, so nothing here can leak to a sibling.
try:
    from .motion_common import compose_parts as _parts, compose_legacy as _legacy
except ImportError:  # pragma: no cover -- flat test imports
    from motion_common import compose_parts as _parts, compose_legacy as _legacy


def _google_chronological(inputs, tail):
    """Chronological action, endpoint, then ONE camera behaviour.

    Both Google lanes want the same SHAPE and neither exposes a numeric motion
    control, so the prompt is the only lever. They are bound separately anyway,
    so either can diverge without touching the other.
    """
    core = _parts(inputs, include_camera=False)
    if not core:
        return _legacy(inputs)
    camera = str((inputs or {}).get("camera") or "").strip().strip(",")
    move = camera if camera else "one small camera move"
    return ("%s, the action developing decisively across the shot -- turning, "
            "reaching, rising or crossing the space as the beat calls for -- "
            "and finishing on a clear final position. Camera: %s. %s"
            % (core.rstrip(" ,."), move, tail))


def compose_googleveovideoengine(self, inputs):
    """`GoogleVeoVideoEngine` -- Veo 3.1 Lite default; 4/6/8s; no proven motion knob and no established cloud motion maximum, so this stays at one action arc."""
    return _google_chronological(inputs, "One continuous take, no cuts.")


GoogleVeoVideoEngine.compose_prompt = compose_googleveovideoengine
