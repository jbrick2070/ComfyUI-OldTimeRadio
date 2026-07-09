"""Google Gemini/Nano Banana direct BYO-key image adapter.

Direct Google API still-image lane for OTR image roles. This is not a Comfy
Cloud Partner node and it never invokes a local image model. The first shipped
surface is text-to-image only: requests contain a prompt plus an image response
format, and any init/reference image request fails loud before network until the
editing/reference path is explicitly wired and tested.

Import-time stays light: no Google SDK, PIL, NumPy, Torch, network, or Comfy
runtime imports happen at module scope.
"""
from __future__ import annotations

import base64
import hashlib
import os
from pathlib import Path

from .registry import register
from .._otr_google_api.client import (
    GoogleAPIError,
    GoogleAPIRequestShapeError,
    create_interaction,
    resolve_api_key,
)
from .._otr_shared.role_compat import ROLES
from .._otr_story_brief_helpers import append_visual_safety_clause

DEFAULT_MODEL = "gemini-3.1-flash-image"
SUPPORTED_MODELS = (
    "gemini-3.1-flash-image",
    "gemini-3.1-flash-lite-image",
    "gemini-3-pro-image",
)
SUPPORTED_ASPECTS = (
    "1:1", "3:2", "2:3", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9",
)
REQUEST_MIME_TYPES = ("image/jpeg",)
SUPPORTED_MIME_TYPES = ("image/png", "image/jpeg")
SUPPORTED_IMAGE_SIZES = {
    "gemini-3.1-flash-image": ("1K", "2K", "4K"),
    "gemini-3.1-flash-lite-image": ("1K",),
    "gemini-3-pro-image": ("1K", "2K", "4K"),
}
_TIMEOUT_S = 300


class GoogleImageError(GoogleAPIError):
    """Raised for sanitized Google image invoke/response failures."""


def _req_get(request, key, default=None):
    if isinstance(request, dict):
        return request.get(key, default)
    return getattr(request, key, default)


def _first_present(request, *keys, default=None):
    for key in keys:
        value = _req_get(request, key, None)
        if value is not None:
            return value
    return default


def _selected_model() -> str:
    model = str(
        os.environ.get("OTR_GOOGLE_IMAGE_MODEL_ID")
        or os.environ.get("OTR_GOOGLE_IMAGE_MODEL")
        or DEFAULT_MODEL
    ).strip()
    if model not in SUPPORTED_MODELS:
        raise GoogleAPIRequestShapeError(
            "google_image.generate: unsupported model %r; expected one of %r"
            % (model, SUPPORTED_MODELS)
        )
    return model


def _selected_mime() -> str:
    mime = str(os.environ.get("OTR_GOOGLE_IMAGE_MIME") or "image/jpeg").strip().lower()
    if mime not in REQUEST_MIME_TYPES:
        raise GoogleAPIRequestShapeError(
            "google_image.generate: unsupported MIME %r; expected one of %r"
            % (mime, REQUEST_MIME_TYPES)
        )
    return mime


def _selected_image_size(model: str) -> str:
    allowed = SUPPORTED_IMAGE_SIZES[model]
    size = str(os.environ.get("OTR_GOOGLE_IMAGE_SIZE") or "1K").strip().upper()
    if size not in allowed:
        raise GoogleAPIRequestShapeError(
            "google_image.generate: image_size %r is unsupported for %s; "
            "expected one of %r" % (size, model, allowed)
        )
    return size


def _canvas_wh(request) -> tuple[int, int]:
    try:
        w = int(_first_present(request, "width", "w", default=832) or 832)
        h = int(_first_present(request, "height", "h", default=1216) or 1216)
    except (TypeError, ValueError) as exc:
        raise GoogleAPIRequestShapeError(
            "google_image.generate: request width/height must be integers"
        ) from exc
    return max(1, w), max(1, h)


def _aspect_for_request(request) -> str:
    env = str(os.environ.get("OTR_GOOGLE_IMAGE_ASPECT") or "").strip()
    if env:
        if env not in SUPPORTED_ASPECTS:
            raise GoogleAPIRequestShapeError(
                "google_image.generate: unsupported aspect_ratio %r; expected one "
                "of %r" % (env, SUPPORTED_ASPECTS)
            )
        return env
    w, h = _canvas_wh(request)
    ratio = w / float(h)
    best = min(
        SUPPORTED_ASPECTS,
        key=lambda s: abs(ratio - (_ratio_value(s))),
    )
    return best


def _ratio_value(aspect: str) -> float:
    left, right = aspect.split(":", 1)
    return float(left) / float(right)


def _prompt(request) -> str:
    text = str(_req_get(request, "prompt") or _req_get(request, "text_prompt") or "").strip()
    if not text:
        raise GoogleAPIRequestShapeError(
            "google_image.generate: blank prompt (no request sent)"
        )
    return append_visual_safety_clause(text)


def _reject_reference_inputs(request) -> None:
    init_image = _req_get(request, "init_image")
    reference_images = _req_get(request, "reference_images")
    if init_image or reference_images:
        raise GoogleAPIRequestShapeError(
            "google_image.generate: init/reference images are not supported by "
            "this text-to-image adapter yet (no request sent)"
        )


def _interaction_payload(model: str, request) -> dict:
    _reject_reference_inputs(request)
    return {
        "model": model,
        "input": _prompt(request),
        "response_format": {
            "type": "image",
            "mime_type": _selected_mime(),
            "aspect_ratio": _aspect_for_request(request),
            "image_size": _selected_image_size(model),
        },
    }


def _image_block(data: str, block: dict) -> dict:
    mime = str(block.get("mime_type") or block.get("mimeType") or "image/png").lower()
    if mime not in SUPPORTED_MIME_TYPES:
        raise GoogleAPIRequestShapeError(
            "Google image response MIME %r is unsupported; expected one of %r"
            % (mime, SUPPORTED_MIME_TYPES)
        )
    return {"data": data, "mime_type": mime}


def _extract_image_data(response: dict) -> dict:
    if not isinstance(response, dict):
        raise GoogleAPIRequestShapeError("Google image response was not a JSON object")
    for key in ("output_image", "outputImage"):
        block = response.get(key)
        if isinstance(block, dict):
            data = block.get("data")
            if isinstance(data, str) and data.strip():
                return _image_block(data, block)

    for step in response.get("steps") or []:
        if not isinstance(step, dict):
            continue
        for block in step.get("content") or []:
            if not isinstance(block, dict):
                continue
            data = block.get("data")
            kind = str(block.get("type") or "").lower()
            mime = str(block.get("mime_type") or block.get("mimeType") or "").lower()
            if (
                isinstance(data, str)
                and data.strip()
                and (kind == "image" or mime.startswith("image/"))
            ):
                return _image_block(data, block)
    raise GoogleAPIRequestShapeError(
        "Google image response did not include image data in output_image, "
        "outputImage, or steps[].content[]"
    )


def _write_provider_image(response: dict) -> dict:
    block = _extract_image_data(response)
    try:
        image_bytes = base64.b64decode(block["data"], validate=True)
    except Exception as exc:  # noqa: BLE001
        raise GoogleAPIRequestShapeError(
            "Google image response data was not valid base64: %s" % exc
        ) from exc
    if not image_bytes:
        raise GoogleImageError("Google image response bytes were empty")
    suffix = ".jpg" if block["mime_type"] == "image/jpeg" else ".png"
    digest = hashlib.sha256(image_bytes).hexdigest()
    from .._otr_paths import otr_shared_tmp_dir

    out_dir = Path(otr_shared_tmp_dir()) / "google_image"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / ("google_image_%s%s" % (digest[:16], suffix))
    path.write_bytes(image_bytes)
    return {
        "path": str(path),
        "content_type": block["mime_type"],
        "duration_s": None,
        "provider_job_id": None,
        "raw_meta": {},
    }


@register
class GoogleImageEngine:
    """Registered as ``google_image``. Direct Gemini image generation, BYO key."""

    name = "google_image"
    roles = ROLES
    default_roles = ()
    commercial_clean = True
    requires_flag = None
    required_inputs = ("text_prompt",)
    native = False

    def load(self) -> None:
        return None

    def unload(self) -> None:
        return None

    def assert_usable(self, host_caps, profile, request_template=None):  # noqa: ARG002
        resolve_api_key()
        return self.name

    def prepare(self, host_caps, profile, session_ctx):  # noqa: ARG002
        return {}

    def teardown(self, prepared) -> None:  # noqa: ARG002
        return None

    def render_image(self, request, prepared=None):  # noqa: ARG002
        model = _selected_model()
        api_key = resolve_api_key()
        payload = _interaction_payload(model, request)
        response = create_interaction(
            payload,
            timeout_s=_TIMEOUT_S,
            max_retries=0,
            _api_key=api_key,
        )
        raw = _write_provider_image(response)
        from .._otr_shared.cloud_media_canonical import canonicalize_image

        w, h = _canvas_wh(request)
        asset = canonicalize_image(raw, {"w": w, "h": h, "format": "PNG"})
        return str(asset.path)


GoogleImage = GoogleImageEngine()

__all__ = [
    "DEFAULT_MODEL",
    "GoogleImage",
    "GoogleImageEngine",
    "REQUEST_MIME_TYPES",
    "SUPPORTED_ASPECTS",
    "SUPPORTED_IMAGE_SIZES",
    "SUPPORTED_MIME_TYPES",
    "SUPPORTED_MODELS",
    "_extract_image_data",
    "_interaction_payload",
]
