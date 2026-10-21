"""A character's photo, from the page to ComfyUI's input folder.

In My Story a person may give any character a photo. The pack reads it from
ComfyUI's input folder by file name (OTR_ImageDirector's `character_photos`
widget, one `Name = file` line per character) and paints that character's
portrait from it, so every still of the character follows the same face.

Standard library only. The image is never decoded here: its first bytes must be
a PNG, JPEG or WebP signature, it must fit under MAX_PHOTO_BYTES, and ComfyUI's
own upload route does the rest. The stored name is derived from the content,
so the same photo uploaded twice is one file and a name can never carry a path.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
import urllib.error
import urllib.request
import uuid

MAX_PHOTO_BYTES = 12 * 1024 * 1024

_SIGNATURES = (
    (b"\x89PNG\r\n\x1a\n", ".png", "image/png"),
    (b"\xff\xd8\xff", ".jpg", "image/jpeg"),
)
#: What the pack accepts as a photo file name (it re-checks on its side).
SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,96}\.(png|jpg|jpeg|webp)$", re.IGNORECASE)


class PhotoError(ValueError):
    """The photo cannot be used; the message says why, for a person."""


def _sniff(raw: bytes) -> tuple:
    for sig, ext, mime in _SIGNATURES:
        if raw.startswith(sig):
            return ext, mime
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return ".webp", "image/webp"
    raise PhotoError("that file is not a PNG, JPEG or WebP picture")


def decode_photo(data: str) -> tuple:
    """`(raw bytes, extension, mime)` from a data URL or bare base64 string."""
    if not isinstance(data, str) or not data:
        raise PhotoError("no picture was sent")
    if data.startswith("data:"):
        data = data.split(",", 1)[-1]
    # base64 is 4/3 the size of the bytes; refuse before decoding a huge body.
    if len(data) > MAX_PHOTO_BYTES * 4 // 3 + 8:
        raise PhotoError(f"the picture is over {MAX_PHOTO_BYTES // (1024 * 1024)} MB")
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        raise PhotoError("the picture did not arrive intact; try again") from None
    if len(raw) > MAX_PHOTO_BYTES:
        raise PhotoError(f"the picture is over {MAX_PHOTO_BYTES // (1024 * 1024)} MB")
    ext, mime = _sniff(raw)
    return raw, ext, mime


def stored_name(raw: bytes, ext: str) -> str:
    """The content-addressed file name the photo is stored under."""
    return "otr_cast_" + hashlib.sha256(raw).hexdigest()[:20] + ext


def _multipart(fields: dict, file_field: str, filename: str, raw: bytes, mime: str) -> tuple:
    boundary = "----RadioPlayStudio" + uuid.uuid4().hex
    parts = []
    for key, value in fields.items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'
            .encode("utf-8"))
    parts.append(
        (f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; '
         f'filename="{filename}"\r\nContent-Type: {mime}\r\n\r\n').encode("utf-8"))
    parts.append(raw)
    parts.append(f"\r\n--{boundary}--\r\n".encode("utf-8"))
    return b"".join(parts), "multipart/form-data; boundary=" + boundary


def upload_photo(comfyui_url: str, data: str, *, opener=urllib.request.urlopen) -> str:
    """Store the photo in ComfyUI's input folder and return its file name."""
    raw, ext, mime = decode_photo(data)
    name = stored_name(raw, ext)
    body, content_type = _multipart({"overwrite": "true", "type": "input"},
                                    "image", name, raw, mime)
    req = urllib.request.Request(
        comfyui_url.rstrip("/") + "/upload/image", data=body, method="POST",
        headers={"Content-Type": content_type})
    try:
        with opener(req, timeout=60) as resp:
            reply = json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as exc:
        raise PhotoError(f"ComfyUI refused the picture (HTTP {exc.code})") from None
    except (urllib.error.URLError, OSError) as exc:
        raise PhotoError(f"cannot reach ComfyUI to store the picture: {exc}") from None
    stored = str(reply.get("name") or "")
    if reply.get("subfolder") or not SAFE_NAME.match(stored):
        raise PhotoError("ComfyUI stored the picture somewhere the episode cannot read it")
    return stored
