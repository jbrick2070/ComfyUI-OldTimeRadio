"""Cloud media canonicalization contract -- S0 skeleton (pass04 sec 6).

S0 ships the TYPES + validation guard + dispatch signatures; the real
per-modality canonicalizers land with their lane sprints (S1 stills,
S2 voice/music, S3 video). Everything here is fail-closed: partial or
invalid media never reaches an episode path.

PartnerResult (DS R4 #2): the exact shape invoke_partner_node returns
and the ONLY input shape canonicalizers accept. Downloads stream to a
temp path -- adapters never hold whole media in memory.

Loudness: the reference constant is resolved at S2 from the EXISTING
local lane (verify-at-build #11) -- do not invent a value here.
LOUDNESS_REFERENCE_SOURCE documents where it must come from.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TypedDict

from .cloud_media_backend import CloudErrorCode, CloudMediaError

__all__ = [
    "PartnerResult",
    "CanonicalAsset",
    "CANONICALIZER_VERSION",
    "LOUDNESS_REFERENCE_SOURCE",
    "validate_partner_result",
    "canonicalize_audio",
    "canonicalize_image",
    "canonicalize_video",
    "cloud_delivery_wh",
]


def cloud_delivery_wh(request_w, request_h, *, land_env, port_env,
                      land_default="1920x1080", port_default="1080x1920"):
    """The TRUE 1080p cloud DELIVERY canvas (orientation-preserving), env-overridable.

    The provider clip/still is conformed to THIS canvas by ``canonicalize_*`` -- NOT
    the smaller per-family request canvas (which would DOWNSCALE a 1080p provider
    output, e.g. word_razzle's 1472x832 request -> a 1472x832 clip). CLOUD-LANE ONLY:
    no local engine reads this, so locals keep their own render resolution
    (kibitz-grounded 2026-07-03). Portrait is chosen when the request canvas is
    TALLER than wide (rh > rw) -- the cloud analogue of "HuMo portrait excepted from
    the landscape bump". Unknown / zero request -> landscape default.
    """
    try:
        rw, rh = int(request_w or 0), int(request_h or 0)
    except (TypeError, ValueError):
        rw = rh = 0
    portrait = rh > rw > 0
    spec = (os.environ.get(port_env, port_default) if portrait
            else os.environ.get(land_env, land_default))
    try:
        parts = str(spec).lower().split("x", 1)
        return (max(1, int(parts[0])), max(1, int(parts[1])))
    except (ValueError, IndexError, AttributeError):
        return (1080, 1920) if portrait else (1920, 1080)

#: bumped on ANY output-contract change (DS R3 S-2: simple integers).
CANONICALIZER_VERSION = 1

#: verify-at-build #11 -- S2 must point this at the local lane's real
#: loudness handling (constant/module), not a fresh LUFS convention.
LOUDNESS_REFERENCE_SOURCE = "UNRESOLVED: locate existing local-lane loudness reference (S2)"


class PartnerResult(TypedDict):
    """Return shape of invoke_partner_node (pass04 sec 3)."""

    path: str                 # temp file the download streamed to
    content_type: str         # e.g. "video/mp4", "audio/wav", "image/png"
    duration_s: Optional[float]
    provider_job_id: Optional[str]
    raw_meta: dict


@dataclass(frozen=True)
class CanonicalAsset:
    """Output of every canonicalizer (pass04 sec 6 / GPT R2 #9)."""

    path: Path
    sha256: str
    media_type: str  # "audio" | "image" | "video"
    duration_s: Optional[float]
    width: Optional[int]
    height: Optional[int]
    fps: Optional[float]
    container: Optional[str]
    provider_job_id: Optional[str]
    validation_warnings: tuple = ()


def validate_partner_result(raw: dict) -> PartnerResult:
    """Fail-closed shape check before any canonicalization work."""
    if not isinstance(raw, dict):
        raise CloudMediaError(CloudErrorCode.CORRUPT_OUTPUT,
                              "partner result is not a mapping")
    missing = [k for k in ("path", "content_type") if not raw.get(k)]
    if missing:
        raise CloudMediaError(CloudErrorCode.CORRUPT_OUTPUT,
                              f"partner result missing {missing}")
    path = Path(raw["path"])
    if not path.is_file() or path.stat().st_size == 0:
        raise CloudMediaError(
            CloudErrorCode.CORRUPT_OUTPUT,
            f"partner result file missing or empty: {path}")
    return PartnerResult(
        path=str(path),
        content_type=str(raw["content_type"]),
        duration_s=raw.get("duration_s"),
        provider_job_id=raw.get("provider_job_id"),
        raw_meta=dict(raw.get("raw_meta") or {}),
    )


def _not_built_yet(modality: str, sprint: str):
    raise NotImplementedError(
        f"canonicalize_{modality} lands in {sprint} (S0 ships the contract "
        f"only) -- see docs/2026-07-02-cloud-engines/roundtable/pass04_plan.md"
    )


def canonicalize_audio(raw: PartnerResult, request: dict, session) -> CanonicalAsset:
    """S2. WAV 44.1kHz, stereo_policy channels, loudness matched to the
    existing local reference, +/-250ms per-line tolerance w/ head/tail
    silence padding, actual_duration_s emitted to line metadata."""
    _not_built_yet("audio", "S2 (voice + music lane)")


def canonicalize_image(raw: PartnerResult, request: dict, session=None) -> CanonicalAsset:
    """S1 (2026-07-03). Conform a provider still to the ROLE canvas:

    - decode the provider image (PNG/JPEG/WEBP -- the bridge already streamed
      it to a temp file) and force sRGB RGB (drop alpha / ICC quirks);
    - resize to the EXACT role canvas WITHOUT distortion: scale-to-COVER then
      centre-crop to ``w x h`` (stills get Ken-Burns panned by still_pan /
      still_flat downstream, so a fill -- never letterbox bars -- is the right
      conform for a background/portrait plate);
    - re-encode a real sRGB PNG on disk (the dispatcher's ``_coerce_pixels``
      reads a ``.png`` PATH and enforces a minimum byte floor);
    - sha256 of the canonical bytes.

    ``request`` supplies ``{"w", "h"}`` (both required, fail-closed) and
    optionally ``format`` (default ``"PNG"``) + ``out_path`` (default: a fresh
    png beside the input with a ``.canon.png`` suffix). ``session`` is accepted
    for signature parity with the audio/video canonicalizers but is unused here
    (image conform is a pure pixel op)."""
    import hashlib
    validated = validate_partner_result(dict(raw))
    src = Path(validated["path"])
    try:
        w = int(request["w"])
        h = int(request["h"])
    except (KeyError, TypeError, ValueError):
        raise CloudMediaError(
            CloudErrorCode.MALFORMED_CONFIG,
            "canonicalize_image request must carry integer w/h "
            f"(got {request!r})")
    if w <= 0 or h <= 0:
        raise CloudMediaError(
            CloudErrorCode.MALFORMED_CONFIG,
            f"canonicalize_image needs positive w/h (got {w}x{h})")
    fmt = str(request.get("format") or "PNG").upper()
    if fmt != "PNG":
        raise CloudMediaError(
            CloudErrorCode.MALFORMED_CONFIG,
            f"canonicalize_image only emits PNG (got format={fmt!r})")
    out_path = Path(request.get("out_path") or
                    src.with_suffix("")).with_suffix(".canon.png")
    try:
        from PIL import Image
        with Image.open(str(src)) as im:
            im = im.convert("RGB")          # sRGB, drop alpha/palette/ICC
            src_w, src_h = im.size
            if src_w <= 0 or src_h <= 0:
                raise CloudMediaError(
                    CloudErrorCode.CORRUPT_OUTPUT,
                    f"provider image {src} decoded to a zero dimension")
            # scale-to-COVER: the larger ratio fills the canvas, centre-crop
            # the overflow -> exact w x h, no distortion, no bars.
            scale = max(w / src_w, h / src_h)
            new_w = max(w, int(round(src_w * scale)))
            new_h = max(h, int(round(src_h * scale)))
            im = im.resize((new_w, new_h), Image.LANCZOS)
            left = (new_w - w) // 2
            top = (new_h - h) // 2
            im = im.crop((left, top, left + w, top + h))
            im.save(str(out_path), format="PNG")
    except CloudMediaError:
        raise
    except Exception as exc:
        raise CloudMediaError(CloudErrorCode.CORRUPT_OUTPUT,
                              f"image conform failed for {src}: {exc}")
    if not out_path.is_file() or out_path.stat().st_size == 0:
        raise CloudMediaError(
            CloudErrorCode.CORRUPT_OUTPUT,
            f"canonical image not written or empty: {out_path}")
    sha = hashlib.sha256()
    with open(out_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            sha.update(chunk)
    warnings = ()
    if (src_w, src_h) != (w, h):
        warnings = (f"provider still {src_w}x{src_h} conformed to "
                    f"{w}x{h} (cover+crop)",)
    return CanonicalAsset(
        path=out_path,
        sha256=sha.hexdigest(),
        media_type="image",
        duration_s=None,
        width=w, height=h, fps=None,
        container="png",
        provider_job_id=validated.get("provider_job_id"),
        validation_warnings=warnings,
    )


def _ffprobe_streams(path: str) -> dict:
    """``{"video": [...], "audio": [...], "duration_s": float}`` via ffprobe.
    Fail-closed CORRUPT_OUTPUT on any probe failure -- partial media never
    proceeds."""
    import json as _json
    import subprocess
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json",
             "-show_streams", "-show_format", str(path)],
            capture_output=True, text=True, timeout=120)
        if out.returncode != 0:
            raise RuntimeError(out.stderr.strip()[-300:])
        doc = _json.loads(out.stdout or "{}")
    except CloudMediaError:
        raise
    except Exception as exc:
        raise CloudMediaError(CloudErrorCode.CORRUPT_OUTPUT,
                              f"ffprobe failed on {path}: {exc}")
    streams = doc.get("streams") or []
    dur_raw = ((doc.get("format") or {}).get("duration"))
    try:
        duration = float(dur_raw)
    except (TypeError, ValueError):
        # pass04 sec 6: actual_duration_s validators fail with a NAMED
        # missing-field error on cloud runs -- never a silent None.
        raise CloudMediaError(
            CloudErrorCode.CORRUPT_OUTPUT,
            f"provider clip {path} has no format.duration "
            f"(actual_duration_s unresolvable)")
    return {
        "video": [s for s in streams if s.get("codec_type") == "video"],
        "audio": [s for s in streams if s.get("codec_type") == "audio"],
        "duration_s": duration,
    }


def canonicalize_video(raw: PartnerResult, request: dict, session=None) -> CanonicalAsset:
    """S3 (2026-07-02). Conform a provider clip to the ROLE contract:

    - provider audio ALWAYS stripped (``-an``; must_strip_audio=True across
      shipped rows -- master audio is frozen upstream, mux is LAST), with a
      POST-STRIP PROOF (re-probe: zero audio streams) recorded on the asset;
    - role canvas (fit + pad, never distort) + role fps + h264/yuv420p/bt709
      mp4 (the CanonicalClip container contract every local engine ships);
    - ``actual_duration_s`` measured from the OUTPUT (named error when the
      provider clip carries no duration);
    - sha256 of the canonical bytes.

    ``request`` supplies ``{"w", "h", "fps"}`` (all required, fail-closed) and
    optionally ``out_path`` (default: a fresh mp4 beside the input with a
    ``.canon.mp4`` suffix)."""
    import hashlib
    import subprocess
    validated = validate_partner_result(dict(raw))
    src = Path(validated["path"])
    try:
        w = int(request["w"])
        h = int(request["h"])
        fps = int(request["fps"])
    except (KeyError, TypeError, ValueError):
        raise CloudMediaError(
            CloudErrorCode.MALFORMED_CONFIG,
            "canonicalize_video request must carry integer w/h/fps "
            f"(got {request!r})")
    probe = _ffprobe_streams(str(src))
    if not probe["video"]:
        raise CloudMediaError(CloudErrorCode.CORRUPT_OUTPUT,
                              f"provider clip {src} has no video stream")
    out_path = Path(request.get("out_path") or
                    src.with_suffix("")).with_suffix(".canon.mp4")
    vf = (f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
          f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,fps={fps},format=yuv420p")
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-an",
           "-vf", vf, "-c:v", "libx264", "-preset", "medium", "-crf", "18",
           "-colorspace", "bt709", "-color_primaries", "bt709",
           "-color_trc", "bt709", "-movflags", "+faststart", str(out_path)]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if res.returncode != 0:
            raise RuntimeError(res.stderr.strip()[-300:])
    except CloudMediaError:
        raise
    except Exception as exc:
        raise CloudMediaError(CloudErrorCode.CORRUPT_OUTPUT,
                              f"ffmpeg conform failed for {src}: {exc}")
    post = _ffprobe_streams(str(out_path))
    if post["audio"]:
        raise CloudMediaError(
            CloudErrorCode.CORRUPT_OUTPUT,
            f"audio strip FAILED: canonical {out_path} still carries "
            f"{len(post['audio'])} audio stream(s)")
    sha = hashlib.sha256()
    with open(out_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            sha.update(chunk)
    warnings = ()
    if probe["audio"]:
        warnings = (f"provider audio stripped ({len(probe['audio'])} "
                    f"stream(s); strip proof: 0 in output)",)
    return CanonicalAsset(
        path=out_path,
        sha256=sha.hexdigest(),
        media_type="video",
        duration_s=post["duration_s"],
        width=w, height=h, fps=float(fps),
        container="mp4",
        provider_job_id=validated.get("provider_job_id"),
        validation_warnings=warnings,
    )
