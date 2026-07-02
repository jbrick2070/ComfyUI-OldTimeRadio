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
]

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


def canonicalize_image(raw: PartnerResult, request: dict, session) -> CanonicalAsset:
    """S1. Exact role canvas, sRGB PNG; portrait-hash / in-character
    checks re-run on cloud output."""
    _not_built_yet("image", "S1 (stills lane)")


def canonicalize_video(raw: PartnerResult, request: dict, session) -> CanonicalAsset:
    """S3. Role fps/res/container; provider audio ALWAYS stripped via
    ffmpeg (must_strip_audio=True across shipped rows; master audio is
    frozen upstream, mux is LAST); strip proof recorded in the cache
    manifest."""
    _not_built_yet("video", "S3 (video lane)")
