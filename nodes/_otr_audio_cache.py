"""Audio cache -- PROTOCOL + canonical record (plan piece 5; impl is Wave 1f).

This module defines the *interface* the Wave-1f cache implementation satisfies,
plus the single canonical sidecar record and the one place the cache key is
derived. It does NO disk IO and holds no implementation -- importing it is
side-effect-free (C-5).

Why a protocol now (before the impl):
  * **One key (I-6).** ``cache_key_for(request)`` is the single definition of how
    the audio cache keys: it is exactly the ``ResolvedVoiceRequest.cache_key``
    (sha256 over the IN_KEY identity fields). The engine never keys on a raw
    widget float; the cache never invents its own key.
  * **One record.** :class:`AudioCacheRecord` is the canonical sidecar shape the
    impl writes, the release gate (E.5) scans, and the Wave-0 cache-sidecar JSON
    schema mirrors. Coding both producer and consumer against this dataclass
    keeps them from drifting.
  * **Release safety (G0).** Every record carries ``allowed_for_release`` and
    ``commercial_clean``; the release gate refuses any record that is not
    releasable or is missing the commercial boolean (fail-closed, I-8).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields as _dc_fields
from typing import Iterable, Optional, Protocol, runtime_checkable

CACHE_SCHEMA_VERSION = "1"


# ---------------------------------------------------------------------------
# Canonical sidecar record (the JSON cache-sidecar schema in config/ mirrors it)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AudioCacheRecord:
    """One cached audio entry's metadata sidecar.

    ``cache_key`` ties the record to the frozen ``ResolvedVoiceRequest`` that
    produced it (I-6). ``request_schema_version`` drives slim migration (a record
    whose version != the build target is re-rendered, Wave 1f). The three
    ``*_version`` fields participate in IS_CHANGED so a projection/template bump
    invalidates cleanly (E.5). ``allowed_for_release`` + ``commercial_clean``
    feed the release gate (I-8).
    """

    cache_key: str
    request_schema_version: str = ""
    cache_schema_version: str = CACHE_SCHEMA_VERSION
    role: str = ""
    engine_name: str = ""
    engine_impl_version: str = ""
    voice_ref_id: Optional[str] = None
    sample_rate: int = 0
    channels: int = 1
    commercial_clean: Optional[bool] = None
    allowed_for_release: bool = False
    audio_path: str = ""
    audio_sha256: str = ""
    prepare_text_version: str = ""
    delivery_projection_version: str = ""
    engine_prompt_template_version: str = ""

    def to_dict(self) -> dict:
        """Plain dict for JSON serialization (the persisted sidecar)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "AudioCacheRecord":
        """Build from a (possibly forward-compatible) dict; unknown keys are
        ignored so a newer sidecar on disk never crashes an older reader.
        ``cache_key`` is required."""
        known = {f.name for f in _dc_fields(cls)}
        kwargs = {k: v for k, v in (data or {}).items() if k in known}
        if "cache_key" not in kwargs:
            raise ValueError("AudioCacheRecord.from_dict: missing 'cache_key'")
        return cls(**kwargs)


def cache_key_for(request) -> str:
    """The audio cache key for a ``ResolvedVoiceRequest`` -- its identity key.

    Single source of the keying rule (I-6): the cache key IS the request's
    ``cache_key`` (sha256 over IN_KEY). Anything that reads or writes the cache
    routes through here so producer and consumer can never disagree on the key.
    """
    return request.cache_key


def record_from_request(
    request,
    *,
    audio_path: str = "",
    audio_sha256: str = "",
    allowed_for_release: bool = False,
    prepare_text_version: str = "",
    delivery_projection_version: str = "",
    engine_prompt_template_version: str = "",
) -> AudioCacheRecord:
    """Build the sidecar record for a resolved request + its rendered audio.

    Pure: copies the identity-relevant fields off the frozen request so the
    Wave-1f writer and the release-gate scanner share one mapping.
    """
    return AudioCacheRecord(
        cache_key=cache_key_for(request),
        request_schema_version=getattr(request, "request_schema_version", ""),
        role=getattr(request, "role", ""),
        engine_name=getattr(request, "engine_name", ""),
        engine_impl_version=getattr(request, "engine_impl_version", ""),
        voice_ref_id=getattr(request, "voice_ref_id", None),
        sample_rate=int(getattr(request, "sample_rate", 0) or 0),
        channels=int(getattr(request, "channels", 1) or 1),
        commercial_clean=getattr(request, "commercial_clean", None),
        allowed_for_release=bool(allowed_for_release),
        audio_path=audio_path,
        audio_sha256=audio_sha256,
        prepare_text_version=prepare_text_version,
        delivery_projection_version=delivery_projection_version,
        engine_prompt_template_version=engine_prompt_template_version,
    )


# ---------------------------------------------------------------------------
# Cache PROTOCOL -- the interface Wave 1f implements
# ---------------------------------------------------------------------------


@runtime_checkable
class AudioCache(Protocol):
    """Read/write interface for the per-line audio cache (impl: Wave 1f).

    Implementations must key strictly on ``cache_key_for(request)`` and must
    persist an :class:`AudioCacheRecord` sidecar beside each cached buffer.
    """

    def key_for(self, request) -> str:
        """Return the cache key for ``request`` (``cache_key_for``)."""
        ...

    def has(self, request) -> bool:
        """True iff a cached entry exists for ``request``."""
        ...

    def get(self, request) -> Optional[AudioCacheRecord]:
        """Return the cached record for ``request`` or ``None`` on a miss."""
        ...

    def put(self, request, audio, *, allowed_for_release: bool = False) -> AudioCacheRecord:
        """Persist ``audio`` for ``request`` and return its sidecar record."""
        ...

    def iter_records(self) -> Iterable[AudioCacheRecord]:
        """Iterate every persisted record (the release-gate manifest scan)."""
        ...
