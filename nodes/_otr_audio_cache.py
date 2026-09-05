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

import hashlib
import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, fields as _dc_fields
from typing import Iterable, List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np

log = logging.getLogger("OTR")

CACHE_SCHEMA_VERSION = "2"


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
    actual_sample_rate: Optional[int] = None
    provider_model_id: str = ""
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
    actual_sample_rate: Optional[int] = None,
    provider_model_id: str = "",
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
        actual_sample_rate=(int(actual_sample_rate) if actual_sample_rate is not None else None),
        provider_model_id=str(provider_model_id or ""),
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


    def get(self, request) -> Optional[AudioCacheRecord]:
        """Return the cached record for ``request`` or ``None`` on a miss."""
        ...

    def put(
        self, request, audio, *,
        allowed_for_release: bool = False,
        actual_sample_rate: Optional[int] = None,
        provider_model_id: str = "",
    ) -> AudioCacheRecord:
        """Persist ``audio`` for ``request`` and return its sidecar record."""
        ...

    def load(self, request) -> "Optional[Tuple[dict, AudioCacheRecord]]":
        """Return (audio_dict, record) on a verified hit; None on any failure
        (missing/corrupt payload, hash mismatch, rate/channel drift, etc.).
        Corruption is a silent miss with ONE bounded warning log line."""
        ...

    def iter_records(self) -> Iterable[AudioCacheRecord]:
        """Iterate every persisted record (the release-gate manifest scan)."""
        ...


# ===========================================================================
# Wave 1f -- implementation + slim migration
# ===========================================================================
from ._otr_resolved_request import REQUEST_SCHEMA_VERSION  # noqa: E402

# The registry-path (cast-locked) ledger schema. A pre-cast-lock ledger replayed
# through the registry path predates voice_ref_id and must be re-rendered.
LEDGER_SCHEMA_VERSION_TARGET = "2"


def needs_rerender(record, *, target_request_schema_version: str = REQUEST_SCHEMA_VERSION) -> bool:
    """True iff a cached record's request schema differs from the build target.

    The slim-migration rule (G0): a record whose ``request_schema_version`` is not
    the current target is treated as a cache MISS so the audio is re-rendered.
    """
    return str(getattr(record, "request_schema_version", "")) != str(target_request_schema_version)


class FileAudioCache:
    """File-backed :class:`AudioCache` implementation (G0).

    One sidecar JSON (``<cache_key>.json``) plus one audio buffer
    (``<cache_key>.npy``) per entry, both named by the I-6 cache key, so a gated
    model identifier can never leak into a cache filename (the sha key is already
    opaque). All disk IO happens in the methods -- constructing the cache does no
    IO (C-5). ``get`` applies the slim migration: a version-drifted record reads
    as a miss so the audio is re-rendered.
    """

    def __init__(self, cache_dir, *, request_schema_version: str = REQUEST_SCHEMA_VERSION):
        self.cache_dir = str(cache_dir)
        self.request_schema_version = str(request_schema_version)

    # -- key / paths --
    def key_for(self, request) -> str:
        return cache_key_for(request)

    def _sidecar_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.json")

    # -- read --
    def has(self, request) -> bool:
        return os.path.exists(self._sidecar_path(self.key_for(request)))

    def get(self, request) -> Optional[AudioCacheRecord]:
        key = self.key_for(request)
        path = self._sidecar_path(key)
        if not os.path.exists(path):
            return None                    # never cached -- the definitional miss
        try:
            with open(path, "r", encoding="utf-8") as fh:
                record = AudioCacheRecord.from_dict(json.load(fh))
        except Exception as exc:  # noqa: BLE001 -- still a miss, but NOT a silent one
            # A PRESENT-but-unparseable sidecar is the worst corruption this
            # cache can show, and it used to be the only one that said nothing
            # while nine lesser ones warned. `put()` publishes the sidecar LAST
            # via os.replace precisely so its presence IS the commit signal, so
            # a garbled one means the commit marker itself is damaged -- torn
            # write, external tool, or BOM contamination from a stray
            # PowerShell cmdlet, which this project has met before.
            # It is also a CLOUD cache: swallowing this silently re-bills the
            # provider and leaves no trace of why.
            log.warning("[OTR audio cache] unreadable sidecar key=%s cls=%s: %s",
                        key, type(exc).__name__, exc)
            return None
        if needs_rerender(record, target_request_schema_version=self.request_schema_version):
            return None  # slim migration: schema drift -> re-render
        return record

    # -- write --
    def put(
        self, request, audio, *,
        allowed_for_release: bool = False,
        actual_sample_rate: Optional[int] = None,
        provider_model_id: str = "",
    ) -> AudioCacheRecord:
        key = self.key_for(request)
        os.makedirs(self.cache_dir, exist_ok=True)
        audio_path, audio_sha = self._write_audio_atomic(audio, self.cache_dir, key)
        record = record_from_request(
            request,
            audio_path=audio_path,
            audio_sha256=audio_sha,
            allowed_for_release=allowed_for_release,
            prepare_text_version=_prepare_text_version(),
            delivery_projection_version=_delivery_projection_version(),
            engine_prompt_template_version="1",
            actual_sample_rate=actual_sample_rate,
            provider_model_id=provider_model_id,
        )
        # Sidecar published LAST via tmp + os.replace so sidecar presence is
        # the "committed" signal. A crash between .npy replace and sidecar
        # replace leaves an orphan .npy that the next put overwrites.
        sidecar_final = self._sidecar_path(key)
        fd, tmp_side = tempfile.mkstemp(
            prefix=f"{key}.", suffix=".json.tmp", dir=self.cache_dir,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(record.to_dict(), fh, sort_keys=True, separators=(",", ":"))
            os.replace(tmp_side, sidecar_final)
        except Exception:
            try:
                os.unlink(tmp_side)
            except OSError:
                pass
            raise
        return record

    # -- load (verified reconstruction; corruption / drift is silent miss) --
    def load(self, request) -> "Optional[Tuple[dict, AudioCacheRecord]]":
        """Return (audio_dict, record) on a verified hit, else None. Emits ONE
        bounded log.warning per miss-due-to-corruption naming the cache key
        and failure class."""
        try:
            rec = self.get(request)
            if rec is None:
                return None
            key = self.key_for(request)
            if rec.cache_key != key:
                log.warning("[OTR audio cache] cache_key mismatch on load key=%s", key)
                return None
            # r4 Fable gate SF#3: derive the payload path from cache_dir + key
            # rather than trusting the sidecar's absolute audio_path. When an
            # episode dir is renamed / moved (pipeline's rename_episode step),
            # the absolute path stored at put-time is stale but the payload
            # traveled WITH the sidecar; derivation lets the hit survive the
            # rename and keeps rec.audio_path as forensics only.
            audio_path = os.path.join(self.cache_dir, f"{key}.npy")
            if not os.path.isfile(audio_path):
                log.warning("[OTR audio cache] audio file missing on load key=%s", key)
                return None
            req_sr = int(getattr(request, "sample_rate", 0) or 0)
            req_ch = int(getattr(request, "channels", 0) or 0)
            if int(rec.sample_rate or 0) != req_sr:
                log.warning("[OTR audio cache] sample_rate mismatch on load key=%s", key)
                return None
            if int(rec.channels or 0) != req_ch:
                log.warning("[OTR audio cache] channels mismatch on load key=%s", key)
                return None
            arr = np.load(audio_path, allow_pickle=False)
            if not hasattr(arr, "dtype"):
                log.warning("[OTR audio cache] npy load produced non-array key=%s", key)
                return None
            digest = hashlib.sha256(
                f"{arr.dtype}|{arr.shape}|".encode("utf-8") + arr.tobytes()
            ).hexdigest()
            if rec.audio_sha256 and digest != rec.audio_sha256:
                log.warning("[OTR audio cache] sha256 mismatch on load key=%s", key)
                return None
            if arr.ndim != 3 or int(arr.shape[1]) != (req_ch or 1):
                log.warning(
                    "[OTR audio cache] shape mismatch on load key=%s shape=%s",
                    key, tuple(arr.shape),
                )
                return None
            arr = np.ascontiguousarray(arr).astype(np.float32, copy=False)
            sr = rec.actual_sample_rate or rec.sample_rate
            if not isinstance(sr, int) or sr <= 0:
                log.warning("[OTR audio cache] invalid sample_rate on load key=%s", key)
                return None
            import torch  # lazy: keep import-time light (C-5)

            audio = {"waveform": torch.from_numpy(arr), "sample_rate": int(sr)}
            from ._otr_resolved_request import assert_audio_batch_contract

            assert_audio_batch_contract(audio, where="FileAudioCache.load")
            return audio, rec
        except Exception as exc:  # noqa: BLE001 -- whole-body miss boundary
            log.warning(
                "[OTR audio cache] load failed key=%s cls=%s: %s",
                getattr(request, "cache_key", "?"), type(exc).__name__, exc,
            )
            return None

    # -- scan --
    def iter_records(self) -> Iterable[AudioCacheRecord]:
        if not os.path.isdir(self.cache_dir):
            return
        for name in sorted(os.listdir(self.cache_dir)):
            if not name.endswith(".json"):
                continue
            try:
                with open(os.path.join(self.cache_dir, name), "r", encoding="utf-8") as fh:
                    yield AudioCacheRecord.from_dict(json.load(fh))
            except Exception:  # noqa: BLE001 -- skip an unreadable sidecar
                continue

    def releasable_records(self) -> List[AudioCacheRecord]:
        """Records marked ``allowed_for_release`` (the release manifest, G0)."""
        return [r for r in self.iter_records() if r.allowed_for_release]

    @staticmethod
    def _write_audio_atomic(audio, cache_dir: str, key: str) -> Tuple[str, str]:
        """Persist AUDIO to <cache_dir>/<key>.npy atomically via tempfile +
        os.replace. Returns (final_path, sha256 over dtype|shape|bytes).

        Hashing includes dtype and shape so header corruption cannot
        reinterpret identical payload bytes and still pass a hash check.
        """
        wf = audio["waveform"] if isinstance(audio, dict) else audio
        if hasattr(wf, "detach"):
            arr = wf.detach().to("cpu").contiguous().numpy()
        else:
            arr = np.asarray(wf)
        arr = np.ascontiguousarray(arr)
        fd, tmp_path = tempfile.mkstemp(
            prefix=f"{key}.", suffix=".npy.tmp", dir=cache_dir,
        )
        try:
            with os.fdopen(fd, "wb") as fh:
                np.save(fh, arr)
            final = os.path.join(cache_dir, f"{key}.npy")
            os.replace(tmp_path, final)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        digest = hashlib.sha256(
            f"{arr.dtype}|{arr.shape}|".encode("utf-8") + arr.tobytes()
        ).hexdigest()
        return final, digest


def _prepare_text_version() -> str:
    try:
        from ._otr_script_prep import PREPARE_TEXT_VERSION

        return PREPARE_TEXT_VERSION
    except Exception:  # noqa: BLE001
        return ""


def _delivery_projection_version() -> str:
    try:
        from ._otr_delivery_profiles import DELIVERY_PROJECTION_VERSION

        return DELIVERY_PROJECTION_VERSION
    except Exception:  # noqa: BLE001
        return ""
