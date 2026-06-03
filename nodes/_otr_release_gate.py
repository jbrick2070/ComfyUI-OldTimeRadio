"""Release gate (plan E.5 / I-8, Wave 1 / 1e).

One manifest scan over every commercial-bearing item in an episode -- roles,
voice-bank entries, audio cache sidecars, and ``audio_meta`` -- enforcing the
three-state commercial rule (I-8):

  * ``commercial_clean is True``  -> clean, silent ship.
  * ``commercial_clean is False`` -> known-gated: a NON-blocking warning recorded
    into ``cast_report`` + ``audio_meta``; the episode still renders. (Under an
    optional ``strict_commercial`` release the gate blocks instead.)
  * missing / null / non-boolean  -> FAIL CLOSED stop-ship.

The gate reuses the audio-engine :class:`EngineUsabilityReason` taxonomy so the
whole stack speaks one error language: ``MALFORMED_CONFIG`` for a missing/null
commercial flag or a record that is not ``allowed_for_release`` (G0), and
``NONCOMMERCIAL_BLOCKED`` for a gated item under a strict commercial release.

Also provides the cache-write-layer filename mangle (I-8): a gated buffer's
on-disk filename is hashed so a non-commercial model identifier never leaks into
an output filename.

Import-time is side-effect-free (C-5). UTF-8, no BOM, ASCII-only source.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Iterable, Tuple

from ._otr_audio_engines import EngineUnusable, EngineUsabilityReason

# Sentinel distinguishing an absent commercial_clean key from a present None.
_MISSING = object()


@dataclass(frozen=True)
class ReleaseReport:
    """Outcome of a release scan. ``warnings`` is recorded into cast_report +
    audio_meta (I-8); ``clean`` is True iff no gated items were seen."""

    scanned: int = 0
    gated: int = 0
    warnings: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def clean(self) -> bool:
        return self.gated == 0


def _as_dict(item) -> dict:
    if hasattr(item, "to_dict"):
        return item.to_dict()
    if isinstance(item, dict):
        return item
    raise EngineUnusable(
        "<item>", "", EngineUsabilityReason.MALFORMED_CONFIG,
        f"release item must be a dict or AudioCacheRecord, got {type(item).__name__}",
    )


def _label(item: dict) -> str:
    for key in ("cache_key", "voice_ref_id", "engine_name", "role"):
        val = item.get(key)
        if val:
            return str(val)
    return "<item>"


def assert_release_clean(
    items: Iterable,
    *,
    strict_commercial: bool = False,
    require_allowed_for_release: bool = False,
) -> ReleaseReport:
    """Scan ``items`` for the three-state commercial rule; FAIL CLOSED.

    ``items`` is any mix of dicts and :class:`AudioCacheRecord`s drawn from the
    roles, the voice bank, the cache sidecars, and ``audio_meta``. Returns a
    :class:`ReleaseReport` (warnings for the gated-but-shipping items). Raises
    :class:`EngineUnusable` when an item lacks a boolean ``commercial_clean``
    (stop-ship), when ``strict_commercial`` and an item is gated, or when
    ``require_allowed_for_release`` and a record is not releasable (G0).
    """
    scanned = 0
    gated = 0
    warnings = []
    for raw in items or ():
        item = _as_dict(raw)
        scanned += 1
        label = _label(item)
        role = str(item.get("role") or "")

        commercial = item.get("commercial_clean", _MISSING)
        if commercial is _MISSING or commercial is None or not isinstance(commercial, bool):
            raise EngineUnusable(
                label, role, EngineUsabilityReason.MALFORMED_CONFIG,
                "missing/null commercial_clean -- fail-closed stop-ship (I-8)",
            )
        if commercial is False:
            gated += 1
            warnings.append(
                f"{label}: known-gated (commercial_clean=false) -- non-blocking "
                f"warning, still renders (I-8)"
            )
            if strict_commercial:
                raise EngineUnusable(
                    label, role, EngineUsabilityReason.NONCOMMERCIAL_BLOCKED,
                    "commercial_clean=false under a strict_commercial release",
                )

        if require_allowed_for_release and "allowed_for_release" in item:
            if not item.get("allowed_for_release"):
                raise EngineUnusable(
                    label, role, EngineUsabilityReason.MALFORMED_CONFIG,
                    "cache record is not allowed_for_release (G0 release refusal)",
                )

    return ReleaseReport(scanned=scanned, gated=gated, warnings=tuple(warnings))


def mangle_release_filename(filename: str, commercial_clean) -> str:
    """Cache-write-layer filename mangle (I-8).

    A commercially-clean buffer keeps its filename. A gated (or unknown) buffer's
    stem is replaced with a stable hash so a non-commercial model identifier
    never leaks into an output filename. The extension is preserved.
    """
    if commercial_clean is True:
        return filename
    base = os.path.basename(filename or "")
    stem, ext = os.path.splitext(base)
    digest = hashlib.sha256(stem.encode("utf-8")).hexdigest()[:12]
    return f"gated_{digest}{ext}"
