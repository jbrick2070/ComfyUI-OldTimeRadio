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

from dataclasses import dataclass, field
from typing import Tuple

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
