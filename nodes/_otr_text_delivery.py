"""Single TTS delivery-text resolver (720-bakeoff C2 / S2 P1.3).

The voice nodes must speak the right string for a line: the CANONICAL
text drives identity/proof, but what actually goes to the TTS engine is
the DELIVERY text. For two lane families those differ:

* LEGACY lanes (science_news, public_domain, shakespeare, media_archive,
  untagged ledgers) run `phase_7_audio_readiness`, which rewrites
  `line.text` in place -- so by the time the voice node reads it, the
  canonical text IS already the delivery text. The resolver returns it
  UNCHANGED: passthrough, byte-identical to the pre-C2 spine. This is the
  contract the science_news byte-parity fixture locks.

* CONTENT-OWNED lanes (scifi_fable2 and any pack that owns its content
  loop) keep canonical `text` sealed against a proof map, so Phase 7 is
  skipped by policy. Instead they STAMP the delivery onto a separate
  `text_for_tts` field (`_otr_readiness.stamp_text_for_tts_delivery`).
  The resolver returns that stamped field -- but ONLY after verifying it
  is present, non-empty, AND fresh (its stored source-sha matches the
  live canonical text). An absent / empty / STALE stamp is a terminal
  error raised BEFORE any audio is generated: the pipeline must re-stamp,
  never speak text that does not match the sealed canonical line.

This is the ONE place the canonical-vs-delivery decision is made. Every
voice-node surface (line filtering, neutral prep, adapter prep, delivery
vectors, request hashing) consumes its result so the two never drift.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

from ._otr_readiness import text_for_tts_source_sha256

LEGACY = "legacy"
CONTENT_OWNED = "content_owned"


class TextDeliveryError(RuntimeError):
    """A content-owned line reached the voice gate without a valid
    ``text_for_tts`` stamp (absent / empty / stale). Terminal BEFORE
    generation -- the delivery text must match the sealed canonical
    line, never a stale or missing value."""


def delivery_mode_for_meta(meta: "Dict[str, Any] | None") -> str:
    """Resolve the delivery mode for an episode from its ledger meta.

    Uses the SAME `resolve_freeze_policy` decision that gates the freeze
    cascade, so the voice lane and the cascade can never disagree about
    which family a bank belongs to. `run_legacy_content_passes` True
    (legacy / untagged) -> LEGACY passthrough; False (content-owned) ->
    CONTENT_OWNED stamped delivery."""
    try:
        from ._otr_freeze_cascade import resolve_freeze_policy
        policy = resolve_freeze_policy(meta or {})
        return LEGACY if policy.run_legacy_content_passes else CONTENT_OWNED
    except Exception:  # noqa: BLE001 -- resolver import/edge: default legacy
        # A resolution failure defaults to LEGACY passthrough: the
        # conservative choice preserves the byte-identical spine and never
        # invents a stamped-delivery requirement for a lane that has none.
        return LEGACY


def _s(v: Any) -> str:
    return "" if v is None else str(v)


def resolve_line_delivery(
    line_row: "Dict[str, Any]",
    mode: str,
) -> "Tuple[str, str]":
    """Return ``(canonical, delivery)`` for a single line row.

    - LEGACY: ``delivery == canonical`` (passthrough). Zero behavior
      change: canonical text was already normalized in place by Phase 7.
    - CONTENT_OWNED: ``delivery`` is the stamped ``text_for_tts``, but
      only after verifying it is present, non-empty, and its stored
      ``text_for_tts_source_sha256`` matches the live canonical text.
      Otherwise raise `TextDeliveryError` (terminal before generation).
    """
    canonical = _s(line_row.get("text"))
    if mode != CONTENT_OWNED:
        return canonical, canonical

    delivery = _s(line_row.get("text_for_tts"))
    stored_sha = _s(line_row.get("text_for_tts_source_sha256"))
    line_id = _s(line_row.get("line_id")) or "?"
    if not delivery.strip():
        raise TextDeliveryError(
            f"content-owned line {line_id!r} has no text_for_tts delivery "
            f"stamp (canonical={canonical[:60]!r}). The content-owned lane "
            f"must stamp delivery before the voice gate -- refusing to "
            f"speak an unstamped line."
        )
    want_sha = text_for_tts_source_sha256(canonical)
    if not stored_sha or stored_sha != want_sha:
        raise TextDeliveryError(
            f"content-owned line {line_id!r} has a STALE text_for_tts "
            f"stamp: stored source sha {stored_sha or '<none>'} != live "
            f"canonical sha {want_sha} (the canonical text changed after "
            f"the delivery was stamped). Re-stamp delivery -- refusing to "
            f"speak text that does not match the sealed line."
        )
    return canonical, delivery


__all__ = [
    "LEGACY",
    "CONTENT_OWNED",
    "TextDeliveryError",
    "delivery_mode_for_meta",
    "resolve_line_delivery",
]
