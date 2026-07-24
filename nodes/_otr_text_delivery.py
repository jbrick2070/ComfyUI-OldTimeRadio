"""Single TTS delivery-text resolver (720-bakeoff C2 / S2 P1.3).

The voice nodes must speak the right string for a line: the CANONICAL
text drives identity/proof, but what actually goes to the TTS engine is
the DELIVERY text. For two lane families those differ:

* LEGACY lanes stamp a pronunciation-only text_for_tts projection during
  Phase 7. The resolver uses it only when its source hash matches canonical
  text; a missing or stale stamp safely falls back to canonical text.

* CONTENT-OWNED lanes stamp the same projection before their proof-owned tail.
  Missing, empty, or stale delivery remains terminal for those lanes because
  speaking unstamped text would violate their sealed authorship contract.

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
    cascade, so the voice lane and the cascade cannot disagree about which
    family a bank belongs to. Inline safety-cleanup lanes use LEGACY delivery;
    producer-owned read-only lanes require CONTENT_OWNED stamped delivery.
    A tagged bank that cannot resolve is a structural configuration error."""
    from ._otr_freeze_cascade import resolve_freeze_policy

    policy = resolve_freeze_policy(meta or {})
    if policy.terminal_error:
        raise TextDeliveryError(policy.terminal_error)
    return LEGACY if policy.run_inline_safety_cleanup else CONTENT_OWNED


def _s(v: Any) -> str:
    return "" if v is None else str(v)


def resolve_line_delivery(
    line_row: "Dict[str, Any]",
    mode: str,
) -> "Tuple[str, str]":
    """Return ``(canonical, delivery)`` for a single line row.

    - LEGACY: use a non-empty, fresh delivery stamp when present; otherwise
      fall back to canonical text.
    - CONTENT_OWNED: require a non-empty, fresh delivery stamp.
    """
    canonical = _s(line_row.get("text"))
    delivery = _s(line_row.get("text_for_tts"))
    stored_sha = _s(line_row.get("text_for_tts_source_sha256"))
    want_sha = text_for_tts_source_sha256(canonical)
    if mode != CONTENT_OWNED:
        if delivery.strip() and stored_sha == want_sha:
            return canonical, delivery
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
