"""Bake-off bank-variant id helper (pure, zero-dependency leaf module).

A bake-off variant bank id is a base bank id with a trailing ``_v2`` or ``_v3``
suffix (e.g. ``shakespeare_v2``, ``science_news_v3``). Variant rows mirror their
base family and reuse the base's story_rules pack and family-keyed behaviour
(style pool, adaptation cast-name surfacing, strict-v4 science selection). This
module maps a variant id back to its base id so those FAMILY-behaviour sites
resolve correctly.

Provenance is deliberately NOT a caller of this helper: content-authorship
(``_otr_content_authorship.build_receipt``) proves the EXACT selected lane
authored the content, so ``owner_bank`` must stay the real selected id
(``shakespeare_v2``), never the base. Base-mapping is for behaviour, not for
identity/provenance.

Kept dependency-free on purpose: it is imported by routing, story_rules,
source_payload, the style catalog, and the writer, so it must never sit on an
import cycle.
"""

from __future__ import annotations

# Ordered longest-first is unnecessary (the suffixes are disjoint and equal
# length), but the tuple is the single source of truth for the accepted set.
BAKEOFF_VARIANT_SUFFIXES: tuple[str, ...] = ("_v2", "_v3")


def base_source_bank_id(bank_id: str) -> str:
    """Return the base bank id for a bake-off variant id.

    ``<base>_v2`` / ``<base>_v3`` -> ``<base>``. A base id, or any id without a
    known variant suffix, is returned unchanged. Non-str input is returned
    as-is (callers pass registry-validated strings; this stays total).
    """
    if not isinstance(bank_id, str):
        return bank_id
    for suffix in BAKEOFF_VARIANT_SUFFIXES:
        if bank_id.endswith(suffix):
            return bank_id[: -len(suffix)]
    return bank_id


def is_bakeoff_variant(bank_id: str) -> bool:
    """True iff ``bank_id`` carries a bake-off ``_v2``/``_v3`` variant suffix."""
    return isinstance(bank_id, str) and bank_id.endswith(BAKEOFF_VARIANT_SUFFIXES)


__all__ = [
    "BAKEOFF_VARIANT_SUFFIXES",
    "base_source_bank_id",
    "is_bakeoff_variant",
]
