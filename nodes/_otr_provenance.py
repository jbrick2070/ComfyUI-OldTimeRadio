"""Source-provenance normalizer (v4 campaign, P1(viii)).

Different lanes carry different rights sidecars:
  * public_domain (``source_rights_from_unit``): ``license_status`` in
    {public_domain_us, cc0, research_only} + ``license_url`` + ``source_url`` +
    ``source_label``.
  * shakespeare (Folger manifest): ``license_label`` + ``commercial_use_allowed``
    (bool) + ``source_url`` / ``license_url`` / ``source_label``.
  * synthetic/original: ``license_label`` == "synthetic original".

This maps every shape onto ONE normalized record so the spoken coda and the
printed credits render the right acknowledgement without a per-lane branch, and
so a single deterministic gate can enforce the operator rule that a
``research_only`` source BLOCKS publication.

Durable ledger keys (sole writer = ``OTR_LedgerScriptWriter.run``, opt-in via
``defaults.provenance_normalize``):
  * ``meta["provenance"]``            -- the normalized record (below)
  * ``meta["provenance_coda_line"]``  -- the spoken acknowledgement
  * ``meta["credits_source_line"]``   -- the printed credit (only when the bank
                                         default did not already set one)

Terminal = ``_otr_ledger_freeze._check_g14_provenance_publish`` (in
``run_gap_audit`` -> the one path every family crosses): raises
``FreezeAssertionError`` when ``blocks_publish`` is True (research_only), so the
episode never freezes and therefore never publishes. Pure; never raises.
Self-contained. UTF-8 no BOM, SFW.
"""
from __future__ import annotations

from typing import Any, Dict

__all__ = [
    "normalize_provenance",
    "spoken_coda_line",
    "printed_credit_line",
]

# license_status values the public-domain fetcher emits.
_PD_PUBLISHABLE = {"public_domain_us", "cc0"}
_PD_BLOCKED = {"research_only"}


def normalize_provenance(source_rights: Any) -> "Dict[str, Any]":
    """Map any lane's ``source_rights`` sidecar onto one normalized record.

    Returned keys: ``status`` (public_domain_us | cc0 | research_only |
    licensed_commercial | licensed_noncommercial | synthetic | unknown),
    ``commercial_use_allowed`` (bool | None), ``license_label`` (str),
    ``source_label`` (str), ``source_url`` (str), ``license_url`` (str),
    ``blocks_publish`` (bool -- True only for research_only). Never raises."""
    sr = source_rights if isinstance(source_rights, dict) else {}

    def _s(key: str) -> str:
        return str(sr.get(key) or "").strip()

    license_label = _s("license_label")
    source_label = _s("source_label")
    source_url = _s("source_url")
    license_url = _s("license_url")
    status_raw = _s("license_status").lower()

    status = "unknown"
    commercial: "Any" = None
    if status_raw in _PD_PUBLISHABLE or status_raw in _PD_BLOCKED:
        status = status_raw
        commercial = status_raw in _PD_PUBLISHABLE  # research_only -> False
    elif "commercial_use_allowed" in sr and isinstance(
        sr.get("commercial_use_allowed"), bool
    ):
        commercial = bool(sr["commercial_use_allowed"])
        status = "licensed_commercial" if commercial else "licensed_noncommercial"
    elif license_label.lower() == "synthetic original":
        status = "synthetic"
        commercial = True

    return {
        "status": status,
        "commercial_use_allowed": commercial,
        "license_label": license_label,
        "source_label": source_label,
        "source_url": source_url,
        "license_url": license_url,
        "blocks_publish": status in _PD_BLOCKED,
    }


_CODA_BY_STATUS = {
    "public_domain_us": "Tonight's tale was adapted from a work in the public domain.",
    "cc0": "Tonight's tale was adapted from a work dedicated to the public domain.",
    "research_only": "Tonight's tale draws on a source cleared for research use only.",
    "synthetic": "Tonight's tale was an original work, created for this broadcast.",
}


def spoken_coda_line(provenance: Any) -> str:
    """A short spoken acknowledgement for the announcer coda. Empty for a
    licensed/unknown status where the pack authors its own line. Never raises."""
    prov = provenance if isinstance(provenance, dict) else {}
    status = str(prov.get("status") or "")
    if status in _CODA_BY_STATUS:
        return _CODA_BY_STATUS[status]
    if status in ("licensed_commercial", "licensed_noncommercial"):
        label = str(prov.get("license_label") or "").strip()
        src = str(prov.get("source_label") or "").strip()
        if src and label:
            return f"Tonight's tale was adapted from {src}, used under {label}."
        if src:
            return f"Tonight's tale was adapted from {src}."
    return ""


def printed_credit_line(provenance: Any) -> str:
    """The printed ``credits_source_line`` for the credits roll. Never raises."""
    prov = provenance if isinstance(provenance, dict) else {}
    status = str(prov.get("status") or "")
    src = str(prov.get("source_label") or "").strip()
    label = str(prov.get("license_label") or "").strip()
    if status in ("public_domain_us", "cc0"):
        tag = "public domain" if status == "public_domain_us" else "CC0 / public domain"
        return f"adapted from {src} ({tag})" if src else f"adapted from a {tag} work"
    if status == "research_only":
        return f"adapted from {src} (research use only)" if src else \
            "adapted from a source cleared for research use only"
    if status in ("licensed_commercial", "licensed_noncommercial"):
        if src and label:
            return f"adapted from {src}, used under {label}"
        if label:
            return f"used under {label}"
        if src:
            return f"adapted from {src}"
    if status == "synthetic":
        return "an original story generated by machine for this broadcast"
    return ""
