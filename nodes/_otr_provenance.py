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
  * ``meta["noncommercial_notice"]``  -- a plain warning, present ONLY when the
                                         source forbids commercial use

The spoken line names the SOURCE, never the licence identifier; the licence
text rides on the printed credit. Attribution under CC BY is satisfied "in the
manner specified", which does not mean read aloud mid-drama.

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
    "noncommercial_notice",
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


#: THE SAME FOUR STATUSES, SAID PROPERLY WHEN WE KNOW WHAT WE ADAPTED.
#: Each takes ``work_title`` and ``author``. The cc0 line keeps the exact
#: phrase "dedicated to the public domain" that
#: ``tests/test_public_domain_library_roll.py`` pins, so naming the work ADDS
#: to that sentence rather than replacing it.
_NAMED_CODA_BY_STATUS = {
    "public_domain_us": "Tonight's tale was adapted from {work_title}, by {author}.",
    "cc0": ("Tonight's tale was adapted from {work_title}, by {author}, "
            "a work dedicated to the public domain."),
    "research_only": ("Tonight's tale draws on {work_title}, by {author}, "
                      "a source cleared for research use only."),
}

#: A LICENSED source names the AUTHOR AND THE WORK, never the edition or the
#: licensor. Shakespeare's lane is `licensed_noncommercial` (Folger CC BY-NC),
#: which is why it fell through to "" and closed on "We've lost our signal."
#: naming nobody.
_LICENSED_NAMED_CODA = "Tonight's scene was drawn from {author}'s {work_title}."

_LICENSED_STATUSES = frozenset({"licensed_commercial", "licensed_noncommercial"})


def spoken_coda_line(provenance: Any, identity: Any = None) -> str:
    """A short spoken acknowledgement for the announcer coda.

    ``identity`` is an optional :class:`_otr_source_identity.SourceIdentity`.
    WITHOUT it this returns exactly what it always returned, so every existing
    caller and test is unaffected; WITH it the sentence names the work and its
    author.

    THE 2026-08-05 LICENSED-SOURCE RULING IS SUPERSEDED FOR AUTHOR AND WORK
    (operator, 2026-08-15). That ruling stopped the announcer reciting a
    LICENCE, and its own reasoning was "Folger publishes the edition and
    Shakespeare wrote the play" -- yet what shipped also stopped it naming the
    author and the play, which is the opposite of what that sentence argues.
    Naming Shakespeare is not a licence claim. The licensor and the licence are
    still never spoken; they remain print-only via ``printed_credit_line``.

    Never raises. An identity that cannot name a work degrades to the generic
    sentence, so a missing field costs the listener some detail and costs the
    render nothing.
    """
    prov = provenance if isinstance(provenance, dict) else {}
    status = str(prov.get("status") or "")

    work_title = str(getattr(identity, "work_title", "") or "").strip()
    author = str(getattr(identity, "author", "") or "").strip()
    if work_title and author:
        if status in _LICENSED_STATUSES:
            return _LICENSED_NAMED_CODA.format(
                work_title=work_title, author=author)
        template = _NAMED_CODA_BY_STATUS.get(status)
        if template:
            return template.format(work_title=work_title, author=author)

    if status in _CODA_BY_STATUS:
        return _CODA_BY_STATUS[status]
    # A LICENSED SOURCE GETS NO SPOKEN LINE (operator ruling 2026-08-05):
    # "I get it, thanks to Folger, but they didn't write it -- Shakespeare did."
    #
    # This used to append "used under CC BY-NC 3.0 (Folger Shakespeare
    # Library)", so a listener heard a licence identifier mid-drama. On
    # 2026-08-04 the licence was dropped and the EDITION NAME kept, which left
    # the announcer thanking the licensor of a public-domain play -- crediting
    # the wrong party to anyone actually listening. Now neither is spoken.
    #
    # Attribution is not lost and was never legally required in the audio: CC BY
    # asks for credit "in the manner specified", not credit read aloud, and
    # printed_credit_line() below still emits the full "adapted from <source>,
    # used under <licence>" for the credits roll, with noncommercial_notice()
    # warning the operator separately. An empty line here routes the episode to
    # its ordinary sign-off and stamps spoken_coda_source="none".
    return ""


def noncommercial_notice(provenance: Any) -> str:
    """A plain-language warning for a source that forbids commercial use.

    Empty unless the source is explicitly non-commercial. This exists because
    the rights data was already correct and threaded -- ``commercial_use_allowed``
    is validated, carried, and normalized -- but no HUMAN surface ever showed
    it, so nothing told a publisher that an episode must not be sold. Operator
    directive 2026-08-04: "tell people if they do [monetize] they need to not
    ship and sell shakespeare episodes." Never raises.
    """
    prov = provenance if isinstance(provenance, dict) else {}
    if prov.get("commercial_use_allowed") is not False:
        return ""
    src = str(prov.get("source_label") or "").strip()
    label = str(prov.get("license_label") or "").strip()
    who = f"{src} " if src else ""
    # The label often already carries the org in parentheses, so nesting a
    # second pair reads badly ("Folger Shakespeare text (CC BY-NC 3.0 (Folger
    # Shakespeare Library))").
    under = f" licensed {label}" if label else ""
    return (
        f"NON-COMMERCIAL SOURCE: this episode adapts {who}text{under}, which "
        "does NOT permit commercial use. Do not sell it, and do not publish it "
        "on a monetized channel. Personal and non-commercial sharing is fine."
    )


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
