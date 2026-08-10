"""Provenance for every CONCRETE provider model this pack ships.

WHY THIS EXISTS. `tencent/hy3:free` sat in the OpenRouter dropdown until its
promo ended and the slug stopped resolving. Chunk A (2026-08-07) fixed that ONE
lane. The same "concrete version pin, no date, nothing able to notice it went
stale" shape was left live in six others. This module closes that, and
`tests/test_slug_provenance.py` enforces it.

WHAT CHANGED 2026-08-10, AND WHY IT HAD TO. The old schema was
`slug -> (lane, date | UNVERIFIED)`, keyed by the SHIPPED SELECTOR. That guard
read as a totality claim and was not one: `eng_cloud_image` ships the display
name `Nano Banana 2 (Gemini 3.1 Flash Image)`, and `cloud_media_invoke.py`
resolves it at invoke time to the real id `gemini-3.1-flash-image-preview`. The
guard was green while dating the LABEL and never seeing the id that actually went
on the wire. Provenance is now keyed by **`(provider_id, authority_lane)`** and
built from `_otr_shared.slug_inventory`, the same collector the verifier uses, so
there is no second list to drift.

The key is a PAIR because one id can legitimately arrive through two authorities:
`gemini-3.1-flash-lite-image` is offered directly by `eng_google_image` (BYO key,
Google decides) and as the Comfy selector `Nano Banana 2 Lite` (Comfy's partner
catalog decides whether Comfy serves it). Collapsing those loses a real fact.

THE HONEST PART, and why entries are not all dates. A date is a CLAIM that a
human checked the model against the authority that decides whether it resolves.
Stamping today on thirty-odd unchecked ids would manufacture exactly the defect
this pack keeps finding -- a field that reads as evidence and is not (BUG-12.86).
So `verification_kind` says what KIND of checking happened, and the date is only
ever attached to a kind that had one.

WHY KIND AND DATE ARE TWO FIELDS. Both review lanes proposed encoding this as
`"kind@YYYY-MM-DD"`. Packing two facts into one string is the positional-parse
defect already present elsewhere in this pack; it is not repeated here.

WHAT THIS TABLE COVERS, SAID OUT LOUD. It covers the CHECKED-IN static and
default model selections of the listed lanes and registered engines -- and
nothing more. It is NOT a claim that the pack ships no unlisted model:
`_otr_google_api/models.py` folds disk-cached ids into slot choices at runtime,
so no static enumeration can cover every runtime-selectable model. Understating a
guard is the cheap failure; overstating one is how BUG-12.86 got promoted, and
that bug is why this file exists.

Pure data + stdlib. No engine imports, so this module stays cold-import clean.
"""
from __future__ import annotations

from typing import Dict, NamedTuple, Optional, Tuple

# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #

#: A hit in the authority's OWN catalog. This is verification.
CATALOG_LISTED = "catalog_listed"

#: Seen upstream, but NOT in the authority that decides. The six Comfy LLM rows
#: were checked against OpenRouter's catalog, which proves the id exists in the
#: world -- not that Comfy Cloud serves it. Recording that as verification would
#: overstate it, and recording it as nothing would throw away real work.
SIGNAL_LISTED = "signal_listed"

#: Nobody has checked, or the check could not settle it. No date, ever.
UNVERIFIED = "unverified"

VERIFICATION_KINDS = (CATALOG_LISTED, SIGNAL_LISTED, UNVERIFIED)

#: The kinds that REQUIRE a real, non-future ISO date. `unverified` requires the
#: absence of one -- a date on an unverified row is a contradiction, not a bonus.
DATED_KINDS = (CATALOG_LISTED, SIGNAL_LISTED)


class ProvenanceRecord(NamedTuple):
    """What is known about one `(provider_id, authority_lane)` pair."""

    authority_lane: str
    verification_kind: str
    verified_on: Optional[str]


#: Who can settle each lane. Naming the authority is the difference between a
#: TODO and an actionable one.
LANE_AUTHORITY: Dict[str, str] = {
    "comfy": "Comfy Cloud partner-node catalog (NOT OpenRouter -- presence "
             "there is a signal, not proof that Comfy serves it)",
    "comfy_image": "Comfy Cloud partner-node catalog (display names)",
    "elevenlabs": "ElevenLabs GET /v1/models",
    "google_api": "Google generativelanguage models.list",
    "google_image": "Google generativelanguage models.list, image-capable",
    "google_lyria": "Google generativelanguage models.list, music-capable",
    "google_tts": "Google generativelanguage models.list, speech-capable",
}


def _catalog(when: str) -> ProvenanceRecord:
    return ProvenanceRecord("", CATALOG_LISTED, when)


def _signal(when: str) -> ProvenanceRecord:
    return ProvenanceRecord("", SIGNAL_LISTED, when)


_UNSET = ProvenanceRecord("", UNVERIFIED, None)

#: The 2026-08-10 direct-Google catalog run. A COMPLETE terminal-page listing of
#: 52 ids; measurement recorded in
#: `docs/2026-08-10-MEASUREMENT-google-catalog-slug-provenance.md`. Only a
#: completed listing may support any verdict, and this one was.
_GOOGLE_RUN = "2026-08-10"

#: The 2026-08-07 OpenRouter sweep. A SIGNAL for the Comfy lane, never authority.
_COMFY_SIGNAL = "2026-08-07"

#: (provider_id, authority_lane) -> ProvenanceRecord.
#:
#: `authority_lane` is repeated inside the record so a row read on its own still
#: says who owns it; the test asserts the two agree.
SLUG_PROVENANCE: Dict[Tuple[str, str], ProvenanceRecord] = {
    # --- comfy LLM lane: OpenRouter-sighted, Comfy-unproven ------------------
    ("google/gemini-3.5-flash", "comfy"):      _signal(_COMFY_SIGNAL),
    ("deepseek/deepseek-v3.2", "comfy"):       _signal(_COMFY_SIGNAL),
    ("mistralai/mistral-large-2512", "comfy"): _signal(_COMFY_SIGNAL),
    ("x-ai/grok-4.20", "comfy"):               _signal(_COMFY_SIGNAL),
    ("openai/gpt-5.5", "comfy"):               _signal(_COMFY_SIGNAL),
    ("anthropic/claude-opus-4.7", "comfy"):    _signal(_COMFY_SIGNAL),

    # --- google_api text -----------------------------------------------------
    # Pointers (`*-latest`) carry no version claim and are excluded from this
    # table entirely, by `is_pointer`.
    ("gemini-3.5-flash", "google_api"):      _catalog(_GOOGLE_RUN),
    ("gemini-3.1-flash-lite", "google_api"): _catalog(_GOOGLE_RUN),
    ("gemini-2.5-flash", "google_api"):      _catalog(_GOOGLE_RUN),
    ("gemini-2.5-pro", "google_api"):        _catalog(_GOOGLE_RUN),
    # MEASURED ABSENT on 2026-08-10 -- both were looked for in the completed
    # listing and were not there. They stay `unverified` rather than gaining a
    # fourth "checked and gone" kind, because absence is a LIVE fact the verifier
    # reports (exit 2) and freezing it into data would be a date-shaped claim
    # about someone else's service. The verifier names them on every run.
    ("gemini-2.0-flash", "google_api"):      _UNSET,
    ("gemini-2.0-flash-lite", "google_api"): _UNSET,

    # --- elevenlabs: its own endpoint was not fetched ------------------------
    ("eleven_multilingual_v2", "elevenlabs"): _UNSET,
    ("eleven_v3", "elevenlabs"):              _UNSET,

    # --- comfy_image: display selectors, Comfy is the authority --------------
    # These four groups have no reachable catalog, so nothing can settle them
    # short of a Comfy partner-catalog fetch.
    ("seedream 5.0 lite", "comfy_image"):   _UNSET,
    ("seedream-4-5-251128", "comfy_image"): _UNSET,
    ("seedream-4-0-250828", "comfy_image"): _UNSET,
    ("Krea 2 Medium Turbo", "comfy_image"): _UNSET,
    ("Krea 2 Medium", "comfy_image"):       _UNSET,
    ("Krea 2 Large", "comfy_image"):        _UNSET,
    ("photon-flash-1", "comfy_image"):      _UNSET,
    ("photon-1", "comfy_image"):            _UNSET,
    # The nano rows resolve to real Google ids, so Google's catalog CAN see
    # them -- but Comfy still decides whether Comfy serves them, which is why a
    # hit here is a SIGNAL and not authority verification.
    ("gemini-3.1-flash-lite-image", "comfy_image"): _signal(_GOOGLE_RUN),
    # THE NAMED EXCEPTION (build spec section 0A). Present in the 2026-08-10
    # catalog, and that settles nothing: runtime sends this id to a Vertex proxy
    # -- `cloud_media_invoke.py` `/proxy/vertexai/gemini/{model_id}` -- not to
    # the endpoint that was fetched. Catalog presence proves the id is LISTED,
    # never that the ROUTE works, and dating it off the wrong endpoint would be
    # the evidence-shaped field this module exists to refuse.
    ("gemini-3.1-flash-image-preview", "comfy_image"): _UNSET,

    # --- google_image / lyria / tts: direct authority, all seen --------------
    ("gemini-3.1-flash-image", "google_image"):      _catalog(_GOOGLE_RUN),
    ("gemini-3.1-flash-lite-image", "google_image"): _catalog(_GOOGLE_RUN),
    ("gemini-3-pro-image", "google_image"):          _catalog(_GOOGLE_RUN),
    ("lyria-3-clip-preview", "google_lyria"):        _catalog(_GOOGLE_RUN),
    ("gemini-2.5-flash-preview-tts", "google_tts"):  _catalog(_GOOGLE_RUN),
    ("gemini-2.5-pro-preview-tts", "google_tts"):    _catalog(_GOOGLE_RUN),
    ("gemini-3.1-flash-tts-preview", "google_tts"):  _catalog(_GOOGLE_RUN),
}

# Fill in the lane each record belongs to, from its own key. Written once, here,
# so a hand-edited row cannot disagree with its key -- and the test still checks.
SLUG_PROVENANCE = {
    key: rec._replace(authority_lane=key[1])
    for key, rec in SLUG_PROVENANCE.items()
}

# --------------------------------------------------------------------------- #
# Preview rules
# --------------------------------------------------------------------------- #

#: A slug carrying one of these advertises a PRICE inside an IDENTIFIER.
#: Promises expire; identifiers do not.
FREE_MARKERS = (":free", "-free")

_PREVIEW_TOKEN = "preview"

#: Preview ids allowed to rest at `unverified`, each with a stated reason. An
#: EXCEPTION LIST, not a silent skip: without it the preview rule would be
#: unshippable on day one, and a rule nobody can ship gets deleted instead of
#: obeyed.
PREVIEW_DATE_EXCEPTIONS: Dict[Tuple[str, str], str] = {
    ("gemini-3.1-flash-image-preview", "comfy_image"):
        "runtime sends this id to a Vertex proxy, not the catalog endpoint that "
        "can be fetched, so catalog presence cannot prove the route works "
        "(build spec 2026-08-09 section 0A -- escalated, operator's call)",
}

#: Directed preview -> stable pairs, written down rather than derived.
#: Strip-and-compare derivation would invent pairs that do not exist and miss
#: ones that do; a table is checked against the inventory instead.
PREVIEW_TO_STABLE: Dict[str, str] = {
    "gemini-3.1-flash-image-preview": "gemini-3.1-flash-image",
}


def has_preview_token(provider_id: str) -> bool:
    """True when `preview` appears as a whole `-`-delimited segment.

    A TOKEN-BOUNDARY test, never a substring match. All three placements exist in
    this pack's own lanes and each must be caught:

      * suffix -- `lyria-3-clip-preview`
      * infix  -- `gemini-2.5-flash-preview-tts`
      * infix, reversed -- `gemini-3.1-flash-tts-preview`

    And a substring match would fire on an id merely CONTAINING the letters, such
    as a hypothetical `previewer-1`, which is a different model.
    """
    if not isinstance(provider_id, str):
        return False
    return _PREVIEW_TOKEN in provider_id.split("-")


def is_pointer(slug: str) -> bool:
    """True for a routing pointer that resolves upstream at request time.

    Pointers need no date -- there is no version claim to go stale. Covers both
    spellings in the tree: OpenRouter's `~author/family-latest` and Google's
    bare `...-latest`.
    """
    return isinstance(slug, str) and (slug.startswith("~") or slug.endswith("-latest"))


def unverified_identities() -> Dict[Tuple[str, str], str]:
    """`(provider_id, lane) -> lane`, for everything nobody has settled.

    Not a failure; a worklist. The VERIFIER prints it -- a pytest print lands
    under capture and is invisible at the `-q` baseline the project actually
    runs, which silently gutted the original mitigation.
    """
    return {key: rec.authority_lane
            for key, rec in SLUG_PROVENANCE.items()
            if rec.verification_kind == UNVERIFIED}


__all__ = [
    "CATALOG_LISTED", "SIGNAL_LISTED", "UNVERIFIED", "VERIFICATION_KINDS",
    "DATED_KINDS", "ProvenanceRecord", "LANE_AUTHORITY", "SLUG_PROVENANCE",
    "FREE_MARKERS", "PREVIEW_DATE_EXCEPTIONS", "PREVIEW_TO_STABLE",
    "has_preview_token", "is_pointer", "unverified_identities",
]
