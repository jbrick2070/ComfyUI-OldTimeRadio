"""The ONE inventory of every provider model this pack OFFERS (non-video).

WHY THIS EXISTS. The provenance guard used to build its own list inline, and the
verifier script would have built a third. Three lists that must agree, with
nothing forcing them to, is the defect the guard was written to catch -- so there
is one collector, here, and both the test and the verifier import it.

WHAT A RECORD IS, AND WHY IT IS NOT JUST A STRING. The same provider id can be
offered by more than one adapter through more than one billing surface. A plain
``{slug: where}`` dict silently collapses those, and it already would today:
``gemini-3.1-flash-lite-image`` is offered BOTH directly by ``eng_google_image``
and as the Comfy selector ``Nano Banana 2 Lite``. Those are different auth and
billing paths for the same model, and a guard that cannot tell them apart cannot
reason about either.

SCOPE. Audio, image and the static LLM lanes. **No video** -- the video lanes are
out of scope by operator ruling (2026-08-09) and nothing here may import them.
``tests/test_slug_inventory_scope.py`` proves that mechanically in a fresh
process rather than trusting this comment.

WHAT IT COLLECTS, AND WHAT IT DELIBERATELY DOES NOT. Checked-in constants and
declared defaults ONLY. It never calls a runtime selector: ``resolve_model_id()``
reads environment overrides and the Google slot builder folds disk-cached ids, so
either would make the inventory differ between machines and turn a shared guard
into a local opinion.
"""
from __future__ import annotations

from typing import Iterable, List, NamedTuple, Optional

#: A Google catalog id, when this record can be checked against Google's
#: ``models.list``. ``None`` means no authority endpoint is reachable for it.
CATALOG_GOOGLE = "google"


class InventoryRecord(NamedTuple):
    """One (source, surface, offered value) the pack ships.

    ``source_key``      stable name of the source -- a registered engine name,
                        or a static lane's dotted constant path.
    ``source_kind``     ``"engine"`` or ``"static_lane"``. Not every source is an
                        engine: the Comfy LLM list and the Google text tuples are
                        module constants with no adapter behind them.
    ``selection_surface`` distinguishes selector groups within one source, and is
                        the key the per-surface preview/stable rule uses.
    ``offered_value``   exactly what the operator picks (may be a display name).
    ``provider_id``     what actually goes on the wire, never carrying a
                        ``models/`` prefix.
    ``authority_lane``  who decides whether this slug resolves.
    ``catalog_target``  which live catalog can check it, or ``None``. Kept
                        SEPARATE from ``authority_lane`` because a Comfy surface
                        can still expose a Google provider id.
    """

    source_key: str
    source_kind: str
    selection_surface: str
    offered_value: str
    provider_id: str
    authority_lane: str
    catalog_target: Optional[str]


#: Engines that legitimately ship NO concrete model collection. Keyed on the
#: reason, not on a bare name, so a future engine of the same shape is covered
#: without editing an allowlist.
#:
#: ``sonilo`` routes through a Comfy PARTNER ROW (``_PARTNER_ROW =
#: "cloud_sonilo_music"``), which is a product identifier rather than a model
#: slug -- there is no version pin in it to go stale.
EXEMPT_ENGINES = {
    "sonilo": "ships a partner row, not a model slug -- no version pin to date",
}


def _norm(provider_id: str) -> str:
    """Strip exactly one leading ``models/`` so ids compare equal to catalog
    names. Google returns ``models/{id}``; everything else ships bare."""
    prefix = "models/"
    return provider_id[len(prefix):] if provider_id.startswith(prefix) else provider_id


def collect_inventory() -> List[InventoryRecord]:
    """Every offered model, as records. Imports are LAZY and local so merely
    importing this module costs nothing and cannot create a cycle."""
    from .google_image_model_ids import resolve_selector_to_model_id

    out: List[InventoryRecord] = []

    def add(source_key: str, source_kind: str, surface: str,
            values: Iterable[str], lane: str, catalog: Optional[str],
            resolve: bool = False) -> None:
        for offered in values or ():
            if not isinstance(offered, str) or not offered:
                continue
            provider = resolve_selector_to_model_id(offered) if resolve else offered
            out.append(InventoryRecord(
                source_key=source_key,
                source_kind=source_kind,
                selection_surface=surface,
                offered_value=offered,
                provider_id=_norm(provider),
                authority_lane=lane,
                catalog_target=catalog,
            ))

    # --- static lanes: module constants, no adapter behind them -------------
    from .. import _otr_comfy_backend as comfy
    add("_otr_comfy_backend.COMFY_LLM_MODELS", "static_lane", "comfy_llm",
        comfy.COMFY_LLM_MODELS, "comfy", None)

    from .._otr_google_api import models as gmodels
    for const in ("GOOGLE_API_LEGACY_TEXT_MODELS",
                  "GOOGLE_API_STABLE_TEXT_MODELS",
                  "GOOGLE_API_STATIC_TEXT_MODELS"):
        add("_otr_google_api.models.%s" % const, "static_lane", "google_text",
            getattr(gmodels, const, ()), "google_api", CATALOG_GOOGLE)

    # --- engines ------------------------------------------------------------
    from .._otr_audio_engines import eng_cloud_elevenlabs as el
    add("elevenlabs", "engine", "elevenlabs_tts",
        el._SUPPORTED_MODELS, "elevenlabs", None)

    from .._otr_audio_engines import eng_google_tts as gtts
    add("google_tts", "engine", "google_tts",
        gtts._SUPPORTED_MODELS, "google_tts", CATALOG_GOOGLE)

    from .._otr_audio_engines import eng_google_lyria as lyria
    add("google_lyria", "engine", "google_lyria",
        lyria.SUPPORTED_MODELS, "google_lyria", CATALOG_GOOGLE)

    from .._otr_image_engines import eng_google_image as gimg
    add("google_image", "engine", "google_image",
        gimg.SUPPORTED_MODELS, "google_image", CATALOG_GOOGLE)

    # Comfy partner IMAGE selectors. The nano rows are DISPLAY NAMES that
    # resolve to Google ids, so they are checkable against Google's catalog even
    # though Comfy is the authority for whether Comfy serves them. The other
    # three groups are ByteDance / Krea / Luma and have no reachable catalog.
    from .._otr_image_engines import eng_cloud_image as ci
    add("cloud_nano_banana_2", "engine", "comfy_nano",
        getattr(ci, "_NANO_MODELS", ()), "comfy_image", CATALOG_GOOGLE,
        resolve=True)
    for const, surface in (("_SEEDREAM_MODELS", "comfy_seedream"),
                           ("_KREA_MODELS", "comfy_krea"),
                           ("_LUMA_PHOTON_MODELS", "comfy_photon")):
        add("cloud_image.%s" % const, "engine", surface,
            getattr(ci, const, ()), "comfy_image", None, resolve=True)

    return out


def google_catalog_records(records: Iterable[InventoryRecord]
                           ) -> List[InventoryRecord]:
    """Only the records a Google catalog fetch may be asked about.

    Without this projection the verifier would send ElevenLabs models and Comfy
    display selectors into Google's catalog and report them 'missing' -- a
    confident wrong answer, which is the failure mode this whole chunk exists to
    prevent."""
    return [r for r in records if r.catalog_target == CATALOG_GOOGLE]


def provider_identities(records: Iterable[InventoryRecord]) -> set:
    """The ``(provider_id, authority_lane)`` pairs provenance is keyed by."""
    return {(r.provider_id, r.authority_lane) for r in records}


__all__ = [
    "CATALOG_GOOGLE", "InventoryRecord", "EXEMPT_ENGINES",
    "collect_inventory", "google_catalog_records", "provider_identities",
]
