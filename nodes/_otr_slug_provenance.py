"""Provenance for every CONCRETE provider slug this pack ships.

WHY THIS EXISTS. `tencent/hy3:free` sat in the OpenRouter dropdown until its
promo ended and the slug stopped resolving. Chunk A (2026-08-07) fixed that ONE
lane: `_otr_model_catalog.OPENROUTER_VERIFIED_ON_BY_ID` dates every concrete
OpenRouter id and `tests/test_openrouter_slug_curation.py` fails when an undated
one is added. The same "concrete version pin, no date, nothing able to notice it
went stale" shape was left live in five other lanes (GO_FORWARD item 6). This
module closes that, and `tests/test_slug_provenance.py` enforces it.

THE HONEST PART, and the reason entries are not all dates. A verified-on date is
a CLAIM that someone checked the slug against the authority that decides whether
it resolves. Stamping today's date on thirty-odd slugs nobody has checked would
manufacture precisely the defect this pack keeps finding -- a field that reads as
evidence and is not (BUG-12.86) -- and would be worse than no date, because it
would look settled. So an entry is EITHER:

  * an ISO date -- someone verified this slug against its authority on that day; or
  * UNVERIFIED -- nobody has, and the `authority` field says who could.

Both are legal. What is NOT legal is a shipped concrete slug with no entry at
all, because that is the state where staleness is invisible. The test counts the
UNVERIFIED ones out loud rather than letting them hide.

EACH LANE HAS A DIFFERENT AUTHORITY. This is why one global "check the catalog"
sweep cannot work and why the field is per-lane:

  * comfy        -- Comfy Cloud's partner-node catalog decides whether Comfy
                    SERVES a slug. OpenRouter presence is a SIGNAL, not proof.
  * google_api   -- Google's own generativelanguage model list.
  * elevenlabs   -- the ElevenLabs models endpoint.
  * comfy_image  -- Comfy Cloud partner catalog (display names, not ids).
  * google_image -- Google's image-capable model list.

Pure data + stdlib. No imports from the engine namespaces, so this module stays
cold-import clean and the test can read it without loading torch.
"""
from __future__ import annotations

#: Marker for "no one has verified this against its authority". Deliberately a
#: loud string rather than None -- a None reads as an accident, this reads as a
#: recorded gap.
UNVERIFIED = "UNVERIFIED"

#: Who can settle each lane. Naming the authority is the difference between a
#: TODO and an actionable one.
LANE_AUTHORITY: dict[str, str] = {
    "comfy": "Comfy Cloud partner-node catalog (NOT OpenRouter -- presence "
             "there is a signal, not proof that Comfy serves it)",
    "google_api": "Google generativelanguage models.list",
    "elevenlabs": "ElevenLabs GET /v1/models",
    "comfy_image": "Comfy Cloud partner-node catalog (display names)",
    "google_image": "Google generativelanguage models.list, image-capable",
}

#: slug -> (lane, ISO date | UNVERIFIED)
#:
#: The comfy dates are REAL and carried forward, not invented: a 2026-08-07
#: session checked all six against the live OpenRouter catalog and found every
#: one present. That is recorded as a SIGNAL date with the caveat above -- it is
#: evidence the id still exists upstream, not proof Comfy serves it.
#: Everything else is UNVERIFIED because nobody has looked.
SLUG_PROVENANCE: dict[str, tuple[str, str]] = {
    # --- comfy: _otr_comfy_backend.COMFY_LLM_MODELS ---------------------
    "google/gemini-3.5-flash":      ("comfy", "2026-08-07"),
    "deepseek/deepseek-v3.2":       ("comfy", "2026-08-07"),
    "mistralai/mistral-large-2512": ("comfy", "2026-08-07"),
    "x-ai/grok-4.20":               ("comfy", "2026-08-07"),
    "openai/gpt-5.5":               ("comfy", "2026-08-07"),
    "anthropic/claude-opus-4.7":    ("comfy", "2026-08-07"),

    # --- google_api: the three text-model tuples ------------------------
    # `*-latest` ids are POINTERS and need no date -- they resolve upstream,
    # which is the whole reason the curation policy prefers them. They are
    # excluded from the shipped-concrete set by the test, not listed here.
    "gemini-3.5-flash":       ("google_api", UNVERIFIED),
    "gemini-3.1-flash-lite":  ("google_api", UNVERIFIED),
    "gemini-2.5-flash":       ("google_api", UNVERIFIED),
    "gemini-2.5-pro":         ("google_api", UNVERIFIED),
    "gemini-2.0-flash":       ("google_api", UNVERIFIED),
    "gemini-2.0-flash-lite":  ("google_api", UNVERIFIED),

    # --- elevenlabs: eng_cloud_elevenlabs._SUPPORTED_MODELS -------------
    "eleven_multilingual_v2": ("elevenlabs", UNVERIFIED),
    "eleven_v3":              ("elevenlabs", UNVERIFIED),

    # --- comfy_image: eng_cloud_image display-name lists ----------------
    "Nano Banana 2 (Gemini 3.1 Flash Image)": ("comfy_image", UNVERIFIED),
    "Nano Banana 2 Lite":                     ("comfy_image", UNVERIFIED),
    "seedream 5.0 lite":                      ("comfy_image", UNVERIFIED),
    "seedream-4-5-251128":                    ("comfy_image", UNVERIFIED),
    "seedream-4-0-250828":                    ("comfy_image", UNVERIFIED),
    "Krea 2 Medium Turbo":                    ("comfy_image", UNVERIFIED),
    "Krea 2 Medium":                          ("comfy_image", UNVERIFIED),
    "Krea 2 Large":                           ("comfy_image", UNVERIFIED),
    "photon-flash-1":                         ("comfy_image", UNVERIFIED),
    "photon-1":                               ("comfy_image", UNVERIFIED),

    # --- google_image: eng_google_image.SUPPORTED_MODELS ----------------
    "gemini-3.1-flash-image":      ("google_image", UNVERIFIED),
    "gemini-3.1-flash-lite-image": ("google_image", UNVERIFIED),
    "gemini-3-pro-image":          ("google_image", UNVERIFIED),
}

#: A slug carrying one of these advertises a PRICE inside an IDENTIFIER.
#: Promises expire; identifiers do not. Same rule as the OpenRouter guard.
FREE_MARKERS = (":free", "-free")


def is_pointer(slug: str) -> bool:
    """True for a routing pointer that resolves upstream at request time.

    Pointers need no verified-on date -- there is no version claim to go stale.
    Covers both spellings in the tree: OpenRouter's `~author/family-latest` and
    Google's bare `...-latest`.
    """
    return slug.startswith("~") or slug.endswith("-latest")


def unverified_slugs() -> dict[str, str]:
    """slug -> lane, for every entry nobody has checked. Not a failure; a
    worklist. Kept as a function so a caller can print it in one line."""
    return {slug: lane for slug, (lane, when) in SLUG_PROVENANCE.items()
            if when == UNVERIFIED}


__all__ = ["UNVERIFIED", "LANE_AUTHORITY", "SLUG_PROVENANCE", "FREE_MARKERS",
           "is_pointer", "unverified_slugs"]
