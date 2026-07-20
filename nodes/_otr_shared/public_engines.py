"""Public video-engine name resolver -- ONE dep-free source of truth (video tiers).

The video-tiers build (2026-07-20) gives four engines a stable, user-facing PUBLIC
id shown in the OTR_VideoDirector menu, decoupled from the ~420 internal id refs that
stay untouched (additive only -- no rename):

    public id             -> internal engine id
    ltx_8gb               -> ltx_8gb            (identity; new 8GB LTX 0.9.8 engine)
    wan_8gb               -> wan_ti2v
    ltx23_16gb_audio_in   -> ltx_audio_in
    ltx23_16gb_video      -> ltx_video

This module is the SINGLE place that maps a menu/saved/profile string back to the
concrete internal engine id: it strips the display suffix (`wan_8gb (16:9)`), maps a
public id to its internal id, THEN maps a renamed engine's legacy id to its current
id (the `_LEGACY_ENGINE_ALIASES` MOVED here from otr_video_director so both the
director and every other boundary read ONE table).

The friendly prose labels (`Wan 2.2 TI2V 5B - 8GB`, ...) live in `_PUBLIC_LABEL` for
the static widget tooltip + docs ONLY -- they are NEVER the combo value / saved value
(the menu value = the short public id + the existing aspect suffix).

stdlib-only, cold-import clean (V-12): importing this module pulls in NOTHING (no
torch / no registry / no ComfyUI). UTF-8, no BOM, ASCII-only.
"""
from __future__ import annotations

#: Public menu id -> internal engine id (the four video-tier rows). ltx_8gb maps to
#: itself (its internal id IS its public id).
_PUBLIC_ENGINES = {
    "ltx_8gb": "ltx_8gb",
    "wan_8gb": "wan_ti2v",
    "ltx23_16gb_audio_in": "ltx_audio_in",
    "ltx23_16gb_video": "ltx_video",
}

#: Legacy engine-id aliases (renamed engines) -- MOVED here from otr_video_director
#: so the resolver, the director, the applier, the render driver and the capability
#: profiles all read ONE table. A saved graph / old ledger carrying the pre-rename
#: name resolves to the current engine so the pick keeps working.
_LEGACY_ENGINE_ALIASES = {
    "flat_still": "still_flat",
    "flux_still": "still_pan",
    "still_kenburns": "still_motion",
    "visualizer": "viz_green",
}

#: Internal engine id -> its public menu id (inverse of _PUBLIC_ENGINES; the label
#: builder maps an internal id to the public token shown in the dropdown).
_INTERNAL_TO_PUBLIC = {v: k for k, v in _PUBLIC_ENGINES.items()}

#: Friendly prose labels -- TOOLTIP / DOCS ONLY, never the combo/saved value.
_PUBLIC_LABEL = {
    "ltx_8gb": "LTX 0.9.8 2B - 8GB",
    "wan_8gb": "Wan 2.2 TI2V 5B - 8GB",
    "ltx23_16gb_audio_in": "LTX 2.3 - 16GB Audio In",
    "ltx23_16gb_video": "LTX 2.3 - 16GB Video",
}

# Bijection guard: unique internals (no two public ids share one internal engine),
# so _INTERNAL_TO_PUBLIC never collapses a row -- exact_menu_option_for stays 1:1.
assert len(_PUBLIC_ENGINES) == len(_INTERNAL_TO_PUBLIC), (
    "public_engines: _PUBLIC_ENGINES is not a bijection (a duplicate internal id "
    "collapses _INTERNAL_TO_PUBLIC): " + repr(_PUBLIC_ENGINES))


def resolve_engine_id(value) -> str:
    """Resolve a menu / saved / profile string to its concrete internal engine id.

    Order (each step idempotent for a value the step does not own):
      1. strip the display suffix -- the token BEFORE the first ' (' (so
         ``'wan_8gb (16:9)'`` -> ``'wan_8gb'``; a bare id / the ADD_CUSTOM sentinel
         has no ' (' and passes through);
      2. PUBLIC -> internal (``'wan_8gb'`` -> ``'wan_ti2v'``);
      3. LEGACY -> current (``'visualizer'`` -> ``'viz_green'``).

    A bare internal id, an unknown id, and the ``'+ Add Custom Model'`` sentinel all
    pass through unchanged. Pure; never raises."""
    bare = str(value or "").split(" (", 1)[0]
    resolved = _PUBLIC_ENGINES.get(bare, bare)          # public -> internal
    return _LEGACY_ENGINE_ALIASES.get(resolved, resolved)  # then legacy -> current


__all__ = [
    "_PUBLIC_ENGINES", "_LEGACY_ENGINE_ALIASES", "_INTERNAL_TO_PUBLIC",
    "_PUBLIC_LABEL", "resolve_engine_id",
]
