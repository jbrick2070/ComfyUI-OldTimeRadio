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
    "ltx23_16gb_audio_in": "ltx_audio_in",
    "ltx23_16gb_video": "ltx_video",
    # --- the low/high naming convention starts here (lane 1, 2026-08-11) ---
    # `<model><version>_<low|high>_<capability>`. The `<vramtier>gb` token above
    # is RETIRED by operator ruling 2026-08-09: it encoded "the card this lane
    # was built for", which drifted badly from measured usage (`wan_8gb` really
    # consumes 12.5-13.2 GiB and cannot run on an 8 GB card at production
    # canvas). `low` / `high` is deliberately COARSE so a user self-selects by
    # their own hardware, and it survives measurement drift.
    #
    # Renamed lanes MOVE their old public id into _LEGACY_ENGINE_ALIASES; they
    # never keep a second row here, because two public ids on one internal id
    # collapses _INTERNAL_TO_PUBLIC and trips the module-scope bijection assert
    # below at IMPORT time -- which, since the director imports this module
    # unguarded, empties most of the ComfyUI node menu rather than failing one
    # lane cleanly.
    "wan22_high_i2v": "wan_i2v",
    # Lane 2, 2026-08-11. The id STATES what the lane is, per the operator's
    # 2026-08-10 refinements: audio-conditioned lanes say `audio_in` (HuMo is
    # audio-driven and now says so), and the aspect is in the id rather than
    # only in the label suffix -- the bare `humo14_high_face` hid that its
    # sibling renders 480x832. `high` comes from a measurement receipt: 13.06
    # GiB warm at 832x480x97 under the humo_diet boot.
    "humo14_high_audio_in_wide": "humo_14B_169",
    # Lane 3, 2026-08-11. The LONG-BEAT lane (the 1.7B renders to 177 frames,
    # 7.08 s, where the 14B stops at 97) and the auto-downgrade target. Its
    # landscape twin closes with it: same checkpoint, same VRAM class, the
    # aspect is the whole difference, so the aspect is in the id.
    "humo17_high_audio_in_portrait": "humo_1.7B",
    "humo17_high_audio_in_wide": "humo_1.7B_169",
    # Lane 4, 2026-08-11: the last HuMo tier. The 2026-06-09 keystone, and
    # the only one of the four whose id was previously just "humo".
    "humo14_high_audio_in_portrait": "humo",
    # Lane 5, 2026-08-11 -- the first MOVE rather than an add.
    # `wan_8gb` left this table for _LEGACY_ENGINE_ALIASES in the
    # same edit: two public ids on one internal id collapses
    # _INTERNAL_TO_PUBLIC and trips the bijection assert at IMPORT
    # time, which empties the ComfyUI menu rather than failing one
    # lane. MOVE, never ADD.
    "wan22_high_video": "wan_ti2v",
    # Lane 6, 2026-08-11. `fastwan_8gb` was an IDENTITY row (its public id
    # WAS its internal id), so it needs no alias row on the way out --
    # a bare internal id already passes through resolve_engine_id step 3.
    # The label sells THROUGHPUT, never quality: same motion, same canvas,
    # same VRAM as wan22_high_video, about 2.7x sooner.
    "wan22_high_fast": "fastwan_8gb",
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
    # RESOLVED BY THE OPERATOR 2026-08-11: the naming was decided all along and
    # `wan21` was a single mistyped version number in the spec that everything
    # downstream inherited. The lane loads a Wan 2.2 weight
    # (wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors, recipe
    # wan22_14b_i2v_single_pass_v1), so `wan22_high_i2v` is correct and stands;
    # the spec and the transplant plan now say so too.
    #
    # This row stays anyway, and NOT because the spec needs it -- it does not.
    # The string briefly existed in a reviewed document, so it survives in this
    # session's kibitz runs and handoff log, and a person pasting from any of
    # those should land on the lane rather than on a not-registered error. It is
    # the cheapest possible way to make a retired typo harmless.
    "wan21_high_i2v": "wan_i2v",
    # MOVED out of _PUBLIC_ENGINES in lane 5 (2026-08-11). Every
    # saved graph, profile and variant carrying `wan_8gb` still
    # resolves through here; it just no longer renders as a menu
    # option. The token was retired because it encoded the card
    # the lane was built for, and this lane really consumes
    # 12.5-13.2 GiB -- it cannot run on an 8 GB card at
    # production canvas.
    "wan_8gb": "wan_ti2v",
}

#: Internal engine id -> its public menu id (inverse of _PUBLIC_ENGINES; the label
#: builder maps an internal id to the public token shown in the dropdown).
_INTERNAL_TO_PUBLIC = {v: k for k, v in _PUBLIC_ENGINES.items()}

#: Friendly prose labels -- TOOLTIP / DOCS ONLY, never the combo/saved value.
_PUBLIC_LABEL = {
    "ltx_8gb": "LTX 0.9.8 2B - 8GB",
    # "3-step" rather than "fast" or "better": this tier renders the SAME motion at
    # the SAME canvas for the SAME VRAM as wan_8gb, about 2.7x sooner. Naming it a
    # quality or longer-clip upgrade would mis-sell it.
    "wan22_high_fast": (
        "FastWan 2.2 TI2V 5B 3-step - high VRAM (the SAME motion at the "
        "SAME canvas for the SAME VRAM as wan22_high_video, ~2.7x sooner)"),
    "ltx23_16gb_audio_in": "LTX 2.3 - 16GB Audio In",
    "ltx23_16gb_video": "LTX 2.3 - 16GB Video",
    # "high" is the measured bucket, not a quality claim: 13.93 GiB warm at
    # 832x480x33 against a 14.5 GiB gate. The rung is named because only f33
    # has warm evidence -- the f177 the contract allows is model-legal and not
    # machine-qualified.
    "wan22_high_i2v": "Wan 2.2 I2V 14B fp8 - high VRAM (13.9 GiB warm at f33)",
    "wan22_high_video": (
        "Wan 2.2 TI2V 5B Q5 - high VRAM (12.1 GiB warm at 832x480x193; "
        "the default WAN lane)"),
    "humo14_high_audio_in_wide": (
        "HuMo 14B fp8 16:9 - audio-driven face, high VRAM "
        "(13.06 GiB warm at 832x480x97 on the humo_diet boot)"),
    "humo17_high_audio_in_portrait": (
        "HuMo 1.7B portrait - audio-driven face, high VRAM, LONG beats "
        "(12.84 GiB warm at 480x832x129 on the humo_diet boot; 177 frames)"),
    "humo17_high_audio_in_wide": (
        "HuMo 1.7B 16:9 - audio-driven face, high VRAM, LONG beats "
        "(same checkpoint as the portrait tier; UNMEASURED at this aspect)"),
    "humo14_high_audio_in_portrait": (
        "HuMo 14B fp8 portrait - audio-driven face, high VRAM "
        "(13.22 GiB warm at 480x832x97 on the humo_diet boot)"),
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


#: The engine ids RETIRED by the 2026-08-06 SFX-bed rip. A user-saved workflow
#: or an external API client may still name one; the contract is a NAMED
#: refusal -- a stale selection must never silently resolve to another engine.
#: These ids are DATA consulted by :func:`check_retired_engine`: never a row in
#: the registry's ``CAPABILITIES``, never importable as an adapter. IMMUTABLE:
#: append here only when another engine is retired, never remove.
RETIRED_ENGINE_IDS = frozenset({
    "cloud_vidu_q2_pro_fast_720p_sfx",
    "google_vid_sfx_omni",
    "google_vid_sfx_veo_fast",
    "google_vid_sfx_veo_lite",
    "google_vid_sfx_veo_pro",
})


class RetiredEngineError(ValueError):
    """A RETIRED engine id was selected; FAIL CLOSED with a NAMED error.

    ONE policy, one spelling: every selection boundary raises THIS type with
    THIS ``reason_code`` and THIS message shape, so five boundaries can never
    drift into five incompatible "named" failures. Boundary wrappers may
    translate the exception but never restate the policy.
    """

    reason_code = "retired_engine"      # lower-case; ONE spelling, test-pinned

    def __init__(self, engine_id: str):
        self.engine_id = str(engine_id)
        super().__init__(
            "video engine '%s' is retired and is no longer selectable"
            % self.engine_id)


def check_retired_engine(engine_id) -> None:
    """Raise :class:`RetiredEngineError` iff ``engine_id`` is retired.

    Call AFTER public/legacy-name resolution (``resolve_engine_id``) at every
    selection boundary, so an alias cannot slip past. Empty / unknown ids pass
    through untouched -- this guard owns ONLY the retired set."""
    if str(engine_id or "") in RETIRED_ENGINE_IDS:
        raise RetiredEngineError(str(engine_id))


__all__ = [
    "_PUBLIC_ENGINES", "_LEGACY_ENGINE_ALIASES", "_INTERNAL_TO_PUBLIC",
    "_PUBLIC_LABEL", "resolve_engine_id",
    "RETIRED_ENGINE_IDS", "RetiredEngineError", "check_retired_engine",
]
