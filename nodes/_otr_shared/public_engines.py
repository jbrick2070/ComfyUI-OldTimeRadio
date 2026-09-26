"""Public video-engine name resolver -- ONE dep-free source of truth (video tiers).

Public ids follow the ``<model><version>_<low|high>_<capability>`` convention
(lane 1, 2026-08-11), decoupled from the internal ids that stay untouched.

This module is the SINGLE place that maps a menu/saved/profile string back to the
concrete internal engine id: it strips the display suffix (`ltx098_low_video (16:9)`), maps a
public id to its internal id, THEN maps a renamed engine's legacy id to its current
id (the `_LEGACY_ENGINE_ALIASES` MOVED here from otr_video_director so both the
director and every other boundary read ONE table).

The friendly prose labels (`LTX 0.9.8 2B - low VRAM`, ...) live in `_PUBLIC_LABEL` for
the static widget tooltip + docs ONLY -- they are NEVER the combo value / saved value
(the menu value = the short public id + the existing aspect suffix).

stdlib-only, cold-import clean (V-12): importing this module pulls in NOTHING (no
torch / no registry / no ComfyUI). UTF-8, no BOM, ASCII-only.
"""
from __future__ import annotations

#: Public menu id -> internal engine id.
_PUBLIC_ENGINES = {
    # --- the low/high naming convention starts here (lane 1, 2026-08-11) ---
    # `<model><version>_<low|high>_<capability>`. The `<vramtier>gb` token above
    # is RETIRED by operator ruling 2026-08-09: it encoded "the card this lane
    # was built for", which drifted badly from measured usage. `low` / `high`
    # is deliberately COARSE so a user self-selects by
    # their own hardware, and it survives measurement drift.
    #
    # Renamed lanes MOVE their old public id into _LEGACY_ENGINE_ALIASES; they
    # never keep a second row here, because two public ids on one internal id
    # collapses _INTERNAL_TO_PUBLIC and trips the module-scope bijection assert
    # below at IMPORT time -- which, since the director imports this module
    # unguarded, empties most of the ComfyUI node menu rather than failing one
    # lane cleanly.
    #
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
    # Lane 8, 2026-08-11. `ltx_8gb` was an IDENTITY row, so it needs NO alias
    # on the way out -- a bare internal id
    # already passes through resolve_engine_id step 3, and adding one would
    # imply an internal rename that never happened. The internal id KEEPS the
    # `8gb` token; only the public surface loses it.
    #
    # `low` is MEASURED here, not inherited from the retired token: 9,106 MB
    # absolute / 6,835 MB net, cold, at 512x288x161 -- the cheapest lane in the
    # roster. Until this lane's own
    # smoke ran, the marker was provisional in the evidence manifest's own
    # words ("NO measurement of any kind on this box").
    "ltx098_low_video": "ltx_8gb",
    # Lane 19, 2026-08-12 -- the first NEW ENGINE in the campaign rather than a
    # rename, so this is an ADD with no alias to move.
    #
    # `low` is MEASURED on the local 5080 under this lane's own boot contract:
    # the controlling legal-floor receipt is 124 model / 129 canvas frames at
    # 864x480 and 6,315 MB cold absolute. That measured allocation does not
    # qualify a physical 8 GB card; 8 GB remains an explicit lab question.
    #
    # `video` and not `audio_in`: H3 natively produces audio, and this lane
    # deliberately does not decode it -- there is no audio VAE in its graph. The
    # audio-conditioned route is lane 20's `h3_low_audio_in`, a SEPARATE public
    # id on a SEPARATE internal id.
    "h3_low_video": "minimax_h3_video",
    # Lane 20, 2026-08-12. The SECOND public id on the H3 stack, and it maps to
    # a SEPARATE internal engine -- which is the whole reason lane 19 registered
    # only one adapter. Two public ids on ONE internal id collapses
    # _INTERNAL_TO_PUBLIC and trips the bijection assert below at IMPORT time.
    #
    # `audio_in` states the capability: this lane conditions on the beat's own
    # audio through MiniMaxH3ReferenceToVideo. Its
    # `low` is the same measured bucket as its sibling: the controlling REF2VA
    # receipt is 864x480, 124 model / 129 canvas frames, 6,678 MB cold absolute
    # on the 5080. This is not a physical-8-GB support receipt.
    "h3_low_audio_in": "minimax_h3_audio_in",
    # LTX 2.5, 2026-08-19. The silent lane, on the 16 GB mix4x8 DiT.
    #
    # `video` and not `audio_in`, matching `h3_low_video`: LTX 2.5 natively
    # produces audio and this lane deliberately does not decode it. Three
    # public ids on three internal ids, never one id with a switch, because two
    # public ids on one internal id collapses _INTERNAL_TO_PUBLIC and trips the
    # bijection assert below AT IMPORT (lesson L5).
    "ltx25_high_video": "ltx25_video",
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
    # MEASURED, and deliberately NOT an 8 GB claim. The retired `8gb` token
    # encoded the card the lane was built for; this label states what the lane
    # was measured to COST (6.8 GiB net at 512x288x161, cold) and lets the user
    # decide what that fits on. Saying "runs on an 8 GB card" would repeat the
    # exact mistake lane 5 retired the token for -- net cost is not the whole
    # story on a card whose desktop already eats some of it.
    "ltx098_low_video": (
        "LTX 0.9.8 2B - low VRAM (6.8 GiB net at 512x288x161; "
        "the cheapest local video lane, ~22 s a beat)"),
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
    # "low" is the measured bucket and the label names the leg it came from, per
    # the lane-1 convention. It also states the two things a user has to know
    # before picking this lane: it needs its own boot, and it is SLOW -- a 5 s
    # beat is minutes, not seconds, because a 21 GB DiT streams through a 16 GB
    # card. Neither is a quality claim in either direction.
    "h3_low_video": (
        "MiniMax H3 33B silent video - low VRAM (7.3 GiB at 864x480; needs the "
        "sage-free h3 boot, and it is the slowest local lane by far)"),
    # Same two warnings as its sibling, plus what it actually adds: this is the
    # only LOCAL lane that conditions on a reference PORTRAIT and audio together.
    "h3_low_audio_in": (
        "MiniMax H3 33B audio-in - low VRAM (6.9-7.2 GiB at 864x480; reference "
        "portrait + the beat's own audio; needs the sage-free h3 boot, and it "
        "is as slow as its silent sibling)"),
    "ltx25_high_video": (
        "LTX 2.5 Distilled mix4x8 HQ two-stage silent video - 16 GB "
        "(832x480 first stage, 1664x960 refined decode, one 3.88 s rung)"),
    # GHOST SIGNAL (2026-08-22). A LABEL ONLY -- there is deliberately no
    # `_PUBLIC_ENGINES` self-alias, because the resolver already passes a bare
    # internal id through unchanged (the existing identity-engine precedent) and
    # a self-alias would put a duplicate in the bijection for no gain.
    #
    # NO `low` OR `high` TOKEN, and that is the whole naming decision. Those
    # tokens carry MEASURED cost semantics in this repo -- every one of the
    # labels above names a real measured bucket -- and no measurement campaign
    # was run for this lane. "very-low-VRAM-targeted" is a design target, so the
    # label says targeted and nothing stronger.
    # ONE GHOST LANE (operator, 2026-08-23: "delete any animatediff that are
    # not haunted"). The five siblings are RETIRED and tombstoned below. The
    # adapter IS this lane's identity, so the label says so plainly. "Apache-2.0"
    # is in the label deliberately: the golden lane's module had no licence at
    # all, and for anyone deciding what to build on that is the most load-bearing
    # fact about this one.
    "animatediff15_v3_haunted_video": (
        "AnimateDiff v3 haunted -- Ghost Signal (official v3 module + the "
        "removable domain adapter, Apache-2.0; degraded transmission look)"),
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
         ``'ltx098_low_video (16:9)'`` -> ``'ltx098_low_video'``; a bare id /
         the ADD_CUSTOM sentinel has no ' (' and passes through);
      2. PUBLIC -> internal (``'ltx098_low_video'`` -> ``'ltx_8gb'``);
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
    # THE NON-HAUNTED GHOST LANES, RETIRED 2026-08-23 (operator: "delete any
    # animatediff that are not haunted"). All five were PUBLIC, menu-selectable
    # ids, and animatediff15_video carried the lane's published proof
    # (signal_lost_the_constables_knock_20260822_050116, 8/8 beats) -- so a saved
    # graph naming it is a graph that once worked, which is exactly why these are
    # NAMED tombstones rather than the generic unregistered-engine refusal.
    #
    # THE CLASSES ARE NOT ALL GONE: GhostSignalV3HauntedEngine inherits
    # GhostSignalV3Engine inherits GhostSignalEngine, so those two SURVIVE as the
    # winner's own machinery -- unregistered, not deleted. Only the true leaves
    # (v2, and the h3/h5 cadence pair) had their code removed.
    "animatediff15_video",
    "animatediff15_v2_video",
    "animatediff15_v3_video",
    "animatediff15_h3_video",
    "animatediff15_h5_video",
    "cloud_vidu_q2_pro_fast_720p_sfx",
    "google_vid_sfx_omni",
    "google_vid_sfx_veo_fast",
    "google_vid_sfx_veo_lite",
    "google_vid_sfx_veo_pro",
    # The dormant 3D / dark family, RETIRED 2026-08-23 (lean-mean order 4).
    # All five were registered/selectable in alpha builds before their
    # unregister commits (character_3d talkers 2026-06-29, the other two
    # 2026-06-30), so an alpha-era saved workflow may still name one. NAMED
    # tombstones rather than the generic unregistered-engine refusal at the
    # director boundary, because "retired" is the truthful diagnosis for a
    # graph that was once valid -- "not registered" reads as a broken install.
    # The adapter FILES are deleted in the same change; resurrection means a
    # real forward, a fresh registration, AND removing the id from this set.
    "triposg_talk",
    "hunyuan3d_talk",
    "trellis_talk",
    "triposr",
    "still_parallax",
    # THE CLOUD PIXVERSE WORD-CARD LANE, RETIRED 2026-09-17 (operator:
    # "WORD_RAZZLE / CLOUD GETS RIPPED"). The adapter was CloudWordRazzleEngine
    # on partner row cloud_pixverse_i2v. Local razzle_ltx_8gb stays; this id
    # is a named tombstone so a saved graph or force-map still fails as
    # RetiredEngineError, never as a silent remap and never as "not registered".
    "word_razzle",
    # THE SEVEN "native"-NAMED LTX 2.5 IDS, RENAMED 2026-09-25 (0b2 step 2,
    # operator: "'native' meant 'ComfyUI's own loaders, not GGUF'; GGUF is
    # gone, the word is noise"). These were the VideoDirector's DEFAULT
    # widget values on four shipped, gallery-published workflows
    # (otr_8gb_ltx25_native_foley/mime/audio_in, otr_24gb_native_foley) --
    # not an obscure lane like wan_i2v above, which the same rename policy
    # deliberately left OFF this set. Plain rename, no alias (operator:
    # "don't worry about back compat") -- but a stale saved copy of one of
    # those four workflows still names these ENGINE ids, and deserves the
    # named RetiredEngineError ("retired and no longer selectable"), not
    # the generic unregistered-engine refusal that reads like a broken
    # install. (A leftover old workflow FILENAME is a different miss: that
    # is a missing file, and this set does not see it.)
    "ltx25_native_foley_16gb",
    "ltx25_native_foley_24gb",
    "ltx25_native_foley_blackwell",
    "ltx25_native_mime_16gb",
    "ltx25_native_mime_24gb",
    "ltx25_native_audio_in_16gb",
    "ltx25_native_audio_in_24gb",
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
