"""OTR_VideoDirector -- the per-role model-selection UI (A-S1/W1).

Captures POLICY only (V-6): per-role video model + per-role image model,
canvas/fps, seed mode. It emits ONE ``video_policy_json`` STRING that
``OTR_ShotLock`` consumes -- an explicit string socket (testable, no hidden
coupling).

Model-agnostic, no "primary": each per-role COMBO is the FULL static video
registry + a ``+ Add Custom Model`` sentinel (V-6: the enum is the full list;
role compatibility is filtered at execute time via the shared
``role_compat.py``, NEVER by mutating the COMBO). A role's pick that does not fit
the role fails closed at execute (named error), never a silent swap.

Determinism (V-7): there is NO widget named ``seed`` (a literal ``seed`` widget
is the dynamic-mutation trap); the seed knob is ``request_seed`` + a
``seed_mode``. No ``model_id`` widget (V-11). Import-time is side-effect-free;
module scope imports only stdlib + the dep-free registry / role_compat. UTF-8,
no BOM, ASCII source.
"""
from __future__ import annotations

import json
import logging

log = logging.getLogger("OTR")

from ._otr_video_engines import registry as _vreg
from ._otr_image_engines import registry as _ireg
from ._otr_shared import role_compat as _rc
from ._otr_shared import role_slots as _role_slots


def _engine_for_role(resolved_video, effective_by_role, role):
    """The engine a per-role derivation should read (2026-07-25, chunk 1b).

    When the caller supplies the FROZEN route map, that map wins -- derived
    values (still aspect, talking flag) must describe the engine that actually
    renders, not the one the operator picked, or a redirected beat gets a still
    sized for an engine that never runs. When it is absent, this falls back to
    the picked slot map, which preserves every pre-1b caller's meaning exactly.
    """
    if effective_by_role:
        eid = str((effective_by_role or {}).get(role) or "")
        if eid:
            return eid
    return _role_slots.engine_id_for_role(resolved_video, role)


# Video-tiers (2026-07-20): the public-name resolver is the SINGLE source of truth
# for the menu id <-> internal id mapping AND the legacy-alias table (moved out of
# this module). Dep-free / cold-import clean.
from ._otr_shared.public_engines import (
    _INTERNAL_TO_PUBLIC,
    _LEGACY_ENGINE_ALIASES,
    check_retired_engine,
    resolve_engine_id,
)

#: Sentinel COMBO entry that opens the "declare a custom model" path. When a role
#: is set to this, its real engine id is read from the ``custom_models_json``
#: widget (role_key -> engine_id). The custom adapter itself loads in CW-4+.
ADD_CUSTOM = "+ Add Custom Model"

#: Default image sources until the image registry lands (C1). Flux = "gen 1"
#: (the spine's default image source); swapping it never touches video models.
IMAGE_DEFAULTS = ("Flux (gen 1)",)

#: render_aspect -> the human suffix shown after the engine id in the dropdown.
#: The label is DERIVED from each engine's render_aspect, never a hand-maintained
#: map -- it can never drift and it auto-labels every engine (portrait HuMo vs
#: 16:9 HuMo at a glance). An engine with no/unknown render_aspect gets NO suffix
#: (the bare id), so the saved value is unchanged.
_ASPECT_SUFFIX = {"portrait": " (portrait)", "wide": " (16:9)"}


def _aspect_suffix(engine_id) -> str:
    """The display suffix for ``engine_id`` derived from its ``render_aspect``
    (portrait -> ' (portrait)', wide -> ' (16:9)'). Unknown engine / unknown
    aspect -> '' (bare id). Resolves a public / legacy id to its internal id
    first so the registry read hits the concrete engine. Pure, never raises."""
    try:
        eng = _vreg.get_engine(resolve_engine_id(engine_id))
        aspect = getattr(eng, "render_aspect", None)
    except Exception:  # noqa: BLE001 -- unknown engine -> no suffix
        aspect = None
    return _ASPECT_SUFFIX.get(aspect, "")


#: Plain-language behaviour caveat shown AFTER the id+aspect+VRAM suffixes for an
#: audio-reactive visualizer -- a family ``"abstract"`` engine that mints NO scene
#: image (``accepts_still is False``). Makes it obvious in the dropdown that a
#: music/announcer/scene pick of one of these IGNORES the scene still and reacts
#: to audio instead (the operator's "abstract/viz_green = audio-reactive, no scene
#: image" label ask, 2026-07-01 E4). Starts with ``' ('`` so
#: :func:`_engine_id_from_pick` strips it with the other suffixes.
_DESCRIPTOR_ABSTRACT = " (audio-reactive, no scene image)"


def _descriptor_suffix(engine_id) -> str:
    """The behaviour-descriptor suffix for ``engine_id``, DERIVED from registry
    attributes (never a hand-maintained per-engine map -- same no-drift contract
    as the aspect + VRAM suffixes). Today only the audio-reactive visualizers
    (family ``"abstract"`` that mint no still) carry one; every other engine ->
    ``''`` (bare label unchanged). Pure registry read, never raises."""
    try:
        eng = _vreg.get_engine(resolve_engine_id(engine_id))
        family = getattr(eng, "family", None)
        accepts_still = getattr(eng, "accepts_still", None)
    except Exception:  # noqa: BLE001 -- unknown engine -> no descriptor
        return ""
    if family == "abstract" and accepts_still is False:
        return _DESCRIPTOR_ABSTRACT
    return ""


def _label_for(engine_id) -> str:
    """The dropdown LABEL for an INTERNAL engine id:
    ``'<public-or-internal id><aspect suffix><descriptor suffix>'`` (e.g.
    ``'wan_8gb (16:9)'`` for internal ``wan_ti2v``, ``'humo_1.7B (portrait)'``,
    or ``'viz_green (16:9) (audio-reactive, no scene image)'``).

    Video-tiers (2026-07-20): the visible token is the PUBLIC menu id when the engine
    has one (``_INTERNAL_TO_PUBLIC``), else the bare internal id. The four
    original tier rows read ``wan_8gb`` / ``ltx_8gb`` / ``ltx23_16gb_audio_in``
    / ``ltx23_16gb_video``; ALL FOUR of those ``<vramtier>gb`` spellings were
    retired one lane at a time by the video transplant build (lanes 5-9,
    2026-08-11) and now resolve through ``_LEGACY_ENGINE_ALIASES``, so the live
    menu reads ``wan22_high_video`` / ``ltx098_low_video`` /
    ``ltx23_low_audio_in`` / ``ltx23_high_video``. Every other engine keeps its
    id. The suffixes are
    still DERIVED from the engine's own ``render_aspect`` / family (passed the
    INTERNAL id). Every suffix starts with ``' ('`` so
    :func:`_engine_id_from_pick`'s resolver strips them all and round-trips the label
    back to the internal id (``_engine_id_from_pick(_label_for('wan_ti2v'))=='wan_ti2v'``)."""
    public = _INTERNAL_TO_PUBLIC.get(engine_id, engine_id)
    return "%s%s%s" % (public, _aspect_suffix(engine_id),
                       _descriptor_suffix(engine_id))


# _LEGACY_ENGINE_ALIASES now lives in _otr_shared.public_engines (the SINGLE source
# of truth for the menu-id <-> internal-id + legacy-alias mapping) and is imported
# above; every boundary reads that one table. (2026-06-29 flat_still/flux_still/
# still_kenburns renames + 2026-06-30 visualizer->viz_green are preserved there.)


def _engine_id_from_pick(pick) -> str:
    """Parse a dropdown pick back to the concrete internal engine id (the
    saved/looked-up VALUE) via the shared resolver: strip the ' (' suffix, map a
    PUBLIC menu id (``'wan_8gb (16:9)'`` -> ``'wan_ti2v'``) then a LEGACY id
    (``'visualizer'`` -> ``'viz_green'``) to the current internal id. A bare
    internal value (old saved graphs) and the ADD_CUSTOM sentinel pass through
    unchanged -- fully back-compatible."""
    return resolve_engine_id(pick)

#: Which role(s) each video slot must be compatible with (fail-closed filter).
#: The ONE shared map (nodes/_otr_shared/role_slots.py). Three first-class video
#: slots -- announcer / music / character (2026-07-03: the legacy catch-all
#: video slot + its migration fallback were retired; character is its own slot).
VIDEO_SLOT_ROLES = _role_slots.VIDEO_SLOT_ROLES
SEED_MODES = ("request_hash", "fixed")


def _video_model_combo() -> list:
    """Every REGISTERED video engine + the custom sentinel (registry IS the menu).

    Built from ``registry.all_engine_names()`` -- there is NO validated-subset
    filter (C4, 2026-06-29): a registered engine is SELECTABLE and renders (it may
    hard-fail LOUD; validation is the operator's MANUAL process, never a code gate).
    ``+ Add Custom Model`` stays the escape hatch for an explicitly-declared engine.

    Each entry is the engine's PUBLIC-or-internal label (``wan_8gb (16:9)`` for
    internal ``wan_ti2v``, ``humo (portrait)``, ...) so the tier rows read by their
    public name; the SAVED value is that same label (``direct()`` resolves it back
    to the internal id via :func:`_engine_id_from_pick`). Deduped by PUBLIC id so a
    public row can never appear twice (a defensive no-op given the bijection)."""
    seen: set = set()
    out: list = []
    for internal in _vreg.all_engine_names():
        public = _INTERNAL_TO_PUBLIC.get(internal, internal)
        if public in seen:
            continue
        seen.add(public)
        out.append(_label_for(internal))
    return out + [ADD_CUSTOM]


def exact_menu_option_for(internal_id) -> str:
    """The UNIQUE live-combo option (menu label) that resolves to ``internal_id``.

    Used by the applier + build_variants to write the EXACT string the UI would save
    for a given internal engine (e.g. ``wan_ti2v`` -> ``'wan_8gb (16:9)'``), so a
    generated variant stores a real, round-trippable menu value rather than a bare
    internal id. Fails LOUD on 0 or >1 matches (a menu/registry inconsistency)."""
    opts = [o for o in _video_model_combo()
            if o != ADD_CUSTOM and _engine_id_from_pick(o) == internal_id]
    if len(opts) != 1:
        raise ValueError(
            "exact_menu_option_for(%r): expected exactly ONE live combo option "
            "resolving to it, found %d: %r" % (internal_id, len(opts), opts))
    return opts[0]


def _image_model_combo() -> list:
    """Every REGISTERED image engine + the custom sentinel (registry IS the menu).

    Built from ``image_registry.all_engine_names()`` -- there is NO validated-subset
    filter (C4, 2026-06-29): every registered image engine is SELECTABLE (validation
    is the operator's MANUAL process, never a code gate). Falls back to
    ``IMAGE_DEFAULTS`` only if the registry is somehow empty so a box-fresh graph
    still validates. ``+ Add Custom Model`` is the escape hatch."""
    names = list(_ireg.all_engine_names()) or list(IMAGE_DEFAULTS)
    return names + [ADD_CUSTOM]


def _registry_descriptors() -> list:
    """role_compat descriptors for every registered video engine."""
    descs = []
    for name in _vreg.all_engine_names():
        eng = _vreg.get_engine(name)
        descs.append({
            "engine_id": name,
            "roles": tuple(getattr(eng, "roles", ())),
            "required_inputs": tuple(getattr(eng, "required_inputs", ())),
        })
    return descs


class OTRVideoDirector:
    """Registered as ``OTR_VideoDirector``. Per-role model-selection policy."""

    CATEGORY = "OldTimeRadio/v2/video"
    FUNCTION = "direct"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_policy_json",)
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        video = _video_model_combo()
        image = _image_model_combo()
        return {
            "required": {
                "announcer_video_model": (video, {
                    "tooltip": "Video model for ANNOUNCER beats (role A).",
                }),
                "music_video_model": (video, {
                    "tooltip": "Video model for MUSIC beats (role B).",
                }),
                "character_video_model": (video, {
                    "tooltip": (
                        "Video model for CHARACTER beats (face + audio, role C). "
                        "Uses the saved dropdown value; profiles only retarget "
                        "this when explicitly applied."
                    ),
                }),
                "announcer_image_model": (image, {
                    "tooltip": "Image source for the announcer (feeds its video).",
                }),
                "music_image_model": (image, {
                    "tooltip": "Image source for music beats.",
                }),
                "character_image_model": (image, {
                    "tooltip": "Image source for character beats (kept slot).",
                }),
                "fps": ("INT", {"default": 25, "min": 1, "max": 60}),
                "canvas_w": ("INT", {"default": 832, "min": 16, "max": 7680}),
                "canvas_h": ("INT", {"default": 480, "min": 16, "max": 4320}),
                "seed_mode": (list(SEED_MODES), {
                    "default": SEED_MODES[0],
                    "tooltip": "request_hash (deterministic) | fixed.",
                }),
                "request_seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xFFFFFFFF,
                    "tooltip": "Base seed (NOT named 'seed' on purpose, V-7).",
                }),
            },
            "optional": {
                "custom_models_json": ("STRING", {
                    "multiline": True,
                    "default": "{}",
                    "tooltip": (
                        "When a role is set to '+ Add Custom Model', map the "
                        "role key to a custom engine id here, e.g. "
                        '{"character_video_model": "my_engine"}.'
                    ),
                }),
                "gate_in": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "forceInput": True,
                    "tooltip": "Optional ordering signal (opaque STRING).",
                }),
                # S5 platform-portability (2026-07-10): explicit device/dtype
                # policy widgets (append-only; widget slots 12-13). Defaults
                # = nv50 baseline; emitted in the v2 policy and enforced at
                # the adapter boundary (S4).
                "device_policy": (["cuda", "cpu", "mps"], {
                    "default": "cuda",
                    "tooltip": "EXPLICIT local video render device (platform "
                               "profile value). No auto-detect.",
                }),
                "dtype_policy": (["fp8_ok", "no_fp8", "no_fp8_no_fp4"], {
                    "default": "fp8_ok",
                    "tooltip": "Dtype lanes allowed for local video engines "
                               "(fp8/fp4 artifacts are OFF on ROCm/MPS "
                               "tiers).",
                }),
                # WAN 8GB launch contract (2026-07-24): the low-VRAM tier's
                # render-length ceiling, appended LAST (widget slot 14) so no
                # saved positional widget vector shifts (BUG-LOCAL-097).
                "max_render_frames": ("INT", {
                    "default": 0, "min": 0, "max": 240,
                    "tooltip": (
                        "Absolute per-clip RENDER-length ceiling in frames for "
                        "local video engines (0 = unpinned = the engine's own "
                        "maximum). Beat LENGTH is never changed by it -- this "
                        "caps what an engine renders, never what the episode "
                        "plays. TWO LANES CONSUME IT DIFFERENTLY. ltx_8gb "
                        "takes it as a COVERAGE-PLANNING cap: the beat is "
                        "partitioned into real clips of at most this many "
                        "frames and chained, so the ceiling decides how many "
                        "clips cover the beat. The 8GB Wan tier takes it as an "
                        "adapter-side NATIVE cap: it pins 17, renders 17 real "
                        "frames, and ping-pong-extends them to the beat's "
                        "audio length rather than requesting the whole "
                        "177-frame span and dying in the VRAM cost model. A "
                        "value at or above an engine's own maximum has no "
                        "effect on that engine."
                    ),
                }),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        # Fail-closed role/engine validation runs on the direct() path so a
        # box-fresh graph (empty registry) validates clean here.
        return True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Re-run whenever the ROUTING ENVIRONMENT changes (2026-07-25, 1b).

        This node is where effective routing is FROZEN, and it freezes state
        that lives outside its inputs: ``OTR_FORCE_ENGINE_MAP`` and
        ``OTR_ENABLE_HUMO_HOSTS``. Without this, ComfyUI would serve a cached
        policy across an env change and every downstream consumer -- MetaBrief,
        ShotLock, the dispatcher -- would plan against a route the operator has
        already changed. Widget-only changes are still handled by ComfyUI's
        ordinary input hashing; this covers ONLY the ambient state.
        """
        from ._otr_shared import route_freeze as _rf
        return _rf.snapshot_fingerprint(_rf.routing_env_snapshot())

    # ------------------------------------------------------------------ #
    def direct(self, announcer_video_model, music_video_model,
               character_video_model, announcer_image_model,
               music_image_model, character_image_model,
               fps, canvas_w, canvas_h,
               seed_mode, request_seed,
               custom_models_json="{}",
               gate_in="",
               device_policy="cuda", dtype_policy="fp8_ok",
               max_render_frames=0):
        warnings: list = []
        custom = self._parse_custom(custom_models_json, warnings)
        descriptors = _registry_descriptors()

        # Parse each dropdown pick (an aspect-labelled string like
        # "humo (portrait)") back to its bare engine id BEFORE resolve/validate.
        # A bare legacy value (old saved graph) and the ADD_CUSTOM sentinel pass
        # through unchanged, so this is fully back-compatible.
        video_models = {
            "announcer_video_model": _engine_id_from_pick(announcer_video_model),
            "music_video_model": _engine_id_from_pick(music_video_model),
            "character_video_model": _engine_id_from_pick(character_video_model),
        }
        resolved_video = {}
        for slot, picked in video_models.items():
            resolved_video[slot] = self._resolve_and_validate(
                slot, picked, custom, descriptors, warnings
            )
        # NO FALLBACKS (operator 2026-07-03): every video role must resolve to a
        # real engine. An empty NON-custom slot fails LOUD here (never a silent
        # lane); an '+ Add Custom Model' pick stays a warning (declare-later).
        for role, slot in _role_slots.ROLE_TO_VIDEO_SLOT.items():
            rv = resolved_video.get(slot, {})
            if not rv.get("engine_id") and not rv.get("custom"):
                raise ValueError(
                    "OTR_VideoDirector: video role %r (slot %r) resolved to an "
                    "EMPTY engine. Pick a model -- NO FALLBACK." % (role, slot))

        # THE ROUTE FREEZE (2026-07-25, multi-clip coverage chunk 1b; kibitz r4
        # codex gpt-5.6-sol high + agy, then a six-way grounded Sonnet fan-out).
        # This node is the UNIQUE COMMON ANCESTOR of both downstream branches --
        # verified against workflows/otr_canonical.json's link list, node 89
        # (MetaBrief) and node 90 (ShotLock) are INDEPENDENT branches that
        # reconverge at node 91, with no 89->90 edge. Ascending node ids are NOT
        # an execution order. So a freeze taken at ShotLock can never inform the
        # image phase, and node 87 is the only point that can inform both.
        # Everything downstream consumes THIS map instead of re-deriving the
        # route from ambient env (the four-mirror defect retired at chunk 1a).
        from ._otr_shared import route_freeze as _rf
        routing_snapshot = _rf.routing_env_snapshot()
        effective_video = _rf.freeze_role_engines(
            resolved_video, snapshot=routing_snapshot)
        for _role, _eff in sorted(effective_video.items()):
            _picked = _role_slots.engine_id_for_role(resolved_video, _role)
            if _eff != _picked:
                log.warning(
                    "[OTR_VideoDirector] ROUTE FREEZE: role=%s picked=%s -> "
                    "EFFECTIVE=%s (force map and/or radio-is-host redirect); "
                    "stills, prompts and shots all plan against %s",
                    _role, _picked, _eff, _eff)

        # THE MEMBERSHIP BOUNDARY (lean-mean order 3, 2026-08-23). Every
        # non-empty engine id this node hands downstream -- the per-slot PICKS
        # and the post-freeze EFFECTIVE map -- must be a REGISTERED video
        # engine, proven HERE, at the unique common ancestor, seconds into the
        # graph. Until now the director deliberately skipped ids the registry
        # did not know (`if engine_id in known:` above, the CW-1 carve-out),
        # so a stale saved pick or a typo'd custom id sailed through and was
        # caught one node later by OTR_ImageDirector's 3D granularity lock --
        # whose message ("add requires_mesh_portrait=True/False") misdiagnoses
        # a stale workflow as an adapter bug. Downstream of THAT guard,
        # ShotLock / MetaBrief / the dispatcher tolerate unknown ids by
        # design, so once the dormant-3D family (and the lock riding on it)
        # retires in order 4, an unknown id would plan portraits and mint
        # stills for minutes and then die mid-render at assert_usable -- the
        # forbidden failure shape. Hence this boundary lands FIRST, additive,
        # with the old guard untouched until this commit is pushed.
        #
        # Scope, deliberately the UNION of picked and effective: a pick that
        # only renders because a force map happens to mask it is a booby trap
        # that detonates the day the env var is unset, so it fails here too,
        # with a message that names both knobs. Unresolved custom slots
        # ({"engine_id": "", "custom": True}) stay a WARNING (declare-later,
        # the carve-out above). RETIRED ids never reach this check, though the
        # two paths fail with DIFFERENT named errors (QA-verified, not
        # assumed): a retired PICK raises RetiredEngineError from
        # _resolve_and_validate, while a retired FORCE-MAP value raises
        # RouteFreezeError from parse_force_map -- which catches everything
        # except its own type and re-wraps, embedding the retired message in
        # its text. Both fire inside/before freeze_role_engines above, so
        # neither can fall through to the generic message below.
        _unregistered = []
        for _slot, _rv in sorted(resolved_video.items()):
            _eid = str(_rv.get("engine_id") or "")
            if _eid and not _vreg.is_registered(resolve_engine_id(_eid)):
                _unregistered.append(f"slot {_slot!r} picked {_eid!r}")
        for _role, _eff_id in sorted(effective_video.items()):
            _eid = str(_eff_id or "")
            if _eid and not _vreg.is_registered(resolve_engine_id(_eid)):
                _unregistered.append(f"role {_role!r} effective {_eid!r}")
        if _unregistered:
            _fmap = str(routing_snapshot.get("force_engine_map", "") or "")
            raise ValueError(
                "OTR_VideoDirector: %s -- not a registered video engine "
                "(registered: %s). A stale saved workflow, a typo'd "
                "custom_models_json entry, or a force-map value names an "
                "engine this build does not register; fix the pick or re-save "
                "the workflow.%s NO FALLBACK -- failing here, seconds in, "
                "instead of mid-render at assert_usable." % (
                    "; ".join(_unregistered),
                    ", ".join(sorted(_vreg.all_engine_names())),
                    (" OTR_FORCE_ENGINE_MAP is active (%r) -- the map or the "
                     "pick it masks may be the stale half." % _fmap)
                    if _fmap else ""))

        policy = {
            # S4 platform-portability (2026-07-10): version 2 adds the
            # explicit device/dtype policy fields (defaults = nv50
            # baseline; the S5 widgets feed them 1:1). Every consumer
            # (ImageDirector, ShotLock, dispatcher, render_driver)
            # asserts policy_version == 2.
            "policy_version": 2,
            "device_policy": str(device_policy or "cuda"),
            "dtype_policy": str(dtype_policy or "fp8_ok"),
            # WAN 8GB launch contract (2026-07-24): the per-tier render-length
            # ceiling rides the SAME v2 policy channel as device/dtype, so a
            # production episode leg carries it without depending on the
            # server's boot environment. 0 = unpinned (every shipped tier).
            "max_render_frames": max(0, int(max_render_frames or 0)),
            "video_models": resolved_video,
            # THE FROZEN ROUTE (chunk 1b): role -> the engine that will ACTUALLY
            # render the beat, force map and radio-is-host redirect already
            # applied. ``video_models`` above stays the PICKED map -- both are
            # stamped so an equality failure downstream can be replayed and
            # audited rather than guessed at.
            "effective_video_models": effective_video,
            # The exact routing environment this policy was frozen from.
            # ShotLock rejects a mismatch against the live environment.
            "routing_env_snapshot": routing_snapshot,
            # Per-role still aspect, resolved from each role's EFFECTIVE engine, so
            # the image node mints character stills to MATCH the engine that
            # actually renders (portrait humo_1.7B vs wide humo_1.7B_169).
            # Opaque to everyone who does not size stills.
            #
            # EFFECTIVE, NOT PICKED (2026-07-25, chunk 1b) -- this was a LIVE
            # DEFAULT-ENV BUG, not a latent one. Picking a portrait HuMo for
            # announcer_visual with OTR_ENABLE_HUMO_HOSTS unset redirects the
            # render to the WIDE ltx_audio_in console, but the aspect map was
            # derived from the PICKED portrait engine -- so a 832x1216 portrait
            # still was minted and the wide render centre-cropped it. The
            # consequence is recorded verbatim in eng_ltx_av.py:345-347:
            # "the director defaulted to a 832x1216 PORTRAIT still that the wide
            # render then centre-cropped, lopping the subject's head off."
            "aspects": self._role_aspects(resolved_video, effective_video),
            # Per-role TALKING flag (S4b 2026-07-02): whether the engine
            # lip-syncs (wants_talking_prompt, the ia2v register), so
            # the image node mints FACE-FORWARD portraits for that lane --
            # proof8 showed brief-styled profile portraits cannot drive lips.
            # Also EFFECTIVE as of chunk 1b: MetaBrief's _effective_talking_roles
            # already upgraded False->True off the effective engine, but only in
            # that one direction and only in that one node.
            "talking": self._role_talking(resolved_video, effective_video),
            "image_models": {
                "announcer_image_model": announcer_image_model,
                "music_image_model": music_image_model,
                "character_image_model": character_image_model,
            },
            "canvas": {"w": int(canvas_w), "h": int(canvas_h), "fps": int(fps)},
            "seed": {"mode": seed_mode, "request_seed": int(request_seed)},
            "warnings": warnings,
        }
        for w in warnings:
            log.warning("[OTR_VideoDirector] %s", w)
        return (json.dumps(policy, ensure_ascii=True, separators=(",", ":")),)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_custom(custom_models_json, warnings) -> dict:
        try:
            data = json.loads(custom_models_json or "{}")
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items()}
            warnings.append("custom_models_json is not a JSON object; ignored")
        except (ValueError, TypeError):
            warnings.append("custom_models_json is not valid JSON; ignored")
        return {}

    @staticmethod
    def _role_aspects(resolved_video, effective_by_role=None):
        """Map each video ROLE to its EFFECTIVE engine's ``render_aspect`` so
        stills match the engine that actually renders. One entry PER ROLE
        (announcer_visual / music_visual / character_video). Unknown / custom /
        unresolved picks -> 'portrait' (the safe legacy look). Pure registry
        read, no side effects.

        ``effective_by_role`` is the frozen route map (chunk 1b). It is
        OPTIONAL so the many existing callers that pass only the slot map keep
        their meaning -- without it this resolves the PICKED engine exactly as
        before, which is correct for any caller that has no route to freeze."""
        def _asp_for_role(role):
            eid = _engine_for_role(resolved_video, effective_by_role, role)
            try:
                eng = _vreg.get_engine(eid)
                if getattr(eng, "render_aspect", "portrait") == "wide":
                    return "wide"
            except Exception:  # noqa: BLE001 -- unknown engine -> portrait
                pass
            return "portrait"
        return {
            "announcer_visual": _asp_for_role("announcer_visual"),
            "music_visual": _asp_for_role("music_visual"),
            "character_video": _asp_for_role("character_video"),
        }

    @staticmethod
    def _role_talking(resolved_video, effective_by_role=None):
        """Map each video ROLE to whether its EFFECTIVE engine renders TALKING
        lip-sync (the engine's ``wants_talking_prompt`` hook -- the ia2v
        register), so stills can be minted face-forward for that lane (S4b).
        Hook errors resolve False here: the RENDER path stays the loud
        enforcer of a misconfigured recipe; the director only styles stills.

        ``effective_by_role`` is the frozen route map (chunk 1b), OPTIONAL for
        the same reason as :meth:`_role_aspects`."""
        def _talk_for_role(role):
            eid = _engine_for_role(resolved_video, effective_by_role, role)
            try:
                eng = _vreg.get_engine(eid)
                fn = getattr(eng, "wants_talking_prompt", None)
                return bool(fn()) if callable(fn) else False
            except Exception:  # noqa: BLE001 -- unknown/misconfigured -> False
                return False
        return {
            "announcer_visual": _talk_for_role("announcer_visual"),
            "music_visual": _talk_for_role("music_visual"),
            "character_video": _talk_for_role("character_video"),
        }

    @staticmethod
    def _resolve_and_validate(slot, picked, custom, descriptors, warnings):
        """Resolve the sentinel + fail-closed role/engine compatibility check."""
        engine_id = picked
        is_custom = picked == ADD_CUSTOM
        if is_custom:
            engine_id = custom.get(slot, "")
            if not engine_id:
                warnings.append(
                    f"{slot} is '+ Add Custom Model' but custom_models_json has "
                    f"no '{slot}' entry; left unresolved"
                )
                return {"engine_id": "", "custom": True, "unresolved": True}

        # A RETIRED id (a stale saved graph / custom_models_json entry) fails
        # with the NAMED policy error -- never a silent resolve to another
        # engine. Runs AFTER pick/custom resolution (the pick was already
        # public/legacy-resolved by _engine_id_from_pick).
        check_retired_engine(resolve_engine_id(engine_id))

        # Validate only against engines the registry actually knows about
        # (CW-1 registry is empty -> nothing to validate; correct once adapters
        # land). A known engine that fits NONE of the slot's roles fails closed.
        known = {d["engine_id"] for d in descriptors}
        if engine_id in known:
            if _vreg.is_registered(engine_id):
                eng = _vreg.get_engine(engine_id)
                if not getattr(eng, "invocable", True):
                    reason = getattr(eng, "invocability_reason", "not runnable")
                    raise _vreg.EngineNotRunnableError(engine_id, reason, slot)

            roles = VIDEO_SLOT_ROLES[slot]
            fits_any = any(
                _rc.engine_fits_role(
                    next(d for d in descriptors if d["engine_id"] == engine_id),
                    role,
                )
                for role in roles
            )
            if not fits_any:
                raise ValueError(
                    f"OTR_VideoDirector: engine '{engine_id}' does not fit any "
                    f"role for slot '{slot}' {roles} (incompatible required "
                    f"inputs). Pick a compatible model -- no silent swap."
                )
        return {"engine_id": engine_id, "custom": is_custom}
