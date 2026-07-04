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
    aspect -> '' (bare id). Pure registry read, never raises."""
    try:
        eng = _vreg.get_engine(str(engine_id))
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
        eng = _vreg.get_engine(str(engine_id))
        family = getattr(eng, "family", None)
        accepts_still = getattr(eng, "accepts_still", None)
    except Exception:  # noqa: BLE001 -- unknown engine -> no descriptor
        return ""
    if family == "abstract" and accepts_still is False:
        return _DESCRIPTOR_ABSTRACT
    return ""


def _label_for(engine_id) -> str:
    """The dropdown LABEL for an engine id:
    ``'<id><aspect suffix><descriptor suffix>'`` (e.g.
    ``'humo_1.7B (portrait)'`` or
    ``'viz_green (16:9) (audio-reactive, no scene image)'``). The descriptor
    suffix (2026-07-01 E4) is auto-derived from the engine's family/``accepts_still``
    (:func:`_descriptor_suffix`) -- never hand-maintained, same contract as
    the aspect suffix. Every suffix starts with ``' ('`` so
    :func:`_engine_id_from_pick`'s first-``' ('``-truncation strips them all
    together; a no-aspect, no-descriptor engine still round-trips to
    the bare id."""
    return "%s%s%s" % (engine_id, _aspect_suffix(engine_id),
                       _descriptor_suffix(engine_id))


#: Legacy engine-id aliases (renamed engines). A saved graph or old ledger that
#: still carries the pre-rename name resolves to the current engine so the pick
#: keeps working. 2026-06-29: flat_still -> still_flat, flux_still -> still_pan
#: (the misleading "flux" name was dropped -- the engine is ffmpeg, never Flux);
#: still_kenburns -> still_motion (the always-renders radio floor was renamed).
#: 2026-06-30 (item 2): visualizer -> viz_green (companion to viz_mxc_cpu /
#: viz_mxc_mandala; the old name no longer has a CAPABILITIES row).
_LEGACY_ENGINE_ALIASES = {"flat_still": "still_flat", "flux_still": "still_pan",
                          "still_kenburns": "still_motion",
                          "visualizer": "viz_green"}


def _engine_id_from_pick(pick) -> str:
    """Parse a dropdown pick back to the bare engine id (the saved/looked-up
    VALUE). Take the token BEFORE the first ' (' so a suffixed label
    (``'humo (portrait)'``) yields ``'humo'``; a bare legacy value with no
    suffix (old saved graphs) passes through unchanged; the ADD_CUSTOM sentinel
    (no ' (') is preserved so the custom path still triggers. A renamed engine's
    old id is mapped via :data:`_LEGACY_ENGINE_ALIASES` so old picks resolve."""
    s = str(pick or "")
    if s == ADD_CUSTOM:
        return s
    idx = s.find(" (")
    bare = s[:idx] if idx != -1 else s
    return _LEGACY_ENGINE_ALIASES.get(bare, bare)

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

    Each entry is the engine's aspect-DERIVED label (``humo (portrait)`` vs
    ``humo_1.7B_169 (16:9)``) so the two HuMo paths are obvious at a glance; the
    SAVED value stays the bare engine id (``direct()`` parses the label back via
    :func:`_engine_id_from_pick`, and a bare legacy value still resolves)."""
    names = list(_vreg.all_engine_names())
    return [_label_for(n) for n in names] + [ADD_CUSTOM]


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
                        "Shipped default humo_14B_169 (the Route-A 14B lane)."
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
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        # Fail-closed role/engine validation runs on the direct() path so a
        # box-fresh graph (empty registry) validates clean here.
        return True

    # ------------------------------------------------------------------ #
    def direct(self, announcer_video_model, music_video_model,
               character_video_model, announcer_image_model,
               music_image_model, character_image_model,
               fps, canvas_w, canvas_h,
               seed_mode, request_seed,
               custom_models_json="{}",
               gate_in=""):
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

        policy = {
            "policy_version": 1,
            "video_models": resolved_video,
            # Per-role still aspect, resolved from each slot's selected engine, so
            # the image node mints character stills to MATCH the chosen video
            # engine with ONE dropdown pick (portrait humo_1.7B vs wide
            # humo_1.7B_169). Opaque to everyone who does not size stills.
            "aspects": self._role_aspects(resolved_video),
            # Per-role TALKING flag (S4b 2026-07-02): whether the selected
            # engine lip-syncs (wants_talking_prompt, the ia2v register), so
            # the image node mints FACE-FORWARD portraits for that lane --
            # proof8 showed brief-styled profile portraits cannot drive lips.
            "talking": self._role_talking(resolved_video),
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
    def _role_aspects(resolved_video):
        """Map each video ROLE to its SELECTED engine's ``render_aspect`` so
        stills match their video engine. One entry PER ROLE
        (announcer_visual / music_visual / character_video), each resolved
        through the shared per-role map (the per-role slot, then the legacy
        character fallback for character). Unknown / custom / unresolved
        picks -> 'portrait' (the safe legacy look). Pure registry read, no
        side effects."""
        def _asp_for_role(role):
            eid = _role_slots.engine_id_for_role(resolved_video, role)
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
    def _role_talking(resolved_video):
        """Map each video ROLE to whether its SELECTED engine renders TALKING
        lip-sync (the engine's ``wants_talking_prompt`` hook -- the ia2v
        register), so stills can be minted face-forward for that lane (S4b).
        Hook errors resolve False here: the RENDER path stays the loud
        enforcer of a misconfigured recipe; the director only styles stills."""
        def _talk_for_role(role):
            eid = _role_slots.engine_id_for_role(resolved_video, role)
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

        # Validate only against engines the registry actually knows about
        # (CW-1 registry is empty -> nothing to validate; correct once adapters
        # land). A known engine that fits NONE of the slot's roles fails closed.
        known = {d["engine_id"] for d in descriptors}
        if engine_id in known:
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
