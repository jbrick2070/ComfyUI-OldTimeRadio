"""OTR_ImageDirector -- per-role image granularity / fresh-cap / seed policy (C1).

Captures POLICY only (V-6) and emits ONE ``image_policy_json`` STRING that
``OTR_MetaBriefImagePromptGen`` + ``OTR_ImageGenDispatcher`` consume.

Image-MODEL selection lives in ONE place -- ``OTR_VideoDirector`` (operator
2026-06-18: "only in one place not two"). This node no longer carries its own
per-role image-model dropdowns; it reads the picks from the wired
``video_policy_json["image_models"]`` and owns only the per-role granularity, the
fresh-mode cap, and the seed mode. A slot absent from the policy defaults to the
gen-1 engine (``flux_gen1``) with a LOUD warning.

Role compatibility of each pick is still filtered at execute time via the SHARED
``role_compat.py`` (AS-1) -- a pick that does not fit its role fails closed
(named error), never a silent Flux swap.

3D granularity LOCK (PASS-IMG MUST-FIX #2, hardened per 3D plan section 3): if
a role's paired VIDEO engine declares ``requires_mesh_portrait`` (the REAL
capability field on the adapter -- never a hard-coded family check), that
role's image granularity is hard-locked to ``per_object`` and ``per_beat``
RAISES (fail-closed; fresh-per-beat -> mesh-rebuild-per-beat). 3D-awareness is
CHARACTER-level: ``per_object`` means one clean front-facing portrait per
character used GLOBALLY -- there is NO "3D-ready image mode" widget anywhere
(mesh-friendly framing, if ever needed, is an M4 PROMPT change, V-11). Mesh
retry policy is BOUNDED (one mesh_portrait variant, once per object) and lives
in the 3D adapter -- never a DAG loop here.

``video_policy_json`` is REQUIRED + fail-closed (3D plan section 3): the
OTR_VideoDirector policy must be WIRED provider-before-consumer; an empty or
malformed policy raises instead of silently disabling the 3D lock. An
UNREGISTERED video engine in the policy also raises (its capability cannot be
read -- covers custom_models_json adapters; fail closed, never a silent
not-3D guess).

Determinism (V-7): NO widget named ``seed`` (use ``request_seed``); no
``model_id`` widget (V-11). Cold-import clean: module scope imports only
stdlib + the dep-free registries / role_compat.
"""
from __future__ import annotations

import json
import logging

log = logging.getLogger("OTR")

from ._otr_image_engines import registry as _ireg
from ._otr_video_engines import registry as _vreg
from ._otr_shared import role_compat as _rc
from ._otr_shared import role_slots as _role_slots
from ._otr_image_engines.schemas import GRANULARITY_MODES

#: Sentinel COMBO entry that opens the "declare a custom model" path (mirrors the
#: video director). The custom engine id is read from ``custom_models_json``.
ADD_CUSTOM = "+ Add Custom Model"

#: Which role(s) each image slot must be compatible with (fail-closed filter) --
#: identical mapping to the video director's slots so the shared filter agrees.
IMAGE_SLOT_ROLES = {
    "announcer_image_model": ("announcer_visual",),
    "music_image_model": ("music_visual",),
    "other_beats_image_model": (
        "character_video", "scene_broll", "background_abstract",
    ),
}
SEED_MODES = ("request_hash", "fixed")
#: Operator-confirmable default ceiling for FRESH (per_beat) generation; the hard
#: cap is the audio-derived beat budget applied in the dispatcher (PASS-IMG #3).
DEFAULT_FRESH_CAP = 15


def _image_model_combo() -> list:
    """Every REGISTERED image engine + the custom sentinel (registry IS the menu).

    Built from ``registry.all_engine_names()`` -- there is NO validated-subset
    filter (C4, 2026-06-29): every registered image engine is SELECTABLE (validation
    is the operator's MANUAL process, never a code gate). ``+ Add Custom Model``
    stays the escape hatch."""
    names = list(_ireg.all_engine_names())
    return names + [ADD_CUSTOM]


def _registry_descriptors() -> list:
    """role_compat descriptors for every registered image engine."""
    descs = []
    for name in _ireg.all_engine_names():
        eng = _ireg.get_engine(name)
        descs.append({
            "engine_id": name,
            "roles": tuple(getattr(eng, "roles", ())),
            "required_inputs": tuple(getattr(eng, "required_inputs", ())),
        })
    return descs


def _is_3d_engine(engine_id: str, slot: str = "") -> bool:
    """True if ``engine_id`` is a VIDEO engine that declares the
    ``requires_mesh_portrait`` capability (3D plan section 3 -- the REAL
    schema/adapter field, never a hard-coded family check).

    FAIL CLOSED (never a silent not-3D guess):
    * a non-empty UNREGISTERED engine raises -- its capability cannot be read
      (covers custom_models_json adapters that never registered);
    * a registered ``character_3d``-family engine that does NOT declare the
      capability raises -- the family says 3D but the lock cannot prove it.
    An empty engine_id (an unresolved custom slot) is not 3D -- it cannot
    render at all and the video director already warned LOUDLY.
    """
    if not engine_id:
        return False
    where = f" (video slot {slot!r})" if slot else ""
    if not _vreg.is_registered(engine_id):
        raise ValueError(
            f"OTR_ImageDirector: video engine '{engine_id}'{where} is not "
            f"registered, so its requires_mesh_portrait capability cannot be "
            f"read. The 3D granularity lock FAILS CLOSED on unknown engines "
            f"-- register the adapter (it must declare requires_mesh_portrait"
            f"=True/False) before wiring it into the video policy."
        )
    eng = _vreg.get_engine(engine_id)
    cap = getattr(eng, "requires_mesh_portrait", None)
    if cap is None:
        if getattr(eng, "family", "") == "character_3d":
            raise ValueError(
                f"OTR_ImageDirector: video engine '{engine_id}'{where} is "
                f"character_3d-family but declares no requires_mesh_portrait "
                f"capability; the 3D granularity lock FAILS CLOSED -- add "
                f"requires_mesh_portrait=True to the adapter."
            )
        return False
    return bool(cap)


def three_d_locked_slots(video_policy: dict) -> set:
    """Image slots whose paired VIDEO engine requires a mesh portrait ->
    per_object lock (CHARACTER-level: one portrait per character, global).

    Reads ``video_policy['video_models'][video_slot].engine_id`` for the video
    slot that pairs with each image slot. Pure over the policy dict; raises
    via :func:`_is_3d_engine` when an engine's capability cannot be read
    (fail-closed, 3D plan section 3)."""
    vm = (video_policy or {}).get("video_models") or {}
    # Route-A: each IMAGE slot is locked if ANY of its paired ROLES' video engine
    # requires a mesh portrait. other_beats_image_model now pairs with the three
    # per-role video slots (character / scene_broll / background_abstract), each
    # resolved via the ONE shared map (per-role slot + legacy fallback).
    img_slot_roles = {
        "announcer_image_model": (_rc.Role.ANNOUNCER_VISUAL.value,),
        "music_image_model": (_rc.Role.MUSIC_VISUAL.value,),
        "other_beats_image_model": (
            _rc.Role.CHARACTER_VIDEO.value,
            _rc.Role.SCENE_BROLL.value,
            _rc.Role.BACKGROUND_ABSTRACT.value,
        ),
    }
    locked = set()
    for img_slot, roles in img_slot_roles.items():
        for role in roles:
            engine_id = _role_slots.engine_id_for_role(vm, role)
            if _is_3d_engine(engine_id, slot=_role_slots.slot_for_role(role)):
                locked.add(img_slot)
                break
    return locked


#: Image-prompt ROLE -> the video slot that renders it (the role->engine join the
#: prompt fork needs). Route-A: the ONE shared per-role map
#: (nodes/_otr_shared/role_slots.py); aliased here for any importer of this name.
_ROLE_TO_VIDEO_SLOT = _role_slots.ROLE_TO_VIDEO_SLOT


def _is_mesh_fodder_engine(engine_id: str) -> bool:
    """True if ``engine_id`` is a registered VIDEO engine declaring the
    ``requires_mesh_fodder`` capability (the 3D image-streams routing gate).

    TOLERANT, unlike :func:`_is_3d_engine`: mesh-fodder routing is additive and
    opt-in, so an empty / unregistered / custom engine is simply NOT-fodder
    (False) -- it must never raise and block a normal episode. The capability
    is read off the registered adapter, never an engine-name/family check."""
    if not engine_id or not _vreg.is_registered(engine_id):
        return False
    return bool(getattr(_vreg.get_engine(engine_id), "requires_mesh_fodder",
                        False))


def mesh_fodder_roles_from_video_policy(video_policy: dict) -> list:
    """The IMAGE-PROMPT roles whose paired VIDEO engine requires clean mesh
    fodder (sorted, deterministic). OTR_MetaBriefImagePromptGen reads this off
    the forwarded image policy and forks those beats to a mesh_fodder subject +
    a scene_background_plate instead of one cinematic scene still. Pure over the
    policy dict; tolerant (a non-fodder / unknown engine just drops out)."""
    vm = (video_policy or {}).get("video_models") or {}
    roles = set()
    for role in _role_slots.ROLE_TO_VIDEO_SLOT:
        engine_id = _role_slots.engine_id_for_role(vm, role)
        if _is_mesh_fodder_engine(engine_id):
            roles.add(role)
    return sorted(roles)


def enforce_3d_granularity_lock(granularity_by_slot: dict, locked_slots: set,
                                warnings: list) -> dict:
    """FAIL CLOSED: every 3D-locked slot must already be ``per_object``;
    anything else RAISES (PASS-IMG MUST-FIX #2 hardened per 3D plan section 3
    -- the old coercion-with-a-warning silently hid a mesh-rebuild-per-beat
    policy; the docstring always said per_beat is BANNED for 3D, now the code
    matches it). Returns the (unchanged) granularity dict on success."""
    out = dict(granularity_by_slot)
    bad = sorted(s for s in locked_slots if out.get(s) != "per_object")
    if bad:
        raise ValueError(
            "OTR_ImageDirector: 3D granularity lock violation -- slot(s) "
            f"{bad} pair with a requires_mesh_portrait video engine and MUST "
            f"be per_object (one portrait per character, used globally); "
            f"per_beat would rebuild the mesh per beat. Set the granularity "
            f"widget(s) to per_object -- there is no coercion and no "
            f"'3D-ready mode' widget (fail-closed by design)."
        )
    return out


class OTRImageDirector:
    """Registered as ``OTR_ImageDirector``. Per-role granularity / fresh-cap / seed
    policy; image-model picks come from OTR_VideoDirector via video_policy_json."""

    CATEGORY = "OldTimeRadio/v2/image"
    FUNCTION = "direct"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("image_policy_json",)
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        gran = list(GRANULARITY_MODES)
        return {
            "required": {
                # Image-MODEL selection lives in ONE place -- OTR_VideoDirector
                # (operator 2026-06-18: "only in one place not two"). This node no
                # longer has its own image-model dropdowns; it reads the per-role
                # picks from the wired video_policy_json["image_models"] and owns
                # only the granularity / fresh-cap / seed policy.
                "announcer_granularity": (gran, {"default": "per_object"}),
                "music_granularity": (gran, {"default": "per_object"}),
                "other_beats_granularity": (gran, {
                    "default": "per_object",
                    "tooltip": (
                        "per_object: one image reused per character/prop "
                        "(cheapest; maps to mesh-once for 3D). per_beat: a fresh "
                        "image per beat (capped to the audio beat budget). 3D "
                        "roles are hard-locked to per_object."
                    ),
                }),
                "fresh_cap": ("INT", {
                    "default": DEFAULT_FRESH_CAP, "min": 1, "max": 512,
                    "tooltip": (
                        "Ceiling on FRESH (per_beat) renders per episode; the "
                        "dispatcher additionally hard-caps to the audio-derived "
                        "beat budget (never over-generate)."
                    ),
                }),
                "seed_mode": (list(SEED_MODES), {"default": SEED_MODES[0]}),
                "request_seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xFFFFFFFF,
                    "tooltip": "Base seed (NOT named 'seed' on purpose, V-7).",
                }),
                # REQUIRED + forceInput (3D plan section 3): ComfyUI only
                # enforces a WIRED connection for required inputs, and
                # forceInput keeps this a link socket -- without it ComfyUI
                # would auto-generate a multiline text widget (V-11 / the
                # static-shell violation). Appended LAST so the widget order
                # of the slots above is byte-identical in saved workflows
                # (forceInput inputs never serialize into widgets_values).
                "video_policy_json": ("STRING", {
                    "forceInput": True,
                    "tooltip": (
                        "OTR_VideoDirector policy (REQUIRED; wire the "
                        "provider before this consumer). Used to detect "
                        "requires_mesh_portrait video engines for the 3D "
                        "granularity lock; opaque otherwise. Empty/malformed "
                        "FAILS CLOSED."
                    ),
                }),
            },
            "optional": {
                "custom_models_json": ("STRING", {
                    "multiline": True, "default": "{}",
                    "tooltip": (
                        "When a role is '+ Add Custom Model', map the role key to "
                        'a custom engine id, e.g. {"music_image_model":"z_image"}.'
                    ),
                }),
                "gate_in": ("STRING", {
                    "multiline": True, "default": "", "forceInput": True,
                    "tooltip": "Optional ordering signal (opaque STRING).",
                }),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    # ------------------------------------------------------------------ #
    def direct(self, announcer_granularity,
               music_granularity, other_beats_granularity, fresh_cap,
               seed_mode, request_seed, video_policy_json="",
               custom_models_json="{}", gate_in=""):
        warnings: list = []
        custom = self._parse_json_obj(custom_models_json, "custom_models_json", warnings)
        video_policy = self._parse_video_policy_required(video_policy_json)
        descriptors = _registry_descriptors()

        # SINGLE SOURCE OF TRUTH (operator 2026-06-18: "only in one place not two").
        # Image-MODEL picks live ONLY in OTR_VideoDirector, carried per-role in
        # video_policy["image_models"]. This node has no image-model widgets; it
        # reads those picks verbatim. A slot absent from the policy defaults to the
        # gen-1 engine (flux_gen1, always usable) with a LOUD warning -- never a
        # silent crash on a box-fresh / partially-wired policy. No other swap: the
        # dispatcher hard-fails if the picked engine cannot run.
        vp_images = video_policy.get("image_models")
        vp_images = vp_images if isinstance(vp_images, dict) else {}
        picks = {}
        for slot in ("announcer_image_model", "music_image_model",
                     "other_beats_image_model"):
            vp_pick = vp_images.get(slot)
            if isinstance(vp_pick, str) and vp_pick.strip():
                picks[slot] = vp_pick
            else:
                warnings.append(
                    f"{slot}: no pick in video_policy['image_models']; "
                    f"defaulting to flux_gen1 (gen-1). Set it in OTR_VideoDirector.")
                picks[slot] = "flux_gen1"
        resolved = {}
        for slot, picked in picks.items():
            resolved[slot] = self._resolve_and_validate(
                slot, picked, custom, descriptors, warnings
            )

        granularity = {
            "announcer_image_model": announcer_granularity,
            "music_image_model": music_granularity,
            "other_beats_image_model": other_beats_granularity,
        }
        locked = three_d_locked_slots(video_policy)
        granularity = enforce_3d_granularity_lock(granularity, locked, warnings)

        cap = int(fresh_cap)
        if cap < 1:
            warnings.append(f"fresh_cap {cap} < 1; clamped to 1")
            cap = 1

        policy = {
            "policy_version": 1,
            "image_models": resolved,
            "granularity": granularity,
            "locked_3d_slots": sorted(locked),
            # Per-role still aspect, forwarded from OTR_VideoDirector so MetaBrief
            # mints each character still to MATCH its selected video engine
            # (portrait vs 16:9) with one dropdown pick. {} -> portrait (legacy).
            "aspects": (video_policy.get("aspects")
                        if isinstance(video_policy.get("aspects"), dict) else {}),
            # Per-role SELECTED video engine, forwarded so the image dispatcher can
            # SKIP a still whose video engine does not consume init_image (the
            # visualizer / abstract procedural floor). An all-procedural episode then
            # invokes NO image model -- accessible for users with no image/video
            # models. {} -> dispatch every still (legacy behaviour).
            "video_models": (video_policy.get("video_models")
                             if isinstance(video_policy.get("video_models"), dict)
                             else {}),
            # Other-beats clip plan {clip_mode, pool_n}, forwarded from
            # OTR_VideoDirector so the still phase pools the other-beats stills to
            # match the VIDEO clip pool (pool_n_loop -> N shared stills, not one per
            # beat). {} -> unique_per_beat (one still per beat, legacy).
            "other_beats": (video_policy.get("other_beats")
                            if isinstance(video_policy.get("other_beats"), dict)
                            else {}),
            # 3D image streams (2026-06-21): the IMAGE-prompt roles whose paired
            # video engine requires_mesh_fodder. MetaBrief forks those beats to a
            # clean mesh_fodder subject + a scene_background_plate (NOT one
            # cinematic scene still) so Hunyuan3D meshes an isolated subject, not
            # the whole environment (the clay blob). [] -> no fork (legacy look).
            "mesh_fodder_roles": mesh_fodder_roles_from_video_policy(video_policy),
            "fresh_cap": cap,
            "seed": {"mode": seed_mode, "request_seed": int(request_seed)},
            "warnings": warnings,
        }
        for w in warnings:
            log.warning("[OTR_ImageDirector] %s", w)
        return (json.dumps(policy, ensure_ascii=True, separators=(",", ":")),)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_json_obj(raw, label, warnings) -> dict:
        try:
            data = json.loads(raw or "{}")
            if isinstance(data, dict):
                return data
            warnings.append(f"{label} is not a JSON object; ignored")
        except (ValueError, TypeError):
            warnings.append(f"{label} is not valid JSON; ignored")
        return {}

    @staticmethod
    def _parse_video_policy_required(raw) -> dict:
        """FAIL-CLOSED video-policy parse (3D plan section 3): the policy is
        a REQUIRED wired input; empty, malformed, non-object, or missing its
        ``video_models`` dict RAISES -- a silently-ignored policy is exactly
        the drift that disabled the 3D granularity lock."""
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(
                "OTR_ImageDirector: video_policy_json is EMPTY -- wire "
                "OTR_VideoDirector.video_policy_json into this input "
                "(provider before consumer); the 3D granularity lock cannot "
                "run on an empty policy (fail-closed)."
            )
        try:
            data = json.loads(raw)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"OTR_ImageDirector: video_policy_json is not valid JSON "
                f"({exc}); fail-closed -- fix the wired policy, it is never "
                f"ignored."
            ) from exc
        if not isinstance(data, dict) or not isinstance(
                data.get("video_models"), dict):
            raise ValueError(
                "OTR_ImageDirector: video_policy_json must be the "
                "OTR_VideoDirector policy object (a JSON object carrying a "
                "'video_models' dict); got something else -- fail-closed."
            )
        return data

    @staticmethod
    def _resolve_and_validate(slot, picked, custom, descriptors, warnings):
        """Resolve the sentinel + fail-closed role/engine compatibility check
        via the SHARED role_compat filter (AS-1); never a silent swap."""
        engine_id = picked
        is_custom = picked == ADD_CUSTOM
        if is_custom:
            engine_id = str(custom.get(slot, "") or "")
            if not engine_id:
                warnings.append(
                    f"{slot} is '+ Add Custom Model' but custom_models_json has "
                    f"no '{slot}' entry; left unresolved"
                )
                return {"engine_id": "", "custom": True, "unresolved": True}

        known = {d["engine_id"] for d in descriptors}
        if engine_id in known:
            roles = IMAGE_SLOT_ROLES[slot]
            desc = next(d for d in descriptors if d["engine_id"] == engine_id)
            fits_any = any(_rc.engine_fits_role(desc, role) for role in roles)
            if not fits_any:
                raise ValueError(
                    f"OTR_ImageDirector: engine '{engine_id}' does not fit any "
                    f"role for slot '{slot}' {roles} (incompatible required "
                    f"inputs). Pick a compatible image model -- no silent swap."
                )
        return {"engine_id": engine_id, "custom": is_custom}
