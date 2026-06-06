"""OTR_ImageDirector -- the per-role image-model selection UI (C1).

The image-side mirror of ``OTR_VideoDirector``, one level upstream: it captures
POLICY only (V-6) -- per-role image engine + per-role granularity + the
fresh-mode cap + seed mode -- and emits ONE ``image_policy_json`` STRING that
``OTR_MetaBriefImagePromptGen`` + ``OTR_ImageGenDispatcher`` consume.

Model-agnostic, no "primary": each per-role COMBO is the FULL static image
registry + a ``+ Add Custom Model`` sentinel; role compatibility is filtered at
execute time via the SHARED ``role_compat.py`` (AS-1) -- NEVER by mutating the
COMBO, and NEVER a private copy of the filter. A pick that does not fit its role
fails closed (named error), never a silent Flux swap.

3D granularity LOCK (PASS-IMG MUST-FIX #2): if a role's VIDEO engine is a
``character_3d`` family (Subproject B), that role's image granularity is
hard-locked to ``per_object`` and ``per_beat`` is rejected (fresh-per-beat ->
mesh-rebuild-per-beat). Determinism (V-7): NO widget named ``seed`` (use
``request_seed``); no ``model_id`` widget (V-11). Cold-import clean: module scope
imports only stdlib + the dep-free registries / role_compat.
"""
from __future__ import annotations

import json
import logging

log = logging.getLogger("OTR")

from ._otr_image_engines import registry as _ireg
from ._otr_video_engines import registry as _vreg
from ._otr_shared import role_compat as _rc
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
    """Full static image registry + the custom sentinel (V-6)."""
    return list(_ireg.all_engine_names()) + [ADD_CUSTOM]


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


def _is_3d_engine(engine_id: str) -> bool:
    """True if ``engine_id`` is a registered VIDEO engine of the character_3d
    family (Subproject B). Used to drive the granularity LOCK. Fail-soft: an
    unknown engine is not 3D."""
    if not engine_id or not _vreg.is_registered(engine_id):
        return False
    return getattr(_vreg.get_engine(engine_id), "family", "") == "character_3d"


def three_d_locked_slots(video_policy: dict) -> set:
    """Image slots whose paired VIDEO engine is character_3d -> per_object lock.

    Reads ``video_policy['video_models'][video_slot].engine_id`` for the video
    slot that pairs with each image slot. Pure; fail-soft on a malformed policy.
    """
    vm = (video_policy or {}).get("video_models") or {}
    pair = {
        "announcer_image_model": "announcer_video_model",
        "music_image_model": "music_video_model",
        "other_beats_image_model": "other_beats_video_model",
    }
    locked = set()
    for img_slot, vid_slot in pair.items():
        entry = vm.get(vid_slot)
        engine_id = entry.get("engine_id") if isinstance(entry, dict) else str(entry or "")
        if _is_3d_engine(engine_id):
            locked.add(img_slot)
    return locked


def enforce_3d_granularity_lock(granularity_by_slot: dict, locked_slots: set,
                                warnings: list) -> dict:
    """Force ``per_object`` on every 3D-locked slot; WARN if the user picked
    ``per_beat`` (PASS-IMG MUST-FIX #2 -- per_beat is BANNED for 3D)."""
    out = dict(granularity_by_slot)
    for slot in locked_slots:
        if out.get(slot) != "per_object":
            warnings.append(
                f"{slot}: granularity locked to per_object (paired video engine "
                f"is character_3d; per_beat would rebuild the mesh per beat)"
            )
            out[slot] = "per_object"
    return out


class OTRImageDirector:
    """Registered as ``OTR_ImageDirector``. Per-role image-model + granularity policy."""

    CATEGORY = "OldTimeRadio/v2/image"
    FUNCTION = "direct"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("image_policy_json",)
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        image = _image_model_combo()
        gran = list(GRANULARITY_MODES)
        return {
            "required": {
                "announcer_image_model": (image, {
                    "tooltip": "Image engine for the ANNOUNCER (feeds its video).",
                }),
                "music_image_model": (image, {
                    "tooltip": "Image engine for MUSIC beats.",
                }),
                "other_beats_image_model": (image, {
                    "tooltip": "Image engine for all OTHER beats (character / scene / bg).",
                }),
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
            },
            "optional": {
                "custom_models_json": ("STRING", {
                    "multiline": True, "default": "{}",
                    "tooltip": (
                        "When a role is '+ Add Custom Model', map the role key to "
                        'a custom engine id, e.g. {"music_image_model":"z_image"}.'
                    ),
                }),
                "video_policy_json": ("STRING", {
                    "multiline": True, "default": "{}", "forceInput": True,
                    "tooltip": (
                        "OTR_VideoDirector policy. Used ONLY to detect 3D roles "
                        "for the granularity lock; opaque otherwise."
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
    def direct(self, announcer_image_model, music_image_model,
               other_beats_image_model, announcer_granularity,
               music_granularity, other_beats_granularity, fresh_cap,
               seed_mode, request_seed, custom_models_json="{}",
               video_policy_json="{}", gate_in=""):
        warnings: list = []
        custom = self._parse_json_obj(custom_models_json, "custom_models_json", warnings)
        video_policy = self._parse_json_obj(video_policy_json, "video_policy_json", warnings)
        descriptors = _registry_descriptors()

        picks = {
            "announcer_image_model": announcer_image_model,
            "music_image_model": music_image_model,
            "other_beats_image_model": other_beats_image_model,
        }
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
