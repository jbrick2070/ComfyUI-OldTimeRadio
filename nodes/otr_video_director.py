"""OTR_VideoDirector -- the per-role model-selection UI (A-S1/W1).

Captures POLICY only (V-6): per-role video model + per-role image model, the
Other-Beats clip mode, canvas/fps, seed mode, fallback policy. It emits ONE
``video_policy_json`` STRING that ``OTR_ShotLock`` consumes -- an explicit string
socket (testable, no hidden coupling).

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

#: Sentinel COMBO entry that opens the "declare a custom model" path. When a role
#: is set to this, its real engine id is read from the ``custom_models_json``
#: widget (role_key -> engine_id). The custom adapter itself loads in CW-4+.
ADD_CUSTOM = "+ Add Custom Model"

#: Default image sources until the image registry lands (C1). Flux = "gen 1"
#: (the spine's default image source); swapping it never touches video models.
IMAGE_DEFAULTS = ("Flux (gen 1)",)

#: Which role(s) each video slot must be compatible with (fail-closed filter).
VIDEO_SLOT_ROLES = {
    "announcer_video_model": ("announcer_visual",),
    "music_video_model": ("music_visual",),
    "other_beats_video_model": (
        "character_video", "scene_broll", "background_abstract",
    ),
}
CLIP_MODES = ("unique_per_beat", "pool_n_loop")
SEED_MODES = ("request_hash", "fixed")


def _video_model_combo() -> list:
    """TESTED-ONLY video engines + the custom sentinel (2026-06-17 gate).

    Lists only GPU-VALIDATED engines (``registry.validated_engine_names()``) so
    the operator cannot pick an untested model from the UI. Every engine remains
    REGISTERED (V-6: ``all_engine_names()`` is untouched, so role_compat /
    assert_usable / the force-map experiment knob still see the full set); this
    narrows the *display* list only. ``+ Add Custom Model`` stays the escape hatch
    for an explicitly-declared engine. Falls back to the full registry only if the
    validated set is somehow empty, so a box-fresh graph still validates."""
    names = list(_vreg.validated_engine_names()) or list(_vreg.all_engine_names())
    return names + [ADD_CUSTOM]


def _image_model_combo() -> list:
    """Image-source COMBO sourced from the C1 image registry (V-6: full static
    list + the custom sentinel). 'Flux' is just gen 1 (engine ``flux_gen1``), no
    longer a hardcoded string. Falls back to ``IMAGE_DEFAULTS`` only if the image
    registry is somehow empty, so a box-fresh graph still validates."""
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
                "other_beats_video_model": (video, {
                    "tooltip": (
                        "Video model for all OTHER beats (role C: character / "
                        "scene b-roll / background)."
                    ),
                }),
                "announcer_image_model": (image, {
                    "tooltip": "Image source for the announcer (feeds its video).",
                }),
                "music_image_model": (image, {
                    "tooltip": "Image source for music beats.",
                }),
                "other_beats_image_model": (image, {
                    "tooltip": "Image source for other beats.",
                }),
                "other_beats_clip_mode": (list(CLIP_MODES), {
                    "default": CLIP_MODES[0],
                    "tooltip": (
                        "unique_per_beat: one clip per beat (real-time). "
                        "pool_n_loop: render N unique clips, tile/loop them "
                        "across the whole other-beats timeline (cheapest; N "
                        "TOTAL renders, capped to the audio duration)."
                    ),
                }),
                "other_beats_n": ("INT", {
                    "default": 8, "min": 1, "max": 256,
                    "tooltip": (
                        "Pool size N for pool_n_loop (clamped to what the "
                        "other-beats span actually uses; over-set -> warn)."
                    ),
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
                "allow_auto_fallback": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "episode_duration_target": ("STRING", {
                    "default": "auto",
                    "tooltip": (
                        "Target episode length mm:ss, or 'auto' to derive from "
                        "the frozen audio (the binding source of truth)."
                    ),
                }),
                "custom_models_json": ("STRING", {
                    "multiline": True,
                    "default": "{}",
                    "tooltip": (
                        "When a role is set to '+ Add Custom Model', map the "
                        "role key to a custom engine id here, e.g. "
                        '{"other_beats_video_model": "my_engine"}.'
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
               other_beats_video_model, announcer_image_model,
               music_image_model, other_beats_image_model,
               other_beats_clip_mode, other_beats_n, fps, canvas_w, canvas_h,
               seed_mode, request_seed, allow_auto_fallback,
               episode_duration_target="auto", custom_models_json="{}",
               gate_in=""):
        warnings: list = []
        custom = self._parse_custom(custom_models_json, warnings)
        descriptors = _registry_descriptors()

        video_models = {
            "announcer_video_model": announcer_video_model,
            "music_video_model": music_video_model,
            "other_beats_video_model": other_beats_video_model,
        }
        resolved_video = {}
        for slot, picked in video_models.items():
            resolved_video[slot] = self._resolve_and_validate(
                slot, picked, custom, descriptors, warnings
            )

        # Clamp N (warn) -- the hard cap against the audio duration is applied
        # later in OTR_ShotLock's audio-derived clip budget.
        n = int(other_beats_n)
        if n < 1:
            warnings.append(f"other_beats_n {n} < 1; clamped to 1")
            n = 1

        policy = {
            "policy_version": 1,
            "video_models": resolved_video,
            # Per-role still aspect, resolved from each slot's selected engine, so
            # the image node mints character stills to MATCH the chosen video
            # engine with ONE dropdown pick (portrait humo_1.7B vs wide
            # humo_1.7B_169). Opaque to everyone who does not size stills.
            "aspects": self._role_aspects(resolved_video),
            "image_models": {
                "announcer_image_model": announcer_image_model,
                "music_image_model": music_image_model,
                "other_beats_image_model": other_beats_image_model,
            },
            "other_beats": {"clip_mode": other_beats_clip_mode, "pool_n": n},
            "canvas": {"w": int(canvas_w), "h": int(canvas_h), "fps": int(fps)},
            "seed": {"mode": seed_mode, "request_seed": int(request_seed)},
            "allow_auto_fallback": bool(allow_auto_fallback),
            "episode_duration_target": str(episode_duration_target or "auto"),
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
        """Map each still-bearing role to the SELECTED engine's ``render_aspect``
        so character stills match their video engine: announcer_visual <-
        announcer_video_model, character_video <- other_beats_video_model.
        Unknown / custom / unresolved picks -> 'portrait' (the safe legacy look).
        Pure: a registry read, no side effects."""
        def _asp(slot):
            eid = (resolved_video.get(slot) or {}).get("engine_id") or ""
            try:
                eng = _vreg.get_engine(eid)
                if getattr(eng, "render_aspect", "portrait") == "wide":
                    return "wide"
            except Exception:  # noqa: BLE001 -- unknown engine -> portrait
                pass
            return "portrait"
        return {
            "announcer_visual": _asp("announcer_video_model"),
            "music_visual": _asp("music_video_model"),
            "character_video": _asp("other_beats_video_model"),
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
