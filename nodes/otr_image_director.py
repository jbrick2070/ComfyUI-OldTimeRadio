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

The 3D granularity LOCK is GONE (lean-mean order 4, 2026-08-23). It hard-locked
any slot whose paired video engine declared ``requires_mesh_portrait`` to
``per_object`` -- and its own comment admitted it was DORMANT: the only
declarers (triposg_talk / hunyuan3d_talk / trellis_talk) were unregistered on
2026-06-29 and are now retired outright (files deleted, ids in
RETIRED_ENGINE_IDS), so it returned an empty set on every real run. The
capability field itself is removed from the video schemas in the same change,
so nothing can silently re-arm it. Its unregistered-engine rejection half was
not lost -- it MOVED upstream to OTR_VideoDirector's registry-membership
boundary (order 3), which fails a stale/unknown id seconds in, with a truthful
message. A 3D re-forward re-adds a capability + a lock DELIBERATELY, with its
own arc; nothing here is a scaffold for one.

``video_policy_json`` is REQUIRED + fail-closed: the OTR_VideoDirector policy
must be WIRED provider-before-consumer; an empty or malformed policy raises.
(A hand-built policy string naming an unknown engine is no longer rejected
HERE -- the director boundary owns membership now, and an out-of-contract
graph that bypasses it fails at the render gate's assert_usable instead.)

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
    # rip-sfx-broll (2026-07-01): the character image slot is KEPT --
    # character stills ride it; the retired_role_a / retired_role_b
    # pairings died with those roles.
    "character_image_model": ("character_video",),
}
SEED_MODES = ("request_hash", "fixed")
#: Operator-confirmable default ceiling for FRESH (per_beat) generation; the hard
#: cap is the audio-derived beat budget applied in the dispatcher (PASS-IMG #3).
DEFAULT_FRESH_CAP = 15


# `_image_model_combo` was removed 2026-08-28: it built a dropdown this node
# no longer declares. Image-MODEL selection lives in ONE place, OTR_VideoDirector
# (operator 2026-06-18: "only in one place not two"), which has its own live
# copy; this one fed no widget and had no production caller.


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


# (lean-mean order 4, 2026-08-23) `_is_3d_engine` and `three_d_locked_slots`
# were here -- the dormant 3D granularity lock. See the module docstring for
# the retirement record; the mesh-FODDER routing below is a different, LIVE
# capability (requires_mesh_fodder, mesh_stage) and is untouched.


#: `_ROLE_TO_VIDEO_SLOT` was removed 2026-08-28. It aliased
#: `_otr_shared/role_slots.ROLE_TO_VIDEO_SLOT` "for any importer of this name"
#: and there was no such importer -- the dispatcher defines its own identical
#: alias. Import the shared map directly.


def _is_mesh_fodder_engine(engine_id: str) -> bool:
    """True if ``engine_id`` is a registered VIDEO engine declaring the
    ``requires_mesh_fodder`` capability (the 3D image-streams routing gate).

    TOLERANT by design: mesh-fodder routing is additive and opt-in, so an
    empty / unregistered / custom engine is simply NOT-fodder (False) -- it
    must never raise and block a normal episode. The capability is read off the
    registered adapter, never an engine-name/family check. (The stricter
    `_is_3d_engine` this used to contrast itself with was retired with the 3D
    family, lean-mean order 4.)"""
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
    # Honor render-time engine rewrites (force-map + radio-is-host redirect): this
    # image-phase fodder decision runs BEFORE render mutates shot engines. Same
    # seam _still_needed_for_role uses. Env unset/no-redirect -> unchanged.
    try:
        from .otr_image_gen_dispatcher import (  # type: ignore
            _effective_video_engine_for_role as _eff)
    except Exception:  # noqa: BLE001 -- never block on the override resolver
        _eff = None
    for role in _role_slots.ROLE_TO_VIDEO_SLOT:
        engine_id = _role_slots.engine_id_for_role(vm, role)
        if _eff is not None:
            engine_id = _eff(role, engine_id)
        if _is_mesh_fodder_engine(engine_id):
            roles.add(role)
    return sorted(roles)


# (lean-mean order 4, 2026-08-23) `enforce_3d_granularity_lock` was here.
# Its dispatcher-side twin (the locked_3d_slots HALT) is removed in the same
# change; the policy field they shared is no longer emitted.


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
                "announcer_granularity": (gran, {
                    "default": "per_object",
                    "tooltip": "How often announcer beats get a fresh still: "
                               "per_object = one reused image (cheapest), "
                               "per_beat = fresh per beat, capped by fresh_cap "
                               "and the audio beat budget.",
                }),
                "music_granularity": (gran, {
                    "default": "per_object",
                    "tooltip": "How often music-card beats get a fresh still: "
                               "per_object = one reused card, per_beat = fresh "
                               "per cue, same caps as announcer_granularity.",
                }),
                "character_granularity": (gran, {
                    "default": "per_object",
                    "tooltip": (
                        "per_object: one image reused per character/prop "
                        "(cheapest). per_beat: a fresh image per beat, capped "
                        "by fresh_cap and the audio-derived beat budget -- "
                        "more variety, more render time."
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
                "seed_mode": (list(SEED_MODES), {
                    "default": SEED_MODES[0],
                    "tooltip": "request_hash: seeds derive deterministically "
                               "from each request (same episode re-renders "
                               "byte-identical stills). fixed: every still "
                               "uses request_seed verbatim -- for A/B "
                               "comparisons.",
                }),
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
                        "provider before this consumer). Carries the frozen "
                        "route, device policy and per-role aspects; "
                        "empty/malformed FAILS CLOSED."
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
                # S5 platform-portability (2026-07-10): explicit dtype policy
                # widget (append-only; widget slot 7). Default = nv50
                # baseline; emitted in the v2 policy and enforced at the
                # adapter boundary (S4).
                "dtype_policy": (["fp8_ok", "no_fp8", "no_fp8_no_fp4"], {
                    "default": "fp8_ok",
                    "tooltip": "Dtype lanes allowed for local image engines "
                               "(fp8/fp4 artifacts are OFF on ROCm/MPS "
                               "tiers).",
                }),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    # ------------------------------------------------------------------ #
    def direct(self, announcer_granularity,
               music_granularity, character_granularity, fresh_cap,
               seed_mode, request_seed, video_policy_json="",
               custom_models_json="{}", gate_in="",
               dtype_policy="fp8_ok"):
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
                     "character_image_model"):
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
            "character_image_model": character_granularity,
        }
        cap = int(fresh_cap)
        if cap < 1:
            warnings.append(f"fresh_cap {cap} < 1; clamped to 1")
            cap = 1

        policy = {
            # S4 platform-portability (2026-07-10): version 2 adds the
            # explicit dtype policy (default = nv50 baseline; the S5
            # widget feeds it). device_policy forwards from the video
            # policy so the dispatcher sees ONE device truth.
            "policy_version": 2,
            "dtype_policy": str(dtype_policy or "fp8_ok"),
            "device_policy": str(
                (video_policy.get("device_policy")
                 if isinstance(video_policy, dict) else None) or "cuda"),
            "image_models": resolved,
            "granularity": granularity,
            # Per-role still aspect, forwarded from OTR_VideoDirector so MetaBrief
            # mints each character still to MATCH its selected video engine
            # (portrait vs 16:9) with one dropdown pick. {} -> portrait (legacy).
            "aspects": (video_policy.get("aspects")
                        if isinstance(video_policy.get("aspects"), dict) else {}),
            # Per-role TALKING flag (S4b 2026-07-02), forwarded so MetaBrief
            # mints FACE-FORWARD portraits for lip-sync lanes. {} -> legacy.
            "talking": (video_policy.get("talking")
                        if isinstance(video_policy.get("talking"), dict) else {}),
            # Per-role SELECTED video engine, forwarded so the image dispatcher can
            # SKIP a still whose video engine does not consume init_image (the
            # visualizer / abstract procedural floor). An all-procedural episode then
            # invokes NO image model -- accessible for users with no image/video
            # models. {} -> dispatch every still (legacy behaviour).
            "video_models": (video_policy.get("video_models")
                             if isinstance(video_policy.get("video_models"), dict)
                             else {}),
            # THE FROZEN ROUTE (2026-07-25, multi-clip coverage chunk 1b),
            # forwarded verbatim from OTR_VideoDirector. This payload is built
            # KEY BY KEY, so an upstream key is NOT forwarded automatically --
            # every field above needed its own line and so do these two. Node 89
            # (MetaBrief) and node 91 (the dispatcher) both hang off THIS node,
            # so this is the only path by which the image branch can learn the
            # route that node 90 (ShotLock) freezes on the other branch.
            # {} -> a pre-1b policy; consumers fall back to resolving the route
            # themselves exactly as they did before.
            "effective_video_models": (
                video_policy.get("effective_video_models")
                if isinstance(video_policy.get("effective_video_models"), dict)
                else {}),
            "routing_env_snapshot": (
                video_policy.get("routing_env_snapshot")
                if isinstance(video_policy.get("routing_env_snapshot"), dict)
                else {}),
            # (The character {clip_mode, pool_n} passthrough died with the
            # pooling rip, 2026-07-01 -- every beat is per-beat now.)
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
        """FAIL-CLOSED video-policy parse: the policy is a REQUIRED wired
        input; empty, malformed, non-object, or missing its ``video_models``
        dict RAISES. The rule outlived the 3D granularity lock that first
        motivated it -- a silently-ignored policy means this node's per-role
        engine join is guessing, which is the drift the fail-closed parse
        exists to stop."""
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(
                "OTR_ImageDirector: video_policy_json is EMPTY -- wire "
                "OTR_VideoDirector.video_policy_json into this input "
                "(provider before consumer); the per-role engine join cannot "
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
