"""triposr -- the LOWER-TIER local 3D mesher (dark scaffold).

The lighter, license-clean sibling of ``mesh_stage``: a single still ->
**TripoSR** image-to-mesh reconstruction (~sub-second, 6-8 GB) -> cached GLB ->
the SAME pinned portable Blender turntable stage -> straight-alpha frame
directory. Where ``mesh_stage`` uses Hunyuan3D-2mv (Tencent community license,
``commercial_clean=False``, medium/8 GB), ``triposr`` uses **TripoSR (MIT)** so
it is ``commercial_clean=True`` AND fits the 8 GB tier -- exactly the recorded
"license-hedge mesher" the ``mesh_stage`` docstring points at (the cache key
already carries a mesher id/version, so the two never collide).

This is the LOWER-TIER 3D pick (operator directive 2026-06-18): MIT, ~7 GB,
fast. It is a STATIC mesher -- camera/object turntable motion only, NEVER
lip-sync (no ``audio_ref`` input; family ``image_to_video``, NOT
``character_3d`` -- the family token is verified against ``schemas.FAMILIES``).
Talking 3D characters remain the parked toolkit lane (triposg_talk et al.); the
character path stays HuMo-2D.

DEFAULT-OFF / dark (empty ``default_roles`` + gated behind ``OTR_ENABLE_TRIPOSR``)
so it shows in the static per-role dropdown (V-6) but is never a default and
FAILS CLOSED until the operator (1) installs a TripoSR ComfyUI node + the MIT
weights, (2) enables the flag, and (3) GPU-validates it. It is deliberately
NOT in ``registry.VALIDATED_ENGINES`` -- the tested-only dropdown gate hides it
until that validation lands (add ``"triposr"`` to ``VALIDATED_ENGINES`` then).

VERIFY-AT-BUILD: the live render graph (the TripoSR node class + widget names)
must be captured from a running ``/object_info`` before the forward is wired --
the project's probe-first discipline (the wan_ti2v / LTX pattern). Until then
``load`` / ``prepare`` / ``render_clip`` raise NAMED errors; the fallback chain
``triposr -> still_parallax`` (matches ``mesh_stage``) keeps an episode rendering.

Config (env): ``OTR_ENABLE_TRIPOSR`` opt-in flag; ``OTR_TRIPOSR_WEIGHTS_DIR``
the SHA-pinned local weights folder (local-files-only, never a render-time
download); ``OTR_BLENDER_EXE`` the SAME pinned portable Blender that
``mesh_stage`` uses for the turntable stage.

Cold-import clean (V-12): module scope imports only stdlib + the dep-free
registry. No torch / diffusers / comfy at module scope. UTF-8, no BOM, ASCII.
"""
from __future__ import annotations

import os

from .registry import EngineUnusable, EngineUsabilityReason

#: Mesher identity (mirrors mesh_stage.MESHER_ID/_VERSION -- the cache key carries
#: this so a TripoSR GLB never collides with a hy3d2mv GLB for the same portrait).
MESHER_ID = "triposr"
MESHER_VERSION = "1"

#: 3D sub-ceiling (documented; enforced in the live render_clip, not the scaffold).
_VRAM_CEILING_MB_3D = 14_000


def _assert_usable_triposr(name):
    """Fail-closed guard for the lower-tier TripoSR mesher (flag -> weights ->
    Blender). Raises a NAMED ``EngineUnusable`` at the first missing
    prerequisite so the operator sees EXACTLY what is absent. Re-runs on EVERY
    attempt (no cached verdict)."""
    # 1. opt-in flag
    if os.getenv("OTR_ENABLE_TRIPOSR", "0") != "1":
        raise EngineUnusable(
            name, "character_video", EngineUsabilityReason.GATED_BY_FLAG,
            "triposr is opt-in (the lower-tier MIT 3D mesher); set "
            "OTR_ENABLE_TRIPOSR=1 ONLY after a TripoSR ComfyUI node + MIT "
            "weights are installed AND the forward is GPU-validated",
            kind="video",
        )
    # 2. TripoSR weights (local_files_only -- never a download at render time)
    weights = os.environ.get("OTR_TRIPOSR_WEIGHTS_DIR", "")
    if not weights or not os.path.isdir(weights):
        raise EngineUnusable(
            name, "character_video", EngineUsabilityReason.MISSING_MODEL,
            "TripoSR weights dir missing (set OTR_TRIPOSR_WEIGHTS_DIR to the "
            "SHA-256-pinned local MIT weights folder; fetched in a separate "
            "acquisition step, never at render time)",
            kind="video",
        )
    # 3. Blender executable (the shared portable turntable stage; never bpy pip)
    blender = os.environ.get("OTR_BLENDER_EXE", "")
    if not blender or not os.path.exists(blender):
        raise EngineUnusable(
            name, "character_video", EngineUsabilityReason.MISSING_MODEL,
            "Blender executable missing (set OTR_BLENDER_EXE to the standalone "
            "Blender binary -- the SAME portable Blender mesh_stage uses for the "
            "turntable stage)",
            kind="video",
        )


class TripoSREngine:
    """TripoSR (MIT) image->mesh -> Blender turntable stage (dark scaffold).

    The LOWER-TIER 3D mesher (operator directive 2026-06-18). MIT license ->
    ``commercial_clean=True``; medium VRAM class (~7 GB, the 8 GB tier).
    STATIC mesher (family ``image_to_video``; turntable/camera motion only,
    NEVER lip-sync). DEFAULT-OFF / dark until the operator installs the node +
    weights and GPU-validates the forward (then promote into
    ``registry.VALIDATED_ENGINES`` so it lists in the tested-only dropdown).
    """

    name = "triposr"
    # STATIC mesher chain (init still -> mesh -> Blender turntable frames);
    # verified against schemas.FAMILIES at build, NOT character_3d (no audio).
    family = "image_to_video"
    #: Still-aspect identity (2026-06-17): triposr meshes a static PROP/SCENE (not
    #: a character), so it takes a WIDE 16:9 init still like the other non-HuMo
    #: engines. It is NOT a requires_mesh_portrait character mesher (those are the
    #: eng_character_3d talkers), so it does NOT get the portrait still-feed.
    render_aspect = "wide"
    # Mirrors mesh_stage: a static mesher can fill the music/announcer/character
    # visual slots (it needs only an init_image, supplied by every role).
    roles = ("music_visual", "announcer_visual", "character_video")
    default_roles = ()
    required_inputs = ("init_image",)
    commercial_clean = True            # TripoSR: MIT license
    requires_flag = "OTR_ENABLE_TRIPOSR"
    engine_version = "1"
    fallback_engine = None               # NO FALLBACKS (2026-07-02): fail LOUD
    #: 3D sub-ceiling (documented; enforced in the live render_clip).
    vram_ceiling_mb = _VRAM_CEILING_MB_3D

    def assert_usable(self, host_caps, profile, request_template=None):
        _assert_usable_triposr(self.name)
        return self.name

    def load(self):
        raise RuntimeError(
            "triposr is a dark scaffold (2026-06-18): the live forward is "
            "VERIFY-AT-BUILD -- capture the TripoSR node class + widget names "
            "from a running /object_info, install the MIT weights, then wire "
            "the mesh->Blender-stage forward. Set OTR_ENABLE_TRIPOSR=1 ONLY "
            "after that GPU validation."
        )

    def unload(self):
        pass

    def prepare(self, host_caps, profile, session_ctx):
        raise NotImplementedError(
            "triposr.prepare: dark scaffold -- the live TripoSR mesh-once "
            "forward is not yet implemented (VERIFY-AT-BUILD pending the node "
            "+ weights install)"
        )

    def render_clip(self, request, prepared):
        raise NotImplementedError(
            "triposr.render_clip: dark scaffold -- not yet implemented; "
            "fails LOUD (NO FALLBACKS, 2026-07-02)"
        )

    def canonicalize(self, raw, request, profile):
        raise NotImplementedError("triposr.canonicalize: dark scaffold")

    def teardown(self, prepared):
        pass


__all__ = ["TripoSREngine", "MESHER_ID", "MESHER_VERSION"]
