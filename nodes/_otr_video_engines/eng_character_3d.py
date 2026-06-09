"""character_3d engine adapters -- dark scaffold (Phase 3 / B opt-in).

Two 3D talking-character adapters that share the ``character_3d`` family:

* ``Hunyuan3DTalkEngine`` (name ``hunyuan3d_talk``) -- Tencent HunyuanVideo-Talk
  mesh-to-talking-head pipeline; ``commercial_clean=False`` (verify-at-build).
  Gated behind ``OTR_ENABLE_CHARACTER_3D``.

* ``TrellisTalkEngine`` (name ``trellis_talk``) -- Microsoft TRELLIS image-to-3D
  driven forward; MIT license, ``commercial_clean=True``.
  Gated behind ``OTR_ENABLE_TRELLIS_TALK``.

Both are DEFAULT-OFF / dark (empty ``default_roles``): they appear in the static
per-role dropdown (V-6, imported unconditionally in ``__init__.py``) but are never
a default for any role and FAIL CLOSED until the Phase 5 GPU keystone is cleared
(real meshes + ARKit-52 template + cu128 toolchain + probe_c < 20% binding GO).

Isolation: both run in their OWN per-engine cu128 sidecar venv (NOT the ComfyUI
cu130 venv; mirrors ``eng_latentsync`` Path-B isolation); the venv python path is
``OTR_B_SIDECAR_PYTHON`` (``OTR_TRELLIS_SIDECAR_PYTHON`` for trellis). The
``assert_usable`` check order is:
  1. opt-in flag
  2. sidecar venv
  3. mesh directory (>=1 .obj/.glb file)
  4. ARKit-52 template .npz
Each failure raises a NAMED ``EngineUnusable(MISSING_MODEL, ...)`` with a
distinguishing detail string (no new enum value needed).

``load()`` raises NAMED RuntimeError (dark path -- no live forward yet).
``render_clip()`` raises NAMED NotImplementedError (dark -- Phase 5 forward).

Fallback chain (B): both declare ``fallback_engine = "humo"`` so the chain
``hunyuan3d_talk -> humo -> latentsync -> still_kenburns`` matches the
``render_driver.SYNTH_FALLBACKS`` overlay WITHOUT requiring a soak re-cert now.
Trellis-first (chain A: ``trellis_talk -> humo``) defers to Phase 5.

VRAM ceiling (3D): 14 000 MB (``_VRAM_CEILING_MB_3D`` = 14_000) -- the
sub-ceiling that leaves headroom for the HuMo audio-driven fallback. The
ceiling is NOT enforced here (dark path); it is documented for Phase 5.

Cold-import clean (V-12): this module imports only stdlib + the dep-free
registry. No torch / diffusers / comfy at module scope. UTF-8, no BOM, ASCII.
"""
from __future__ import annotations

import os

from .registry import EngineUnusable, EngineUsabilityReason, register

_THIS = os.path.abspath(__file__)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS)))
_COMFY_ROOT = os.path.dirname(os.path.dirname(_REPO_ROOT))

#: 3D sidecar VRAM sub-ceiling (NOT enforced in the dark scaffold; for Phase 5).
_VRAM_CEILING_MB_3D = 14_000

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _assert_usable_3d(name, flag_env, venv_env, flag_label, venv_label):
    """Shared fail-closed guard for both 3D adapters (flag -> venv -> mesh -> npz).

    Raises NAMED ``EngineUnusable(MISSING_MODEL, ...)`` at the first missing
    prerequisite so the operator sees EXACTLY what is absent.
    """
    # 1. opt-in flag
    if os.getenv(flag_env, "0") != "1":
        raise EngineUnusable(
            name, "character_3d", EngineUsabilityReason.GATED_BY_FLAG,
            "%s is opt-in (Phase 5 asset-gated); set %s=1 AND clear the "
            "Phase 5 keystone (real meshes + ARKit-52 template + cu128 "
            "toolchain + probe_c < 20%% binding GO) before enabling"
            % (flag_label, flag_env),
            kind="video",
        )
    # 2. cu128 sidecar venv
    venv = os.environ.get(venv_env, "")
    if not venv or not os.path.exists(venv):
        raise EngineUnusable(
            name, "character_3d", EngineUsabilityReason.MISSING_MODEL,
            "%s sidecar venv not found at %r (set %s to the venv python "
            "executable path, e.g. <OTR_CU128_HOME>/%s/.venv/Scripts/python.exe)"
            % (flag_label, venv, venv_env, venv_label),
            kind="video",
        )
    # 3. mesh directory
    mesh_dir = os.environ.get("OTR_B_MESH_DIR", "")
    if not mesh_dir or not os.path.isdir(mesh_dir):
        raise EngineUnusable(
            name, "character_3d", EngineUsabilityReason.MISSING_MODEL,
            "character_3d mesh directory missing (set OTR_B_MESH_DIR to a "
            "folder containing >=1 .obj/.glb mesh files; ~25 expected for a "
            "full episode cast)",
            kind="video",
        )
    meshes = [
        f for f in os.listdir(mesh_dir)
        if f.lower().endswith((".obj", ".glb"))
    ]
    if not meshes:
        raise EngineUnusable(
            name, "character_3d", EngineUsabilityReason.MISSING_MODEL,
            "character_3d mesh directory %r exists but contains no .obj/.glb "
            "files (drop real mesh files before enabling %s)"
            % (mesh_dir, flag_label),
            kind="video",
        )
    # 4. ARKit-52 template .npz
    arkit_npz = os.environ.get("OTR_B_ARKIT_TEMPLATE_NPZ", "")
    if not arkit_npz or not os.path.exists(arkit_npz):
        raise EngineUnusable(
            name, "character_3d", EngineUsabilityReason.MISSING_MODEL,
            "ARKit-52 template .npz missing (set OTR_B_ARKIT_TEMPLATE_NPZ to "
            "the .npz path; required by the Slot1->Wrap->Slot2 keystone before "
            "%s can render)" % flag_label,
            kind="video",
        )


# ---------------------------------------------------------------------------
# Hunyuan3DTalk adapter
# ---------------------------------------------------------------------------

@register
class Hunyuan3DTalkEngine:
    """Tencent HunyuanVideo-Talk 3D talking-head adapter (dark scaffold).

    DEFAULT-OFF / dark (Phase 5 GPU keystone required). ``commercial_clean``
    is False -- verify-at-build per the OTR license matrix.
    """

    name = "hunyuan3d_talk"
    family = "character_3d"
    # Drives the talking-character role; a 3D talking head also covers the
    # announcer slot (where audio_ref + init_image are both available).
    roles = ("announcer_visual", "character_video")
    default_roles = ()
    required_inputs = ("audio_ref", "init_image")
    commercial_clean = False          # Tencent HunyuanVideo-Talk -- verify-at-build
    requires_flag = "OTR_ENABLE_CHARACTER_3D"
    engine_version = "1"
    #: Fallback B: hunyuan3d_talk -> humo (matches render_driver.SYNTH_FALLBACKS).
    fallback_engine = "humo"
    #: 3D sub-ceiling (documented; enforced in Phase 5 render_clip).
    vram_ceiling_mb = _VRAM_CEILING_MB_3D

    def assert_usable(self, host_caps, profile, request_template=None):
        _assert_usable_3d(
            self.name,
            "OTR_ENABLE_CHARACTER_3D",
            "OTR_B_SIDECAR_PYTHON",
            "hunyuan3d_talk",
            "hunyuan3d_talk",
        )
        return self.name

    def load(self):
        raise RuntimeError(
            "hunyuan3d_talk is a dark scaffold (Phase 3); "
            "the live GPU forward is Phase 5 (requires Phase 5 keystone: "
            "real meshes + ARKit-52 template + probe_c < 20%% binding GO + "
            "cu128 toolchain). Set OTR_ENABLE_CHARACTER_3D=1 ONLY after that."
        )

    def unload(self):
        pass

    def prepare(self, host_caps, profile, session_ctx):
        raise NotImplementedError(
            "hunyuan3d_talk.prepare: dark scaffold -- Phase 5 live forward "
            "not yet implemented"
        )

    def render_clip(self, request, prepared):
        raise NotImplementedError(
            "hunyuan3d_talk.render_clip: dark scaffold -- Phase 5 live forward "
            "not yet implemented; fallback_engine='humo' handles the chain in "
            "assert_usable / render_driver fallback resolution"
        )

    def canonicalize(self, raw, request, profile):
        raise NotImplementedError(
            "hunyuan3d_talk.canonicalize: dark scaffold"
        )

    def teardown(self, prepared):
        pass


# ---------------------------------------------------------------------------
# TrellisTalk adapter
# ---------------------------------------------------------------------------

@register
class TrellisTalkEngine:
    """Microsoft TRELLIS image-to-3D driven adapter (dark scaffold).

    MIT license -> ``commercial_clean=True``. DEFAULT-OFF / dark (Phase 5
    keystone required; trellis-first chain A deferred to Phase 5).
    """

    name = "trellis_talk"
    family = "character_3d"
    roles = ("announcer_visual", "character_video")
    default_roles = ()
    required_inputs = ("audio_ref", "init_image")
    commercial_clean = True           # TRELLIS: MIT license
    requires_flag = "OTR_ENABLE_TRELLIS_TALK"
    engine_version = "1"
    #: Fallback B: trellis_talk -> humo (chain A deferred to Phase 5).
    fallback_engine = "humo"
    #: 3D sub-ceiling (documented; enforced in Phase 5 render_clip).
    vram_ceiling_mb = _VRAM_CEILING_MB_3D

    def assert_usable(self, host_caps, profile, request_template=None):
        _assert_usable_3d(
            self.name,
            "OTR_ENABLE_TRELLIS_TALK",
            "OTR_TRELLIS_SIDECAR_PYTHON",
            "trellis_talk",
            "trellis_talk",
        )
        return self.name

    def load(self):
        raise RuntimeError(
            "trellis_talk is a dark scaffold (Phase 3); "
            "the live GPU forward is Phase 5 (requires Phase 5 keystone: "
            "real meshes + ARKit-52 template + probe_c < 20%% binding GO + "
            "cu128 toolchain). Set OTR_ENABLE_TRELLIS_TALK=1 ONLY after that."
        )

    def unload(self):
        pass

    def prepare(self, host_caps, profile, session_ctx):
        raise NotImplementedError(
            "trellis_talk.prepare: dark scaffold -- Phase 5 live forward "
            "not yet implemented"
        )

    def render_clip(self, request, prepared):
        raise NotImplementedError(
            "trellis_talk.render_clip: dark scaffold -- Phase 5 live forward "
            "not yet implemented; fallback_engine='humo' handles the chain in "
            "assert_usable / render_driver fallback resolution"
        )

    def canonicalize(self, raw, request, profile):
        raise NotImplementedError(
            "trellis_talk.canonicalize: dark scaffold"
        )

    def teardown(self, prepared):
        pass


__all__ = ["Hunyuan3DTalkEngine", "TrellisTalkEngine"]
