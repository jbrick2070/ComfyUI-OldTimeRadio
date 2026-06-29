"""character_3d engine adapters -- dark scaffold (W7-pre migration slice).

Three 3D talking-character adapters that share the ``character_3d`` family:

* ``TripoSGTalkEngine`` (name ``triposg_talk``) -- the v1 NO-COMPILE lane
  (3D plan section 4: TripoSG geometry -> template shrinkwrap -> Rhubarb ->
  Blender alpha). MIT license, ``commercial_clean=True``. Gated behind
  ``OTR_ENABLE_TRIPOSG_TALK``; conditional on the S-3D-0 sidecar probe.

* ``Hunyuan3DTalkEngine`` (name ``hunyuan3d_talk``) -- Tencent HunyuanVideo-Talk
  mesh-to-talking-head pipeline; ``commercial_clean=False`` (verify-at-build).
  Gated behind ``OTR_ENABLE_CHARACTER_3D``. DEFERRED (cu128 TOOLKIT lane).

* ``TrellisTalkEngine`` (name ``trellis_talk``) -- Microsoft TRELLIS image-to-3D
  driven forward; MIT license, ``commercial_clean=True``.
  Gated behind ``OTR_ENABLE_TRELLIS_TALK``. DEFERRED (cu128 TOOLKIT lane).

All three are DEFAULT-OFF / dark (empty ``default_roles``): they appear in the
static per-role dropdown (V-6, imported unconditionally in ``__init__.py``) but
are never a default for any role and FAIL CLOSED until their gate clears
(triposg: S-3D-0 + T2b keystone; hunyuan/trellis: the cu128 toolkit lane).
All three declare ``requires_mesh_portrait = True`` -- the capability field the
ImageDirector granularity lock reads (3D plan section 3; never a hard-coded
family check).

Isolation: each runs in its OWN per-engine cu128 sidecar venv (NOT the ComfyUI
cu130 venv; the Path-B sidecar isolation pattern); the venv python paths
are ``OTR_TRIPOSG_SIDECAR_PYTHON`` / ``OTR_B_SIDECAR_PYTHON`` (hunyuan) /
``OTR_TRELLIS_SIDECAR_PYTHON`` (trellis).

``assert_usable`` check order (hunyuan/trellis -- UNCHANGED semantics):
  1. opt-in flag
  2. sidecar venv
  3. mesh directory (>=1 .obj/.glb file)
  4. ARKit-52 template .npz

``assert_usable`` check order (triposg_talk -- its OWN helper, 3D plan 7.1):
  1. opt-in flag
  2. sidecar venv (``OTR_TRIPOSG_SIDECAR_PYTHON``)
  3. TripoSG weights dir (``OTR_TRIPOSG_WEIGHTS_DIR``, local-files-only)
  4. ARKit-52 template .npz (+ ``OTR_B_ARKIT_TEMPLATE_HASH`` pin match when set)
  5. Rhubarb binary (``OTR_RHUBARB_EXE``)
  6. Blender executable (``OTR_BLENDER_EXE``)
NEVER the generated-mesh directory: ``OTR_B_MESH_DIR`` is probe_c EVIDENCE only
-- gating on it deadlocks ``prepare()``, which is what GENERATES meshes. Every
check re-runs on EVERY attempt (no cached "missing"). Each failure raises a
NAMED ``EngineUnusable(MISSING_MODEL, ...)`` with a distinguishing detail
string (no new enum value needed).

``load()`` raises NAMED RuntimeError (dark path -- no live forward yet).
``render_clip()`` raises NAMED NotImplementedError (dark -- the W7 forward).

Fallback chain: all three declare ``fallback_engine = "humo"`` so the chain
``triposg_talk -> humo -> humo_1.7B -> still_motion`` matches
the ``render_driver.SYNTH_FALLBACKS`` overlay WITHOUT requiring a soak re-cert
now. Trellis-first (chain A: ``trellis_talk -> humo``) defers with its lane.

VRAM ceiling (3D): 14 000 MB (``_VRAM_CEILING_MB_3D`` = 14_000) -- the
sub-ceiling that leaves headroom for the HuMo audio-driven fallback. The
ceiling is NOT enforced here (dark path); it is documented for Phase 5.

Cold-import clean (V-12): this module imports only stdlib + the dep-free
registry. No torch / diffusers / comfy at module scope. UTF-8, no BOM, ASCII.
"""
from __future__ import annotations

import os

from .registry import EngineUnusable, EngineUsabilityReason

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


def _arkit_template_check(name):
    """Shared ARKit-52 template gate: the .npz must exist; when the
    ``OTR_B_ARKIT_TEMPLATE_HASH`` pin is set the file's sha256 must MATCH it
    (3D plan 7.1 -- a stale/wrong template fails closed, never a bad wrap).
    Re-computed on every call (no cached verdict)."""
    arkit_npz = os.environ.get("OTR_B_ARKIT_TEMPLATE_NPZ", "")
    if not arkit_npz or not os.path.exists(arkit_npz):
        raise EngineUnusable(
            name, "character_3d", EngineUsabilityReason.MISSING_MODEL,
            "ARKit-52 template .npz missing (set OTR_B_ARKIT_TEMPLATE_NPZ to "
            "the converter-produced .npz; see the T1 template spike)",
            kind="video",
        )
    pin = os.environ.get("OTR_B_ARKIT_TEMPLATE_HASH", "").strip().lower()
    if pin:
        import hashlib
        h = hashlib.sha256()
        with open(arkit_npz, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        digest = h.hexdigest()
        if digest != pin:
            raise EngineUnusable(
                name, "character_3d", EngineUsabilityReason.MISSING_MODEL,
                "ARKit-52 template hash MISMATCH: sha256(%s)=%s but "
                "OTR_B_ARKIT_TEMPLATE_HASH pins %s (stale or wrong template; "
                "re-run the T1 converter or fix the pin)"
                % (os.path.basename(arkit_npz), digest[:16], pin[:16]),
                kind="video",
            )


def _assert_usable_triposg(name):
    """TripoSG's OWN fail-closed guard (3D plan 7.0/7.1) -- deliberately NOT
    the shared ``_assert_usable_3d`` (hunyuan/trellis keep their existing
    semantics + tests unchanged).

    Order: flag -> sidecar venv -> TripoSG weights -> ARKit template (+hash
    pin) -> Rhubarb binary -> Blender exe. NEVER the generated-mesh dir
    (``OTR_B_MESH_DIR`` is probe_c evidence only; gating on it deadlocks
    ``prepare()``). Every check re-runs on EVERY attempt."""
    # 1. opt-in flag
    if os.getenv("OTR_ENABLE_TRIPOSG_TALK", "0") != "1":
        raise EngineUnusable(
            name, "character_3d", EngineUsabilityReason.GATED_BY_FLAG,
            "triposg_talk is opt-in (S-3D-0 sidecar probe + T2b keystone "
            "gated); set OTR_ENABLE_TRIPOSG_TALK=1 ONLY after the S-3D-0 "
            "no-compile probe passes and the operator green-lights the lane",
            kind="video",
        )
    # 2. cu128 sidecar venv (prebuilt wheels only -- the no-compile lane)
    venv = os.environ.get("OTR_TRIPOSG_SIDECAR_PYTHON", "")
    if not venv or not os.path.exists(venv):
        raise EngineUnusable(
            name, "character_3d", EngineUsabilityReason.MISSING_MODEL,
            "triposg_talk sidecar venv not found at %r (set "
            "OTR_TRIPOSG_SIDECAR_PYTHON to the S-3D-0-built venv python "
            "executable; wheels-only, no source builds)" % venv,
            kind="video",
        )
    # 3. TripoSG weights (local_files_only -- never a download at render time)
    weights = os.environ.get("OTR_TRIPOSG_WEIGHTS_DIR", "")
    if not weights or not os.path.isdir(weights):
        raise EngineUnusable(
            name, "character_3d", EngineUsabilityReason.MISSING_MODEL,
            "TripoSG weights dir missing (set OTR_TRIPOSG_WEIGHTS_DIR to the "
            "SHA-256-pinned local weights folder; weights are fetched in a "
            "separate acquisition step, never at render time)",
            kind="video",
        )
    # 4. ARKit-52 template .npz (+ optional pinned-hash match)
    _arkit_template_check(name)
    # 5. Rhubarb binary (CPU viseme driver)
    rhubarb = os.environ.get("OTR_RHUBARB_EXE", "")
    if not rhubarb or not os.path.exists(rhubarb):
        raise EngineUnusable(
            name, "character_3d", EngineUsabilityReason.MISSING_MODEL,
            "Rhubarb binary missing (set OTR_RHUBARB_EXE to the rhubarb "
            "executable path; MIT, CPU-only viseme driver)",
            kind="video",
        )
    # 6. Blender executable (standalone; never `pip install bpy`)
    blender = os.environ.get("OTR_BLENDER_EXE", "")
    if not blender or not os.path.exists(blender):
        raise EngineUnusable(
            name, "character_3d", EngineUsabilityReason.MISSING_MODEL,
            "Blender executable missing (set OTR_BLENDER_EXE to the "
            "standalone Blender binary, e.g. under C:\\ComfyUI-Models\\tools)",
            kind="video",
        )


# ---------------------------------------------------------------------------
# TripoSGTalk adapter (the v1 no-compile lane; W7-pre dark scaffold)
# ---------------------------------------------------------------------------

class TripoSGTalkEngine:
    """TripoSG geometry -> shrinkwrap -> Rhubarb -> Blender alpha (dark).

    The v1 NO-COMPILE character_3d lane (3D plan section 4): prebuilt cu128
    wheels + SDPA, no source builds, conditional on the S-3D-0 probe. MIT
    license -> ``commercial_clean=True``. DEFAULT-OFF / dark until the
    operator green-lights the lane (S-3D-0 + T2b keystone).
    """

    name = "triposg_talk"
    family = "character_3d"
    # Drives the talking-character role; a 3D talking head also covers the
    # announcer slot (where audio_ref + init_image are both available).
    roles = ("announcer_visual", "character_video")
    default_roles = ()
    required_inputs = ("audio_ref", "init_image")
    commercial_clean = True           # TripoSG: MIT license
    requires_flag = "OTR_ENABLE_TRIPOSG_TALK"
    engine_version = "1"
    #: The v1 chain: triposg_talk -> humo (matches render_driver.SYNTH_FALLBACKS).
    fallback_engine = "humo"
    #: 3D sub-ceiling (documented; enforced in the W7 render_clip).
    vram_ceiling_mb = _VRAM_CEILING_MB_3D
    #: 3D capability (plan section 3): consumes a clean front-facing MESH
    #: PORTRAIT; the ImageDirector granularity lock reads this field.
    requires_mesh_portrait = True
    #: Still-aspect identity (2026-06-17): a mesh-portrait talker needs a PORTRAIT
    #: still to feed the mesher even when the final video is 16:9 -- the one 3D
    #: exception to the "non-HuMo = wide" rule.
    render_aspect = "portrait"

    def assert_usable(self, host_caps, profile, request_template=None):
        _assert_usable_triposg(self.name)
        return self.name

    def load(self):
        raise RuntimeError(
            "triposg_talk is a dark scaffold (W7-pre); the live forward is "
            "W7/B1 (requires the S-3D-0 no-compile sidecar probe PASS + the "
            "T2b keystone GO). Set OTR_ENABLE_TRIPOSG_TALK=1 ONLY after that."
        )

    def unload(self):
        pass

    def prepare(self, host_caps, profile, session_ctx):
        raise NotImplementedError(
            "triposg_talk.prepare: dark scaffold -- the W7 live forward "
            "(mesh-once per character) is not yet implemented"
        )

    def render_clip(self, request, prepared):
        raise NotImplementedError(
            "triposg_talk.render_clip: dark scaffold -- the W7 live forward "
            "is not yet implemented; fallback_engine='humo' handles the chain "
            "in assert_usable / render_driver fallback resolution"
        )

    def canonicalize(self, raw, request, profile):
        raise NotImplementedError(
            "triposg_talk.canonicalize: dark scaffold"
        )

    def teardown(self, prepared):
        pass


# ---------------------------------------------------------------------------
# Hunyuan3DTalk adapter
# ---------------------------------------------------------------------------

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
    #: 3D capability (plan section 3): consumes a mesh portrait.
    requires_mesh_portrait = True
    #: Still-aspect identity (2026-06-17): mesh-portrait talker -> portrait still.
    render_aspect = "portrait"

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
    #: 3D capability (plan section 3): consumes a mesh portrait.
    requires_mesh_portrait = True
    #: Still-aspect identity (2026-06-17): mesh-portrait talker -> portrait still.
    render_aspect = "portrait"

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


__all__ = ["TripoSGTalkEngine", "Hunyuan3DTalkEngine", "TrellisTalkEngine"]
