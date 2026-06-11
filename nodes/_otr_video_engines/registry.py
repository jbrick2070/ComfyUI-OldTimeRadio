"""Pluggable video-engine registry -- the model-agnostic video superstructure.

Each video *role* -- announcer visual (A), music visual (B), other-beats visual
(C: character / scene b-roll / background) -- picks its engine from this shared
registry instead of being hardcoded to one model. Adapters self-register on
import; the ``OTR_VideoDirector`` builds its per-role dropdown from the FULL
static registry (V-6) and filters by role compatibility at execute time
(``nodes/_otr_shared/role_compat.py``); ``OTR_ShotLock`` calls
:func:`assert_usable` to fail closed on an incompatible pick.

This mirrors the SHIPPED audio registry pattern (AS-4) via the dependency-free
:mod:`nodes._otr_shared.engine_registry_base` -- it does NOT import the audio
package (which hard-imports ``torch``; that would break the video cold-import
invariant V-12). One pattern, three parallel namespaces (audio frozen; video
here; image in C1).

Cold-import-clean: this module imports only ``typing`` + the dep-free shared
base. No torch / transformers / diffusers at module scope.
"""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from .._otr_shared.engine_registry_base import (
    EngineCore,
    EngineRegistry,
    EngineUnusable,
    EngineUsabilityReason,
)

__all__ = [
    "VideoEngine",
    "EngineUnusable",
    "EngineUsabilityReason",
    "register",
    "get_engine",
    "is_registered",
    "all_engine_names",
    "engines_for_role",
    "default_engine_for_role",
    "assert_usable",
]


@runtime_checkable
class VideoEngine(Protocol):
    """Contract every video-engine adapter implements.

    A structural SUPERSET of the shipped ``AudioEngine(Protocol)`` core (see
    :class:`~nodes._otr_shared.engine_registry_base.EngineCore`) -- anchored on
    that proven shape (AS-4), NOT a fresh divergent protocol. The registry only
    reads the CORE members (``name`` / ``roles`` / ``default_roles`` /
    ``commercial_clean`` / ``requires_flag`` / ``load`` / ``unload``); the render
    lifecycle below is the per-clip contract that ``OTR_VideoRenderBatch`` walks
    (built out by the CW-4+ adapters). Adapters duck-type -- inheritance is never
    required.

    ``family`` is one of: ``audio_driven_face`` | ``lipsync_overlay`` |
    ``image_to_video`` | ``text_to_video`` | ``static_image_gen`` |
    ``static_motion`` | ``abstract`` (the 3D ``character_3d`` family ships with
    Subproject B). ``required_inputs`` declares the request-level inputs the
    engine needs (tokens shared with ``role_compat``: ``text_prompt`` /
    ``init_image`` / ``audio_ref`` / ``base_clip_ref``).

    Reduced-capability engines may set ``canonicalize = None`` (e.g. an
    image-gen adapter reused via the same pattern in C1): the registry never
    calls render-lifecycle methods, so optional ones may be absent or ``None``.
    """

    # --- registry-facing core (mirrors AudioEngine(Protocol)) ---
    name: str
    roles: tuple
    default_roles: tuple
    commercial_clean: bool
    requires_flag: Optional[str]

    def load(self) -> None: ...
    def unload(self) -> None: ...

    # --- video-specific identity ---
    family: str
    required_inputs: tuple

    # --- render lifecycle (CW-4+ adapters implement; not called by registry) ---
    def assert_usable(self, host_caps, profile, request_template=None): ...
    def prepare(self, host_caps, profile, session_ctx): ...
    def render_clip(self, request, prepared): ...
    def canonicalize(self, raw, request, profile): ...
    def teardown(self, prepared) -> None: ...


# One registry instance for the video namespace (its own dict; no audio
# cross-pollution). Module-level functions bind to it so the public API matches
# the shipped audio registry's function surface 1:1 (AS-4 "one pattern").
_VIDEO_REGISTRY = EngineRegistry("video")

register = _VIDEO_REGISTRY.register
get_engine = _VIDEO_REGISTRY.get_engine
is_registered = _VIDEO_REGISTRY.is_registered
all_engine_names = _VIDEO_REGISTRY.all_engine_names
engines_for_role = _VIDEO_REGISTRY.engines_for_role
default_engine_for_role = _VIDEO_REGISTRY.default_engine_for_role
assert_usable = _VIDEO_REGISTRY.assert_usable

# Re-export the protocol core under the video namespace for adapters that prefer
# an explicit base reference (duck-typing still works without it).
__all__.append("EngineCore")


# ---------------------------------------------------------------------------
# GATE B S1 -- per-engine capability DECLARATIONS (the registry TABLE, not the
# adapters). Consumed by nodes/_otr_shared/capability_profiles.py to DERIVE the
# per-profile enable-set -- never hand-listed per profile. A new engine ships
# its own row here; zero profile edits.
#
# Keys per row (validated by capability_profiles.validate_declaration):
#   vram_class          cpu | light | medium | heavy (GPU residency class)
#   vram_estimate_mb    DRAFT estimates pending operator probe runs (Lever-1
#                       register) -- policy-grade, not benchmark-grade.
#   required_toolchain  None, or "cu128_toolkit" (source builds; operator-
#                       blocked per the 3D plan -- keeps hunyuan/trellis dark).
#   requires_sidecar    True when the engine runs in an isolated sidecar venv.
#   cpu_ok              True when the engine can run with no GPU at all
#                       (procgen/CPU lanes; the cpu_floor tier filter).
#   model_requirements  informational model-asset ids for the S5 wizard.
# ---------------------------------------------------------------------------
CAPABILITIES = {
    "abstract": {"vram_class": "cpu", "vram_estimate_mb": 0, "required_toolchain": None,
                 "requires_sidecar": False, "cpu_ok": True, "model_requirements": []},
    "still_kenburns": {"vram_class": "cpu", "vram_estimate_mb": 0, "required_toolchain": None,
                       "requires_sidecar": False, "cpu_ok": True, "model_requirements": []},
    "station_card": {"vram_class": "cpu", "vram_estimate_mb": 0, "required_toolchain": None,
                     "requires_sidecar": False, "cpu_ok": True, "model_requirements": []},
    "visualizer": {"vram_class": "cpu", "vram_estimate_mb": 0, "required_toolchain": None,
                   "requires_sidecar": False, "cpu_ok": True, "model_requirements": []},
    "flux_still": {"vram_class": "heavy", "vram_estimate_mb": 12000, "required_toolchain": None,
                   "requires_sidecar": False, "cpu_ok": False,
                   "model_requirements": ["flux.1-dev"]},
    "humo": {"vram_class": "heavy", "vram_estimate_mb": 14000, "required_toolchain": None,
             "requires_sidecar": False, "cpu_ok": False,
             "model_requirements": ["HuMo-17B"]},
    "humo_1.7B": {"vram_class": "medium", "vram_estimate_mb": 7000, "required_toolchain": None,
                  "requires_sidecar": False, "cpu_ok": False,
                  "model_requirements": ["HuMo-1.7B"]},
    "latentsync": {"vram_class": "medium", "vram_estimate_mb": 6500, "required_toolchain": None,
                   "requires_sidecar": True, "cpu_ok": False,
                   "model_requirements": ["latentsync-1.5"]},
    "ltx_video": {"vram_class": "heavy", "vram_estimate_mb": 12500, "required_toolchain": None,
                  "requires_sidecar": False, "cpu_ok": False,
                  "model_requirements": ["ltx-video-2b"]},
    "wan_i2v": {"vram_class": "heavy", "vram_estimate_mb": 14000, "required_toolchain": None,
                "requires_sidecar": False, "cpu_ok": False,
                "model_requirements": ["wan2.1-i2v"]},
    "hunyuan3d_talk": {"vram_class": "heavy", "vram_estimate_mb": 14000,
                       "required_toolchain": "cu128_toolkit", "requires_sidecar": True,
                       "cpu_ok": False, "model_requirements": ["hunyuan3d-2"]},
    "trellis_talk": {"vram_class": "heavy", "vram_estimate_mb": 14000,
                     "required_toolchain": "cu128_toolkit", "requires_sidecar": True,
                     "cpu_ok": False, "model_requirements": ["trellis"]},
}
__all__.append("CAPABILITIES")
