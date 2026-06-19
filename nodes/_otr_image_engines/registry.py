"""Pluggable image-engine registry -- the model-agnostic image superstructure (C1).

Each image *role* -- announcer image (A), music image (B), other-beats/character
image (C) -- picks its engine from this shared registry instead of being
hardcoded to Flux. Adapters self-register on import; ``OTR_ImageDirector`` builds
its per-role dropdown from the FULL static registry (V-6) and filters by role
compatibility at execute time (``nodes/_otr_shared/role_compat.py``);
``OTR_ImageGenDispatcher`` calls :func:`assert_usable` to fail closed on an
incompatible / disabled pick (NO silent Flux fallback).

This mirrors the SHIPPED audio registry pattern (AS-4) via the dependency-free
:mod:`nodes._otr_shared.engine_registry_base`. The image protocol is a REDUCED
``prompt -> image`` contract -- it has NO ``canonicalize`` (that is a video clip
concept); a reduced-capability adapter is exactly what the base registry was
designed to hold. One pattern, three parallel namespaces (audio frozen; video;
image here).

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
    "ImageEngine",
    "EngineCore",
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
class ImageEngine(Protocol):
    """Contract every image-engine adapter implements.

    A structural SUPERSET of the shipped ``AudioEngine(Protocol)`` core (see
    :class:`~nodes._otr_shared.engine_registry_base.EngineCore`) -- anchored on
    that proven shape (AS-4), NOT a fresh divergent protocol. The registry only
    reads the CORE members (``name`` / ``roles`` / ``default_roles`` /
    ``commercial_clean`` / ``requires_flag`` / ``load`` / ``unload``); the
    ``prompt -> image`` lifecycle below is the per-image contract the dispatcher
    walks. Adapters duck-type -- inheritance is never required.

    ``required_inputs`` declares the request-level inputs the engine needs
    (tokens shared with ``role_compat``: an image engine is ``text_prompt`` and
    optionally ``init_image`` for an edit/img2img variant). There is
    deliberately NO ``canonicalize`` (a video-clip concern) -- the image adapter
    is the reduced ``prompt -> .png`` set the seam contract (AS-4) specifies.
    """

    # --- registry-facing core (mirrors AudioEngine(Protocol)) ---
    name: str
    roles: tuple
    default_roles: tuple
    commercial_clean: bool
    requires_flag: Optional[str]

    def load(self) -> None: ...
    def unload(self) -> None: ...

    # --- image-specific identity ---
    required_inputs: tuple

    # --- prompt -> image lifecycle (adapters implement; not called by registry) ---
    def assert_usable(self, host_caps, profile, request_template=None): ...
    def prepare(self, host_caps, profile, session_ctx): ...
    def render_image(self, request, prepared): ...
    def teardown(self, prepared) -> None: ...


# One registry instance for the image namespace (its own dict; no audio/video
# cross-pollution). Module-level functions bind to it so the public API matches
# the shipped audio + video registry function surface 1:1 (AS-4 "one pattern").
_IMAGE_REGISTRY = EngineRegistry("image")

register = _IMAGE_REGISTRY.register
get_engine = _IMAGE_REGISTRY.get_engine
is_registered = _IMAGE_REGISTRY.is_registered
all_engine_names = _IMAGE_REGISTRY.all_engine_names
engines_for_role = _IMAGE_REGISTRY.engines_for_role
default_engine_for_role = _IMAGE_REGISTRY.default_engine_for_role
assert_usable = _IMAGE_REGISTRY.assert_usable


# ---------------------------------------------------------------------------
# GATE B S1 -- per-engine capability DECLARATIONS (the registry TABLE, not the
# adapters). Consumed by nodes/_otr_shared/capability_profiles.py to DERIVE
# the per-profile enable-set -- never hand-listed per profile. A new engine
# ships its own row here; zero profile edits. vram_estimate_mb values are
# DRAFT estimates pending operator probe runs.
# ---------------------------------------------------------------------------
CAPABILITIES = {
    "flux_gen1": {"vram_class": "heavy", "vram_estimate_mb": 12000, "required_toolchain": None,
                  "requires_sidecar": False, "cpu_ok": False,
                  "model_requirements": ["flux.1-dev"]},
    "flux2_klein": {"vram_class": "medium", "vram_estimate_mb": 8000, "required_toolchain": None,
                    "requires_sidecar": False, "cpu_ok": False,
                    "model_requirements": ["flux.2-klein"]},
    "hidream_i1": {"vram_class": "heavy", "vram_estimate_mb": 14000, "required_toolchain": None,
                   "requires_sidecar": False, "cpu_ok": False,
                   "model_requirements": ["hidream-i1"]},
    # MEASURED 2026-06-18 on the 5080: the lumina2 split-file recipe stages the
    # Gemma-2 TE (4986 MB) then the 2.6B diffusion (4977 MB) sequentially; the
    # steady diffusion+VAE residency is ~7 GB (TE offloads before sampling) and
    # the render-window resident peak read ~12.2 GB, well under the 14.5 GB
    # ceiling. The engine's render_image reclaims after decode (single-resident).
    "lumina_image": {"vram_class": "medium", "vram_estimate_mb": 7000, "required_toolchain": None,
                     "requires_sidecar": False, "cpu_ok": False,
                     "model_requirements": ["lumina-image-2"]},
    "qwen_image": {"vram_class": "heavy", "vram_estimate_mb": 14000, "required_toolchain": None,
                   "requires_sidecar": False, "cpu_ok": False,
                   "model_requirements": ["qwen-image"]},
    "sd35_large": {"vram_class": "heavy", "vram_estimate_mb": 12000, "required_toolchain": None,
                   "requires_sidecar": False, "cpu_ok": False,
                   "model_requirements": ["sd3.5-large"]},
    # MEASURED 2026-06-18 on the 5080 (nvfp4 + qwen3-4b fp8 TE): the nvfp4
    # diffusion steady residency is ~4.3-5 GB (TE offloaded before sampling); the
    # transient TE+diffusion LOAD peak hit ~10 GB but ComfyUI manages it down. The
    # policy estimate is the steady residency, which fits the 8GB tier budget.
    "z_image_turbo": {"vram_class": "medium", "vram_estimate_mb": 5000, "required_toolchain": None,
                      "requires_sidecar": False, "cpu_ok": False,
                      "model_requirements": ["z-image-turbo"]},
}
__all__.append("CAPABILITIES")


# ---------------------------------------------------------------------------
# TESTED-ONLY DROPDOWN GATE (2026-06-17 operator directive). The per-role image
# COMBO in OTR_ImageDirector lists ONLY engines VALIDATED in production -- the
# untested peers (hidream_i1, qwen_image, sd35_large) are HIDDEN from the
# dropdown. Display gate only: every engine stays REGISTERED (V-6 --
# all_engine_names() unchanged), and the "+ Add Custom Model" sentinel remains
# the escape hatch. Mirrors the video registry's VALIDATED_ENGINES (a separate
# frozenset, NOT a CAPABILITIES key, because validate_declaration rejects unknown
# CAPABILITIES keys).
#
# Validated IMAGE engines: flux_gen1 (production default, wired x3 in
# otr_scifi_16gb_full.json), z_image_turbo + flux2_klein (2026-06-18), and
# lumina_image (2026-06-18, the lightweight Apache-2.0 low-VRAM peer).
# ---------------------------------------------------------------------------
VALIDATED_ENGINES = frozenset({
    "flux_gen1",
    # z_image_turbo: GPU-VERIFIED 2026-06-18 on the RTX 5080 (nvfp4 diffusion +
    # qwen3-4b fp8 TE + ae VAE; clean on-prompt filmic still at ~10 GB, 8 steps,
    # 12 s). The low-VRAM commercial-clean (Apache-2.0) image option.
    "z_image_turbo",
    # flux2_klein: GPU-VERIFIED 2026-06-18 on the RTX 5080 (FLUX.2 [klein] 4B GGUF
    # Q4_K_M + the klein-MATCHED qwen_3_4b TE [7680-wide; flux2-dev's Mistral was
    # 15360 -> the mat1/mat2 error] + flux2-vae). In a full flux2->silent-LTX episode
    # the dispatcher minted real stills end-to-end: "[OTR.image.flux2_klein] minted
    # still 832x480 ... steps=20 guidance=4.00", TE staged 7671 MB, no dim error.
    # Apache-2.0 (commercial-clean). Stays opt-in via OTR_ENABLE_FLUX2_KLEIN.
    "flux2_klein",
    # lumina_image: GPU-VERIFIED 2026-06-18 on the RTX 5080 (Lumina-Image 2.0,
    # the native split-file flow recipe: UNETLoader lumina_2_model_bf16 +
    # CLIPLoader[type=lumina2] gemma_2_2b TE + VAELoader lumina2_ae ->
    # ModelSamplingAuraFlow -> KSampler -> VAEDecode). The exact engine recipe
    # minted a real 1216x832 still in 23.5 s (model_type FLOW; TE 4986 MB +
    # diffusion 4977 MB staged sequentially; resident peak ~12.2 GB << 14.5 GB
    # ceiling -- and the engine's render_image reclaims after decode). Apache-2.0
    # (commercial-clean). The lightweight low-VRAM peer; opt-in via
    # OTR_ENABLE_LUMINA. Smoke: scripts/_otr_lumina_image_smoke.py.
    "lumina_image",
})
__all__.append("VALIDATED_ENGINES")


def validated_engine_names() -> list:
    """Registered image engines that are VALIDATED, sorted.

    The intersection of the live registry with :data:`VALIDATED_ENGINES` -- the
    tested-only source for the OTR_ImageDirector per-role dropdowns.
    """
    return sorted(set(_IMAGE_REGISTRY.all_engine_names()) & VALIDATED_ENGINES)


__all__.append("validated_engine_names")
