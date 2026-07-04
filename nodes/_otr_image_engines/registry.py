"""Pluggable image-engine registry -- the model-agnostic image superstructure (C1).

Each image *role* -- announcer image (A), music image (B), character
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
# ships its own row here; zero profile edits.
# ---------------------------------------------------------------------------
CAPABILITIES = {
    "flux_gen1": {"required_toolchain": None,
                  "requires_sidecar": False, "cpu_ok": False,
                  "model_requirements": ["flux.1-dev"]},
    "flux2_klein": {"required_toolchain": None,
                    "requires_sidecar": False, "cpu_ok": False,
                    "model_requirements": ["flux.2-klein"]},
    # hidream_i1 CAPABILITIES row REMOVED 2026-06-29 (C3): the dark scaffold
    # (NotImplementedError render) is unregistered; the registry-consistency
    # invariant forbids a row without a registered engine.
    # MEASURED 2026-06-18 on the 5080: the lumina2 split-file recipe stages the
    # Gemma-2 TE (4986 MB) then the 2.6B diffusion (4977 MB) sequentially; the
    # steady diffusion+VAE residency is ~7 GB (TE offloads before sampling) and
    # the render-window resident peak read ~12.2 GB, well under the 14.5 GB
    # ceiling. The engine's render_image reclaims after decode (single-resident).
    "lumina_image": {"required_toolchain": None,
                     "requires_sidecar": False, "cpu_ok": False,
                     "model_requirements": ["lumina-image-2"]},
    "qwen_image": {"required_toolchain": None,
                   "requires_sidecar": False, "cpu_ok": False,
                   "model_requirements": ["qwen-image"]},
    # sd35_large CAPABILITIES row REMOVED 2026-06-29 (C3): the dark scaffold
    # (NotImplementedError render) is unregistered; the registry-consistency
    # invariant forbids a row without a registered engine.
    # MEASURED 2026-06-18 on the 5080 (nvfp4 + qwen3-4b fp8 TE): the nvfp4
    # diffusion steady residency is ~4.3-5 GB (TE offloaded before sampling); the
    # transient TE+diffusion LOAD peak hit ~10 GB but ComfyUI manages it down.
    "z_image_turbo": {"required_toolchain": None,
                      "requires_sidecar": False, "cpu_ok": False,
                      "model_requirements": ["z-image-turbo"]},
    # Cloud partner STILLS (S1 stills lane 2026-07-03): no local weights, no
    # VRAM, CPU-side (the provider does the compute; canonicalize_image is a
    # pure PIL op). The registry-consistency invariant (test_capability_profiles
    # :217) requires ONE row per registered engine and vice versa.
    "cloud_recraft": {"required_toolchain": None,
                      "requires_sidecar": False, "cpu_ok": True, "model_requirements": []},
    "cloud_flux_pro": {"required_toolchain": None,
                       "requires_sidecar": False, "cpu_ok": True, "model_requirements": []},
    "cloud_nano_banana_2": {"required_toolchain": None,
                            "requires_sidecar": False, "cpu_ok": True, "model_requirements": []},
    "cloud_seedream_2": {"required_toolchain": None,
                         "requires_sidecar": False, "cpu_ok": True, "model_requirements": []},
    # `ideo` -- plain cloud Ideogram scene-still (S1+1). node_key cloud_ideogram_v4.
    "ideo": {"required_toolchain": None,
             "requires_sidecar": False, "cpu_ok": True, "model_requirements": []},
}
__all__.append("CAPABILITIES")


# ---------------------------------------------------------------------------
# VALIDATED_ENGINES + validated_engine_names() REMOVED 2026-06-29 (C4 -- "registry
# IS the menu"): there is NO validated-subset dropdown filter. Every REGISTERED
# image engine is SELECTABLE; the per-role director COMBO is built from
# all_engine_names() (validation is the operator's MANUAL process, never a code
# gate). The "+ Add Custom Model" sentinel remains the escape hatch.
# ---------------------------------------------------------------------------
