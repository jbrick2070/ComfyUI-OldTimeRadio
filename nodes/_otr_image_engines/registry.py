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
