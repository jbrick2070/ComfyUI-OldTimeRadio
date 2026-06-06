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
