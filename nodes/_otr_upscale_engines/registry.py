"""Pluggable upscale-engine registry -- the model-agnostic post-render upscale/enhance
superstructure (2026-08-08, queue item 8 / D-2 codicil rebuild).

A parallel namespace to ``nodes/_otr_video_engines`` and ``nodes/_otr_image_engines`` --
same proven pattern (AS-4), its OWN registry dict, zero cross-pollution with the other
namespaces. Every upscale engine (``off`` pass-through, ``spandrel_esrgan`` Real-ESRGAN,
future rows) is a pluggable adapter selected via one ``upscale_stage.engine`` profile
field per tier.

The registry replaces the retired ``nodes/rtx_upscale.py`` (RTX-VSR, NVIDIA-only) which
was ripped in the same commit as this package. The D-2 codicil pinned every design
constraint: registry rows with ``device_backends`` per engine (nvidia/amd/mps/cpu),
per-tier profile values, fail-loud on unsupported hardware, and the HONEST-SWITCH LAW
(widgets + working engine land together; ``off`` is a real registered engine).

Cold-import-clean: this module imports only ``typing`` + the dep-free shared base. No
torch / spandrel at module scope (adapters import lazily inside load / upscale_frames).
"""
from __future__ import annotations

from typing import Optional, Protocol, TYPE_CHECKING, runtime_checkable

if TYPE_CHECKING:
    import torch

from .._otr_shared.engine_registry_base import (
    EngineCore,
    EngineRegistry,
    EngineUnusable,
    EngineUsabilityReason,
    EngineNotRunnableError,
)

__all__ = [
    "UpscaleEngine",
    "EngineCore",
    "EngineUnusable",
    "EngineUsabilityReason",
    "EngineNotRunnableError",
    "register",
    "get_engine",
    "is_registered",
    "all_engine_names",
    "engines_for_role",
    "default_engine_for_role",
    "assert_usable",
    "audit_engine_roster",
    "CAPABILITIES",
    "RETIRED_UPSCALE_ENGINE_IDS",
]


@runtime_checkable
class UpscaleEngine(Protocol):
    """Contract every upscale-engine adapter implements.

    Structural SUPERSET of the shipped ``AudioEngine(Protocol)`` core (via
    :class:`~nodes._otr_shared.engine_registry_base.EngineCore`). Adapters
    duck-type -- inheritance is never required.

    ``upscale_frames`` takes a BHWC float32 tensor in [0, 1] on the engine's
    load-time device and returns a BHWC float32 tensor in [0, 1] on the same
    device, upscaled by ``intrinsic_scale``. The caller is responsible for
    reading ``self.device`` and moving inputs there before the call. spandrel
    enforces batch=1 at the descriptor layer, so batch-N adapters loop
    internally (see ``eng_spandrel_esrgan.SpandrelEsrgan.upscale_frames``).
    """

    # --- registry-facing core (mirrors AudioEngine(Protocol)) ---
    name: str
    roles: tuple                    # ALWAYS ("upscale_stage",)
    default_roles: tuple            # ("upscale_stage",) for off; () for others
    commercial_clean: bool
    requires_flag: Optional[str]    # ALWAYS None (registry IS the menu)

    # --- upscale-namespace identity ---
    device_backends: tuple          # subset of ("cuda", "cpu"); NO "mps" until proven
    requires_vendor: Optional[str]  # None | "nvidia" | "amd" | "apple"
    intrinsic_scale: int            # off=1; a 2x model=2; 4x=4
    device: Optional["torch.device"]  # SET BY load(); None until loaded

    def load(self, device: "torch.device") -> None: ...      # MUST set self.device
    def unload(self) -> None: ...                            # MUST clear self.device + release descriptor

    def upscale_frames(self, frames: "torch.Tensor") -> "torch.Tensor": ...


# One registry instance for the upscale namespace (its own dict; no cross-pollution).
_UPSCALE_REGISTRY = EngineRegistry("upscale")

register = _UPSCALE_REGISTRY.register
get_engine = _UPSCALE_REGISTRY.get_engine
is_registered = _UPSCALE_REGISTRY.is_registered
all_engine_names = _UPSCALE_REGISTRY.all_engine_names
engines_for_role = _UPSCALE_REGISTRY.engines_for_role
default_engine_for_role = _UPSCALE_REGISTRY.default_engine_for_role
assert_usable = _UPSCALE_REGISTRY.assert_usable


# ---------------------------------------------------------------------------
# Per-engine capability DECLARATIONS (the registry TABLE, mirroring
# _otr_video_engines/registry.py:CAPABILITIES). Consumed by
# nodes/_otr_shared/capability_profiles.py cross-validation. A new engine
# ships its own row here; zero per-profile edits.
#
# MPS is deliberately NOT in device_backends for either engine in the first
# ship: this Windows/NVIDIA box cannot produce a live MPS receipt, and the
# operator's HONEST-SWITCH LAW forbids advertising a capability without
# proof. Add "mps" when a Mac user provides an integration receipt.
# ---------------------------------------------------------------------------
CAPABILITIES = {
    "off": {
        "required_toolchain": None, "requires_sidecar": False,
        "device_backends": ["cuda", "cpu"], "requires_vendor": None,
        "needs_fp8_te": False, "needs_fp4_te": False,
        "practical_without_gpu": True, "sidecar_conditional": False,
        "model_requirements": []},
    "spandrel_esrgan": {
        "required_toolchain": None, "requires_sidecar": False,
        "device_backends": ["cuda", "cpu"], "requires_vendor": None,
        "needs_fp8_te": False, "needs_fp4_te": False,
        "practical_without_gpu": True, "sidecar_conditional": False,
        "model_requirements": ["real-esrgan-x2plus"]},
}


# The engine ids RETIRED. Empty at ship (nothing to retire yet). Follows the
# shipped video precedent at nodes/_otr_shared/public_engines.RETIRED_ENGINE_IDS
# so a future retirement has a clear home. IMMUTABLE: append here only when
# another engine is retired, never remove.
RETIRED_UPSCALE_ENGINE_IDS: frozenset = frozenset()


def audit_engine_roster() -> dict:
    """Compare the EXPECTED roster against what actually registered.

    Mirrors ``nodes/_otr_video_engines/registry.py:audit_engine_roster``.
    Every adapter import in ``_otr_upscale_engines/__init__.py`` is wrapped in
    ``try: ... except Exception: pass`` so a packaging quirk cannot break the
    namespace import. The cost is that a broken adapter fails silently: it
    never registers, vanishes from the dropdown, and nothing says so.
    :data:`CAPABILITIES` is the independent expected roster.

    Returns ``{"missing": (...), "unexpected": (...)}``.
    """
    expected = frozenset(CAPABILITIES)
    registered = frozenset(all_engine_names())
    return {
        "missing": tuple(sorted(expected - registered)),
        "unexpected": tuple(sorted(registered - expected)),
    }
