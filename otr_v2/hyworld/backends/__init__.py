"""
otr_v2.hyworld.backends  --  video stack backend registry
=========================================================

Day 1 harness for the 14-day video stack sprint.  Each backend is a
small class with a ``run(job_dir: Path) -> None`` method.  The worker
resolves a backend by name (from the ``OTR_HYWORLD_BACKEND`` env var,
plumbed through by ``bridge.py``) and hands the job directory over.

Contract is intentionally minimal so the existing HyworldBridge +
HyworldPoll + HyworldRenderer trio stays untouched: the backend writes
``STATUS.json`` and per-shot assets into ``io/hyworld_out/<job_id>/``
exactly like ``worker.run_stub`` does today.

Registry entries load lazily -- importing this package must never pull
torch, diffusers, or any GPU-heavy dependency.  Real backends (FLUX,
PuLID, LTX, Wan2.1, Florence-2) land on Days 2-7 of the sprint and
each will live in its own file, imported on demand.

No module-level side effects.  Safe to import from torch-free unit tests.
"""

from __future__ import annotations

from typing import Callable, Dict

# Registry: name -> zero-arg factory returning a backend instance.
# Factories import their module lazily so ``backends`` can be imported
# without dragging in the whole dependency graph.
_REGISTRY: Dict[str, Callable] = {}


def register(name: str, factory: Callable) -> None:
    """Register a backend factory under ``name`` (lower-case, dash-free)."""
    key = name.strip().lower()
    if not key:
        raise ValueError("backend name cannot be empty")
    _REGISTRY[key] = factory


def resolve(name: str):
    """Look up and instantiate a backend by name.  Raises KeyError if unknown."""
    key = (name or "").strip().lower()
    if key not in _REGISTRY:
        raise KeyError(
            f"unknown backend {name!r}; known: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[key]()


def list_backends() -> list[str]:
    """Return the registered backend names in sorted order."""
    return sorted(_REGISTRY)


# -- Day 1-7 registration.  Each factory defers its module import so
# ``backends`` stays torch-free at import time.  Day 2 added FLUX anchor;
# Day 3 added PuLID portraits; Day 4 added FLUX + ControlNet keyframes;
# Day 5 added LTX-2.3 motion (FLUX still -> LTX handoff);
# Day 6 added Wan2.1 1.3B I2V loops (FLUX still -> Wan2.1 handoff);
# Day 7 will add Florence-2 masks for SDXL inpaint compositing.

def _make_placeholder_test():
    from . import placeholder_test
    return placeholder_test.PlaceholderTestBackend()


def _make_flux_anchor():
    from . import flux_anchor
    return flux_anchor.FluxAnchorBackend()


def _make_pulid_portrait():
    from . import pulid_portrait
    return pulid_portrait.PulidPortraitBackend()


def _make_flux_keyframe():
    from . import flux_keyframe
    return flux_keyframe.FluxKeyframeBackend()


def _make_ltx_motion():
    from . import ltx_motion
    return ltx_motion.LtxMotionBackend()


def _make_wan21_loop():
    from . import wan21_loop
    return wan21_loop.Wan21LoopBackend()


register("placeholder_test", _make_placeholder_test)
register("flux_anchor", _make_flux_anchor)
register("pulid_portrait", _make_pulid_portrait)
register("flux_keyframe", _make_flux_keyframe)
register("ltx_motion", _make_ltx_motion)
register("wan21_loop", _make_wan21_loop)

__all__ = ["register", "resolve", "list_backends"]
