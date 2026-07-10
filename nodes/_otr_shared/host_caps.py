"""Real host capability facts for adapter-level enforcement (S4).

docs/2026-07-09-platform-portability-final.md section 3: the adapter-level
``assert_usable(host_caps, profile, ...)`` protocol EXISTED on every video
and image adapter but received empty dicts (render_driver) or was never
called at all (image dispatcher) -- the campaign's most consequential
wiring catch. This module is the ONE builder both dispatch paths share.

Minimum shape (verify-at-build register): ``cuda_available``,
``mps_available``, ``vendor``, ``total_vram_gb``. Truthful on lying CUDA
stacks: a runtime that answers ``is_available()`` but cannot open device 0
reports ``cuda_available=False`` (same ruling as the S0 ``_detect_host``
fix). Stdlib-only at import; torch is probed lazily and failure to probe
NEVER raises (detection must not crash -- enforcement belongs to the
adapters consuming these facts).
"""
from __future__ import annotations

__all__ = ["build_host_caps"]


def build_host_caps() -> dict:
    caps = {
        "cuda_available": False,
        "mps_available": False,
        "vendor": "none",
        "total_vram_gb": 0.0,
    }
    try:
        import torch
        try:
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                props = torch.cuda.get_device_properties(0)
                caps["cuda_available"] = True
                caps["total_vram_gb"] = round(
                    props.total_memory / (1024 ** 3), 2)
                caps["vendor"] = ("amd"
                                  if getattr(torch.version, "hip", None)
                                  else "nvidia")
        except Exception:  # noqa: BLE001 -- unusable CUDA = no CUDA
            caps["cuda_available"] = False
            caps["total_vram_gb"] = 0.0
            caps["vendor"] = "none"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            caps["mps_available"] = True
            if caps["vendor"] == "none":
                caps["vendor"] = "apple"
    except Exception:  # noqa: BLE001 -- detection must never crash
        pass
    return caps
