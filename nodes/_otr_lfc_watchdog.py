"""LFC cascade-entry VRAM telemetry.

The cascade reads :func:`_torch_vram_allocated_gb` once at entry and stamps
the sample for diagnostics. It is telemetry only: no ceiling or prose-quality
policy lives here.
"""
from __future__ import annotations

import logging


log = logging.getLogger("OTR.lfc_watchdog")


__all__ = ["_torch_vram_allocated_gb"]


def _torch_vram_allocated_gb() -> float:
    """Return CUDA-allocated VRAM in GB, or 0.0 when unavailable."""
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            return 0.0
        return float(torch.cuda.memory_allocated()) / 1024**3
    except Exception as exc:  # noqa: BLE001 -- telemetry must stay nonterminal
        log.debug(
            "[LFC:watchdog] VRAM allocation read failed: %s; returning 0.0",
            exc,
        )
        return 0.0
