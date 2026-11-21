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
    """Return accelerator memory this process has allocated, in GB, or 0.0
    when none can be read. CUDA first, unchanged; on Apple Silicon, the
    tensors Metal holds for this process (the ledger stamped 0.0 on a Mac)."""
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return float(torch.cuda.memory_allocated()) / 1024**3
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return float(torch.mps.current_allocated_memory()) / 1024**3
        return 0.0
    except Exception as exc:  # noqa: BLE001 -- telemetry must stay nonterminal
        log.debug(
            "[LFC:watchdog] VRAM allocation read failed: %s; returning 0.0",
            exc,
        )
        return 0.0
