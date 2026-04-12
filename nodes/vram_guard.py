"""
VRAMGuard — Pre-load watermark check.

Called immediately before every SD3.5, Flux, and LTX model load.
If current VRAM exceeds the watermark, raises VRAMWatermarkExceeded
so the pipeline can fall back to a static clip instead of OOMing
mid-render (which is unrecoverable).
"""

import logging

import torch

log = logging.getLogger("OTR")


class VRAMWatermarkExceeded(RuntimeError):
    """Raised when VRAM exceeds the pre-load watermark threshold."""
    pass


def vram_guard(threshold_gb: float, label: str) -> None:
    """Check VRAM against threshold. Raise if exceeded.

    Args:
        threshold_gb: Maximum allowed VRAM in GB before loading a model.
        label: Context string for the error message.

    Raises:
        VRAMWatermarkExceeded: If current allocation exceeds threshold.
    """
    if not torch.cuda.is_available():
        return

    allocated_gb = torch.cuda.memory_allocated() / 1e9

    if allocated_gb > threshold_gb:
        msg = f"{label}: {allocated_gb:.2f} GB > {threshold_gb} GB"
        log.error("[VRAMGuard] %s", msg)
        raise VRAMWatermarkExceeded(msg)

    log.info(
        "[VRAMGuard] %s: %.2f GB / %.1f GB ceiling - OK",
        label, allocated_gb, threshold_gb
    )
