"""
MemoryBoundary — Explicit VRAM reclamation between model phases.

Dereference != VRAM release on Blackwell. This function forces the full
unload-flush-sync-sleep sequence that the driver actually needs.

Without it, loading LTX-Video after SD3.5 will OOM even when Python
shows zero live references to the old model.
"""

import json
import logging
import os
import time

import torch

log = logging.getLogger("OTR")

_V2_CATEGORY = "OldTimeRadio/v2.0 Visual Drama Engine"

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _runtime_log(msg):
    """Append a timestamped line to otr_runtime.log."""
    try:
        from datetime import datetime
        ts = datetime.now().strftime("%H:%M:%S")
        with open(os.path.join(_REPO_ROOT, "otr_runtime.log"), "a", encoding="utf-8") as f:
            f.write(f"[{ts}] [v2] {msg}\n")
    except Exception:
        pass


def memory_boundary(sleep_s: float, label: str) -> dict:
    """Full VRAM reclamation: unload, flush, sync, sleep, log.

    Args:
        sleep_s: Seconds to sleep after sync. Blackwell driver reclamation
                 latency is non-negotiable. 1.8s minimum on RTX 5080,
                 2.3s for SD->LTX transition, 3.0s for defrag.
        label:   Human-readable label for the VRAM peak log.

    Returns:
        Dict with ts, label, pre_gb, post_gb, sleep_s.
    """
    import gc
    from datetime import datetime

    pre_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

    # 1. ComfyUI model management
    try:
        import comfy.model_management as mm
        mm.unload_all_models()
        mm.soft_empty_cache()
    except ImportError:
        log.warning("[MemoryBoundary] comfy.model_management not available")

    # 2. Python GC
    gc.collect()

    # 3. CUDA cache flush + synchronize
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # 4. Mandatory sleep
    time.sleep(sleep_s)

    # 5. Post-boundary measurement
    post_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

    # 6. Log to vram_peaks.jsonl
    entry = {
        "ts": datetime.now().isoformat(),
        "label": label,
        "pre_gb": round(pre_gb, 3),
        "post_gb": round(post_gb, 3),
        "sleep_s": sleep_s,
    }
    try:
        log_dir = os.path.join(_REPO_ROOT, "logs")
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "vram_peaks.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass

    log.info(
        "[MemoryBoundary] %s: %.2f GB -> %.2f GB (slept %.1fs)",
        label, pre_gb, post_gb, sleep_s
    )
    _runtime_log(
        f"MemoryBoundary [{label}]: {pre_gb:.2f}GB -> {post_gb:.2f}GB "
        f"(sleep {sleep_s:.1f}s)"
    )

    # Reset peak tracker for next phase
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    return entry


# ---------------------------------------------------------------------------
# ComfyUI Node
# ---------------------------------------------------------------------------

class MemoryBoundaryNode:
    """Explicit VRAM reclamation between model phases.

    # This passthrough enforces ComfyUI execution order.
    # Removing it produces non-deterministic OOMs on Blackwell. Do not remove.
    """

    CATEGORY = _V2_CATEGORY
    FUNCTION = "execute"
    RETURN_TYPES = ("IMAGE", "STRING", "FLOAT")
    RETURN_NAMES = ("images_passthrough", "json_passthrough", "post_vram_gb")
    OUTPUT_NODE = True
    DESCRIPTION = (
        "v2.0 Memory Boundary - Forces full VRAM reclamation. "
        "Required between SD3.5 and LTX-Video phases on Blackwell."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # This passthrough enforces ComfyUI execution order.
                # Removing it produces non-deterministic OOMs on Blackwell.
                # Do not remove.
                "images": ("IMAGE", {
                    "tooltip": "Anchor images to pass through. Required dependency link.",
                }),
                "sleep_seconds": ("FLOAT", {
                    "default": 1.8, "min": 0.5, "max": 10.0, "step": 0.1,
                    "tooltip": "Blackwell driver reclamation sleep.",
                }),
                "phase_label": ("STRING", {
                    "default": "sd35_to_ltx",
                }),
            },
            "optional": {
                "json_data": ("STRING", {
                    "multiline": True, "default": "{}",
                }),
            },
        }

    def execute(self, images, sleep_seconds=1.8, phase_label="sd35_to_ltx",
                json_data="{}"):
        result = memory_boundary(sleep_s=sleep_seconds, label=phase_label)
        return (images, json_data, result["post_gb"])
