"""
MemoryBoundary — Explicit VRAM reclamation between model phases.

Critical on RTX 5080 Blackwell: Python dereference != driver release.
This node forces a full unload + cache flush + synchronize + mandatory
sleep to let the Blackwell driver actually reclaim the allocation.

Without this, loading LTX-Video after SD3.5 will OOM even though
Python shows 0 references to the SD3.5 model.

See V2_actionplan.md Section 5 for validated peak numbers.
"""

import json
import logging
import os
import time

import torch

log = logging.getLogger("OTR")

_V2_CATEGORY = "OldTimeRadio/v2.0 Visual Drama Engine"

# ---------------------------------------------------------------------------
# Runtime log — shared helper (same path as v2_preview.py)
# ---------------------------------------------------------------------------

def _runtime_log(msg):
    """Append a timestamped line to otr_runtime.log (repo root)."""
    try:
        from datetime import datetime
        ts = datetime.now().strftime("%H:%M:%S")
        log_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "otr_runtime.log"
        )
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] [v2] {msg}\n")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# VRAM peak logger — writes to v2/logs/vram_peaks.jsonl
# ---------------------------------------------------------------------------

def _log_vram_peak(phase, peak_gb, current_gb):
    """Append a JSON line to the VRAM drift-detection log."""
    try:
        from datetime import datetime
        log_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "logs"
        )
        os.makedirs(log_dir, exist_ok=True)
        entry = {
            "ts": datetime.now().isoformat(),
            "phase": phase,
            "peak_gb": round(peak_gb, 3),
            "current_gb": round(current_gb, 3),
        }
        with open(os.path.join(log_dir, "vram_peaks.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Core boundary function — usable standalone or from the node
# ---------------------------------------------------------------------------

def memory_boundary(sleep_s=1.8, phase_label=""):
    """Full VRAM reclamation: unload all models, flush caches, sync, sleep.

    Args:
        sleep_s: Seconds to sleep after sync. Blackwell driver reclamation
                 latency is non-negotiable — tested at 1.8s minimum on
                 RTX 5080 FE. Increase to 2.5s if OOM on LTX load.
        phase_label: Human-readable label for logging.
    """
    import gc

    pre_gb = 0.0
    if torch.cuda.is_available():
        pre_gb = torch.cuda.memory_allocated() / (1024 ** 3)

    # Step 1: ComfyUI model management — unload everything
    try:
        import comfy.model_management as mm
        mm.unload_all_models()
        mm.soft_empty_cache()
    except ImportError:
        log.warning("[MemoryBoundary] comfy.model_management not available")

    # Step 2: Python garbage collection
    gc.collect()

    # Step 3: CUDA cache flush + synchronize
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Step 4: Mandatory sleep — Blackwell driver reclamation latency
    time.sleep(sleep_s)

    # Log the result
    post_gb = 0.0
    peak_gb = 0.0
    if torch.cuda.is_available():
        post_gb = torch.cuda.memory_allocated() / (1024 ** 3)
        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        # Reset peak tracker for the next phase
        torch.cuda.reset_peak_memory_stats()

    label = phase_label or "boundary"
    log.info(
        "[MemoryBoundary] %s: %.2f GB -> %.2f GB (peak was %.2f GB, slept %.1fs)",
        label, pre_gb, post_gb, peak_gb, sleep_s
    )
    _runtime_log(
        f"MemoryBoundary [{label}]: {pre_gb:.2f}GB -> {post_gb:.2f}GB "
        f"(peak {peak_gb:.2f}GB, sleep {sleep_s:.1f}s)"
    )
    _log_vram_peak(f"boundary_{label}", peak_gb, post_gb)

    return post_gb


# ---------------------------------------------------------------------------
# ComfyUI Node
# ---------------------------------------------------------------------------

class MemoryBoundaryNode:
    """Explicit VRAM reclamation between SD3.5 and LTX-Video phases.

    Place this node between the image generation outputs (CharacterForge,
    ScenePainter) and the LTX-Video checkpoint loader. It forces a full
    model unload + CUDA cache flush + driver sleep before the next model
    loads.

    The passthrough outputs let you wire dependencies in the workflow
    graph without losing data flow.
    """

    CATEGORY = _V2_CATEGORY
    FUNCTION = "execute"
    RETURN_TYPES = ("IMAGE", "STRING", "FLOAT")
    RETURN_NAMES = ("images_passthrough", "json_passthrough", "post_vram_gb")
    OUTPUT_NODE = True
    DESCRIPTION = (
        "v2.0 Memory Boundary — Forces full VRAM reclamation between "
        "model phases. Required on RTX 5080 to prevent OOM when "
        "transitioning from SD3.5 to LTX-Video."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sleep_seconds": ("FLOAT", {
                    "default": 1.8,
                    "min": 0.5,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": (
                        "Blackwell driver reclamation sleep. 1.8s validated "
                        "on RTX 5080 FE. Increase to 2.5s if OOM on next model load."
                    ),
                }),
                "phase_label": ("STRING", {
                    "default": "sd35_to_ltx",
                    "tooltip": "Label for VRAM peak log (v2/logs/vram_peaks.jsonl)",
                }),
            },
            "optional": {
                "images": ("IMAGE", {
                    "tooltip": "Anchor images to pass through (from ScenePainter/Compositor)",
                }),
                "json_data": ("STRING", {
                    "multiline": True,
                    "default": "{}",
                    "tooltip": "JSON data to pass through (production_plan, prompts, etc.)",
                }),
            },
        }

    def execute(self, sleep_seconds=1.8, phase_label="sd35_to_ltx",
                images=None, json_data="{}"):

        post_gb = memory_boundary(
            sleep_s=sleep_seconds,
            phase_label=phase_label,
        )

        # Passthrough: forward inputs unchanged so downstream nodes
        # can depend on this boundary without losing data flow.
        if images is None:
            images = torch.zeros([1, 64, 64, 3], dtype=torch.float32)

        return (images, json_data, post_gb)
