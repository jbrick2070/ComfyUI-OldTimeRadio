"""
OTR_VRAMGuardian — Explicit VRAM flush node for workflow graphs.
================================================================

Wraps the existing `force_vram_offload()` infrastructure from `_vram_log.py`
into a first-class ComfyUI node. Users wire it between heavy GPU nodes to
guarantee a clean VRAM slate.

Design:
  - Passthrough: Accepts an optional trigger input (any type) and forwards it
    unchanged. This lets you wire it inline between two nodes without breaking
    the graph.
  - Logs a VRAM snapshot before and after the flush so `otr_runtime.log` shows
    the actual reclaim.
  - Zero VRAM footprint. The node itself is pure Python orchestration.

v1.5 Phase 1 — Jeffrey Brick
"""

import logging

from ._vram_log import force_vram_offload, vram_snapshot

log = logging.getLogger("OTR")


class VRAMGuardian:
    """Explicit VRAM flush point for ComfyUI workflow graphs."""

    CATEGORY = "OldTimeRadio"
    FUNCTION = "flush"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("passthrough",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "trigger": ("STRING", {
                    "default": "",
                    "tooltip": "Wire any upstream output here to control execution order. Value is passed through unchanged."
                }),
            },
        }

    def flush(self, trigger=""):
        """Execute a full VRAM flush and return the trigger value unchanged."""
        # Snapshot before flush
        before = vram_snapshot("vram_guardian_before_flush")

        # Execute the nuclear flush
        force_vram_offload()

        # Snapshot after flush
        after = vram_snapshot("vram_guardian_after_flush")

        reclaimed_gb = max(0.0, before["current_gb"] - after["current_gb"])
        log.info(
            "[VRAMGuardian] Flush complete. Reclaimed %.3f GB "
            "(%.3f GB -> %.3f GB)",
            reclaimed_gb, before["current_gb"], after["current_gb"],
        )

        return (trigger,)


NODE_CLASS_MAPPINGS = {"OTR_VRAMGuardian": VRAMGuardian}
NODE_DISPLAY_NAME_MAPPINGS = {"OTR_VRAMGuardian": "🛡️ VRAM Guardian"}
