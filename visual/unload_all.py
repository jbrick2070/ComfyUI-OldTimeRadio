"""
unload_all.py  --  OTR_UnloadAll ComfyUI node
==============================================
Post-render VRAM release pass. Drop this between VAEDecode and
SaveImage (or any terminal node) so the FLUX checkpoint, CLIP text
encoder, and VAE all get evicted from GPU memory as soon as the
image has been decoded.

Problem it solves (Jeffrey, 2026-04-23):
    After a successful FLUX render on the RTX 5080 Laptop, VRAM
    stays parked at ~7 GB (44% of the 16 GB ceiling) because
    ComfyUI keeps loaded models resident for the next prompt. If
    the NEXT graph run is Mistral-Nemo (6 GB NF4) or LTX-2B, the
    retained FLUX allocation pushes combined peak over 14.5 GB
    and thrashes.

What it does:
    1. Pass the IMAGE tensor through unchanged (so the downstream
       SaveImage / PreviewImage node fires normally).
    2. Call comfy.model_management.unload_all_models() which
       internally calls free_memory(1e30, get_torch_device()). This
       drops every currently-loaded-model from GPU to CPU.
    3. Call comfy.model_management.soft_empty_cache(force=True) to
       hand the freed pages back to the CUDA allocator, so
       nvidia-smi / LibreHardwareMonitor actually reflect the drop.

Design notes:
    - No torch import at module scope -- lazy inside execute().
    - OUTPUT_NODE=False so it doesn't force execution; it only runs
      when the downstream SaveImage asks for the IMAGE.
    - Accepts an optional "also_unload_llm_polish" flag so the
      visual-prompt polish LLM (Mistral-Nemo) is released in the
      same pass, matching checkpoint_loader_gated.py's pattern in
      reverse.
    - Never raises: a failed empty_cache should NOT kill the graph
      (the image is already decoded and on its way to disk).

Usage in the TEST workflow::

    VAEDecode.IMAGE  --->  OTR_UnloadAll.image  --->  SaveImage.images

After this runs, LibreHardwareMonitor should show VRAM drop back to
~1-2 GB (ComfyUI's own resident overhead), not 7 GB.
"""

from __future__ import annotations

import logging

log = logging.getLogger("OTR.visual.unload_all")


class UnloadAll:
    """IMAGE passthrough that evicts all loaded models from VRAM."""

    CATEGORY = "OTR/v2/Visual"
    FUNCTION = "execute"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": (
                        "Pass the decoded IMAGE through this node. "
                        "Typically wire VAEDecode.IMAGE in and "
                        "SaveImage.images out."
                    ),
                }),
            },
            "optional": {
                "unload_checkpoint": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Call comfy.model_management.unload_all_models() "
                        "to evict MODEL + CLIP + VAE from GPU. Turn off "
                        "only if you want the same checkpoint resident "
                        "for the next queued run (faster, but holds "
                        "~7 GB VRAM)."
                    ),
                }),
                "unload_llm_polish": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Also call visual.llm_polish.unload() to release "
                        "Mistral-Nemo (prompt polish LLM). Safe to leave "
                        "on; idempotent when LLM is not loaded."
                    ),
                }),
                "empty_cache": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Call soft_empty_cache(force=True) after "
                        "unloading so CUDA returns pages to the pool "
                        "and LibreHardwareMonitor reflects the drop."
                    ),
                }),
            },
        }

    def execute(
        self,
        image,
        unload_checkpoint: bool = True,
        unload_llm_polish: bool = True,
        empty_cache: bool = True,
    ):
        # Step 1: release the prompt-polish LLM (idempotent if unloaded).
        if unload_llm_polish:
            try:
                from . import llm_polish  # type: ignore
                llm_polish.unload()
                log.info("[UnloadAll] llm_polish.unload() called")
            except ImportError:
                try:
                    import llm_polish  # type: ignore
                    llm_polish.unload()
                    log.info("[UnloadAll] llm_polish.unload() called (flat import)")
                except ImportError:
                    log.debug("[UnloadAll] llm_polish module unavailable")
            except Exception as exc:  # noqa: BLE001
                log.warning("[UnloadAll] llm_polish.unload() errored: %s", exc)

        # Step 2: release the checkpoint (MODEL + CLIP + VAE) via
        # ComfyUI's own model_management API.
        if unload_checkpoint:
            try:
                import comfy.model_management as mm  # type: ignore
                mm.unload_all_models()
                log.info("[UnloadAll] comfy.model_management.unload_all_models() called")
            except Exception as exc:  # noqa: BLE001
                log.warning("[UnloadAll] unload_all_models() errored: %s", exc)

        # Step 3: hand freed pages back to the CUDA allocator so
        # external monitors (LHM / nvidia-smi) see the drop.
        if empty_cache:
            try:
                import comfy.model_management as mm  # type: ignore
                mm.soft_empty_cache(force=True)
                log.info("[UnloadAll] soft_empty_cache(force=True) called")
            except Exception as exc:  # noqa: BLE001
                # Fall back to raw torch if comfy module path changes
                try:
                    import gc
                    import torch
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    log.info("[UnloadAll] torch.cuda.empty_cache() fallback")
                except Exception as exc2:  # noqa: BLE001
                    log.warning(
                        "[UnloadAll] cache cleanup errored: %s / %s",
                        exc,
                        exc2,
                    )

        return (image,)


__all__ = ["UnloadAll"]
