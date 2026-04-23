"""
checkpoint_loader_gated.py  --  OTR_CheckpointLoaderGated ComfyUI node
=======================================================================
Wraps ComfyUI's stock CheckpointLoaderSimple with two additions:

1. A `trigger` STRING input (forceInput=True) so ComfyUI's scheduler
   cannot dispatch the checkpoint load until the upstream node that
   produces the trigger has finished. This solves the "CheckpointLoader
   runs in parallel with Mistral polish" VRAM-thrash: without a data
   dependency, ComfyUI runs CheckpointLoaderSimple as soon as it has
   a free slot, which on a 16 GB card can collide with Mistral-Nemo's
   6 GB NF4 footprint and push peak past the 14.5 GB ceiling.

2. An explicit `llm_polish.unload()` call right before the checkpoint
   load. This drops Mistral-Nemo from RAM+VRAM entirely (del model +
   gc.collect + torch.cuda.empty_cache) so FLUX's ~11 GB load starts
   from a clean slate. Same pattern bridge.py's _pre_spawn_vram_flush
   uses for the sidecar path.

Returns (MODEL, CLIP, VAE) -- drop-in replacement for
CheckpointLoaderSimple in the stock FLUX example graph. The `ckpt_name`
widget/input mirrors CheckpointLoaderSimple exactly so workflow JSON
can swap the node type without remapping any other fields.

Design notes:
  - No torch / diffusers / model imports at module scope so node-scan
    stays fast. Heavy work only inside load().
  - Delegates the actual checkpoint load to ComfyUI's own
    CheckpointLoaderSimple class (nodes.CheckpointLoaderSimple) to
    avoid duplicating or diverging from upstream loader behavior.
  - trigger value is logged but not otherwise inspected -- any STRING
    is accepted.
"""

from __future__ import annotations

import logging

log = logging.getLogger("OTR.visual.checkpoint_loader_gated")


def _folder_paths():
    """Lazy import so node-scan time stays cheap."""
    import folder_paths  # type: ignore
    return folder_paths


class CheckpointLoaderGated:
    """CheckpointLoaderSimple + trigger input + pre-load LLM unload."""

    CATEGORY = "OTR/v2/Visual"
    FUNCTION = "load"
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("MODEL", "CLIP", "VAE")

    @classmethod
    def INPUT_TYPES(cls):
        fp = _folder_paths()
        return {
            "required": {
                "trigger": ("STRING", {
                    "default": "",
                    "forceInput": True,
                    "tooltip": (
                        "Any upstream STRING link. ComfyUI will wait "
                        "for the source node to finish before this "
                        "checkpoint loader runs. Wire it to the "
                        "OTR_VisualExtractFluxPrompt prompt output "
                        "so Mistral polish completes first."
                    ),
                }),
                "ckpt_name": (fp.get_filename_list("checkpoints"), {
                    "tooltip": (
                        "FLUX checkpoint to load. Expects a Comfy-Org "
                        "packaged single-file safetensors like "
                        "flux1-dev-fp8.safetensors or "
                        "flux1-schnell-fp8.safetensors."
                    ),
                }),
            },
            "optional": {
                "unload_llm_polish": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Release Mistral-Nemo NF4 from RAM+VRAM before "
                        "loading FLUX. Prevents the combined 17+ GB "
                        "footprint from overshooting the 16 GB ceiling. "
                        "Turn off only if you explicitly want both "
                        "resident."
                    ),
                }),
            },
        }

    def load(
        self,
        trigger: str,
        ckpt_name: str,
        unload_llm_polish: bool = True,
    ):
        log.info(
            "[CheckpointLoaderGated] triggered (trigger_len=%d ckpt=%s "
            "unload_llm=%s)",
            len(trigger or ""),
            ckpt_name,
            unload_llm_polish,
        )

        # Step 1: release the prompt-polish LLM so FLUX has clean VRAM.
        if unload_llm_polish:
            try:
                # Prefer package-relative import -- works whether the
                # node pack is loaded as a subpackage or a flat module.
                from . import llm_polish  # type: ignore
                llm_polish.unload()
                log.info(
                    "[CheckpointLoaderGated] llm_polish.unload() called "
                    "(Mistral-Nemo released if it was loaded)"
                )
            except ImportError:
                try:
                    import llm_polish  # type: ignore
                    llm_polish.unload()
                    log.info(
                        "[CheckpointLoaderGated] llm_polish.unload() "
                        "called (fallback import)"
                    )
                except ImportError:
                    log.debug(
                        "[CheckpointLoaderGated] llm_polish module "
                        "unavailable; skipping unload"
                    )
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "[CheckpointLoaderGated] llm_polish.unload() "
                    "errored (%s); proceeding with checkpoint load",
                    exc,
                )

        # Step 2: delegate to ComfyUI's stock CheckpointLoaderSimple so
        # we inherit whatever loader fixes land in upstream ComfyUI. No
        # duplicated loader logic, no risk of divergence.
        import nodes  # type: ignore  # ComfyUI's built-in nodes module
        loader = nodes.CheckpointLoaderSimple()
        log.info(
            "[CheckpointLoaderGated] loading %s via "
            "nodes.CheckpointLoaderSimple",
            ckpt_name,
        )
        # CheckpointLoaderSimple.load_checkpoint returns (MODEL, CLIP, VAE)
        # as a tuple -- pass it straight through.
        return loader.load_checkpoint(ckpt_name)


__all__ = ["CheckpointLoaderGated"]
