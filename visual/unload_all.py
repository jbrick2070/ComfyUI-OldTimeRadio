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
        """Unload models + clean CUDA, with the full chain ported from
        `nodes/batch_humo_render.py::_hard_reset_cuda_context`.

        Sprint H §3.7 fix (Jeffrey 2026-05-17 round-robin Track B
        verdict): the pre-fix `execute()` ran a 3-step subset of the
        cleanup chain BatchHumoRender uses for VRAM-pressure recovery.
        Track B isolation test proved the LTXAVTextEncoderLoader +
        Gemma 3 12B + LTX 2.3 22B path works correctly on cold VRAM
        (15.92 GB peak, 9.71s encode wall, no access violation), and
        the production crash was FLUX co-residence + dynamic
        offloader fragmentation. Beefing up `OTR_UnloadAll.execute()`
        to match the BatchHumoRender hard-reset chain gives the
        workflow's existing unload-between-phases node enough
        cleanup power to clear FLUX before the LTX text encoder
        load at order 26.

        Full chain (order matters -- model patches release VRAM
        FIRST so the allocator's free-block coalesce runs against
        actually-free pages):

            1. llm_polish.unload()              [gated by unload_llm_polish]
            2. mm.unload_all_models()           [gated by unload_checkpoint]
            3. gc.collect()                     [gated by empty_cache]
            4. mm.soft_empty_cache(force=True)  [gated by empty_cache]
            5. torch.cuda.synchronize()         [gated by empty_cache]
            6. torch.cuda.empty_cache()         [gated by empty_cache]
            7. torch.cuda.ipc_collect()         [gated by empty_cache]

        Plus before/after VRAM telemetry (allocated + reserved) so
        the operator can see whether the chain actually reclaimed
        VRAM or just ran in place -- mirrors BatchHumoRender's
        round-robin Item H telemetry.

        Best-effort throughout: every step is wrapped so a missing
        API on an older torch/Comfy combo doesn't escalate cleanup
        into a second crash. The image passthrough at end is
        unconditional -- the downstream SaveImage / consumer
        always receives the tensor regardless of cleanup outcome.
        """
        import gc

        steps_run: list[str] = []
        steps_failed: list[str] = []

        # Step 1: release the prompt-polish LLM (idempotent if unloaded).
        if unload_llm_polish:
            try:
                from . import llm_polish  # type: ignore
                llm_polish.unload()
                steps_run.append("llm_polish.unload")
            except ImportError:
                try:
                    import llm_polish  # type: ignore
                    llm_polish.unload()
                    steps_run.append("llm_polish.unload (flat)")
                except ImportError:
                    pass  # module unavailable -- not an error
            except Exception as exc:  # noqa: BLE001
                steps_failed.append(f"llm_polish.unload ({exc})")

        # Import the CUDA-side helpers once; treat their absence as a
        # soft skip rather than a hard fail. The image passthrough
        # still fires below.
        try:
            import comfy.model_management as mm  # type: ignore
        except Exception as exc:  # noqa: BLE001
            steps_failed.append(f"import comfy.model_management ({exc})")
            mm = None  # type: ignore[assignment]

        try:
            import torch  # type: ignore
        except Exception as exc:  # noqa: BLE001
            steps_failed.append(f"import torch ({exc})")
            log.warning(
                "[UnloadAll] chain SKIPPED: %s; image passthrough only",
                "; ".join(steps_failed),
            )
            return (image,)

        # Telemetry: capture before-state so the log answers
        # "did the cleanup chain actually reclaim VRAM, or did it
        # just run cleanly while the pool stayed fragmented?".
        before_allocated_gib = float("nan")
        before_reserved_gib = float("nan")
        try:
            if torch.cuda.is_available():
                before_allocated_gib = torch.cuda.memory_allocated() / (1024 ** 3)
                before_reserved_gib = torch.cuda.memory_reserved() / (1024 ** 3)
        except Exception:  # noqa: BLE001
            pass

        # Step 2: release the checkpoint (MODEL + CLIP + VAE) via
        # ComfyUI's own model_management API. mm.unload_all_models
        # internally calls free_memory(1e30, get_torch_device()),
        # which drops every currently-loaded-model patcher from GPU
        # to CPU.
        if unload_checkpoint and mm is not None:
            try:
                mm.unload_all_models()
                steps_run.append("mm.unload_all_models")
            except Exception as exc:  # noqa: BLE001
                steps_failed.append(f"mm.unload_all_models ({exc})")

        # Steps 3-7: chain ported from BatchHumoRender hard-reset.
        # All gated by `empty_cache` widget so the operator can
        # selectively run just the unload portion if needed.
        if empty_cache:
            try:
                gc.collect()
                steps_run.append("gc.collect")
            except Exception as exc:  # noqa: BLE001
                steps_failed.append(f"gc.collect ({exc})")

            if mm is not None:
                try:
                    mm.soft_empty_cache(force=True)
                    steps_run.append("mm.soft_empty_cache")
                except Exception as exc:  # noqa: BLE001
                    steps_failed.append(f"mm.soft_empty_cache ({exc})")

            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    steps_run.append("torch.cuda.synchronize")
            except Exception as exc:  # noqa: BLE001
                steps_failed.append(f"torch.cuda.synchronize ({exc})")

            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    steps_run.append("torch.cuda.empty_cache")
            except Exception as exc:  # noqa: BLE001
                steps_failed.append(f"torch.cuda.empty_cache ({exc})")

            try:
                if torch.cuda.is_available():
                    torch.cuda.ipc_collect()
                    steps_run.append("torch.cuda.ipc_collect")
            except Exception as exc:  # noqa: BLE001
                steps_failed.append(f"torch.cuda.ipc_collect ({exc})")

        # Telemetry: capture after-state + log reclaim delta.
        after_allocated_gib = float("nan")
        after_reserved_gib = float("nan")
        try:
            if torch.cuda.is_available():
                after_allocated_gib = torch.cuda.memory_allocated() / (1024 ** 3)
                after_reserved_gib = torch.cuda.memory_reserved() / (1024 ** 3)
        except Exception:  # noqa: BLE001
            pass

        # Sprint H §3.7 telemetry (Jeffrey 2026-05-17 round-robin
        # Step 2): explicit delta line so the next iter's comfy log
        # immediately answers "did mm.unload_all_models() actually
        # evict FLUX?". If delta is small (< 5 GiB), FLUX survived
        # as a user-managed CheckpointLoaderSimple patcher outside
        # `mm.loaded_models()` and the follow-up commit needs
        # explicit MODEL/CLIP/VAE patcher dereference before this
        # chain. If delta is large (>= 20 GiB), the chain did its
        # job and the gate node was the missing piece.
        try:
            alloc_delta_gib = before_allocated_gib - after_allocated_gib
        except Exception:  # noqa: BLE001
            alloc_delta_gib = float("nan")
        try:
            reserved_delta_gib = before_reserved_gib - after_reserved_gib
        except Exception:  # noqa: BLE001
            reserved_delta_gib = float("nan")

        if steps_failed:
            log.warning(
                "[UnloadAll] partial: ran=%s; failed=%s; "
                "allocated %.2f -> %.2f GiB (delta=%.2f), "
                "reserved %.2f -> %.2f GiB (delta=%.2f)",
                steps_run, steps_failed,
                before_allocated_gib, after_allocated_gib, alloc_delta_gib,
                before_reserved_gib, after_reserved_gib, reserved_delta_gib,
            )
        else:
            log.info(
                "[UnloadAll] OK: %s; "
                "allocated %.2f -> %.2f GiB (delta=%.2f), "
                "reserved %.2f -> %.2f GiB (delta=%.2f)",
                ", ".join(steps_run),
                before_allocated_gib, after_allocated_gib, alloc_delta_gib,
                before_reserved_gib, after_reserved_gib, reserved_delta_gib,
            )

        return (image,)


__all__ = ["UnloadAll"]
