"""
flux_branch_gate.py  --  OTR_FluxBranchGate ComfyUI node
=========================================================

Topology gate: takes a STRING (writer ledger completion signal) +
MODEL + CLIP + VAE (from FLUX CheckpointLoaderSimple), returns the
MODEL/CLIP/VAE unchanged.

Sprint H §3.7 follow-up fix (Jeffrey 2026-05-17 round-robin Step A,
after LtxBranchGate at b5c1441 proved the pattern):

ComfyUI's executor fires `CheckpointLoaderSimple` immediately at
graph start because it has zero declared INPUT_TYPES sockets --
filename widget only. By the time `OTR_LedgerScriptWriter` reaches
its style_picker phase (mid-writer; needs to re-acquire a Gemma
slot via `request_slot`), FLUX has already resident at 22.7 GB on
the 16 GB physical card via dynamic offloader. Writer's Gemma load
hits CUDA OOM at `Currently allocated: 30.91 GiB` -- documented in
worker_iter_001.json from §3.7 retest #7.

Mirror of LtxBranchGate. Pass-through with a single sequencing
edge:

  inputs:  gate_signal  (STRING, required)  -- bound to
                                              OTR_LedgerFreezeCascade.script_json
                                              ("writer fully done + LLM
                                              unloaded" signal; the freeze
                                              cascade's finally block calls
                                              `_OTRML.unload_llm()` per
                                              commit 12.12)
           model        (MODEL, required)   -- bound to
                                              CheckpointLoaderSimple.MODEL
           clip         (CLIP, required)    -- bound to
                                              CheckpointLoaderSimple.CLIP
           vae          (VAE, required)     -- bound to
                                              CheckpointLoaderSimple.VAE
  outputs: model        (MODEL)             -- identity passthrough
           clip         (CLIP)              -- identity passthrough
           vae          (VAE)               -- identity passthrough

Wired between CheckpointLoaderSimple (node 22) and every FLUX
consumer that previously read its MODEL / CLIP / VAE outputs
directly (in workflows/otr_scifi_16gb_full.json: nodes 42, 23, 59).

Topology consequence: every FLUX-consuming node downstream now
depends on the gate's outputs, which depend on BOTH
CheckpointLoaderSimple (data) AND OTR_LedgerFreezeCascade (sequencing).
Comfy's executor will not fire any FLUX consumer until the freeze
cascade has emitted its script_json -- by which time `unload_llm()`
has already evicted Gemma from VRAM in the cascade's finally block.

Telemetry on execute(): log `torch.cuda.memory_allocated()` at fire
time. Next iter's comfy log surfaces "[FluxBranchGate] fire: VRAM
allocated=X.XX GB; writer ledger signal received". This tells us
whether ComfyUI's executor deferred FLUX's GPU materialization
until the gate (low allocated value) OR whether FLUX got pre-loaded
to GPU regardless of consumer readiness (high allocated value). The
latter would require replacing CheckpointLoaderSimple with an OTR
wrapper that defers `model_management.load_models_gpu()`.

What this DOES NOT do:
  - Does NOT inspect MODEL/CLIP/VAE internals.
  - Does NOT mutate the wrapped objects.
  - Does NOT consume VRAM beyond the log call.
  - Does NOT do per-line work or anything order-sensitive
    beyond signaling dependency.

OUTPUT_NODE = False: intermediate dependency edge, not a sink.

Jeffrey 2026-05-17.
"""

from __future__ import annotations

import logging

log = logging.getLogger("OTR.visual.flux_branch_gate")


class FluxBranchGate:
    """STRING + MODEL + CLIP + VAE pass-through that forces FLUX
    consumers to wait for the writer ledger completion signal."""

    CATEGORY = "OTR/v2/Visual"
    FUNCTION = "gate"
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gate_signal": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "forceInput": True,
                    "tooltip": (
                        "Wire to OTR_LedgerFreezeCascade.script_json "
                        "(node 62 slot 1) -- the 'writer fully done + "
                        "Gemma unloaded' signal. The freeze cascade's "
                        "finally block calls _OTRML.unload_llm() per "
                        "commit 12.12, so by the time this STRING "
                        "becomes ready, Gemma is evicted from VRAM. "
                        "Gate is forceInput=True (socket-only)."
                    ),
                }),
                "model": ("MODEL", {
                    "tooltip": (
                        "Wire to CheckpointLoaderSimple.MODEL "
                        "(FLUX checkpoint). Passed through unchanged."
                    ),
                }),
                "clip": ("CLIP", {
                    "tooltip": (
                        "Wire to CheckpointLoaderSimple.CLIP. "
                        "Passed through unchanged."
                    ),
                }),
                "vae": ("VAE", {
                    "tooltip": (
                        "Wire to CheckpointLoaderSimple.VAE. "
                        "Passed through unchanged."
                    ),
                }),
            },
        }

    def gate(self, gate_signal, model, clip, vae):
        # Sprint H §3.7 telemetry (Jeffrey 2026-05-17 round-robin
        # Step A): log VRAM allocated at fire time. Tells the next
        # iter's comfy log whether ComfyUI's executor actually
        # deferred FLUX's GPU materialization until the gate
        # (allocated ~0 GB) or pre-loaded FLUX regardless of
        # consumer readiness (allocated >= 20 GB).
        alloc_gib = float("nan")
        try:
            import torch  # local import; harmless if absent
            if torch.cuda.is_available():
                alloc_gib = torch.cuda.memory_allocated() / (1024 ** 3)
        except Exception:  # noqa: BLE001
            pass

        log.info(
            "[FluxBranchGate] fire: VRAM allocated=%.2f GiB; "
            "writer ledger signal received (len=%d)",
            alloc_gib,
            len(gate_signal) if gate_signal else 0,
        )

        return (model, clip, vae)


__all__ = ["FluxBranchGate"]
