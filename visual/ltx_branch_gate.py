"""
ltx_branch_gate.py  --  OTR_LtxBranchGate ComfyUI node
=======================================================

Topology gate: takes an IMAGE (from OTR_UnloadAll) and a CLIP
(from LTXAVTextEncoderLoader), returns the CLIP unchanged.

Sprint H §3.7 fix (Jeffrey 2026-05-17 round-robin Step 2):

ComfyUI's executor topo-sorts the graph by data dependencies, not
by `order` widgets. `LTXAVTextEncoderLoader` (ComfyUI Desktop's
LTX-V text encoder loader, in `comfy_extras/nodes_lt_audio.py`)
has ZERO declared INPUT_TYPES sockets -- only widgets for
filenames. The executor therefore fires it as soon as graph
execution begins, in parallel with the FLUX-side chain. By the
time `OTR_UnloadAll` (whose IMAGE input depends on the FLUX
environment-stills branch) is ready to fire, LTX text encoder
has already attempted to load -- on top of FLUX's resident
22.7 GB. Result: VRAM contention, allocator fragmentation, and
the `0xC0000005` access violation at
`torch/storage.py:468 __getitem__` documented in
worker_iter_001.json from §3.7 retest #6.

The fix is purely topological. This gate node:

  inputs:  gate_image (IMAGE, required)  -- bound to OTR_UnloadAll.image
           clip       (CLIP, required)   -- bound to LTXAVTextEncoderLoader.CLIP
  outputs: clip       (CLIP)             -- identity passthrough

Wired between LTXAVTextEncoderLoader and its current downstream
consumer (e.g. CLIPTextEncode -> OTR_BatchLTXRender on node 55).
The CLIP value flows through unchanged, but the gate's CLIP-OUTPUT
node only becomes "ready" once BOTH its inputs are ready -- so the
downstream consumer is now backpressured behind OTR_UnloadAll's
completion. Comfy's executor lazily evaluates CLIP outputs (the
loader's weights don't materialize until something actually
references the CLIP), so the gate effectively defers the LTX text
encoder weight materialization until FLUX has been unloaded.

What this DOES NOT do:
  - Does NOT inspect CLIP internals.
  - Does NOT mutate the CLIP object.
  - Does NOT consume VRAM.
  - Does NOT log anything (passthrough silent by design).
  - Does NOT do per-line work or anything order-sensitive
    beyond signaling dependency.

OUTPUT_NODE = False: this is an intermediate dependency edge,
not a sink. Downstream consumers (CLIPTextEncode, BatchLTXRender)
are the actual outputs.

Failure mode if the topology doesn't actually defer the load:
the access violation will recur. Telemetry on OTR_UnloadAll
(VRAM allocated before/after the chain) will tell us whether
the chain actually evicted FLUX or whether FLUX survived as a
user-managed CheckpointLoaderSimple patcher outside
`mm.loaded_models()`. That's the follow-up commit's branch
(explicit MODEL/CLIP/VAE patcher dereference before the chain).

Jeffrey 2026-05-17.
"""

from __future__ import annotations

import logging

log = logging.getLogger("OTR.visual.ltx_branch_gate")


class LtxBranchGate:
    """Two-input one-output passthrough that forces topological
    dependency from OTR_UnloadAll completion to the LTX text encoder
    branch's consumer."""

    CATEGORY = "OTR/v2/Visual"
    FUNCTION = "gate"
    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gate_image": ("IMAGE", {
                    "tooltip": (
                        "Wire to OTR_UnloadAll's IMAGE output. This is "
                        "the dependency edge -- the gate cannot produce "
                        "its CLIP output until this IMAGE is ready, "
                        "which means OTR_UnloadAll must have completed "
                        "its cleanup chain first."
                    ),
                }),
                "clip": ("CLIP", {
                    "tooltip": (
                        "Wire to LTXAVTextEncoderLoader's CLIP output. "
                        "Passed through unchanged. The downstream LTX "
                        "consumer reads from this gate's CLIP output "
                        "instead of directly from the loader, which is "
                        "what creates the topological deferral."
                    ),
                }),
            },
        }

    def gate(self, gate_image, clip):
        # Silent identity passthrough. The work happens entirely in
        # ComfyUI's executor topology, not in this function body.
        log.debug(
            "[LtxBranchGate] passthrough -- gate satisfied; "
            "downstream LTX consumer unblocked"
        )
        return (clip,)


__all__ = ["LtxBranchGate"]
