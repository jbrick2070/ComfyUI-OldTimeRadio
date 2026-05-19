"""nodes/_otr_deferred_loaders.py -- gate-signal-dependent loader wrappers.

Sprint H 3.7 Path G (Jeffrey 2026-05-18). Retest #13 produced the
campaign's core architectural answer:

    [FluxBranchGate] fire: VRAM allocated=22.18 GiB; writer ledger
                    signal received (len=26762)

22 GiB at gate-fire time means ComfyUI's executor pre-loaded FLUX
to GPU at graph-start regardless of any downstream gate. The gate
defers FLUX *consumers* (works correctly for that scope) but the
loader itself materializes earlier in the executor's topo order
because it has zero input dependencies.

Same finding applies to the LTX text encoder loader (community node
in ComfyUI core, comfy_extras/nodes_lt_audio.py:LTXAVTextEncoderLoader
at line 167-204). It loads at graph-start regardless of LtxBranchGate.

The fix landed in this module: deferred-loader wrappers that take
`gate_signal` (STRING, forceInput=True) as a required input. The
forceInput dependency means ComfyUI's executor topo-sort puts the
loader downstream of whatever produces the gate_signal. The loader
cannot fire until upstream emits it.

Wiring contract:
    gate_signal -> OTR_LedgerFreezeCascade.script_json (same source
                   FluxBranchGate / LtxBranchGate already use).
    By the time script_json is emitted, the freeze cascade's
    finally block has already run `_OTRML.unload_llm()` (commit
    12.12), so the writer's Gemma slot is evicted from VRAM. The
    deferred loader fires with the GPU mostly empty (the loader
    sees only residual offloader allocations, ~0-2 GiB).

VRAM telemetry: every fire logs allocated VRAM at the moment the
loader runs. Next iter's comfy log surfaces:

    [DeferredCheckpointLoader] fire: VRAM allocated=X.XX GiB;
                              gate_signal len=N; ckpt=<name>
    [DeferredLtxTextEncoderLoader] fire: VRAM allocated=X.XX GiB;
                              gate_signal len=N; text_encoder=<name>;
                              ckpt=<name>

When both fire at low VRAM (< 5 GiB), the deferred-loader contract
is proven and the legacy gates (FluxBranchGate, LtxBranchGate)
can be retired in a follow-up commit.

Architectural notes (Jeffrey 2026-05-18 Path G):
  * delegate to comfy.sd internals; do NOT reimplement the loader.
  * lean prompts policy doesn't apply (these are loaders, not LLM
    calls).
  * the wrappers do NOT touch model_management.load_models_gpu
    explicitly -- comfy.sd.load_checkpoint_guess_config + load_clip
    own that call internally. The wrappers' contribution is the
    topological dependency on gate_signal that delays the call.
"""
from __future__ import annotations

import logging
from typing import Any

import folder_paths

log = logging.getLogger("OTR.nodes._otr_deferred_loaders")


def _vram_allocated_gib() -> float:
    """Best-effort VRAM allocation snapshot. NaN if torch not
    importable or CUDA missing. Pure telemetry -- not load-bearing.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 ** 3)
    except Exception:  # noqa: BLE001
        pass
    return float("nan")


def _vram_telemetry_snapshot() -> dict:
    """BUG-LOCAL-231 PARTIAL investigation (2026-05-19): capture
    a richer VRAM snapshot at deferred-loader fire/complete moments
    to disambiguate the audio-residue hypothesis from the alt-a/b/c
    hypotheses. Returns a dict with:

      - allocated_gib:  torch.cuda.memory_allocated() in GiB.
                        Python-side, doesn't include allocator-held
                        reserved blocks.
      - reserved_gib:   torch.cuda.memory_reserved() in GiB.
                        Caching allocator's reserved blocks --
                        closer to what LHM sees as physically used.
      - lhm_used_mb:    GPU Memory Used from LHM HTTP endpoint.
                        Driver-side truth; includes any non-torch
                        allocations (browser WebGL, Comfy Electron,
                        Steam shader cache, etc.). NaN if LHM is
                        not running or unreachable.

    All three are best-effort. Pure telemetry. No exceptions
    propagated -- a failed LHM fetch returns NaN for that field,
    not a raise. Per Jeffrey 2026-05-19 directive: this telemetry
    is what disambiguates the BUG-LOCAL-231 audio-residue hypothesis
    across cold-launch smokes.
    """
    snap = {
        "allocated_gib": float("nan"),
        "reserved_gib":  float("nan"),
        "lhm_used_mb":   float("nan"),
    }
    try:
        import torch
        if torch.cuda.is_available():
            snap["allocated_gib"] = torch.cuda.memory_allocated() / (1024 ** 3)
            snap["reserved_gib"]  = torch.cuda.memory_reserved()  / (1024 ** 3)
    except Exception:  # noqa: BLE001
        pass
    # LHM HTTP fetch -- Jeffrey runs LibreHardwareMonitor at
    # localhost:8085. Single shallow GPU Memory Used probe;
    # 1.5s timeout keeps this off the hot path if LHM is down.
    try:
        import json as _json
        import urllib.request
        with urllib.request.urlopen(
            "http://localhost:8085/data.json", timeout=1.5,
        ) as resp:
            data = _json.loads(resp.read().decode("utf-8"))
        # LHM returns a recursive {Children: [...]} tree.
        # Walk it iteratively looking for "GPU Memory Used"
        # leaf -- typically reports like "15921.0 MB".
        stack = [data]
        while stack:
            node = stack.pop()
            if not isinstance(node, dict):
                continue
            if (node.get("Text") == "GPU Memory Used"
                    and isinstance(node.get("Value"), str)):
                val = node["Value"].strip()
                if val.endswith(" MB"):
                    try:
                        snap["lhm_used_mb"] = float(val[:-3])
                        break
                    except ValueError:
                        pass
            for child in (node.get("Children") or []):
                stack.append(child)
    except Exception:  # noqa: BLE001
        pass
    return snap


# ---------------------------------------------------------------------------
# OTR_DeferredCheckpointLoader -- FLUX checkpoint with gate_signal dependency
# ---------------------------------------------------------------------------


class DeferredCheckpointLoader:
    """ComfyUI checkpoint loader that does NOT fire until gate_signal
    is ready. Mirrors CheckpointLoaderSimple's surface (MODEL, CLIP,
    VAE outputs; ckpt_name combo input) but adds `gate_signal` as a
    required forceInput STRING.

    Wired in workflows/otr_scifi_16gb_full.json node 22 (was the
    stock CheckpointLoaderSimple loading flux1-dev-fp8.safetensors)
    with gate_signal from OTR_LedgerFreezeCascade.script_json.

    See module docstring for design rationale + Path G context.
    """

    CATEGORY = "OTR/loaders"
    FUNCTION = "load"
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    OUTPUT_NODE = False
    # S30 B6 opt-in: ckpt_name is a model-file picker, not an LLM
    # model_id. The two-slot LLM rule does not apply.
    NON_LLM_MODEL_WIDGET_OK = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gate_signal": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "forceInput": True,
                    "tooltip": (
                        "Required STRING input wired from "
                        "OTR_LedgerFreezeCascade.script_json (or any "
                        "upstream node that emits the writer's "
                        "ledger sentinel). forceInput=True makes the "
                        "ComfyUI executor topo-sort defer this "
                        "loader's GPU materialization until the "
                        "writer phase is finished + Gemma is "
                        "evicted. Without this dependency the stock "
                        "CheckpointLoaderSimple pre-loads to GPU at "
                        "graph-start regardless of downstream gates "
                        "(retest #13 confirmed 22.18 GiB pre-loaded)."
                    ),
                }),
                "ckpt_name": (
                    folder_paths.get_filename_list("checkpoints"),
                    {"tooltip": "Checkpoint to load. Same picker shape "
                                "as the stock CheckpointLoaderSimple."},
                ),
            },
        }

    def load(self, gate_signal, ckpt_name):
        # BUG-LOCAL-231 PARTIAL investigation (2026-05-19): the
        # original `VRAM allocated=X.XX GiB` only captures
        # torch.cuda.memory_allocated() -- Python's live tensor
        # bytes. To disambiguate the audio-residue hypothesis from
        # alt-a/b/c (browser/Comfy reserve/D3D spillover), also log
        # torch.cuda.memory_reserved() (caching allocator reserve --
        # closer to LHM physical use) AND LibreHardwareMonitor's
        # GPU Memory Used (true driver-side state). One log line
        # per fire/complete moment; LHM fetch is 1.5s-timeout
        # best-effort so a missing LHM doesn't slow the load.
        before = _vram_telemetry_snapshot()
        log.info(
            "[DeferredCheckpointLoader] fire: VRAM allocated=%.2f GiB "
            "reserved=%.2f GiB lhm_used=%.0f MB; gate_signal len=%d; ckpt=%s",
            before["allocated_gib"], before["reserved_gib"],
            before["lhm_used_mb"],
            len(gate_signal) if gate_signal else 0,
            ckpt_name,
        )

        # Delegate to Comfy core's load_checkpoint_guess_config --
        # same call CheckpointLoaderSimple makes internally. No
        # re-implementation; this wrapper's only contribution is the
        # gate_signal forceInput dependency that delays the load.
        import comfy.sd
        ckpt_path = folder_paths.get_full_path_or_raise(
            "checkpoints", ckpt_name,
        )
        out = comfy.sd.load_checkpoint_guess_config(
            ckpt_path,
            output_vae=True,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths(
                "embeddings",
            ),
        )

        after = _vram_telemetry_snapshot()
        log.info(
            "[DeferredCheckpointLoader] load complete: "
            "VRAM allocated=%.2f -> %.2f GiB (delta=%.2f) "
            "reserved=%.2f -> %.2f GiB (delta=%.2f) "
            "lhm_used=%.0f -> %.0f MB (delta=%.0f); ckpt=%s",
            before["allocated_gib"], after["allocated_gib"],
            after["allocated_gib"] - before["allocated_gib"],
            before["reserved_gib"], after["reserved_gib"],
            after["reserved_gib"] - before["reserved_gib"],
            before["lhm_used_mb"], after["lhm_used_mb"],
            after["lhm_used_mb"] - before["lhm_used_mb"],
            ckpt_name,
        )
        return out[:3]


# ---------------------------------------------------------------------------
# OTR_DeferredLtxTextEncoderLoader -- LTX text encoder with gate_signal
# ---------------------------------------------------------------------------


class DeferredLtxTextEncoderLoader:
    """ComfyUI LTX text encoder loader that does NOT fire until
    gate_signal is ready. Mirrors LTXAVTextEncoderLoader's surface
    (CLIP output; text_encoder + ckpt_name + device combo inputs)
    but adds `gate_signal` as a required forceInput STRING.

    Wired in workflows/otr_scifi_16gb_full.json node 57 (was the
    stock LTXAVTextEncoderLoader loading gemma_3_12B_it_fp4_mixed +
    ltx-2.3-22b-dev) with gate_signal from
    OTR_LedgerFreezeCascade.script_json.

    See module docstring for design rationale + Path G context.

    Implementation note: LTXAVTextEncoderLoader.execute() in
    comfy_extras/nodes_lt_audio.py (line 192-204) hard-codes
    comfy.sd.CLIPType.LTXV. We mirror that.
    """

    CATEGORY = "OTR/loaders"
    FUNCTION = "load"
    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)
    OUTPUT_NODE = False
    NON_LLM_MODEL_WIDGET_OK = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gate_signal": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "forceInput": True,
                    "tooltip": (
                        "Required STRING input wired from "
                        "OTR_LedgerFreezeCascade.script_json. "
                        "forceInput=True defers this loader until "
                        "the writer phase emits the ledger "
                        "sentinel. Prevents the FLUX/LTX co-residence "
                        "crash retest #7 + #13 surfaced at "
                        "nodes_lt_audio.py:203 / "
                        "torch/storage.py:468."
                    ),
                }),
                "text_encoder": (
                    folder_paths.get_filename_list("text_encoders"),
                    {"tooltip": "Gemma text encoder safetensors."},
                ),
                "ckpt_name": (
                    folder_paths.get_filename_list("checkpoints"),
                    {"tooltip": "LTX checkpoint that carries the "
                                "second half of the CLIP weights."},
                ),
            },
            "optional": {
                "device": (
                    ["default", "cpu"],
                    {"default": "default",
                     "tooltip": "Match the stock "
                                "LTXAVTextEncoderLoader's device "
                                "combo."},
                ),
            },
        }

    def load(self, gate_signal, text_encoder, ckpt_name,
             device="default"):
        before_gib = _vram_allocated_gib()
        log.info(
            "[DeferredLtxTextEncoderLoader] fire: VRAM allocated=%.2f GiB; "
            "gate_signal len=%d; text_encoder=%s; ckpt=%s; device=%s",
            before_gib,
            len(gate_signal) if gate_signal else 0,
            text_encoder,
            ckpt_name,
            device,
        )

        # Mirror LTXAVTextEncoderLoader.execute() in
        # comfy_extras/nodes_lt_audio.py:193-204 verbatim. Same
        # CLIPType.LTXV constant, same load_clip call shape, same
        # device combo behavior. The wrapper's only contribution is
        # the gate_signal dependency.
        import comfy.sd
        clip_type = comfy.sd.CLIPType.LTXV
        clip_path1 = folder_paths.get_full_path_or_raise(
            "text_encoders", text_encoder,
        )
        clip_path2 = folder_paths.get_full_path_or_raise(
            "checkpoints", ckpt_name,
        )
        model_options: dict[str, Any] = {}
        if device == "cpu":
            import torch
            model_options["load_device"] = torch.device("cpu")
            model_options["offload_device"] = torch.device("cpu")

        clip = comfy.sd.load_clip(
            ckpt_paths=[clip_path1, clip_path2],
            embedding_directory=folder_paths.get_folder_paths(
                "embeddings",
            ),
            clip_type=clip_type,
            model_options=model_options,
        )

        after_gib = _vram_allocated_gib()
        log.info(
            "[DeferredLtxTextEncoderLoader] load complete: "
            "VRAM allocated=%.2f -> %.2f GiB (delta=%.2f); "
            "text_encoder=%s; ckpt=%s",
            before_gib, after_gib, after_gib - before_gib,
            text_encoder, ckpt_name,
        )
        return (clip,)


__all__ = [
    "DeferredCheckpointLoader",
    "DeferredLtxTextEncoderLoader",
]
