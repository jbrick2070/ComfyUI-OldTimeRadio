"""OTR_VideoRenderBatch -- the in-process render entry that walks the registry
video engines via :mod:`nodes._otr_video_engines.render_driver` (A-S7.5).

Two modes: ``soak`` runs the full-episode A-S7.5 soak (the A-ship gate -- 40
beats, all roles, a forced mid-episode character_3d OOM converging
hunyuan3d_talk -> humo -> latentsync -> still_kenburns with LOUD restamps, run
TWICE back-to-back for determinism, frozen audio untouched); ``single`` renders
ONE shot via one engine (the focused in-process forward validation). Emits the
structured render report as a JSON STRING. Model-agnostic: no model is "primary".

Cold-import clean (V-12): heavy work + the driver import are LAZY inside the
FUNCTION; module scope imports only stdlib. UTF-8, no BOM, ASCII-only source.
"""
from __future__ import annotations

import json
import logging

log = logging.getLogger("OTR")


class OTRVideoRenderBatch:
    """Registered as ``OTR_VideoRenderBatch``. Walks the model-agnostic render
    loop in-process (NODE_CLASS_MAPPINGS populated). OUTPUT_NODE so it can be the
    terminal of a render-gate workflow."""

    CATEGORY = "OldTimeRadio/v2/video"
    FUNCTION = "render"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("render_report_json",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["soak", "single"], {"default": "soak"}),
                "beats": ("INT", {"default": 40, "min": 1, "max": 400}),
                "oom_index": ("INT", {"default": 20, "min": 0, "max": 399}),
                "frame_count": ("INT", {"default": 25, "min": 1, "max": 240}),
            },
            "optional": {
                "engine": ("STRING", {"default": "humo"}),
                "portrait_path": ("STRING", {"default": ""}),
                "audio_path": ("STRING", {"default": ""}),
            },
        }

    def render(self, mode, beats, oom_index, frame_count,
               engine="humo", portrait_path="", audio_path=""):
        # NOTE: this MUST run inside ComfyUI's executor thread (i.e. submitted via
        # /prompt, not a background HTTP-route thread): only there does ComfyUI's
        # model_management evict the umt5/whisper encoders between encode and
        # sample, keeping the heavy in-process forward under the VRAM ceiling.
        import os
        from ._otr_video_engines import render_driver as _rd
        assets = {"init_image": portrait_path or "", "audio_ref": audio_path or ""}
        if mode == "single":
            report = _rd.render_single(engine, assets=assets,
                                       frame_count=int(frame_count))
            name = "node_single_%s.json" % engine
        else:
            report = _rd.run_gpu_soak(n_beats=int(beats), oom_index=int(oom_index),
                                      frame_count=int(frame_count), assets=assets)
            name = "node_soak.json"
        ok = bool(report.get("ok"))
        payload = json.dumps(report, ensure_ascii=True, default=str)
        try:                                  # durable report the operator polls
            base = os.environ.get("OTR_OUTPUT_DIR") or "."
            out_dir = os.path.join(base, "otr", "aship")
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
                f.write(payload)
        except Exception as exc:              # noqa: BLE001
            log.warning("[OTR_VideoRenderBatch] report write failed: %s", exc)
        log.warning("[OTR_VideoRenderBatch] mode=%s ok=%s -> %s", mode, ok, name)
        return {"ui": {"text": [payload[:6000]]}, "result": (payload,)}


__all__ = ["OTRVideoRenderBatch"]
