"""OTR_VideoRenderBatch -- the in-process render entry that walks the registry
video engines via :mod:`nodes._otr_video_engines.render_driver` (A-S7.5).

Three modes: ``soak`` runs the full-episode A-S7.5 soak (the A-ship gate -- 40
beats, all roles, a forced mid-episode character_3d OOM converging
hunyuan3d_talk -> humo -> latentsync -> still_kenburns with LOUD restamps, run
TWICE back-to-back for determinism, frozen audio untouched); ``single`` renders
ONE shot via one engine (the focused in-process forward validation); ``episode``
renders one REAL per-beat clip per shot from a ShotLock-planned ledger
(``run_real_episode``) and emits a beat-ordered clip manifest for the downstream
OTR_SilentComposite. Emits the structured render report as a JSON STRING.
Model-agnostic: no model is "primary".

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
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("render_report_json", "clip_manifest_json")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["soak", "single", "episode"], {"default": "soak"}),
                "beats": ("INT", {"default": 40, "min": 1, "max": 400}),
                "oom_index": ("INT", {"default": 20, "min": 0, "max": 399}),
                "frame_count": ("INT", {"default": 25, "min": 1, "max": 240}),
            },
            "optional": {
                "engine": ("STRING", {"default": "humo"}),
                "portrait_path": ("STRING", {"default": ""}),
                "audio_path": ("STRING", {"default": ""}),
                "patched_ledger_json": ("STRING", {
                    "default": "{}", "multiline": True, "forceInput": True,
                    "tooltip": (
                        "mode=episode: the OTR_ShotLock-planned (and image-genned) "
                        "ledger JSON. run_real_episode renders one REAL per-beat "
                        "clip per shot; the emitted clip manifest feeds "
                        "OTR_SilentComposite."
                    ),
                }),
                "master_audio_path": ("STRING", {
                    "default": "", "forceInput": True,
                    "tooltip": (
                        "mode=episode: path to the FROZEN master mix (MP4 or WAV). "
                        "Beats whose ledger line has no per-line *_wav_path get "
                        "audio_ref filled by slicing [start_s, start_s+dur_s] from "
                        "this file (read-only ffmpeg; master is NEVER mutated). "
                        "Wire from OTR_SignalLostVideo.video_path (the procgen mp4 "
                        "carries the frozen master audio). Leave unset to degrade "
                        "LOUD on missing per-line wavs."
                    ),
                }),
                "image_done": ("STRING", {
                    "multiline": True, "default": "", "forceInput": True,
                    "tooltip": (
                        "Image-done gate (mirrors audio_done; ST-0.2 still-spine). "
                        "Wire from OTR_ImageGenDispatcher.image_done so the video "
                        "render cannot start before every episode still exists on "
                        "disk (W4). Opaque STRING; the token is never parsed."
                    ),
                }),
            },
        }

    def render(self, mode, beats, oom_index, frame_count,
               engine="humo", portrait_path="", audio_path="",
               patched_ledger_json="{}", master_audio_path="",
               image_done=""):
        # ``image_done`` is the W4 ordering gate (opaque STRING token from
        # OTR_ImageGenDispatcher): consuming it forces ComfyUI to finish the
        # image phase -- every episode still on disk -- before this node runs.
        # NOTE: this MUST run inside ComfyUI's executor thread (i.e. submitted via
        # /prompt, not a background HTTP-route thread): only there does ComfyUI's
        # model_management evict the umt5/whisper encoders between encode and
        # sample, keeping the heavy in-process forward under the VRAM ceiling.
        # Single resident heavy engine; the per-beat detach reclaim lives inside
        # the engines (eng_humo: wrapper_bridge.reclaim_idle_models, NO
        # unload_all_models). The frozen audio section is read-only throughout.
        import os
        from ._otr_video_engines import render_driver as _rd
        manifest_payload = ""
        if mode == "episode":
            report, manifest_payload, name = self._render_episode(
                _rd, patched_ledger_json,
                master_audio_path=str(master_audio_path or ""))
        elif mode == "single":
            assets = {"init_image": portrait_path or "", "audio_ref": audio_path or ""}
            report = _rd.render_single(engine, assets=assets,
                                       frame_count=int(frame_count))
            name = "node_single_%s.json" % engine
        else:
            assets = {"init_image": portrait_path or "", "audio_ref": audio_path or ""}
            report = _rd.run_gpu_soak(n_beats=int(beats), oom_index=int(oom_index),
                                      frame_count=int(frame_count), assets=assets)
            name = "node_soak.json"
        ok = bool(report.get("ok"))
        payload = json.dumps(report, ensure_ascii=True, default=str)
        # OUTPUT HYGIENE (operator directive 2026-06-09): otr/obs holds ONLY
        # the final playable episode mp4 -- JSON run artifacts go to otr/state.
        sub = "state" if mode == "episode" else "aship"
        try:                                  # durable artifacts the operator polls
            base = os.environ.get("OTR_OUTPUT_DIR") or "."
            out_dir = os.path.join(base, "otr", sub)
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
                f.write(payload)
            if manifest_payload:
                with open(os.path.join(out_dir, "node_episode_manifest.json"),
                          "w", encoding="utf-8") as f:
                    f.write(manifest_payload)
        except Exception as exc:              # noqa: BLE001
            log.warning("[OTR_VideoRenderBatch] report write failed: %s", exc)
        log.warning("[OTR_VideoRenderBatch] mode=%s ok=%s -> %s", mode, ok, name)
        return {"ui": {"text": [payload[:6000]]},
                "result": (payload, manifest_payload)}

    @staticmethod
    def _render_episode(_rd, patched_ledger_json, master_audio_path=""):
        """Render one REAL episode from a ShotLock-planned ledger ->
        ``(report, clip_manifest_json, report_name)``. Fail-soft: a bad or empty
        ledger yields an error report + an empty manifest so the graph never
        crashes (the procgen floor still carries the visual elsewhere).

        ``master_audio_path``: path to the FROZEN master mix; forwarded to
        :func:`run_real_episode` for per-beat audio slicing.  Read-only."""
        try:
            ledger = json.loads(patched_ledger_json or "{}")
        except (ValueError, TypeError) as exc:
            return ({"ok": False, "mode": "episode",
                     "error": "patched_ledger_json is not valid JSON: %s" % exc},
                    "", "node_episode_report.json")
        if not isinstance(ledger, dict) or not (ledger.get("video") or {}).get("shots"):
            return ({"ok": False, "mode": "episode",
                     "error": "ledger has no video.shots (run OTR_ShotLock first)"},
                    "", "node_episode_report.json")
        episode_id = str(ledger.get("episode_id")
                         or (ledger.get("meta") or {}).get("episode_id") or "")
        ep = _rd.run_real_episode(ledger,
                                  master_audio_path=str(master_audio_path or ""))
        manifest = _rd.build_clip_manifest(ep, episode_id=episode_id)
        # Round 5 F2 (warn-only): the per-beat brief-composed prompts must
        # actually DIFFER -- an all-equal sha set means the beat clauses never
        # landed (the 2026-06-10 "one terse prompt x3" eyeball failure).
        diversity = _rd.ltx_prompt_diversity_status(ep.get("trace"))
        if not diversity.get("ok"):
            log.warning("[OTR_VideoRenderBatch] LTX prompt diversity FAILED: "
                        "%s brief-composed prompts all identical (%s)",
                        diversity.get("n"), diversity.get("sha8s"))
        report = {
            "ok": manifest["clip_count"] > 0, "mode": "episode",
            "episode_id": episode_id, "n_beats": manifest["n_beats"],
            "clip_count": manifest["clip_count"],
            "engine_histogram": manifest["engine_histogram"],
            "video_revision": manifest["video_revision"],
            "prompt_diversity": diversity,
            "vram_peak_mb": ep.get("vram_peak_mb"), "trace": ep.get("trace"),
        }
        return (report, json.dumps(manifest, ensure_ascii=True, default=str),
                "node_episode_report.json")


__all__ = ["OTRVideoRenderBatch"]
