"""OTR_VideoRenderBatch -- the in-process render entry that walks the registry
video engines via :mod:`nodes._otr_video_engines.render_driver` (A-S7.5).

Three modes: ``soak`` runs the full-episode A-S7.5 soak (the A-ship gate -- 40
beats, all roles, a forced mid-episode character_3d OOM converging
triposg_talk -> humo -> humo_1.7B -> still_motion with LOUD
restamps, run TWICE back-to-back for determinism, frozen audio untouched);
``single`` renders
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


def _build_render_engines_payload(manifest, vram_peak_mb):
    """Pure: the ``meta.render_engines`` payload. Preserves the existing keys
    (histogram / video_revision / by_role / vram_peak_mb) and ADDS the S-E5
    recipe receipt -- a per-beat ``delivered_engine`` + recipe/quant/LoRA/canvas/
    peak block + a by-engine roll-up -- so every saved episode self-documents
    "what did I use?" (the durable twin of no-fallbacks + dropdown labels). An
    engine that emits no receipt stamps recipe=None (never drops the row). No
    I/O; unit-testable."""
    by_role: dict[str, dict[str, int]] = {}
    per_clip: list = []
    by_engine: dict[str, dict] = {}
    for clip in (manifest or {}).get("clips") or []:
        role = str(clip.get("role") or "?")
        eng = str(clip.get("engine_id") or "?")
        by_role.setdefault(role, {})
        by_role[role][eng] = by_role[role].get(eng, 0) + 1
        receipt = {
            "recipe": clip.get("recipe"), "quant": clip.get("quant"),
            "use_lora": clip.get("use_lora"),
            "render_canvas": clip.get("render_canvas"),
            "vram_peak_mb": clip.get("vram_peak_mb"),
            # family = the engine's render family (audio_driven_face / text_to_video /
            # image_to_video ...) -- the human-readable suffix the credits MODELS.VIDEO
            # rows show (e.g. "humo . audio-driven face"). Sourced from the manifest
            # clip row (render_driver stamps clip["family"]); None on engines that
            # don't stamp it (the None IS the receipt, never a fallback).
            "family": clip.get("family"),
        }
        per_clip.append({"shot_id": clip.get("shot_id"), "role": role,
                         "delivered_engine": eng, **receipt})
        by_engine.setdefault(eng, receipt)
    return {
        "histogram": (manifest or {}).get("engine_histogram") or {},
        "video_revision": (manifest or {}).get("video_revision"),
        "by_role": by_role,
        "vram_peak_mb": vram_peak_mb,
        "per_clip": per_clip,        # E5
        "by_engine": by_engine,      # E5
    }


def _stamp_render_engines_meta(manifest, vram_peak_mb):
    """Stamp the per-role VIDEO engine map + histogram + VRAM peak + the E5
    recipe receipt into the production-ledger meta so the episode treatment /
    credits sheet reports exactly which engine + recipe rendered each beat.

    S2 durable-persistence contract (credits enrichment 2026-07-03): this
    stamp was previously best-effort (bare ``led.save()`` -- but ``save()``
    returns None on failure and never raises, so a miss was SILENT and the
    MOTION credit line would be missing at OTR_CreditsRoll time). Now routed
    through ``stamp_durable``: LOUD LedgerStampError on save failure in
    production; test-mode injects in-memory only.

    (This node already operates on the SINGLETON (not a wire-parsed local
    dict), so there is no local-to-singleton copy step here -- the payload is
    handed straight to the stamp helper.)"""
    payload = _build_render_engines_payload(manifest, vram_peak_mb)
    from .production_ledger import stamp_durable

    stamp_durable(
        meta_updates={"render_engines": payload},
        source="video_render_batch",
    )


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
                "mode": (["soak", "single", "episode"], {"default": "soak", "tooltip": (
                    "Run mode. 'episode' is the production path (renders one REAL "
                    "per-beat clip per shot from patched_ledger_json). 'soak' and "
                    "'single' are diagnostic harness modes -- some widgets below are "
                    "read ONLY in a specific mode (see engine / oom_index)."
                )}),
                "beats": ("INT", {"default": 40, "min": 1, "max": 400, "tooltip": (
                    "Diagnostic harness only (mode=soak): synthetic beat count. "
                    "Ignored in mode=episode (beats come from the planned ledger)."
                )}),
                "oom_index": ("INT", {"default": 20, "min": 0, "max": 399, "tooltip": (
                    "Diagnostic harness only -- read ONLY in mode=soak (the beat index "
                    "at which to inject a simulated OOM for fallback-trail testing). "
                    "Inert in mode=single and mode=episode."
                )}),
                "frame_count": ("INT", {"default": 25, "min": 1, "max": 240, "tooltip": (
                    "Diagnostic harness only (mode=soak): per-clip frame count. "
                    "Ignored in mode=episode (per-shot frame budget is planned upstream)."
                )}),
            },
            "optional": {
                "engine": ("STRING", {"default": "humo_1.7B", "tooltip": (
                    "Diagnostic harness only -- read ONLY in mode=single (forces the "
                    "single engine to render). Inert in mode=soak and mode=episode "
                    "(episode routes each beat by its planned per-role engine)."
                )}),
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
                        "Wire from OTR_EpisodeAssembler.output_path (the frozen "
                        "master WAV). Leave unset to degrade "
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
               engine="humo_1.7B", portrait_path="", audio_path="",
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
        # OUTPUT HYGIENE (output-tree contract OH-1, 2026-06-11): ALL JSON
        # run artifacts (episode reports AND the soak/single probe reports
        # that used to land in the now-retired top-level otr/aship) go to
        # the per-machine state tier episodes/_shared/state via the ONE
        # path authority -- never a hand-composed otr/<sub> join.
        try:                                  # durable artifacts the operator polls
            from ._otr_paths import otr_state_dir
            out_dir = str(otr_state_dir())
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
        # S-F VISUAL SMOKE CAPTURE: persist the exact node-92 INPUT (the
        # OTR_ShotLock-planned ledger + the master audio path) as a forensic
        # artifact -- a twin of the node_episode_report write below. This is the
        # ONLY place that ledger shape (video.shots + images.images) exists on
        # the wire; nothing else persists it, so scripts/otr_visual_smoke.py
        # bakes its bundle from this. Best-effort, never raises; no behavior
        # change to the render (read-only of the inputs).
        try:
            import os as _os
            from ._otr_paths import otr_state_dir as _sd
            _sdir = str(_sd())
            _os.makedirs(_sdir, exist_ok=True)
            with open(_os.path.join(_sdir, "node_episode_input.json"),
                      "w", encoding="utf-8") as _cf:
                json.dump({"ledger": ledger,
                           "master_audio_path": str(master_audio_path or "")},
                          _cf, ensure_ascii=True, default=str)
        except Exception as _cap_exc:  # noqa: BLE001 -- never break the render
            log.warning("[OTR_VideoRenderBatch] smoke-input capture skipped: %s",
                        _cap_exc)
        episode_id = str(ledger.get("episode_id")
                         or (ledger.get("meta") or {}).get("episode_id") or "")
        # Per-beat motion clause (opt-in OTR_LTX_MOTION_CLAUSE=1; default OFF -> no-op,
        # byte-identical). Pre-render pass: fills ledger['video']['shots'][i]
        # ['motion_clause'] from each beat's dialogue + cast; render reads it read-only.
        try:
            from ._otr_motion_clause import (  # type: ignore
                enabled as _mc_on, generate_motion_clauses as _mc_gen,
                make_writer_generate_fn as _mc_fn, make_name_resolver as _mc_names)
            if _mc_on():
                _mc_counts = _mc_gen(ledger, generate_fn=_mc_fn(),
                                     name_resolver=_mc_names(ledger))
                log.warning("[OTR_VideoRenderBatch] motion_clause: %s", _mc_counts)
        except Exception as exc:  # noqa: BLE001 -- never break the render
            log.warning("[OTR_VideoRenderBatch] motion_clause skipped: %s", exc)
        ep = _rd.run_real_episode(ledger,
                                  master_audio_path=str(master_audio_path or ""))
        # CLIP-FILL Piece 4: move each rendered per-beat clip out of the
        # janitor-swept _shared/tmp scratch tier into the durable
        # episodes/<ep>/clips/ workspace BEFORE building the manifest, so the
        # manifest (and the composite) reference stable paths the sweep won't
        # delete. Best-effort + LOUD; never aborts the render.
        _rd.persist_episode_clips(ep, episode_id)
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
        # Stamp the per-role render-engine map + VRAM peak into the
        # production-ledger meta for the credits roll. S2 durable-persistence
        # contract (credits enrichment 2026-07-03): this stamp is REQUIRED --
        # OTR_CreditsRoll raises on a missing MOTION receipt, so a swallowed
        # save failure here would just move the crash to mux time with less
        # context. LedgerStampError propagates (no silent skip).
        _stamp_render_engines_meta(manifest, ep.get("vram_peak_mb"))
        return (report, json.dumps(manifest, ensure_ascii=True, default=str),
                "node_episode_report.json")


__all__ = ["OTRVideoRenderBatch"]
