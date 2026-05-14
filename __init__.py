"""
ComfyUI-OldTimeRadio — AI-Powered Sci-Fi Radio Drama Generator
================================================================

Generates full-length sci-fi anthology radio dramas using:
  - LLM local inference (Gemma series, Nemo, etc.) for story writing + director
  - Bark (Suno) TTS with emotional bracket tags [sighs] [whispers] etc.
  - 48kHz stereo spatial audio mastering (Haas effect, mid-side widening)
  - Procedural SFX (theremin, static, room tone)

Self-contained: drop into custom_nodes/ and go. No external node deps.

Audio:  LedgerScriptWriter -> FreezeCascade -> BatchBark -> SceneSequencer -> AudioEnhance -> EpisodeAssembler
Video:  EpisodeAssembler -> SignalLostVideo -> .mp4 + _treatment.txt (cast, voices, full script, stats)
        (legacy "Director" stage was removed in voice-path-cleanbreak S2)

BEST PRACTICE (per comfyui-custom-node-survival-guide Section 8):
  Uses isolated per-node loading so a broken dependency in one node
  doesn't prevent the rest from loading.

v1.0  2026-04-04  Jeffrey Brick — initial release
v1.4  2026-04-10  Jeffrey Brick — VRAM Hardening (v1.4 Flagship, 2GB Sovereignty)
"""

import importlib
import logging
import os
import warnings

log = logging.getLogger("OTR")

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL LOG / WARNING SUPPRESSION — runs once before any node module loads.
#
# Three separate systems produce noise that we don't want:
#   1. HuggingFace Hub telemetry and ETag network checks → env vars
#   2. transformers' own logging (INFO/WARNING level) → hf_logging verbosity
#   3. Python's warnings system (FutureWarning/UserWarning from transformers
#      internals and Bark's hardcoded max_length=20 kwarg) → filterwarnings
#
# Individual node files (bark_tts.py, batch_bark_generator.py) also have
# targeted filterwarnings calls as a belt-and-suspenders fallback.
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# SAFETENSORS CONVERSION MOCK — NOW HANDLED IN prestartup_script.py (earlier)
# ─────────────────────────────────────────────────────────────────────────────
# (the nuclear mock runs before this file is even executed)

# 1. Hub telemetry — disable before any transformers/huggingface_hub import
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# 2. transformers + huggingface_hub logging — errors only, no INFO/WARNING chatter
#    These are two separate logging systems — both need to be silenced.
#    The HF_TOKEN "unauthenticated requests" warning comes from huggingface_hub,
#    not transformers. No token needed — we run local_files_only=True throughout.
try:
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
except Exception:
    pass  # transformers not installed yet — will be caught at node load time
try:
    import huggingface_hub.utils._logging as hfh_logging
    hfh_logging.set_verbosity_error()
except Exception:
    pass

# 3. Python warnings — broad module-scoped filter for transformers FutureWarnings
#    (deprecation notices for APIs we don't control, e.g. Bark's generate() kwargs)
warnings.filterwarnings("ignore", category=FutureWarning, module=r"transformers\..*")
warnings.filterwarnings("ignore", category=UserWarning,   module=r"transformers\..*")

# 4. HF_TOKEN bake-in — ComfyUI's desktop process does NOT inherit user-scope
#    env vars from HKCU\Environment, so gated models (Gemma, Mistral, FLUX-dev)
#    401 on first download.  Pull the token from the user registry now and
#    export it into os.environ so every downstream loader picks it up.
try:
    from .visual._hf_token import ensure_hf_token
    ensure_hf_token()
except Exception as _hf_err:
    log.debug("[OldTimeRadio] HF_TOKEN bake-in skipped: %s", _hf_err)

# ─────────────────────────────────────────────────────────────────────────────
# ISOLATED PER-NODE LOADING
# If one node fails to import (e.g. missing transformers, parler_tts lib),
# the rest still load and work. This is critical for partial installs.
# ─────────────────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

_NODE_MODULES = {
    # key = NODE_CLASS_MAPPINGS key (permanent public ID — never rename)
    # value = (module_path, class_name, display_name)
    "OTR_LedgerScriptWriter": (".nodes.OTR_LedgerScriptWriter", "OTR_LedgerScriptWriter", " LPL Script Writer (v2.0)"),
    # Ledger Freeze Cascade (LFC sprint, 2026-05-11). Renamed from
    # OTR_LedgerScriptReviewer in commit 2 of 14. Wires AFTER
    # OTR_LedgerScriptWriter, BEFORE OTR_SceneSequencer. Existing
    # 3-pass cast-gated reviewer (Phases 1, 2, 9) is now wrapped by
    # Phase 0 (gap_audit_pre) at entry and Phase 10 (gap_audit_post +
    # freeze) at exit. The legacy OTR_LedgerScriptReviewer name is
    # registered as an alias below so existing workflow JSONs load.
    "OTR_LedgerFreezeCascade": (".nodes.OTR_LedgerFreezeCascade", "OTR_LedgerFreezeCascade", " LFC Ledger Freeze Cascade (v2.0)"),
    # Clean-break commit 12.11 (2026-05-12, C1 design call).
    # Standalone per-phase nodes for the heavy LLM phases. The main
    # cascade still owns the full chain; these are ADDITIONAL entry
    # points so operators can per-phase skip / rerun from the
    # canvas without rerunning the upstream cascade. Each defaults
    # OFF so dropping a node on the canvas is a no-op until opted
    # in.
    "OTR_LFCPhase4Scene": (".nodes.OTR_LFCPhase4Scene", "OTR_LFCPhase4Scene", " LFC Phase 4 - Scene Coherence"),
    "OTR_LFCPhase5Voice": (".nodes.OTR_LFCPhase5Voice", "OTR_LFCPhase5Voice", " LFC Phase 5 - Voice Drift"),
    "OTR_LFCPhase6Arc":   (".nodes.OTR_LFCPhase6Arc",   "OTR_LFCPhase6Arc",   " LFC Phase 6 - Episode Arc"),
    # Voice-path-cleanbreak Sprint 2 (2026-05-12): OTR_LLMDirector deleted.
    # The LLMDirector class was the legacy LLM-derived production plan
    # generator. P2 severed the voice side (commit 446ec81); Sprint 2
    # migrated the two remaining video-side consumers (OTR_SignalLostVideo,
    # OTR_VideoPlan) to read meta.visual_plan + meta.voice_assignments +
    # meta.style directly from the L3 ledger stamped by the writer.
    # The deleted Director class + registration + workflow node +
    # workflow links 17 + 38 all delete in lockstep. See:
    #   docs/voice-path-cleanbreak-execution-plan.md Sprint 2
    #   docs/2026-05-12-voice-path-cleanbreak-qa.md §3 + Q3
    #
    # Voice-path-cleanbreak 2026-05-12 (P3): the legacy single-line
    # nodes OTR_BarkTTS / OTR_SFXGenerator / OTR_VoiceRender plus the
    # pre-L3 parser-list reader OTR_BatchKokoroGenerator are deleted.
    # Their registrations are removed in lockstep with the file deletes.
    "OTR_SceneSequencer":     (".nodes.scene_sequencer",     "SceneSequencer",       " Scene Sequencer"),
    "OTR_EpisodeAssembler":   (".nodes.scene_sequencer",     "EpisodeAssembler",     " Episode Assembler"),
    "OTR_AudioEnhance":       (".nodes.audio_enhance",       "AudioEnhance",         " Spatial Audio Enhance"),
    "OTR_BatchBarkGenerator": (".nodes.batch_bark_generator", "BatchBarkGenerator",   " Batch Bark Generator"),
    "OTR_BatchAudioGenGenerator":(".nodes.batch_audiogen_generator", "BatchAudioGenGenerator"," Batch AudioGen (Foley)"),
    "OTR_BatchProceduralSFX": (".nodes.batch_procedural_sfx", "BatchProceduralSFX",   " Batch Procedural SFX (Obsidian)"),
    "OTR_SignalLostVideo":    (".nodes.video_engine",          "SignalLostVideoRenderer", " Signal Lost Video"),
    "OTR_ProjectStateLoader": (".nodes.project_state",         "ProjectStateLoader",      " Project State Loader"),
    "OTR_KokoroAnnouncer":    (".nodes.kokoro_announcer",      "KokoroAnnouncer",         " Kokoro Announcer"),
    "OTR_MusicGenTheme":      (".nodes.musicgen_theme",        "MusicGenTheme",           " MusicGen Theme"),
    "OTR_VRAMGuardian":       (".nodes.vram_guardian",          "VRAMGuardian",            " VRAM Guardian"),
    "OTR_VRAMContextTest":    (".nodes.vram_context_test",     "VRAMContextTest",         " VRAM Context Test (diagnostics)"),
    # v2.0 Visual Generation Trio
    # Sidecar-isolated visual (stills/portraits/motion) generation from
    # the L3 ledger (replaces the legacy OTR Director output the visual
    # bridge used to read). Audio path NEVER touched. Falls back to
    # OTR_SignalLostVideo on failure.  See docs/OTR_PIPELINE_EXPLAINER.md
    "OTR_VisualBridge":         (".visual.bridge",            "VisualBridge",         " Visual Bridge"),
    "OTR_VisualPoll":           (".visual.poll",              "VisualPoll",           " Visual Poll"),
    "OTR_VisualRenderer":       (".visual.renderer",          "VisualRenderer",       " Visual Renderer"),
    "OTR_VisualPromptCoercion": (".visual.prompt_coercion",   "VisualPromptCoercion", " Visual Prompt Coercion"),
    "OTR_VisualLLMSelector":    (".visual.llm_selector",      "VisualLLMSelector",    " Visual LLM Selector"),
    "OTR_VisualExtractFluxPrompt": (".visual.flux_prompt_extractor", "VisualExtractFluxPrompt", " Visual Extract FLUX Prompt"),
    "OTR_CheckpointLoaderGated":   (".visual.checkpoint_loader_gated", "CheckpointLoaderGated", " Checkpoint Loader (gated)"),
    "OTR_UnloadAll":               (".visual.unload_all",              "UnloadAll",             " Unload All (VRAM release)"),
    "OTR_BatchFluxRender":         (".visual.batch_flux_render",       "BatchFluxRender",       " Batch FLUX Render"),
    # BUG-LOCAL-078 fix (2026-05-03 EVENING). Per-cast portrait render.
    # Generates one clean head-and-shoulders FLUX portrait for each
    # cast member, saves to per-episode portraits/<char_id>_portrait.png,
    # stamps cast[i].portrait_path into the ledger so HuMo's tier 1
    # portrait lookup hits instead of falling through to the env-still
    # tier 4 stopgap.
    "OTR_BatchFluxPortraitRender": (".visual.batch_flux_portrait_render", "BatchFluxPortraitRender", " Batch FLUX Portrait Render (per-cast)"),
    # v2.0 MIT-original video-chain nodes (replace VideoHelperSuite-GPL deps)
    "OTR_VideoConcat":             (".nodes.otr_video_concat",         "OTRVideoConcat",        " OTR Video Concat"),
    # v2.0 read-only ledger/script adapter for multi-pass FLUX rendering
    # (legacy "Director" naming retired in voice-path-cleanbreak S23.9).
    "OTR_VideoPlan":               (".nodes.otr_video_plan",           "OTRVideoPlan",          " OTR Video Plan"),
    # v2.0 multi-clip shot expansion (needs shot durations from audio timeline)
    "OTR_ShotDurationCalculator":  (".nodes.otr_shot_duration_calculator", "OTRShotDurationCalculator", " OTR Shot Duration Calculator"),
    # OTR_PostAudioVideoPipeline -- DELETED S27 (commit lands in s27-
    # cleanbreak-tail). The class was a subprocess trigger for the
    # pre-2026-04-27 HuMo batch + concat pipeline; it was superseded
    # in-graph by OTR_BatchHumoRender + OTR_VideoComposite. S26 kept
    # the registration "so any old workflow JSON that still references
    # it loads without error" -- exactly the back-compat-for-old-data
    # pattern S27 was authorized to delete. Old workflow JSONs that
    # still name the type now fail to load loudly via the workflow
    # validator's DELETED_NODE_TYPES sentinel; they must be re-saved.
    # S26 Sprint 3 (T1.2): opt-in execution-time workflow contract
    # validator. Reads the workflow JSON from disk and runs the same
    # validate_workflow_contract check the S16.6 CI test runs. Place
    # as the first node in a workflow to catch contract drift at queue
    # time. ADR: docs/2026-05-13-S14_2-active-validation-ADR.md.
    "OTR_WorkflowValidator":       (".nodes._otr_workflow_validator", "WorkflowValidator", " Workflow Validator (opt-in, S14.2)"),
    # v2.0 in-graph batch HuMo lip-sync renderer. Loads HuMo + Lora +
    # CLIP + VAE + Whisper once, loops every ledger line internally,
    # writes per-line clips at output/otr/videos/<ep_id>/<line_id>.mp4.
    # Replaces scripts/render_humo_batch.py for production use; the
    # CLI script stays as an ad-hoc smoke tool.
    "OTR_BatchHumoRender":         (".nodes.batch_humo_render", "BatchHumoRender", " Batch HuMo Render"),
    # v2.0 in-graph batch LTX-2 renderer for non-character ledger lines
    # (announcer / music_open / music_close / music_inter / sfx).
    # Loads LTX-Video 2B + T5 once, loops every non-character ledger
    # line internally, feeds the radio_bookend.png as BOTH start and
    # end keyframes via LTXVAddGuide for seamless loop, writes per-line
    # clips alongside HuMo's output at
    # output/otr/videos/<ep_id>/<line_id>.mp4. Architecture locked
    # 2026-05-01 with Jeffrey after BUG-LOCAL-129 settled that HuMo
    # cannot animate non-face references; LTX is the answer for
    # "the radio is the visual performer for non-dialogue."
    "OTR_BatchLTXRender":          (".nodes.batch_ltx_render", "BatchLTXRender", " Batch LTX Render (radio for music/sfx/announcer)"),
    # v2.0 in-graph episode compositor. Pillarbox HuMo center 624x1080
    # in 1920x1080 canvas + additive-blend SignalLostVideo proc gen at
    # 50% opacity + audio mux from proc gen. Single ffmpeg invocation.
    # Replaces scripts/render_episode_concat.py for production use.
    "OTR_VideoComposite":          (".nodes.video_composite", "VideoComposite", " Video Composite (1080p)"),
    # v2.0 final-stage RTX VSR upscaler. Path-in / path-out wrapper around
    # NVIDIA's RTXVideoSuperResolution that preserves C7 audio identity:
    # decodes video frames in chunks via ffmpeg pipe, runs nvvfx HW-accel
    # upscale, encodes silent libx264 yuv420p, then muxes the original mp4
    # audio with -c:a copy (zero audio re-encode). Bypassable via the
    # `bypass` widget for raw 832x480 deliverables.
    "OTR_RTXUpscale":              (".nodes.rtx_upscale", "RTXUpscale", " RTX VSR Upscale (1080p)"),
    # BUG-LOCAL-028 fix (2026-05-03): per-episode-aware image save sink.
    # Replaces stock SaveImage nodes whose hardcoded filename_prefix
    # couldn't track the in-flight episode_id. Reads the Ledger singleton
    # at runtime and routes images to output/otr/episodes/<ep>/stills/
    # or .../portraits/. Falls back to legacy flat dirs in headless/test.
    "OTR_SaveToEpisodeWorkspace":  (".nodes.otr_save_to_episode_workspace", "SaveToEpisodeWorkspace", " Save to Episode Workspace (FLUX stills/portraits)"),
    # BUG-LOCAL-030 Phase B (2026-05-03 EVENING): post-RTXUpscale procgen
    # visual blend. Overlays 1920x1080 native procgen on the upscaled
    # HuMo + LTX composite at delivery res. Audio passes through with
    # -c:a copy so C7 byte-identity holds end-to-end. Fills the visible
    # HuMo black pillarbox bars from Phase A simple-pillarbox composite
    # with the SIGNAL LOST CRT signature (audio-reactive scanlines +
    # flicker over the otherwise-static black surround).
    "OTR_PostUpscaleProcgenBlend": (".nodes.otr_post_upscale_procgen_blend", "PostUpscaleProcgenBlend", " Post-Upscale Procgen Blend (1080p)"),
    # BUG-LOCAL-031 QA tee (2026-05-03 EVENING). Per-stage video copy
    # so smoke workflows can preserve the output of every pipeline
    # node before the next node potentially clobbers the canonical
    # filename. Pure side-effect; passes the upstream path through
    # unchanged so production wiring is not affected.
    "OTR_SaveCopy": (".nodes.otr_save_copy", "SaveCopy", " Save Copy (per-stage QA tee)"),
}

for node_name, (module_path, class_name, display_name) in _NODE_MODULES.items():
    try:
        mod = importlib.import_module(module_path, package=__name__)
        cls = getattr(mod, class_name)

        # Single canonical registration (OTR_ prefix only). The legacy
        # bare-name (NodeName) alias mirror loop was deleted in the
        # voice-path-cleanbreak 2026-05-12 sprint per the Standing
        # Directive. Saved workflow JSONs that reference bare-name
        # node types are expected to be rewritten against the OTR_
        # prefix; there is no parallel legacy-workflow path.
        NODE_CLASS_MAPPINGS[node_name] = cls
        NODE_DISPLAY_NAME_MAPPINGS[node_name] = display_name

    except Exception as e:
        log.warning("[OldTimeRadio] Failed to load '%s': %s", node_name, e)
        print(f"[OldTimeRadio] Skipped '{node_name}': {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Clean-break v2.0-alpha (2026-05-12): _RENAME_ALIASES dict removed. No
# back-compat surface. Every workflow JSON references the current canonical
# class names directly. Legacy class names (OTR_Gemma4Director,
# OTR_LedgerScriptReviewer, OTR_Gemma4ScriptWriter, OTR_LLMScriptWriter) are
# DEAD -- attempting to load a workflow that references one will fail loudly,
# which is the desired behaviour during a greenfield rewrite.
# ─────────────────────────────────────────────────────────────────────────────

_loaded = sum(1 for k in NODE_CLASS_MAPPINGS if k.startswith("OTR_"))
_total = len(_NODE_MODULES)
if _loaded == _total:
    print(f"[OldTimeRadio] OK - All {_total} nodes loaded successfully")
else:
    print(f"[OldTimeRadio] Loaded {_loaded}/{_total} nodes ({_total - _loaded} failed)")

# =====================================================================
# HTTP route: GET /otr/latest_ledger
# Exposes the freshest *_ledger.json from output/otr/audio/ (or the legacy
# output/old_time_radio/ fallback) as plain JSON over ComfyUI's existing
# HTTP server. Lets the live-run-tail Cowork artifact poll a single URL
# without needing Desktop Commander or any MCP transport.
# Wrapped in try/except so a server import failure cannot break node load.
# =====================================================================
try:
    import json as _otr_json
    import os as _otr_os
    from glob import glob as _otr_glob
    from server import PromptServer as _otr_PromptServer  # type: ignore
    from aiohttp import web as _otr_web  # type: ignore
    import folder_paths as _otr_folder_paths  # type: ignore

    _OTR_CORS_HEADERS = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
        "Cache-Control": "no-store",
    }

    @_otr_PromptServer.instance.routes.get("/otr/latest_ledger")
    async def _otr_latest_ledger(request):
        try:
            output_dir = _otr_folder_paths.get_output_directory()
            search_dirs = [
                _otr_os.path.join(output_dir, "otr", "audio"),
                _otr_os.path.join(output_dir, "old_time_radio"),
            ]
            candidates = []
            for d in search_dirs:
                if _otr_os.path.isdir(d):
                    candidates.extend(_otr_glob(_otr_os.path.join(d, "*_ledger.json")))
            if not candidates:
                return _otr_web.json_response({
                    "ok": False,
                    "reason": "no ledger files found",
                    "searched": search_dirs,
                }, headers=_OTR_CORS_HEADERS)
            candidates.sort(key=lambda p: _otr_os.path.getmtime(p), reverse=True)
            latest = candidates[0]
            with open(latest, "r", encoding="utf-8") as f:
                ledger = _otr_json.load(f)
            return _otr_web.json_response({
                "ok": True,
                "filename": _otr_os.path.basename(latest),
                "fullpath": latest,
                "mtime": _otr_os.path.getmtime(latest),
                "size": _otr_os.path.getsize(latest),
                "ledger": ledger,
            }, headers=_OTR_CORS_HEADERS)
        except Exception as exc:
            return _otr_web.json_response(
                {"ok": False, "reason": str(exc)},
                status=500,
                headers=_OTR_CORS_HEADERS,
            )

    @_otr_PromptServer.instance.routes.options("/otr/latest_ledger")
    async def _otr_latest_ledger_options(request):
        return _otr_web.Response(status=204, headers=_OTR_CORS_HEADERS)

    print("[OldTimeRadio] HTTP route registered: GET /otr/latest_ledger (with CORS)")
except Exception as _otr_route_err:
    print(f"[OldTimeRadio] HTTP route registration skipped: {_otr_route_err}")

# Phase 2A (2026-05-11): expose web/ for the act_count widget JS
# extension. ComfyUI auto-serves anything under WEB_DIRECTORY at
# /extensions/<custom-node>/ on server start; the JS file at
# web/js/otr_act_count_widget.js registers an `OTR.ActCountWidget`
# extension that lives-updates the act_count dropdown's valid range
# whenever target_words changes on OTR_LedgerScriptWriter nodes.
# Python validator (compute_episode_budget) is still authoritative;
# this JS is purely UI feedback.
WEB_DIRECTORY = "./web"

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]
