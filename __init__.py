"""
ComfyUI-OldTimeRadio — AI-Powered Sci-Fi Radio Drama Generator
================================================================

Generates full-length sci-fi anthology radio dramas using:
  - LLM local inference (Gemma series, Nemo, etc.) for story writing + director
  - Bark (Suno) TTS with emotional bracket tags [sighs] [whispers] etc.
  - 48kHz stereo spatial audio mastering (Haas effect, mid-side widening)
  - Procedural SFX (theremin, static, room tone)

Self-contained: drop into custom_nodes/ and go. No external node deps.

Audio:  ScriptWriter -> Director -> BatchBark -> SceneSequencer -> AudioEnhance -> EpisodeAssembler
Video:  EpisodeAssembler -> SignalLostVideo -> .mp4 + _treatment.txt (cast, voices, full script, stats)

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
    "OTR_LLMScriptWriter":    (".nodes.story_orchestrator", "LLMScriptWriter", " LLM Story Writer"),
    "OTR_LLMDirector":        (".nodes.story_orchestrator", "LLMDirector",      " LLM Director"),
    "OTR_BarkTTS":            (".nodes.bark_tts",           "BarkTTSNode",          " Bark TTS (Suno)"),
    "OTR_SFXGenerator":       (".nodes.sfx_generator",      "SFXGenerator",         " SFX Generator"),
    "OTR_SceneSequencer":     (".nodes.scene_sequencer",     "SceneSequencer",       " Scene Sequencer"),
    "OTR_EpisodeAssembler":   (".nodes.scene_sequencer",     "EpisodeAssembler",     " Episode Assembler"),
    "OTR_AudioEnhance":       (".nodes.audio_enhance",       "AudioEnhance",         " Spatial Audio Enhance"),
    "OTR_BatchBarkGenerator": (".nodes.batch_bark_generator", "BatchBarkGenerator",   " Batch Bark Generator"),
    "OTR_BatchKokoroGenerator":(".nodes.batch_kokoro_generator", "BatchKokoroGenerator"," Batch Kokoro (4GB)"),
    "OTR_BatchAudioGenGenerator":(".nodes.batch_audiogen_generator", "BatchAudioGenGenerator"," Batch AudioGen (Foley)"),
    "OTR_BatchProceduralSFX": (".nodes.batch_procedural_sfx", "BatchProceduralSFX",   " Batch Procedural SFX (Obsidian)"),
    "OTR_SignalLostVideo":    (".nodes.video_engine",          "SignalLostVideoRenderer", " Signal Lost Video"),
    "OTR_ProjectStateLoader": (".nodes.project_state",         "ProjectStateLoader",      " Project State Loader"),
    "OTR_KokoroAnnouncer":    (".nodes.kokoro_announcer",      "KokoroAnnouncer",         " Kokoro Announcer"),
    "OTR_MusicGenTheme":      (".nodes.musicgen_theme",        "MusicGenTheme",           " MusicGen Theme"),
    "OTR_VRAMGuardian":       (".nodes.vram_guardian",          "VRAMGuardian",            " VRAM Guardian"),
    # v2.0 Visual Generation Trio
    # Sidecar-isolated visual (stills/portraits/motion) generation from
    # OTR Director output. Audio path NEVER touched. Falls back to
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
    # v2.0 MIT-original video-chain nodes (replace VideoHelperSuite-GPL deps)
    "OTR_VideoConcat":             (".nodes.otr_video_concat",         "OTRVideoConcat",        " OTR Video Concat"),
    # v2.0 read-only Director/script adapter for multi-pass FLUX rendering
    "OTR_VideoPlan":               (".nodes.otr_video_plan",           "OTRVideoPlan",          " OTR Video Plan"),
    # v2.0 multi-clip shot expansion (needs shot durations from audio timeline)
    "OTR_ShotDurationCalculator":  (".nodes.otr_shot_duration_calculator", "OTRShotDurationCalculator", " OTR Shot Duration Calculator"),
}

for node_name, (module_path, class_name, display_name) in _NODE_MODULES.items():
    try:
        mod = importlib.import_module(module_path, package=__name__)
        cls = getattr(mod, class_name)

        # Primary registration (OTR_ prefix)
        NODE_CLASS_MAPPINGS[node_name] = cls
        NODE_DISPLAY_NAME_MAPPINGS[node_name] = display_name

        # Legacy alias registration
        # If the primary ID is "OTR_NodeName", also map "NodeName" to it.
        # This restores styled widgets in older production workflows.
        if node_name.startswith("OTR_"):
            legacy_name = node_name[4:]
            if legacy_name not in NODE_CLASS_MAPPINGS:
                NODE_CLASS_MAPPINGS[legacy_name] = cls

    except Exception as e:
        log.warning("[OldTimeRadio] Failed to load '%s': %s", node_name, e)
        print(f"[OldTimeRadio] Skipped '{node_name}': {e}")

# ─────────────────────────────────────────────────────────────────────────────
# RENAME ALIASES — keep older workflow JSONs loading after we generalize node IDs
#
# The story-writing + directing nodes were originally named after the Gemma
# model family ("OTR_Gemma4ScriptWriter" / "OTR_Gemma4Director"), but Gemma is
# only ONE of several LLM choices the dropdown exposes (Mistral-Nemo, Qwen2.5,
# etc.). The canonical IDs are now OTR_LLMScriptWriter / OTR_LLMDirector.
#
# Old workflow JSONs that reference the Gemma4 IDs continue to load via these
# aliases. Safe to remove these entries once all production workflows have been
# re-saved against the new IDs.
# ─────────────────────────────────────────────────────────────────────────────
_RENAME_ALIASES = {
    "OTR_Gemma4ScriptWriter": "OTR_LLMScriptWriter",
    "OTR_Gemma4Director":     "OTR_LLMDirector",
}
for _old_id, _new_id in _RENAME_ALIASES.items():
    if _new_id in NODE_CLASS_MAPPINGS and _old_id not in NODE_CLASS_MAPPINGS:
        NODE_CLASS_MAPPINGS[_old_id] = NODE_CLASS_MAPPINGS[_new_id]

_loaded = sum(1 for k in NODE_CLASS_MAPPINGS if k.startswith("OTR_") and k not in _RENAME_ALIASES)
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

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
