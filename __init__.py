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
    "OTR_Gemma4ScriptWriter": (".nodes.story_orchestrator", "LLMScriptWriter", "📻 LLM Story Writer"),
    "OTR_Gemma4Director":     (".nodes.story_orchestrator", "LLMDirector",      "🎬 LLM Director"),
    "OTR_BarkTTS":            (".nodes.bark_tts",           "BarkTTSNode",          "🎙️ Bark TTS (Suno)"),
    "OTR_SFXGenerator":       (".nodes.sfx_generator",      "SFXGenerator",         "💥 SFX Generator"),
    "OTR_SceneSequencer":     (".nodes.scene_sequencer",     "SceneSequencer",       "🎞️ Scene Sequencer"),
    "OTR_EpisodeAssembler":   (".nodes.scene_sequencer",     "EpisodeAssembler",     "📼 Episode Assembler"),
    "OTR_AudioEnhance":       (".nodes.audio_enhance",       "AudioEnhance",         "🔊 Spatial Audio Enhance"),
    "OTR_BatchBarkGenerator": (".nodes.batch_bark_generator", "BatchBarkGenerator",   "⚡ Batch Bark Generator"),
    "OTR_BatchKokoroGenerator":(".nodes.batch_kokoro_generator", "BatchKokoroGenerator","⚡ Batch Kokoro (4GB)"),
    "OTR_BatchAudioGenGenerator":(".nodes.batch_audiogen_generator", "BatchAudioGenGenerator","⚡ Batch AudioGen (Foley)"),
    "OTR_BatchProceduralSFX": (".nodes.batch_procedural_sfx", "BatchProceduralSFX",   "⚡ Batch Procedural SFX (Obsidian)"),
    "OTR_SignalLostVideo":    (".nodes.video_engine",          "SignalLostVideoRenderer", "📺 Signal Lost Video"),
    "OTR_ProjectStateLoader": (".nodes.project_state",         "ProjectStateLoader",      "📖 Project State Loader"),
    "OTR_KokoroAnnouncer":    (".nodes.kokoro_announcer",      "KokoroAnnouncer",         "🎙️ Kokoro Announcer"),
    "OTR_MusicGenTheme":      (".nodes.musicgen_theme",        "MusicGenTheme",           "🎺 MusicGen Theme"),
    "OTR_VRAMGuardian":       (".nodes.vram_guardian",          "VRAMGuardian",            "🛡️ VRAM Guardian"),
    # ── v2.0 Visual Generation Trio ─────────────────────────────────
    # Sidecar-isolated visual (stills/portraits/motion) generation from
    # OTR Director output. Audio path NEVER touched. Falls back to
    # OTR_SignalLostVideo on failure.  See docs/OTR_PIPELINE_EXPLAINER.md
    "OTR_VisualBridge":         (".otr_v2.visual.bridge",            "VisualBridge",         "🌐 Visual Bridge"),
    "OTR_VisualPoll":           (".otr_v2.visual.poll",              "VisualPoll",           "⏳ Visual Poll"),
    "OTR_VisualRenderer":       (".otr_v2.visual.renderer",          "VisualRenderer",       "🎬 Visual Renderer"),
    "OTR_VisualPromptCoercion": (".otr_v2.visual.prompt_coercion",   "VisualPromptCoercion", "🧹 Visual Prompt Coercion"),
    "OTR_VisualLLMSelector":    (".otr_v2.visual.llm_selector",      "VisualLLMSelector",    "🔀 Visual LLM Selector"),
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
        print(f"[OldTimeRadio] ⚠️  Skipped '{node_name}': {e}")

_loaded = sum(1 for k in NODE_CLASS_MAPPINGS if k.startswith("OTR_"))
_total = len(_NODE_MODULES)
if _loaded == _total:
    print(f"[OldTimeRadio] OK - All {_total} nodes loaded successfully")
else:
    print(f"[OldTimeRadio] ⚠️  Loaded {_loaded}/{_total} nodes ({_total - _loaded} failed)")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
