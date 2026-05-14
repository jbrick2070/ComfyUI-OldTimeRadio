r"""
Bark TTS Node for ComfyUI - Old-Time Radio Edition
====================================================

Wraps Suno's Bark model for expressive character voice generation.
Bark excels at emotional delivery, laughs, sighs, and dramatic pauses -
perfect for radio drama characters.

Bark voice presets:
  v2/en_speaker_0 through v2/en_speaker_9 - varied English voices
  Each preset has a distinct timbre, pitch, and speaking style.

Special Bark tokens (insert in text):
  [laughter]  - laughing
  [laughs]    - brief laugh
  [sighs]     - sigh
  [music]     - musical interlude
  [gasps]     - gasp
  ...         - hesitation/ellipsis

v1.0  2026-04-04  Jeffrey Brick
"""

import json
import logging
import os
import re
import warnings

import numpy as np

# BEST PRACTICE (Section 8): Lazy heavy imports - torch, numpy, transformers
# imported inside methods only. Node registers instantly at startup.

log = logging.getLogger("OTR")


def _move_to_device(obj, device):
    """Recursively move tensors and numpy arrays to the target device.

    BarkProcessor returns voice presets as a nested dict ('history_prompt')
    containing numpy arrays for semantic/coarse/fine prompts. A flat
    dict comprehension misses these - this walks the full tree.
    """
    import torch
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_move_to_device(v, device) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_move_to_device(v, device) for v in obj)
    elif isinstance(obj, np.ndarray):
        return torch.from_numpy(obj).to(device)
    elif hasattr(obj, "to") and callable(obj.to):
        return obj.to(device)
    return obj

# -----------------------------------------------------------------------------
# LOG CLEANUP - compliant fixes handle most warnings at the source.
# These catch any residual library noise (urllib3/httpx cache-check spam,
# edge-case transformers warnings from Bark's internal sub-model pipeline).
#
# WHY warnings.filterwarnings() HERE:
#   Bark's internal generate calls hardcode max_length=20 as an explicit kwarg
#   inside its own sub-model pipeline (suno/bark source, not our code).
#   When we pass max_new_tokens, transformers sees BOTH and fires a UserWarning
#   on every single sub-model call (~20+ per line of dialogue).
#   We cannot intercept this via generation_config patching because Bark passes
#   max_length=20 as a direct kwarg that overrides the config object.
#   The only clean fix without forking Bark is filterwarnings() at module load.
# -----------------------------------------------------------------------------
import warnings
warnings.filterwarnings(
    "ignore",
    message=r".*Both.*`max_new_tokens`.*`max_length`.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*`max_length` is deprecated.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*attention_mask.*pad_token_id.*not set.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*Passing.*`generation_config`.*together with generation-related arguments.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*Setting `pad_token_id` to `eos_token_id`.*",
    category=UserWarning,
)

logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub.file_download").setLevel(logging.WARNING)

# Bounded cache with device tracking (Section 34, Section 5)
_BARK_CACHE = {"model": None, "processor": None, "device": None}


def _load_bark(model_id="suno/bark", device=None):
    """Load Bark model and processor. Caches globally with device tracking.

    BEST PRACTICES (per survival guide):
      - Section 3:  Lazy load, explicit unload available
      - Section 5:  Device alignment via cache tracking
      - Section 40: Manual VRAM management

    Use torch_dtype= (not dtype=) - BarkModel wraps its own from_pretrained
    and passes kwargs through to transformers, which expects the standard kwarg.

    Device fallback: CUDA if available, CPU otherwise (with warning).
    """
    global _BARK_CACHE
    import torch

    # Auto-detect device: CUDA if available, CPU fallback
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            log.warning("[Bark] CUDA not available. Falling back to CPU. TTS will be slow.")

    # Device change invalidation (Section 34)
    if (_BARK_CACHE["model"] is not None and
            str(_BARK_CACHE["device"]) != str(device)):
        log.info("Bark device changed, reloading")
        _unload_bark()

    if _BARK_CACHE["model"] is None:
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        # -- VRAM Hardening v1.4: Strict Handoff --
        # If Gemma is in VRAM, evict it now before loading Bark.
        # S30 B4b: route through the modern loader's unload_llm.
        try:
            from ._otr_model_loader import unload_llm
            unload_llm()
        except ImportError:
            pass
        except Exception as handoff_err:
            log.warning("[Bark] LLM handoff failed: %s", handoff_err)

        log.info(f"Loading Bark model: {model_id} on {device}")
        
        # v1.4.10 Hardening: Force cache_dir to our local Hub directory
        cache_dir_path = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
        
        try:
            from transformers import AutoProcessor, BarkModel

            # First load: download & cache. Subsequent loads: skip HTTP checks.
            try:
                processor = AutoProcessor.from_pretrained(model_id, local_files_only=True, cache_dir=cache_dir_path)
                log.info("Bark processor loaded from cache (no HTTP checks)")
            except OSError as local_err:
                log.info("[Bark] local_files_only=True failed for processor (%s), attempting Hub fallback...", local_err)
                try:
                    processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir_path)
                    log.info("Bark processor downloaded and cached")
                except Exception as hub_err:
                    log.error("[Bark] Hub fallback failed. Ensure model is downloaded or Hub is reachable: %s", hub_err)
                    raise RuntimeError(f"Failed to load Bark processor '{model_id}'. Is it downloaded? Hub error: {hub_err}") from hub_err

            # Load to target device (CUDA or CPU fallback).
            # On CUDA: Use device_map for direct CUDA load (avoids CPU intermediate state)
            # On CPU: Standard load with dtype=torch.float32 (CPU doesn't support float16 well)
            device_map = f"{device}:0" if device == "cuda" else device
            dtype = torch.float16 if device == "cuda" else torch.float32

            try:
                model = BarkModel.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                    device_map=device_map,
                    local_files_only=True,
                    cache_dir=cache_dir_path,
                )
                log.info(f"Bark model loaded from cache on {device} (no HTTP checks)")
            except OSError as local_err:
                log.info("[Bark] local_files_only=True failed for model (%s), attempting Hub fallback...", local_err)
                try:
                    model = BarkModel.from_pretrained(
                        model_id,
                        torch_dtype=dtype,
                        device_map=device_map,
                        cache_dir=cache_dir_path,
                    )
                    log.info(f"Bark model downloaded and cached on {device}")
                except Exception as hub_err:
                    log.error("[Bark] Hub fallback failed. Ensure model is downloaded or Hub is reachable: %s", hub_err)
                    raise RuntimeError(f"Failed to load Bark model '{model_id}'. Is it downloaded? Hub error: {hub_err}") from hub_err

            # -- STRICT DEVICE SENTRY --
            # Force all sub-models to target device explicitly to prevent any internal
            # state from being stranded on wrong device.
            model.to(device)
            for sub in ("semantic", "coarse_acoustics", "fine_acoustics"):
                sm = getattr(model, sub, None)
                if sm is not None:
                    sm.to(device)

            # -- FIX: Patch generation configs - parent model + all sub-models --
            # Bark's BarkModel and its sub-models ship with max_length=20 in
            # their GenerationConfig. When we call model.generate() with
            # max_new_tokens, transformers sees BOTH and logs a deprecation
            # warning for every single sub-model call (~20+ lines per line of
            # dialogue). Setting max_length=None on all configs suppresses this.
            # We also set pad_token_id explicitly so the "pad_token_id not set"
            # warning doesn't fire either.
            _configs_to_patch = [model]
            for sub_name in ("semantic", "coarse_acoustics", "fine_acoustics"):
                sub = getattr(model, sub_name, None)
                if sub is not None:
                    _configs_to_patch.append(sub)

            for obj in _configs_to_patch:
                if hasattr(obj, "generation_config"):
                    obj.generation_config.max_length = None
                    if obj.generation_config.pad_token_id is None:
                        eos = obj.generation_config.eos_token_id
                        obj.generation_config.pad_token_id = (
                            eos[0] if isinstance(eos, list) else eos
                        )

            _BARK_CACHE["model"] = model
            _BARK_CACHE["processor"] = processor
            _BARK_CACHE["device"] = device
            log.info("Bark loaded: %s on cuda (gen-config patched)", type(model).__name__)
        except Exception as e:
            log.exception("Failed to load Bark: %s", e)
            raise
    return _BARK_CACHE["model"], _BARK_CACHE["processor"]


def _unload_bark():
    """Explicitly unload Bark to free VRAM (Section 3, Section 40).

    gc.collect() before empty_cache() ensures Python destroys the model object
    before PyTorch attempts to reclaim VRAM.
    """
    global _BARK_CACHE
    import gc
    import torch
    if _BARK_CACHE["model"] is not None:
        del _BARK_CACHE["model"]
        del _BARK_CACHE["processor"]
        _BARK_CACHE = {"model": None, "processor": None, "device": None}
        gc.collect()
        torch.cuda.empty_cache()
        log.info("Bark unloaded, VRAM freed (gc.collect + empty_cache)")


# Voice-path-cleanbreak 2026-05-12 (P3, commit 83d7f17): the OTR_BarkTTS
# node class (BarkTTSNode) was deleted (legacy single-line node, unused
# in any active workflow). The _load_bark loader remains because
# batch_bark_generator.py imports it directly. Library-only module --
# no node class, no NODE_CLASS_MAPPINGS.
#
# Voice-path-cleanbreak Sprint 7.2 (2026-05-12): module renamed
# nodes/_bark_lib.py -> nodes/_otr_bark_lib.py per docs/conventions.md
# (project-prefix discipline for private library modules).
# Underscore prefix marks this as a private internal library; otr_
# prefix scopes the name to this project; _lib suffix flags it as
# library-only (no node class). Importers updated in lockstep:
#   nodes/batch_bark_generator.py
#   nodes/scene_sequencer.py (inline-Bark fallback)
#   nodes/story_orchestrator.py (Bark health check + VRAM unload)
#   tests/test_bark_ledger.py (patch target for _load_bark)
