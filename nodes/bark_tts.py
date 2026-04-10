r"""
Bark TTS Node for ComfyUI — Old-Time Radio Edition
====================================================

Wraps Suno's Bark model for expressive character voice generation.
Bark excels at emotional delivery, laughs, sighs, and dramatic pauses —
perfect for radio drama characters.

Bark voice presets:
  v2/en_speaker_0 through v2/en_speaker_9 — varied English voices
  Each preset has a distinct timbre, pitch, and speaking style.

Special Bark tokens (insert in text):
  [laughter]  — laughing
  [laughs]    — brief laugh
  [sighs]     — sigh
  [music]     — musical interlude
  [gasps]     — gasp
  ...         — hesitation/ellipsis

v1.0  2026-04-04  Jeffrey Brick
"""

import json
import logging
import os
import re
import warnings

import numpy as np

# BEST PRACTICE (Section 8): Lazy heavy imports — torch, numpy, transformers
# imported inside methods only. Node registers instantly at startup.

log = logging.getLogger("OTR")


def _move_to_device(obj, device):
    """Recursively move tensors and numpy arrays to the target device.

    BarkProcessor returns voice presets as a nested dict ('history_prompt')
    containing numpy arrays for semantic/coarse/fine prompts. A flat
    dict comprehension misses these — this walks the full tree.
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

# ─────────────────────────────────────────────────────────────────────────────
# LOG CLEANUP — compliant fixes handle most warnings at the source.
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
# ─────────────────────────────────────────────────────────────────────────────
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

    Use torch_dtype= (not dtype=) — BarkModel wraps its own from_pretrained
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

        # ── VRAM Hardening v1.4: Strict Handoff ──
        # If Gemma is in VRAM, evict it now before loading Bark.
        try:
            from .story_orchestrator import _unload_llm
            _unload_llm()
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

            # ── STRICT DEVICE SENTRY ──
            # Force all sub-models to target device explicitly to prevent any internal
            # state from being stranded on wrong device.
            model.to(device)
            for sub in ("semantic", "coarse_acoustics", "fine_acoustics"):
                sm = getattr(model, sub, None)
                if sm is not None:
                    sm.to(device)

            # ── FIX: Patch generation configs — parent model + all sub-models ──
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


class BarkTTSNode:
    """Generate expressive character speech using Suno Bark."""

    CATEGORY = "OldTimeRadio"
    FUNCTION = "speak"
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "generation_info")

    # Bark speaker descriptions for UI reference
    SPEAKER_DESCRIPTIONS = {
        "v2/en_speaker_0": "Male, deep, authoritative (commander)",
        "v2/en_speaker_1": "Male, warm, conversational (pilot)",
        "v2/en_speaker_2": "Female, clipped, precise (officer/british-adjacent)",
        "v2/en_speaker_3": "Male, young, energetic (rebel)",
        "v2/en_speaker_4": "Female, warm, expressive (explorer)",
        "v2/en_speaker_5": "Male, older, gravelly (scientist)",
        "v2/en_speaker_6": "Male, neutral, broadcast (android)",
        "v2/en_speaker_7": "Female, young, bright (hacker)",
        "v2/en_speaker_8": "Male, deep, dramatic (engineer/LEMMY)",
        "v2/en_speaker_9": "Female, mature, sophisticated (senator)",
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "Ladies and gentlemen... [sighs] what I'm about to tell you... may change everything.",
                    "tooltip": "Text to speak. Bark tokens: [laughter] [sighs] [gasps] [music]"
                }),
                "voice_preset": ([
                    "v2/en_speaker_0", "v2/en_speaker_1", "v2/en_speaker_2",
                    "v2/en_speaker_3", "v2/en_speaker_4", "v2/en_speaker_5",
                    "v2/en_speaker_6", "v2/en_speaker_7", "v2/en_speaker_8",
                    "v2/en_speaker_9",
                ], {
                    "default": "v2/en_speaker_6",
                    "tooltip": "Bark speaker preset — each has unique voice character"
                }),
            },
            "optional": {
                "model_id": ("STRING", {"default": "suno/bark"}),
                "max_chunk_length": ("INT", {
                    "default": 200, "min": 50, "max": 400, "step": 50,
                    "tooltip": "Max characters per generation chunk (Bark works best < 200)"
                }),
                "generation_temperature": ("FLOAT", {
                    "default": 0.7, "min": 0.1, "max": 1.5, "step": 0.05,
                    "tooltip": "Higher = more expressive but less stable"
                }),
                "silence_padding_ms": ("INT", {
                    "default": 100, "min": 0, "max": 1000, "step": 50,
                    "tooltip": "Silence between chunks in ms"
                }),
            },
        }

    def speak(self, text, voice_preset, model_id="suno/bark",
              max_chunk_length=200, generation_temperature=0.7,
              silence_padding_ms=100):

        # Lazy imports (Section 8)
        import numpy as np
        import torch

        log.info(f"[BarkTTS] Generating: '{text[:60]}...' voice={voice_preset}")
        model, processor = _load_bark(model_id)
        sample_rate = model.generation_config.sample_rate  # typically 24000

        # Chunk long text (Bark works best with shorter segments)
        chunks = self._chunk_text(text, max_chunk_length)
        log.info(f"[BarkTTS] Split into {len(chunks)} chunks")

        all_audio = []
        silence = np.zeros(int(sample_rate * silence_padding_ms / 1000), dtype=np.float32)

        for i, chunk in enumerate(chunks):
            # S29: Allow users to cancel long TTS generation
            try:
                import comfy.model_management
                comfy.model_management.throw_exception_if_processing_interrupted()
            except ImportError:
                pass

            log.info(f"[BarkTTS] Chunk {i+1}/{len(chunks)}: '{chunk[:40]}...'")

            inputs = processor(chunk, voice_preset=voice_preset)
            # Recursively move ALL processor outputs to target device (CUDA or CPU).
            # Includes nested 'history_prompt' dict with voice preset numpy arrays.
            target_device = torch.device(_BARK_CACHE["device"])
            inputs = _move_to_device(inputs, target_device)

            # SENTRY AUDIT: Final verification before generation
            device_errors = []
            expected_device_type = _BARK_CACHE["device"]
            if inputs["input_ids"].device.type != expected_device_type:
                device_errors.append(f"input_ids on {inputs['input_ids'].device}")
            if "history_prompt" in inputs:
                for k, v in inputs["history_prompt"].items():
                    if torch.is_tensor(v) and v.device.type != expected_device_type:
                        device_errors.append(f"history_prompt.{k} on {v.device}")
            
            if device_errors:
                log.error(f"[BarkSentry] REJECTING: {', '.join(device_errors)}")
                raise RuntimeError(f"Device alignment failure: {device_errors}")

            _orig_tensor = torch.tensor
            _orig_arange = torch.arange
            def _tensor_cuda(*args, **kwargs):
                if "device" not in kwargs:
                    kwargs["device"] = "cuda"
                return _orig_tensor(*args, **kwargs)
            def _arange_cuda(*args, **kwargs):
                if "device" not in kwargs:
                    kwargs["device"] = "cuda"
                return _orig_arange(*args, **kwargs)
            torch.tensor = _tensor_cuda
            torch.arange = _arange_cuda
            try:
                with torch.no_grad():
                    output = model.generate(
                        **inputs,
                        do_sample=True,
                        temperature=generation_temperature,
                    )
            finally:
                torch.tensor = _orig_tensor
                torch.arange = _orig_arange

            audio_np = output.cpu().numpy().squeeze()
            all_audio.append(audio_np)

            if i < len(chunks) - 1 and silence_padding_ms > 0:
                all_audio.append(silence)

        # Concatenate all chunks
        combined = np.concatenate(all_audio)

        # ComfyUI AUDIO type contract (Section 26): dict with waveform + sample_rate
        waveform = torch.from_numpy(combined).float().unsqueeze(0).unsqueeze(0)  # (B=1, C=1, T)
        audio_out = {"waveform": waveform, "sample_rate": sample_rate}

        info = json.dumps({
            "voice_preset": voice_preset,
            "chunks": len(chunks),
            "total_samples": len(combined),
            "duration_sec": round(len(combined) / sample_rate, 2),
            "sample_rate": sample_rate,
        })

        log.info(f"[BarkTTS] Done: {len(combined)/sample_rate:.1f}s @ {sample_rate}Hz")
        return (audio_out, info)

    def _chunk_text(self, text, max_len):
        """Split text into chunks at sentence boundaries."""
        if len(text) <= max_len:
            return [text]

        chunks = []
        # Split on sentence endings first
        sentences = re.split(r'(?<=[.!?])\s+', text)
        current = ""

        for sentence in sentences:
            if len(current) + len(sentence) + 1 > max_len and current:
                chunks.append(current.strip())
                current = sentence
            else:
                current = f"{current} {sentence}" if current else sentence

        if current.strip():
            chunks.append(current.strip())

        return chunks if chunks else [text]
