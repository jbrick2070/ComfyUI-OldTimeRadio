"""
Batch AudioGen Generator - high-fidelity generative Foley for "Signal Lost".
==========================================================================

Replaces the previous "silence" or procedural-only SFX with high-quality sound
effects generated via facebook/audiogen-medium. 

Architectural Highlights:
  - Contextual Matching: Parses the script_json for [SFX: ...] tags and matches 
    them in order to the sfx_plan dictionary from Gemma4Director.
  - Per-Prompt Caching: SHA-256 hashed filenames under models/sfx_cache/ ensure
    we never waste VRAM or time generating the same sound twice for the same episode.
  - VRAM Discipline: Loads the model only if uncached cues exist. unloads it
    immediately after generation, returning the memory window for Bark TTS or 
    Video rendering.
  - Native transformers implementation: Low friction, no complex audiocraft 
    dependency issues.

v1.5 AudioGen Integration - Jeffrey Brick
"""

import gc
import hashlib
import json
import logging
import os
import re

import numpy as np
import torch

from ._vram_log import force_vram_offload

log = logging.getLogger("OTR")

AUDIOGEN_MODEL_ID = "facebook/audiogen-medium"
AUDIOGEN_SAMPLE_RATE = 32000 # Native rate for AudioGen
CACHE_SUBDIR = "sfx_cache"


def _cache_dir() -> str:
    """Return models/sfx_cache, creating it if needed."""
    try:
        import folder_paths
        base = os.path.join(folder_paths.models_dir, CACHE_SUBDIR)
    except Exception:
        here = os.path.dirname(os.path.abspath(__file__))
        base = os.path.normpath(os.path.join(here, "..", "..", "..", "models", CACHE_SUBDIR))
    os.makedirs(base, exist_ok=True)
    return base


def _cache_key(prompt: str, duration_sec: float, episode_seed: str) -> str:
    """Deterministic cache filename."""
    payload = f"{duration_sec}|{prompt}|{episode_seed}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:16]
    # Sanitize prompt for filename context (first 20 chars)
    safe_name = re.sub(r'[^a-zA-Z0-9]', '_', prompt[:20]).lower()
    return f"sfx_{safe_name}_{digest}.wav"


def _save_wav(path: str, waveform: np.ndarray, sample_rate: int) -> None:
    try:
        import soundfile as sf
        sf.write(path, waveform, sample_rate, subtype="FLOAT")
    except Exception as exc:
        log.warning("[BatchAudioGen] Failed to write cache %s: %s", path, exc)


def _load_cached_wav(path: str) -> torch.Tensor | None:
    if not os.path.exists(path):
        return None
    try:
        import soundfile as sf
        data, sr = sf.read(path, dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1) # force mono
        tensor = torch.from_numpy(np.asarray(data, dtype=np.float32))
        return tensor.unsqueeze(0).unsqueeze(0), sr # (1, 1, T), sr
    except Exception as exc:
        log.warning("[BatchAudioGen] Failed to read cache %s: %s", path, exc)
        return None


class BatchAudioGenGenerator:
    """Generates a batch of SFX cues from a script using AudioGen."""

    CATEGORY = "OldTimeRadio"
    FUNCTION = "generate"
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("sfx_audio_clips", "batch_log")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "script_json": ("STRING", {"multiline": True, "default": "[]"}),
                "production_plan_json": ("STRING", {"multiline": True, "default": "{}"}),
            },
            "optional": {
                "episode_seed": ("STRING", {"default": ""}),
                "model_id": (["facebook/audiogen-medium", "facebook/audiogen-small", "3", "3.0", 3, 3.0], {"default": "facebook/audiogen-medium"}),
                "guidance_scale": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10.0, "step": 0.5}),
                "default_duration": ("FLOAT", {"default": 3.0, "min": 0.5, "max": 10.0, "step": 0.5}),
            }
        }

    def generate(self, script_json, production_plan_json, episode_seed="", 
                 model_id="facebook/audiogen-medium", guidance_scale=3.0, default_duration=3.0):
        
        # [EMOJI] MANDATORY VRAM POWER WASH (Clean slate before start)
        force_vram_offload()
        
        # UI JSON back-compat fix
        if str(model_id) in ["3", "3.0"]:
            model_id = "facebook/audiogen-medium"
            
        batch_log = ["=== Batch AudioGen Generator ==="]
        
        try:
            script = json.loads(script_json)
            plan = json.loads(production_plan_json)
        except Exception as exc:
            log.error("[BatchAudioGen] Parse failed: %s", exc)
            return ({"waveform": torch.zeros(1, 1, 10), "sample_rate": 32000}, f"Error: {exc}")

        sfx_plan = plan.get("sfx_plan", [])
        
        # v1.5: Consume SFX cues directly from the canonical parser output.
        # The parser emits {"type": "sfx", "description": "..."} items inline
        # with dialogue. No duplicate regex - single source of truth.
        sfx_items = [item for item in script if item.get("type") == "sfx"]
        sfx_tags = [item.get("description", "") for item in sfx_items]
        
        if not sfx_tags:
            batch_log.append("No SFX cues found in script_json. Returning silence.")
            return ({"waveform": torch.zeros(1, 1, 10), "sample_rate": 32000}, "\n".join(batch_log))

        batch_log.append(f"Found {len(sfx_tags)} SFX cues in script.")
        
        # 2. Match tags to plan prompts
        render_queue = []
        for i, tag in enumerate(sfx_tags):
            prompt = tag
            duration = default_duration
            
            # Try to match to sfx_plan by order or description
            if i < len(sfx_plan):
                plan_entry = sfx_plan[i]
                prompt = plan_entry.get("generation_prompt") or plan_entry.get("description") or tag
            
            cache_path = os.path.join(_cache_dir(), _cache_key(prompt, duration, episode_seed))
            render_queue.append({
                "index": i,
                "tag": tag,
                "prompt": prompt,
                "duration": duration,
                "cache_path": cache_path
            })

        # 3. Check cache
        final_clips = [None] * len(render_queue)
        to_generate_indices = []
        
        for i, item in enumerate(render_queue):
            cached = _load_cached_wav(item["cache_path"])
            if cached:
                item["audio"], sr = cached
                final_clips[i] = item["audio"]
                batch_log.append(f"  [{i}] CACHE HIT: {item['tag'][:30]}")
            else:
                to_generate_indices.append(i)
                batch_log.append(f"  [{i}] MISS: {item['tag'][:30]}")

        # 4. Generate missing cues
        if to_generate_indices:
            try:
                from transformers import AutoProcessor, AudiogenForConditionalGeneration
            except ImportError as exc:
                batch_log.append(f"Error: transformers AudioGen not available: {exc}")
                # Fallback to silence for missing
                for idx in to_generate_indices:
                    final_clips[idx] = torch.zeros(1, 1, int(AUDIOGEN_SAMPLE_RATE * default_duration))
            else:
                batch_log.append(f"Loading {model_id}...")
                device = "cuda" if torch.cuda.is_available() else "cpu"
                dtype = torch.float16 if device == "cuda" else torch.float32
                
                processor = AutoProcessor.from_pretrained(model_id)
                model = AudiogenForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype).to(device)
                model.eval()
                
                tokens_per_sec = 50 # AudioGen specific approx
                
                try:
                    for idx in to_generate_indices:
                        item = render_queue[idx]
                        prompt = item["prompt"]
                        duration = item["duration"]
                        max_new_tokens = int(duration * tokens_per_sec)
                        
                        batch_log.append(f"  Generating [{idx}]: {prompt[:50]}...")
                        inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(device)
                        
                        with torch.no_grad():
                            audio_values = model.generate(
                                **inputs, 
                                max_new_tokens=max_new_tokens,
                                do_sample=True,
                                guidance_scale=guidance_scale
                            )
                        
                        audio_np = audio_values[0, 0].detach().cpu().float().numpy()
                        # Peak normalize
                        peak = np.abs(audio_np).max() or 1.0
                        audio_np = (audio_np / peak * 0.9).astype(np.float32)
                        
                        _save_wav(item["cache_path"], audio_np, AUDIOGEN_SAMPLE_RATE)
                        final_clips[idx] = torch.from_numpy(audio_np).unsqueeze(0).unsqueeze(0)
                        
                finally:
                    # Explicit VRAM cleanup
                    if 'model' in locals():
                        model.cpu()
                        del model
                    if 'processor' in locals():
                        del processor
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    batch_log.append("Model unloaded, VRAM cleared.")

        # 5. Build batched AUDIO output
        # ComfyUI AUDIO is a list or a dict with 'waveform' tensor.
        # Batching requires padding to max length.
        max_samples = max(clip.shape[2] for clip in final_clips)
        batched_waveform = torch.zeros(len(final_clips), 1, max_samples)
        
        for i, clip in enumerate(final_clips):
            samples = clip.shape[2]
            batched_waveform[i, 0, :samples] = clip[0, 0, :samples]
            
        return ({"waveform": batched_waveform, "sample_rate": AUDIOGEN_SAMPLE_RATE}, "\n".join(batch_log))

NODE_CLASS_MAPPINGS = {"OTR_BatchAudioGenGenerator": BatchAudioGenGenerator}
NODE_DISPLAY_NAME_MAPPINGS = {"OTR_BatchAudioGenGenerator": "[FAST] Batch AudioGen (Foley)"}
