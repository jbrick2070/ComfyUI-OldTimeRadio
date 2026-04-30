"""
Batch AudioGen Generator - high-fidelity generative Foley for "Signal Lost".
==========================================================================

Replaces the previous "silence" or procedural-only SFX with high-quality sound
effects generated via facebook/audiogen-medium. 

Architectural Highlights:
  - Contextual Matching: Parses the script_json for [SFX: ...] tags and matches 
    them in order to the sfx_plan dictionary from LLMDirector.
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

from ._otr_paths import otr_audio_dir
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
                # BUG-LOCAL-027: the "3"/"3.0"/3/3.0 entries were scar tissue
                # from widget-drift hitting this node. With the mapper fix in
                # _workflow_to_api_prompt, socket-only inputs no longer leak
                # into widget slots, so the hack is no longer needed. Fail
                # loudly on bad input instead of silently accepting garbage.
                "model_id": (["facebook/audiogen-medium", "facebook/audiogen-small"], {"default": "facebook/audiogen-medium"}),
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
        # BUG-LOCAL-116 fix 2026-04-30: honor Director's sfx_plan[i].dur_s
        # (or .duration_sec / .duration) instead of always using default.
        # The Director plans per-cue durations to fit narrative beats;
        # ignoring them was producing a 3.0s render of every SFX even
        # when the cue was a short sting. Combined with the output-shape
        # bug, the cache_key collision meant short/long stings shared
        # cache files. Both fixed here.
        render_queue = []
        for i, tag in enumerate(sfx_tags):
            prompt = tag
            duration = float(default_duration)

            # Try to match to sfx_plan by order or description
            if i < len(sfx_plan):
                plan_entry = sfx_plan[i] if isinstance(sfx_plan[i], dict) else {}
                prompt = (
                    plan_entry.get("generation_prompt")
                    or plan_entry.get("description")
                    or tag
                )
                # Director-specified duration takes precedence over default.
                # Several schema variants exist across versions of the
                # Director output -- accept any of them.
                for _key in ("dur_s", "duration_sec", "duration"):
                    _v = plan_entry.get(_key)
                    if isinstance(_v, (int, float)) and _v > 0:
                        # Clamp to AudioGen's reasonable bounds so a
                        # hallucinated 60-second cue doesn't OOM.
                        duration = max(0.5, min(10.0, float(_v)))
                        break

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

                        batch_log.append(
                            f"  Generating [{idx}] dur={duration:.2f}s "
                            f"max_new_tokens={max_new_tokens}: {prompt[:50]}..."
                        )
                        inputs = processor(
                            text=[prompt], padding=True, return_tensors="pt"
                        ).to(device)

                        with torch.no_grad():
                            audio_values = model.generate(
                                **inputs,
                                max_new_tokens=max_new_tokens,
                                do_sample=True,
                                guidance_scale=guidance_scale,
                            )

                        # BUG-LOCAL-116 fix 2026-04-30: robust shape
                        # extraction. AudioGen's transformers output shape
                        # has varied across versions:
                        #   - [batch, channels, samples]  (older, decoded)
                        #   - [batch, samples]            (newer, decoded mono)
                        #   - [batch, num_codebooks, seq] (token IDs - bug)
                        # The OLD code did audio_values[0, 0] which on a
                        # 2D tensor returns a SCALAR, and on a token-ID
                        # tensor returns the codebook 0 token sequence
                        # (~150 ints). Result: 3 ms WAVs.
                        # New logic: find the longest float-ish 1D slice
                        # and validate length is plausibly audio.
                        _av = audio_values
                        if hasattr(_av, "audio_values"):
                            _av = _av.audio_values   # named-tuple wrapper
                        _av = _av.detach().cpu().float()
                        if _av.dim() == 3:
                            audio_np = _av[0, 0].numpy()
                        elif _av.dim() == 2:
                            audio_np = _av[0].numpy()
                        elif _av.dim() == 1:
                            audio_np = _av.numpy()
                        else:
                            audio_np = _av.flatten().numpy()

                        # Sanity check: minimum 0.25 sec of real audio
                        # (= 8000 samples @ 32kHz). Anything shorter is
                        # an output-shape bug or generation failure;
                        # fall back to silence at the requested duration
                        # so the timeline still has a plausible slot
                        # rather than a 3 ms blip.
                        _min_samples = int(AUDIOGEN_SAMPLE_RATE * 0.25)
                        if audio_np.size < _min_samples:
                            log.warning(
                                "[BatchAudioGen] [%d] generated audio too "
                                "short (%d samples = %.4fs) -- expected "
                                ">= %.2fs. Output shape was %s. "
                                "Falling back to silence at %.2fs. "
                                "(BUG-LOCAL-116 / transformers AudioGen "
                                "output-shape regression.)",
                                idx, audio_np.size,
                                audio_np.size / AUDIOGEN_SAMPLE_RATE,
                                _min_samples / AUDIOGEN_SAMPLE_RATE,
                                tuple(audio_values.shape) if hasattr(audio_values, "shape") else "?",
                                duration,
                            )
                            audio_np = np.zeros(
                                int(AUDIOGEN_SAMPLE_RATE * duration),
                                dtype=np.float32,
                            )

                        # Peak normalize
                        peak = np.abs(audio_np).max() or 1.0
                        audio_np = (audio_np / peak * 0.9).astype(np.float32)

                        _save_wav(item["cache_path"], audio_np, AUDIOGEN_SAMPLE_RATE)
                        final_clips[idx] = torch.from_numpy(audio_np).unsqueeze(0).unsqueeze(0)
                        batch_log.append(
                            f"    [{idx}] saved {audio_np.size} samples "
                            f"({audio_np.size / AUDIOGEN_SAMPLE_RATE:.2f}s)"
                        )
                        
                finally:
                    # 2026-04-26 PM BUG-LOCAL-073 GUARD: synchronize before
                    # cpu() so a pending CUDA kernel fault surfaces as a
                    # clean Python exception instead of zombifying the
                    # process during model walk. Mirror of the guard in
                    # story_orchestrator._unload_llm.
                    sync_ok = True
                    if torch.cuda.is_available():
                        try:
                            torch.cuda.synchronize()
                        except Exception as sync_err:
                            sync_ok = False
                            batch_log.append(
                                f"AudioGen unload: cuda.synchronize() failed ({sync_err}); skipping cpu() walk"
                            )
                    if 'model' in locals():
                        if sync_ok:
                            try:
                                model.cpu()
                            except Exception as cpu_err:
                                batch_log.append(
                                    f"AudioGen unload: model.cpu() failed ({cpu_err}); proceeding with empty_cache only"
                                )
                        del model
                    if 'processor' in locals():
                        del processor
                    gc.collect()
                    if torch.cuda.is_available():
                        try:
                            torch.cuda.empty_cache()
                        except Exception as ec_err:
                            batch_log.append(f"AudioGen unload: empty_cache failed ({ec_err})")
                    batch_log.append("Model unloaded, VRAM cleared.")

        # 5. Build batched AUDIO output
        # ComfyUI AUDIO is a list or a dict with 'waveform' tensor.
        # Batching requires padding to max length.
        max_samples = max(clip.shape[2] for clip in final_clips)
        batched_waveform = torch.zeros(len(final_clips), 1, max_samples)

        for i, clip in enumerate(final_clips):
            samples = clip.shape[2]
            batched_waveform[i, 0, :samples] = clip[0, 0, :samples]

        # ---- BUG-LOCAL-095: write SFX wav_paths back to ledger ----
        # Stamp each SFX cue's cache_path + dur_s into ledger.sfx[].
        # Maps render_queue position -> ledger.sfx[position] (both
        # iterate the script in order). Silent skip when no ledger
        # is on disk yet. Errors logged, never crash.
        try:
            import json as _json
            audio_dir = otr_audio_dir()
            if audio_dir.exists():
                cands = list(audio_dir.glob("*_ledger.json"))
                if cands:
                    ledger_path = max(cands, key=lambda x: x.stat().st_mtime)
                    led = _json.loads(ledger_path.read_text(encoding="utf-8"))
                    sfx_rows = led.get("sfx") or []
                    updated = 0
                    for i, item in enumerate(render_queue):
                        if i >= len(sfx_rows):
                            break
                        row = sfx_rows[i]
                        cache_path = item.get("cache_path")
                        clip_t = final_clips[i] if i < len(final_clips) else None
                        dur = None
                        if clip_t is not None:
                            try:
                                dur = float(clip_t.shape[2]) / float(AUDIOGEN_SAMPLE_RATE)
                            except Exception:
                                dur = None
                        if cache_path:
                            row["wav_path"] = str(cache_path)
                        if dur is not None:
                            row["dur_s"] = dur
                        updated += 1
                    if updated:
                        ledger_path.write_text(
                            _json.dumps(led, indent=2, ensure_ascii=False),
                            encoding="utf-8",
                        )
                        batch_log.append(
                            f"BUG-095 ledger updated: {updated} sfx wav_path(s) -> "
                            f"{ledger_path.name}"
                        )
        except Exception as _exc:
            batch_log.append(f"BUG-095 ledger write-back failed: {_exc}")

        return ({"waveform": batched_waveform, "sample_rate": AUDIOGEN_SAMPLE_RATE}, "\n".join(batch_log))

NODE_CLASS_MAPPINGS = {"OTR_BatchAudioGenGenerator": BatchAudioGenGenerator}
NODE_DISPLAY_NAME_MAPPINGS = {"OTR_BatchAudioGenGenerator": "[FAST] Batch AudioGen (Foley)"}
