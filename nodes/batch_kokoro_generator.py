r"""
OTR_BatchKokoroGenerator — Character-Grouped Parallel TTS Generation
====================================================================

Pre-computes ALL dialogue TTS audio before the SceneSequencer runs,
using the low-VRAM Kokoro v1.0 neural engine instead of Bark.

This is the cornerstone of the "Obsidian Edition" 4GB VRAM overhaul.

Pipeline position: Director -> BatchKokoroGenerator -> SceneSequencer

v1.0  2026-04-09
"""

import json
import logging
import gc
import numpy as np
import torch

from .story_orchestrator import _runtime_log
from .kokoro_announcer import _kokoro_model_dir, _ensure_voice_file, KOKORO_SAMPLE_RATE

log = logging.getLogger("OTR")

# Curated Kokoro valid voices for Obsidian 4GB mode
# Note: af_bella = Female energetic, af_sky = Female neutral, af_nicole = Female whispery
#       am_adam = Male younger, am_onyx = Male deep, am_michael = Male older (Lemmy default)
_KOKORO_VALID_VOICES = [
    "af_bella", "af_sky", "af_nicole", "af_alloy", "af_aoede", "af_kore", "af_sarah", 
    "am_adam", "am_onyx", "am_michael", "am_echo", "am_eric", "am_fenrir", "am_puck"
]

def _voice_preset_for_character(character, voice_map, voice_traits=""):
    """Determine Kokoro voice preset for a character."""
    voice_info = voice_map.get(character, {})
    preset = voice_info.get("voice_preset") or voice_info.get("bark_preset")
    if preset in _KOKORO_VALID_VOICES:
        return preset

    char_normalized = character.upper().replace(" ", "_")
    for map_key, map_val in voice_map.items():
        key_normalized = map_key.upper().replace(" ", "_")
        if (key_normalized == char_normalized or
                key_normalized in char_normalized or
                char_normalized in key_normalized):
            preset = map_val.get("voice_preset") or map_val.get("bark_preset")
            if preset in _KOKORO_VALID_VOICES:
                return preset

    # Fallback to random assignment
    import random
    rng = random.Random(hash(character))
    
    traits_lower = voice_traits.lower() if voice_traits else ""
    is_female = "female" in traits_lower or "woman" in traits_lower or "girl" in traits_lower
    is_male   = "male" in traits_lower or "man" in traits_lower or "boy" in traits_lower

    if is_female:
        pool = [v for v in _KOKORO_VALID_VOICES if v.startswith("af_")]
    elif is_male:
        pool = [v for v in _KOKORO_VALID_VOICES if v.startswith("am_")]
    else:
        pool = _KOKORO_VALID_VOICES

    preset = rng.choice(pool)
    return preset


def _clean_text_for_kokoro(text):
    """Normalize text for Kokoro. Kokoro doesn't support Bark bracket tokens."""
    import re
    # Strip Structural tags
    text = re.sub(r'\[(?:VOICE|ENV|SFX|MUSIC):[^\]]*\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'===.*?===', '', text)
    # Strip bracket/asterisk parentheticals completely (as Kokoro will simply read them aloud!)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\*.*?\*', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


class BatchKokoroGenerator:
    """Pre-compute all dialogue TTS via Kokoro."""

    CATEGORY = "OldTimeRadio"
    FUNCTION = "generate_batch"
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("tts_audio_clips", "batch_log")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "script_json": ("STRING", {
                    "multiline": True, "default": "[]",
                }),
                "production_plan_json": ("STRING", {
                    "multiline": True, "default": "{}",
                }),
            },
            "optional": {
                "speed": ("FLOAT", {
                    "default": 1.0, "min": 0.7, "max": 1.3, "step": 0.05,
                }),
            },
        }

    def generate_batch(self, script_json, production_plan_json, speed=1.0):
        script = json.loads(script_json) if isinstance(script_json, str) else script_json
        plan = json.loads(production_plan_json) if isinstance(production_plan_json, str) else production_plan_json
        voice_map = plan.get("voice_assignments", {})

        dialogue_items = []
        skipped_announcer = 0
        for i, item in enumerate(script):
            if item.get("type") == "dialogue" and item.get("line", "").strip():
                character_name = item.get("character_name", "UNKNOWN")
                if character_name.strip().upper() == "ANNOUNCER":
                    skipped_announcer += 1
                    continue
                voice_traits = item.get("voice_traits", "")
                preset = _voice_preset_for_character(character_name, voice_map, voice_traits)
                dialogue_items.append({
                    "script_idx": i,
                    "character_name": character_name,
                    "preset": preset,
                    "line": item["line"],
                })

        total_lines = len(dialogue_items)
        log.info("[BatchKokoro] Found %d dialogue lines", total_lines)

        if total_lines == 0:
            empty = {"waveform": torch.zeros(1, 1, 24000), "sample_rate": 24000}
            return (empty, "No dialogue lines found")

        # Unload Gemma
        try:
            from .story_orchestrator import _unload_llm
            _unload_llm()
        except Exception:
            pass

        # Load Kokoro Pipeline
        try:
            from kokoro import KPipeline
        except ImportError as exc:
            log.error("[BatchKokoro] kokoro package not installed: %s", exc)
            return ({"waveform": torch.zeros(1, 1, 24000), "sample_rate": KOKORO_SAMPLE_RATE}, "Missing kokoro pkg")

        pipeline = KPipeline(lang_code="a") # "a" for American English (default general)

        results = {}
        batch_log = []
        generated = 0
        
        for item in dialogue_items:
            idx = item["script_idx"]
            preset = item["preset"]
            line = _clean_text_for_kokoro(item["line"])
            
            try:
                _ensure_voice_file(preset)
                generator = pipeline(
                    line,
                    voice=preset,
                    speed=speed,
                    split_pattern=r"\n+",
                )
                
                segments = []
                for _, _, audio_data in generator:
                    if torch.is_tensor(audio_data):
                        audio_np = audio_data.detach().cpu().numpy()
                    else:
                        audio_np = np.asarray(audio_data, dtype=np.float32)
                    segments.append(audio_np.astype(np.float32).squeeze())

                if not segments:
                    raise RuntimeError("pipeline produced no audio")

                clip_np = np.concatenate(segments) if len(segments) > 1 else segments[0]
                peak = float(np.max(np.abs(clip_np))) or 1.0
                clip_np = clip_np / peak * 0.9  # peak-normalize to -1 dBFS
                
                results[idx] = (clip_np, KOKORO_SAMPLE_RATE)
                batch_log.append(f"  [{idx}] {item['character_name']}: {line[:45]}...")
                generated += 1
                
                if generated % 5 == 0:
                    _runtime_log(f"BatchKokoro: {generated}/{total_lines} complete")

            except Exception as exc:
                log.warning("[BatchKokoro] Failed line %d: %s", idx, exc)
                batch_log.append(f"  [{idx}] {item['character_name']}: FAILED - {exc}")
                results[idx] = (np.zeros(int(KOKORO_SAMPLE_RATE * len(line.split()) / 2.5), dtype=np.float32), KOKORO_SAMPLE_RATE)

        # Assemble batch
        target_sr = KOKORO_SAMPLE_RATE
        clips = []
        for item in dialogue_items:
            audio_np, sr = results[item["script_idx"]]
            clip_t = torch.from_numpy(audio_np).float()
            clips.append(clip_t)

        from torch.nn.utils.rnn import pad_sequence
        if clips:
            padded = pad_sequence(clips, batch_first=True)
            batch_tensor = padded.unsqueeze(1).cpu()
            max_len = padded.shape[-1]
        else:
            batch_tensor = torch.zeros(1, 1, target_sr)
            max_len = target_sr

        audio_out = {"waveform": batch_tensor, "sample_rate": target_sr}
        batch_log.append(f"--- Generated: {generated}/{total_lines} lines ---")
        
        try:
            if hasattr(pipeline, "model"):
                pipeline.model.to("cpu")
            del pipeline
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return (audio_out, "\n".join(batch_log))
