"""
Batch Procedural SFX - lightweight, zero-VRAM Foley for "Signal Lost".
=====================================================================

Designed for the Obsidian Edition (4GB VRAM). Mirroring the BatchAudioGen
logic, this node parses the script and generates SFX cues, but uses the
existing procedural synthesis engine (math-based) instead of a generative 
model.

v1.5 AudioGen Integration - Jeffrey Brick
"""

import json
import logging
import os
import re

import numpy as np
import torch

from .sfx_generator import SFX_GENERATORS

log = logging.getLogger("OTR")

SAMPLE_RATE = 48000 # Procedural default


class BatchProceduralSFX:
    """Generates a batch of SFX cues from a script using procedural synthesis."""

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
                "default_duration": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "volume_db": ("FLOAT", {"default": 0.0, "min": -30.0, "max": 6.0, "step": 1.0}),
            }
        }

    def generate(self, script_json, production_plan_json, default_duration=2.0, volume_db=0.0):
        batch_log = ["=== Batch Procedural SFX (Obsidian) ==="]
        
        try:
            script = json.loads(script_json)
        except Exception as exc:
            log.error("[BatchProceduralSFX] Parse failed: %s", exc)
            return ({"waveform": torch.zeros(1, 1, 10), "sample_rate": SAMPLE_RATE}, f"Error: {exc}")

        # 1. Extract all [SFX:] tags
        sfx_tags = []
        for line in script:
            content = line.get("content", "")
            matches = re.findall(r'\[SFX:\s*(.*?)\]', content)
            for m in matches:
                sfx_tags.append(m.strip().lower())
        
        if not sfx_tags:
            batch_log.append("No [SFX:] tags found. Returning silence.")
            return ({"waveform": torch.zeros(1, 1, 10), "sample_rate": SAMPLE_RATE}, "\n".join(batch_log))

        batch_log.append(f"Found {len(sfx_tags)} SFX cues.")
        
        # 2. Match tags to available procedural generators
        # We use fuzzy matching to try to find a generator like 'door_knock' if tag is 'knock'
        available_types = list(SFX_GENERATORS.keys())
        
        final_clips = []
        for i, tag in enumerate(sfx_tags):
            chosen_type = "radio_tuning" # Default fallback
            
            # Simple keyword matching
            for t in available_types:
                if t in tag or tag in t:
                    chosen_type = t
                    break
            
            # Additional semantic aliases
            if "knock" in tag or "door" in tag: chosen_type = "door_knock"
            elif "beep" in tag or "computer" in tag or "digital" in tag: chosen_type = "sci_fi_beep"
            elif "static" in tag or "noise" in tag: chosen_type = "white_noise"
            elif "boom" in tag or "thud" in tag: chosen_type = "explosion"
            elif "theremin" in tag or "eerie" in tag: chosen_type = "theremin"
            
            batch_log.append(f"  [{i}] Tag: '{tag[:20]}' -> Procedural Model: '{chosen_type}'")
            
            generator = SFX_GENERATORS[chosen_type]
            audio_np = generator(default_duration, SAMPLE_RATE)
            
            # Apply volume
            if volume_db != 0.0:
                gain = 10.0 ** (volume_db / 20.0)
                audio_np *= gain
            
            # Peak normalize
            peak = np.abs(audio_np).max()
            if peak > 0.95:
                audio_np *= (0.95 / peak)
                
            final_clips.append(torch.from_numpy(audio_np).float().unsqueeze(0).unsqueeze(0))

        # 3. Build batched AUDIO output
        max_samples = max(clip.shape[2] for clip in final_clips)
        batched_waveform = torch.zeros(len(final_clips), 1, max_samples)
        
        for i, clip in enumerate(final_clips):
            samples = clip.shape[2]
            batched_waveform[i, 0, :samples] = clip[0, 0, :samples]
            
        return ({"waveform": batched_waveform, "sample_rate": SAMPLE_RATE}, "\n".join(batch_log))

NODE_CLASS_MAPPINGS = {"OTR_BatchProceduralSFX": BatchProceduralSFX}
NODE_DISPLAY_NAME_MAPPINGS = {"OTR_BatchProceduralSFX": "[FAST] Batch Procedural SFX (Obsidian)"}
