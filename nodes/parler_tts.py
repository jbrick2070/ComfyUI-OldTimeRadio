r"""
Parler-TTS Node for ComfyUI — Old-Time Radio Edition
======================================================

Wraps Parler-TTS for controllable voice generation via text descriptions.
Unlike Bark (preset voices), Parler lets you describe the voice you want:
  "A male announcer speaking in a 1940s transatlantic accent, warm baritone,
   through a vintage radio microphone with slight room reverb"

Perfect for the ANNOUNCER and NARRATOR roles where you need precise control
over the vocal quality to match that authentic old-time radio feel.

v1.0  2026-04-04  Jeffrey Brick
"""

import json
import logging
import re

# Lazy heavy imports (Section 8) — torch, numpy inside methods only

log = logging.getLogger("OTR")

# Bounded cache with device tracking (Section 34, Section 5)
_PARLER_CACHE = {"model": None, "tokenizer": None, "desc_tokenizer": None, "device": None}


def _load_parler(model_id="parler-tts/parler-tts-large-v1", device="cuda"):
    """Load Parler-TTS model and tokenizers with device tracking.

    BEST PRACTICES (per survival guide):
      - Section 3:  Lazy load + explicit unload
      - Section 5:  Device alignment
      - Section 34: Cache invalidation on device change

    Use torch_dtype= (not dtype=) — ParlerTTS wraps its own from_pretrained()
    and passes kwargs through to transformers, which expects the standard kwarg.
    """
    global _PARLER_CACHE
    import torch

    if (_PARLER_CACHE["model"] is not None and
            str(_PARLER_CACHE["device"]) != str(device)):
        log.info("Parler device changed, reloading")
        _unload_parler()

    if _PARLER_CACHE["model"] is None:
        log.info(f"Loading Parler-TTS: {model_id}")
        try:
            from parler_tts import ParlerTTSForConditionalGeneration
            from transformers import AutoTokenizer

            # Parler wraps transformers base — needs torch_dtype, not dtype
            model = ParlerTTSForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16,  # Parler-specific: uses transformers kwarg
            ).to(device).eval()

            tokenizer = AutoTokenizer.from_pretrained(model_id)
            desc_tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")

            _PARLER_CACHE["model"] = model
            _PARLER_CACHE["tokenizer"] = tokenizer
            _PARLER_CACHE["desc_tokenizer"] = desc_tokenizer
            _PARLER_CACHE["device"] = device
            log.info("Parler-TTS loaded: %s on %s", type(model).__name__, device)
        except Exception as e:
            log.exception("Failed to load Parler-TTS: %s", e)
            raise
    return _PARLER_CACHE["model"], _PARLER_CACHE["tokenizer"], _PARLER_CACHE["desc_tokenizer"]


def _unload_parler():
    """Explicitly unload Parler to free VRAM.

    gc.collect() before empty_cache() ensures Python destroys the model object
    before PyTorch attempts to reclaim VRAM.
    """
    global _PARLER_CACHE
    import gc
    import torch
    if _PARLER_CACHE["model"] is not None:
        del _PARLER_CACHE["model"]
        del _PARLER_CACHE["tokenizer"]
        del _PARLER_CACHE["desc_tokenizer"]
        _PARLER_CACHE = {"model": None, "tokenizer": None, "desc_tokenizer": None, "device": None}
        gc.collect()
        torch.cuda.empty_cache()
        log.info("Parler unloaded, VRAM freed (gc.collect + empty_cache)")


# Preset voice descriptions for common old-time radio roles
OTR_VOICE_PRESETS = {
    "announcer_male": (
        "A male voice speaking clearly and dramatically with a 1940s transatlantic accent, "
        "warm baritone, confident, slightly formal, as if speaking into a vintage ribbon "
        "microphone in a radio studio. Clean recording, moderate pace."
    ),
    "announcer_female": (
        "A female voice speaking clearly with a 1940s mid-Atlantic accent, "
        "warm mezzo-soprano, poised and professional, the voice of a golden-age "
        "radio hostess. Clean recording, measured pace."
    ),
    "detective_noir": (
        "A male voice speaking in a low, world-weary tone with subtle cynicism, "
        "like a hardboiled detective narrating in a dimly lit office. Deep baritone, "
        "slow and deliberate, with a slight rasp. Mono recording."
    ),
    "scientist_nervous": (
        "A male voice speaking quickly with nervous energy, slightly higher pitch, "
        "educated diction, the breathless excitement of a scientist making a terrible "
        "discovery. Clean recording."
    ),
    "commander_authority": (
        "A male voice speaking with deep authority and gravitas, measured pace, "
        "military precision, like a starship commander giving orders. Deep bass, "
        "resonant, calm under pressure."
    ),
    "ingenue_young": (
        "A young female voice speaking with brightness and vulnerability, "
        "clear soprano, slightly breathless, earnest and questioning. "
        "The ingenue of a radio drama. Clean recording."
    ),
    "villain_sinister": (
        "A male voice speaking slowly with dark amusement, low pitch with "
        "subtle menace, savoring each word. Slightly theatrical, like a radio "
        "villain revealing their plan. Clean recording."
    ),
    "alien_otherworldly": (
        "A voice speaking with an unusual, slightly mechanical cadence, "
        "as if English is not the speaker's native language. Flat affect "
        "with occasional emphasis on unexpected syllables. Medium pitch."
    ),
    "narrator_documentary": (
        "A male voice speaking with warm authority, clear enunciation, "
        "like a 1950s documentary narrator. Rich baritone, measured pace, "
        "educational but engaging. Studio recording quality."
    ),
    "operator_radio": (
        "A female voice speaking with clipped precision, slightly nasal, "
        "like a radio operator or telephone switchboard operator from the 1950s. "
        "Crisp consonants, urgent but controlled."
    ),
}


class ParlerTTSNode:
    """Generate voice audio using Parler-TTS with text-described voice styles."""

    CATEGORY = "OldTimeRadio"
    FUNCTION = "speak"
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "generation_info")

    @classmethod
    def INPUT_TYPES(cls):
        preset_list = list(OTR_VOICE_PRESETS.keys()) + ["custom"]
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "Good evening, ladies and gentlemen. Tonight, Transmission From Tomorrow brings you a tale of science... and of terror.",
                    "tooltip": "The dialogue or narration text to speak"
                }),
                "voice_preset": (preset_list, {
                    "default": "announcer_male",
                    "tooltip": "Preset voice description or 'custom' for your own"
                }),
            },
            "optional": {
                "custom_description": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Custom voice description (used when preset is 'custom')"
                }),
                "model_id": ("STRING", {
                    "default": "parler-tts/parler-tts-large-v1",
                }),
                "max_chunk_length": ("INT", {
                    "default": 300, "min": 50, "max": 600, "step": 50,
                    "tooltip": "Max chars per chunk (Parler handles longer text than Bark)"
                }),
                "temperature": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 2.0, "step": 0.05,
                }),
            },
        }

    def speak(self, text, voice_preset, custom_description="",
              model_id="parler-tts/parler-tts-large-v1",
              max_chunk_length=300, temperature=1.0):

        # Get voice description
        if voice_preset == "custom" and custom_description:
            description = custom_description
        else:
            description = OTR_VOICE_PRESETS.get(voice_preset, OTR_VOICE_PRESETS["announcer_male"])

        # Lazy imports (Section 8)
        import numpy as np
        import torch

        log.info(f"[ParlerTTS] Generating: '{text[:60]}...' preset={voice_preset}")
        model, tokenizer, desc_tokenizer = _load_parler(model_id)

        # Chunk text
        chunks = self._chunk_text(text, max_chunk_length)
        log.info(f"[ParlerTTS] Split into {len(chunks)} chunks")

        all_audio = []
        sample_rate = model.config.sampling_rate  # typically 44100

        for i, chunk in enumerate(chunks):
            # S29: Allow users to cancel long TTS generation
            try:
                import comfy.model_management
                comfy.model_management.throw_exception_if_processing_interrupted()
            except ImportError:
                pass

            log.info(f"[ParlerTTS] Chunk {i+1}/{len(chunks)}")

            # Tokenize and align to model device (Section 5)
            desc_inputs = desc_tokenizer(
                description, return_tensors="pt", padding=True
            ).to(model.device)
            text_inputs = tokenizer(
                chunk, return_tensors="pt", padding=True
            ).to(model.device)

            with torch.no_grad():
                output = model.generate(
                    input_ids=desc_inputs.input_ids,
                    attention_mask=desc_inputs.attention_mask,
                    prompt_input_ids=text_inputs.input_ids,
                    prompt_attention_mask=text_inputs.attention_mask,
                    temperature=temperature,
                    do_sample=True,
                )

            audio_np = output.cpu().numpy().squeeze()
            all_audio.append(audio_np)

        # Concatenate
        combined = np.concatenate(all_audio)

        # ComfyUI AUDIO type contract (Section 26): dict with waveform + sample_rate
        waveform = torch.from_numpy(combined).float().unsqueeze(0).unsqueeze(0)  # (B=1, C=1, T)
        audio_out = {"waveform": waveform, "sample_rate": sample_rate}

        info = json.dumps({
            "voice_preset": voice_preset,
            "description": description[:100],
            "chunks": len(chunks),
            "duration_sec": round(len(combined) / sample_rate, 2),
            "sample_rate": sample_rate,
        })

        log.info(f"[ParlerTTS] Done: {len(combined)/sample_rate:.1f}s @ {sample_rate}Hz")
        return (audio_out, info)

    def _chunk_text(self, text, max_len):
        """Split text at sentence boundaries."""
        if len(text) <= max_len:
            return [text]

        chunks = []
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
