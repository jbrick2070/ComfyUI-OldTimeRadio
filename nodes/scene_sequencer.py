r"""
Scene Sequencer + Episode Assembler - Orchestrate the Full Radio Show
======================================================================

Two nodes:
  1. SceneSequencer - Takes a parsed script JSON and production plan,
     renders each line through the appropriate TTS engine (Bark or Parler),
     inserts SFX/music cues at the right moments, and outputs a scene.
     Features: Intelligent pacing (breath buffers, BEAT/PAUSE tags), continuous
     room tone bed, Gemma Director voice_map dispatch.

  2. EpisodeAssembler - Takes multiple rendered scenes, adds act breaks,
     opening/closing themes, and assembles the complete episode WAV.

These nodes tie together the Gemma 4 Director output with all the
audio generation nodes into a complete pipeline.

v1.0  2026-04-04  Jeffrey Brick
"""

import json
import logging
import math
import os
import re

import numpy as np
import torch

from .story_orchestrator import _runtime_log

log = logging.getLogger("OTR")


def _move_to_device(obj, device):
    """Recursively move tensors and numpy arrays to the target device.

    BarkProcessor returns voice presets as a nested dict ('history_prompt')
    containing numpy arrays for semantic/coarse/fine prompts. A flat
    dict comprehension misses these - this walks the full tree.
    """
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
# LOG CLEANUP - suppress urllib3/httpx cache-check spam from HuggingFace
# -----------------------------------------------------------------------------
logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub.file_download").setLevel(logging.WARNING)


def _trim_trailing_silence(clip_np, threshold=1e-4, min_samples=100):
    """Strip trailing zero-padding from a 1-D float32 numpy array.

    ComfyUI batched AUDIO tensors are zero-padded to uniform length.
    This removes the silent tail so dialogue clips don't insert
    unnatural dead air in the assembled episode.

    Keeps at least `min_samples` to avoid returning an empty array
    for genuinely quiet clips.
    """
    abs_amp = np.abs(clip_np)
    # Find last sample above noise floor
    above = np.where(abs_amp > threshold)[0]
    if len(above) == 0:
        # Entire clip is below threshold - return a tiny slice
        return clip_np[:min_samples] if len(clip_np) > min_samples else clip_np
    last_idx = above[-1]
    # Keep a small tail (50ms at 48kHz - 2400 samples) for natural decay
    tail_pad = min(2400, len(clip_np) - last_idx - 1)
    end = min(last_idx + tail_pad + 1, len(clip_np))
    return clip_np[:end]


def _normalize_clip(clip_np, target_peak=0.85):
    """Normalize a 1-D float32 clip to a target peak amplitude.

    Bark outputs vary wildly in volume between characters and takes.
    This brings every dialogue clip to a consistent level so the Commander
    doesn't whisper while the Pilot screams.

    Uses peak normalization (not RMS) to preserve dynamics within each clip
    while matching overall loudness across clips.
    """
    peak = np.abs(clip_np).max()
    if peak < 1e-6:
        return clip_np  # silence - don't amplify noise floor
    return (clip_np * (target_peak / peak)).astype(np.float32)


def _resample_audio(clip_np, src_rate, dst_rate):
    """Resample a 1-D float32 numpy array.

    Path selection (RTX 5080 optimized):
      - CUDA available + clip > 5s - torchaudio.functional.resample on GPU
        (8-12x faster than scipy for full scenes, sinc-interpolated)
      - Otherwise - scipy.signal.resample_poly (high-quality CPU path)
      - No scipy - np.interp linear fallback
    """
    if src_rate == dst_rate:
        return clip_np.astype(np.float32)

    # GPU fast path for anything longer than ~5 seconds
    try:
        import torch
        import torchaudio
        if torch.cuda.is_available() and len(clip_np) > int(src_rate * 5):
            wav = torch.from_numpy(clip_np).unsqueeze(0).float().cuda()  # [1, T]
            resampled = torchaudio.functional.resample(wav, src_rate, dst_rate)
            log.info("[SceneSequencer] Resample %dHz-%dHz: GPU torchaudio (%d samples)",
                     src_rate, dst_rate, len(clip_np))
            return resampled.squeeze(0).cpu().numpy().astype(np.float32)
    except ImportError:
        pass  # fall through to CPU paths

    # CPU path: scipy polyphase (high quality, anti-aliased)
    g = math.gcd(int(dst_rate), int(src_rate))
    up = int(dst_rate) // g
    down = int(src_rate) // g

    try:
        from scipy.signal import resample_poly
        return resample_poly(clip_np, up, down).astype(np.float32)
    except ImportError:
        log.warning("[SceneSequencer] scipy not available - falling back to linear "
                    "interpolation for resampling. Install scipy for proper anti-aliasing.")
        new_len = int(len(clip_np) * dst_rate / src_rate)
        return np.interp(
            np.linspace(0, len(clip_np) - 1, new_len),
            np.arange(len(clip_np)),
            clip_np
        ).astype(np.float32)

DEFAULT_OUT = os.path.join(os.path.expanduser("~"), "Documents", "ComfyUI", "output", "otr", "audio")


# -----------------------------------------------------------------------------
# ROOM TONE BED - continuous background that fills silence between dialogue
# -----------------------------------------------------------------------------

def _generate_room_tone(duration_sec, sample_rate=48000, intensity=0.03, descriptors=""):
    """Generate a dynamic background bed based on Canonical 1.0 ENV descriptors.

    Uses the descriptors (e.g. 'night city street, distant traffic') to skew
    the noise profile and add textures like wind, sirens, or electronic hums.

    Path selection (RTX 5080 optimized):
      - CUDA + duration > 60s - GPU torch path (noise + sin on Tensor Cores)
      - Otherwise - CPU numpy path (low overhead for short beds)
    """
    import torch
    n_samples = int(duration_sec * sample_rate)
    desc = descriptors.lower()
    _use_gpu = (torch.cuda.is_available() and duration_sec >= 60)

    if _use_gpu:
        # -- GPU path: all noise + trig on CUDA ------------------------------
        dev = torch.device("cuda")
        t = torch.arange(n_samples, dtype=torch.float32, device=dev) / sample_rate

        # Base: tape hiss
        hiss = torch.randn(n_samples, dtype=torch.float32, device=dev)
        hiss_cutoff = 800 if ("wind" in desc or "storm" in desc) else 4000
        hiss_intensity = intensity * 1.5 if ("wind" in desc or "storm" in desc) else intensity
        # FFT bandpass on GPU (replaces scipy.sosfilt)
        freqs = torch.fft.rfftfreq(n_samples, d=1.0 / sample_rate, device=dev)
        mask = ((freqs >= 100) & (freqs <= hiss_cutoff)).float()
        hiss = torch.fft.irfft(torch.fft.rfft(hiss) * mask, n=n_samples)
        hiss *= hiss_intensity * 0.6

        # Mains hum
        hum_freq = 50 if "euro" in desc else 60
        hum_amp = intensity * 0.15 if ("electronic" in desc or "fluorescent" in desc or "ship" in desc) else intensity * 0.1
        hum = torch.sin(2 * math.pi * hum_freq * t) * hum_amp

        # Textures
        texture = torch.zeros(n_samples, dtype=torch.float32, device=dev)
        if "traffic" in desc or "street" in desc:
            texture += torch.sin(2 * math.pi * 30 * t) * (intensity * 0.2)
        if "siren" in desc:
            siren_mod = torch.sin(2 * math.pi * 0.2 * t) * 100 + 400
            texture += torch.sin(2 * math.pi * siren_mod * t) * (intensity * 0.05)

        # Sporadic crackle (stays on CPU - tiny loop, negligible cost)
        crackle = np.zeros(n_samples, dtype=np.float32)
        n_pops = int(duration_sec * (8 if "vinyl" in desc else 3))
        pop_positions = np.random.randint(0, n_samples, size=n_pops)
        for pos in pop_positions:
            p_len = np.random.randint(int(sample_rate * 0.001), int(sample_rate * 0.004))
            end = min(pos + p_len, n_samples)
            crackle[pos:end] += np.linspace(1.0, 0, end - pos) * intensity * 0.4
        crackle_t = torch.from_numpy(crackle).to(dev, non_blocking=True)

        result = hiss + hum + texture + crackle_t
        log.info("[SceneSequencer] Room tone: GPU path (%.1fs, %d samples)", duration_sec, n_samples)
        return result.cpu().numpy()

    # -- CPU path: numpy (low overhead for short beds) -----------------------
    hiss = np.random.randn(n_samples).astype(np.float32)
    if "wind" in desc or "storm" in desc:
        cutoff = 800
        intensity *= 1.5
    else:
        cutoff = 4000

    try:
        from scipy.signal import butter, sosfilt
        sos = butter(4, [100, cutoff], btype='bandpass', fs=sample_rate, output='sos')
        hiss = sosfilt(sos, hiss).astype(np.float32)
    except Exception:
        pass
    hiss *= intensity * 0.6

    # Mains Hum
    hum_freq = 50 if "euro" in desc else 60
    hum_amp = intensity * 0.3 if ("electronic" in desc or "fluorescent" in desc or "ship" in desc) else intensity * 0.1
    t = np.arange(n_samples, dtype=np.float32) / sample_rate
    hum = np.sin(2 * np.pi * hum_freq * t) * hum_amp

    # Textures
    texture = np.zeros(n_samples, dtype=np.float32)
    if "traffic" in desc or "street" in desc:
        texture += np.sin(2 * np.pi * 30 * t) * (intensity * 0.2)
    if "siren" in desc:
        siren_mod = np.sin(2 * np.pi * 0.2 * t) * 100 + 400
        texture += np.sin(2 * np.pi * siren_mod * t) * (intensity * 0.05)

    # Sporadic crackle
    crackle = np.zeros(n_samples, dtype=np.float32)
    n_pops = int(duration_sec * (8 if "vinyl" in desc else 3))
    pop_positions = np.random.randint(0, n_samples, size=n_pops)
    for pos in pop_positions:
        p_len = np.random.randint(int(sample_rate * 0.001), int(sample_rate * 0.004))
        end = min(pos + p_len, n_samples)
        crackle[pos:end] += np.linspace(1.0, 0, end - pos) * intensity * 0.4

    return hiss + hum + texture + crackle


# -----------------------------------------------------------------------------
# INLINE BARK TTS - called by SceneSequencer for dynamic dialogue generation
# -----------------------------------------------------------------------------

# Default voice preset rotation for characters without explicit assignments
_BARK_VOICE_PRESETS = [
    # -- English (native) --
    "v2/en_speaker_0",  # Male, deep, authoritative (announcer)
    "v2/en_speaker_1",  # Male, warm, conversational
    "v2/en_speaker_2",  # Male, calm, measured (sounds male/neutral in practice)
    "v2/en_speaker_3",  # Male, young, energetic
    "v2/en_speaker_4",  # Female, warm, expressive
    "v2/en_speaker_5",  # Male, older, gravelly
    "v2/en_speaker_6",  # Male, neutral, broadcast
    "v2/en_speaker_7",  # Male, sharp, anxious (androgynous but reads male)
    "v2/en_speaker_8",  # Male, deep, dramatic
    "v2/en_speaker_9",  # Female, mature, sophisticated
    # -- International accented English --
    # European presets render English clearly with accent flavor.
    # Adds vocal diversity without sacrificing intelligibility.
    "v2/de_speaker_0",  # German male, precise, clipped
    "v2/de_speaker_4",  # German female, clear, analytical
    "v2/fr_speaker_0",  # French male, smooth, baritone
    "v2/fr_speaker_4",  # French female, warm, elegant
    "v2/es_speaker_0",  # Spanish male, warm, authoritative
    "v2/es_speaker_9",  # Spanish female, mature, expressive
    "v2/it_speaker_0",  # Italian male, dramatic, animated
    "v2/it_speaker_4",  # Italian female, expressive, warm
    "v2/pt_speaker_0",  # Portuguese male, soft, thoughtful
    "v2/pt_speaker_4",  # Portuguese female, gentle, clear
]

_FEMALE_PRESETS = [
    # en_speaker_2 and en_speaker_7 removed - sound male/androgynous in practice
    "v2/en_speaker_4", "v2/en_speaker_9",
    "v2/de_speaker_4", "v2/fr_speaker_4", "v2/es_speaker_9",
    "v2/it_speaker_4", "v2/pt_speaker_4",
]
_MALE_PRESETS = [
    "v2/en_speaker_0", "v2/en_speaker_1", "v2/en_speaker_3",
    "v2/en_speaker_5", "v2/en_speaker_6", "v2/en_speaker_8",
    "v2/de_speaker_0", "v2/fr_speaker_0", "v2/es_speaker_0",
    "v2/it_speaker_0", "v2/pt_speaker_0",
]

# Stable character-preset cache so the same character always gets the same voice
_CHARACTER_VOICE_CACHE = {}


def _voice_preset_for_character(voice_tag, voice_map, voice_traits=""):
    """Determine Bark voice preset for a character/voice_tag.

    Priority:
      1. Cached assignment (stable across the episode)
      2. Director's voice_assignments (from LLMDirector voice_map_json)
      3. Gender-aware hash fallback using voice_traits from script
    """
    if voice_tag in _CHARACTER_VOICE_CACHE:
        return _CHARACTER_VOICE_CACHE[voice_tag]

    # Direct match from Director's voice map (Director maps Tag -> Preset)
    voice_info = voice_map.get(voice_tag, {})
    preset = voice_info.get("voice_preset") or voice_info.get("bark_preset")
    if preset and preset.startswith("v2/"):
        _CHARACTER_VOICE_CACHE[voice_tag] = preset
        return preset

    # Gender-aware hash fallback with 93/7 English-native/international ratio.
    # ~93% chance of English native, ~7% of international accented English.
    import random as _rng_mod
    traits_lower = voice_traits.lower() if voice_traits else ""
    is_female = "female" in traits_lower or "woman" in traits_lower or "girl" in traits_lower
    is_male   = "male" in traits_lower or "man" in traits_lower or "boy" in traits_lower

    # Deterministic seed per voice_tag so same character always gets same voice
    rng = _rng_mod.Random(hash(voice_tag))
    use_intl = rng.random() < 0.07  # 7% chance of international preset

    if is_female:
        en_pool   = [p for p in _FEMALE_PRESETS if p.startswith("v2/en_")]
        intl_pool = [p for p in _FEMALE_PRESETS if not p.startswith("v2/en_")]
        label = "female"
    elif is_male:
        en_pool   = [p for p in _MALE_PRESETS if p.startswith("v2/en_")]
        intl_pool = [p for p in _MALE_PRESETS if not p.startswith("v2/en_")]
        label = "male"
    else:
        en_pool   = [p for p in _BARK_VOICE_PRESETS if p.startswith("v2/en_")]
        intl_pool = [p for p in _BARK_VOICE_PRESETS if not p.startswith("v2/en_")]
        label = "unknown-gender"

    pool = intl_pool if (use_intl and intl_pool) else en_pool
    if not pool:
        pool = _BARK_VOICE_PRESETS
    preset = rng.choice(pool)
    _CHARACTER_VOICE_CACHE[voice_tag] = preset
    pool_tag = "international" if (use_intl and intl_pool) else "English-native"
    log.info("[VoiceMap] No Director mapping for '%s' (%s), assigned %s from %s %s pool",
             voice_tag, traits_lower[:30], preset, pool_tag, label)
    return preset


def _clean_text_for_bark(text):
    """Clean and normalize dialogue text for Bark TTS.

    Bark accepts a specific set of non-speech tokens in square brackets.
    This function:
      1. Strips structural tags that must never reach Bark ([VOICE:], [ENV:],
         [SFX:], [MUSIC:], === scene headers ===)
      2. Converts common parenthetical stage directions to Bark token equivalents
      3. Converts asterisk actions (*laughs*) to Bark tokens
      4. Preserves - music notation (Bark renders humming/singing)
      5. Preserves valid Bark non-verbal tokens already in the text
      6. Strips any remaining unrecognized square-bracket tags
      7. Collapses whitespace

    Bark's full supported token set (suno/bark v1):
      [laughter]      sustained laughter
      [laughs]        brief laugh
      [sighs]         audible sigh
      [music]         musical interlude / humming
      [gasps]         sharp gasp
      [clears throat] throat clear before speaking
      [coughs]        cough
      [pants]         breathless panting (exertion)
      [sobs]          crying/sobbing
      [grunts]        effort grunt
      [groans]        pain or frustration groan
      [whistles]      whistle
      [sneezes]       sneeze
    - text -         sung / hummed phrase

    Tokens NOT supported by Bark (spoken as literal words - must be stripped):
      [whispers] [shouts] [nervous laugh] etc.
    """
    import re

    # -- Step 1: Strip structural / non-Bark tags -----------------------------
    text = re.sub(r'\[VOICE:[^\]]*\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[(?:ENV|SFX|MUSIC):[^\]]*\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'===.*?===', '', text)

    # -- Step 2: Drop ALL parenthetical stage directions ----------------------
    # BUG-LOCAL-101 (2026-04-28 PM): mirrors the change in
    # batch_bark_generator._clean_text_for_bark. The two functions must stay
    # behaviorally identical (test_scene_sequencer_clean_matches_batcher).
    # Parens get dropped wholesale rather than translated to Bark non-verbal
    # tokens, because rendered tokens add unwanted breath/throat audio
    # leading into dialogue (Stellar Shadows 2026-04-28: l004 "(panting)"
    # produced "Let's work I guess..." in Whisper transcription).
    text = re.sub(r'\([^)]{1,80}\)\s*', '', text)

    # -- Step 3: Asterisk actions - Bark tokens -------------------------------
    _ASTERISK_TO_BARK = [
        ("laugh",   "[laughs]"),
        ("chuckl",  "[laughs]"),
        ("sigh",    "[sighs]"),
        ("gasp",    "[gasps]"),
        ("groan",   "[groans]"),
        ("sob",     "[sobs]"),
        ("cough",   "[coughs]"),
        ("grunt",   "[grunts]"),
    ]
    def _translate_asterisk(m):
        inner = m.group(1).lower().strip()
        for stem, token in _ASTERISK_TO_BARK:
            if stem in inner:
                return token + " "
        return ""

    text = re.sub(r'\*([^*]{1,60})\*', _translate_asterisk, text)

    # -- Step 4: Strip unrecognized bracket tags -------------------------------
    _BARK_VALID_TOKENS = {
        "[laughter]", "[laughs]", "[sighs]", "[music]", "[gasps]",
        "[clears throat]", "[coughs]", "[pants]", "[sobs]", "[grunts]",
        "[groans]", "[whistles]", "[sneezes]",
    }
    def _filter_bracket_tag(m):
        inner = m.group(0)[1:-1].strip().lower()
        inner = re.sub(r'\s+', ' ', inner)
        tag = f"[{inner}]"
        return tag if tag in _BARK_VALID_TOKENS else ""

    text = re.sub(r'\[[^\]]{1,40}\]', _filter_bracket_tag, text)

    # -- Step 5: Normalize whitespace -----------------------------------------
    text = re.sub(r'  +', ' ', text).strip()
    return text


def _chunk_text_for_bark(text, max_len=180):
    """Split text into Bark-friendly chunks at sentence boundaries."""
    import re
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


def _generate_bark_for_line(text, voice_preset, temperature=0.7):
    """Generate TTS audio for a single dialogue line using Bark.

    Returns (audio_np_1d, sample_rate).  Handles chunking internally.
    Reuses the bark_tts module's model cache so the model loads once.
    """
    import torch

    # Clean the text: strip parenthetical directions, keep Bark-compatible tags
    text = _clean_text_for_bark(text)
    if not text:
        # Nothing left after cleaning - return tiny silence
        return np.zeros(2400, dtype=np.float32), 24000

    # Import the shared Bark loader from our bark_tts module
    from .bark_tts import _load_bark

    model, processor = _load_bark("suno/bark")
    sample_rate = model.generation_config.sample_rate  # 24000

    chunks = _chunk_text_for_bark(text)
    all_audio = []
    silence_pad = np.zeros(int(sample_rate * 0.08), dtype=np.float32)  # 80ms gap

    for chunk in chunks:
        inputs = processor(chunk, voice_preset=voice_preset)
        # Recursively move ALL processor outputs to CUDA - including the
        # nested 'history_prompt' dict with voice preset numpy arrays.
        inputs = _move_to_device(inputs, torch.device("cuda"))

        if "attention_mask" not in inputs and "input_ids" in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

        assert inputs["input_ids"].device.type == "cuda", "input_ids not on CUDA before generate"

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
                    temperature=temperature,
                )
        finally:
            torch.tensor = _orig_tensor
            torch.arange = _orig_arange

        audio_np = output.cpu().numpy().squeeze()
        all_audio.append(audio_np)
        if len(chunks) > 1:
            all_audio.append(silence_pad)

    return np.concatenate(all_audio), sample_rate


class SceneSequencer:
    """Render a script scene: TTS for each line, SFX cues, pauses."""

    CATEGORY = "OldTimeRadio"
    FUNCTION = "sequence"
    RETURN_TYPES = ("AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("scene_audio", "render_log", "scene_manifest_json")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "script_json": ("STRING", {
                    "multiline": True,
                    "default": "[]",
                    "tooltip": "Parsed script JSON from LLMScriptWriter"
                }),
                "production_plan_json": ("STRING", {
                    "multiline": True,
                    "default": "{}",
                    "tooltip": "Production plan JSON from LLMDirector"
                }),
            },
            "optional": {
                "tts_audio_clips": ("AUDIO", {
                    "tooltip": "Pre-rendered TTS audio clips (from Bark/Parler batch). "
                               "If provided, dialogue lines use these clips instead of "
                               "placeholder silence. Clips are matched to dialogue lines "
                               "in order. ANNOUNCER lines are NOT expected here - they "
                               "flow through announcer_audio_clips on a separate bus."
                }),
                "announcer_audio_clips": ("AUDIO", {
                    "tooltip": "Pre-rendered ANNOUNCER audio clips from KokoroAnnouncer. "
                               "Consumed in script order for dialogue lines whose "
                               "character_name is ANNOUNCER. Keeps the Voice of God "
                               "bookends separated from the Bark character pool."
                }),
                "sfx_audio_clips": ("AUDIO", {
                    "tooltip": "Pre-rendered SFX audio clips (from SFXGenerator batch). "
                               "Matched to [SFX:] cues in script order."
                }),
                "start_line": ("INT", {
                    "default": 0, "min": 0, "max": 9999,
                    "tooltip": "First line to render (for chunked processing)"
                }),
                "end_line": ("INT", {
                    "default": 999, "min": 1, "max": 9999,
                    "tooltip": "Last line to render"
                }),
                "output_dir": ("STRING", {"default": ""}),
                "default_tts": (["bark", "parler", "kokoro"], {
                    "default": "bark",
                    "tooltip": "Default TTS engine when not specified in production plan"
                }),
                # v1.5 Phase 3: Time-Alignment Offset Pins
                "dialogue_offset_ms": ("FLOAT", {
                    "default": 0.0, "min": -500.0, "max": 500.0, "step": 10.0,
                    "tooltip": "Shift all dialogue clips on the timeline (ms). "
                               "Positive = delay, negative = advance."
                }),
                "sfx_offset_ms": ("FLOAT", {
                    "default": 0.0, "min": -500.0, "max": 500.0, "step": 10.0,
                    "tooltip": "Shift all SFX clips on the timeline (ms). "
                               "Positive = delay, negative = advance."
                }),
            },
        }

    def _extract_clips_from_audio(self, audio_input):
        """Extract individual clips from a batched AUDIO input.

        If the AUDIO has batch dim > 1, each batch element is a separate clip.
        If batch dim == 1, it's a single long clip that we return as-is.
        """
        if audio_input is None:
            return []

        if isinstance(audio_input, dict):
            waveform = audio_input.get("waveform")
            sr = audio_input.get("sample_rate", 48000)
        else:
            waveform = audio_input
            sr = 48000

        if waveform is None:
            return []

        # Ensure 3D: (B, C, T)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)
        elif waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)

        # Return list of (waveform_np, sample_rate) per batch element.
        # Strip trailing zero-padding from batched tensors (ComfyUI pads
        # shorter clips to match the longest in the batch).
        clips = []
        for b in range(waveform.shape[0]):
            clip_np = waveform[b].cpu().numpy().squeeze()
            clip_np = _trim_trailing_silence(clip_np)
            if len(clip_np) > 0:
                clips.append((clip_np, sr))

        return clips

    def sequence(self, script_json, production_plan_json,
                 tts_audio_clips=None, sfx_audio_clips=None,
                 announcer_audio_clips=None,
                 start_line=0, end_line=999, output_dir=DEFAULT_OUT,
                 default_tts="bark",
                 dialogue_offset_ms=0.0, sfx_offset_ms=0.0):

        _runtime_log("SceneSequencer: Starting 1.0 audio assembly...")
        # Schema l3 (2026-04-28): track wall-clock for meta.phase_ms.scene_sequencer.
        import time as _time
        _phase_t0 = _time.time()
        script = json.loads(script_json) if isinstance(script_json, str) else script_json
        plan = json.loads(production_plan_json) if isinstance(production_plan_json, str) else production_plan_json

        voice_map = plan.get("voice_assignments", {})
        pacing = plan.get("pacing", {})
        # PACING: Breath buffer + dramatic pauses (v1.4 - 50% duration reduction)
        breath_ms = pacing.get("breath_pause_ms", 200)          # between every dialogue line
        beat_pause_ms = pacing.get("beat_pause_ms", 750)        # [BEAT] tag - dramatic beat
        pause_ms = pacing.get("pause_ms", 1000)                 # [PAUSE] tag - longer pause
        scene_transition_ms = pacing.get("scene_transition_ms", 1250)
        act_break_ms = pacing.get("act_break_ms", 2500)

        # Guard: fall back to DEFAULT_OUT if output_dir is empty/None
        if not output_dir or not output_dir.strip():
            output_dir = DEFAULT_OUT
        os.makedirs(output_dir, exist_ok=True)

        # Free LLM VRAM before TTS generation - Bark needs GPU headroom.
        # LLM is done by this point (script + plan already generated).
        try:
            from .story_orchestrator import _unload_llm
            _unload_llm()
            log.info("[SceneSequencer] Freed LLM VRAM for inline TTS")
        except Exception:
            pass  # LLM may already be unloaded or not imported

        # Extract pre-rendered clips from batched AUDIO inputs
        tts_clips = self._extract_clips_from_audio(tts_audio_clips)
        sfx_clips = self._extract_clips_from_audio(sfx_audio_clips)
        announcer_clips = self._extract_clips_from_audio(announcer_audio_clips)
        tts_clip_idx = 0
        sfx_clip_idx = 0
        announcer_clip_idx = 0
        log.info(
            "[SceneSequencer] Pre-rendered clips: %d TTS, %d SFX, %d ANNOUNCER",
            len(tts_clips), len(sfx_clips), len(announcer_clips),
        )

        # We accumulate silence/audio segments as numpy arrays
        sample_rate = 48000  # standardize output
        all_segments = []
        sfx_timeline = []
        render_log = []
        manifest = []
        
        # Canonical 1.0+ state tracking
        current_character_name = None
        current_env = "silent room"
        env_timeline = []  # List of (start_sample, end_sample, desc)

        # v1.5 Phase 3: Convert TA offset from ms to samples
        dialogue_offset_samples = int(dialogue_offset_ms * sample_rate / 1000.0)
        sfx_offset_samples = int(sfx_offset_ms * sample_rate / 1000.0)
        if dialogue_offset_ms != 0.0 or sfx_offset_ms != 0.0:
            _runtime_log(f"SceneSequencer: TA_Offset active: dialogue={dialogue_offset_ms:+.0f}ms, sfx={sfx_offset_ms:+.0f}ms")

        lines_to_render = script[start_line:end_line]
        log.info(f"[SceneSequencer] Rendering Canonical 1.0 items {start_line}-{min(end_line, len(script))}")

        current_sample_pos = 0

        # BUG-LOCAL-106 (2026-04-29): track per-dialogue-line positions
        # in scene_audio space so we can write authoritative start_s /
        # dur_s back to ledger.lines[] at the end of this method. The
        # BUG-094 word-share estimator was placing HuMo clips at
        # cumulative-bark-only positions which DON'T account for the
        # breath/beat/SFX gaps SceneSequencer inserts here -- net
        # result was lip-sync drift growing line-by-line and HuMo
        # rendering motion during music intros (deep_earth_echoes
        # 2026-04-28 ledger showed l001 placed at 0.0s but the actual
        # announcer audio in the master started at 0:09).
        # We record positions in SCENE-AUDIO space (no opening theme).
        # EpisodeAssembler shifts these to MASTER-MIX space when it
        # prepends the opening_theme.
        dialogue_positions: list[dict] = []

        for i, item in enumerate(lines_to_render):
            item_type = item.get("type", "dialogue")
            global_idx = start_line + i
            
            # S29: Interrupt check
            if i % 10 == 0:
                try:
                    import comfy.model_management
                    comfy.model_management.throw_exception_if_processing_interrupted()
                except ImportError:
                    pass

            segment_np = None
            
            # -- CANONICAL 1.0 TOKENS --------------------------------------
            
            if item_type == "environment":
                current_env = item.get("description", "default room")
                render_log.append(f"[{global_idx}] ENV: {current_env}")
                continue

            elif item_type == "scene_break":
                segment_np = np.zeros(int(sample_rate * 0.5), dtype=np.float32)
                render_log.append(f"[{global_idx}] === SCENE {item.get('scene', '?')} ===")

            elif item_type == "pause":
                # Default to beat_pause_ms if duration not in script.
                # Remove the legacy 200ms cap (confused with beat_pause_ms name).
                dur_ms = item.get("duration_ms", beat_pause_ms)
                segment_np = np.zeros(int(sample_rate * dur_ms / 1000), dtype=np.float32)
                render_log.append(f"[{global_idx}] (beat) {dur_ms}ms")

            elif item_type == "sfx":
                desc = item.get("description", "unknown sound")
                if sfx_clip_idx < len(sfx_clips):
                    clip_np, clip_sr = sfx_clips[sfx_clip_idx]
                    sfx_segment = _resample_audio(clip_np, clip_sr, sample_rate)
                    sfx_segment = _normalize_clip(sfx_segment, target_peak=0.85)
                    sfx_clip_idx += 1
                    # v1.5: Apply SFX TA_Offset
                    sfx_pos = max(0, current_sample_pos + sfx_offset_samples)
                    sfx_timeline.append((sfx_pos, sfx_segment, desc))
                    render_log.append(f"[{global_idx}] SFX Overlay: {desc}")
                else:
                    render_log.append(f"[{global_idx}] SFX: {desc} (MISSING)")
                    
                # Add a tiny 0.1s breath to dialogue timeline to give SFX impact room
                segment_np = np.zeros(int(sample_rate * 0.1), dtype=np.float32)

            elif item_type == "dialogue":
                character_name = item.get("character_name", "UNKNOWN")
                voice_traits = item.get("voice_traits", "")
                line = item.get("line", "")
                preset = _voice_preset_for_character(character_name, voice_map, voice_traits)

                is_announcer = character_name.strip().upper() == "ANNOUNCER"

                if is_announcer and announcer_clip_idx < len(announcer_clips):
                    # Dedicated Kokoro announcer bus - clean, no Bark filler sounds.
                    clip_np, clip_sr = announcer_clips[announcer_clip_idx]
                    segment_np = _resample_audio(clip_np, clip_sr, sample_rate)
                    segment_np = _normalize_clip(segment_np)
                    announcer_clip_idx += 1
                    render_log.append(f"[{global_idx}] ANNOUNCER (Kokoro): {line[:40]}...")
                elif tts_clip_idx < len(tts_clips):
                    clip_np, clip_sr = tts_clips[tts_clip_idx]
                    segment_np = _resample_audio(clip_np, clip_sr, sample_rate)
                    segment_np = _normalize_clip(segment_np)
                    tts_clip_idx += 1
                    render_log.append(f"[{global_idx}] {character_name}: {line[:40]}...")
                else:
                    log.info(f"[SceneSequencer] Inline Bark [{global_idx}] {character_name}")
                    bark_np, bark_sr = _generate_bark_for_line(line, preset)
                    segment_np = _resample_audio(bark_np, bark_sr, sample_rate)
                    segment_np = _normalize_clip(segment_np)
                    render_log.append(f"[{global_idx}] {character_name}: {line[:40]}...")

                current_character_name = character_name

            elif item_type == "direction":
                render_log.append(f"[{global_idx}] DIRECTION: {item.get('text', '')[:50]}")

            # -- Accumulate Audio and Track Environment Span --------------
            if segment_np is not None:
                # v1.5: Apply dialogue TA_Offset for dialogue/announcer items
                if item_type == "dialogue" and dialogue_offset_samples != 0:
                    offset_silence = max(0, dialogue_offset_samples)
                    if offset_silence > 0:
                        segment_np = np.concatenate([
                            np.zeros(offset_silence, dtype=np.float32),
                            segment_np
                        ])
                seg_len = len(segment_np)
                env_timeline.append((current_sample_pos, current_sample_pos + seg_len, current_env))
                all_segments.append(segment_np)
                # BUG-LOCAL-106: capture authoritative scene-audio
                # position for every dialogue line so the ledger
                # write-back below can give BatchHumoRender real
                # placements instead of the BUG-094 word-share
                # estimator's wrong-when-music-or-pauses-exist guess.
                if item_type == "dialogue":
                    # ROADMAP P0 step 4b (2026-04-30): tag the line's
                    # speaker_role so the ledger write-back can stamp
                    # it onto ledger.lines[].  Announcer dialogue
                    # routes BatchHumoRender to the radio still I2V
                    # ref; everything else uses the cast portrait.
                    _ch_upper = item.get("character_name", "").strip().upper()
                    _role = "announcer" if _ch_upper == "ANNOUNCER" else "character"
                    dialogue_positions.append({
                        "text": (item.get("line") or "").strip(),
                        "speaker": item.get("character_name", "UNKNOWN"),
                        "speaker_role": _role,
                        "start_s": float(current_sample_pos) / float(sample_rate),
                        "dur_s": float(seg_len) / float(sample_rate),
                    })
                current_sample_pos += seg_len

        # Log clip usage stats
        render_log.append(f"--- Audio units assembled: {len(all_segments)}")

        # Concatenate all dialogue/SFX segments
        if all_segments:
            combined = np.concatenate(all_segments)
        else:
            combined = np.zeros(int(sample_rate * 1), dtype=np.float32)

        # -- CANONICAL 1.0 ENVIRONMENT MIXING --------------------------
        total_len = len(combined)
        final_bed = np.zeros(total_len, dtype=np.float32)
        room_intensity = plan.get("vintage_settings", {}).get("room_tone_intensity", 0.01)
        
        for start, end, desc in env_timeline:
            span_len_sec = (end - start) / sample_rate
            # Generate a specialized texture for this description
            bed_segment = _generate_room_tone(span_len_sec, sample_rate, intensity=room_intensity, descriptors=desc)
            fit_len = min(len(bed_segment), end - start)
            final_bed[start : start + fit_len] += bed_segment[:fit_len]
            
        combined = combined + final_bed
        render_log.append(f"--- Layered {len(env_timeline)} environment segments")

        # -- SFX DUCKING & OVERLAY ------------------------------------------
        max_sfx_end = 0
        for start_pos, sfx_np, _ in sfx_timeline:
            end_pos = start_pos + len(sfx_np)
            if end_pos > max_sfx_end:
                max_sfx_end = end_pos
                
        if max_sfx_end > len(combined):
            pad = np.zeros(max_sfx_end - len(combined), dtype=np.float32)
            combined = np.concatenate([combined, pad])
            
        if sfx_timeline:
            render_log.append(f"--- Overlaying {len(sfx_timeline)} SFX cues with ducking")
            for start_pos, sfx_np, desc in sfx_timeline:
                end_pos = start_pos + len(sfx_np)
                # Duck main mix (dialogue+bed) down to 70% underneath SFX
                combined[start_pos:end_pos] *= 0.7
                # Mix in SFX at 85% to prevent clipping
                combined[start_pos:end_pos] += sfx_np * 0.85

        total_len = len(combined)
        total_sec = total_len / sample_rate
        _runtime_log(f"SceneSequencer: 1.0 Mix complete ({total_sec:.1f}s)")

        waveform = torch.from_numpy(combined).float().unsqueeze(0).unsqueeze(0)
        audio_out = {"waveform": waveform, "sample_rate": sample_rate}
        log_text = "\n".join(render_log)
        manifest_json = json.dumps(manifest, indent=2)

        # Schema l3 ledger write-back: phase_ms.scene_sequencer +
        # audio_gates "post_scene_sequencer" sha256-of-leading-1KB.
        # Best-effort: any failure logs WARNING but never aborts the
        # render. Audio integrity gate per CLAUDE.md C7 -- if two
        # consecutive runs disagree on this hash after a no-op change
        # we know audio drift slipped in.
        try:
            from . import _otr_ledger as _OTRL  # type: ignore
            from ._otr_paths import otr_audio_dir, otr_legacy_audio_dir
            _phase_ms = int((_time.time() - _phase_t0) * 1000)
            _ledger_p = _OTRL.find_most_recent_ledger(
                [otr_audio_dir(), otr_legacy_audio_dir()]
            )
            if _ledger_p is not None:
                _led = _OTRL.load_ledger_safe(_ledger_p)
                if _led is not None:
                    _OTRL.record_phase_ms(_led, "scene_sequencer", _phase_ms)
                    _wb = combined.tobytes()[: _OTRL.GATE_HASH_BYTES]
                    _gate = _OTRL.audio_gate_record(
                        gate_name="post_scene_sequencer",
                        waveform_bytes=_wb,
                        dur_s=float(total_sec),
                        sample_count=int(total_len),
                        sample_rate=int(sample_rate),
                    )
                    _OTRL.append_audio_gate(_led, _gate)

                    # BUG-LOCAL-106: write authoritative scene-audio
                    # positions back to ledger.lines[]. Match by
                    # text (same strategy as BatchBark BUG-096) so
                    # the order of dialogue_positions vs ledger.lines
                    # is robust to ANNOUNCER routing / SFX gaps.
                    # Positions are in SCENE-AUDIO space here;
                    # EpisodeAssembler shifts them to MASTER-MIX
                    # space when it prepends the opening_theme.
                    _ledger_lines = _led.get("lines") or []
                    _text_to_idx: dict[str, list[int]] = {}
                    for _li, _ln in enumerate(_ledger_lines):
                        _t = (_ln.get("text") or "").strip()
                        if _t:
                            _text_to_idx.setdefault(_t, []).append(_li)
                    _matched = 0
                    for _pos in dialogue_positions:
                        _cands = _text_to_idx.get(_pos["text"]) or []
                        if not _cands:
                            continue
                        _ledger_idx = _cands.pop(0)
                        _row = _ledger_lines[_ledger_idx]
                        _row["start_s"] = float(_pos["start_s"])
                        _row["dur_s"] = float(_pos["dur_s"])
                        # Mark these positions as scene-audio-relative
                        # so EpisodeAssembler knows whether to shift.
                        _row["start_s_space"] = "scene_audio"
                        # ROADMAP P0 step 4b (2026-04-30): stamp
                        # speaker_role so BatchHumoRender's
                        # ref-image swap branches on the right value.
                        # Default to "character" if dialogue_positions
                        # somehow didn't supply a role (defensive).
                        _row["speaker_role"] = (
                            _pos.get("speaker_role") or "character"
                        )
                        _matched += 1

                    # BUG-LOCAL-107 (authoritative-log expansion):
                    # write-back SFX cue placements to ledger.sfx[].
                    # SceneSequencer's sfx_timeline holds (start_sample,
                    # sfx_np, desc) tuples in script order; ledger.sfx[]
                    # is also in script order so we walk both in
                    # parallel. Positions are in SCENE-AUDIO space;
                    # EpisodeAssembler shifts them to MASTER-MIX
                    # alongside lines[] / clips[].
                    _ledger_sfx = _led.get("sfx") or []
                    _sfx_matched = 0
                    # ROADMAP P0 step 4b (2026-04-30): also collect
                    # SFX entries in a parallel list for mirroring
                    # into ledger.lines[] below.  This is what gives
                    # BatchHumoRender wall-to-wall iteration coverage
                    # without it having to walk a separate ledger
                    # array.  ledger.sfx[] stays populated for any
                    # back-compat consumers.
                    _sfx_to_mirror_into_lines = []
                    for _sfx_idx, (_sfx_pos, _sfx_np, _sfx_desc) in enumerate(sfx_timeline):
                        if _sfx_idx >= len(_ledger_sfx):
                            break
                        _sfx_row = _ledger_sfx[_sfx_idx]
                        _sfx_row["start_s"] = float(_sfx_pos) / float(sample_rate)
                        _sfx_row["dur_s"] = float(len(_sfx_np)) / float(sample_rate)
                        _sfx_row["start_s_space"] = "scene_audio"
                        # Capture the description SceneSequencer
                        # actually placed (for diagnostic clarity).
                        if _sfx_desc and not _sfx_row.get("description"):
                            _sfx_row["description"] = str(_sfx_desc)
                        _sfx_matched += 1
                        # Mirror entry candidate.
                        _sfx_to_mirror_into_lines.append({
                            "line_id": _sfx_row.get("sfx_id") or _sfx_row.get("id") or f"sfx_{_sfx_idx:03d}",
                            "speaker": "SFX",
                            "speaker_role": "sfx",
                            "text": str(_sfx_desc) if _sfx_desc else "",
                            "start_s": float(_sfx_pos) / float(sample_rate),
                            "dur_s": float(len(_sfx_np)) / float(sample_rate),
                            "start_s_space": "scene_audio",
                            "shot_id": _sfx_row.get("shot_id"),
                        })

                    # ROADMAP P0 step 4b: append SFX mirrors to
                    # ledger.lines[] so BatchHumoRender's existing
                    # speaker_role-aware loop picks them up.  Skip
                    # if any line already has a matching line_id
                    # (idempotent across reruns; protects resume
                    # behavior).  ledger.sfx[] is NOT removed -- it
                    # remains the canonical SFX index for any
                    # consumer that walks SFX explicitly.
                    if _sfx_to_mirror_into_lines:
                        _existing_ids = {
                            ln.get("line_id") for ln in _ledger_lines
                            if isinstance(ln, dict) and ln.get("line_id")
                        }
                        _appended = 0
                        for _mirror in _sfx_to_mirror_into_lines:
                            if _mirror["line_id"] in _existing_ids:
                                continue
                            _ledger_lines.append(_mirror)
                            _appended += 1
                        if _appended:
                            _led["lines"] = _ledger_lines
                            log.info(
                                "[SceneSequencer] mirrored %d SFX cue(s) "
                                "into ledger.lines[] with speaker_role=sfx "
                                "(BatchHumoRender wall-to-wall coverage)",
                                _appended,
                            )

                    _gc = _OTRL.lookup_git_commit(
                        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    )
                    if _gc:
                        _OTRL.set_meta(_led, "git_commit", _gc)
                    _OTRL.save_ledger_safe(_ledger_p, _led)
                    log.info(
                        "[SceneSequencer] schema-l3 ledger update: phase_ms=%d "
                        "gate=post_scene_sequencer dur=%.1fs "
                        "lines_positioned=%d/%d sfx_positioned=%d/%d",
                        _phase_ms, total_sec,
                        _matched, len(dialogue_positions),
                        _sfx_matched, len(sfx_timeline),
                    )
        except Exception as _meta_exc:
            log.warning(
                "[SceneSequencer] schema-l3 ledger update failed: %s", _meta_exc
            )

        return (audio_out, log_text, manifest_json)


class EpisodeAssembler:
    """Assemble multiple scenes into a complete episode with intro/outro."""

    CATEGORY = "OldTimeRadio"
    FUNCTION = "assemble"
    OUTPUT_NODE = True
    RETURN_TYPES = ("AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("episode_audio", "output_path", "episode_info")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scene_audio": ("AUDIO",),
                "episode_title": ("STRING", {"default": "The Last Frequency"}),
            },
            "optional": {
                "opening_theme_audio": ("AUDIO",),
                "closing_theme_audio": ("AUDIO",),
                "opening_duration_sec": ("FLOAT", {
                    "default": 10.0, "min": 0.0, "max": 60.0, "step": 1.0,
                    "tooltip": "Max duration of opening theme"
                }),
                "closing_duration_sec": ("FLOAT", {
                    "default": 8.0, "min": 0.0, "max": 60.0, "step": 1.0,
                    "tooltip": "Max duration of closing theme"
                }),
                "crossfade_ms": ("INT", {
                    "default": 500, "min": 0, "max": 3000, "step": 100,
                    "tooltip": "Crossfade between theme and content"
                }),
            },
        }

    def assemble(self, scene_audio, episode_title,
                 opening_theme_audio=None, closing_theme_audio=None,
                 opening_duration_sec=10.0, closing_duration_sec=8.0,
                 crossfade_ms=500):

        # Schema l3: phase wall-clock for meta.phase_ms.episode_assembler.
        import time as _time
        _phase_t0 = _time.time()

        # Extract main scene waveform
        if isinstance(scene_audio, dict):
            main_waveform = scene_audio["waveform"]
            sample_rate = scene_audio["sample_rate"]
        else:
            main_waveform = scene_audio
            sample_rate = 48000

        # Ensure 3D
        if main_waveform.dim() == 1:
            main_waveform = main_waveform.unsqueeze(0).unsqueeze(0)
        elif main_waveform.dim() == 2:
            main_waveform = main_waveform.unsqueeze(0)

        xfade_samples = int(sample_rate * crossfade_ms / 1000.0)

        segments = []

        # Opening theme
        if opening_theme_audio is not None:
            opening = self._extract_waveform(opening_theme_audio, target_sr=sample_rate)
            max_samples = int(opening_duration_sec * sample_rate)
            if opening.shape[-1] > max_samples:
                opening = opening[:, :, :max_samples]
            segments.append(opening)

        # Main content
        segments.append(main_waveform)

        # Closing theme
        if closing_theme_audio is not None:
            closing = self._extract_waveform(closing_theme_audio, target_sr=sample_rate)
            max_samples = int(closing_duration_sec * sample_rate)
            if closing.shape[-1] > max_samples:
                closing = closing[:, :, :max_samples]
            segments.append(closing)

        # Match channel counts across all segments
        max_channels = max(s.shape[1] for s in segments)
        matched = []
        for s in segments:
            while s.shape[1] < max_channels:
                s = torch.cat([s, s[:, :1, :]], dim=1)
            matched.append(s)

        # Assemble with real crossfade overlaps between adjacent segments
        # Instead of hard-cutting, we overlap the tail of segment A with the
        # head of segment B using equal-power (sqrt) fades for smooth transitions.
        episode_waveform = matched[0]
        for i in range(1, len(matched)):
            nxt = matched[i]
            xf = min(xfade_samples, episode_waveform.shape[-1], nxt.shape[-1])
            if xf > 0:
                # Equal-power crossfade curves (sqrt for constant energy)
                t = torch.linspace(0.0, 1.0, xf, device=episode_waveform.device)
                fade_out = torch.sqrt(1.0 - t)  # tail of current segment
                fade_in = torch.sqrt(t)          # head of next segment

                # Overlap region: blend tail of current with head of next
                tail = episode_waveform[:, :, -xf:] * fade_out
                head = nxt[:, :, :xf] * fade_in
                blended = tail + head

                # Stitch: everything before overlap + blended + remainder of next
                episode_waveform = torch.cat([
                    episode_waveform[:, :, :-xf],
                    blended,
                    nxt[:, :, xf:]
                ], dim=-1)
            else:
                # Fallback: straight concat if segments too short for crossfade
                episode_waveform = torch.cat([episode_waveform, nxt], dim=-1)

        log.info("[EpisodeAssembler] Assembled %d segments with %dms crossfades",
                 len(matched), crossfade_ms)

        # Final peak normalize to -1.0 dBFS - runs AFTER crossfades so
        # overlapping segments can't push the mix into clipping.
        peak = episode_waveform.abs().max()
        if peak > 1e-8:
            target_linear = 10.0 ** (-1.0 / 20.0)  # -1.0 dBFS
            episode_waveform = episode_waveform * (target_linear / peak)
        log.info("[EpisodeAssembler] Final normalize: -1.0 dBFS (post-crossfade)")

        # Video-only pipeline - MP4 is written by OTR_SignalLostVideo.
        # No WAV or PNG files are saved here.
        output_path = "(video-only - MP4 written by OTR_SignalLostVideo)"

        from datetime import datetime as _dt
        audio_out = {"waveform": episode_waveform, "sample_rate": sample_rate}

        total_sec = episode_waveform.shape[-1] / sample_rate
        info_dict = {
            "title": episode_title,
            "duration_sec": round(total_sec, 1),
            "duration_min": round(total_sec / 60, 1),
            "sample_rate": sample_rate,
            "channels": episode_waveform.shape[1],
            "output_path": output_path,
            "timestamp": _dt.now().isoformat(),
        }
        info = json.dumps(info_dict, indent=2)

        log.info(f"[EpisodeAssembler] '{episode_title}' - {total_sec/60:.1f} min")

        # Schema l3 ledger write-back: phase_ms.episode_assembler +
        # audio_gates "post_episode_assembler" sha256 + BUG-LOCAL-106
        # opening-theme offset shift on lines[].start_s and
        # clips[].start_s. Best-effort; failures are warned not raised.
        try:
            from . import _otr_ledger as _OTRL  # type: ignore
            from ._otr_paths import otr_audio_dir, otr_legacy_audio_dir
            _phase_ms = int((_time.time() - _phase_t0) * 1000)
            _ledger_p = _OTRL.find_most_recent_ledger(
                [otr_audio_dir(), otr_legacy_audio_dir()]
            )
            if _ledger_p is not None:
                _led = _OTRL.load_ledger_safe(_ledger_p)
                if _led is not None:
                    _OTRL.record_phase_ms(_led, "episode_assembler", _phase_ms)
                    # post_episode_assembler audio gate.
                    _ew_cpu = (
                        episode_waveform.detach().cpu().numpy()
                        if hasattr(episode_waveform, "detach")
                        else episode_waveform
                    )
                    _wb = bytes(_ew_cpu.tobytes()[: _OTRL.GATE_HASH_BYTES])
                    _sample_count = int(episode_waveform.shape[-1])
                    _gate = _OTRL.audio_gate_record(
                        gate_name="post_episode_assembler",
                        waveform_bytes=_wb,
                        dur_s=float(total_sec),
                        sample_count=_sample_count,
                        sample_rate=int(sample_rate),
                    )
                    _OTRL.append_audio_gate(_led, _gate)

                    # BUG-LOCAL-106: shift lines[].start_s and
                    # clips[].start_s by the actual master-timeline
                    # offset of the scene audio. Compute the offset
                    # from the segments that were prepended BEFORE
                    # the scene segment. With the equal-power
                    # crossfade in this method, scene audio starts
                    # at (sum of pre-scene segment lengths) -
                    # (number of crossfades crossed * xfade_samples).
                    # Most builds: opening_theme prepended -> scene
                    # starts at opening_theme_dur_samples - xfade.
                    # Compute generically: segments[] is built in
                    # order, scene_audio is at index `_scene_idx`.
                    # _shift_samples = sum of all segments before
                    # scene minus xfade_samples * num_crossfades.
                    try:
                        # `segments` and `xfade_samples` are local to
                        # this method (defined above). We track the
                        # scene's index in the segments list -- with
                        # the current code, segments is opening +
                        # scene + closing so scene is at index 1 if
                        # opening_theme present, else 0.
                        _scene_idx = (
                            1 if opening_theme_audio is not None else 0
                        )
                        _shift_samples = 0
                        for _i, _seg in enumerate(segments[:_scene_idx]):
                            _shift_samples += int(_seg.shape[-1])
                        # Each pre-scene segment that crossfades into
                        # the next subtracts xfade_samples from the
                        # shift (the overlap region eats time off
                        # both sides). With opening->scene only, one
                        # crossfade boundary applies.
                        if _scene_idx > 0:
                            _shift_samples -= int(xfade_samples) * _scene_idx
                            _shift_samples = max(0, _shift_samples)
                        _shift_s = float(_shift_samples) / float(sample_rate)
                    except Exception:
                        _shift_s = 0.0

                    _shifted_lines = 0
                    _shifted_clips = 0
                    _shifted_sfx = 0
                    if _shift_s > 0.0:
                        for _ln in (_led.get("lines") or []):
                            # Only shift entries that are in
                            # scene-audio space (set by SceneSequencer
                            # write-back). Avoids double-shifting on
                            # re-runs.
                            if (
                                _ln.get("start_s_space") == "scene_audio"
                                and isinstance(_ln.get("start_s"), (int, float))
                            ):
                                _ln["start_s"] = float(_ln["start_s"]) + _shift_s
                                _ln["start_s_space"] = "master_mix"
                                _shifted_lines += 1
                        for _cl in (_led.get("clips") or []):
                            # clips[] start_s is currently written by
                            # BatchHumoRender BEFORE EpisodeAssembler
                            # in some workflow orderings. If clips
                            # already exist when this runs, shift
                            # them too. Use a flag to be idempotent.
                            if _cl.get("start_s_space") in (None, "scene_audio") and isinstance(_cl.get("start_s"), (int, float)):
                                _cl["start_s"] = float(_cl["start_s"]) + _shift_s
                                _cl["start_s_space"] = "master_mix"
                                _shifted_clips += 1
                        # BUG-LOCAL-107: same shift for SFX cues so
                        # ledger.sfx[].start_s is master-mix space too.
                        for _sx in (_led.get("sfx") or []):
                            if (
                                _sx.get("start_s_space") == "scene_audio"
                                and isinstance(_sx.get("start_s"), (int, float))
                            ):
                                _sx["start_s"] = float(_sx["start_s"]) + _shift_s
                                _sx["start_s_space"] = "master_mix"
                                _shifted_sfx += 1

                    # BUG-LOCAL-107: stamp the music cue placements.
                    # Opening theme starts at master t=0 and ends at
                    # opening_theme_dur. Closing theme starts at
                    # (scene_end - xfade) + scene_dur - xfade
                    # = total_master - closing_dur. Write them by
                    # cue_id match against ledger.music[].
                    _music_rows = _led.get("music") or []
                    if _music_rows and segments:
                        _opening_dur_s = (
                            float(segments[0].shape[-1]) / float(sample_rate)
                            if opening_theme_audio is not None else 0.0
                        )
                        _closing_dur_s = (
                            float(segments[-1].shape[-1]) / float(sample_rate)
                            if closing_theme_audio is not None and len(segments) > 1
                            else 0.0
                        )
                        _master_total_s = (
                            float(episode_waveform.shape[-1]) / float(sample_rate)
                        )
                        for _mc in _music_rows:
                            _cue = (_mc.get("cue_id") or "").lower()
                            if _cue == "opening" and opening_theme_audio is not None:
                                _mc["start_s"] = 0.0
                                _mc["dur_s"] = _opening_dur_s
                                _mc["start_s_space"] = "master_mix"
                            elif _cue == "closing" and closing_theme_audio is not None:
                                _mc["start_s"] = max(
                                    0.0, _master_total_s - _closing_dur_s
                                )
                                _mc["dur_s"] = _closing_dur_s
                                _mc["start_s_space"] = "master_mix"

                    # BUG-LOCAL-107: append crossfade boundaries to
                    # ledger.transitions[] so post-mortem can audit
                    # the seam between opening->scene and scene->closing.
                    _xfade_ms = int(crossfade_ms)
                    if opening_theme_audio is not None and len(segments) >= 2:
                        _OTRL.append_transition(
                            _led,
                            from_line_id="opening_theme",
                            to_line_id="scene_audio",
                            crossfade_ms=_xfade_ms,
                            boundary_s=(
                                float(segments[0].shape[-1]) / float(sample_rate)
                            ),
                        )
                    if closing_theme_audio is not None and len(segments) >= 2:
                        _OTRL.append_transition(
                            _led,
                            from_line_id="scene_audio",
                            to_line_id="closing_theme",
                            crossfade_ms=_xfade_ms,
                            boundary_s=(
                                float(episode_waveform.shape[-1]) / float(sample_rate)
                                - float(segments[-1].shape[-1]) / float(sample_rate)
                            ),
                        )

                    if _shift_s > 0.0:
                        log.info(
                            "[EpisodeAssembler] BUG-106 master-mix shift: "
                            "+%.3fs applied to %d line(s) + %d clip(s) "
                            "+ %d sfx",
                            _shift_s, _shifted_lines, _shifted_clips,
                            _shifted_sfx,
                        )
                    _gc = _OTRL.lookup_git_commit(
                        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    )
                    if _gc:
                        _OTRL.set_meta(_led, "git_commit", _gc)
                    _OTRL.save_ledger_safe(_ledger_p, _led)
                    log.info(
                        "[EpisodeAssembler] schema-l3 ledger update: "
                        "phase_ms=%d gate=post_episode_assembler dur=%.1fs",
                        _phase_ms, total_sec,
                    )
        except Exception as _meta_exc:
            log.warning(
                "[EpisodeAssembler] schema-l3 ledger update failed: %s", _meta_exc
            )

        return (audio_out, output_path, info)

    def _extract_waveform(self, audio, target_sr=None):
        """Extract waveform tensor from AUDIO input, resampling to target_sr if needed.

        MusicGen emits at 32 kHz, Bark at 24 kHz, Kokoro at 24 kHz, and the
        main scene bus runs at 48 kHz. Themes coming in on the opening /
        closing inputs must be rate-matched before they can be concatenated
        with the main content, otherwise they play at the wrong speed and
        pitch. Resampling happens here at the concatenation boundary - the
        same pattern SceneSequencer uses for TTS clips via _resample_audio.
        """
        if isinstance(audio, dict):
            wf = audio.get("waveform")
            src_sr = int(audio.get("sample_rate") or 0) or None
        else:
            wf = audio
            src_sr = None
        if wf.dim() == 1:
            wf = wf.unsqueeze(0).unsqueeze(0)
        elif wf.dim() == 2:
            wf = wf.unsqueeze(0)

        if target_sr and src_sr and src_sr != target_sr:
            # Use torchaudio if available for high-quality resampling, fall
            # back to numpy-based _resample_audio (already imported in this
            # module for the TTS bus) for correctness without an extra dep.
            try:
                import torchaudio.functional as AF
                wf = AF.resample(wf, src_sr, target_sr)
            except Exception:
                import numpy as _np
                chans = []
                for c in range(wf.shape[1]):
                    clip_np = wf[0, c].cpu().numpy().astype(_np.float32)
                    resampled = _resample_audio(clip_np, src_sr, target_sr)
                    chans.append(torch.from_numpy(resampled).float())
                wf = torch.stack(chans, dim=0).unsqueeze(0)
            log.info("[EpisodeAssembler] Resampled theme %d Hz -> %d Hz",
                     src_sr, target_sr)
        return wf
