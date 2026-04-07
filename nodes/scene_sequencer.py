r"""
Scene Sequencer + Episode Assembler — Orchestrate the Full Radio Show
======================================================================

Two nodes:
  1. SceneSequencer — Takes a parsed script JSON and production plan,
     renders each line through the appropriate TTS engine (Bark or Parler),
     inserts SFX/music cues at the right moments, and outputs a scene.
     Features: Intelligent pacing (breath buffers, BEAT/PAUSE tags), continuous
     room tone bed, Gemma Director voice_map dispatch.

  2. EpisodeAssembler — Takes multiple rendered scenes, adds act breaks,
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

from .gemma4_orchestrator import _runtime_log

log = logging.getLogger("OTR")


def _move_to_device(obj, device):
    """Recursively move tensors and numpy arrays to the target device.

    BarkProcessor returns voice presets as a nested dict ('history_prompt')
    containing numpy arrays for semantic/coarse/fine prompts. A flat
    dict comprehension misses these — this walks the full tree.
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


# ─────────────────────────────────────────────────────────────────────────────
# LOG CLEANUP — suppress urllib3/httpx cache-check spam from HuggingFace
# ─────────────────────────────────────────────────────────────────────────────
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
        # Entire clip is below threshold — return a tiny slice
        return clip_np[:min_samples] if len(clip_np) > min_samples else clip_np
    last_idx = above[-1]
    # Keep a small tail (50ms at 48kHz ≈ 2400 samples) for natural decay
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
        return clip_np  # silence — don't amplify noise floor
    return (clip_np * (target_peak / peak)).astype(np.float32)


def _resample_audio(clip_np, src_rate, dst_rate):
    """Resample a 1-D float32 numpy array.

    Path selection (RTX 5080 optimized):
      - CUDA available + clip > 5s → torchaudio.functional.resample on GPU
        (8-12x faster than scipy for full scenes, sinc-interpolated)
      - Otherwise → scipy.signal.resample_poly (high-quality CPU path)
      - No scipy → np.interp linear fallback
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
            log.info("[SceneSequencer] Resample %dHz→%dHz: GPU torchaudio (%d samples)",
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
        log.warning("[SceneSequencer] scipy not available — falling back to linear "
                    "interpolation for resampling. Install scipy for proper anti-aliasing.")
        new_len = int(len(clip_np) * dst_rate / src_rate)
        return np.interp(
            np.linspace(0, len(clip_np) - 1, new_len),
            np.arange(len(clip_np)),
            clip_np
        ).astype(np.float32)

DEFAULT_OUT = os.path.join(os.path.expanduser("~"), "Documents", "ComfyUI", "output", "old_time_radio")


# ─────────────────────────────────────────────────────────────────────────────
# ROOM TONE BED — continuous background that fills silence between dialogue
# ─────────────────────────────────────────────────────────────────────────────

def _generate_room_tone(duration_sec, sample_rate=48000, intensity=0.03, descriptors=""):
    """Generate a dynamic background bed based on Canonical 1.0 ENV descriptors.

    Uses the descriptors (e.g. 'night city street, distant traffic') to skew
    the noise profile and add textures like wind, sirens, or electronic hums.

    Path selection (RTX 5080 optimized):
      - CUDA + duration > 60s → GPU torch path (noise + sin on Tensor Cores)
      - Otherwise → CPU numpy path (low overhead for short beds)
    """
    import torch
    n_samples = int(duration_sec * sample_rate)
    desc = descriptors.lower()
    _use_gpu = (torch.cuda.is_available() and duration_sec >= 60)

    if _use_gpu:
        # ── GPU path: all noise + trig on CUDA ──────────────────────────────
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
        hum_amp = intensity * 0.3 if ("electronic" in desc or "fluorescent" in desc or "ship" in desc) else intensity * 0.1
        hum = torch.sin(2 * math.pi * hum_freq * t) * hum_amp

        # Textures
        texture = torch.zeros(n_samples, dtype=torch.float32, device=dev)
        if "traffic" in desc or "street" in desc:
            texture += torch.sin(2 * math.pi * 30 * t) * (intensity * 0.2)
        if "siren" in desc:
            siren_mod = torch.sin(2 * math.pi * 0.2 * t) * 100 + 400
            texture += torch.sin(2 * math.pi * siren_mod * t) * (intensity * 0.05)

        # Sporadic crackle (stays on CPU — tiny loop, negligible cost)
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

    # ── CPU path: numpy (low overhead for short beds) ───────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# INLINE BARK TTS — called by SceneSequencer for dynamic dialogue generation
# ─────────────────────────────────────────────────────────────────────────────

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
    # en_speaker_2 and en_speaker_7 removed — sound male/androgynous in practice
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

# Stable character→preset cache so the same character always gets the same voice
_CHARACTER_VOICE_CACHE = {}


def _voice_preset_for_character(voice_tag, voice_map, voice_traits=""):
    """Determine Bark voice preset for a character/voice_tag.

    Priority:
      1. Cached assignment (stable across the episode)
      2. Director's voice_assignments (from Gemma4Director voice_map_json)
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

    # Gender-aware hash fallback: use voice_traits to pick from the right pool
    traits_lower = voice_traits.lower() if voice_traits else ""
    if "female" in traits_lower or "woman" in traits_lower or "girl" in traits_lower:
        pool = _FEMALE_PRESETS
        label = "female"
    elif "male" in traits_lower or "man" in traits_lower or "boy" in traits_lower:
        pool = _MALE_PRESETS
        label = "male"
    else:
        pool = _BARK_VOICE_PRESETS
        label = "unknown-gender"

    idx = hash(voice_tag) % len(pool)
    preset = pool[idx]
    _CHARACTER_VOICE_CACHE[voice_tag] = preset
    log.info("[VoiceMap] No Director mapping for '%s' (%s), hash-assigned %s from %s pool",
             voice_tag, traits_lower[:30], preset, label)
    return preset


def _clean_text_for_bark(text):
    """Clean and normalize dialogue text for Bark TTS.

    Bark accepts a specific set of non-speech tokens in square brackets.
    This function:
      1. Strips structural tags that must never reach Bark ([VOICE:], [ENV:],
         [SFX:], [MUSIC:], === scene headers ===)
      2. Converts common parenthetical stage directions to Bark token equivalents
      3. Converts asterisk actions (*laughs*) to Bark tokens
      4. Preserves ♪ music notation (Bark renders humming/singing)
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
    ♪ text ♪         sung / hummed phrase

    Tokens NOT supported by Bark (spoken as literal words — must be stripped):
      [whispers] [shouts] [nervous laugh] etc.
    """
    import re

    # ── Step 1: Strip structural / non-Bark tags ─────────────────────────────
    text = re.sub(r'\[VOICE:[^\]]*\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[(?:ENV|SFX|MUSIC):[^\]]*\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'===.*?===', '', text)

    # ── Step 2: Parenthetical stage directions → Bark tokens ─────────────────
    _PAREN_TO_BARK = [
        ("laughter",        "[laughter]"),
        ("laugh",           "[laughs]"),
        ("chuckl",          "[laughs]"),
        ("giggl",           "[laughs]"),
        ("sigh",            "[sighs]"),
        ("gasp",            "[gasps]"),
        ("clears throat",   "[clears throat]"),
        ("clear",           "[clears throat]"),
        ("cough",           "[coughs]"),
        ("pant",            "[pants]"),
        ("breath",          "[pants]"),
        ("sob",             "[sobs]"),
        ("cry",             "[sobs]"),
        ("weep",            "[sobs]"),
        ("grunt",           "[grunts]"),
        ("strain",          "[grunts]"),
        ("groan",           "[groans]"),
        ("moan",            "[groans]"),
        ("whistle",         "[whistles]"),
        ("sneeze",          "[sneezes]"),
        # Unsupported tokens — drop direction, let voice preset carry the tone
        ("whisper",         ""),
        ("quiet",           ""),
        ("soft",            ""),
        ("shout",           ""),
        ("yell",            ""),
        ("scream",          ""),
        ("nervous",         "[sighs]"),
        ("anxious",         "[sighs]"),
        ("excited",         ""),
        ("angry",           ""),
    ]

    def _translate_paren(m):
        inner = m.group(1).lower().strip()
        for stem, token in _PAREN_TO_BARK:
            if stem in inner:
                return (token + " ") if token else ""
        return ""

    text = re.sub(r'\(([^)]{1,80})\)\s*', _translate_paren, text)

    # ── Step 3: Asterisk actions → Bark tokens ───────────────────────────────
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

    # ── Step 4: Strip unrecognized bracket tags ───────────────────────────────
    _BARK_VALID_TOKENS = {
        "[laughter]", "[laughs]", "[sighs]", "[music]", "[gasps]",
        "[clears throat]", "[coughs]", "[pants]", "[sobs]", "[grunts]",
        "[groans]", "[whistles]", "[sneezes]",
    }
    def _filter_bracket_tag(m):
        tag = m.group(0).lower().strip()
        return tag if tag in _BARK_VALID_TOKENS else ""

    text = re.sub(r'\[[^\]]{1,40}\]', _filter_bracket_tag, text)

    # ── Step 5: Normalize whitespace ─────────────────────────────────────────
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
        # Nothing left after cleaning — return tiny silence
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
        # Recursively move ALL processor outputs to CUDA — including the
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
                    "tooltip": "Parsed script JSON from Gemma4ScriptWriter"
                }),
                "production_plan_json": ("STRING", {
                    "multiline": True,
                    "default": "{}",
                    "tooltip": "Production plan JSON from Gemma4Director"
                }),
            },
            "optional": {
                "tts_audio_clips": ("AUDIO", {
                    "tooltip": "Pre-rendered TTS audio clips (from Bark/Parler batch). "
                               "If provided, dialogue lines use these clips instead of "
                               "placeholder silence. Clips are matched to dialogue lines "
                               "in order."
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
                 start_line=0, end_line=999, output_dir=DEFAULT_OUT,
                 default_tts="bark"):

        _runtime_log("SceneSequencer: Starting 1.0 audio assembly...")
        script = json.loads(script_json) if isinstance(script_json, str) else script_json
        plan = json.loads(production_plan_json) if isinstance(production_plan_json, str) else production_plan_json

        voice_map = plan.get("voice_assignments", {})
        pacing = plan.get("pacing", {})
        # PACING: Breath buffer + dramatic pauses
        breath_ms = pacing.get("breath_pause_ms", 400)          # between every dialogue line
        beat_pause_ms = pacing.get("beat_pause_ms", 1500)       # [BEAT] tag — dramatic beat
        pause_ms = pacing.get("pause_ms", 2000)                 # [PAUSE] tag — longer pause
        scene_transition_ms = pacing.get("scene_transition_ms", 2500)
        act_break_ms = pacing.get("act_break_ms", 5000)

        # Guard: fall back to DEFAULT_OUT if output_dir is empty/None
        if not output_dir or not output_dir.strip():
            output_dir = DEFAULT_OUT
        os.makedirs(output_dir, exist_ok=True)

        # Free Gemma4 VRAM before TTS generation — Bark needs GPU headroom.
        # Gemma4 is done by this point (script + plan already generated).
        try:
            from .gemma4_orchestrator import _unload_gemma4
            _unload_gemma4()
            log.info("[SceneSequencer] Freed Gemma4 VRAM for inline TTS")
        except Exception:
            pass  # Gemma4 may already be unloaded or not imported

        # Extract pre-rendered clips from batched AUDIO inputs
        tts_clips = self._extract_clips_from_audio(tts_audio_clips)
        sfx_clips = self._extract_clips_from_audio(sfx_audio_clips)
        tts_clip_idx = 0
        sfx_clip_idx = 0
        log.info(f"[SceneSequencer] Pre-rendered clips: {len(tts_clips)} TTS, {len(sfx_clips)} SFX")

        # We accumulate silence/audio segments as numpy arrays
        sample_rate = 48000  # standardize output
        all_segments = []
        render_log = []
        manifest = []
        
        # Canonical 1.0+ state tracking
        current_character_name = None
        current_env = "silent room"
        env_timeline = []  # List of (start_sample, end_sample, desc)

        lines_to_render = script[start_line:end_line]
        log.info(f"[SceneSequencer] Rendering Canonical 1.0 items {start_line}-{min(end_line, len(script))}")

        current_sample_pos = 0

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
            
            # ── CANONICAL 1.0 TOKENS ──────────────────────────────────────
            
            if item_type == "environment":
                current_env = item.get("description", "default room")
                render_log.append(f"[{global_idx}] ENV: {current_env}")
                continue

            elif item_type == "scene_break":
                segment_np = np.zeros(int(sample_rate * 1.0), dtype=np.float32)
                render_log.append(f"[{global_idx}] === SCENE {item.get('scene', '?')} ===")

            elif item_type == "pause":
                dur_ms = item.get("duration_ms", 800)
                # Cap beats at 200ms — prevents dead air stacking
                dur_ms = min(dur_ms, 200)
                segment_np = np.zeros(int(sample_rate * dur_ms / 1000), dtype=np.float32)
                render_log.append(f"[{global_idx}] (beat) {dur_ms}ms")

            elif item_type == "sfx":
                desc = item.get("description", "unknown sound")
                if sfx_clip_idx < len(sfx_clips):
                    clip_np, clip_sr = sfx_clips[sfx_clip_idx]
                    segment_np = _resample_audio(clip_np, clip_sr, sample_rate)
                    segment_np = _normalize_clip(segment_np)
                    sfx_clip_idx += 1
                    render_log.append(f"[{global_idx}] SFX: {desc}")
                else:
                    segment_np = np.zeros(int(sample_rate * 1.5), dtype=np.float32)
                    render_log.append(f"[{global_idx}] SFX: {desc} (MISSING)")

            elif item_type == "dialogue":
                character_name = item.get("character_name", "UNKNOWN")
                voice_traits = item.get("voice_traits", "")
                line = item.get("line", "")
                preset = _voice_preset_for_character(character_name, voice_map, voice_traits)
                
                if tts_clip_idx < len(tts_clips):
                    clip_np, clip_sr = tts_clips[tts_clip_idx]
                    segment_np = _resample_audio(clip_np, clip_sr, sample_rate)
                    segment_np = _normalize_clip(segment_np)
                    tts_clip_idx += 1
                else:
                    log.info(f"[SceneSequencer] Inline Bark [{global_idx}] {character_name}")
                    bark_np, bark_sr = _generate_bark_for_line(line, preset)
                    segment_np = _resample_audio(bark_np, bark_sr, sample_rate)
                    segment_np = _normalize_clip(segment_np)
                
                render_log.append(f"[{global_idx}] {character_name}: {line[:40]}...")
                current_character_name = character_name

            elif item_type == "direction":
                render_log.append(f"[{global_idx}] DIRECTION: {item.get('text', '')[:50]}")

            # ── Accumulate Audio and Track Environment Span ──────────────
            if segment_np is not None:
                seg_len = len(segment_np)
                env_timeline.append((current_sample_pos, current_sample_pos + seg_len, current_env))
                all_segments.append(segment_np)
                current_sample_pos += seg_len

        # Log clip usage stats
        render_log.append(f"--- Audio units assembled: {len(all_segments)}")

        # Concatenate all dialogue/SFX segments
        if all_segments:
            combined = np.concatenate(all_segments)
        else:
            combined = np.zeros(int(sample_rate * 1), dtype=np.float32)

        # ── CANONICAL 1.0 ENVIRONMENT MIXING ──────────────────────────
        total_len = len(combined)
        final_bed = np.zeros(total_len, dtype=np.float32)
        room_intensity = plan.get("vintage_settings", {}).get("room_tone_intensity", 0.02)
        
        for start, end, desc in env_timeline:
            span_len_sec = (end - start) / sample_rate
            # Generate a specialized texture for this description
            bed_segment = _generate_room_tone(span_len_sec, sample_rate, intensity=room_intensity, descriptors=desc)
            fit_len = min(len(bed_segment), end - start)
            final_bed[start : start + fit_len] += bed_segment[:fit_len]
            
        combined = combined + final_bed
        render_log.append(f"--- Layered {len(env_timeline)} environment segments")
        total_sec = total_len / sample_rate
        _runtime_log(f"SceneSequencer: 1.0 Mix complete ({total_sec:.1f}s)")

        waveform = torch.from_numpy(combined).float().unsqueeze(0).unsqueeze(0)
        audio_out = {"waveform": waveform, "sample_rate": sample_rate}
        log_text = "\n".join(render_log)
        manifest_json = json.dumps(manifest, indent=2)

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
            opening = self._extract_waveform(opening_theme_audio)
            max_samples = int(opening_duration_sec * sample_rate)
            if opening.shape[-1] > max_samples:
                opening = opening[:, :, :max_samples]
            segments.append(opening)

        # Main content
        segments.append(main_waveform)

        # Closing theme
        if closing_theme_audio is not None:
            closing = self._extract_waveform(closing_theme_audio)
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

        # Final peak normalize to -1.0 dBFS — runs AFTER crossfades so
        # overlapping segments can't push the mix into clipping.
        peak = episode_waveform.abs().max()
        if peak > 1e-8:
            target_linear = 10.0 ** (-1.0 / 20.0)  # -1.0 dBFS
            episode_waveform = episode_waveform * (target_linear / peak)
        log.info("[EpisodeAssembler] Final normalize: -1.0 dBFS (post-crossfade)")

        # Video-only pipeline — MP4 is written by OTR_SignalLostVideo.
        # No WAV or PNG files are saved here.
        output_path = "(video-only — MP4 written by OTR_SignalLostVideo)"

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

        log.info(f"[EpisodeAssembler] '{episode_title}' — {total_sec/60:.1f} min")
        return (audio_out, output_path, info)

    def _extract_waveform(self, audio):
        """Extract waveform tensor from AUDIO input."""
        if isinstance(audio, dict):
            wf = audio.get("waveform")
        else:
            wf = audio
        if wf.dim() == 1:
            wf = wf.unsqueeze(0).unsqueeze(0)
        elif wf.dim() == 2:
            wf = wf.unsqueeze(0)
        return wf
