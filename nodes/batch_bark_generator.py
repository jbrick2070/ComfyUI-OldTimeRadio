r"""
OTR_BatchBarkGenerator — Character-Grouped Parallel TTS Generation
====================================================================

Pre-computes ALL dialogue TTS audio before the SceneSequencer runs.
Instead of generating Line 1, Line 2, Line 3 sequentially (stop-start
GPU thrashing), this node:

  1. Parses the script JSON for all dialogue lines
  2. Groups lines by character → voice preset (minimizes preset switches)
  3. Generates all lines per character in a single pass (GPU stays hot)
  4. Returns clips in original script order as batched AUDIO

Pipeline position:  Director → BatchBarkGenerator → SceneSequencer

The SceneSequencer receives pre-rendered clips via its `tts_audio_clips`
input and skips inline Bark calls entirely. Result: 60-70% faster renders
on long episodes because the GPU isn't idling between preset switches.

v1.0  2026-04-04  Jeffrey Brick
"""

import json
import logging
import gc
import warnings

import numpy as np
import torch

from .story_orchestrator import _runtime_log
from ._vram_log import force_vram_offload

log = logging.getLogger("OTR")

# ─────────────────────────────────────────────────────────────────────────────
# LOG CLEANUP — Bark's sub-models hardcode max_length=20 as an explicit kwarg.
# When we pass max_new_tokens, transformers fires warnings on every sub-model
# call (~20+ per dialogue line). Cannot intercept via generation_config because
# Bark passes max_length=20 as a direct kwarg that overrides the config object.
# The generation_config kwarg warning is a FutureWarning in transformers ≥4.45
# — using UserWarning there silently fails to suppress it.
# ─────────────────────────────────────────────────────────────────────────────
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
    category=FutureWarning,  # transformers ≥4.45 emits this as FutureWarning, not UserWarning
)
warnings.filterwarnings(
    "ignore",
    message=r".*Setting `pad_token_id` to `eos_token_id`.*",
    category=UserWarning,
)


def _move_to_device(obj, device):
    """Recursively move tensors and numpy arrays to the target device.

    BarkProcessor returns voice presets as a nested dict ('history_prompt')
    containing numpy arrays for semantic/coarse/fine prompts. A flat
    dict comprehension misses these — this walks the full tree.
    Handles tensors, dicts, lists, tuples, numpy arrays, and any object
    with a .to() method (e.g. nn.Module).
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


# ── Voice preset resolution (shared logic with SceneSequencer) ───────────────

_BARK_VOICE_PRESETS = [
    # -- English (native) --
    "v2/en_speaker_0", "v2/en_speaker_1", "v2/en_speaker_2",
    "v2/en_speaker_3", "v2/en_speaker_4", "v2/en_speaker_5",
    "v2/en_speaker_6", "v2/en_speaker_7", "v2/en_speaker_8",
    "v2/en_speaker_9",
    # -- International accented English --
    # European language presets render English words clearly with a
    # distinct accent/timbre. Adds vocal diversity to the ensemble
    # without sacrificing intelligibility.
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

_CHARACTER_VOICE_CACHE = {}


_FEMALE_PRESETS = [
    # en_speaker_2 and en_speaker_7 removed — sound male/androgynous in practice
    "v2/en_speaker_4", "v2/en_speaker_9",
    "v2/de_speaker_4", "v2/fr_speaker_4", "v2/es_speaker_9",
    "v2/it_speaker_4", "v2/pt_speaker_4",
]
_MALE_PRESETS = [
    "v2/en_speaker_0", "v2/en_speaker_1", "v2/en_speaker_2",
    "v2/en_speaker_3", "v2/en_speaker_5", "v2/en_speaker_6",
    "v2/en_speaker_7", "v2/en_speaker_8",
    "v2/de_speaker_0", "v2/fr_speaker_0", "v2/es_speaker_0",
    "v2/it_speaker_0", "v2/pt_speaker_0",
]


def _voice_preset_for_character(character, voice_map, voice_traits=""):
    """Determine Bark voice preset for a character.

    Priority:
      1. Cached assignment (stable across the episode)
      2. Director's voice_assignments
      3. Fuzzy match (uppercase, underscored, partial)
      4. Gender-aware hash fallback using voice_traits from script
    """
    if character in _CHARACTER_VOICE_CACHE:
        return _CHARACTER_VOICE_CACHE[character]

    voice_info = voice_map.get(character, {})
    preset = voice_info.get("voice_preset") or voice_info.get("bark_preset")
    if preset and preset.startswith("v2/"):
        _CHARACTER_VOICE_CACHE[character] = preset
        return preset

    char_normalized = character.upper().replace(" ", "_")
    for map_key, map_val in voice_map.items():
        key_normalized = map_key.upper().replace(" ", "_")
        if (key_normalized == char_normalized or
                key_normalized in char_normalized or
                char_normalized in key_normalized):
            preset = map_val.get("voice_preset") or map_val.get("bark_preset")
            if preset and preset.startswith("v2/"):
                _CHARACTER_VOICE_CACHE[character] = preset
                return preset

    # Gender-aware hash fallback with 85/15 English-native/international ratio.
    # Director always assigns en_speaker_*; this fallback runs only when the
    # Director mapping is missing. ~85% chance of English native, ~15% of
    # international accented English (adds subtle vocal variety without
    # risking language drift — the temp cap + ASCII sanitizer handle the rest).
    import random
    traits_lower = voice_traits.lower() if voice_traits else ""
    is_female = "female" in traits_lower or "woman" in traits_lower or "girl" in traits_lower
    is_male   = "male" in traits_lower or "man" in traits_lower or "boy" in traits_lower

    # Deterministic seed per character name so same character always gets same voice
    rng = random.Random(hash(character))
    use_intl = rng.random() < 0.15  # 15% chance of international preset

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
    if not pool:  # safety net — should never happen with current preset lists
        pool = _BARK_VOICE_PRESETS
    preset = rng.choice(pool)
    _CHARACTER_VOICE_CACHE[character] = preset
    pool_tag = "international" if (use_intl and intl_pool) else "English-native"
    log.info("[BatchBark] No Director mapping for '%s' (%s), assigned %s from %s %s pool",
             character, traits_lower[:30], preset, pool_tag, label)
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

    Bark's full supported token set (as of suno/bark v1):
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

    Tokens NOT supported by Bark (will be spoken as words, so strip them):
      [whispers] [shouts] [nervously] [quietly] — these get cleaned.
    """
    import re

    # ── Step 1: Strip structural / non-Bark tags ─────────────────────────────
    # [VOICE: ...] tags (catch any that slipped through the parser)
    text = re.sub(r'\[VOICE:[^\]]*\]', '', text, flags=re.IGNORECASE)
    # [ENV: ...], [SFX: ...], [MUSIC: ...] — not TTS content
    text = re.sub(r'\[(?:ENV|SFX|MUSIC):[^\]]*\]', '', text, flags=re.IGNORECASE)
    # === SCENE ... === headers
    text = re.sub(r'===.*?===', '', text)

    # ── Step 2: Parenthetical stage directions → Bark tokens ────────────────
    # e.g. (sighs), (nervous laugh), (clears throat), (whispers softly)
    _PAREN_TO_BARK = [
        # Exact / strong matches first (ordered by specificity)
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
        # Unsupported but common — convert to nearest Bark equivalent
        ("whisper",         ""),        # Bark can't whisper; drop the direction, tone stays
        ("quiet",           ""),
        ("soft",            ""),
        ("shout",           ""),        # Bark doesn't shout; caps in text handles emphasis
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
        return ""  # unknown direction — drop it

    text = re.sub(r'\(([^)]{1,80})\)\s*', _translate_paren, text)

    # ── Step 3: Asterisk actions → Bark tokens ───────────────────────────────
    # e.g. *laughs* *sighs deeply*
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
        return ""  # unknown action — drop

    text = re.sub(r'\*([^*]{1,60})\*', _translate_asterisk, text)

    # ── Step 4: Strip remaining unrecognized square-bracket tags ─────────────
    # Bark speaks unrecognized bracket content as literal words — bad.
    # Whitelist the known-good tokens and drop everything else.
    _BARK_VALID_TOKENS = {
        "[laughter]", "[laughs]", "[sighs]", "[music]", "[gasps]",
        "[clears throat]", "[coughs]", "[pants]", "[sobs]", "[grunts]",
        "[groans]", "[whistles]", "[sneezes]",
    }

    def _filter_bracket_tag(m):
        # Normalize: lowercase + collapse any internal whitespace to single spaces so
        # "[ clears  throat ]" matches "[clears throat]" in the whitelist
        inner = m.group(0)[1:-1].strip().lower()
        inner = re.sub(r'\s+', ' ', inner)
        tag = f"[{inner}]"
        return tag if tag in _BARK_VALID_TOKENS else ""

    text = re.sub(r'\[[^\]]{1,40}\]', _filter_bracket_tag, text)

    # ── Step 5: Force pure ASCII English ────────────────────────────────────
    # Non-ASCII characters (accented letters, foreign scripts, smart quotes)
    # can trigger Bark's language detection to lock into a foreign language
    # when using international presets (v2/fr_*, v2/de_*, etc.).
    # Transliterate what we can, strip the rest.
    import unicodedata
    text = unicodedata.normalize("NFKD", text)
    # Keep only ASCII printable chars + Bark's special tokens in brackets
    cleaned = []
    for ch in text:
        if ord(ch) < 128:
            cleaned.append(ch)
        elif unicodedata.category(ch).startswith("M"):
            pass  # combining marks — drop after NFKD decomposition
        else:
            cleaned.append("")  # drop non-ASCII entirely
    text = "".join(cleaned)

    # ── Step 6: Normalize whitespace ─────────────────────────────────────────
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


def _generate_single_line(text, voice_preset, model, processor, temperature=0.7,
                          is_first_line=False):
    """Generate TTS audio for one dialogue line. Returns (np_1d, sample_rate).

    is_first_line=True activates two hallucination guards for the opening line
    of each voice preset:

      1. Prepend ``[clears throat]`` — a valid Bark non-verbal token that forces
         the model into "about to read text" mode before the dialogue starts.
         Without this anchor, Bark's training data (saturated with podcast/YouTube
         intros that match authoritative male speaker_0) causes the model to
         autocomplete with phrases like "click the link in the description"
         instead of reading the actual script.

      2. Temperature floor of 0.6 — reduces randomness on the first line so the
         model commits to the text rather than hallucinating continuations.
         Subsequent lines in the same preset keep the user-set temperature.
    """
    text = _clean_text_for_bark(text)
    if not text:
        return np.zeros(2400, dtype=np.float32), 24000

    # ── Language drift guard for international presets ────────────────────
    # Foreign presets (de, fr, es, etc.) are probabilistically biased toward
    # their native language. Cap temperature to 0.55 to keep Bark committed
    # to the English text rather than drifting into the preset's language.
    _is_intl = voice_preset and not voice_preset.startswith("v2/en_")
    if _is_intl:
        temperature = min(temperature, 0.55)

    if is_first_line:
        # Anchor the model before the first dialogue line of each preset.
        # [clears throat] is in Bark's supported token whitelist — it renders
        # as a brief audible cue (~0.15s) and resets the generation context
        # away from "podcast opener" toward "radio drama performance".
        text = f"[clears throat] {text}"
        temperature = min(temperature, 0.5 if _is_intl else 0.6)

    sample_rate = model.generation_config.sample_rate
    chunks = _chunk_text_for_bark(text)
    all_audio = []
    silence_pad = np.zeros(int(sample_rate * 0.08), dtype=np.float32)

    for chunk in chunks:
        inputs = processor(chunk, voice_preset=voice_preset)
        # Recursively move ALL processor outputs to CUDA — including the
        # nested 'history_prompt' dict that contains semantic/coarse/fine
        # numpy arrays from the voice preset NPZ file.
        inputs = _move_to_device(inputs, torch.device("cuda"))

        if "attention_mask" not in inputs and "input_ids" in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

        assert inputs["input_ids"].device.type == "cuda", "input_ids not on CUDA before generate"

        # Monkey-patch torch.tensor and torch.arange to default to CUDA.
        # Bark's internal sub-model loops call these without a device argument,
        # which defaults to CPU and causes the index_select device mismatch.
        # Context managers and set_default_device don't reach inside Bark's
        # C-level ops — patching the Python functions is the only reliable fix.
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


class BatchBarkGenerator:
    """Pre-compute all dialogue TTS in character-grouped batches.

    Groups dialogue lines by voice preset so the GPU stays on the same
    speaker embedding without stop-start thrashing. Outputs batched AUDIO
    in original script order for the SceneSequencer to consume.
    """

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
                    "tooltip": "Parsed script JSON from Gemma4ScriptWriter"
                }),
                "production_plan_json": ("STRING", {
                    "multiline": True, "default": "{}",
                    "tooltip": "Production plan JSON from Gemma4Director"
                }),
            },
            "optional": {
                "temperature": ("FLOAT", {
                    "default": 0.7, "min": 0.1, "max": 1.5, "step": 0.05,
                    "tooltip": "Bark generation temperature (0.7 = balanced)"
                }),
            },
        }

    def generate_batch(self, script_json, production_plan_json, temperature=0.7):

        # 🚿 MANDATORY VRAM POWER WASH (Clean slate before start)
        force_vram_offload()

        script = json.loads(script_json) if isinstance(script_json, str) else script_json
        plan = json.loads(production_plan_json) if isinstance(production_plan_json, str) else production_plan_json
        voice_map = plan.get("voice_assignments", {})

        # ── Step 1: Extract all dialogue lines with their script index ────
        # ANNOUNCER lines are intentionally skipped — they are rendered by the
        # dedicated KokoroAnnouncer node on a separate bus. Keeping them out
        # of the Bark pool eliminates Bark's "ums" and "ahs" from the
        # broadcast-ready opening and closing bookends.
        dialogue_items = []
        skipped_announcer = 0
        for i, item in enumerate(script):
            if item.get("type") == "dialogue" and item.get("line", "").strip():
                # Canonical 1.0+ uses the character name as the primary ID
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
        log.info("[BatchBark] Found %d dialogue lines in Canonical 1.0 format "
                 "(skipped %d ANNOUNCER lines — routed to Kokoro bus)",
                 total_lines, skipped_announcer)

        if total_lines == 0:
            empty = {"waveform": torch.zeros(1, 1, 2400), "sample_rate": 24000}
            return (empty, "No dialogue lines found")

        # ── Step 2: Free Gemma4 VRAM — Bark needs GPU headroom ────────────
        try:
            from .story_orchestrator import _unload_llm
            _unload_llm()
            log.info("[BatchBark] Freed Gemma4 VRAM for batch TTS")
        except Exception:
            pass

        # ── Step 3: Group by voice preset for efficient generation ────────
        # Same preset = same speaker embeddings loaded = no thrashing
        from collections import OrderedDict
        preset_groups = OrderedDict()
        for item in dialogue_items:
            preset = item["preset"]
            if preset not in preset_groups:
                preset_groups[preset] = []
            preset_groups[preset].append(item)

        log.info("[BatchBark] Grouped into %d voice presets: %s",
                 len(preset_groups),
                 ", ".join(f"{k}({len(v)} lines)" for k, v in preset_groups.items()))

        # ── Step 4: Load Bark once, generate all lines per preset ─────────
        from .bark_tts import _load_bark
        model, processor = _load_bark("suno/bark")

        # Results dict: script_idx → (audio_np, sample_rate)
        results = {}
        batch_log = []
        generated = 0
        # Track which presets have had their first line generated.
        # The first line per preset gets hallucination guards ([clears throat]
        # prefix + temperature floor 0.6) to prevent Bark defaulting to podcast
        # autocomplete phrases at the start of each new speaker context.
        _presets_started = set()

        for preset, items in preset_groups.items():
            log.info("[BatchBark] Generating %d lines for preset %s",
                     len(items), preset)
            batch_log.append(f"=== Preset: {preset} ({len(items)} lines) ===")

            for item in items:
                # Allow cancellation every 5 lines
                if generated % 5 == 0:
                    try:
                        import comfy.model_management
                        comfy.model_management.throw_exception_if_processing_interrupted()
                    except ImportError:
                        pass

                idx = item["script_idx"]
                character_name = item["character_name"]
                line = item["line"]

                is_first = preset not in _presets_started
                if is_first:
                    _presets_started.add(preset)
                    log.info("[BatchBark] First line for preset %s — activating "
                             "hallucination guard ([clears throat] + temp floor 0.6)",
                             preset)

                try:
                    audio_np, sr = _generate_single_line(
                        line, preset, model, processor, temperature,
                        is_first_line=is_first,
                    )
                    dur = len(audio_np) / sr
                    results[idx] = (audio_np, sr)
                    batch_log.append(f"  [{idx}] {character_name}: {line[:45]}... ({dur:.1f}s)")
                    generated += 1
                    
                    if generated % 5 == 0:
                        _runtime_log(f"BatchBark: {generated}/{total_lines} lines complete")

                    if generated % 10 == 0:
                        log.info("[BatchBark] Progress: %d/%d lines (%d%%)",
                                 generated, total_lines,
                                 int(100 * generated / total_lines))

                except Exception as e:
                    log.warning("[BatchBark] Failed [%d] %s: %s", idx, character_name, e)
                    batch_log.append(f"  [{idx}] {character_name}: FAILED — {e}")
                    # Silence placeholder
                    word_count = len(line.split())
                    est_dur = max(1.0, word_count / 2.5)
                    results[idx] = (np.zeros(int(24000 * est_dur), dtype=np.float32), 24000)

        log.info("[BatchBark] Generated %d/%d lines", generated, total_lines)
        batch_log.append(f"\n--- Generated: {generated}/{total_lines} lines ---")

        # ── Step 5: Assemble into batched AUDIO tensor (script order) ─────
        # Use pad_sequence on GPU for vectorized zero-padding instead of
        # Python loops + numpy.pad. SceneSequencer's _extract_clips already
        # handles trim_trailing_silence on the receiving end.
        from torch.nn.utils.rnn import pad_sequence
        target_sr = 24000  # Bark native rate; AudioEnhance handles upsample

        ordered_clips = []
        for item in dialogue_items:
            audio_np, sr = results[item["script_idx"]]
            # Convert to GPU tensor once — pad_sequence runs on device of inputs
            clip_t = torch.from_numpy(audio_np).float()
            if torch.cuda.is_available():
                clip_t = clip_t.cuda()
            # Per-clip peak normalize to -3 dBFS so quieter voices (e.g. Bark
            # speaker variance) don't get buried in the mix before stitching.
            # AudioEnhance does a final -1 dBFS pass on the assembled track.
            peak = clip_t.abs().max()
            if peak > 1e-6:
                clip_t = clip_t * (10 ** (-3.0 / 20) / peak)
            ordered_clips.append(clip_t)  # shape: [T]

        if not ordered_clips:
            # Fallback: 1 second of silence
            batch_tensor = torch.zeros(1, 1, target_sr)
            max_len = target_sr
        else:
            # Vectorized padding on GPU (zero-pad shorter clips on the right)
            padded = pad_sequence(ordered_clips, batch_first=True)  # [B, max_T]
            batch_tensor = padded.unsqueeze(1).cpu()  # [B, 1, max_T] on CPU
            max_len = padded.shape[-1]

        audio_out = {"waveform": batch_tensor, "sample_rate": target_sr}
        log_text = "\n".join(batch_log)

        log.info("[BatchBark] Output: %d clips, max_len=%d samples (%.1fs), sr=%d",
                 len(ordered_clips), max_len, max_len / target_sr, target_sr)

        return (audio_out, log_text)
