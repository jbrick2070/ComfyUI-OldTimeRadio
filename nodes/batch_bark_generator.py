r"""
OTR_BatchBarkGenerator - Character-Grouped Parallel TTS Generation
====================================================================

Pre-computes ALL dialogue TTS audio before the SceneSequencer runs.
Instead of generating Line 1, Line 2, Line 3 sequentially (stop-start
GPU thrashing), this node:

  1. Parses the script JSON for all dialogue lines
  2. Groups lines by character - voice preset (minimizes preset switches)
  3. Generates all lines per character in a single pass (GPU stays hot)
  4. Returns clips in original script order as batched AUDIO

Pipeline position:  Director - BatchBarkGenerator - SceneSequencer

The SceneSequencer receives pre-rendered clips via its `tts_audio_clips`
input and skips inline Bark calls entirely. Result: 60-70% faster renders
on long episodes because the GPU isn't idling between preset switches.

v1.0  2026-04-04  Jeffrey Brick
"""

import json
import logging
import gc
import time
import warnings

import numpy as np
import torch

from ._otr_paths import otr_episodes_root, otr_legacy_audio_dir
from . import _otr_ledger as _OTRL_PATHS
from .story_orchestrator import _runtime_log
from ._vram_log import force_vram_offload, vram_sentinel

log = logging.getLogger("OTR")

# -----------------------------------------------------------------------------
# LOG CLEANUP - Bark's sub-models hardcode max_length=20 as an explicit kwarg.
# When we pass max_new_tokens, transformers fires warnings on every sub-model
# call (~20+ per dialogue line). Cannot intercept via generation_config because
# Bark passes max_length=20 as a direct kwarg that overrides the config object.
# The generation_config kwarg warning is a FutureWarning in transformers -4.45
# - using UserWarning there silently fails to suppress it.
# -----------------------------------------------------------------------------
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
    category=FutureWarning,  # transformers -4.45 emits this as FutureWarning, not UserWarning
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
    dict comprehension misses these - this walks the full tree.
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

# -----------------------------------------------------------------------------
# LOG CLEANUP - suppress urllib3/httpx cache-check spam from HuggingFace
# -----------------------------------------------------------------------------
logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub.file_download").setLevel(logging.WARNING)


# Voice preset resolution: cast.voice_preset is the only source. The
# legacy Director voice_map fallback + _voice_preset_for_character +
# gender-aware grab-bag helpers were deleted in the voice-path-cleanbreak
# 2026-05-12 sprint (Gate 3). Empty / non-v2 cast.voice_preset is a
# writer contract violation -- BatchBarkGenerator hard-raises ValueError.


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
    - text -         sung / hummed phrase

    Tokens NOT supported by Bark (will be spoken as words, so strip them):
      [whispers] [shouts] [nervously] [quietly] - these get cleaned.
    """
    import re

    # -- Step 1: Strip structural / non-Bark tags -----------------------------
    # [VOICE: ...] tags (catch any that slipped through the parser)
    text = re.sub(r'\[VOICE:[^\]]*\]', '', text, flags=re.IGNORECASE)
    # [ENV: ...], [SFX: ...], [MUSIC: ...] - not TTS content
    text = re.sub(r'\[(?:ENV|SFX|MUSIC):[^\]]*\]', '', text, flags=re.IGNORECASE)
    # === SCENE ... === headers
    text = re.sub(r'===.*?===', '', text)

    # -- Step 2: Drop ALL parenthetical stage directions ---------------------
    # BUG-LOCAL-101 (2026-04-28 PM): the previous behavior translated common
    # parentheticals to Bark non-verbal tokens ((panting) -> [pants],
    # (laughs) -> [laughs], etc.). In practice Bark's rendered nonverbal
    # tokens at the start of a clip add 200-500 ms of breath/throat audio
    # BEFORE the first dialogue word, which the listener perceives as
    # garbled words leading into the line (Stellar Shadows 2026-04-28
    # 0:34 "Let's work I guess..." was the [pants] token rendered for
    # l004's "(panting)" prefix). Drop ALL parens so Bark gets clean
    # dialogue text only. If a writer wants laughter or breath in the
    # output they can write it inline ("Hahaha!", "Mmm...") in the actual
    # dialogue text rather than as a parenthetical performance note.
    text = re.sub(r'\([^)]{1,80}\)\s*', '', text)

    # -- Step 3: Asterisk actions - Bark tokens -------------------------------
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
        return ""  # unknown action - drop

    text = re.sub(r'\*([^*]{1,60})\*', _translate_asterisk, text)

    # -- Step 4: Strip remaining unrecognized square-bracket tags -------------
    # Bark speaks unrecognized bracket content as literal words - bad.
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

    # -- Step 5: Force pure ASCII English ------------------------------------
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
            pass  # combining marks - drop after NFKD decomposition
        else:
            cleaned.append("")  # drop non-ASCII entirely
    text = "".join(cleaned)

    # -- Step 6: Normalize whitespace -----------------------------------------
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

      1. Prepend ``[clears throat]`` - a valid Bark non-verbal token that forces
         the model into "about to read text" mode before the dialogue starts.
         Without this anchor, Bark's training data (saturated with podcast/YouTube
         intros that match authoritative male speaker_0) causes the model to
         autocomplete with phrases like "click the link in the description"
         instead of reading the actual script.

      2. Temperature floor of 0.6 - reduces randomness on the first line so the
         model commits to the text rather than hallucinating continuations.
         Subsequent lines in the same preset keep the user-set temperature.
    """
    text = _clean_text_for_bark(text)
    if not text:
        return np.zeros(2400, dtype=np.float32), 24000

    # -- Language drift guard for international presets --------------------
    # Foreign presets (de, fr, es, etc.) are probabilistically biased toward
    # their native language. Cap temperature to 0.55 to keep Bark committed
    # to the English text rather than drifting into the preset's language.
    _is_intl = voice_preset and not voice_preset.startswith("v2/en_")
    if _is_intl:
        temperature = min(temperature, 0.55)

    if is_first_line:
        # Anchor the model before the first dialogue line of each preset.
        # [clears throat] is in Bark's supported token whitelist - it renders
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
        # Recursively move ALL processor outputs to CUDA - including the
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
        # C-level ops - patching the Python functions is the only reliable fix.
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
                    "tooltip": "L3 ledger JSON from OTR_LedgerFreezeCascade"
                }),
            },
            "optional": {
                "temperature": ("FLOAT", {
                    "default": 0.7, "min": 0.1, "max": 1.5, "step": 0.05,
                    "tooltip": "Bark generation temperature (0.7 = balanced)"
                }),
            },
        }

    @vram_sentinel("bark_batch", max_entry_gb=6.0)
    def generate_batch(self, script_json, temperature=0.7):

        # MANDATORY VRAM POWER WASH (clean slate before start).
        force_vram_offload()

        # Read-side helper: parse the script_json wire input as a v2 ledger
        # dict. load_ledger raises ValueError on the legacy parser-list shape;
        # Bark fails loud here (unlike the critic, which is non-blocking).
        # cast.voice_preset is the only source for voice selection (Gate 3).
        from . import _otr_ledger_consumers as _OTRLC
        led = _OTRLC.load_ledger(script_json)

        # -- Step 1: Walk character lines from the ledger ------------------
        # ANNOUNCER lines (speaker_role == "announcer") are intentionally
        # skipped here - they are rendered by the dedicated KokoroAnnouncer
        # node on a separate bus. Keeping them out of the Bark pool
        # eliminates Bark's "ums" and "ahs" from the broadcast-ready
        # opening and closing bookends. We still tally them for the log
        # so the existing diagnostic line stays informative.
        skipped_announcer = sum(
            1
            for _ln in _OTRLC.iter_lines(led, roles={"announcer"})
            if (_ln.get("text") or "").strip()
        )
        dialogue_items = []
        for i, line in enumerate(_OTRLC.iter_lines(led, roles={"character"})):
            text = (line.get("text") or "").strip()
            if not text:
                continue
            line_id = line.get("line_id")
            character_name = _OTRLC.speaker_name(led, line)
            # Gate 3 (voice-path-cleanbreak, last line of defense):
            # cast.voice_preset is the only source. Empty / non-v2/* is a
            # writer cast-lock contract violation that Gate 1 (writer) and
            # Gate 2 (FreezeCascade G6) should have caught earlier. Bark
            # hard-raises here so a future bypass of either upstream gate
            # surfaces at render time instead of silently re-introducing
            # the deleted Director-voice_map fallback.
            preset = _OTRLC.voice_preset(led, line)
            if not preset or not str(preset).startswith("v2/"):
                raise ValueError(
                    f"BatchBarkGenerator: cast.voice_preset missing or "
                    f"non-v2/* for character {character_name!r} "
                    f"(line_id={line_id}, char_id={line.get('char_id')!r}, "
                    f"got {preset!r}). Writer cast-lock contract violation "
                    f"(Gate 1 + Gate 2 should have caught this upstream)."
                )
            dialogue_items.append({
                "script_idx":     i,
                "line_id":        line_id,
                "character_name": character_name,
                "preset":         preset,
                "line":           text,
            })

        total_lines = len(dialogue_items)
        log.info("[BatchBark] Found %d dialogue lines in Canonical 1.0 format "
                 "(skipped %d ANNOUNCER lines - routed to Kokoro bus)",
                 total_lines, skipped_announcer)

        if total_lines == 0:
            empty = {"waveform": torch.zeros(1, 1, 2400), "sample_rate": 24000}
            return (empty, "No dialogue lines found")

        # -- Step 2: Free LLM VRAM - Bark needs GPU headroom ------------
        try:
            from .story_orchestrator import _unload_llm
            _unload_llm()
            log.info("[BatchBark] Freed LLM VRAM for batch TTS")
        except Exception:
            pass

        # -- Step 3: Group by voice preset for efficient generation --------
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

        # P1 #5: Length-sort within each preset group.
        # Bark pads to the longest sequence in the batch. Sorting by text
        # length groups similar-length lines together, reducing wasted padding.
        # Script order is restored at assembly time via results[script_idx].
        for preset in preset_groups:
            preset_groups[preset].sort(key=lambda item: len(item["line"]))

        log.info("[BatchBark] Length-sorted %d preset groups for reduced padding waste",
                 len(preset_groups))
        _runtime_log(f"BatchBark: Length-sorted {len(preset_groups)} preset groups")

        # -- Step 4: Load Bark once, generate all lines per preset ---------
        from .bark_tts import _load_bark
        model, processor = _load_bark("suno/bark")

        # Schema l3: track overall Bark phase wall-clock for the
        # ledger meta.phase_ms.bark field. Per-line render_ms is
        # captured separately on each item.
        _phase_t0 = time.time()

        # Results dict: script_idx - (audio_np, sample_rate)
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
                    log.info("[BatchBark] First line for preset %s - activating "
                             "hallucination guard ([clears throat] + temp floor 0.6)",
                             preset)

                try:
                    # Schema l3 (BUG-LOCAL-101 audit aid): capture the
                    # exact post-cleanup text fed to Bark so the ledger
                    # records WHAT the TTS actually saw -- catching any
                    # future regression where parens / narration leak
                    # back in. _clean_text_for_bark is idempotent so
                    # calling it twice is harmless.
                    _text_for_tts = _clean_text_for_bark(line)
                    _bark_t0 = time.time()
                    audio_np, sr = _generate_single_line(
                        line, preset, model, processor, temperature,
                        is_first_line=is_first,
                    )
                    _bark_render_ms = int((time.time() - _bark_t0) * 1000)
                    dur = len(audio_np) / sr
                    # Stash the new diagnostic fields onto the item for
                    # the ledger write-back loop below.
                    item["_bark_wav_dur_s"] = float(dur)
                    item["_bark_render_ms"] = int(_bark_render_ms)
                    item["_text_for_tts"] = str(_text_for_tts)
                    # BUG-LOCAL-030 audit-completion (2026-05-03 EVENING):
                    # stash voice_preset + per-line audio_sample_hash so
                    # the ledger write-back loop can stamp the new
                    # forensic fields (tts_engine, voice_preset,
                    # render_ms, generated_dur_s, audio_sample_hash).
                    item["_voice_preset"] = str(preset or "")
                    try:
                        from . import _otr_ledger as _OTRL_HASH  # type: ignore
                        item["_audio_sample_hash"] = (
                            _OTRL_HASH.compute_audio_sample_hash(audio_np)
                        )
                    except Exception:
                        item["_audio_sample_hash"] = ""
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
                    batch_log.append(f"  [{idx}] {character_name}: FAILED - {e}")
                    # Silence placeholder
                    word_count = len(line.split())
                    est_dur = max(1.0, word_count / 2.5)
                    results[idx] = (np.zeros(int(24000 * est_dur), dtype=np.float32), 24000)

        log.info("[BatchBark] Generated %d/%d lines", generated, total_lines)
        batch_log.append(f"\n--- Generated: {generated}/{total_lines} lines ---")

        # -- Step 5: Assemble into batched AUDIO tensor (script order) -----
        # Use pad_sequence on GPU for vectorized zero-padding instead of
        # Python loops + numpy.pad. SceneSequencer's _extract_clips already
        # handles trim_trailing_silence on the receiving end.
        from torch.nn.utils.rnn import pad_sequence
        target_sr = 24000  # Bark native rate; AudioEnhance handles upsample

        ordered_clips = []
        for item in dialogue_items:
            audio_np, sr = results[item["script_idx"]]
            # Convert to GPU tensor once - pad_sequence runs on device of inputs
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

        log.info("[BatchBark] Output: %d clips, max_len=%d samples (%.1fs), sr=%d",
                 len(ordered_clips), max_len, max_len / target_sr, target_sr)

        # ---- BUG-LOCAL-096: write per-line dur_s + start_s back to ledger ----
        # Bark doesn't persist per-line WAVs to disk in the current
        # design (clips live in memory and assemble straight into
        # SceneSequencer's mix), so we cannot stamp `bark_wav_path`.
        # But we DO know the actual generated WAV duration per line
        # via results[script_idx] = (audio_np, sr). Writing dur_s
        # back to ledger.lines[].dur_s + cumulative start_s gives
        # BatchHumoRender real per-line timings instead of the
        # BUG-LOCAL-094 word-share heuristic -- HuMo audio slices
        # then align with actual Bark playback positions in the
        # assembled audio.
        #
        # Stamp strategy (v2 ledger): patch by line["line_id"] via
        # _otr_ledger.patch_line_fields(). Replaces the old
        # text-match-then-stamp block (fragile when two characters
        # said the same short line, e.g. "okay."). dialogue_items
        # carries line_id from the iter_lines walk above.
        # Compute overall Bark phase duration for the meta block.
        _total_bark_ms = int((time.time() - _phase_t0) * 1000)

        try:
            from pathlib import Path as _Path
            # BUG-LOCAL-021 (Phase G): use in-flight singleton, not mtime
            # walker. See _otr_ledger.in_flight_ledger_path docstring.
            ledger_path = _OTRL_PATHS.in_flight_ledger_path()
            if ledger_path is not None:
                led_disk = _OTRL_PATHS.load_ledger_safe(ledger_path)
                if led_disk is None:
                    log.warning(
                        "[BatchBark] in-flight ledger load failed at %s; "
                        "skipping ledger write-back",
                        ledger_path,
                    )
                else:
                    cumulative_start = 0.0
                    updated = 0
                    for item in dialogue_items:
                        idx = item.get("script_idx")
                        line_id = item.get("line_id")
                        if idx not in results or not line_id:
                            continue
                        audio_np, sr = results[idx]
                        try:
                            dur = float(len(audio_np)) / float(sr)
                        except Exception:
                            dur = 0.0
                        # Same fields as the prior text-match block.
                        # bark_wav_dur_s + generated_dur_s = the exact
                        # generated WAV duration; dur_s is the
                        # canonical timeline duration; bark_render_ms +
                        # render_ms = wall-clock generation time;
                        # text_for_tts = post-cleanup string fed to
                        # Bark; voice_preset / audio_sample_hash =
                        # BUG-LOCAL-030 forensic fields.
                        fields: dict = {
                            "dur_s":      dur,
                            "start_s":    cumulative_start,
                            "tts_engine": "bark",
                        }
                        if "_bark_wav_dur_s" in item:
                            fields["bark_wav_dur_s"] = float(item["_bark_wav_dur_s"])
                        if "_bark_render_ms" in item:
                            fields["bark_render_ms"] = int(item["_bark_render_ms"])
                        if "_text_for_tts" in item:
                            fields["text_for_tts"] = str(item["_text_for_tts"])
                        if item.get("_voice_preset"):
                            fields["voice_preset"] = str(item["_voice_preset"])
                        if item.get("_bark_render_ms"):
                            fields["render_ms"] = int(item["_bark_render_ms"])
                        if item.get("_bark_wav_dur_s"):
                            fields["generated_dur_s"] = float(item["_bark_wav_dur_s"])
                        if item.get("_audio_sample_hash"):
                            fields["audio_sample_hash"] = str(item["_audio_sample_hash"])
                        if _OTRL_PATHS.patch_line_fields(led_disk, line_id, fields):
                            cumulative_start += dur
                            updated += 1
                    if updated:
                        # Schema l3: stamp phase timing + git commit +
                        # post_bark audio gate alongside the per-line
                        # writes so the ledger shows which version of
                        # the code produced it.
                        try:
                            _OTRL_PATHS.record_phase_ms(
                                led_disk, "bark", int(_total_bark_ms),
                            )
                            _gc_b = _OTRL_PATHS.lookup_git_commit(
                                _Path(__file__).resolve().parents[1]
                            )
                            if _gc_b:
                                _OTRL_PATHS.set_meta(led_disk, "git_commit", _gc_b)
                            # post_bark audio gate.
                            try:
                                _bt_cpu = (
                                    batch_tensor.detach().cpu().numpy()
                                    if hasattr(batch_tensor, "detach")
                                    else batch_tensor.numpy()
                                )
                                _bt_bytes = bytes(
                                    _bt_cpu.tobytes()[: _OTRL_PATHS.GATE_HASH_BYTES]
                                )
                                _bark_gate = _OTRL_PATHS.audio_gate_record(
                                    gate_name="post_bark",
                                    waveform_bytes=_bt_bytes,
                                    dur_s=float(max_len) / float(target_sr),
                                    sample_count=int(max_len),
                                    sample_rate=int(target_sr),
                                )
                                _OTRL_PATHS.append_audio_gate(led_disk, _bark_gate)
                            except Exception as _gexc:
                                log.warning(
                                    "[BatchBark] post_bark audio gate failed: %s",
                                    _gexc,
                                )
                        except Exception as _meta_exc:
                            log.warning(
                                "[BatchBark] schema-l3 meta stamp failed "
                                "(%s); per-line patches still saved",
                                _meta_exc,
                            )
                        # Single atomic save at the end (architect's
                        # write-back contract).
                        _OTRL_PATHS.save_ledger_safe(ledger_path, led_disk)
                        log.info(
                            "[BatchBark] BUG-096 ledger updated (line_id "
                            "stamping): %d line dur_s + start_s timings "
                            "written to %s",
                            updated, ledger_path.name,
                        )
                        batch_log.append(
                            f"BUG-096 ledger updated (line_id stamping): "
                            f"{updated} line timings -> {ledger_path.name}"
                        )
        except Exception as _exc:
            log.warning("[BatchBark] BUG-096 ledger write-back failed: %s", _exc)

        log_text = "\n".join(batch_log)
        return (audio_out, log_text)
