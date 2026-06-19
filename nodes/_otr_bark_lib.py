r"""
Bark TTS Node for ComfyUI - Old-Time Radio Edition
====================================================

Wraps Suno's Bark model for expressive character voice generation.
Bark excels at emotional delivery, laughs, sighs, and dramatic pauses -
perfect for radio drama characters.

Bark voice presets:
  v2/en_speaker_0 through v2/en_speaker_9 - varied English voices
  Each preset has a distinct timbre, pitch, and speaking style.

Special Bark tokens (insert in text):
  [laughter]  - laughing
  [laughs]    - brief laugh
  [sighs]     - sigh
  [music]     - musical interlude
  [gasps]     - gasp
  ...         - hesitation/ellipsis

v1.0  2026-04-04  Jeffrey Brick
"""

import json
import logging
import os
import re
import warnings

import numpy as np

# BEST PRACTICE (Section 8): Lazy heavy imports - torch, numpy, transformers
# imported inside methods only. Node registers instantly at startup.

log = logging.getLogger("OTR")


def _move_to_device(obj, device):
    """Recursively move tensors and numpy arrays to the target device.

    BarkProcessor returns voice presets as a nested dict ('history_prompt')
    containing numpy arrays for semantic/coarse/fine prompts. A flat
    dict comprehension misses these - this walks the full tree.
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

# -----------------------------------------------------------------------------
# LOG CLEANUP - compliant fixes handle most warnings at the source.
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
# -----------------------------------------------------------------------------
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

    Use torch_dtype= (not dtype=) - BarkModel wraps its own from_pretrained
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

        # -- VRAM Hardening v1.4: Strict Handoff --
        # If Gemma is in VRAM, evict it now before loading Bark.
        # S30 B4b: route through the modern loader's unload_llm.
        try:
            from ._otr_model_loader import unload_llm
            unload_llm()
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

            # -- STRICT DEVICE SENTRY --
            # Force all sub-models to target device explicitly to prevent any internal
            # state from being stranded on wrong device.
            model.to(device)
            for sub in ("semantic", "coarse_acoustics", "fine_acoustics"):
                sm = getattr(model, sub, None)
                if sm is not None:
                    sm.to(device)

            # -- FIX: Patch generation configs - parent model + all sub-models --
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


# -----------------------------------------------------------------------------
# Per-line Bark inference (relocated from batch_bark_generator.py, sprint 1a).
# Self-contained per-line helpers so the engine-registry adapter (eng_bark) and
# the orchestrator voice-health check run Bark inference WITHOUT importing the
# heavy batch node. _clean_text_for_bark stays behaviorally identical to
# scene_sequencer._clean_text_for_bark (regex parity pinned by tests/test_core.py).
# -----------------------------------------------------------------------------

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


def _stage_temps_for_line(temperature, semantic_temp, coarse_temp, fine_temp,
                          voice_preset, is_first_line):
    """Pure: resolve the (semantic, coarse, fine) Bark temps for one line.

    CPU-testable -- no torch / model. Each stage that is ``None`` inherits the
    legacy single ``temperature``. The international + first-line guards are CAPS
    (``min``) applied to the SEMANTIC stage only (content commitment is a
    semantic-stage decision); coarse/fine keep their profile values.
    """
    sem = float(semantic_temp if semantic_temp is not None else temperature)
    crs = float(coarse_temp if coarse_temp is not None else temperature)
    fin = float(fine_temp if fine_temp is not None else temperature)
    is_intl = bool(voice_preset) and not str(voice_preset).startswith("v2/en_")
    if is_intl:
        sem = min(sem, 0.55)
    if is_first_line:
        sem = min(sem, 0.5 if is_intl else 0.6)
    return sem, crs, fin


#: Runaway-length guard (2026-06-18 caption-fit fix). Bark stochastically
#: over-generates -- the SAME line measured 7s on two runs and 14s on a third,
#: which inflated the per-line dur_s -> the burned caption lingered for the full
#: (too-long) clip ("captions don't fit the dialogue"). A low EOS probability
#: floor on the SEMANTIC stage lets bark terminate at sentence end: a GPU sweep
#: (same line x4 per setting) measured off=mean 8.1s/max 9.3 (+ a 14.4s runaway),
#: 0.1=mean 7.4s/max 7.8 (tightest + lowest variance), 0.2=mean 9.1s. 0.1 wins.
#: Env OTR_BARK_MIN_EOS_P overrides; "0" disables (legacy unbounded behavior).
_BARK_DEFAULT_MIN_EOS_P = 0.1


def _resolve_min_eos_p():
    """The semantic-stage min EOS probability (env OTR_BARK_MIN_EOS_P, default
    0.1; <=0 disables). Pure."""
    raw = (os.environ.get("OTR_BARK_MIN_EOS_P") or "").strip()
    if raw == "":
        return _BARK_DEFAULT_MIN_EOS_P
    try:
        return float(raw)
    except ValueError:
        return _BARK_DEFAULT_MIN_EOS_P


def _trim_trailing_silence(audio, sample_rate, *, thresh_rel=0.06,
                           min_keep_s=0.15, win_s=0.05):
    """Trim trailing near-silence from a bark clip so its duration reflects the
    SPEECH, not bark's trailing pad (the residual ~1s tail the EOS guard leaves).

    Energy-based + conservative: window the clip, find the last window whose RMS
    exceeds ``thresh_rel`` x the peak-window RMS, and keep up to ``min_keep_s`` of
    decay after it. Returns the clip UNCHANGED when it is empty or all-silence
    (peak 0) so a silent placeholder line is never zeroed to nothing. Pure /
    CPU-testable (numpy only); never touches leading or inter-chunk audio."""
    a = np.asarray(audio, dtype=np.float32)
    n = a.shape[0] if a.ndim else 0
    if n == 0:
        return a
    win = max(1, int(win_s * sample_rate))
    rms = np.array([float(np.sqrt(np.mean(a[i:i + win] ** 2)))
                    for i in range(0, n, win)], dtype=np.float32)
    peak = float(rms.max()) if rms.size else 0.0
    if peak <= 0.0:
        return a                                   # all-silence -> leave as-is
    above = np.where(rms > peak * thresh_rel)[0]
    if not above.size:
        return a
    keep = (int(above[-1]) + 1) * win + int(min_keep_s * sample_rate)
    return a[:max(1, min(n, keep))]


def _generate_single_line(text, voice_preset, model, processor, temperature=0.7,
                          is_first_line=False, *, semantic_temp=None,
                          coarse_temp=None, fine_temp=None):
    """Generate TTS audio for one dialogue line. Returns (np_1d, sample_rate).

    PER-STAGE TEMPERATURES (2026-06-17 whiny-voice fix): Bark is a three-stage
    pipeline (semantic -> coarse acoustics -> fine acoustics). A single flat
    ``temperature`` over-randomizes the acoustic stages, which is the main driver
    of the thin/whiny timbre. ``semantic_temp`` / ``coarse_temp`` / ``fine_temp``
    set each stage independently and are routed via the transformers
    ``BarkModel.generate`` PREFIXED-kwargs contract (``semantic_temperature`` /
    ``coarse_temperature`` / ``fine_temperature`` -- a prefixed kwarg goes to that
    sub-model and has priority; nothing on the globally-cached model's
    generation_config is mutated, so there is no cross-voice leak). Each stage that
    is left ``None`` falls back to the legacy single ``temperature`` so existing
    callers (the orchestrator health probe) keep their exact behavior.

    is_first_line=True activates two hallucination guards for the opening line
    of each voice preset:

      1. Prepend ``[clears throat]`` - a valid Bark non-verbal token that forces
         the model into "about to read text" mode before the dialogue starts.
         Without this anchor, Bark's training data (saturated with podcast/YouTube
         intros that match authoritative male speaker_0) causes the model to
         autocomplete with phrases like "click the link in the description"
         instead of reading the actual script.

      2. Temperature CAP on the SEMANTIC stage (0.6 en / 0.5 intl) - reduces
         randomness on the first line so the model commits to the text rather than
         hallucinating continuations. The cap is a CEILING (``min``), not a floor:
         it only ever lowers the semantic temp, and only on the first line. The
         coarse/fine stages keep their profile values (the cap is about content
         commitment, which the semantic stage owns).
    """
    import torch
    text = _clean_text_for_bark(text)
    if not text:
        return np.zeros(2400, dtype=np.float32), 24000

    # Per-stage temps + the intl / first-line SEMANTIC caps (pure helper).
    sem, crs, fin = _stage_temps_for_line(
        temperature, semantic_temp, coarse_temp, fine_temp,
        voice_preset, is_first_line)

    if is_first_line:
        # Anchor the model before the first dialogue line of each preset.
        # [clears throat] is in Bark's supported token whitelist - it renders
        # as a brief audible cue (~0.15s) and resets the generation context
        # away from "podcast opener" toward "radio drama performance". (The
        # matching semantic-temp cap is applied in _stage_temps_for_line.)
        text = f"[clears throat] {text}"

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
                # Prefixed-kwargs route (transformers BarkModel.generate): each
                # ``<stage>_temperature`` is forwarded to that sub-model only and
                # takes priority over any unprefixed value. do_sample drives the
                # semantic/coarse sampling; the fine stage reads its temperature
                # directly. No generation_config is mutated -> no cross-voice leak.
                _gen_kwargs = dict(
                    do_sample=True,
                    semantic_temperature=sem,
                    coarse_temperature=crs,
                    fine_temperature=fin,
                )
                # Runaway-length guard (2026-06-18 caption-fit fix): bark
                # stochastically over-generates (same line measured 7s/7s/14s) ->
                # inflated dur_s -> the burned caption lingers. A low EOS-prob
                # floor on the SEMANTIC stage (default 0.1, GPU-swept) makes it
                # stop at sentence end. Routed via the prefixed-kwarg contract like
                # the temps; <=0 (env) disables.
                _eos = _resolve_min_eos_p()
                if _eos > 0:
                    _gen_kwargs["semantic_min_eos_p"] = _eos
                output = model.generate(**inputs, **_gen_kwargs)
        finally:
            torch.tensor = _orig_tensor
            torch.arange = _orig_arange

        audio_np = output.cpu().numpy().squeeze()
        all_audio.append(audio_np)
        if len(chunks) > 1:
            all_audio.append(silence_pad)

    # Trim bark's trailing silence/pad so the clip duration (-> per-line dur_s ->
    # the burned caption window) reflects the SPEECH, not bark's tail.
    audio = _trim_trailing_silence(np.concatenate(all_audio), sample_rate)
    return audio, sample_rate


# Voice-path-cleanbreak 2026-05-12 (P3, commit 83d7f17): the OTR_BarkTTS
# node class (BarkTTSNode) was deleted (legacy single-line node, unused
# in any active workflow). The _load_bark loader remains because
# batch_bark_generator.py imports it directly. Library-only module --
# no node class, no NODE_CLASS_MAPPINGS.
#
# Voice-path-cleanbreak Sprint 7.2 (2026-05-12): module renamed
# nodes/_bark_lib.py -> nodes/_otr_bark_lib.py per docs/conventions.md
# (project-prefix discipline for private library modules).
# Underscore prefix marks this as a private internal library; otr_
# prefix scopes the name to this project; _lib suffix flags it as
# library-only (no node class). Importers updated in lockstep:
#   nodes/batch_bark_generator.py
#   nodes/scene_sequencer.py (inline-Bark fallback)
#   nodes/story_orchestrator.py (Bark health check + VRAM unload)
#   tests/test_bark_ledger.py (patch target for _load_bark)
