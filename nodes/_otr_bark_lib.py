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

try:
    from ._otr_shared import env as otr_env
except ImportError:  # pragma: no cover -- flat test imports
    from _otr_shared import env as otr_env  # type: ignore

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

    # A STAMPED DEVICE THAT DOES NOT EXIST FALLS BACK -- it does not fail.
    #
    # This repo runs one canonical across a CUDA 5080, a CUDA 4060 and an Apple
    # Silicon Mac, and workflows/otr_canonical.json currently stamps
    # voice_device="mps" by operator ruling while the Mac is the machine under
    # test. Before 2026-09-07 bark IGNORED that stamp entirely and auto-picked
    # cuda, so the mismatch was invisible on NVIDIA. Threading the stamp through
    # (the correct fix, so a Mac finally gets its GPU) removed that accidental
    # protection and would have made an unprofiled NVIDIA run try mps and die.
    # Caught by the codex review lane before the operator's boxes ever saw it.
    #
    # So the stamp is HONOURED WHERE IT IS REAL and falls through where it is
    # not. That is strictly safer than both the old behaviour (stamp ignored
    # everywhere) and the naive fix (stamp obeyed blindly), and it needs no
    # per-machine profile to keep NVIDIA working.
    if device is not None:
        _d = str(device).strip().lower()
        if not _d:
            # An empty stamp is not a device. Unreachable through the ledger
            # (CastLock admits only cuda|cpu|mps) but reachable by any caller
            # that bypasses it, and `.to("")` is a confusing failure to debug.
            device = None
            _d = ""
        if _d.startswith("cuda") and not torch.cuda.is_available():
            log.warning("[Bark] ledger asked for %r but this host has no CUDA; "
                        "auto-detecting instead.", device)
            device = None
        elif _d == "mps" and not (
                getattr(torch.backends, "mps", None) is not None
                and torch.backends.mps.is_available()):
            log.warning("[Bark] ledger asked for 'mps' but this host has no MPS; "
                        "auto-detecting instead.", )
            device = None

    # Auto-detect device: CUDA, then MPS, then CPU.
    #
    # This line used to read `"cuda" if torch.cuda.is_available() else "cpu"`,
    # which silently denied Apple Silicon the GPU: mps was never a candidate,
    # so a Mac always took the CPU branch and was told "CUDA not available"
    # about hardware it does not have.
    #
    # MEASURED 2026-09-07 on a Mac mini M4 (torch 2.12.1) -- Bark runs on mps:
    #   mps  40.8 s -> 4.6 s of audio, spectral flatness 0.070, finite
    #   cpu  27.8 s -> 3.0 s of audio, spectral flatness 0.064, finite
    # Structured speech both ways. Note mps is NOT dramatically faster here
    # (roughly a wash per second of audio at this model size), so the fix is
    # about the selection being HONEST, not about a speed win -- and about the
    # log line no longer blaming missing CUDA on a machine that never had any.
    #
    # CUDA IS UNAFFECTED: cuda still wins whenever it is available, so this is
    # purely an added branch for hosts that previously fell through to cpu.
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = "mps"
            log.info("[Bark] Using Apple Silicon GPU (mps).")
        else:
            device = "cpu"
            log.warning("[Bark] No CUDA or MPS device. Falling back to CPU. TTS will be slow.")

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
        cache_dir_path = os.path.join(otr_env.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
        
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
            # Report the device actually used, not a hardcoded "cuda" -- the
            # loader has always been able to land on CPU, and a log line that
            # claims otherwise sends the next reader hunting the wrong thing.
            log.info("Bark loaded: %s on %s (gen-config patched)",
                     type(model).__name__, device)
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
        # THE CUDA CALL WAS BARE, so adding the guard is not an add-only diff
        # and is called out here rather than buried: on CUDA the condition is
        # True and the call still runs, byte-identical behaviour; off CUDA it
        # was already a documented silent no-op. Nothing changes on the 5080.
        if getattr(torch, "cuda", None) and torch.cuda.is_available():
            torch.cuda.empty_cache()
            log.info("Bark unloaded, VRAM freed (gc.collect + empty_cache)")
        # METAL. `del` + gc drop the Python object; PyTorch's MPS caching
        # allocator keeps the ~4.2 GB reserved until something asks for it
        # back, and the line above never asked on this platform.
        #
        # WHY THIS SITE AND NOT THE OTHER TWENTY-FOUR. Most `empty_cache` calls
        # in this pack are followed by a `soft_empty_cache`, which DOES release
        # MPS (comfy/model_management.py), so their window closes on its own --
        # see PBUG-20260908-03's own correction, which retracts the "the pool
        # ratchets every cycle" reading in favour of a bounded 2x window. This
        # one is different because of ORDERING: `load_llm` performs its wash
        # FIRST and calls `_unload_bark()` AFTER it, so on the bark path the
        # writer's weights materialize on top of a dead-but-reserved Bark pool
        # with no wash in between. On a 16 GB unified machine that is the sum
        # of both models at once, and an overrun there REBOOTS THE HOST rather
        # than failing the render.
        #
        # Deliberately NOT paired with `model.to("cpu")`: that would be a real
        # device->host copy on every CUDA teardown -- a silent regression on
        # the box this fix is not for -- and on unified memory the copy is the
        # same physical RAM anyway.
        elif getattr(torch, "mps", None) and torch.backends.mps.is_available():
            torch.mps.empty_cache()
            log.info("Bark unloaded, unified memory returned "
                     "(gc.collect + torch.mps.empty_cache)")


# -----------------------------------------------------------------------------
# Per-line Bark inference (relocated from batch_bark_generator.py, sprint 1a).
# Self-contained per-line helpers so the engine-registry adapter (eng_bark) and
# the orchestrator voice-health check run Bark inference WITHOUT importing the
# heavy batch node. _clean_text_for_bark stays behaviorally identical to
# scene_sequencer._clean_text_for_bark (regex parity pinned by tests/test_core.py).
# -----------------------------------------------------------------------------

def _clean_text_for_bark(text, *, speech_only=False):
    """Clean and normalize dialogue text for Bark TTS.

    speech_only (B1, 2026-06-22): when True, the HIGH-RISK non-speech tokens
    that produce the high-pitched squeal/whine -- [music], [whistles],
    [sneezes], [gasps] -- are DROPPED from the kept-token whitelist (and so
    are the asterisk stage-directions that translate into them, since the
    bracket filter re-runs over the asterisk output). The low-risk emotive
    tokens [laughs]/[sighs] (and the rest) are kept. Default False preserves
    the legacy whitelist exactly (parity with scene_sequencer + test_core).

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
    if speech_only:
        # B1: never TELL bark to make the squeal/whine non-speech sounds for
        # a dialogue line. Dropping these from the whitelist also strips the
        # asterisk-derived ones (e.g. *gasps* -> [gasps] in step 3) because
        # this filter runs AFTER the asterisk translation.
        _BARK_VALID_TOKENS = _BARK_VALID_TOKENS - {
            "[music]", "[whistles]", "[sneezes]", "[gasps]",
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


def _pack_words(text, max_len):
    """Greedy word-pack ``text`` into chunks of at most ``max_len`` chars.

    Splits ONLY on whitespace -- never mid-word. A single word longer than
    ``max_len`` is kept whole (the last resort: a word is never cut). Pure."""
    words = text.split()
    out = []
    cur = ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > max_len:
            out.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}" if cur else w
    if cur:
        out.append(cur)
    return out or ([text.strip()] if text.strip() else [])


def _split_long_sentence(sentence, max_len):
    """B3 (2026-06-22): split ONE overlong sentence without breaking words.

    Fallback ladder: clause delimiters (``, ; :``) FIRST, then whitespace
    word-packing as the last resort. The delimiter stays attached to the
    clause that precedes it. Returns a list whose parts are each <= max_len
    except an unavoidable single over-long word. Pure / CPU-testable."""
    import re
    s = sentence.strip()
    if len(s) <= max_len:
        return [s] if s else []
    parts = re.split(r'(?<=[,;:])\s+', s)
    out = []
    cur = ""
    for part in parts:
        if len(part) > max_len:
            # A clause longer than the budget -> word-pack it.
            if cur.strip():
                out.append(cur.strip())
                cur = ""
            out.extend(_pack_words(part, max_len))
            continue
        if cur and len(cur) + 1 + len(part) > max_len:
            out.append(cur.strip())
            cur = part
        else:
            cur = f"{cur} {part}" if cur else part
    if cur.strip():
        out.append(cur.strip())
    return out or [s]


def _chunk_text_for_bark(text, max_len=180):
    """Split text into Bark-friendly chunks at sentence boundaries.

    B3: a SINGLE sentence longer than ``max_len`` is no longer returned whole
    (which let bark over-generate on a runaway clause). It is split on clause
    punctuation first, then on whitespace -- never mid-word."""
    import re
    if len(text) <= max_len:
        return [text]

    chunks = []
    sentences = re.split(r'(?<=[.!?])\s+', text)
    current = ""
    for sentence in sentences:
        # B3: an overlong single sentence cannot be packed as one chunk --
        # flush the buffer and split the sentence on clauses/words.
        if len(sentence) > max_len:
            if current.strip():
                chunks.append(current.strip())
                current = ""
            chunks.extend(_split_long_sentence(sentence, max_len))
            continue
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
    raw = (otr_env.get("OTR_BARK_MIN_EOS_P") or "").strip()
    if raw == "":
        return _BARK_DEFAULT_MIN_EOS_P
    try:
        return float(raw)
    except ValueError:
        return _BARK_DEFAULT_MIN_EOS_P


def _env_flag(name, default):
    """Parse a boolean env flag. Empty/unset -> ``default``; otherwise
    1/true/yes/on -> True, everything else -> False. Pure."""
    raw = (otr_env.get(name) or "").strip().lower()
    if raw == "":
        return bool(default)
    return raw in ("1", "true", "yes", "on")


def _resolve_bark_speech_only(default=True):
    """B1: whether a dialogue line renders in SPEECH-ONLY mode (drop the
    high-risk squeal tokens). Driven by OTR_BARK_SPEECH_ONLY (default ON --
    bark is the char_voice engine, so all its lines are dialogue). Pure."""
    return _env_flag("OTR_BARK_SPEECH_ONLY", default)


def _resolve_bark_inject_anchor(default_disabled=True):
    """B1: whether to inject the first-line ``[clears throat]`` anchor.
    Driven by OTR_BARK_DISABLE_THROAT_CLEAR (default ON -> anchor OFF for
    dialogue, since the throat-clear is itself an audible artifact at the
    clip head). Set OTR_BARK_DISABLE_THROAT_CLEAR=0 to restore the anchor.
    Pure."""
    return not _env_flag("OTR_BARK_DISABLE_THROAT_CLEAR", default_disabled)


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


#: Default flag threshold for the high-band edge artifact gate. From the
#: B0 corpus scan, speech edges sit < ~0.15 (p90) while a squeal/whine
#: window is > 0.9; 0.5 cleanly separates them.
_HIGH_BAND_ARTIFACT_THRESHOLD = 0.5


def high_band_edge_ratio(audio, sample_rate, *, edge_ms=150, hb_hz=4000.0):
    """QA metric: high-band (> ``hb_hz``) energy fraction at the clip EDGES.

    Returns the MAX over the first and last ``edge_ms`` of the clip of
    (power above ``hb_hz``) / (total power). The high-pitched non-speech
    squeal that B1 prevents (the [music]/[whistles]/throat-clear artifact)
    concentrates almost all of its energy above 4 kHz and sits at the clip
    head/tail; speech edges are low-band. Near-silent edges return 0.0 (a
    quiet edge is not an artifact). Deterministic, numpy-only, CPU-testable."""
    a = np.asarray(audio, dtype=np.float32)
    if a.ndim > 1:
        a = a.reshape(-1)
    n = int(a.shape[0]) if a.ndim else 0
    if n == 0 or sample_rate <= 0:
        return 0.0
    win = max(1, min(n, int(round((edge_ms / 1000.0) * sample_rate))))

    def _ratio(seg):
        seg = np.asarray(seg, dtype=np.float64)
        if seg.size < 2:
            return 0.0
        if float(np.sqrt(np.mean(seg ** 2))) < 1e-4:
            return 0.0  # near-silence is not an artifact
        windowed = seg * np.hanning(seg.size)
        power = np.abs(np.fft.rfft(windowed)) ** 2
        freqs = np.fft.rfftfreq(seg.size, 1.0 / sample_rate)
        total = float(power.sum())
        if total <= 0.0:
            return 0.0
        return float(power[freqs > hb_hz].sum()) / total

    head = _ratio(a[:win])
    tail = _ratio(a[-win:])
    return max(head, tail)


def flag_high_band_artifact(audio, sample_rate, *,
                            threshold=_HIGH_BAND_ARTIFACT_THRESHOLD,
                            edge_ms=150, hb_hz=4000.0):
    """(flagged, ratio): True when the clip's head/tail high-band ratio meets
    ``threshold`` -- a likely squeal/whine artifact. Pure / CPU-testable."""
    ratio = high_band_edge_ratio(audio, sample_rate, edge_ms=edge_ms, hb_hz=hb_hz)
    return (ratio >= threshold, ratio)


def _generate_single_line(text, voice_preset, model, processor, temperature=0.7,
                          is_first_line=False, *, semantic_temp=None,
                          coarse_temp=None, fine_temp=None,
                          inject_first_line_anchor=True, speech_only=False,
                          seed=None):
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
    text = _clean_text_for_bark(text, speech_only=speech_only)
    if not text:
        return np.zeros(2400, dtype=np.float32), 24000

    # Per-stage temps + the intl / first-line SEMANTIC caps (pure helper).
    sem, crs, fin = _stage_temps_for_line(
        temperature, semantic_temp, coarse_temp, fine_temp,
        voice_preset, is_first_line)

    if is_first_line and inject_first_line_anchor:
        # Anchor the model before the first dialogue line of each preset.
        # [clears throat] is in Bark's supported token whitelist - it renders
        # as a brief audible cue (~0.15s) and resets the generation context
        # away from "podcast opener" toward "radio drama performance". (The
        # matching semantic-temp cap is applied in _stage_temps_for_line.)
        # B1 (2026-06-22): the anchor is ITSELF an audible non-speech cue at
        # the artifact-prone clip head, so dialogue defaults to OFF
        # (inject_first_line_anchor=False via eng_bark). The semantic-temp cap
        # (driven by is_first_line) stays regardless -- it is a separate guard.
        text = f"[clears throat] {text}"

    sample_rate = model.generation_config.sample_rate
    chunks = _chunk_text_for_bark(text)
    all_audio = []
    silence_pad = np.zeros(int(sample_rate * 0.08), dtype=np.float32)

    # B2 (2026-06-22): Bark.generate samples (do_sample=True) and binds NO
    # external Generator (eng_bark.supports_external_generator=False), so the
    # ONLY way to make a line reproducible is to seed the global torch RNG
    # before model.generate consumes it. Seed ONCE before the chunk loop so a
    # multi-chunk line generates a single deterministic sequence. The caller
    # (story orchestrator) already runs inside deterministic_inference, so no
    # manual RNG save/restore is needed here. seed=None keeps the legacy
    # unseeded (stochastic) behavior for callers that pass no seed.
    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    # THE DEVICE THE MODEL IS ACTUALLY ON -- resolved once, from the model
    # itself, never assumed (2026-08-25).
    #
    # These three sites used to hardcode the literal "cuda". `_load_bark` has
    # always been device-aware (it picks `cuda if torch.cuda.is_available()
    # else cpu` and builds device_map/dtype from that), so on a Mac, an Intel
    # box, or any CUDA-less install the model loaded happily on CPU and then
    # the FIRST spoken line raised here. Nothing gated it either: the
    # CAPABILITIES row declares bark `device_backends: ["cuda"]`, but that
    # table feeds capability_profiles to derive per-profile enable-sets and is
    # NOT consulted at voice dispatch -- `registry.assert_usable` only checks
    # "registered + role-compatible". So a stranger who picked bark (the
    # zero-setup announcer/char engine, which is exactly what a fresh install
    # reaches for) walked into an AssertionError mid-render, after the story,
    # the casting and every still had already been paid for.
    #
    # Asking the model is the right source: `_load_bark` does `model.to(device)`
    # and forces every sub-model to the same device, so the parameters are
    # uniform and they are the ground truth about where generate() will run.
    # No try/except here, deliberately. A first cut guarded this with
    # `except StopIteration: _dev = torch.device("cuda" if ... else "cpu")`,
    # and review proved that the GUARD ITSELF crashed on every CUDA box:
    # `torch.device("cuda")` carries index=None, `_move_to_device` lands the
    # tensors on `cuda:0`, and `device(type='cuda', index=0)` does NOT equal
    # `device(type='cuda')` -- so the assert below fired every time the
    # fallback ran. A BarkModel always has parameters; if it somehow does not,
    # a StopIteration naming this line is a far better bug report than a
    # confusing device-mismatch assertion three lines later.
    _dev = next(model.parameters()).device

    for chunk in chunks:
        inputs = processor(chunk, voice_preset=voice_preset)
        # Recursively move ALL processor outputs to the model's device -
        # including the nested 'history_prompt' dict that contains
        # semantic/coarse/fine numpy arrays from the voice preset NPZ file.
        inputs = _move_to_device(inputs, _dev)

        if "attention_mask" not in inputs and "input_ids" in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

        # Keep the assert -- it caught real device drift. It was simply
        # comparing against the wrong thing.
        assert inputs["input_ids"].device == _dev, (
            f"input_ids on {inputs['input_ids'].device}, expected {_dev} "
            "before generate"
        )

        # Monkey-patch torch.tensor and torch.arange to default to the model's
        # device. Bark's internal sub-model loops call these without a device
        # argument, which defaults to CPU and causes the index_select device
        # mismatch. Context managers and set_default_device don't reach inside
        # Bark's C-level ops - patching the Python functions is the only
        # reliable fix. Restored in the `finally` below, so this never leaks
        # into the wider ComfyUI process.
        _orig_tensor = torch.tensor
        _orig_arange = torch.arange
        def _tensor_on_dev(*args, **kwargs):
            if "device" not in kwargs:
                kwargs["device"] = _dev
            return _orig_tensor(*args, **kwargs)
        def _arange_on_dev(*args, **kwargs):
            if "device" not in kwargs:
                kwargs["device"] = _dev
            return _orig_arange(*args, **kwargs)
        torch.tensor = _tensor_on_dev
        torch.arange = _arange_on_dev
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


# ---------------------------------------------------------------------------
# THE SPEECH-SHAPE GUARD (PBUG-20260902-03, built 2026-09-12).
#
# Bark's semantic stage can derail on any roll into non-speech tokens: the
# record has a 9-word line that came back as seven seconds of noise floor
# (dominant bin 0 Hz, flatness 0.47-0.56) and two seconds of a steady tone at
# 2,524-2,679 Hz (flatness 0.038), with nothing abnormal in the log. That is
# the silent wrong render the standing rule licenses a guard against. This
# scorer says whether a take is SHAPED like speech; eng_bark re-rolls a take
# that is not, a bounded number of times, and keeps the best.
#
# WHAT IT MEASURES, and why not the PBUG's first draft. The record proposed
# "the fraction of one-second windows whose dominant frequency sits in
# 70-400 Hz". Calibrated on 32 real bark takes across four presets and four
# deliveries (docs/2026-09-12-bark-output-guard/), that criterion scored
# real speech anywhere from 0.00 to 1.00 -- a voice whose formants carry the
# whole-second peak (one preset sat at 400-1,500 Hz on every line) reads as
# "not speech" -- so it would have re-rolled good takes, which on bark costs
# real minutes. What separates speech from the two documented artifacts is
# PITCH: a periodicity in the 70-400 Hz range, frame by frame. A pure tone at
# 2.5 kHz is periodic too, but its FIRST autocorrelation peak sits at 0.38 ms,
# far below the 2.5 ms floor, which is exactly how it is convicted; a noise
# floor has no peak at all and fails on flatness besides.
#
# NUMBERS, all from the calibration set: real takes of the three normal
# presets scored 0.50-1.00 (median 0.94); every synthetic artifact -- pure
# tone, white noise, noise then tone -- scored 0.00. The pass line sits at
# 0.30: under every normal take, above every artifact. One preset scored
# 0.00-0.62; whether it derails or is simply an odd voice is the operator's
# ear (its takes are in obs), and either way the guard re-rolls it at most
# twice per line.
# ---------------------------------------------------------------------------

#: Pitch range a human speaking voice can sit in, for the frame test.
_SPEECH_F0_HZ = (70.0, 400.0)
#: Analysis frame and hop for the pitch test.
_SPEECH_FRAME_S = 0.040
_SPEECH_HOP_S = 0.020
#: Normalised autocorrelation a frame's first peak must reach to count as
#: pitched.
_SPEECH_PITCH_STRENGTH = 0.45
#: A one-second window is speech-shaped when at least this fraction of its
#: frames are pitched ...
_SPEECH_VOICED_FRACTION = 0.30
#: ... and its 20 Hz-8 kHz spectral flatness is under this (a noise floor
#: measured 0.47-0.56 on the record; speech sits far below).
_SPEECH_FLATNESS_MAX = 0.50
_SPEECH_WINDOW_S = 1.0
#: A window whose RMS is under this fraction of the CLIP'S OWN PEAK is a
#: pause, not a verdict, and is left out of the denominator. Relative, not
#: absolute: the clip is peak-normalised first, so a soft take is judged on
#: shape instead of being scored 0 for being soft (codex, 2026-09-12). About
#: -34 dB below the loudest sample, well under any spoken syllable.
_SPEECH_PAUSE_RMS = 0.02
#: A take PASSES the guard at this score or above (see the numbers above).
SPEECH_SHAPE_PASS = 0.30
#: How many EXTRA takes the guard may spend on one line. Bounded on purpose:
#: a bark line costs tens of seconds, so a failing line costs at most three.
BARK_REROLLS_MAX = 2
#: A large odd stride, not `seed + 1`: engine seeds are 63-bit hashes and
#: adjacent integers are not reserved, so a retry seed is pushed far away.
_BARK_REROLL_STRIDE = 0x9E3779B97F4A7C15
_BARK_SEED_MASK = 0x7FFFFFFFFFFFFFFF


def bark_reroll_seed(seed, attempt):
    """The seed for re-roll ``attempt`` (0 is the original line seed).
    Deterministic, so a replay walks the same ladder to the same winner."""
    if attempt <= 0:
        return int(seed)
    return (int(seed) + int(attempt) * _BARK_REROLL_STRIDE) & _BARK_SEED_MASK


def _spectral_flatness(window, sample_rate):
    p = np.abs(np.fft.rfft(window * np.hanning(window.size))) ** 2
    freqs = np.fft.rfftfreq(window.size, 1.0 / sample_rate)
    band = p[(freqs >= 20.0) & (freqs <= 8000.0)]
    pos = band[band > 0]
    if pos.size == 0:
        return 1.0
    return float(np.exp(np.mean(np.log(pos))) / np.mean(pos))


def _pitched_fraction(window, sample_rate):
    """Fraction of 40 ms frames whose FIRST autocorrelation peak after the
    first dip is strong and sits at a lag in the speaking-pitch range.

    THE FIRST LOCAL MAXIMUM, NOT THE STRONGEST ONE (codex, finished-diff
    review 2026-09-12). A voice with a strong second harmonic correlates
    even better at twice its pitch period, so taking the global maximum of
    the tail reads a 200 Hz voice as a 100 Hz one -- still in range here,
    but the same reasoning at the range edge demotes real speech. The first
    local maximum after the correlation first falls away IS the pitch
    period, which is the whole reason a 2.6 kHz tone fails: its first peak
    sits at 0.38 ms, far below the 2.5 ms floor.

    THE LIMIT, MEASURED AND STATED RATHER THAN PAPERED OVER. When a voice's
    SECOND harmonic is several times louder than its fundamental, the first
    peak lands at HALF the pitch period. Below 200 Hz that half-period is
    still inside the range and the voice passes; above it, the voice reads
    an octave high and scores zero. Measured 2026-09-12 across 95-350 Hz:
    every balance passes up to 200 Hz, and only the second-harmonic-dominant
    shape fails above it. Accepting integer multiples of the first peak
    would cover that case and would also let a 2.6 kHz tone through (its
    9-sample period times seven lands squarely in the speaking range),
    which is the defect this guard exists to catch. So the edge stays: such
    a line costs up to three takes and never costs the take. Every bark
    preset measured that day sits at 95-250 Hz on a normal balance.
    """
    n = int(sample_rate * _SPEECH_FRAME_S)
    hop = max(1, int(sample_rate * _SPEECH_HOP_S))
    lag_lo = int(sample_rate / _SPEECH_F0_HZ[1])
    lag_hi = int(sample_rate / _SPEECH_F0_HZ[0])
    if n <= 0 or window.size < n or lag_hi >= n:
        return 0.0
    frames = pitched = 0
    for start in range(0, window.size - n + 1, hop):
        f = window[start:start + n]
        f = f - f.mean()
        energy = float(np.dot(f, f))
        if energy < 1e-9:
            continue
        frames += 1
        ac = np.correlate(f, f, mode="full")[n - 1:] / energy
        # Search no further than the lowest pitch we accept; a peak beyond
        # that is not a speaking voice whatever its strength.
        search = ac[:lag_hi + 2]
        below = np.where(search[1:] < _SPEECH_PITCH_STRENGTH)[0]
        if below.size == 0:
            continue                          # never dips: DC-like, a hum
        first_dip = int(below[0]) + 1
        peak_lag = -1
        for k in range(first_dip + 1, min(lag_hi, search.size - 2) + 1):
            if (search[k] > _SPEECH_PITCH_STRENGTH
                    and search[k] >= search[k - 1] and search[k] >= search[k + 1]):
                peak_lag = k                  # the FIRST qualifying peak
                break
        if lag_lo <= peak_lag <= lag_hi:
            pitched += 1
    return (pitched / frames) if frames else 0.0


def speech_shape_score(audio, sample_rate):
    """0.0-1.0: the fraction of non-silent one-second windows that are shaped
    like speech (pitched in 70-400 Hz, not a noise floor). 0.0 when nothing
    in the clip is loud enough to judge. Deterministic, numpy-only,
    CPU-testable; a few milliseconds a line.

    LEVEL IS NOT THIS FUNCTION'S BUSINESS (codex, finished-diff review
    2026-09-12). The clip is normalised to unit peak before anything is
    measured, so a quiet-but-usable take -- one that clears the engine's
    1e-4 peak gate but whose RMS sits under a fixed floor -- is judged on
    its SHAPE rather than scored 0 and re-rolled twice for being soft. The
    pause floor below is therefore relative to this clip's own peak, and
    every other measure here (normalised autocorrelation, spectral flatness)
    is already scale-free.
    """
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim > 1:
        x = x.reshape(-1)
    sr = int(sample_rate)
    n = int(sr * _SPEECH_WINDOW_S)
    if sr <= 0 or n <= 0 or x.size < n:
        return 0.0
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak <= 0.0:
        return 0.0
    x = x / peak
    judged = shaped = 0
    for start in range(0, x.size - n + 1, n):
        w = x[start:start + n]
        if float(np.sqrt(np.mean(w * w))) < _SPEECH_PAUSE_RMS:
            continue
        judged += 1
        if (_pitched_fraction(w, sr) >= _SPEECH_VOICED_FRACTION
                and _spectral_flatness(w, sr) < _SPEECH_FLATNESS_MAX):
            shaped += 1
    return (shaped / judged) if judged else 0.0
