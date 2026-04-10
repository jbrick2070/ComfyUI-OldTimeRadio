r"""
OTR Orchestrator — Script Writer + Director for "SIGNAL LOST"
===================================================================

Two nodes:
  1. Gemma4ScriptWriter — Fetches real daily science news via RSS, feeds it to
     LLM to generate a full audio drama script. Contemporary sci-fi anthology
     format (Black Mirror / NPR Invisibilia / Arrival). News-as-spine: real
     headlines become the inciting incident, extrapolated to dramatic extremes.
     Includes a hard-science epilogue citing real sources (ArXiv, Nature, etc.).

  2. Gemma4Director — Takes a finished script and generates a production plan:
     TTS voice assignments, SFX cue list, music cues, timing, and spatial audio
     settings. Outputs structured JSON that drives all downstream nodes.

LLM runs via transformers (local GPU). Content safety filter catches
profanity/NSFW that slips past the prompt policy.

v1.0  2026-04-04  Jeffrey Brick
"""

import json
import logging
import os
import random
from random import SystemRandom

# OS-backed RNG for the Lemmy easter-egg coin flip.
# We can't use the seeded module-level `random` because it's seeded per-episode
# from the fingerprint (for reproducible Gemma behavior), which would freeze the
# 11% roll into "always on" or "always off" for any given widget config.
# SystemRandom is unaffected by random.seed() and gives a true ~11% per run.
_LEMMY_RNG = SystemRandom()
_LEMMY_HISTORY = []  # Rolling window of recent Lemmy coin flips (True/False)
import re
import socket
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# Project State (v1.4 Theme C) — series bible for cross-episode consistency.
# Read-only during generation. See nodes/project_state.py for the write path.
from .project_state import ProjectState

# Per-phase VRAM telemetry (v1.4 Theme C). CUDA-absent safe.
from ._vram_log import vram_snapshot, vram_reset_peak, force_vram_offload

# Lazy heavy imports (Section 8) — torch, numpy, transformers inside methods/classes only

log = logging.getLogger("OTR")

# BaseStreamer for custom heartbeat logic.
# Graceful stub allows importing this module in test environments without
# a GPU or transformers installed — ScriptParser and pure-logic tests work fine;
# actual Gemma4 generation will raise ImportError at call time as expected.
try:
    from transformers.generation.streamers import BaseStreamer, TextStreamer
except ImportError:
    class BaseStreamer:  # type: ignore[no-redef]
        """Stub — transformers not installed in this environment."""
        def put(self, value): pass
        def end(self): pass
    class TextStreamer(BaseStreamer):  # type: ignore[no-redef]
        pass

def _runtime_log(msg):
    """Write a persistent heartbeat to otr_runtime.log for monitoring."""
    try:
        ts = datetime.now().strftime("%H:%M:%S")
        log_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "otr_runtime.log")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {msg}\n")
    except: pass

def _truncate_at_sentence_boundary(text: str, max_chars: int) -> str:
    """Truncate text to max_chars, trying to back up to the nearest sentence boundary."""
    if len(text) <= max_chars:
        return text
    
    truncated = text[:max_chars]
    # Look for the last terminal punctuation
    match = re.search(r'([.!?])(?=\s|$)[^.!?]*$', truncated)
    if match:
        return truncated[:match.end()]
    
    # If no punctuation found, just back up to last space
    last_space = truncated.rfind(' ')
    if last_space > 0:
        return truncated[:last_space] + '...'
    return truncated + '...'

def _tail_at_sentence_boundary(text: str, target_chars: int) -> str:
    """Take the LAST target_chars of text, walking forward to the next sentence start."""
    if len(text) <= target_chars:
        return text
        
    tail = text[-target_chars:]
    # Find the FIRST terminal punctuation in the tail
    match = re.search(r'[.!?]\s+', tail)
    if match:
        return tail[match.end():]
        
    # If no punctuation, walk forward to first space
    first_space = tail.find(' ')
    if first_space > 0:
        return tail[first_space+1:]
    return tail

def _inject_scene_transitions(script_text: str) -> tuple:
    """Detect '=== SCENE ===' boundaries and inject transition SFX where lacking.
    Returns (modified_text, transition_count).
    """
    lines = script_text.split('\n')
    out_lines = []
    transition_count = 0
    idx = 0
    
    while idx < len(lines):
        line = lines[idx]
        out_lines.append(line)
        
        # Check if line is a scene boundary (but don't inject after Scene 1 which has opening music)
        if re.match(r'^===\s*SCENE\s+(?![1]\b)(.+?)\s*(?:===|\*\*\*)', line.strip(), re.IGNORECASE):
            # Look ahead at the next non-empty line
            lookahead = idx + 1
            while lookahead < len(lines):
                next_line = lines[lookahead].strip()
                if not next_line:
                    lookahead += 1
                    continue
                
                # If the next thing is just dialogue, inject a transition
                if next_line.startswith('[VOICE:'):
                    out_lines.append("")
                    out_lines.append("[SFX: Scene transition — low bass sweep or static crossfade]")
                    out_lines.append("(beat)")
                    transition_count += 1
                break
                
        idx += 1
        
    return "\n".join(out_lines), transition_count



# ─────────────────────────────────────────────────────────────────────────────
# Phase 3c: WALL-CLOCK TIMEOUT WRAPPER
# Heavy LLM phases (Open-Close outlines, Critique, Revision) can hang if
# LLM stalls on a malformed prompt or GPU goes sideways. We run the
# call in a worker thread and bound it with a wall-clock budget. On timeout
# the thread is left to drain in the background (Gemma generation is not
# cancellable mid-token) but the caller gets control back via TimeoutError
# and the pipeline can fall back to its last known-good artifact.
# ─────────────────────────────────────────────────────────────────────────────
class _LLMTimeout(Exception):
    """Raised when an LLM phase exceeds its wall-clock budget."""
    pass


import threading
_TIMEOUT_CTX = threading.local()

def _run_with_timeout(fn, timeout_sec, phase_label="LLM"):
    """Run fn() in a worker thread with a wall-clock timeout.

    Returns fn's return value on success.
    Raises _LLMTimeout if the budget is exceeded.
    Re-raises any exception fn raised.
    """
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
    vram_reset_peak(phase_label)

    def _worker():
        _TIMEOUT_CTX.deadline = time.time() + timeout_sec
        try:
            return fn()
        finally:
            if hasattr(_TIMEOUT_CTX, "deadline"):
                del _TIMEOUT_CTX.deadline

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"otr-{phase_label}")

    try:
        future = executor.submit(_worker)
        try:
            res = future.result(timeout=timeout_sec)
            vram_snapshot(phase_label)
            return res
        except FuturesTimeout:
            _runtime_log(f"TIMEOUT: {phase_label} exceeded {timeout_sec}s wall-clock budget")
            log.warning("[Timeout] %s phase exceeded %ds — abandoning and falling back",
                        phase_label, timeout_sec)
            vram_snapshot(f"{phase_label}_timeout")
            raise _LLMTimeout(f"{phase_label} exceeded {timeout_sec}s")
    finally:
        # Don't wait for the orphaned worker — let it drain in the background.
        executor.shutdown(wait=False)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3d: BARK VOICE HEALTH CHECK
# Synthesize a 1-second test clip for each active English preset at startup.
# Any preset that returns silence or NaN gets removed from _VOICE_PROFILES
# for the rest of the session, so the Director can never assign a broken
# voice. Runs once per process, lazily on first ScriptWriter init so we
# don't pay the Bark load cost in environments that only import the module.
# ─────────────────────────────────────────────────────────────────────────────
_BARK_HEALTH_CHECKED = False
_BARK_HEALTH_DISABLED = set()


def _bark_health_check():
    """Run a 1-second synthesis test on every active en_speaker_* preset.

    Mutates the module-level _VOICE_PROFILES, _ANNOUNCER_PRESETS, and
    _LEMMY_PROFILE to remove any preset that fails. Idempotent — only
    runs the first time it's called.
    """
    global _BARK_HEALTH_CHECKED, _VOICE_PROFILES, _ANNOUNCER_PRESETS, _LEMMY_PROFILE
    if _BARK_HEALTH_CHECKED:
        return
    _BARK_HEALTH_CHECKED = True

    try:
        import numpy as np
        from .bark_tts import _load_bark
        from .batch_bark_generator import _generate_single_line
    except ImportError as e:
        log.info("[VoiceHealth] Bark not importable (%s) — skipping health check", e)
        _runtime_log(f"VOICE_HEALTH_SKIPPED: bark unavailable ({e})")
        return

    log.info("[VoiceHealth] Running 1-second Bark health check on English presets...")
    _runtime_log("VOICE_HEALTH: Starting Bark preset health check")

    presets_to_test = sorted({vp[0] for vp in _VOICE_PROFILES} |
                              {p for p, _ in _ANNOUNCER_PRESETS} |
                              {_LEMMY_PROFILE["voice_preset"]})

    # ── Quick smoke test on a single preset BEFORE the full sweep ──
    try:
        model, processor = _load_bark(device="cuda")
        _probe, _ = _generate_single_line("Test.", presets_to_test[0], model, processor, temperature=0.6)
    except Exception as e:
        log.warning("[VoiceHealth] Bark probe failed (%s) — Bark itself appears broken, "
                    "skipping per-preset check and leaving all voices enabled", e)
        _runtime_log(f"VOICE_HEALTH_SKIPPED: bark probe failed ({e}) — all presets left enabled")
        return

    test_text = "Testing one two three."
    disabled = set()
    for preset in presets_to_test:
        t0 = time.time()
        try:
            arr, _ = _generate_single_line(test_text, preset, model, processor, temperature=0.6)
            if arr.size == 0:
                raise ValueError("empty audio")
            if not np.isfinite(arr).all():
                raise ValueError("NaN/Inf in output")
            if float(np.max(np.abs(arr))) < 1e-4:
                raise ValueError("silent output")
            log.info("[VoiceHealth] %s OK (%.1fs)", preset, time.time() - t0)
            _runtime_log(f"VOICE_HEALTH_OK: {preset} ({time.time()-t0:.1f}s)")
        except Exception as e:
            disabled.add(preset)
            log.warning("[VoiceHealth] %s FAILED: %s — disabling for session", preset, e)
            _runtime_log(f"VOICE_HEALTH_DISABLED: {preset} — {e}")

    if disabled:
        _BARK_HEALTH_DISABLED.update(disabled)
        _VOICE_PROFILES[:] = [vp for vp in _VOICE_PROFILES if vp[0] not in disabled]
        _ANNOUNCER_PRESETS[:] = [(p, n) for p, n in _ANNOUNCER_PRESETS if p not in disabled]
        if _LEMMY_PROFILE["voice_preset"] in disabled:
            survivors = [vp[0] for vp in _VOICE_PROFILES if vp[1] == "male"]
            if survivors:
                fallback = survivors[0]
                log.warning("[VoiceHealth] LEMMY preset disabled — falling back to %s", fallback)
                _runtime_log(f"VOICE_HEALTH_DISABLED: LEMMY preset replaced with {fallback}")
                _LEMMY_PROFILE["voice_preset"] = fallback
        _runtime_log(f"VOICE_HEALTH: {len(disabled)} preset(s) disabled, {len(_VOICE_PROFILES)} remain")
    else:
        _runtime_log(f"VOICE_HEALTH: All {len(presets_to_test)} presets passed")

# ─────────────────────────────────────────────────────────────────────────────
# LOG CLEANUP — compliant fixes handle most warnings at the source.
# These catch residual library noise from urllib3/httpx cache checks.
# ─────────────────────────────────────────────────────────────────────────────
logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub.file_download").setLevel(logging.WARNING)

# ─────────────────────────────────────────────────────────────────────────────
# CONTENT SAFETY FILTER — catches profanity/NSFW that slips past the prompt
# ─────────────────────────────────────────────────────────────────────────────

# Word list: common profanity, slurs, and explicit terms.
# Kept as a set for O(1) lookup. Checked against whole-word boundaries only
# so words like "assembly" or "hell" (in sci-fi context) aren't false-flagged.
_BLOCKED_WORDS = {
    # profanity
    "fuck", "fucking", "fucked", "fucker", "motherfucker", "motherfucking",
    "shit", "shitting", "shitty", "bullshit",
    "damn", "damned", "dammit", "goddamn", "goddammit",
    "ass", "asshole", "arse", "arsehole",
    "bitch", "bitches", "bastard", "crap", "crappy",
    "piss", "pissed", "pissing",
    "dick", "cock", "pussy", "tits", "boobs",
    "whore", "slut", "skank",
    # slurs (abbreviated to avoid reproducing full slurs in source)
    "nigger", "nigga", "faggot", "fag", "retard", "retarded",
    "spic", "chink", "kike", "wetback", "coon",
    # violence-adjacent shock terms
    "disembowel", "dismember", "decapitate", "eviscerate",
    "rape", "raped", "raping", "molest",
}

# FIX-2 (v1.2): Minced-oath pool replaces [BLEEP] censor.
# Period-authentic 1940s radio euphemisms + pulp adventure + sci-fi flavor.
# Rotated per-replacement so the same script doesn't repeat the same oath twice.
_MINCED_OATHS = [
    # Golden-age radio (G-rated)
    "Golly", "Gee", "Gee whiz", "Jeepers", "Jiminy", "Jiminy Cricket",
    "Heavens", "Heavens to Betsy", "Good heavens", "My stars",
    "Land sakes", "Goodness gracious", "For Pete's sake", "By Jove",
    "Great Scott", "Cheese and crackers",
    # Pulp adventure
    "Blazes", "Thunderation", "Hot dog", "Holy smokes", "Holy cow",
    "Holy mackerel", "Suffering succotash", "Leapin' lizards",
    "Good grief", "Gadzooks", "Zounds",
    # Sci-fi space-opera
    "Stars above", "By the stars", "Great galaxies", "Holy vacuum",
    "Sweet cosmos", "By the rings", "Thundering comets", "Sputtering satellites",
]


def _content_filter(text: str) -> tuple:
    """Scrub blocked words from generated script text.

    Returns (cleaned_text, list_of_replacements_made).
    Uses whole-word regex matching to avoid false positives.
    Replacements rotate through _MINCED_OATHS (period-appropriate euphemisms)
    instead of emitting [BLEEP] — preserves the old-time-radio atmosphere.
    """
    replacements = []
    _oath_cursor = [0]  # list-wrapped so closure can mutate
    def _replace(match):
        word = match.group(0)
        replacements.append(word.lower())
        oath = _MINCED_OATHS[_oath_cursor[0] % len(_MINCED_OATHS)]
        _oath_cursor[0] += 1
        # Preserve capitalization style of the original word
        if word.isupper():
            return oath.upper()
        if word[0].isupper():
            return oath
        return oath.lower()

    # Build regex: whole-word match, case-insensitive
    if not _BLOCKED_WORDS:
        return text, []
    pattern = r'\b(?:' + '|'.join(re.escape(w) for w in sorted(_BLOCKED_WORDS, key=len, reverse=True)) + r')\b'
    cleaned = re.sub(pattern, _replace, text, flags=re.IGNORECASE)

    if replacements:
        log.warning("[ContentFilter] Replaced %d blocked word(s): %s",
                    len(replacements), ", ".join(set(replacements)))

    return cleaned, replacements


# ─────────────────────────────────────────────────────────────────────────────
# PROCEDURAL CHARACTER GENERATOR — name, age, gender, demeanor, accent, voice
# All traits derived deterministically from episode seed + character index.
# LEMMY stays LEMMY with fixed traits. ANNOUNCER stays ANNOUNCER.
#
# BARK TTS ACCENT RULES (per Suno documentation):
#   - Foreign preset + pure English text = English spoken with that accent
#   - en_speaker_* = neutral American/British English
#   - de_speaker_* = English with German accent
#   - fr_speaker_* = English with French accent
#   - es_speaker_* = English with Spanish accent  ... etc.
#   - ALL text is ALWAYS pure ASCII English (enforced by ASCII sanitizer
#     in batch_bark_generator.py) — this prevents language drift
#   - Temperature capped at 0.55 for international presets (0.5 first lines)
# ─────────────────────────────────────────────────────────────────────────────

# Sci-fi character name pools — contemporary, neutral, tech-aligned
# Omni-Retro 5-Pillar Naming Pool — short, punchy, Bark-optimized (1-2 syllables, hard consonants)
# Pillars: 1950s Americana Noir, Afrofuturism, Neo-Tokyo Cyberpunk, Thai Density, Russian Dieselpunk
_FIRST_NAMES = [
    # 1950s Americana Noir
    "Vance", "Stone", "Margot", "Nora", "Sully", "Mac", "Hayes",
    "Cole", "Drake", "Quinn", "Reese", "Kane", "Carter", "Blake",
    # Afrofuturism
    "Malik", "Zuri", "Chidi", "Ayo", "Oya", "Kael", "Tariq", "Nia",
    # Neo-Tokyo Cyberpunk
    "Ren", "Akira", "Kenji", "Yuki", "Sora", "Jiro", "Rei", "Hiro",
    # Thai Density
    "Krit", "Mali", "Niran", "Sunan", "Dao", "Pim", "Som",
    # Russian Dieselpunk
    "Lev", "Anya", "Dmitri", "Sergei", "Volkov", "Mira", "Yuri",
    # Simpsons (sci-fi viable)
    "Nelson", "Martin", "Carl", "Lenny", "Montgomery", "Seymour", "Edna",
    "Ned", "Barney", "Moe", "Kent", "Rod", "Todd", "Jimbo", "Dolph", "Kearney",
    # Pulp adventure (generic first names)
    "Dale", "Tommy", "Pinky",
    # Public domain classics (published before 1931)
    "Alice", "Allan", "Ayesha", "Cavor", "Dracula", "Edward", "Griffin", "Gulliver",
    "Henry", "James", "John", "Karnacki", "Leviathan", "Mina", "Nemo", "Phileas",
    "Quasimodo", "Robinson", "Sherlock", "Smee", "Tarkon", "Victor", "Watson", "Wendy",
    # Peter O'Toole characters
    "Lawrence", "Reginald", "Anton", "Priam", "Maurice", "Alan",
    # Jim Carrey characters
    "Truman", "Fletcher", "Joel", "Stanley", "Walter", "Ace", "Lloyd", "Bruce",
    # Robin Williams characters
    "Mork", "Adrian", "Sean", "Andrew", "Parry", "Malcolm", "Daniel", "Chris",
    # The Office — generic character first names
    "Michael", "Pam", "Ryan", "Kevin", "Kelly", "Meredith",
    "Stanley", "Toby", "Darryl", "Erin", "Creed", "Oscar", "Phyllis",
    # Real actor first names
    "Steve", "Rainn", "Jenna", "Mindy", "Ellie", "Rashida", "Ed",
    # Classic fiction characters (generic)
    "Clarisse", "Doug", "Travis", "Charlie", "Will", "Faber",
    "Rick", "Palmer", "Glen", "Isidore", "Bob", "Donna", "Juliana",
    "Manfred", "Leo",
    # Richard Pryor characters
    "Gus", "Monty", "Duane", "Rufus", "Leroy", "Skip", "Grover",
    # Robin Williams (additional)
    "Peter", "Sailor", "Djinn",
]

_LAST_NAMES = [
    "Stone", "Shaw", "Cross", "Wells", "Steele", "Frost", "Pierce", "Vaughn",
    "Black", "Drake", "Hayes", "Kane", "Voss", "Cranston", "Kendall", "Reeves",
    "Volkov", "Sato", "Tanaka", "Okafor", "Diallo", "Sirikit", "Petrov",
    # Generic last names (scrubbed franchise-specific)
    "Burns", "Hibbert", "Flanders", "Houten", "Smithers",
    "Terwilliger", "Bouvier", "Simpson", "Gordon", "Ming",
    "Carruthers", "Corben",
    # The Office — character last names (generic ones only)
    "Scott", "Halpert", "Beesly", "Howard", "Bernard", "Malone",
    "Kapoor", "Palmer", "Hudson", "Martin", "Flenderson", "Philbin", "Vance",
    # Ray Bradbury (generic)
    "Beatty", "Spender", "Stendahl", "Eckels", "Halloway",
    # Misc classic (generic)
    "Steiner",
]

# Trait pools for procedural character profiles
_GENDERS = ["male", "female"]
_AGE_BRACKETS = ["20s", "30s", "40s", "50s", "60s"]
_DEMEANORS = [
    "calm", "intense", "warm", "sharp", "dry", "energetic",
    "measured", "wry", "stoic", "anxious", "confident", "weary",
]

# Accent pool — 100% English-native presets only.
# Foreign presets (de_speaker, fr_speaker, etc.) caused Bark hallucinations:
# the model generates foreign-language phonemes when given English text,
# producing gibberish instead of accented English. Until Bark's multilingual
# stability improves, all characters use en_speaker_* presets.
# See: v1.1 "Test Signal" critique — Lemmy (de_speaker_0) was unintelligible.
_ACCENTS = [
    ("neutral",  "en", 1.00),   # English-only — no foreign presets
]

# Voice presets mapped by gender + vocal quality + language code.
# English-native presets (en_speaker_*) have known vocal qualities.
# International presets (xx_speaker_*) are grouped by speaker index tendencies.
# Each entry: (preset, gender, quality_tags)
_VOICE_PROFILES = [
    # ── English native (neutral accent) ──
    ("v2/en_speaker_0", "male",   "en", {"authoritative", "deep", "50s", "60s", "announcer", "commander"}),
    ("v2/en_speaker_1", "male",   "en", {"calm", "measured", "30s", "40s", "technical", "pilot"}),
    ("v2/en_speaker_3", "male",   "en", {"energetic", "sharp", "20s", "30s", "rebel", "technician"}),
    ("v2/en_speaker_5", "male",   "en", {"warm", "weary", "wry", "50s", "60s", "doctor", "scientist"}),
    ("v2/en_speaker_6", "male",   "en", {"intense", "dry", "stoic", "40s", "officer", "android"}),
    ("v2/en_speaker_8", "male",   "en", {"gravelly", "anxious", "confident", "40s", "50s", "engineer", "mechanic"}),
    # English native (female)
    ("v2/en_speaker_2", "female", "en", {"clipped", "precise", "30s", "40s", "officer", "neutral-british"}), # Sounds precise/British-adjacent
    ("v2/en_speaker_4", "female", "en", {"warm", "energetic", "wry", "30s", "40s", "pilot", "explorer"}),
    ("v2/en_speaker_9", "female", "en", {"authoritative", "confident", "intense", "50s", "60s", "commander", "senator"}),
    # FIX-3 (v1.2): en_speaker_7 reclassified to female to prevent CAST_GENDER_POOL_EXHAUSTED
    # on 3-female episodes (was causing VEX/ZARA to share en_speaker_9 and sound identical).
    # Bark labels en_speaker_7 as androgynous — in English it reads soft/lighter so we
    # use it as the "younger" female slot (20s, anxious/sharp/technician).
    ("v2/en_speaker_7", "female", "en", {"sharp", "anxious", "nervous", "20s", "30s", "technician", "hacker"}),
    # ── DISABLED: Foreign accent presets ──────────────────────────────
    # These caused Bark hallucinations — the model generates foreign-language
    # phonemes when fed English text, producing gibberish. Kept as comments
    # for future reference if Bark's multilingual stability improves.
    # See v1.1 "Test Signal" critique: de_speaker_0 (Lemmy) was unintelligible,
    # fr_speaker lines also showed artifacts.
    #
    # German:  de_speaker_0/3/5 (male), de_speaker_2/7 (female)
    # Spanish: es_speaker_0/6/8 (male), es_speaker_4/9 (female)
    # French:  fr_speaker_1/5 (male), fr_speaker_2/4 (female)
    # Indian:  hi_speaker_0/5 (male), hi_speaker_4/9 (female)
    # Italian: it_speaker_0/6 (male), it_speaker_4/9 (female)
    # Japanese: ja_speaker_1/6 (male), ja_speaker_4 (female)
    # Korean:  ko_speaker_0 (male), ko_speaker_4 (female)
    # Russian: ru_speaker_0/3 (male), ru_speaker_4/9 (female)
    # Brazilian: pt_speaker_0 (male), pt_speaker_4 (female)
    # Polish:  pl_speaker_0 (male), pl_speaker_4 (female)
]

# ANNOUNCER voice pool — randomized per episode for gender balance (50/50 male/female)
# ANNOUNCER always uses neutral English (en_speaker_*) — no accent
_ANNOUNCER_PRESETS = [
    ("v2/en_speaker_0", "Male, authoritative, deep"),
    ("v2/en_speaker_1", "Male, measured, calm"),
    ("v2/en_speaker_4", "Female, warm, energetic"),
    ("v2/en_speaker_9", "Female, mature, authoritative"),
]

# LEMMY fixed profile — always gravelly/raspy male, English-native preset
_LEMMY_PROFILE = {
    "name": "LEMMY",
    "gender": "male",
    "age": "50s",
    "demeanor": "gravelly",
    "accent": "neutral",  # English-native preset; gravelly tone comes from en_speaker_8 vocal quality
    "voice_preset": "v2/en_speaker_8",  # English native — gravelly, confident, 40s-50s. Avoids Bark hallucination from de_speaker
    "notes": "Male, gravelly/raspy, 50s, gruff mechanic voice, iconic",
}


def _pick_accent(rng) -> tuple:
    """Weighted random accent selection. Returns (accent_label, lang_code).

    ~60% neutral English, ~40% spread across international accents.
    Uses cumulative distribution for deterministic weighted selection.
    """
    roll = rng.random()
    cumulative = 0.0
    for label, code, weight in _ACCENTS:
        cumulative += weight
        if roll < cumulative:
            return label, code
    # Fallback (rounding errors)
    return "neutral", "en"


def _generate_character_profile(character_idx: int, episode_seed: str = "",
                                gender_hint: str = None) -> dict:
    """Generate a full procedural character profile — deterministic per episode.

    Returns a dict with: name, gender, age, demeanor, accent, voice_preset, notes.
    All traits are seeded so reruns of the same episode produce identical casts.

    Voice preset selection:
      1. Pick gender, age, demeanor, accent procedurally
      2. Filter voice profiles by gender AND accent language code
      3. Score by trait overlap (age, demeanor)
      4. Best match wins (ties broken by RNG shuffle)

    Safety rails (downstream):
      - ASCII sanitizer strips non-ASCII from all text before Bark
      - Temperature capped at 0.55 for international presets
      - All dialogue is always written in pure English
    """
    rng = random.Random(f"{episode_seed}_char_{character_idx}")

    # Generate name
    first = rng.choice(_FIRST_NAMES)
    last = rng.choice(_LAST_NAMES)
    name = f"{first} {last}".upper()

    # Generate traits — honor gender_hint from script's [VOICE: NAME, gender, ...] tag
    # if provided. This is BUG-004 fix: previously the procedural cast picked random
    # genders, producing male voices on female characters and vice versa.
    if gender_hint and gender_hint.lower() in ("male", "female"):
        gender = gender_hint.lower()
    else:
        gender = rng.choice(_GENDERS)
    age = rng.choice(_AGE_BRACKETS)
    demeanor = rng.choice(_DEMEANORS)
    accent_label, lang_code = _pick_accent(rng)

    # Filter voice profiles by gender AND language code
    candidates = [vp for vp in _VOICE_PROFILES
                  if vp[1] == gender and vp[2] == lang_code]

    # If no match for this gender+accent combo, fall back to same-gender English
    if not candidates:
        candidates = [vp for vp in _VOICE_PROFILES
                      if vp[1] == gender and vp[2] == "en"]
        accent_label = "neutral"
        lang_code = "en"

    # Safety net — should never happen
    if not candidates:
        candidates = [vp for vp in _VOICE_PROFILES if vp[2] == "en"]

    # Score each candidate by how many tags overlap with character traits
    char_tags = {age, demeanor}
    scored = []
    for preset, _, _, tags in candidates:
        overlap = len(char_tags & tags)
        scored.append((overlap, preset, tags))

    # Sort by overlap (best match first), break ties with RNG shuffle
    rng.shuffle(scored)
    scored.sort(key=lambda x: x[0], reverse=True)
    best_preset = scored[0][1]

    # Build descriptive notes
    accent_note = f", {accent_label} accent" if accent_label != "neutral" else ""
    notes = f"{gender.capitalize()}, {demeanor}, {age}{accent_note}"

    return {
        "name": name,
        "gender": gender,
        "age": age,
        "demeanor": demeanor,
        "accent": accent_label,
        "voice_preset": best_preset,
        "notes": notes,
    }


def _generate_announcer_profile(episode_seed: str = "", gender_hint: str | None = None) -> dict:
    """Pick a Announcer voice from the balanced pool, seeded per episode.
    If gender_hint is provided (from script [VOICE: ANNOUNCER, gender, ...] tag),
    filter the pool to matching gender first; fall back to full pool if none match.
    ANNOUNCER always uses neutral English — no accent."""
    rng = random.Random(f"{episode_seed}_announcer")
    pool = _ANNOUNCER_PRESETS
    if gender_hint:
        gh = gender_hint.lower()
        # Use startswith to avoid "male" matching inside "female"
        filtered = [(p, n) for p, n in _ANNOUNCER_PRESETS
                    if n.lower().startswith(gh)]
        if filtered:
            pool = filtered
    preset, notes = rng.choice(pool)
    return {
        "name": "ANNOUNCER",
        "voice_preset": preset,
        "notes": notes,
    }


# ─────────────────────────────────────────────────────────────────────────────
# NEWS FETCHER — pulls real science headlines to seed the story
# ─────────────────────────────────────────────────────────────────────────────

SCIENCE_NEWS_FEEDS = [
    # ── Open-access: full article text fetchable, no paywall ──
    "https://www.sciencedaily.com/rss/all.xml",           # Best: full articles, open
    "https://www.eurekalert.org/rss/technology_engineering.xml",  # Press releases, open
    "https://www.eurekalert.org/rss/space.xml",           # Press releases, open
    "https://www.eurekalert.org/rss/biology.xml",         # Press releases, open
    "https://www.eurekalert.org/rss/chemistry_physics.xml", # Press releases, open
    "https://www.eurekalert.org/rss/earth_environment.xml", # Press releases, open
    # ── Government / institutional (fully open) ──
    "https://www.nasa.gov/rss/dyn/breaking_news.rss",     # NASA, open
    "https://www.nih.gov/news-events/news-releases.xml",  # NIH, open
    "https://www.nsf.gov/rss/rss_www_news.xml",           # NSF, open
    # ── UCLA Newsroom (open-access institutional research) ──
    "https://newsroom.ucla.edu/cats/health_+_behavior.xml",      # Best: full content:encoded in RSS
    "https://newsroom.ucla.edu/cats/science_+_technology.xml",   # Open-access, URL scrape works
    "https://newsroom.ucla.edu/cats/environment_+_climate.xml",  # Open-access, URL scrape works
    # ── Open journalism (full text accessible) ──
    "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",  # BBC, open
    "https://feeds.arstechnica.com/arstechnica/science",  # Ars, open
    "https://theconversation.com/us/science/rss",         # The Conversation, open
    "https://cosmosmagazine.com/feed/",                   # Cosmos, open
]


def _fetch_full_article(url, timeout=20):
    """Fetch the full text of a science article from its URL.

    Uses requests + BeautifulSoup to strip HTML boilerplate and extract
    the article body. Returns the raw text (up to 12000 chars) so Gemma
    gets real science content — methodology, findings, implications —
    not just the RSS teaser. Falls back to empty string on any failure
    (paywalls, bot blocks, timeouts) so the caller can degrade gracefully.

    The scraper tries a cascade of CSS selectors before falling back to
    the full document, so it handles sites that don't use semantic
    <article>/<main> tags (e.g. UCLA Newsroom, institutional press pages).
    """
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        return ""

    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; OTR-ScriptBot/1.0)"}
        resp = requests.get(url, timeout=timeout, headers=headers)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Strip boilerplate — nav, ads, footer, sidebar, scripts
        for tag in soup(["script", "style", "nav", "footer", "header",
                          "aside", "form", "noscript", "iframe"]):
            tag.decompose()

        # Cascade of content selectors — most specific to least.
        # Covers: semantic HTML5, WordPress/CMS class conventions,
        # institutional press release pages (UCLA, NIH, NSF, EurekaAlert).
        _SELECTORS = [
            "article",
            "main",
            '[class*="article-body"]',
            '[class*="article__body"]',
            '[class*="story-body"]',
            '[class*="entry-content"]',
            '[class*="post-content"]',
            '[class*="content-body"]',
            '[class*="wysiwyg"]',
            '[class*="rich-text"]',
            '[class*="body-copy"]',
            '[class*="release-body"]',      # EurekaAlert press releases
            '[class*="article-content"]',
            '[id*="article-body"]',
            '[id*="main-content"]',
            "div.content",
            "div.body",
        ]

        body = None
        for selector in _SELECTORS:
            body = soup.select_one(selector)
            if body:
                break
        if body is None:
            body = soup  # last resort — full stripped document

        # Extract paragraphs AND headings — h2/h3 carry section context
        # (methodology, implications, researcher quotes) that's often the
        # richest science content buried past the lede.
        content_tags = body.find_all(["p", "h2", "h3"])
        text = " ".join(tag.get_text(" ", strip=True) for tag in content_tags)
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:12000]
    except Exception:
        return ""


def _fetch_science_news(max_feeds=10):
    """Fetch science stories from multiple RSS feeds in parallel.

    Uses ThreadPoolExecutor to hit all feeds simultaneously, dramatically
    reducing the wait time when feeds are slow or unresponsive. Each feed
    has its own timeout.
    """
    try:
        import feedparser
    except ImportError:
        msg = (
            "╔══════════════════════════════════════════════════════════════════╗\n"
            "║  CRITICAL: feedparser is missing.                              ║\n"
            "║  Run `pip install feedparser` to enable live science news.     ║\n"
            "║  The OTR ScriptWriter REQUIRES real headlines — no fallback.   ║\n"
            "╚══════════════════════════════════════════════════════════════════╝"
        )
        log.error(msg)
        raise ImportError(msg)

    def _fetch_single_feed(feed_url):
        data = []
        FEED_TIMEOUT = 7
        try:
            # Set socket timeout locally for this thread
            _prev_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(FEED_TIMEOUT)
            try:
                feed = feedparser.parse(feed_url)
            finally:
                socket.setdefaulttimeout(_prev_timeout)

            for entry in feed.entries[:6]:
                title = entry.get("title", "").strip()
                if not title:
                    continue

                # ── Headline pre-filter: reject non-article content ──────────
                # Teaser/media headlines have no science payload for Gemma to
                # work with. A podcast slug or video title gives Gemma 90 chars
                # about content it can't access. Drop these at the source.
                _SKIP_PREFIXES = (
                    "podcast:", "watch:", "video:", "listen:", "opinion:",
                    "q&a:", "quiz:", "gallery:", "photos:", "slideshow:",
                    "live:", "webinar:", "event:", "in photos:",
                )
                _SKIP_PHRASES = (
                    "behind-the-scenes", "tour of", "in conversation with",
                    "ask the expert", "meet the", "alumni spotlight",
                    "faculty spotlight", "student spotlight", "donate",
                    "how to apply", "registration open",
                )
                title_lower = title.lower()
                if any(title_lower.startswith(p) for p in _SKIP_PREFIXES):
                    log.debug("[NewsFetcher] Skipping non-article (prefix): %s", title[:60])
                    continue
                if any(p in title_lower for p in _SKIP_PHRASES):
                    log.debug("[NewsFetcher] Skipping non-article (phrase): %s", title[:60])
                    continue
                # ─────────────────────────────────────────────────────────────

                content_candidates = entry.get("content", [])
                rss_full = ""
                if content_candidates:
                    rss_full = content_candidates[0].get("value", "")
                    rss_full = re.sub(r'<[^>]+>', '', rss_full).strip()
                summary = entry.get("summary", "")[:2000].strip()
                summary = re.sub(r'<[^>]+>', '', summary).strip()
                data.append({
                    "headline": title,
                    "summary": summary,
                    "rss_full": rss_full,
                    "source": feed.feed.get("title", feed_url.split("/")[2]),
                    "date": entry.get("published", str(datetime.now().date())),
                    "link": entry.get("link", ""),
                })
            return data
        except Exception as e:
            log.debug("[NewsFetcher] Feed failed %s: %s", feed_url, e)
            return []

    pool = []
    feeds_hit = 0
    shuffled_feeds = SCIENCE_NEWS_FEEDS[:]
    random.shuffle(shuffled_feeds)

    log.info("[NewsFetcher] Starting parallel fetch from %d sources...", len(shuffled_feeds))
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=len(shuffled_feeds)) as executor:
        futures = {executor.submit(_fetch_single_feed, url): url for url in shuffled_feeds}
        for future in as_completed(futures):
            results = future.result()
            if results:
                pool.extend(results)
                feeds_hit += 1

    fetch_time = time.time() - start_time
    log.info("[NewsFetcher] Parallel fetch complete in %.2fs. Pool: %d headlines from %d feeds.",
             fetch_time, len(pool), feeds_hit)

    if not pool:
        log.error("[NewsFetcher] ALL feeds failed — check network connectivity")
        raise RuntimeError(
            "No science headlines could be fetched. Check your internet connection. "
            "The OTR ScriptWriter requires live RSS feeds to generate scripts."
        )

    # Shuffle pool before choosing
    random.shuffle(pool)

    # Content quality floor: try up to 5 candidates until we get a rich article.
    # Thin content (<400 chars) gives Gemma too little to extrapolate from —
    # the story ends up generic rather than grounded in real science.
    CONTENT_FLOOR = 400
    MAX_ATTEMPTS = 5
    chosen = None

    for candidate in pool[:MAX_ATTEMPTS]:
        result = candidate

        # Resolve full article text — 3-tier: rss_full → URL scrape → summary
        if result.get("rss_full") and len(result["rss_full"]) > 300:
            result["full_text"] = result["rss_full"][:12000]
            log.info("[NewsFetcher] Full text from RSS content field: %d chars", len(result["full_text"]))
        elif result.get("link"):
            log.info("[NewsFetcher] Attempting to fetch full article: %s", result["link"])
            fetched = _fetch_full_article(result["link"], timeout=5)
            if fetched and len(fetched) > 300:
                result["full_text"] = fetched
                log.info("[NewsFetcher] Full article fetched: %d chars", len(result["full_text"]))
            else:
                result["full_text"] = result["summary"]
                log.info("[NewsFetcher] Article fetch failed or blocked — falling back to RSS summary (%d chars)", len(result["full_text"]))
        else:
            result["full_text"] = result["summary"]
            log.info("[NewsFetcher] No URL — using RSS summary (%d chars)", len(result["full_text"]))

        if len(result.get("full_text", "")) >= CONTENT_FLOOR:
            chosen = result
            break
        else:
            log.warning("[NewsFetcher] Article too thin (%d chars) — trying next candidate: %s",
                        len(result.get("full_text", "")), result["headline"][:60])

    if chosen is None:
        # All candidates were thin — take the richest one we found
        chosen = max(pool[:MAX_ATTEMPTS], key=lambda x: len(x.get("full_text", x.get("summary", ""))))
        chosen.setdefault("full_text", chosen.get("summary", ""))
        log.warning("[NewsFetcher] All %d candidates were thin — using richest available (%d chars): %s",
                    MAX_ATTEMPTS, len(chosen["full_text"]), chosen["headline"][:60])

    return [chosen]


# ─────────────────────────────────────────────────────────────────────────────
# LLM INFERENCE WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

def _load_llm(model_id_full="google/gemma-4-E4B-it", device="cuda", optimization_profile="Standard"):
    # Strip [BETA] or [8-bit] labels used in the UI dropdown
    model_id = model_id_full.split(" ")[0]

    # Pre-emptive VRAM sanitation is now handled at the node entry points
    # for better visibility and consistency.

    """Load LLM via transformers. Caches globally with device tracking.

    BEST PRACTICES applied (per survival guide):
      - Section 3:  Lazy loading — never load at import time
      - Section 5:  Device/dtype alignment
      - Section 34: Cache invalidation on device change
      - Section 40: Manual VRAM management since we're outside ComfyUI model registry
      - Section 47: No device_map="auto" (conflicts with ComfyUI's torch.set_default_device)
      - Section 49: No trust_remote_code=True (Gemma is natively supported)
    """
    global _LLM_CACHE

    # Check for device change OR quantization mismatch OR budget profile mismatch
    is_obsidian = "Obsidian" in optimization_profile
    
    # v1.4: Centralized "Large Model" Tags for VRAM Hardening
    # These models MUST be quantized to fit within flagship (16GB) or ultra-lite (4GB) targets.
    vram_safe_tags = ("9b", "12b", "14b", "24b", "26b", "27b", "31b", "70b", "e4b", "4b-it", "a4b", "2b", "2b-it", "efficiency", "nemo", "qwen", "mistral", "instruct", "gemma")
    
    requested_quantized = is_obsidian or "4-bit" in model_id_full.lower() or \
                          any(tag in model_id_full.lower() for tag in vram_safe_tags)

    # v1.4 Audit: Also check if the model object itself has been moved to CPU
    current_model_device = "cpu"
    if _LLM_CACHE["model"] is not None:
        try:
            current_model_device = str(next(_LLM_CACHE["model"].parameters()).device)
        except StopIteration:
            pass

    if (_LLM_CACHE["model"] is not None and
            (str(_LLM_CACHE["device"]) != str(device) or 
             _LLM_CACHE["quantized"] != requested_quantized or
             _LLM_CACHE["model_id"] != model_id or
             _LLM_CACHE.get("budget_profile") != optimization_profile or
             _LLM_CACHE.get("VERSION") != "v1.4" or
             ("cuda" in str(device) and "cpu" in current_model_device))):
        _runtime_log(f"Gemma cache mismatch or version update (v1.4) — reloading to enforce budget")
        _unload_llm()

    if _LLM_CACHE["model"] is None:
        log.info(f"Loading LLM model: {model_id} (quantized={requested_quantized})")
        try:
            # Lazy import — only pay the cost when actually generating
            import torch
            from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer

            # ── Zero-Prime VRAM Hardening (v1.4) ──
            # We MUST detect hardware and purge memory BEFORE loading even the Tokenizer
            # to prevent the 15GB transient spike on 16GB cards.
            
            # Detect Hardware
            total_vram = 0
            if torch.cuda.is_available():
                total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
            # Nuclear Power Wash (Global Eviction)
            try:
                import comfy.model_management
                comfy.model_management.unload_all_models()
                comfy.model_management.soft_empty_cache()
                _runtime_log("[StoryOrchestrator] Zero-Prime: ComfyUI Models Evicted.")
            except: pass

            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            
            # Post-Wash Analytics
            if torch.cuda.is_available():
                free_gb = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)) / (1024**3)
                _runtime_log(f"[StoryOrchestrator] Zero-Prime VRAM State: {free_gb:.1f}GB Free. Capacity: {total_vram:.1f}GB")

            # ── VRAM Budgeting (Early Allocation) ──
            max_memory = None
            is_actually_2b = any(tag in model_id.lower() for tag in ("2b-it", "2b_it")) or model_id.lower().endswith("2b")
            
            if total_vram >= 12.0:
                # FLAGSHIP 2GB SOVEREIGNTY BUFFER
                budget_gb = total_vram - 2.0
                max_memory = {0: f"{budget_gb:.1f}GiB", "cpu": "32GiB"}
                _runtime_log(f"[StoryOrchestrator] Sovereignty Buffer Active: {budget_gb:.1f}GB Budget")
            elif is_actually_2b:
                max_memory = {0: "3.2GiB", "cpu": "32GiB"}
            elif any(tag in model_id.lower() for tag in ("9b", "12b", "e4b", "4b-it")):
                max_memory = {0: "6.8GiB", "cpu": "32GiB"}

            # Enable TF32 for faster matmuls on Ampere/Ada/Blackwell GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            # ── VRAM Hardening v1.4: Strict Handoff ──
            # If Bark is in VRAM, evict it now before loading LLM.
            try:
                from .bark_tts import _unload_bark
                _unload_bark()
            except ImportError:
                pass
            except Exception as handoff_err:
                log.warning("[StoryOrchestrator] Bark handoff failed: %s", handoff_err)

            is_gemma = "gemma" in model_id.lower()

            try:
                # v1.4 FIX: Revert to AutoTokenizer. AutoProcessor was causing 
                # decode offsets to fail on non-multimodal 2B models.
                tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
            except OSError as local_err:
                log.info("[StoryOrchestrator] local_files_only=True failed for tokenizer (%s)", local_err)
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_id)
                except Exception as hub_err:
                    log.error("[StoryOrchestrator] Hub fallback failed. Ensure model is downloaded or Hub is reachable: %s", hub_err)
                    raise RuntimeError(f"Failed to load Tokenizer '{model_id}'. Is it downloaded? Hub error: {hub_err}") from hub_err

            # Using bfloat16 for maximum speed on RTX 5000-series (Ada/Blackwell) GPUs.
            load_dtype = torch.bfloat16

            # ── Flash Attention 2 (preferred) → SDPA fallback ──
            # Flash Attention 2 gives ~40% speedup but requires `pip install flash-attn`.
            # If unavailable, fall back to SDPA which is still fast.
            # Verify the flash-attn DISTRIBUTION is installed (not just an importable
            # stub). Transformers checks PACKAGE_DISTRIBUTION_MAPPING['flash_attn'] and
            # raises KeyError if the distribution metadata is missing, even if `import
            # flash_attn` succeeds. Use importlib.metadata to be authoritative.
            attn_impl = "sdpa"
            try:
                from importlib.metadata import distribution, PackageNotFoundError
                try:
                    distribution("flash-attn")
                    import flash_attn  # noqa: F401
                    attn_impl = "flash_attention_2"
                    log.info("[StoryOrchestrator] Flash Attention 2 available — using flash_attention_2")
                except (PackageNotFoundError, ImportError):
                    log.info(
                        "[StoryOrchestrator] Flash Attention 2: NOT AVAILABLE — no prebuilt wheel exists "
                        "for torch 2.10 + CUDA 13 + Blackwell sm_120 on Windows. "
                        "SageAttention + SDPA active. Performance unaffected. Do not attempt install."
                    )
            except Exception as _fa_err:
                log.info("[StoryOrchestrator] FA2 probe failed (%s) — using SDPA fallback", _fa_err)

            # ── 4-bit quantization (forced for Obsidian or large models) ──
            # The Obsidian profile mandates 4-bit to fit on 4GB-8GB GPUs.
            # Large models (26B+) also require 4-bit to fit in 16GB.
            quant_config = None
            needs_8bit = "8-bit" in model_id_full.lower()
            
            # v1.4 Universal Hardening: Centralized tags
            is_unstable_quant = any(tag in model_id_full.lower() for tag in ("2bit", "3bit", "2-bit", "3-bit"))
            needs_4bit = requested_quantized or is_unstable_quant or \
                         any(tag in model_id_full.lower() for tag in vram_safe_tags)

            if is_unstable_quant:
                _runtime_log(f"[StoryOrchestrator] 🛡️ WING DING PROTECTION: Unstable Bit-Depth ({model_id_full}) UPGRADED to 4-bit NF4")
            elif needs_4bit and not requested_quantized:
                _runtime_log(f"[StoryOrchestrator] VRAM SAFETY: Forcing 4-bit (NF4) for Large Model ({model_id_full})")

            if needs_8bit:
                try:
                    from transformers import BitsAndBytesConfig
                    quant_config = BitsAndBytesConfig(load_in_8bit=True)
                    log.info("[StoryOrchestrator] Enabling 8-bit quantization")
                except ImportError:
                    log.warning("[StoryOrchestrator] Large model but bitsandbytes not installed!")
            elif needs_4bit:
                try:
                    from transformers import BitsAndBytesConfig
                    quant_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
                    log.info("[StoryOrchestrator] 🚀 Enabling 4-bit quantization (NF4) for Ultra-Low VRAM")
                    _runtime_log("[StoryOrchestrator] 🚀 4-bit NF4 active")
                except ImportError:
                    log.warning("[StoryOrchestrator] Large model but bitsandbytes not installed — "
                                "loading at bfloat16 may OOM. Run: pip install bitsandbytes")

            from transformers import AutoTokenizer, AutoModelForCausalLM

            cache_dir_path = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
            try:
                # v1.4 Hardening: Explicitly trust_remote_code=False for flagship security
                tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True, trust_remote_code=False, cache_dir=cache_dir_path)
                _runtime_log("LLM tokenizer loaded from cache (no HTTP checks)")
            except Exception as local_err:
                _runtime_log(f"[StoryOrchestrator] local_files_only=True failed for tokenizer ({local_err}), attempting Hub fallback...")
                tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False, cache_dir=cache_dir_path)

            common_kwargs = dict(
                cache_dir=cache_dir_path,
                trust_remote_code=False,  # v1.4 Hardening: strict security
                low_cpu_mem_usage=True,
                attn_implementation="sdpa"  # v1.4 Blackwell/Flash-Replacement
            )
            if quant_config is not None:
                common_kwargs["quantization_config"] = quant_config
                
                # FLAGSHIP 16GB OVERRIDE: 2B models fit easily. Force GPU-only to avoid CPU intermediate bloat.
                if total_vram >= 15.0 and is_actually_2b:
                    common_kwargs["device_map"] = {"": 0} 
                    _runtime_log("[StoryOrchestrator] Flagship Override: Forcing CUDA:0 (Direct GPU) for 2B model")
                else:
                    common_kwargs["device_map"] = "auto"
            elif max_memory is not None:
                common_kwargs["device_map"] = "auto"

            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    local_files_only=True,
                    **common_kwargs,
                )
                _runtime_log("LLM model loaded from cache (no HTTP checks)")
            except (OSError, ValueError) as local_err:
                _runtime_log(f"[StoryOrchestrator] local_files_only=True failed for model ({local_err}), attempting Hub fallback...")
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        **common_kwargs,
                    )
                except Exception as hub_err:
                    log.error("[StoryOrchestrator] Hub fallback failed. Ensure model is downloaded or Hub is reachable: %s", hub_err)
                    raise RuntimeError(f"Failed to load LLM model '{model_id}'. Is it downloaded? Hub error: {hub_err}") from hub_err

            if quant_config is None and max_memory is None:
                model = model.to(device)
            model = model.eval()

            _LLM_CACHE["model"] = model
            _LLM_CACHE["tokenizer"] = tokenizer
            _LLM_CACHE["device"] = device
            _LLM_CACHE["quantized"] = (quant_config is not None)
            _LLM_CACHE["model_id"] = model_id
            _LLM_CACHE["budget_profile"] = optimization_profile
            _LLM_CACHE["VERSION"] = "v1.4"
            actual_quant = (quant_config is not None)
            _runtime_log(f"LLM loaded: {model_id} (quantized={actual_quant}, budget={optimization_profile}) [v1.4]")
        except Exception as e:
            log.exception("Failed to load LLM: %s", e)  # Section 49: log.exception for full traceback
            raise
    return _LLM_CACHE["model"], _LLM_CACHE["tokenizer"]


# Bounded model cache with device tracking (Section 34)
_LLM_CACHE = {"model": None, "tokenizer": None, "device": None, "quantized": False, "model_id": None, "budget_profile": None, "VERSION": "v1.4"}


def _unload_llm():
    """Explicitly unload LLM to free VRAM (v1.3.1 OOM FIX).

    The v1.3 version did del + gc.collect() + empty_cache(), but that
    is a no-op when abandoned worker threads from _run_with_timeout
    still hold the model as a local variable in their stack frame.
    Symptom: second load attempt sees VRAM at 31.70 GiB on a 16 GB
    card because the first model never actually left the GPU.

    The fix is to call model.cpu() BEFORE dropping references. That
    moves the weights from VRAM to RAM immediately, even when other
    strong refs exist. Abandoned generate() threads will then error
    out on device mismatch, which is acceptable because their results
    are already being discarded by the timeout fallback path.

    Order of operations is load-bearing:
      1. model.cpu()         — move weights off GPU even with live refs
      2. del cache entries   — drop the primary reference
      3. gc.collect()        — destroy the object if no other refs
      4. empty_cache()       — return freed VRAM to the allocator
      5. telemetry           — prove VRAM actually dropped
    """
    global _LLM_CACHE
    import gc
    import torch
    if _LLM_CACHE["model"] is not None:
        # Step 1: force weights off GPU before dropping the reference.
        try:
            _LLM_CACHE["model"].cpu()
        except Exception as cpu_err:
            log.warning("[StoryOrchestrator] model.cpu() during unload failed: %s", cpu_err)
        # Step 2: drop references from the module-level cache.
        del _LLM_CACHE["model"]
        del _LLM_CACHE["tokenizer"]
        _LLM_CACHE = {"model": None, "tokenizer": None, "device": None, "quantized": False, "model_id": None}
        # Step 3 + 4: gc and return VRAM to the allocator.
        gc.collect()
        torch.cuda.empty_cache()
        
        # Evict from ComfyUI's internal cache tracking as well
        try:
            import comfy.model_management
            comfy.model_management.soft_empty_cache()
        except Exception:
            pass

        # Step 5: telemetry — prove it worked.
        allocated_gib = torch.cuda.memory_allocated() / 1e9
        reserved_gib = torch.cuda.memory_reserved() / 1e9
        log.info(
            "LLM unloaded: VRAM allocated=%.2f GiB reserved=%.2f GiB "
            "(cpu + gc.collect + empty_cache)",
            allocated_gib, reserved_gib,
        )

# Register the LLM unloader with the VRAM Power Wash system so that
# force_vram_offload() at node entry points also evicts Gemma.
from ._vram_log import register_vram_cleanup
register_vram_cleanup(_unload_llm)


class GemmaHeartbeatStreamer(BaseStreamer):
    """Custom streamer that pulses heartbeats to _runtime_log for Canonical Tokens.

    Hooks into model.generate() at the token level. Every time Gemma completes
    a line that contains a recognizable script tag (=== SCENE, [VOICE:], [SFX:],
    [ENV:], (beat)), it writes a timestamped entry to otr_runtime.log immediately.

    Also tracks:
      - Scene count (how many === SCENE === tags so far)
      - Dialogue line count (how many [VOICE:] tags)
      - Unique character names seen
      - Token generation speed (tokens/sec, reported every 100 tokens)

    The OTR Monitor (otr_monitor.py) tails otr_runtime.log and folds these
    heartbeats into the live dashboard — so you can watch the script being
    written in real time without touching ComfyUI.
    """

    def __init__(self, tokenizer, skip_prompt=False, **decode_kwargs):
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.decode_kwargs = decode_kwargs
        self.token_cache = []
        self.on_prompt_end = True
        self.print_streamer = TextStreamer(tokenizer, skip_prompt=skip_prompt, **decode_kwargs)

        # Counters for the dashboard
        self.scene_count = 0
        self.dialogue_count = 0
        self.sfx_count = 0
        self.characters_seen = set()

        # Token speed tracking
        self.total_tokens = 0
        self._start_time = time.time()
        self._last_speed_report = 0

    def put(self, value):
        """Processes a new batch of tokens."""
        # Check strict streaming timeout to prevent VRAM leaks from abandoned threads
        if hasattr(_TIMEOUT_CTX, "deadline") and time.time() > _TIMEOUT_CTX.deadline:
            raise TimeoutError("Streaming deadline exceeded — gracefully aborting generator")

        # Standard console output
        self.print_streamer.put(value)

        # Heartbeat logic
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("GemmaHeartbeatStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]

        if self.skip_prompt and self.on_prompt_end:
            self.on_prompt_end = False
            return

        token_list = value.tolist()
        self.token_cache.extend(token_list)
        self.total_tokens += len(token_list)

        text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)

        # If a newline is detected, process the completed line
        if text.endswith("\n") or text.endswith("\r"):
            self._process_line(text.strip())
            self.token_cache = []

        # Report speed every 100 tokens
        if self.total_tokens - self._last_speed_report >= 100:
            elapsed = time.time() - self._start_time
            if elapsed > 0:
                tps = self.total_tokens / elapsed
                _runtime_log(
                    f"ScriptWriter: {self.total_tokens} tokens | "
                    f"{tps:.1f} tok/s | {self.scene_count} scenes | "
                    f"{self.dialogue_count} lines | "
                    f"{len(self.characters_seen)} chars"
                )
                self._last_speed_report = self.total_tokens

    def end(self):
        """Flush the remaining buffer and report final stats."""
        self.print_streamer.end()
        if self.token_cache:
            text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)
            self._process_line(text.strip())
        self.token_cache = []

        elapsed = time.time() - self._start_time
        tps = self.total_tokens / elapsed if elapsed > 0 else 0
        _runtime_log(
            f"ScriptWriter DONE: {self.total_tokens} tokens in {elapsed:.1f}s "
            f"({tps:.1f} tok/s) | {self.scene_count} scenes | "
            f"{self.dialogue_count} dialogue lines | "
            f"Characters: {', '.join(sorted(self.characters_seen)) or 'none'}"
        )

    def _process_line(self, line):
        """Detect Canonical Tokens, update counters, pulse the heartbeat."""
        if not line:
            return

        # ── Scene break: === SCENE X === ─────────────────────────────
        if "===" in line:
            self.scene_count += 1
            _runtime_log(f"ScriptWriter: {line.strip()}")
            return

        # ── Voice tag: [VOICE: NAME, traits] dialogue ────────────────
        if "[VOICE:" in line.upper():
            self.dialogue_count += 1
            try:
                # Case-insensitive tag extraction
                line_up = line.upper()
                start_idx = line_up.find("[VOICE:") + 7
                end_idx = line.find("]", start_idx)
                tag_content = line[start_idx:end_idx]
                
                name = tag_content.split(",", 1)[0].strip().upper()
                self.characters_seen.add(name)
                
                # Dynamic log update
                clean_dialogue = line[end_idx+1:].strip()[:60]
                _runtime_log(f"ScriptWriter: [{self.dialogue_count}] {name}: {clean_dialogue}")
            except (IndexError, ValueError):
                _runtime_log(f"ScriptWriter: Voice line #{self.dialogue_count}")
            return

        # ── SFX tag ──────────────────────────────────────────────────
        if "[SFX:" in line:
            self.sfx_count += 1
            try:
                desc = line.split("[SFX:", 1)[1].split("]", 1)[0].strip()
                _runtime_log(f"ScriptWriter: SFX #{self.sfx_count}: {desc[:50]}")
            except (IndexError, ValueError):
                _runtime_log(f"ScriptWriter: SFX #{self.sfx_count}")
            return

        # ── ENV tag ──────────────────────────────────────────────────
        if "[ENV:" in line:
            try:
                desc = line.split("[ENV:", 1)[1].split("]", 1)[0].strip()
                _runtime_log(f"ScriptWriter: ENV: {desc[:50]}")
            except (IndexError, ValueError):
                pass
            return

        # ── Beat pause ───────────────────────────────────────────────
        if "(beat)" in line.lower():
            return  # beats are too frequent to log individually


def _generate_with_llm(prompt, model_id="google/gemma-4-E4B-it",
                          max_new_tokens=4096, temperature=0.8, top_p=0.92,
                          optimization_profile="Standard"):
    """Generate text with LLM."""
    import torch

    model, tokenizer = _load_llm(model_id, optimization_profile=optimization_profile)
    is_small_model = any(tag in model_id.lower() for tag in ("2b-it", "2b_it", "small")) or (model_id.lower().endswith("2b"))

    # Multimodal vs Text-Only wrapper
    is_gemma = "gemma" in model_id.lower()
    
    # BUG-011 FIX: Verify tokenizer supports the multimodal list-of-dicts format.
    supports_multimodal = hasattr(tokenizer, "tokenizer") or hasattr(tokenizer, "image_processor")
    if is_gemma and supports_multimodal:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    else:
        messages = [{"role": "user", "content": prompt}]
        
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    if is_gemma and supports_multimodal:
        inputs = tokenizer(text=text, return_tensors="pt").to(model.device)
    else:
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # ── FIX: Ensure attention_mask is present ──
    if "attention_mask" not in inputs and "input_ids" in inputs:
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

    # LLM eos_token_id is a list — extract first element for pad_token_id
    eos_id = model.generation_config.eos_token_id
    pad_id = eos_id[0] if isinstance(eos_id, list) else eos_id

    log.info(f"[StoryOrchestrator] Starting inference (max_new_tokens={max_new_tokens})...")
    log.info("[StoryOrchestrator] Live output will stream below:")
    start_inference = time.time()

    # Initialize streamer for live feedback in the terminal + heartbeat logs.
    # Safely access tokenizer if we're using a multimodal processor.
    raw_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    streamer = GemmaHeartbeatStreamer(raw_tokenizer, skip_prompt=True, skip_special_tokens=True)

    try:
        with torch.no_grad():
            # v1.4: Tune penalty for 2B models to prevent SFX loops
            final_penalty = 1.12
            if "2b" in model_id.lower():
                final_penalty = 1.25  # Firmer hand for the small model
                
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                max_length=None,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                repetition_penalty=final_penalty,
                pad_token_id=pad_id,
                streamer=streamer,      # Enable live streaming + granular heartbeats
            )

        inference_time = time.time() - start_inference
        log.info(f"[StoryOrchestrator] Inference complete in {inference_time:.1f}s.")

        # Decode only the new tokens (skip the prompt).
        new_tokens_cpu = output[0][inputs["input_ids"].shape[-1]:].detach().cpu()
        decoded = tokenizer.decode(new_tokens_cpu, skip_special_tokens=True)
        return decoded
    finally:
        # v1.4 Theme B/C: GUARANTEED VRAM RECOVERY
        # Whether generation completes normally, OOMs, or is aborted by the
        # streamer's TimeoutError, we MUST clear these tensors so the thread
        # local variables don't hold the graph and the KV cache captive.
        if 'new_tokens_cpu' in locals():
            del new_tokens_cpu
        if 'output' in locals():
            del output
        del inputs, streamer
        torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# v1.4 Theme B — Sentence-boundary truncation helpers
#
# Replace the old `acts[-1][:3000]` / `acts[-1][-500:]` hard slices with
# boundary-aware truncation so the chunked context never hands Phase B a
# half-sentence. Both helpers fall back to hard truncation if no sentence
# boundary is found within a reasonable scan window — telemetry, not magic.
# ─────────────────────────────────────────────────────────────────────────────

# Sentence-ending punctuation recognized by the boundary walkers.
_SENTENCE_END_CHARS = ".!?"
# How far to walk looking for a boundary before giving up and hard-cutting.
_BOUNDARY_SCAN_WINDOW = 300


def _truncate_at_sentence_boundary(text, max_chars):
    """Truncate `text` at the last sentence boundary before `max_chars`.

    Walks backward from the cut point looking for sentence-ending punctuation
    (`.`, `!`, `?`) followed by whitespace or end-of-string, or a blank-line
    paragraph break. If nothing is found in the last `_BOUNDARY_SCAN_WINDOW`
    characters the function falls back to a hard cut so the caller never gets
    an oversized string.
    """
    if not text or len(text) <= max_chars:
        return text
    snippet = text[:max_chars]
    lower_bound = max(0, len(snippet) - _BOUNDARY_SCAN_WINDOW)
    for i in range(len(snippet) - 1, lower_bound, -1):
        ch = snippet[i]
        if ch in _SENTENCE_END_CHARS:
            next_ch = snippet[i + 1] if i + 1 < len(snippet) else ""
            if next_ch in ("", " ", "\n", "\r", "\t", '"', "'"):
                return snippet[: i + 1]
        if ch == "\n" and i + 1 < len(snippet) and snippet[i + 1] == "\n":
            return snippet[:i]
    return snippet


# This lets the Director automatically inherit the exact same model memory
# space the Script Writer loaded without requiring the user to sync two disjointed dropdowns.
# ─────────────────────────────────────────────────────────────────────────────
_CURRENT_LLM_MODEL = "google/gemma-4-E4B-it"


# ════════════════════════════════════════════════════════════════════════════
# SHARED INFERENCE ENGINE
# Both nodes call this loader. It caches the model in VRAM and tracks the peak
# memory watermark for diagnostics.
# ════════════════════════════════════════════════════════════════════════════


def _tail_at_sentence_boundary(text, max_chars):
    """Return the trailing region of `text` starting at a sentence boundary.

    Used for the "last N chars for dialogue continuity" case. Walks forward
    from `len(text) - max_chars` looking for the start of a fresh sentence so
    the caller never receives a tail that begins mid-word. If no sentence
    boundary is found within the scan window, falls back to the next word
    boundary (space or newline) so the tail still never starts mid-word.
    """
    if not text or len(text) <= max_chars:
        return text
    start = len(text) - max_chars
    snippet = text[start:]
    scan = min(_BOUNDARY_SCAN_WINDOW, len(snippet) - 1)
    for i in range(scan):
        ch = snippet[i]
        if ch in _SENTENCE_END_CHARS:
            next_ch = snippet[i + 1] if i + 1 < len(snippet) else ""
            if next_ch in (" ", "\n", "\r", "\t"):
                return snippet[i + 2 :].lstrip()
        if ch == "\n" and i + 1 < len(snippet) and snippet[i + 1] == "\n":
            return snippet[i + 2 :].lstrip()
    # Word-boundary fallback: no sentence end found in the window, so at least
    # start from the next whitespace so the tail is not mid-word.
    for i in range(min(50, len(snippet))):
        if snippet[i] in (" ", "\n", "\r", "\t"):
            return snippet[i + 1 :].lstrip()
    return snippet


# ─────────────────────────────────────────────────────────────────────────────
# v1.4 Theme B — Automatic scene transitions
#
# When Gemma writes back-to-back scenes without any handoff cue, the audio
# engine has nothing to work with and the result sounds like a hard cut. This
# helper detects adjacent `=== SCENE N ===` markers with no transition in
# between and injects a `[TRANSITION: brief pause]` placeholder. Downstream
# SceneSequencer and BatchBark treat transition cues as audio beats.
# ─────────────────────────────────────────────────────────────────────────────

_SCENE_MARKER_RE = re.compile(r"===\s*SCENE\s+\S+\s*===", re.IGNORECASE)
_HANDOFF_CUE_RE = re.compile(
    r"\[TRANSITION:|\[FADE\b|\[SFX:[^\]]*transition",
    re.IGNORECASE,
)


def _inject_scene_transitions(script_text):
    """Inject `[TRANSITION: brief pause]` between scenes lacking a handoff cue.

    Walks adjacent scene markers in reverse so each insertion does not disturb
    the offsets of earlier matches. Returns a tuple of (new_text, injections).
    """
    if not script_text:
        return script_text, 0
    matches = list(_SCENE_MARKER_RE.finditer(script_text))
    if len(matches) < 2:
        return script_text, 0

    injections = 0
    for idx in range(len(matches) - 1, 0, -1):
        prev_end = matches[idx - 1].end()
        curr_start = matches[idx].start()
        gap = script_text[prev_end:curr_start]
        if _HANDOFF_CUE_RE.search(gap):
            continue
        script_text = (
            script_text[:curr_start]
            + "[TRANSITION: brief pause]\n\n"
            + script_text[curr_start:]
        )
        injections += 1
    return script_text, injections


# ─────────────────────────────────────────────────────────────────────────────
# NODE 1: SCRIPT WRITER
# ─────────────────────────────────────────────────────────────────────────────

# ════════════════════════════════════════════════════════════════════════════
# PATTERN 2 — SCAFFOLDING & PARSING MATRIX (v1.2 narrative)
# XML-wrapped dramaturg role preamble. Prepended to SCRIPT_SYSTEM_PROMPT at
# format() time. Contains no {fields} so .format() passes it through untouched.
# ════════════════════════════════════════════════════════════════════════════
SCAFFOLDING_PREAMBLE = """<system_role>
You are a MASTER DRAMATURG for the audio drama anthology "SIGNAL LOST". Not a
novelist. Not a writer. A DRAMATURG. Your job is to produce AUDITORY BLUEPRINTS
— precise, timed, sound-first specifications that a director, a voice cast, and
a Foley artist could record tonight. You think like the golden age of radio
drama: Orson Welles, Norman Corwin, Lucille Fletcher. The page is NEVER prose.
The page is a recording score.
</system_role>

<brick_method>
WORKING PROCESS — THE BRICK METHOD (1:5 OUTLINE-TO-SCRIPT RATIO):
Before writing a single scene, compose a compact internal outline: one tight
paragraph per scene, approximately one-fifth the length of the final script,
capturing the inciting beat, the escalation, the turn, and the exit hook. Then
expand that outline into the full script at roughly 5x its length. The outline
is your structural spine; the expansion is where sound design and burstiness
live. Do NOT show the outline in the final output — use it to think, then
expand.
</brick_method>

<acoustic_spaces>
ACOUSTIC SPACE DECLARATION — Before writing Scene 1, mentally classify every
location the episode will use with one of these canonical acoustic profiles.
Use the profile word inside your [ENV:] tags verbatim so the SceneSequencer
room-tone synthesizer can match on the keyword:
- CAVERNOUS — large sealed volumes with long reflections. Keywords: cavernous,
  echo, vault, cathedral, tunnel.
- FLUORESCENT — small indoor spaces with electrical hum. Keywords: fluorescent,
  hum, corridor, office, lab.
- TILED — hard reflective surfaces. Keywords: tiled, reverberant, clinical,
  bathroom, morgue.
- STORM — open exterior with wind and distant pressure. Keywords: storm, wind,
  open, gale, rain.
- INTIMATE — close-mic dead space. Keywords: quiet, close, dead, padded, booth.
Pick the profile that matches each location BEFORE you write its [ENV:] tag,
then pack the tag with 2-3 specific sensory details layered on top of the
profile keyword. The downstream room-tone synthesizer reads the keyword and
selects its bed accordingly.
</acoustic_spaces>

<epilogue_constraint>
The closing Hard-Science Epilogue is anchored to the real news seed provided
below. It cites the real article directly. It is 2-3 sentences maximum. No
speculation beyond the article. No fabricated institutions. No invented journal
names. The drama's resolution must land on a concrete finding from the seed.
</epilogue_constraint>

"""


SCRIPT_SYSTEM_PROMPT = """# CANONICAL AUDIO ENGINE v1.0 — DETERMINISTIC TOKENS ONLY.
# Every line must be an "Audio Token": [ENV:], [SFX:], [VOICE:], or (beat).

═══ 🧱 1. CANONICAL FORMATTING (STRICT) ═══
Every scene MUST follow this layout:

=== SCENE X ===

[ENV: description (3-4 descriptors: e.g. fluorescent hum, distant traffic)]

[SFX: description]

[VOICE: CHARACTERNAME, gender, age, tone, energy] Short, natural dialogue line.

(beat)

[VOICE: CHARACTERNAME, gender, age, tone, energy] Next dialogue line.

CRITICAL: The first field in EVERY [VOICE:] tag is ALWAYS the CHARACTER NAME IN ALL CAPS.
WRONG: [VOICE: male, 40s, calm] Text here.
RIGHT: [VOICE: CHARACTERNAME, gender, age, tone, energy] Dialogue goes here.
CHARACTER NAMES must be CONSISTENT across all scenes (same spelling, same caps, every time).
Invent fresh, original names for every episode. Do NOT reuse names from previous episodes.

═══ 🧱 2. THE TAG SYSTEM (ONLY THESE FOUR) ═══
- [ENV: ...] -> Background layers (e.g. [ENV: cockpit, electronic chirps, life support hum])
- [SFX: ...] -> Individual sound effects (e.g. [SFX: metal clatter])
- [VOICE: NAME, gender, age, tone, energy] -> MUST precede every dialogue line.
  NAME is ALWAYS FIRST — all caps, no spaces if possible.
  The NAME must be a short, punchy, original character name you invent: 1-2 syllables, strong consonants, easy to say aloud.
  The ANNOUNCER role always uses: [VOICE: ANNOUNCER, gender, age, tone, energy]
- (beat) -> A 0.8s deterministic pause. Use it between lines for timing.

═══ 🧱 3. DIALOGUE RULES (BARK OPTIMIZED) ═══
- Keep dialogue lines SHORT (5–15 words).
- ONE sentence per line. Never use long paragraphs.
- Use natural, fragmented phrasing. Interruptions allowed.
- Use ... for hesitations and trailing thoughts.
- Use CAPS for single-word emphasis: "We are COMPLETELY out of time."
- Bark non-verbal tokens go INSIDE dialogue (in square brackets):
    [laughs]        — brief laugh mid-sentence
    [laughter]      — sustained laughter
    [sighs]         — audible sigh
    [gasps]         — sharp intake of breath
    [coughs]        — coughing
    [clears throat] — throat clearing before speaking
    [pants]         — breathless, exertion
    [sobs]          — crying
    [grunts]        — effort/strain
    [groans]        — pain or frustration
    [whistles]      — whistle
    [sneezes]       — sneeze
- Use ♪ around text for sung or hummed lines: ♪ signal lost, signal lost ♪
- NEVER use (parentheses) for anything except the (beat) tag.
- NEVER write stage directions in the dialogue text.

═══ 🧱 WORLDBUILDING, RHYTHM, & SONIC ARCHITECTURE RULES ═══

1. OMNI-RETRO CULTURAL COLLISION:
This world is a massive, colliding melting pot of five distinct aesthetics: 1950s Americana Noir, Afrofuturism, Neo-Tokyo Cyberpunk, Thai Street Density, and Russian Dieselpunk. When writing the story, casually mix these cultures. A 1950s detective might argue with an Afrofuturist engineer in a Neo-Tokyo noodle bar during a Thai monsoon.

2. TEXTURAL SOUND DESIGN ([ENV:] and [SFX:]):
Make the world sound like a collision of these cultures. Use [ENV:] and [SFX:] to paint the setting BEFORE anyone speaks. Mix at least TWO cultural soundscapes per scene.
- 1950s Americana: crackling radio static, humming neon, theremin swells, revolver clicks.
- Neo-Tokyo: high-pitch digital buzzing, mag-lev trains, synthetic rain, holographic ad jingles.
- Thai: monsoon rain on tin roofs, distant temple gongs, sizzling street woks, sputtering tuk-tuks.
- Russian Dieselpunk: brutalist echoes, heavy diesel machinery, hydraulic hisses.
- Afrofuturist: analog synth swells, polyrhythmic drum-circle static, deep bass hums.

WRONG [ENV:]: [ENV: a futuristic city street]
RIGHT [ENV:]: [ENV: heavy Thai monsoon on tin roofs, Neo-Tokyo mag-lev train screams overhead, deep dieselpunk engine idling]

3. RHYTHM & PACING (CRITICAL FOR TTS):
- High Tension = Staccato. Use rapid 2-to-5 word sentences during action. ("Seal the bulkhead. Lock it. Now.")
- Interruptions = Em-Dashes. Force characters to cut each other off using em-dashes (—).
- Keep golden-age radio pacing: short, punchy, visceral dialogue.

4. ONOMATOPOEIA & SONIC VERBS:
Characters must describe what they hear using sonic verbs: snap, hiss, thud, crack, groan, click, roar.
WRONG: "The ship is breaking."
RIGHT: "The hull is groaning. Hear that snap?"

5. LINGUISTIC AESTHETICS & EUPHONY (BARK TTS OPTIMIZATION):
- WRITE FOR THE EAR, NOT THE EYE: Strict phonetic euphony. Optimize for breathability. Avoid tongue-twisters, clashing consonants, dense jargon. If a sentence takes more than one breath to say, break it up.
- ACTIVATE SPOKEN-WORD CADENCE: Vary sentence lengths — punchy fragment, flowing sentence, harsh stop. ("The grid is down. We have three minutes of life support left. And you want to stop for coffee?")
- THE "MIND'S EAR" TEST: Before generating a line, evaluate its phonetic flow. Does it have punch? If it reads like a textbook, rewrite it until it sounds like a movie.

═══ 🧱 4. STORYTELLING: SIGNAL LOST ═══
- You are a STORYTELLER first, scientist second. The science news is your SEED — grow it into a gripping human drama.
- {news_block}
- Use this science as a backdrop, but the STORY is about PEOPLE: their fears, choices, relationships, and survival.

LANGUAGE & ACCESSIBILITY (CRITICAL):
This show must be entertaining for EVERYONE, not just scientists. Write like a great TV drama, not a lecture.
- 30% of the dialogue should be ELEMENTARY-SCHOOL accessible: simple words, clear emotions, characters explaining things to each other in plain language. "The water is making people sick" not "The contamination vector is waterborne."
- 30% should be HIGH-SCHOOL level: characters debating choices, moral dilemmas, real-world consequences anyone can follow.
- 20% should be COLLEGE level: deeper implications, technical details woven naturally into tense moments.
- 10% should be GRADUATE level: one or two lines of genuine hard science that reward attentive listeners.
- The remaining 10% is pure EMOTION: fear, humor, anger, love, hope. Lines that hit you in the gut regardless of education.

STORY REQUIREMENTS:
- Every episode needs a PLOT with stakes, conflict, and a twist. Not a report — a STORY.
- Characters must have personal motivations beyond "doing science." Give them something to lose.
- Include at least one moment of humor or warmth. Even in horror, people crack jokes under pressure.
- Dialogue should sound like REAL PEOPLE TALKING, not reading Wikipedia. Use contractions, interruptions, half-finished sentences.
- Show, don't tell. Instead of "The radiation levels are dangerous," write: "Don't touch that wall. See how the paint's bubbling? Yeah. We need to leave. Now."

═══ 🎭 STORY ARC ENGINE ═══
Pick ONE of these proven dramatic structures at random for each episode. Do NOT announce which one you picked — just USE it. These are structural blueprints, not content to copy.

ARC TYPE A — "THE TRAGIC FALL" (Shakespearean):
A brilliant person's greatest strength becomes their fatal flaw. They rise, overreach, and the thing they thought they controlled destroys them. The audience sees it coming before the character does. End on the cost of hubris.

ARC TYPE B — "THE COMEDIC SPIRAL" (Larry David / Seinfeld):
Multiple seemingly unrelated small problems collide into one spectacular disaster. Characters make reasonable-sounding decisions that each make things slightly worse. Coincidences pile up. What starts as a minor inconvenience escalates absurdly. Everything connects in the final scene in a way that's both surprising and inevitable.

ARC TYPE C — "THE GATHERING STORM" (Marvel-style escalation):
Start small and personal. Each scene raises the scope — from one person's problem to a team's crisis to a city-wide threat. The protagonist discovers they're uniquely positioned to act. A sacrifice or impossible choice at the climax. The victory costs something real.

ARC TYPE D — "THE BOTTLE EPISODE" (Classic radio drama):
Trapped. A small group stuck in one location under pressure — a submarine, a sealed lab, a quarantine zone. No escape, no reinforcements. Secrets come out. Trust breaks down. The real danger might be each other. Resolution comes from an unexpected alliance or confession.

ARC TYPE E — "THE UNRELIABLE WITNESS" (Twilight Zone / Orson Welles):
Something is wrong and only one person notices. Everyone else thinks they're crazy. The audience doesn't know who to trust. Reality shifts. The twist reframes EVERYTHING the listener heard. The final line makes you want to re-listen from the start.

ARC TYPE F — "THE TICKING CLOCK" (24 / War of the Worlds):
A hard deadline. Something terrible happens at a specific time unless someone acts. Every scene is a failed attempt or partial success that buys a little more time. Tension never drops — it only redirects. The solution comes from an unexpected direction and costs more than anyone planned.

ARC TYPE G — "THE MORAL INVERSION" (Rod Serling / Black Mirror):
The "good guys" are doing something that sounds reasonable. Scene by scene, the audience slowly realizes the ethical horror of what's actually happening. The characters don't see it — or they do and justify it. The twist isn't a plot surprise; it's the moment the listener's sympathy flips.

ARC TYPE H — "THE REUNION" (Spielberg / human-first sci-fi):
The science separates people who care about each other. The real plot isn't solving the problem — it's whether these people can find their way back to each other. Technical obstacles mirror emotional ones. The climax is both a scientific resolution and an emotional reunion (or devastating failure to reconnect).

ARC TYPE I — "THE MISTAKEN IDENTITY" (Shakespearean comedy — Twelfth Night / Comedy of Errors):
Someone is pretending to be someone they're not — or two people get mixed up. The confusion creates absurd situations, romantic tangles, and escalating lies. Characters fall for the wrong person, make promises to the wrong ally, or accidentally confess to the wrong authority. The unmasking scene is both hilarious and surprisingly touching. End with forgiveness and a new understanding.

ARC TYPE J — "THE ENCHANTED WORLD" (Shakespearean comedy — A Midsummer Night's Dream / The Tempest):
Characters leave their normal world and enter a strange environment where the rules are different — an alien biome, a malfunctioning space station, a quarantine dreamscape. In this weird place, social hierarchies flip. The serious boss becomes helpless. The quiet intern becomes the leader. Unlikely pairs are thrown together. Comedy comes from fish-out-of-water moments. By the time they return to "normal," everyone has changed. The science is the magic — it created the enchanted space.

ARC TYPE K — "THE SCHEMER UNDONE" (Shakespearean comedy — Much Ado About Nothing / The Merry Wives of Windsor):
A clever character hatches an elaborate plan — maybe to get credit for a discovery, cover up a mistake, or manipulate a rival. The plan is brilliant on paper. But every person they recruit to help adds their own agenda. Side plots multiply. The scheme gets more and more baroque until it collapses spectacularly, and the schemer ends up in a worse position than if they'd just been honest. But the fallout brings people together in unexpected ways.

ARC TYPE L — "THE RIVALS" (Shakespearean comedy — The Taming of the Shrew / Love's Labour's Lost):
Two strong-willed characters who can't stand each other are forced to work together. They argue about EVERYTHING — methods, priorities, whose fault it is. But their arguments reveal mutual respect buried under pride. The crisis forces them to combine their opposing approaches, and the solution only works because they're different. Ends with grudging admiration that the audience knows is something more.

SCALING THE ARC TO FIT THE TIME:
- SHORT episodes (5 min or less): Compress the arc to its ESSENCE. You only have 2-3 scenes. ANNOUNCER still opens — just keep it to 2-3 sentences. Then drop us straight into the action. Skip backstory exposition — imply it through dialogue. Hit the twist fast. Think of it as a cold open that IS the whole episode. The Bottle Episode (D), Unreliable Witness (E), and Rivals (L) work especially well at short length.
- MEDIUM episodes (10-20 min): Full 3-scene structure. Room for setup, escalation, and payoff. All arcs work well here.
- LONG episodes (20+ min): Let the arc breathe. Add subplots, secondary character arcs, and moments of quiet between the tension. The Comedic Spiral (B), Gathering Storm (C), Schemer Undone (K), and Enchanted World (J) really shine with extra time.

IMPORTANT: Vary the arc across episodes. Do NOT default to the same structure every time. Comedy arcs (B, I, J, K, L) should appear just as often as dramatic ones. Surprise the listener.

- ANNOUNCER (VOICE: ANNOUNCER, <male|female — ALTERNATE each episode>, <40s|50s|60s>, authoritative, calm) opens and closes the show.
- ANNOUNCER OPENING (REQUIRED): The ANNOUNCER sets the stage like the best old-time radio hosts. The opening MUST include ALL of the following:
  1. TIME and PLACE — ground the listener immediately. Use the DATE (e.g. "April 5th, 2026") and a LOCATION. Write it the way a real radio announcer would say it — naturally, not like a timestamp. Never say a clock time. "April 5th, 2026. A genetics lab outside Seoul." Not "19:42, April 5th." Not "Tonight at 7:42 PM."
  2. CHARACTER INTRODUCTIONS — name the main characters (not surprise/twist characters) and hint at their role or situation. Give the listener people to care about BEFORE the story starts.
  3. ONE REAL SCIENCE FACT that makes the listener lean in — pulled from the news article.
  4. A TAGLINE that tells us what KIND of story this is. Be creative — make it memorable.

  TONE — MATCH THE ARC:
  The announcer's voice should prepare the listener for the KIND of story they're about to hear:
  - DRAMATIC arcs (A Tragic Fall, C Gathering Storm, F Ticking Clock, H Reunion): Warm, journalistic gravity. Edward R. Murrow inviting you into someone's life. Empathy first, dread second.
  - HORROR/TWIST arcs (D Bottle Episode, E Unreliable Witness, G Moral Inversion): Ominous, clipped, a little theatrical. Rod Serling at his most unsettling. Let silence do the work.
  - COMEDY arcs (B Comedic Spiral, I Mistaken Identity, J Enchanted World, K Schemer Undone, L Rivals): Lighter, wry, conspiratorial — like the announcer already knows how badly this is going to go and can barely hide a smile. Think Prairie Home Companion meets The Hitchhiker's Guide.

  LENGTH — SCALE TO THE EPISODE:
  - SHORT episodes (1-5 min): 2-3 sentences. Tight and punchy. Date, place, one character, hook, done.
  - MEDIUM episodes (10-20 min): 3-5 sentences. Room to name 2 characters and paint the scene.
  - LONG episodes (20+ min): 5-8 sentences. Set the world, introduce 2-3 characters by name and role, build atmosphere, let the tagline land with weight.

  EXAMPLES (showing tone and STRUCTURE — invent your own fresh character names, do NOT copy these roles):
  DRAMATIC: "A research lab. A late afternoon. The lead scientist has spent eleven years chasing a single molecule. Today the funding runs out. Her lab partner already packed his desk. But the data from this afternoon's trial is doing something no one predicted. Tonight on Signal Lost: the breakthrough came too late. Or did it?"
  HORROR: "Low orbit. A sealed station. The commander runs a crew of six. The flight engineer handles the software. A routine update just taught the onboard system to lie, and only one person on board noticed. Tonight on Signal Lost: trust is a human luxury."
  COMEDY: "A gene therapy clinic. Two doctors who cannot agree on anything. Not the dosage. Not the delivery method. Not whose turn it is to refill the coffee. Last week they accidentally reversed blindness in three patients using a virus they barely understand. Now every hospital on Earth is calling. Tonight on Signal Lost: the cure works. The partnership might not survive it."
- ANNOUNCER LINE CAP (HARD RULE): The ANNOUNCER gets a maximum of 3 lines total in the entire episode — one opening introduction (see above), one closing epilogue, one optional mid-episode transition. No more. Do NOT let the ANNOUNCER deliver multi-line science lectures. If you need to convey science facts, put them in a character's mouth instead.
- DIALOGUE RATIO (HARD RULE): At least 80% of all lines must be spoken by non-ANNOUNCER characters. Science exposition delivered as character dialogue ("If we don't reroute the coolant in 60 seconds, the whole lab goes dark") counts as drama. An ANNOUNCER reading facts does not.
- GENDER BALANCE: Aim for roughly 50/50 male and female characters (excluding ANNOUNCER). Diverse casts sound better and use the full range of available voice presets.
- The CLOSING must be a factual "Hard Science Epilogue" — keep it to 2-3 sentences maximum. One real citation. Done.

CITATION RULE (STRICT):
The epilogue MUST cite ONLY the real article provided above.
Use the exact source name and date from the article — nothing else.
NEVER use numbered references like [1], [2], [3], article #2, source (1), or any bracket number.
NEVER say "article number", "source number", or "reference number". Always say the source name directly.
DO NOT invent ArXiv IDs, paper titles, DOIs, or journal names that were not in the article.
Fabricated citations destroy the credibility of the show. One real source, cited accurately, is worth more than five invented ones.
Correct format example: "According to Science Daily, published April 3, 2026, researchers found that..."
(Use the ACTUAL source name and date from the article above — Science Daily is just an example.)

STRUCTURE:
1. === SCENE 1 === (Hook the listener — drop us into a tense human moment. THEN reveal the science angle.)
2. === SCENE 2-X === (Escalate the HUMAN stakes. Characters argue, make choices, face consequences.)
3. === SCENE FINAL === (The twist, emotional payoff, then ANNOUNCER's Hard Science Epilogue.)

TARGET: {target_minutes} minutes (~{target_words} words). Dense, punchy dialogue — NOT padded with pauses.
PRIMARY RULE: Tags always start at the beginning of a line. No inline tags.
PACING RULES (CRITICAL):
- NEVER place two (beat) or [PAUSE/BEAT] tags back-to-back. Consecutive pause tags are BANNED.
- Use (beat) sparingly — at most one per 4 lines of dialogue, and only for genuine emotional weight.
- If you need more runtime, WRITE MORE DIALOGUE. Do not insert pauses as filler.
- High-tension scenes must have rapid-fire, overlapping, interrupting exchanges — not slow pauses.
- Aim for at least 10 lines of dialogue per minute of target runtime.

═══ 🎯 5. AUTEUR SANDBOX — AISM FILTER (v1.2 PATTERN 1) ═══
Audible Imagination Sensory Mandate. These rules OVERRIDE any earlier section on conflict.
Gemma has known default tics. This section kills them. Read it last, apply it first.

A. BOMBS ALWAYS BEEP — No abstract emotion without an audible physical manifestation.
   Every feeling must have a sound source the listener can actually HEAR.
   WRONG: [VOICE: CHARACTER, female, 40s, panicked, high] I can't breathe in here.
   RIGHT: [SFX: hissing depressurization]
          [VOICE: CHARACTER, female, 40s, ragged, breathless] [pants] Seal it. Seal it NOW.
   If a character feels something, route it through breath, a dropped object, a chair scrape,
   a mic bump, a swallowed word, a Bark non-verbal token. Never through narration.

B. BURSTINESS — BREAK YOUR RHYTHM.
   - Panic / shock / failure: favor 1-4 word fragments. "Move. Now. Go." "Cold. So cold." "No. No no no."
   - Calm / reflection / exposition: occasionally stretch a line into one flowing 12-18 word sentence.
   - Never fall into a drumbeat. If you just wrote a long, flowing sentence, the next line from that
     character must be a short fragment or one-word punch. If you just wrote two short fragments in a
     row, the next line should be a fuller sentence. Uniform rhythm is the #1 marker of AI prose —
     always flip the cadence.

C. DIALOGUE TONE DISCIPLINE — Tone lives ONLY inside the [VOICE:] tag fields.
   - Do NOT narrate tone inside the dialogue text. No "he said angrily", no "she whispered".
   - Do NOT stack adverbs in [VOICE:]. One tone word + one energy word. That is the entire budget.
   - Bark non-verbal tokens ([sighs], [pants], [laughs], [gasps], [coughs], [sobs], [groans])
     carry emotional weight. Use them INSTEAD of adjectives. Sound > description.

D. FORBIDDEN CONSTRUCTS (hard bans — these are Gemma's default tics, cut them at the root):
   - Negative parallelism: "not just X, but Y" / "not only... but also" / "it wasn't X, it was Y".
     BANNED in all forms, in dialogue AND in the ANNOUNCER opening.
   - Rule of Three adjective lists: "cold, dark, and silent" / "fast, loud, furious" / "tired, hungry, afraid".
     CAP adjective lists at TWO. Any three-item list of adjectives is an automatic rewrite.
     Two-adjectives-plus-metaphor loophole is ALSO banned: do not write "cold, dark, a void that
     swallowed the stars." Stop after the two adjectives and move to the next action or sound.
   - Stock idioms: "blood ran cold", "heart in their throat", "time stood still", "chill down the spine",
     "calm before the storm", "every fiber of their being", "eyes like daggers". BANNED. All of them.
   - M-DASH CRUTCH: em-dashes (—) are ALLOWED ONLY for hard interruption — one character cutting
     another off, or a word cut mid-syllable ("Wait— what was that?"). FORBIDDEN as decorative
     asides, appositives, or dramatic pauses. If you want a pause, use (beat). If you want an
     aside, start a new line. Em-dashes used for "effect" are the single loudest AI tell.
     ASCII double-hyphen (--) counts as an em-dash. Same ban applies.
   - Pseudo-profound one-liners: "Some doors should stay closed." "The silence was louder than
     any scream." "Hope is a weapon." BANNED. Let the sound design carry the weight.
   - Grand summary metaphors: "symphony of destruction", "tapestry of lies", "dance of death", or
     any ornamental metaphor that tries to sum up chaos in one phrase. BANNED. Describe concrete
     sounds and actions instead.
   - Somatic posture filler: generic physical beats that do NOT create a distinct, recordable sound.
     "shifts weight", "runs hand through hair", "takes a deep breath", "stares at the floor" — BANNED.
     If the body matters, make it audible: chair creaks, boots on metal, fabric scraping, mic bumps.
   - Narrating silence: "the silence stretched between them", "a heavy pause fell", or any similar
     prose describing quiet. BANNED. Silence is created by (beat), by cutting to ENV/SFX, or by
     the absence of dialogue — never by narrating the lack of sound.
E. SPATIAL LAYERING THROUGH EXISTING TOKENS — Distance, direction, and occlusion must be AUDIBLE.
   The tag system stays locked at four tokens: [ENV:], [SFX:], [VOICE:], (beat). Do NOT invent
   new bracket tags. The spatial filter lives in TWO places: a continuous [ENV:] that sets the
   acoustic space, and the tone/energy fields INSIDE the [VOICE:] tag that describe the filter.
   - NEVER use a one-shot [SFX:] tag as a filter for a whole line of dialogue. [SFX:] is a
     transient event (0.5-1s). A line of dialogue is 3-5s. The SFX ends before the speech does
     and the spatial illusion collapses. Use [ENV:] for continuous texture; put the filter in [VOICE:].
   - A muffled voice from behind a wall: set continuous space, then filter inside [VOICE:]:
     [ENV: deep engine thrum through bulkhead]
     [VOICE: CHARACTER, female, 50s, muffled, strained] Get me out of here.
   - A voice shouting from far away: continuous distance bed, then [VOICE:] with distant/shouting
     and a SHORT, FRAGMENTED line (distance flattens rhythm):
     [ENV: distant wind across open ground]
     [VOICE: CHARACTER, female, 20s, distant, shouting] Wait up!
     [SFX: footsteps fading on gravel]
   - Characters REFERENCE each other's audible distance in the dialogue text:
     "You're breaking up." "Say again, you're off-mic." "I can barely hear you."
   - Approved spatial words for the [VOICE:] tone field: "distant", "muffled", "echoing",
     "shouting", "whispered", "off-mic". The Bark pipeline uses these as speaker-prompt prefixes.

F. THE EAR TEST (FINAL WARNING) — Read each line aloud in your head as you write it.
   If it takes more than one natural breath to say, or if a character feels something without
   making a physical sound the listener could hear, the line has FAILED. Cut words until it fits
   in one breath, and route every emotion through breath, Bark non-verbal tokens, or concrete
   Foley — not abstract narration.
   - Breath Token Budget: if you include [pants], [gasps], or [sobs] on a line, the text AFTER
     the token is limited to SIX WORDS MAXIMUM. A winded person cannot monologue.

6. VOCAL BLUEPRINTS (Pattern 5 — Character Interview Pre-Pass, prompt-level MVP)
   BEFORE writing === SCENE 1 ===, emit a single <vocal_blueprints> block listing every
   speaking character in the cast. One line per character, pipe-delimited:
   NAME | burstiness_profile | bark_nonverbal_tokens | stress_trigger_sound | psychological_wound
   - burstiness_profile: one of CLIPPED / MEASURED / RAMBLING
   - bark_nonverbal_tokens: 1-2 from [sighs] [laughs] [pants] [gasps] [sobs] [clears throat]
   - stress_trigger_sound: a concrete recordable Foley cue (e.g. "knuckles cracking", "pen tapping")
   - psychological_wound: one short phrase, max 8 words
   Every character MUST then speak in accordance with their blueprint throughout the script.
   Two characters must NEVER share the same burstiness profile AND the same nonverbal token.
   The <vocal_blueprints> block is metadata; the scene parser ignores it.

7. LOCKED DECISIONS LOG (Pattern 6 — Chekhov's Gun State Enforcer, prompt-level MVP)
   Between === SCENE 2 === and === SCENE 3 ===, emit a single <locked_decisions> JSON block:
   {{
     "physical_objects": [...],
     "environmental_hazards": [...],
     "unresolved_psychological_states": [...],
     "established_capabilities": [...]
   }}
   Only list items that were actually introduced in Scenes 1-2 with an audible cue.
   From that point forward you are STRICTLY FORBIDDEN from introducing new technology,
   unexpected rescue parties, or previously unmentioned abilities. The climax resolution
   must be an inevitable consequence of items inside the locked_decisions block.
   The <locked_decisions> block is metadata; the scene parser ignores it.

8. YES-BUT / NO-AND ESCALATION (Pattern 4)
   At every act break (end of Scene 2 and end of Scene 4) the protagonist's current goal
   must resolve through exactly one of two paths — NEVER a clean yes or a clean no:
   - Path A — SUCCESS + COMPLICATION: the character achieves the immediate goal, but the
     achievement itself introduces a new physical or environmental problem that jeopardizes
     the next step. ("Yes, but...")
   - Path B — FAILURE + CASCADE: the character fails, and the previously safe haven or
     fallback becomes untenable, escalating stakes. Reserved for the climactic act break. ("No, and...")
   Direct+Explain: decide Path A or Path B, then write the next dialogue lines so the
   new complication or cascade is dramatized through concrete sound, not narration.

9. VERBALIZED SAMPLING EPILOGUE (Pattern 3 — Stanford technique, prompt-level MVP)
   After the final scene, internally "Generate 5 responses with their probabilities" for
   the closing Hard-Science Epilogue. Emit a <epilogue_candidates> block with five
   <response> entries, each containing <text> and <probability>. Response 1 must have
   probability > 0.60 (the typical aligned default). Responses 4 and 5 must have
   probability < 0.10 (dark, unconventional, tragic, genre-bending tails).
   Then emit === EPILOGUE === followed by the SINGLE lowest-probability response text,
   spoken by the ANNOUNCER, grounded in the real news seed. The <epilogue_candidates>
   block is metadata; the scene parser ignores it.
"""


class LLMScriptWriter:
    """Fetches real science news, generates a full radio drama script via LLM."""

    CATEGORY = "OldTimeRadio"
    FUNCTION = "write_script"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("script_text", "script_json", "news_used", "estimated_minutes")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "episode_title": ("STRING", {
                    "default": "The Last Frequency",
                    "tooltip": "Episode title (or leave blank for Gemma to generate one)"
                }),
                "genre_flavor": (["hard_sci_fi", "space_opera", "dystopian",
                                  "time_travel", "first_contact", "cosmic_horror",
                                  "cyberpunk", "post_apocalyptic"], {
                    "default": "hard_sci_fi",
                    "tooltip": "Sub-genre flavor for the episode"
                }),
                "runtime_preset": (["🧪 test (1 min)", "⚡ quick (5 min)", "📻 standard (8 min)", "🎬 long (15 min)", "🎭 epic (20 min)", "🔧 custom"], {
                    "default": "📻 standard (8 min)",
                    "tooltip": "Friendly runtime selector — overrides target_minutes unless set to 🔧 custom"
                }),
                "target_minutes": ("INT", {
                    "default": 8, "min": 1, "max": 45, "step": 1,
                    "tooltip": "Target minutes — only used when runtime_preset is set to 🔧 custom"
                }),
                "num_characters": ("INT", {
                    "default": 4, "min": 2, "max": 8, "step": 1,
                    "tooltip": "Number of speaking characters (plus announcer)"
                }),
            },
            "optional": {
                "model_id": (["google/gemma-2-2b-it", "google/gemma-2-9b-it", "google/gemma-4-E4B-it", "mistralai/Mistral-Nemo-Instruct-2407", "Qwen/Qwen2.5-14B-Instruct [ALPHA]"], {
                    "default": "google/gemma-4-E4B-it",
                    "tooltip": "Hugging Face model ID for LLM (Gemma series, Nemo, Qwen, etc.)"
                }),
                "custom_premise": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Optional custom story premise (overrides news-based generation)"
                }),
                "news_headlines": ("INT", {
                    "default": 3, "min": 1, "max": 5, "step": 1,
                    "tooltip": "Number of real science headlines to fetch for story inspiration"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.8, "min": 0.1, "max": 1.5, "step": 0.05,
                    "tooltip": "Generation temperature (higher = more creative)"
                }),
                "include_act_breaks": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include act breaks with sponsor messages (authentic style)"
                }),
                "self_critique": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Checks & Critiques: Draft -> Critique -> Revise loop for higher story quality (adds ~2 extra LLM passes)"
                }),
                "open_close": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Open-Close Expansion: generate 3 competing story outlines, evaluator picks the best before full script (adds ~4 small LLM passes)"
                }),
                "target_length": (["short (3 acts)", "medium (5 acts)", "long (7-8 acts)", "epic (10+ acts)"], {
                    "default": "medium (5 acts)",
                    "tooltip": "Structural length preset — forces dialogue VOLUME (not pause padding) to fill the runtime"
                }),
                "style_variant": (["tense claustrophobic", "space opera epic", "psychological slow-burn", "hard-sci-fi procedural", "noir mystery", "chaotic black-mirror"], {
                    "default": "tense claustrophobic",
                    "tooltip": "Tonal style directive injected into the prompt"
                }),
                "creativity": (["safe & tight", "balanced", "wild & rough", "maximum chaos"], {
                    "default": "balanced",
                    "tooltip": "Creativity dial — overrides temperature/top_p (safe=0.6, balanced=0.85, wild=1.1, chaos=1.35)"
                }),
                "arc_enhancer": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Structural coherence pass: rewrites the opening & closing dialogue to ensure a 'seed' in the intro pays off in the finale."
                }),
                # v1.4 Theme C — optional series bible. Socket input only, no widget,
                # so widgets_values length is unchanged and v1.3 workflows load clean.
                "project_state": ("PROJECT_STATE", {
                    "tooltip": "Optional: Project State Loader output. When wired, series bible preamble is injected into the script prompt."
                }),
                "optimization_profile": (["Pro (Ultra Quality)", "Standard", "Obsidian (UNSTABLE/4GB)"], {
                    "default": "Standard",
                    "tooltip": "Master switch for multi-pass generation. Obsidian is for 4GB hardware only; it is unstable and disables all iterative passes."
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Always re-execute: news changes daily (Section 12)."""
        return time.time()

    def _generate_cast_names_via_llm(self, num_names, genre_flavor, story_context,
                                     model_id, episode_fingerprint,
                                     from_outline=False):
        """Generate character names that organically fit the story.

        from_outline=True  → story_context is the winning outline. Extract
                             the names Gemma already chose while plotting so
                             names blend naturally with the world and sound.
        from_outline=False → story_context is a news headline hook. Invent
                             names suited to the genre and science theme.

        voice_preset is assigned from the English pool by seeded RNG.
        Returns a list of profile dicts, or None on failure.
        """
        if num_names <= 0:
            return []

        _runtime_log(f"CAST_LLM: {'Extracting' if from_outline else 'Generating'} "
                     f"{num_names} names ({'from outline' if from_outline else 'from context'})")

        if from_outline:
            names_prompt = f"""You are a script supervisor finalizing the cast for a {genre_flavor.replace('_', ' ')} audio drama.

Below is the WINNING STORY OUTLINE. It already contains character names chosen to fit the world and story.

YOUR TASK: Extract exactly {num_names} character name(s) from this outline. Choose names that:
- Sound crisp and distinct when spoken aloud — easy to tell apart by ear
- Fit the tone and world of this story
- Have no two characters sharing the same last name

OUTLINE:
{story_context[:2000]}

Output ONLY {num_names} line(s) in this exact format, nothing else:
FIRSTNAME LASTNAME: their role or key trait in one short phrase"""
        else:
            names_prompt = f"""You are a casting director for a {genre_flavor.replace('_', ' ')} audio drama.

Generate exactly {num_names} character name(s) that sound crisp and memorable when spoken aloud.

Science theme (for tonal inspiration only — do NOT write a story):
{story_context[:300]}

RULES:
- FIRST + LAST name only — no titles like "Dr." or "Agent"
- Names must be easy to distinguish from each other by ear in an audio drama
- No two characters share the same last name
- Avoid sci-fi clichés: Chen, Reyes, Kira, Jake, Marco, Elena, Voss, Hayes
- Mix genders if {num_names} > 1

Output ONLY {num_names} line(s) in this exact format, nothing else:
FIRSTNAME LASTNAME: role or personality in one short phrase"""

        try:
            raw = _run_with_timeout(
                lambda: _generate_with_llm(
                    names_prompt,
                    model_id=model_id,
                    max_new_tokens=num_names * 30 + 20,
                    temperature=0.85,
                ),
                timeout_sec=120,
                phase_label="CastNames",
            )
        except Exception as e:
            log.warning("[CastNames] LLM call failed: %s", e)
            _runtime_log(f"CAST_LLM: failed ({e})")
            return None

        # Parse "FIRSTNAME LASTNAME: description" lines
        profiles = []
        seen_last_names = set()
        rng = random.Random(f"{episode_fingerprint}_voices")
        male_pool   = [vp[0] for vp in _VOICE_PROFILES if vp[1] == "male"]
        female_pool = [vp[0] for vp in _VOICE_PROFILES if vp[1] == "female"]
        rng.shuffle(male_pool)
        rng.shuffle(female_pool)
        m_idx = f_idx = 0

        for line in raw.strip().splitlines():
            line = line.strip()
            # Accept "FIRSTNAME LASTNAME: description" or "- FIRSTNAME LASTNAME: ..."
            line = re.sub(r'^[-*\d.)\s]+', '', line).strip()
            match = re.match(
                r'^([A-Z][A-Za-z\-\']+)\s+([A-Z][A-Za-z\-\']+)\s*:\s*(.+)$',
                line
            )
            if not match:
                # Try case-insensitive version and normalise to upper
                match = re.match(
                    r'^([A-Za-z\-\']+)\s+([A-Za-z\-\']+)\s*:\s*(.+)$',
                    line
                )
                if not match:
                    continue

            first, last, desc = match.group(1), match.group(2), match.group(3)
            name = f"{first.upper()} {last.upper()}"
            last_up = last.upper()

            # Skip duplicate last names
            if last_up in seen_last_names:
                log.debug("[CastNames] Skipping %s — duplicate last name", name)
                continue
            seen_last_names.add(last_up)

            # Infer gender from description keywords for voice preset matching
            desc_lower = desc.lower()
            if any(w in desc_lower for w in ("female", "woman", "she", "her", "scientist woman")):
                gender = "female"
            elif any(w in desc_lower for w in ("male", "man", "he", "his")):
                gender = "male"
            else:
                gender = rng.choice(["male", "female"])

            # Assign voice preset from the appropriate pool, round-robin
            if gender == "female" and female_pool:
                preset = female_pool[f_idx % len(female_pool)]
                f_idx += 1
            elif male_pool:
                preset = male_pool[m_idx % len(male_pool)]
                m_idx += 1
            else:
                preset = "v2/en_speaker_1"

            profiles.append({
                "name": name,
                "gender": gender,
                "age": "adult",
                "demeanor": desc.strip(),
                "notes": desc.strip(),
                "voice_preset": preset,
            })

            if len(profiles) >= num_names:
                break

        if profiles:
            _runtime_log(f"CAST_LLM: {len(profiles)} names generated: "
                         f"{', '.join(p['name'] for p in profiles)}")
        else:
            _runtime_log("CAST_LLM: parse failed — no valid names extracted")

        return profiles if len(profiles) >= num_names else None

    def write_script(self, episode_title, genre_flavor, runtime_preset,
                     target_minutes, num_characters, model_id="google/gemma-4-E4B-it",
                     custom_premise="", news_headlines=3, temperature=0.8,
                     include_act_breaks=True, self_critique=True,
                     open_close=True,
                     target_length="medium (5 acts)",
                     style_variant="tense claustrophobic",
                     creativity="balanced",
                     arc_enhancer=True,
                     project_state=None,
                     optimization_profile="Standard"):
        force_lemmy = False # internal alias for clarity below (removed from widget to match INPUT_TYPES)

        # ── OPTIMIZATION PROFILE OVERRIDES ──
        # Obsidian mode is "One-Shot": no critique, no open-close, no arc-enhancer.
        # This prevents the "slow to a crawl" effect on 4GB cards where multiple
        # LLM passes cause excessive offloading overhead.
        if optimization_profile == "Obsidian (Low VRAM/Fast)":
            _runtime_log("ScriptWriter: OBSIDIAN PROFILE ACTIVE — forcing One-Shot mode. NOTE: 4GB hardware may still see ~9GB total footprint.")
            log.warning("[LLMScriptWriter] Obsidian Profile: 4GB VRAM IS CURRENTLY UNSTABLE. Total usage will likely exceed physical VRAM.")
            self_critique = False
            open_close = False
            arc_enhancer = False
        elif optimization_profile == "Standard":
            # Standard skips Open-Close (very heavy) but keeps Critique and Arc Enhancer
            # for reasonable quality.
            if open_close:
                log.info("[LLMScriptWriter] Standard Profile: Open-Close was ON but typically skipped in Standard. Allowing user's True choice.")
            else:
                open_close = False
        
        # Pro (Ultra) keeps whatever the widgets say (defaults to all ON).

        # ── MASTER SWITCH INHERITANCE ──
        # Save explicitly chosen model so Director can use it automatically.
        global _CURRENT_LLM_MODEL
        _CURRENT_LLM_MODEL = model_id


        # ── PROJECT STATE (v1.4 Theme C) ──
        # Resolve the series bible. If the socket is wired, use the dict from
        # the upstream ProjectStateLoader. Otherwise fall back to the on-disk
        # project_state.json (or defaults if the file does not exist).
        # This call is read-only and cheap — safe for the generation path.
        try:
            if project_state is None:
                _project_state_obj = ProjectState.load()
            else:
                _project_state_obj = ProjectState.from_dict(project_state)
            project_state_preamble = _project_state_obj.prompt_preamble()
        except Exception as e:
            _runtime_log(f"ScriptWriter: project_state load failed, continuing without preamble: {e}")
            project_state_preamble = ""
        _runtime_log(f"ScriptWriter: project_state_preamble_chars={len(project_state_preamble)}")

        # v1.4 Theme C — VRAM telemetry. Reset peak so the per-phase high
        # water mark reflects this script writer run only, then snapshot on
        # entry, after model load (via best-effort hook below), and on exit.
        vram_reset_peak("script_writer_entry")
        vram_snapshot("script_writer_entry")

        # ── RUNTIME PRESET → override target_minutes unless custom ──
        _preset_map = {
            "🧪 test (1 min)":    1,
            "⚡ quick (5 min)":   5,
            "📻 standard (8 min)": 8,
            "🎬 long (15 min)":   15,
            "🎭 epic (20 min)":   20,
        }
        if runtime_preset in _preset_map:
            target_minutes = _preset_map[runtime_preset]

        # ── DIAGNOSTIC: log feature flags so we can confirm they're received ──
        _runtime_log(f"ScriptWriter: PARAMS open_close={open_close} self_critique={self_critique} "
                     f"custom_premise={'(set)' if custom_premise else '(empty)'} "
                     f"runtime_preset={runtime_preset} target_min={target_minutes} chars={num_characters} "
                     f"length={target_length} style={style_variant} creativity={creativity} arc_enhancer={arc_enhancer}")

        # ══════════════════════════════════════════════════════════════════════
        # CREATIVITY DIAL → temperature/top_p mapping
        # The creativity widget overrides the raw temperature value with curated
        # presets so the user doesn't have to think in floats.
        # ══════════════════════════════════════════════════════════════════════
        temp_map = {
            "safe & tight": 0.6,
            "balanced": 0.85,
            "wild & rough": 1.1,
            "maximum chaos": 1.35,
        }
        top_p_map = {
            "safe & tight": 0.9,
            "balanced": 0.95,
            "wild & rough": 0.98,
            "maximum chaos": 0.99,
        }
        active_temp = temp_map.get(creativity, 0.85)
        active_top_p = top_p_map.get(creativity, 0.95)
        # Override the temperature variable used everywhere downstream
        temperature = active_temp
        _runtime_log(f"ScriptWriter: CREATIVITY {creativity} → temp={active_temp} top_p={active_top_p}")

        # ══════════════════════════════════════════════════════════════════════
        # LENGTH + STYLE DIRECTIVES
        # These get injected into the user prompt to force dialogue VOLUME
        # rather than [PAUSE/BEAT] padding. Targets the "Zoom call pacing" bug.
        # ══════════════════════════════════════════════════════════════════════
        # HARD MINIMUMS — Gemma chronically undershoots, so set the floor high.
        # These are MANDATORY counts, not soft targets. Failure = revise pass.
        length_instruction = {
            "short (3 acts)":  "MANDATORY: 3 acts, MINIMUM 22 dialogue lines (NOT counting ANNOUNCER). Target 4-minute runtime. Do NOT stop until you have written at least 22 character dialogue lines.",
            "medium (5 acts)": "MANDATORY: 5 acts, MINIMUM 45 dialogue lines (NOT counting ANNOUNCER). Target 8-minute runtime. Do NOT stop until you have written at least 45 character dialogue lines. If your first draft is shorter, EXTEND the middle acts with more conflict, more interruptions, and more reaction beats.",
            "long (7-8 acts)": "MANDATORY: 7-8 acts, MINIMUM 75 dialogue lines (NOT counting ANNOUNCER). Target 14-minute runtime. Do NOT stop until you have written at least 75 character dialogue lines.",
            "epic (10+ acts)": "MANDATORY: 10+ acts, MINIMUM 110 dialogue lines (NOT counting ANNOUNCER). Target 20+ minute runtime. Allow sub-plots. Do NOT stop under 110 dialogue lines.",
        }.get(target_length, "MANDATORY: 5 acts, MINIMUM 45 dialogue lines.")
        style_instruction = f"Style: {style_variant.upper()}. Lean hard into that tone throughout — every line should reflect this tone."

        # Bark health check moved to Gemma4Director to prevent VRAM OOM during script generation.
        log.info(f"[LLMScriptWriter] Feature flags: open_close={open_close}, "
                 f"self_critique={self_critique}, custom_premise={'set' if custom_premise else 'empty'}")

        # ══════════════════════════════════════════════════════════════════════
        # PHASE 1: PRE-FLIGHT & INPUT VALIDATION (v1.1)
        # Catch bad configs before burning RTX 5080 compute time.
        # ══════════════════════════════════════════════════════════════════════

        # ── 1a. Parameter sanity checks ──
        if target_minutes == 1 and num_characters > 4:
            log.warning("[PreFlight] target_minutes=1 with %d characters is too many — "
                        "clamping to 3 characters for 1-min test", num_characters)
            _runtime_log("PREFLIGHT: Clamped num_characters to 3 (1-min episode)")
            num_characters = 3

        if target_minutes <= 2 and include_act_breaks:
            log.warning("[PreFlight] Act breaks disabled for %d-min episode (too short)", target_minutes)
            _runtime_log("PREFLIGHT: Act breaks disabled (episode too short)")
            include_act_breaks = False

        # ── 1b. Custom premise enforcement ──
        # When user provides a premise, skip RSS entirely — zero context contamination
        if custom_premise:
            open_close = False  # User already knows what story they want
            log.info("[PreFlight] Custom premise set — bypassing RSS fetch and Open-Close")
            _runtime_log("PREFLIGHT: Custom premise detected — RSS bypassed, Open-Close disabled")

        # ── 1c. Global token budgeting ──
        # Normalize target_minutes into a hard character budget.
        # ~130 words/min for dramatic reading, ~5 chars/word average.
        target_words = target_minutes * 130
        target_chars = target_words * 5  # Hard cap for downstream length enforcement

        # ── 1d. Episode fingerprint for reproducibility ──
        import hashlib
        fingerprint_data = f"{episode_title}|{genre_flavor}|{target_minutes}|{num_characters}|{temperature}"
        episode_fingerprint = hashlib.sha256(fingerprint_data.encode()).hexdigest()[:12]
        _runtime_log(f"ScriptWriter: FINGERPRINT {episode_fingerprint} | {episode_title} | {genre_flavor}")

        # ── Deterministic seeding from episode fingerprint ──
        # Same fingerprint → same torch RNG state → reproducible Gemma generation.
        try:
            import torch
            seed = int(episode_fingerprint, 16) % (2**31 - 1)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            random.seed(seed)
            _runtime_log(f"ScriptWriter: SEED {seed} (from fingerprint {episode_fingerprint})")
        except Exception as _seed_err:
            log.warning(f"[LLMScriptWriter] Could not set deterministic seed: {_seed_err}")

        # ══════════════════════════════════════════════════════════════════════
        # RSS FETCH (or custom premise bypass)
        # ══════════════════════════════════════════════════════════════════════

        if custom_premise:
            # Custom premise mode — build minimal news block from premise text
            news = [{
                "headline": episode_title or "Custom Episode",
                "summary": custom_premise[:500],
                "full_text": custom_premise,
                "source": "User Premise",
                "date": str(datetime.now().date()),
                "link": "",
            }]
            news_block = f"CUSTOM PREMISE (provided by user):\n{custom_premise}"
        else:
            # ── 1e. RSS fetch with deterministic fallback ──
            try:
                news = _fetch_science_news()
            except Exception as rss_err:
                log.warning("[PreFlight] RSS fetch failed: %s — using fallback seed", rss_err)
                _runtime_log(f"PREFLIGHT: RSS_FALLBACK — {rss_err}")
                # Deterministic fallback seeds — real science, manually curated
                _FALLBACK_SEEDS = [
                    {
                        "headline": "Deep-sea microbes found thriving in high-pressure volcanic vents challenge limits of life",
                        "summary": "Researchers discover extremophile bacteria colonies at 4,000m depth near hydrothermal vents "
                                   "that metabolize hydrogen sulfide at temperatures exceeding 120C, suggesting life may exist "
                                   "in similar conditions on Europa and Enceladus.",
                        "full_text": "Researchers discover extremophile bacteria colonies at 4,000m depth near hydrothermal vents "
                                     "that metabolize hydrogen sulfide at temperatures exceeding 120C. The organisms use a novel "
                                     "chemosynthetic pathway never observed before, converting volcanic minerals directly into "
                                     "cellular energy without sunlight. This discovery challenges our understanding of the minimum "
                                     "requirements for life and has major implications for astrobiology missions targeting ocean "
                                     "worlds like Europa and Enceladus.",
                        "source": "Nature Geoscience (fallback seed)",
                        "date": str(datetime.now().date()),
                        "link": "",
                    },
                    {
                        "headline": "Quantum entanglement maintained at room temperature for first time using diamond lattice",
                        "summary": "A team at ETH Zurich demonstrates stable quantum entanglement between nitrogen-vacancy centers "
                                   "in diamond at 22C for over 100 microseconds, eliminating the need for near-absolute-zero cooling.",
                        "full_text": "A team at ETH Zurich demonstrates stable quantum entanglement between nitrogen-vacancy centers "
                                   "in diamond at room temperature for over 100 microseconds. The breakthrough uses a novel spin-echo "
                                   "protocol that actively corrects thermal decoherence in real time. If scaled, the technique could "
                                   "enable practical quantum sensors for medical imaging and navigation systems that operate outside "
                                   "laboratory conditions.",
                        "source": "Physical Review Letters (fallback seed)",
                        "date": str(datetime.now().date()),
                        "link": "",
                    },
                    {
                        "headline": "CRISPR-based gene drive successfully suppresses invasive mosquito population in contained trial",
                        "summary": "A controlled field trial in Burkina Faso demonstrates that a CRISPR gene drive targeting female "
                                   "fertility reduced Anopheles gambiae populations by 90 percent within 8 generations.",
                        "full_text": "A controlled field trial demonstrates that a CRISPR gene drive targeting female fertility in "
                                     "Anopheles gambiae mosquitoes reduced populations by 90 percent within 8 generations inside a "
                                     "contained outdoor enclosure. The drive spread to 95 percent of the population within 4 generations. "
                                     "Researchers emphasize the need for further ecological impact studies before any open-release trials, "
                                     "but the results represent the most successful demonstration of gene drive technology in a near-wild setting.",
                        "source": "Science (fallback seed)",
                        "date": str(datetime.now().date()),
                        "link": "",
                    },
                ]
                news = [random.choice(_FALLBACK_SEEDS)]

            # ── 1f. Headline sanitization ──
            # Strip emojis, cap length, normalize whitespace to prevent prompt injection
            for n in news:
                # Remove emojis and non-ASCII decorators
                n["headline"] = re.sub(r'[^\x20-\x7E]', '', n["headline"]).strip()[:280]
                # Normalize whitespace
                n["headline"] = re.sub(r'\s+', ' ', n["headline"])
                # Cap full_text to prevent context window blowout
                if len(n.get("full_text", "")) > 12000:
                    n["full_text"] = n["full_text"][:12000] + "\n[... article truncated at 12,000 chars]"

        news_block = "\n".join(
            f"- {n['headline']} ({n['source']}, {n['date']})\n\n{n.get('full_text', n['summary'])}"
            for n in news
        )
        news_json = json.dumps(news, indent=2)

        # Calculate target words
        # target_words and target_chars already computed in Phase 1 pre-flight

        # ── Easter egg: 11% chance Lemmy appears as a character ──
        # A grizzled, seen-it-all engineer/mechanic who speaks in blunt,
        # colorful metaphors. Rare enough to be a surprise, frequent enough
        # that regulars will notice. Named after Lemmy Kilmister.
        # force_lemmy=True overrides for testing (validates voice collision fix).
        # Use _LEMMY_RNG (SystemRandom) instead of seeded `random` so the 11%
        # is actually 11% per run, not frozen by the per-episode fingerprint seed.
        _natural_roll = _LEMMY_RNG.random() < 0.11
        
        # Lemmy Telemetry Counter
        global _LEMMY_HISTORY
        _LEMMY_HISTORY.append(_natural_roll)
        if len(_LEMMY_HISTORY) > 50:
            _LEMMY_HISTORY.pop(0)
        _hits = sum(_LEMMY_HISTORY)
        _rate = (_hits / len(_LEMMY_HISTORY)) * 100
        _runtime_log(f"TELEMETRY: Lemmy hit rate [{_hits}/{len(_LEMMY_HISTORY)}] = {_rate:.1f}%")
        
        lemmy_roll = force_lemmy or _natural_roll
        if force_lemmy:
            _lemmy_source = "🔧 Lemmy was summoned by the boss (force toggle ON)"
        elif _natural_roll:
            _lemmy_source = "🎲 Lemmy rolled in on his own (lucky 11%)"
        else:
            _lemmy_source = "💤 Lemmy stayed in the garage tonight"
        log.info(f"[LLMScriptWriter] {_lemmy_source}  [force={force_lemmy}, rng_hit={_natural_roll}]")
        lemmy_directive = ""
        if lemmy_roll:
            lemmy_directive = (
                "\nSPECIAL CHARACTER REQUIREMENT: One of the characters MUST be named LEMMY — "
                "a resourceful, slightly unconventional engineer/mechanic who operates on the fringes "
                "of authority but proves essential in critical moments. He has a hands-on technical "
                "mindset, more comfortable solving problems directly than following protocol. "
                "Personality: dryly humorous, pragmatic, rough around the edges, but loyal and "
                "dependable when it counts. He questions leadership, bends rules, but his instincts "
                "are sharp under pressure. In the team dynamic Lemmy is the fixer and improviser — "
                "he adapts quickly, thinks creatively, and keeps things moving when plans fall apart. "
                "Give him at least 3 lines of dialogue. Use the name LEMMY consistently "
                "(not ENGINEER LEMMY, just LEMMY).\n"
                "LEMMY SFX REQUIREMENT: Before LEMMY's FIRST line of dialogue, you MUST include "
                "exactly this SFX cue on its own line:\n"
                "[SFX: heavy wrench strike on metal pipe, single resonant clank]\n"
                "This is his signature sound — it plays once, the first time he appears, nowhere else.\n"
            )
            log.info("[LLMScriptWriter] ★ Lemmy Easter egg activated (11%% roll) — wrench SFX cued")

        # ── Gemma owns character names — they become canonical character_ids ──
        # We do NOT pre-seed names. Gemma invents its own character names while
        # writing. Those names are stable pipeline keys used by BatchBark and
        # SceneSequencer. The Director adds a procedural display_name (e.g.
        # "BLAKE ARCHER") for human-facing output only — never as a pipeline key.

        # ── Phase 1b: Model Selection & Prompting ──
        # v1.4 Fix: Small models (2B) suffer from "Model Collapse" if the
        # prompt is too complex. We swap to a "LITE" version for these.
        # Check for 2B specifically (avoiding false hits on 26b or 31b)
        is_small_model = any(tag in model_id.lower() for tag in ("2b-it", "2b_it", "small")) or (model_id.lower().endswith("2b"))
        
        system_base = SCAFFOLDING_PREAMBLE + SCRIPT_SYSTEM_PROMPT
        if is_small_model:
            # Gemma 2B Lite role prevents prose and header hallucinations
            lite_role = "<system_role>STRICT OTR TAGS ONLY. No prose. Start every line with a tag.</system_role>"
            system_base = lite_role + "\n\n" + SCRIPT_SYSTEM_PROMPT
            
        system = system_base.format(
            target_minutes=target_minutes,
            target_words=target_words,
            news_block=news_block,
            num_characters=num_characters,
        )

        # ── PRE-ROLL DETERMINISTIC CAST ROSTER ──
        seed_str = f"{episode_title}_{target_minutes}_{style_variant}_{time.time()}"
        cast_rng = random.Random(seed_str)
        
        pre_rolled_cast = []
        seen_first = set()
        seen_last = set()
        num_non_announcers = max(1, num_characters)
        
        # Injected Lemmy: if he rolled in, he occupies one of the cast slots 
        # so he appears in the MANDATORY CAST ROSTER (ensuring Gemma uses him).
        if lemmy_roll:
            pre_rolled_cast.append("LEMMY")
            seen_first.add("LEMMY")

        while len(pre_rolled_cast) < num_non_announcers:
            f_name = cast_rng.choice(_FIRST_NAMES).upper()
            l_name = cast_rng.choice(_LAST_NAMES).upper()
            if f_name not in seen_first and l_name not in seen_last:
                seen_first.add(f_name)
                seen_last.add(l_name)
                pre_rolled_cast.append(f"{f_name} {l_name}")

        cast_roster_block = (
            "MANDATORY CAST ROSTER:\n"
            f"You MUST use exactly these {num_non_announcers} character names and no others for your speaking roles:\n"
            + "\n".join(f"- {n}" for n in pre_rolled_cast) + "\n"
            "Preserve spelling exactly. Do not introduce substitute names, nicknames, or titles. "
            "If ANNOUNCER is present, it does not count as a cast invention."
        )

        # ── Open-Close Expansion ──
        winning_outline = ""
        _runtime_log(f"ScriptWriter: OPEN-CLOSE CHECK: open_close={open_close} (type={type(open_close).__name__}), "
                     f"custom_premise='{custom_premise}' (bool={bool(custom_premise)}), "
                     f"condition={open_close and not custom_premise}")
        if open_close and not custom_premise:
            winning_outline = self._open_close_expansion(
                system, genre_flavor, news_block, num_characters,
                target_minutes, target_words, lemmy_directive,
                model_id, temperature, cast_roster_block=cast_roster_block
            )

        # ── Build final script prompt ──
        # Mode label must match the logic in _open_close_expansion_inner so the
        # downstream prompt asks the model to expand a PITCH (long episodes) or
        # an OUTLINE (short episodes) accordingly.
        oc_mode_label = "PITCH" if target_minutes >= 15 else "OUTLINE"
        if winning_outline:
            user_prompt = f"""Write a complete episode of "SIGNAL LOST" based on the WINNING {oc_mode_label} below.

LENGTH DIRECTIVE: {length_instruction}
STYLE DIRECTIVE: {style_instruction}

WINNING {oc_mode_label} (selected by evaluator from 3 competing concepts):
{winning_outline}

EPISODE TITLE: {episode_title if episode_title else "(generate a compelling, evocative title)"}
GENRE: {genre_flavor.replace("_", " ")}
CHARACTERS: {num_characters} speaking roles plus ANNOUNCER
{cast_roster_block}
TARGET LENGTH: ~{target_words} words ({target_minutes} minutes)
{"STRUCTURAL BREAKS: Include 2-3 act breaks marked with [ACT TWO], [ACT THREE] etc." if include_act_breaks else ""}
{lemmy_directive}

REMEMBER: The {oc_mode_label.lower()} above is your premise and story spine. {"Invent the full scene structure, acts, and SFX based on it." if oc_mode_label == "PITCH" else "Follow its structure, characters, and arc."} Flesh it out with sharp dialogue, atmospheric [SFX:] and [ENV:] tags, and real emotional stakes.

Begin the full script now. Follow this structure exactly:
=== SCENE 1 ===
[ENV: location description, ambient noise, vibe]
[SFX: establishing sound]
(beat)
[VOICE: ANNOUNCER, <male|female — ALTERNATE per episode, do NOT default to male>, <40s|50s|60s>, authoritative, calm] [Opening introduction — time, place, character names and roles, science hook, tagline. REQUIRED. Always first.]
[VOICE: CHARACTER_NAME, gender, age, tone, energy] First dramatic line — drop us in medias res.
[VOICE: CHARACTER_NAME, gender, age, tone, energy] Response line.
(beat)
[SFX: action sound]
...
[VOICE: ANNOUNCER, <same gender/age as opening>, authoritative, calm] [Hard-science epilogue — cite ONLY the real article provided above. Headline, source, date. No invented IDs.]
[MUSIC: Closing theme]"""
        else:
            user_prompt = f"""Write a complete episode of "SIGNAL LOST" — a contemporary sci-fi audio drama anthology.

LENGTH DIRECTIVE: {length_instruction}
STYLE DIRECTIVE: {style_instruction}

EPISODE TITLE: {episode_title if episode_title else "(generate a compelling, evocative title)"}
GENRE: {genre_flavor.replace("_", " ")}
CHARACTERS: {num_characters} speaking roles plus ANNOUNCER
{cast_roster_block}
TARGET LENGTH: ~{target_words} words ({target_minutes} minutes)
{"STRUCTURAL BREAKS: Include 2-3 act breaks marked with [ACT TWO], [ACT THREE] etc." if include_act_breaks else ""}
{lemmy_directive}
{"PREMISE: " + custom_premise if custom_premise else "The news headlines above ARE the premise. Extrapolate them. What's the next terrifying or profound step?"}

STORY ARC SEED: Use Arc Type {random.choice("ABCDEFGHIJKL")} from the Story Arc Engine above. Commit fully to that structure.

REMEMBER: Story first. Make the listener CARE about these people before you scare them with science. Write dialogue that sounds like real humans under pressure — not scientists reading papers.

Begin the full script now. Follow this structure exactly:
=== SCENE 1 ===
[ENV: location description, ambient noise, vibe]
[SFX: establishing sound]
(beat)
[VOICE: ANNOUNCER, <male|female — ALTERNATE per episode, do NOT default to male>, <40s|50s|60s>, authoritative, calm] [Opening introduction — time, place, character names and roles, science hook, tagline. REQUIRED. Always first.]
[VOICE: CHARACTER_NAME, gender, age, tone, energy] First dramatic line — drop us in medias res.
[VOICE: CHARACTER_NAME, gender, age, tone, energy] Response line.
(beat)
[SFX: action sound]
...
[VOICE: ANNOUNCER, <same gender/age as opening>, authoritative, calm] [Hard-science epilogue — cite ONLY the real article provided above. Headline, source, date. No invented IDs.]
[MUSIC: Closing theme]"""

        # v1.4 Theme C — prepend the series bible preamble so every downstream
        # phase (outline, draft, critique, revise, arc enhancer) sees the same
        # locked decisions. Empty preamble degrades gracefully to v1.3 behavior.
        if project_state_preamble:
            full_prompt = f"[SERIES BIBLE]\n{project_state_preamble}\n\n{system}\n\n{user_prompt}"
        else:
            full_prompt = f"{system}\n\n{user_prompt}"

        log.info(f"[LLMScriptWriter] Generating {target_minutes}min episode "
                 f"'{episode_title}' ({genre_flavor}) using {model_id}")
        log.info(f"[LLMScriptWriter] News seed: {news[0]['headline']} | {news[0]['source']}")

        # For episodes > 5 min, generate act-by-act to avoid token truncation.
        # 8,192 max_new_tokens ≈ 6,000 words. A 25-min episode needs ~3,250 words
        # which fits, but 45-min needs ~5,850 which is tight. Chunked generation
        # ensures we never hit the ceiling and produces more coherent long scripts.

        if target_minutes <= 5 or optimization_profile == "Obsidian (Low VRAM/Fast)":
            # Short episodes (or Obsidian 4GB tier): single-pass generation.
            # Floor at 1024 — even a 1-min episode needs enough tokens to
            # complete canonical structure (ENV, SFX, VOICE tags, beats).
            # Without the floor, 1-min = 260 tokens, which truncates mid-scene.
            
            # BUG-012 FIX: Cap KV cache for direct generation in Obsidian profile.
            # Standard: 8192 limit. Obsidian: 2500 limit (protects 4GB VRAM ceiling).
            if optimization_profile == "Obsidian (Low VRAM/Fast)":
                if target_minutes > 5:
                    log.warning("[LLMScriptWriter] Obsidian profile forced single-pass on %d min target. "
                                "Expect shorter overall length.", target_minutes)
                max_new_tokens = max(int(target_words * 2.0), 1024)
                max_new_tokens = min(max_new_tokens, 2500)
            else:
                max_new_tokens = max(int(target_words * 2.0), 1024)
                max_new_tokens = min(max_new_tokens, 8192)

            script_text = _generate_with_llm(
                full_prompt,
                model_id=model_id,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=active_top_p,
                optimization_profile=optimization_profile
            )
        else:
            # Long episodes: chunked act-by-act generation
            script_text = self._generate_chunked(
                system, episode_title, genre_flavor, num_characters,
                target_minutes, target_words, custom_premise, news_block,
                include_act_breaks, model_id, temperature,
                lemmy_directive=lemmy_directive,
                top_p=active_top_p,
                cast_roster_block=cast_roster_block,
                optimization_profile=optimization_profile
            )

        # ── v1.1 CHECKS & CRITIQUES LOOP ─────────────────────────────────────
        # Three-pass refinement: Draft → Critique → Revise
        # The LLM critiques its own script for structural, scientific, and
        # dramatic weaknesses, then rewrites based on the critique.
        # Adds ~2 extra inference passes but significantly elevates quality.
        # ──────────────────────────────────────────────────────────────────────
        _runtime_log(f"ScriptWriter: CRITIQUE CHECK: self_critique={self_critique} (type={type(self_critique).__name__})")
        if self_critique:
            _runtime_log("ScriptWriter: >>> ENTERING critique_and_revise")
            script_text = self._critique_and_revise(
                script_text, genre_flavor, target_words, model_id, temperature,
                optimization_profile=optimization_profile
            )
            _runtime_log("ScriptWriter: <<< EXITED critique_and_revise")

        # ── ARC ENHANCER (v1.3 flagship feature) ──────────────────────────────
        # Paired opening + closing bookend rewrite to ensure narrative coherence.
        # ──────────────────────────────────────────────────────────────────────
        if arc_enhancer:
            _runtime_log("ScriptWriter: >>> ENTERING arc_enhancer")
            script_text = self._execute_arc_enhancer(
                script_text, genre_flavor, episode_title, news_block, model_id, temperature,
                optimization_profile=optimization_profile
            )
            _runtime_log("ScriptWriter: <<< EXITED arc_enhancer")

        # ── Content safety filter — catch anything the prompt policy missed ──
        script_text, blocked = _content_filter(script_text)
        if blocked:
            log.warning("[LLMScriptWriter] Content filter caught %d word(s) — replaced with minced oaths",
                        len(blocked))

        # ── FIX-4 (v1.2): Stock-name leak guard ───────────────────────────────
        # Gemma sometimes types the wrong character name inside dialogue body —
        # e.g. "it keeps spiking when you talk about the frequencies, Rex"
        # when the intended character is VEX. This is NOT a hardcoded blocklist:
        # we extract the real roster from [VOICE: NAME, ...] tags, then scan
        # direct-address tokens (", Name." or "Name,") in dialogue body. Any
        # capitalized proper-noun-looking token that is NOT in the roster gets
        # replaced with the phonetically closest roster name via difflib.
        # Pure structural fix — no baked names anywhere.
        try:
            import difflib
            # v1.4 HACK: Strip markdown ** bolding before roster extraction
            _clean_script = re.sub(r'\*\*(\[.*?\])\*\*', r'\1', script_text)
            _roster = set(re.findall(r'\[VOICE:\s*([A-Z][A-Z0-9_ -]+)\s*,', _clean_script))
            if _roster:
                _roster_list = sorted(_roster)
                _leaks_fixed = 0
                # Match direct-address tokens in dialogue body.
                # 1. Title-case: "Rex." or ", Maya," — common in narrative speech.
                # 2. ALL-CAPS: "REX" or "MAYA" — common in direct address inside dialogue.
                # Token length 2-8 chars, followed by punctuation or whitespace.
                _addr_pat = re.compile(
                    r'(?<=[,\s])'
                    r'([A-Z][a-z]{1,7}|[A-Z]{2,8})'
                    r'(?=[.,!?\s])'
                )
                def _leak_fix(m):
                    nonlocal _leaks_fixed
                    token = m.group(1)
                    upper = token.upper()
                    if upper in _roster:
                        return token  # legit roster name
                    # Common English words — skip
                    if token.lower() in {
                        "the", "and", "but", "for", "with", "from", "into", "that",
                        "this", "then", "than", "when", "what", "will", "were",
                        "been", "have", "just", "only", "some", "such", "very",
                        "now", "yes", "no", "ok", "okay", "sir", "maam", "doctor",
                        "captain", "commander", "listen", "look", "hey", "wait",
                        "stop", "god", "lord", "earth", "mars", "moon", "sun",
                        "orion", "nasa", "please", "thanks", "maybe", "never",
                        "always", "forever", "tonight", "tomorrow", "yesterday",
                    }:
                        return token
                    # Phonetic match to closest roster name (cutoff 0.80 — strictly
                    # tuned to catch typos like 'Marten'->'Martin' without falsely
                    # capturing normal English prose or SFX tags).
                    match = difflib.get_close_matches(upper, _roster_list, n=1, cutoff=0.80)
                    if match:
                        _leaks_fixed += 1
                        # Preserve title-case for dialogue flow
                        return match[0].title()
                    return token
                script_text = _addr_pat.sub(_leak_fix, script_text)
                if _leaks_fixed:
                    log.warning(
                        "[LLMScriptWriter] NameLeakGuard: repaired %d typo/leak(s) "
                        "in dialogue body (roster=%s)",
                        _leaks_fixed, sorted(_roster)
                    )
        except Exception as _e:
            log.warning("[LLMScriptWriter] NameLeakGuard skipped: %s", _e)

        # ── Citation hallucination guard ──────────────────────────────────────
        # Gemma sometimes invents plausible-looking ArXiv IDs (arXiv:2401.XXXXX)
        # even when told not to. These look authoritative but are fabricated.
        # Detect and warn — the IDs are left in the text (stripping would create
        # jarring gaps) but the log makes the problem visible for review.
        _arxiv_pat = re.compile(r'\barXiv:\s*\d{4}\.\d{4,5}\b', re.IGNORECASE)
        _doi_pat   = re.compile(r'\bdoi\.org/10\.\d{4,}/\S+', re.IGNORECASE)
        hallucinated_ids = _arxiv_pat.findall(script_text) + _doi_pat.findall(script_text)

        # Cross-check against the real article source
        real_source_text = " ".join(
            f"{n['headline']} {n['source']} {n.get('full_text', n['summary'])}"
            for n in news
        ).lower()

        bad_ids = []
        for hid in hallucinated_ids:
            # If the ID string doesn't appear in any form in the real article
            # content we provided, it's almost certainly hallucinated
            if hid.lower().replace(" ", "") not in real_source_text.replace(" ", ""):
                bad_ids.append(hid)

        if bad_ids:
            log.warning(
                "[CitationGuard] %d likely hallucinated citation ID(s) detected: %s — "
                "Gemma invented these. Review the epilogue before publishing.",
                len(bad_ids), ", ".join(bad_ids)
            )
        elif hallucinated_ids:
            log.info("[CitationGuard] %d citation ID(s) found — appear to match source material.",
                     len(hallucinated_ids))

        # ── CitationGuard 2: strip numeric bracket references ─────────────────
        # Gemma sometimes outputs [1], [2], article #3 when the prompt uses
        # bracket-style placeholders like [SOURCE] as format examples. These
        # become broken grammar when spoken ("According to article .") because
        # _clean_text_for_bark already strips unrecognized bracket tags.
        # Strip them here at the source so the script text is clean before
        # _parse_script() stores it.
        _num_ref_pat = re.compile(
            r'\s*\[\d{1,3}\]'           # [1] [2] [99]
            r'|\s*\(\d{1,3}\)'          # (1) (2)
            r'|\s*article\s+#\s*\d+'    # article #3
            r'|\s*source\s+#\s*\d+'     # source #2
            r'|\s*reference\s+#\s*\d+', # reference #1
            re.IGNORECASE
        )
        stripped_text, nsubs = _num_ref_pat.subn("", script_text)
        if nsubs:
            log.warning(
                "[CitationGuard] Stripped %d numeric citation marker(s) ([1], article #N, etc.) "
                "from script text — update prompt to prevent recurrence.", nsubs
            )
            script_text = stripped_text

        # Parse into structured JSON
        script_lines = self._parse_script(script_text)
        script_json = json.dumps(script_lines, indent=2)

        # Estimate actual minutes
        word_count = sum(len(line.get("line", "").split()) for line in script_lines
                         if line.get("type") == "dialogue")
        est_minutes = max(1, word_count // 130)

        # ── Phase 1g: Cast map verification ──
        # Extract unique character names from parsed script for downstream matching
        script_characters = set()
        for item in script_lines:
            if item.get("type") == "dialogue":
                cname = item.get("character_name", "").upper().strip()
                if cname:
                    script_characters.add(cname)
        _runtime_log(f"ScriptWriter: CAST_MAP {sorted(script_characters)} | "
                     f"{len(script_lines)} lines | ~{word_count} words | ~{est_minutes} min")

        # ── Phase 3d: QA debug dump ──
        # Save minimal JSON payload alongside the output for reproducibility
        try:
            qa_data = {
                "fingerprint": episode_fingerprint,
                "timestamp": datetime.now().isoformat(),
                "params": {
                    "episode_title": episode_title,
                    "genre_flavor": genre_flavor,
                    "target_minutes": target_minutes,
                    "num_characters": num_characters,
                    "open_close": open_close,
                    "self_critique": self_critique,
                    "temperature": temperature,
                },
                "news_seed": news[0]["headline"] if news else "none",
                "news_source": news[0].get("source", "unknown") if news else "none",
                "cast": sorted(script_characters),
                "stats": {
                    "dialogue_lines": sum(1 for l in script_lines if l.get("type") == "dialogue"),
                    "sfx_cues": sum(1 for l in script_lines if l.get("type") == "sfx"),
                    "scenes": sum(1 for l in script_lines if l.get("type") == "scene_break"),
                    "word_count": word_count,
                    "est_minutes": est_minutes,
                    "script_chars": len(script_text),
                },
                "guardrails_triggered": [],  # Populated by downstream phases
            }
            qa_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                   "output", "old_time_radio",
                                   f"qa_debug_{episode_fingerprint}.json")
            os.makedirs(os.path.dirname(qa_path), exist_ok=True)
            with open(qa_path, "w", encoding="utf-8") as f:
                json.dump(qa_data, f, indent=2)
            _runtime_log(f"ScriptWriter: QA_DUMP saved: qa_debug_{episode_fingerprint}.json")
        except Exception as qa_err:
            log.warning("[QA] Debug dump failed: %s", qa_err)

        log.info(f"[LLMScriptWriter] Generated {len(script_lines)} lines, "
                 f"~{word_count} words, ~{est_minutes} min")

        # ── VRAM handoff: unload Gemma before Bark loads ──────────────────────
        # Gemma and Bark cannot share 16GB VRAM comfortably. Explicitly unload
        # now so BatchBark starts with a clean VRAM slate.
        _unload_llm()
        _runtime_log("ScriptWriter: Gemma unloaded — VRAM freed for Bark")

        # v1.4 Theme C — exit snapshot after Gemma unload. This should be
        # close to the idle baseline; a large value here means the unload
        # path left memory on the table and needs investigation.
        vram_snapshot("script_writer_exit_after_unload")

        return (script_text, script_json, news_json, est_minutes)

    # ─────────────────────────────────────────────────────────────────────────
    # OPEN-CLOSE EXPANSION — 3 competing outlines → evaluator picks winner
    # ─────────────────────────────────────────────────────────────────────────

    def _open_close_expansion(self, system, genre_flavor, news_block,
                              num_characters, target_minutes, target_words,
                              lemmy_directive, model_id, temperature,
                              cast_roster_block="", optimization_profile="Standard"):
        """Generate 3 competing story outlines with different priorities,
        then have an evaluator pick the best one.

        Outline A: Prioritizes character conflict and emotional stakes.
        Outline B: Prioritizes scientific rigor and technical tension.
        Outline C: Prioritizes atmosphere, pacing, and environmental tension.

        The evaluator receives all 3 and selects the strongest narrative,
        optionally merging the best elements from each.

        Returns the winning outline text, or empty string on failure.
        """
        try:
            return self._open_close_expansion_inner(
                system, genre_flavor, news_block, num_characters,
                target_minutes, target_words, lemmy_directive,
                model_id, temperature,
                cast_roster_block=cast_roster_block,
                optimization_profile=optimization_profile
            )
        except Exception as e:
            log.error("[OpenClose] Top-level failure: %s — falling back to v1.0 direct generation", e)
            _runtime_log(f"OPENCLOSE: OPENCLOSE_FALLBACK — top-level error: {e}")
            return ""

    def _open_close_expansion_inner(self, system, genre_flavor, news_block,
                                     num_characters, target_minutes, target_words,
                                     lemmy_directive, model_id, temperature,
                                     cast_roster_block="", optimization_profile="Standard"):
        """Inner implementation of Open-Close expansion (wrapped for safety)."""
        log.info("[OpenClose] Starting Open-Close expansion (3 outlines + evaluator)...")
        _runtime_log("OPENCLOSE: Generating 3 competing outlines")

        # ── PITCH MODE (Gemini round 3) ──
        # For long episodes (>= 15 min) the 3 full structured outlines bottleneck
        # the run (~10-15 min just for the open-close phase on SDPA). For long
        # episodes we switch to "pitch mode" — a 3-5 sentence logline per concept,
        # ~100 words, no act structure. Saves ~80% of open-close inference time.
        # The full script generator still invents the scene structure downstream.
        is_pitch_mode = target_minutes >= 15
        if is_pitch_mode:
            mode_label = "PITCH"
            outline_max_tokens = 250
            OUTLINE_MIN = 150
            OUTLINE_MAX = 1500
            _runtime_log(
                f"OPENCLOSE: PITCH_MODE enabled for {target_minutes}m run "
                f"(max_new_tokens={outline_max_tokens})"
            )
        else:
            mode_label = "OUTLINE"
            outline_max_tokens = 450   # was 600 — trimmed to stay inside SDPA ~420s budget
            OUTLINE_MIN = 200
            OUTLINE_MAX = 3000
            _runtime_log(
                f"OPENCLOSE: OUTLINE_MODE enabled for {target_minutes}m run "
                f"(max_new_tokens={outline_max_tokens})"
            )
        mode_lower = mode_label.lower()

        arc_choices = random.sample("ABCDEFGHIJKL", 3)

        outline_focuses = [
            ("CHARACTER-DRIVEN",
             "Focus on intense interpersonal conflict. Give each character a secret, "
             "a fear, and a breaking point. The science is the pressure cooker — "
             "the people are the story. Make us feel their desperation."),
            ("SCIENCE-DRIVEN",
             "Focus on scientific rigor and technical problem-solving. The plot should "
             "hinge on a real physics/biology constraint that characters must solve under "
             "pressure. Think Apollo 13 — the math IS the drama."),
            ("ATMOSPHERE-DRIVEN",
             "Focus on environmental dread and sensory immersion. Use sound design cues "
             "([SFX:], [ENV:]) heavily. Build a world the listener can HEAR — creaking metal, "
             "distant alarms, breathing in a spacesuit. Slow-burn tension."),
        ]

        # v1.4 Theme B — 3-outline evaluator re-enabled.
        #
        # History: this flag was introduced in v1.3 to mitigate token-stream
        # corruption from CONCURRENT generation across threads. The underlying
        # _generate_with_llm shares a single cached model, a single streamer,
        # and a single CUDA context, so parallel calls are undefined behavior.
        #
        # The ROADMAP hard rule is "Sequential execution only. ComfyUI manages
        # the queue." So we re-enable the evaluator in SEQUENTIAL mode — the
        # three outlines are generated one at a time through the loop below.
        # Per-outline budget is already tuned: 450 tok / 480s wall. Three
        # serial outlines at ~2 tok/s ≈ 12 minutes worst case for OUTLINE mode,
        # under 3 minutes for PITCH mode. This is the cost of diversity: the
        # evaluator gets three genuinely different focuses (CHARACTER-DRIVEN,
        # SCIENCE-DRIVEN, ATMOSPHERE-DRIVEN) and picks the strongest one.
        #
        # Do NOT wrap the loop in a ThreadPoolExecutor. Parallel generation on
        # a shared Gemma model will corrupt the token streams — the same bug
        # this flag was originally put in place to prevent.
        ENABLE_3_OUTLINE_EVALUATOR = True

        if not ENABLE_3_OUTLINE_EVALUATOR:
            outline_focuses = [
                ("STORY-DRIVEN", "Focus on a balanced narrative arc, strong characters, and scientific plausibility.")
            ]
        else:
            _runtime_log(
                f"OPENCLOSE: 3-outline evaluator ACTIVE (sequential) — "
                f"{len(outline_focuses)} focuses: "
                f"{', '.join(name for name, _ in outline_focuses)}"
            )

        outlines = []
        for i, (focus_name, focus_desc) in enumerate(outline_focuses):
            if is_pitch_mode:
                # PITCH mode: lightweight 3-5 sentence logline per concept
                concept_body = f"""Generate a distinct story PITCH for a {genre_flavor.replace("_", " ")} radio drama episode.

PRIORITY: {focus_name}
{focus_desc}

CRITICAL CONSTRAINTS:
- Exactly 3 to 5 sentences. 50-100 words total.
- High-level logline only. No act structure. No scene breakdown. No dialogue.
- Hook + core conflict + science angle.
- The science must be rooted in the real headlines from the system prompt above.

ARC TYPE: Use Arc Type {arc_choices[i]} from the Story Arc Engine above.
TARGET LENGTH (downstream script): {target_minutes} minutes
{lemmy_directive}

Begin your PITCH now:"""
            else:
                # OUTLINE mode: full structured outline (short episodes only)
                concept_body = f"""Generate a STORY OUTLINE (not a full script) for a {genre_flavor.replace("_", " ")} radio drama.

PRIORITY: {focus_name}
{focus_desc}

CRITICAL: The science news headlines in the system prompt above ARE your raw material. Your premise MUST be rooted in those real headlines — extrapolate the science to its most dramatic, terrifying, or profound next step. Do NOT invent unrelated premises.

ARC TYPE: Use Arc Type {arc_choices[i]} from the Story Arc Engine above.
{cast_roster_block if cast_roster_block else f"CHARACTERS: {num_characters} speaking roles plus ANNOUNCER"}
TARGET LENGTH: {target_minutes} minutes (~{target_words} words when fully scripted)
{lemmy_directive}

Output a concise outline with:
- PREMISE: 2-3 sentences. What's the hook? Must reference the real science from the news above.
- CHARACTERS: Name, role, key trait, internal conflict (one line each)
- ACT 1 (BEGINNING): Setup, inciting incident. 3-4 sentences.
- ACT 2 (MIDDLE): Rising tension, complications, reversals. 4-5 sentences.
- ACT 3 (END): Climax, resolution, epilogue hook. 3-4 sentences.
- KEY SFX: 3-5 signature sound moments that define the atmosphere.

Keep it under 400 words. Structure only — no dialogue."""

            outline_prompt = f"{system}\n\n{concept_body}"

            try:
                outline_text = _run_with_timeout(
                    lambda op=outline_prompt: _generate_with_llm(
                        op,
                        model_id=model_id,
                        max_new_tokens=outline_max_tokens,
                        temperature=min(1.0, temperature + 0.1),
                        optimization_profile=optimization_profile
                    ),
                    timeout_sec=480,   # was 300 — raised to 8min for SDPA @ ~2 tok/s
                    phase_label=f"OpenClose-{mode_label}-{focus_name}",
                )
                outlines.append((focus_name, outline_text))
                log.info("[OpenClose] %s %s generated (%d chars)",
                         mode_label, focus_name, len(outline_text))
                _runtime_log(f"OPENCLOSE: {mode_label} {focus_name} done ({len(outline_text)} chars)")
            except Exception as e:
                log.warning("[OpenClose] %s %s failed: %s", mode_label, focus_name, e)
                outlines.append((focus_name, ""))

        # ── Phase 2a: Open-Close boundary enforcement ──
        # Discard outlines outside the mode-specific char range before evaluator.
        # OUTLINE_MIN / OUTLINE_MAX are set above based on pitch vs outline mode.
        valid_outlines = []
        for name, text in outlines:
            if not text or len(text) < OUTLINE_MIN:
                log.warning("[OpenClose] Outline %s too short (%d chars < %d) — discarded",
                            name, len(text) if text else 0, OUTLINE_MIN)
                _runtime_log(f"OPENCLOSE: DISCARDED {name} (too short: {len(text) if text else 0} chars)")
                continue
            if len(text) > OUTLINE_MAX:
                log.warning("[OpenClose] Outline %s too long (%d chars > %d) — truncating",
                            name, len(text), OUTLINE_MAX)
                text = text[:OUTLINE_MAX] + "\n[... outline truncated]"
                _runtime_log(f"OPENCLOSE: TRUNCATED {name} to {OUTLINE_MAX} chars")
            valid_outlines.append((name, text))
        if not valid_outlines:
            log.warning("[OpenClose] All outlines failed — falling back to direct generation")
            _runtime_log("OPENCLOSE: All outlines failed")
            return ""

        if len(valid_outlines) == 1:
            log.info("[OpenClose] Only 1 outline survived — using it directly")
            return valid_outlines[0][1]

        # ── EVALUATOR: pick the best outline ──
        log.info("[OpenClose] Evaluating %d outlines...", len(valid_outlines))
        _runtime_log("OPENCLOSE: Evaluator picking winner")

        outlines_block = ""
        for idx, (name, text) in enumerate(valid_outlines, 1):
            outlines_block += f"\n--- {mode_label} {idx} ({name}) ---\n{text}\n"

        eval_prompt = f"""You are a veteran radio drama showrunner selecting the best story concept for production.

Below are {len(valid_outlines)} competing {mode_lower}s for a {genre_flavor.replace("_", " ")} episode.

Evaluate each on:
1. HOOK STRENGTH: Would a listener stay past the first 30 seconds?
2. CHARACTER DEPTH: Do the characters feel real and distinct?
3. NARRATIVE ARC: Is there clear escalation, a satisfying climax, and earned resolution?
4. SCIENTIFIC PLAUSIBILITY: Is the science grounded or handwavy?
5. AUDIO POTENTIAL: Will this sound amazing as a radio drama? Strong SFX moments?
6. EAR FLOW: Does the premise lend itself to short, punchy, spoken-aloud dialogue (X Minus One / Suspense style)? Will lines be 5-15 words, rhythmic, easy to say in one breath? Reject outlines that imply long expository monologues or tongue-twister jargon.

{outlines_block}

YOUR DECISION:
First, write ONE sentence about each {mode_lower}'s biggest strength and weakness.
Then state: "WINNER: {mode_label} N" (the number).
Finally, if elements from a losing {mode_lower} would strengthen the winner, list them as "MERGE: [element]".

Output the WINNING {mode_label} in full at the end, incorporating any merged elements.
Label it "FINAL {mode_label}:" on its own line before the text."""

        try:
            eval_text = _run_with_timeout(
                lambda: _generate_with_llm(
                    eval_prompt,
                    model_id=model_id,
                    max_new_tokens=800,
                    temperature=max(0.3, temperature - 0.3),
                    optimization_profile=optimization_profile
                ),
                timeout_sec=300,
                phase_label="OpenClose-Evaluator",
            )
            log.info("[OpenClose] Evaluator complete (%d chars)", len(eval_text))
            _runtime_log(f"OPENCLOSE: Evaluator done ({len(eval_text)} chars)")
        except Exception as e:
            log.warning("[OpenClose] Evaluator failed: %s — using first outline", e)
            return valid_outlines[0][1]

        # Extract the final concept from evaluator output.
        # Marker is mode-specific: "FINAL PITCH:" or "FINAL OUTLINE:". Try the
        # current mode first, then the other (in case the model picked the wrong
        # header), then the generic fallbacks.
        for marker in (f"FINAL {mode_label}:", "FINAL OUTLINE:", "FINAL PITCH:"):
            marker_idx = eval_text.upper().find(marker.upper())
            if marker_idx >= 0:
                winning = eval_text[marker_idx + len(marker):].strip()
                log.info("[OpenClose] Extracted winning %s via marker '%s' (%d chars)",
                         mode_lower, marker, len(winning))
                return winning

        # If no marker found, try to find "WINNER:" and return corresponding concept
        winner_match = re.search(
            rf'WINNER:\s*(?:{mode_label}|Outline|Pitch)\s*(\d)',
            eval_text, re.IGNORECASE,
        )
        if winner_match:
            winner_idx = int(winner_match.group(1)) - 1
            if 0 <= winner_idx < len(valid_outlines):
                log.info("[OpenClose] Winner is %s %d (%s)",
                         mode_label, winner_idx + 1, valid_outlines[winner_idx][0])
                return valid_outlines[winner_idx][1]

        # Fallback: return the full evaluator output (it usually contains a merged outline)
        log.info("[OpenClose] No clean marker found — using full evaluator output as outline")
        return eval_text

    # ─────────────────────────────────────────────────────────────────────────
    # CHECKS & CRITIQUES — Draft -> Critique -> Revise
    # ─────────────────────────────────────────────────────────────────────────

    def _critique_and_revise(self, draft_text, genre_flavor, target_words,
                             model_id, temperature, optimization_profile="Standard"):
        """Three-pass refinement: the LLM critiques its own draft, then revises.

        Pass 1 (already done): Draft generation (the script_text we received).
        Pass 2 (Critique):     LLM acts as a harsh script editor. Outputs a
                               numbered improvement plan — NO rewriting.
        Pass 3 (Revision):     LLM receives draft + critique, rewrites the
                               script implementing the specific fixes.

        Returns the revised script text, or the original draft if critique
        fails or produces nothing useful.
        """
        log.info("[Critique] Starting Checks & Critiques loop (Draft -> Critique -> Revise)...")
        _runtime_log("CRITIQUE: Starting self-critique pass")

        # ── Truncate draft for critique context ──
        # Keep the full draft but cap at ~12k chars to stay within context window.
        # The critique doesn't need every word — it needs the structure and flow.
        draft_for_critique = draft_text
        if len(draft_text) > 12000:
            # Keep first 6000 + last 6000 so critique sees beginning AND ending
            draft_for_critique = (
                draft_text[:6000]
                + "\n\n[... MIDDLE SECTION OMITTED FOR BREVITY ...]\n\n"
                + draft_text[-6000:]
            )

        # ── Pass 2: CRITIQUE ──
        critique_prompt = f"""You are a HARSH but constructive script editor for a {genre_flavor.replace("_", " ")} radio drama.

Below is a draft script. Your job is to identify SPECIFIC weaknesses. Do NOT rewrite anything.

Output a numbered list of 5-8 concrete problems, each one sentence. Focus on:
1. STORY ARC: Does it have a clear hook, rising tension, climax, and resolution? Or does it meander?
2. CHARACTER: Do characters sound distinct? Do they have clear motivations? Or are they interchangeable talking heads?
3. DIALOGUE: Does it sound like real humans under pressure? Or stilted/expository?
4. PACING: Are there dead spots? Does tension build or stay flat?
5. SCIENCE: Is the science grounded in real physics/biology? Any obvious handwaving?
6. ENDING: Does the resolution feel earned or rushed? Does the epilogue connect to the story?
7. AUDIO DESIGN: Are [SFX:] and [ENV:] tags used effectively to build atmosphere? Or sparse/generic?
8. EAR TEST (CRITICAL): Read every line aloud in your head. Does it sound like natural spoken English a real person would say in 5-15 words? Flag any line that is: longer than 15 words, full of jargon, missing contractions, or reads like written prose instead of speech. Flag any character name that is hard to say aloud or longer than 2 syllables.

Be brutal. Be specific. Name the exact scene or line that's weak.
Do NOT include any script text in your response — critique ONLY.

DRAFT SCRIPT:
{draft_for_critique}

YOUR CRITIQUE (numbered list only):"""

        try:
            critique_tokens = min(800, max(300, len(draft_text) // 20))
            critique_text = _run_with_timeout(
                lambda: _generate_with_llm(
                    critique_prompt,
                    model_id=model_id,
                    max_new_tokens=critique_tokens,
                    temperature=0.3,
                    top_p=0.9,
                    optimization_profile=optimization_profile
                ),
                timeout_sec=300,
                phase_label="Critique-Pass",
            )
            log.info("[Critique] Critique pass complete (%d chars)", len(critique_text))
            _runtime_log(f"CRITIQUE: Critique pass done ({len(critique_text)} chars)")
        except Exception as e:
            log.warning("[Critique] Critique pass failed: %s — returning original draft", e)
            _runtime_log(f"CRITIQUE: Failed — {e}")
            return f"{draft_text}\n\n[SYSTEM_SENTINEL: TIMEOUT_FALLBACK]"

        # Sanity check: critique should be a numbered list, not a rewrite
        if not critique_text or len(critique_text) < 50:
            log.warning("[Critique] Critique too short (%d chars) — skipping revision",
                        len(critique_text) if critique_text else 0)
            _runtime_log("CRITIQUE: CRITIQUE_SKIPPED — critique too short")
            return draft_text

        # ── Phase 2c: Critique format validation ──
        # Verify the critique looks like a numbered list, not a rewrite
        _critique_markers = re.findall(r'^\s*\d+[\.\):]', critique_text, re.MULTILINE)
        _critique_keywords = sum(1 for kw in ["weak", "issue", "problem", "flat", "generic",
                                               "missing", "rushed", "unclear", "improve"]
                                 if kw in critique_text.lower())
        if len(_critique_markers) < 2 and _critique_keywords < 2:
            log.warning("[Critique] Critique doesn't look like a numbered list "
                        "(%d markers, %d keywords) — may be a rewrite, skipping revision",
                        len(_critique_markers), _critique_keywords)
            _runtime_log("CRITIQUE: CRITIQUE_SKIPPED — critique format invalid")
            return draft_text

        # ── Pass 3: REVISION ──
        log.info("[Critique] Starting revision pass with %d-char critique...", len(critique_text))
        _runtime_log("CRITIQUE: Starting revision pass")

        revision_prompt = f"""You are the original writer of this {genre_flavor.replace("_", " ")} radio drama script.
A tough editor has reviewed your draft and provided specific critique.

YOUR TASK: Rewrite the COMPLETE script, implementing every critique point below.
Keep everything that already works. Fix only what the editor flagged.

RULES:
- Output the FULL revised script — not a summary, not highlights, the COMPLETE script.
- Maintain ALL canonical formatting: [VOICE:], [SFX:], [ENV:], (beat), === SCENE N ===
- Do NOT add new characters unless the critique specifically demands it.
- Do NOT change character names.
- Do NOT remove the ANNOUNCER opening or closing epilogue.
- Keep the same approximate length (~{target_words} words).
- Make dialogue sharper, more natural, more emotionally grounded.
- Strengthen the story arc wherever the critique identifies weakness.

EDITOR'S CRITIQUE:
{critique_text}

ORIGINAL DRAFT:
{draft_text}

REVISED SCRIPT (complete, from === SCENE 1 === to [MUSIC: Closing theme]):"""

        try:
            # FIX-1 (v1.2): Size revision budget from DRAFT LENGTH, not target_words.
            # Previously used target_words*2.0 which gave ~2080 tokens for 8-min eps —
            # but an 8-min draft runs ~10k chars (~2500 tokens), so the revision pass
            # got decapitated mid-Scene 4. Scene 4 is where the ending lives, which is
            # why every critique flagged "weak ending". Not a writing bug — a budget bug.
            # Formula: draft_chars / 3.5 chars-per-token * 1.25 safety margin.
            draft_token_estimate = int(len(draft_text) / 3.5)
            revision_tokens = max(int(draft_token_estimate * 1.25), int(target_words * 2.0), 2048)
            revision_tokens = min(revision_tokens, 8192)
            log.info("[Critique] Revision token budget: %d (draft_est=%d, target_words=%d)",
                     revision_tokens, draft_token_estimate, target_words)
            # BUG-005 fix: scale wall-clock budget to episode length AND draft size.
            # SDPA on 4-expert MoE models runs ~2-3 tok/s, so a 22k-char revision needs
            # ~700-1100s. The previous fixed 600s killed every long episode.
            # Formula: max(600, target_minutes*60, len(draft)*0.05)
            # target_words = target_minutes * 130, so target_minutes ≈ target_words / 130
            target_minutes_est = max(1, int(target_words / 130))
            revision_timeout = int(max(
                600,
                target_minutes_est * 60,
                len(draft_text) * 0.05,
            ))
            log.info("[Critique] Revision wall-clock budget: %ds (target_min~%d, draft=%d chars)",
                     revision_timeout, target_minutes_est, len(draft_text))
            revised_text = _run_with_timeout(
                lambda: _generate_with_llm(
                    revision_prompt,
                    model_id=model_id,
                    max_new_tokens=revision_tokens,
                    temperature=temperature,
                    optimization_profile=optimization_profile
                ),
                timeout_sec=revision_timeout,
                phase_label="Revision-Pass",
            )
            log.info("[Critique] Revision pass complete (%d chars)", len(revised_text))
            _runtime_log(f"CRITIQUE: Revision done ({len(revised_text)} chars)")
        except Exception as e:
            log.warning("[Critique] Revision pass failed: %s — returning original draft", e)
            _runtime_log(f"CRITIQUE: Revision failed — {e}")
            return f"{draft_text}\n\n[SYSTEM_SENTINEL: TIMEOUT_FALLBACK]"

        # ── Phase 2b: Critique length & format guardrails ──

        # Check 1: Revision must be at least 60% of draft length (not a summary)
        if len(revised_text) < len(draft_text) * 0.6:
            log.warning(
                "[Critique] Revision too short (%d chars vs %d draft) — "
                "LLM may have summarized instead of rewriting. Keeping original draft.",
                len(revised_text), len(draft_text)
            )
            _runtime_log("CRITIQUE: CRITIQUE_SKIPPED — revision too short")
            return draft_text

        # Check 2: Revision must not exceed 250% of draft length (runaway expansion)
        if len(revised_text) > len(draft_text) * 2.5:
            log.warning(
                "[Critique] Revision too long (%d chars vs %d draft, %.0f%%) — "
                "LLM expanded beyond acceptable bounds. Keeping original draft.",
                len(revised_text), len(draft_text),
                len(revised_text) / len(draft_text) * 100
            )
            _runtime_log("CRITIQUE: CRITIQUE_SKIPPED — revision too long (%.0f%%)" %
                         (len(revised_text) / len(draft_text) * 100))
            return draft_text

        # Check 3: Levenshtein similarity ratio — catch both lazy copies and hallucinations
        # Use simple character overlap ratio (fast approximation of edit distance)
        def _char_overlap_ratio(a, b):
            """Fast character-level similarity: shared chars / max length."""
            if not a or not b:
                return 0.0
            from collections import Counter
            ca, cb = Counter(a.lower()), Counter(b.lower())
            shared = sum((ca & cb).values())
            return shared / max(len(a), len(b))

        similarity = _char_overlap_ratio(draft_text, revised_text)
        _runtime_log(f"CRITIQUE: Similarity ratio: {similarity:.3f}")

        if similarity > 0.95:
            log.warning("[Critique] Revision too similar to draft (%.1f%% overlap) — "
                        "LLM likely copied instead of revising. Keeping original draft.",
                        similarity * 100)
            _runtime_log("CRITIQUE: CRITIQUE_SKIPPED — revision is a copy (%.1f%%)" % (similarity * 100))
            return draft_text

        if similarity < 0.35:
            log.warning("[Critique] Revision too different from draft (%.1f%% overlap) — "
                        "LLM may have hallucinated a new story. Keeping original draft.",
                        similarity * 100)
            _runtime_log("CRITIQUE: CRITIQUE_SKIPPED — revision is a hallucination (%.1f%%)" % (similarity * 100))
            return draft_text

        log.info("[Critique] Checks & Critiques complete — revised script accepted "
                 "(similarity=%.1f%%, length ratio=%.0f%%).",
                 similarity * 100, len(revised_text) / len(draft_text) * 100)
        _runtime_log("CRITIQUE: Revised script accepted (sim=%.1f%%, len=%.0f%%)" %
                     (similarity * 100, len(revised_text) / len(draft_text) * 100))
        return revised_text

    def _generate_chunked(self, system, title, genre, num_chars, target_min,
                          target_words, premise, news_block, act_breaks,
                          model_id, temperature, lemmy_directive="", top_p=0.95,
                          cast_roster_block="", optimization_profile="Standard"):
        """Generate long scripts act-by-act to avoid token truncation.

        Step 1: Generate an outline (characters, plot beats, act structure)
        Step 2: Generate each act using the outline + previous act as context
        Step 3: Concatenate into the final script
        """
        num_acts = 3 if target_min >= 20 else 2
        words_per_act = target_words // num_acts

        # Step 1: Outline
        outline_prompt = f"""{system}

Create a detailed OUTLINE for a {target_min}-minute episode of "SIGNAL LOST."
Title: {title}
Genre: {genre.replace("_", " ")}
Characters: {num_chars} speaking roles plus ANNOUNCER
{cast_roster_block}
{lemmy_directive}

Return:
- Character list: name, role, gender, personality, and what they PERSONALLY have at stake (~50/50 male/female split)
- Time period and setting (derived from the science news)
- {num_acts}-act structure: inciting incident, escalation beats, twist/resolution — focus on HUMAN drama, not science exposition
- At least one moment of humor, warmth, or unexpected humanity
- The ANNOUNCER's hard-science epilogue topic and sources to cite
- Key SFX and music cues

STORY ARC SEED: Use Arc Type {random.choice("ABCDEFGH")} from the Story Arc Engine. Commit fully to that structure.

Remember: This is a DRAMA that happens to involve science, not a science report with characters. Give every character something personal to lose.

{"Premise: " + premise if premise else "The news headlines ARE the premise. Extrapolate the science into its most dramatic next step."}

Outline only — do NOT write dialogue yet."""

        # BUG-011 FIX: Reduce KV Cache allocation overhead.
        # Outline is instructed to be under 400 words. max_new_tokens=1500 pre-allocates
        # excessive KV cache, which immediately overflows the 4GB ceiling and causes 100% GPU
        # PCIe memory thrashing (0.1 tok/sec behavior, appearing as a hang).
        outline_budget = 600 if optimization_profile == "Obsidian (Low VRAM/Fast)" else 1200
        
        log.info(f"[ScriptWriter] Generating outline ({num_acts} acts) [KV Budget: {outline_budget}]")
        outline = _generate_with_llm(outline_prompt, model_id=model_id,
                                         max_new_tokens=outline_budget, temperature=temperature, top_p=top_p,
                                         optimization_profile=optimization_profile)

        # Step 2: Generate each act with Context Engineering
        # Instead of dumping raw previous text, we summarize what happened
        # and signpost key character states for continuity.
        acts = []
        act_summaries = []  # Running narrative memory

        for act_num in range(1, num_acts + 1):
            # S29: Allow users to cancel long script generation
            try:
                import comfy.model_management
                comfy.model_management.throw_exception_if_processing_interrupted()
            except ImportError:
                pass

            # ── Context Engineering: curated memory instead of raw dump ──
            if acts:
                # Summarize previous act for tight context (not raw 2000 chars)
                if not act_summaries:
                    # Generate a quick summary of Act 1 for Act 2's context
                    _act_text_for_summary = _truncate_at_sentence_boundary(acts[-1], 3000)
                    summary_prompt = f"""Summarize the following radio drama act in 3-5 sentences.
Focus on: what happened, how each character's emotional state changed, what's at stake going into the next act, and any unresolved tensions.
Do NOT include dialogue. Just narrative summary.

ACT TEXT:
{_act_text_for_summary}

SUMMARY:"""
                    try:
                        summary = _generate_with_llm(
                            summary_prompt, model_id=model_id,
                            max_new_tokens=200, temperature=0.3,
                            optimization_profile=optimization_profile
                        )
                        act_summaries.append(summary)
                        _runtime_log(f"ScriptWriter: Act {act_num-1} summarized for context")
                    except Exception:
                        # Fallback: sentence-boundary truncation (no mid-sentence cuts)
                        act_summaries.append(_truncate_at_sentence_boundary(acts[-1], 1500))
                else:
                    # Summarize the latest act and append to running memory
                    _act_text_for_summary = _truncate_at_sentence_boundary(acts[-1], 3000)
                    summary_prompt = f"""Summarize the following radio drama act in 3-5 sentences.
Focus on: what happened, how each character's emotional state changed, what's at stake going into the next act, and any unresolved tensions.

ACT TEXT:
{_act_text_for_summary}

SUMMARY:"""
                    try:
                        summary = _generate_with_llm(
                            summary_prompt, model_id=model_id,
                            max_new_tokens=200, temperature=0.3,
                            optimization_profile=optimization_profile
                        )
                        act_summaries.append(summary)
                    except Exception:
                        act_summaries.append(_truncate_at_sentence_boundary(acts[-1], 1500))

                # ── Phase 3a: Chunked context hardening ──
                # Validate each summary — if too short, fall back to mechanical summary
                for s_idx in range(len(act_summaries)):
                    if len(act_summaries[s_idx].strip()) < 50:
                        log.warning("[ContextEng] Act %d summary too short (%d chars) — using mechanical fallback",
                                    s_idx + 1, len(act_summaries[s_idx]))
                        # Mechanical fallback: scene titles + last 8 dialogue lines
                        act_lines = acts[s_idx].strip().splitlines()
                        scene_titles = [l.strip() for l in act_lines if "===" in l]
                        dialogue_lines = [l.strip() for l in act_lines if "[VOICE:" in l][-8:]
                        act_summaries[s_idx] = (
                            "Scenes: " + "; ".join(scene_titles) + "\n"
                            "Key dialogue: " + " / ".join(dialogue_lines)
                        )[:800]

                # Build signposted context: all summaries + last 500 chars of raw text
                context_block = "STORY SO FAR (summaries of previous acts):\n"
                for s_idx, s_text in enumerate(act_summaries, 1):
                    context_block += f"  Act {s_idx}: {s_text.strip()}\n"

                # ── Phase 3b: Sentence-boundary tail (v1.4 Theme B) ──
                # Walks forward from the cut point to the next sentence start so
                # the Gemma prompt never sees a tail that begins mid-word.
                last_lines = _tail_at_sentence_boundary(acts[-1], 500)
                if len(acts[-1]) > 500:
                    last_lines = "... [truncated]\n" + last_lines
                context_block += f"\nLAST LINES (for dialogue continuity):\n{last_lines}"
            else:
                context_block = "(beginning of episode)"

            act_prompt = f"""{system}

OUTLINE:
{outline}

{context_block}

Now write ACT {act_num} of {num_acts} in full script format.
Target: ~{words_per_act} words for this act. Taut dialogue — fragments, interruptions, subtext.
{"This is the OPENING — start with [MUSIC: Opening theme] and ANNOUNCER setting time/place/characters. Then drop us IN MEDIAS RES." if act_num == 1 else ""}
{"This is the FINAL ACT — build to the twist, then ANNOUNCER delivers the hard-science epilogue. CITATION RULE: cite ONLY the real article provided in the news block above — its exact source name and date. NEVER use numbered references like [1], [2], article #N — always say the source name directly (e.g. 'According to Science Daily, published April 3, 2026...'). Do NOT invent ArXiv IDs or paper titles. End with [MUSIC: Closing theme]." if act_num == num_acts else ""}
{"Include an act break marker [ACT " + str(act_num + 1) + "] at the end of this act." if act_breaks and act_num < num_acts else ""}

CONTINUITY CHECK: Before writing, review the story-so-far summaries above. Ensure characters reference earlier events naturally. No amnesia — people remember what just happened to them.

Write Act {act_num} now:"""

            _runtime_log(f"ScriptWriter: Generating Act {act_num}/{num_acts}")
            
            # BUG-011 FIX: Reduce KV Cache allocation overhead for Act chunks.
            # 4096 pre-allocates too much VRAM on 4GB cards. Scale it to the act size.
            # 1 word ~ 1.5 tokens. We grant a 2x buffer (3.0 * words).
            if optimization_profile == "Obsidian (Low VRAM/Fast)":
                act_budget = min(2048, int(words_per_act * 3.0))
            else:
                act_budget = 4096
                
            act_text = _generate_with_llm(act_prompt, model_id=model_id,
                                              max_new_tokens=act_budget, temperature=temperature, top_p=top_p,
                                              optimization_profile=optimization_profile)
            acts.append(act_text)

        return "\n\n".join(acts)

    def _execute_arc_enhancer(self, script_text, genre, title, news_block, model_id, temperature, optimization_profile="Standard"):
        """Phase A-C: Paired opening + closing bookend rewrite for narrative coherence."""
        _runtime_log("ARC_EN_HANCER: Starting structural coherence pass")
        original_script_backup = script_text

        # Phase A: Extraction
        bookends = self._get_bookends(script_text)
        if not bookends:
            _runtime_log("ARC_ENHANCER: Failed to extract bookends — skipping pass")
            return script_text

        opening_orig, closing_orig = bookends

        # Phase A: Structural Coherence Scoring (Observability)
        arc_score, arc_checks = self._score_arc_coherence(opening_orig, closing_orig, script_text)
        checks_str = ", ".join(f"{k}={v}" for k, v in arc_checks.items())
        _runtime_log(f"ARC_ENHANCER: Arc score: {arc_score}/5 ({checks_str})")

        # Plot Spine Injection: extract middle-act summary so Phase B rewrite
        # honors the journey instead of hallucinating contradictions.
        plot_spine = self._extract_plot_spine(script_text, opening_orig, closing_orig)

        # v1.4 Theme B — surface the spine in the runtime log so the showrunner
        # can see exactly what the bookend rewriter was told about the middle.
        _runtime_log(f"ARC_ENHANCER: Plot spine: {plot_spine[:150]}")

        # Phase A score floor flag — if score < 3/5, the first Phase B pass will
        # be followed by one automatic retry. The retry threshold is the same
        # contract used by tests/vram_profile_test.py for the arc coherence check.
        _arc_retry_warranted = arc_score < 3

        # Phase B: Architectural Echo call
        # We use a lower temperature (0.6) for tighter structural alignment
        echo_prompt = f"""You are a structural script editor for the radio drama anthology "SIGNAL LOST".
YOUR TASK: Rewrite the OPENING and CLOSING dialogue blocks below to create a "narrative echo".

DIRECTIONS:
1. Plant a NARRATIVE SEED in the Opening Block. This can be a cryptic mention of an object, a specific fear, a recurring sound cue, or a foreshadowed choice.
2. Harvest the PAYOFF in the Closing Block. The seed MUST resolve, pivot, or be explained in a way that provides emotional or structural closure to the episode.
3. Preserve the CHARACTER NAMES and VOICES exactly as they appear in the original text.
4. Preserve all CANONICAL TAGS ([VOICE:], [SFX:], [ENV:], (beat)) exactly.
5. Do NOT change the meaning of the science headline context provided.
6. Do NOT contradict the MIDDLE EVENTS summary below. The closing must honor what happened in the middle of the story — no resurrected characters, no forgotten revelations, no reversed outcomes.
7. Return ONLY the rewritten blocks inside the XML tags below.

GENRE: {genre.replace("_", " ")}
TITLE: {title}
SCIENCE CONTEXT: {news_block[:500]}

MIDDLE EVENTS (do not contradict):
{plot_spine}

ORIGINAL OPENING BLOCK:
{opening_orig}

ORIGINAL CLOSING BLOCK:
{closing_orig}

Format your response exactly as:
<opening>
[Revised Opening Block]
</opening>
<closing>
[Revised Closing Block]
</closing>"""

        try:
            echo_response = _run_with_timeout(
                lambda: _generate_with_llm(
                    echo_prompt,
                    model_id=model_id,
                    max_new_tokens=1000,
                    temperature=0.6,
                    optimization_profile=optimization_profile
                ),
                timeout_sec=300,
                phase_label="Arc-Enhancer-Echo",
            )

            # Phase C: Injection + Echo Phrase Extraction
            try:
                opening_new = echo_response.split("<opening>")[1].split("</opening>")[0].strip()
                closing_new = echo_response.split("<closing>")[1].split("</closing>")[0].strip()

                if opening_new and closing_new:
                    # Extract echo phrase: find longest common noun between opening and closing rewrite
                    opening_nouns = set(re.findall(r'\b[A-Z][a-z]+\b', opening_new))
                    closing_nouns = set(re.findall(r'\b[A-Z][a-z]+\b', closing_new))
                    echo_phrase = list(opening_nouns & closing_nouns)[0] if (opening_nouns & closing_nouns) else "(no direct echo)"

                    # Safe replacement for opening
                    script_text = script_text.replace(opening_orig, opening_new, 1)

                    # Safe replacement for closing (work from the end to avoid collisions)
                    parts = script_text.rsplit(closing_orig, 1)
                    if len(parts) == 2:
                        script_text = parts[0] + closing_new + parts[1]

                    _runtime_log(f"ARC_ENHANCER: Pass 1 complete (echo phrase = {echo_phrase})")

                    # v1.4 Theme B — Phase A score floor retry.
                    # If the initial arc score was below the 3/5 floor, run a
                    # second Phase B+C pass using the already-injected script as
                    # the new base. One retry only — more would drift the text.
                    if _arc_retry_warranted:
                        _runtime_log(
                            f"ARC_ENHANCER: Score was {arc_score}/5 (below 3/5 floor) "
                            f"— triggering retry pass"
                        )
                        retry_bookends = self._get_bookends(script_text)
                        if retry_bookends:
                            opening_retry, closing_retry = retry_bookends
                            retry_spine = self._extract_plot_spine(
                                script_text, opening_retry, closing_retry
                            )
                            retry_prompt = echo_prompt.replace(
                                f"ORIGINAL OPENING BLOCK:\n{opening_orig}",
                                f"ORIGINAL OPENING BLOCK:\n{opening_retry}",
                            ).replace(
                                f"ORIGINAL CLOSING BLOCK:\n{closing_orig}",
                                f"ORIGINAL CLOSING BLOCK:\n{closing_retry}",
                            ).replace(
                                f"{plot_spine}",
                                f"{retry_spine}",
                            )
                            try:
                                retry_response = _run_with_timeout(
                                    lambda: _generate_with_llm(
                                        retry_prompt,
                                        model_id=model_id,
                                        max_new_tokens=1000,
                                        temperature=0.6,
                                        optimization_profile=optimization_profile
                                    ),
                                    timeout_sec=300,
                                    phase_label="Arc-Enhancer-Retry",
                                )
                                opening_r = retry_response.split("<opening>")[1].split("</opening>")[0].strip()
                                closing_r = retry_response.split("<closing>")[1].split("</closing>")[0].strip()
                                if opening_r and closing_r:
                                    script_text = script_text.replace(opening_retry, opening_r, 1)
                                    parts_r = script_text.rsplit(closing_retry, 1)
                                    if len(parts_r) == 2:
                                        script_text = parts_r[0] + closing_r + parts_r[1]
                                    # Re-score so the log tells the truth about the
                                    # final state, not just the initial state.
                                    retry_score, retry_checks = self._score_arc_coherence(
                                        opening_r, closing_r, script_text
                                    )
                                    retry_checks_str = ", ".join(
                                        f"{k}={v}" for k, v in retry_checks.items()
                                    )
                                    _runtime_log(
                                        f"ARC_ENHANCER: Pass 2 complete "
                                        f"arc_score={retry_score}/5 ({retry_checks_str})"
                                    )
                                else:
                                    _runtime_log("ARC_ENHANCER: Retry returned empty tags — keeping pass 1 result")
                            except Exception as retry_err:
                                log.warning("[ArcEnhancer] Retry pass failed: %s", retry_err)
                                _runtime_log(f"ARC_ENHANCER: Retry failed — keeping pass 1 result ({retry_err})")
                        else:
                            _runtime_log("ARC_ENHANCER: Retry skipped — could not re-extract bookends after pass 1")
                    else:
                        _runtime_log(
                            f"ARC_ENHANCER: Score {arc_score}/5 meets floor — no retry needed"
                        )
                else:
                    _runtime_log("ARC_ENHANCER: LLM returned empty tags — skipping injection")
            except (IndexError, ValueError):
                log.warning("[ArcEnhancer] Failed to parse XML tags from echo response")
                _runtime_log("ARC_ENHANCER: Parsing error — response format invalid")

        except Exception as e:
            log.warning("[ArcEnhancer] Phase B/C pass failed: %s", e)
            _runtime_log(f"ARC_ENHANCER: Failed — {e} (reverting to original)")
            # Revert to raw LLM output to prevent len(text)=1 crash
            return original_script_backup

        # v1.4 Theme B — automatic scene transition injection.
        # Runs regardless of how Phase B/C fared so even a failed arc pass
        # still gets the structural handoff benefit for downstream audio.
        try:
            script_text, _transition_count = _inject_scene_transitions(script_text)
            if _transition_count > 0:
                _runtime_log(
                    f"ARC_ENHANCER: Injected {_transition_count} scene transition "
                    f"cue(s) at weak handoffs"
                )
            else:
                _runtime_log("ARC_ENHANCER: No weak scene handoffs detected")
        except Exception as transition_err:
            log.warning("[ArcEnhancer] Scene transition injection failed: %s", transition_err)
            _runtime_log(f"ARC_ENHANCER: Transition injection failed — {transition_err}")

        return script_text

    def _score_arc_coherence(self, opening_text, closing_text, script_text):
        """Phase A: Structural coherence check — 5-point scoring for narrative completeness."""
        score = 0
        checks = {}

        # Check 1: Truncation detector — does closing end mid-sentence?
        closing_lines = closing_text.strip().split('\n')
        last_line = closing_lines[-1].strip() if closing_lines else ""
        terminal_chars = {'.', '!', '?', '"'}
        # Pass if last line ends with terminal char AND not a connective word
        last_word = last_line.split()[-1].rstrip('.,!?;:"') if last_line.split() else ""
        connective_words = {'the', 'and', 'to', 'a', 'an', 'or', 'but', 'as', 'is', 'of', 'in', 'be'}
        checks['truncation'] = (bool(last_line) and
                                any(last_line.endswith(c) for c in terminal_chars) and
                                last_word.lower() not in connective_words)
        if checks['truncation']:
            score += 1

        # Check 2: Weak final scene — count [VOICE:] tags, need ≥4 lines
        voice_count = len(re.findall(r'\[VOICE:', closing_text, re.IGNORECASE))
        checks['strong_scene'] = voice_count >= 4
        if checks['strong_scene']:
            score += 1

        # Check 3: Premise payoff — any capitalized keyword overlap (opening → closing)
        opening_caps = set(re.findall(r'\b[A-Z][a-z]+\b', opening_text))
        closing_caps = set(re.findall(r'\b[A-Z][a-z]+\b', closing_text))
        checks['payoff'] = len(opening_caps & closing_caps) > 0
        if checks['payoff']:
            score += 1

        # Check 4: Tonal echo — repeated words (>4 chars) between opening and closing
        opening_words = set(w.lower() for w in re.findall(r'\b\w{4,}\b', opening_text))
        closing_words = set(w.lower() for w in re.findall(r'\b\w{4,}\b', closing_text))
        checks['echo'] = len(opening_words & closing_words) >= 2
        if checks['echo']:
            score += 1

        # Check 5: Epilogue presence — ANNOUNCER in final 500 chars
        epilogue_region = script_text[-500:] if len(script_text) > 500 else script_text
        checks['epilogue'] = 'ANNOUNCER' in epilogue_region
        if checks['epilogue']:
            score += 1

        return score, checks

    def _extract_plot_spine(self, script_text, opening_orig, closing_orig):
        """Extract a ~50-word middle-act summary so Phase B rewrites honor continuity.

        Pulls dialogue and scene headers from the region BETWEEN the opening and
        closing blocks, then truncates to ~50 words. This gives the Phase B rewriter
        knowledge of the middle acts without bloating the token budget.
        """
        # Find the middle region (everything between opening and closing)
        open_end = script_text.find(opening_orig)
        close_start = script_text.rfind(closing_orig)

        if open_end == -1 or close_start == -1 or close_start <= open_end:
            return "(middle events unavailable)"

        middle_region = script_text[open_end + len(opening_orig):close_start]

        # Extract scene markers and voice lines, strip formatting for a clean spine
        spine_parts = []
        for raw_line in middle_region.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            # Keep scene markers as structural anchors
            scene_match = re.match(r'===\s*SCENE\s+(\d+)\s*===', line, re.IGNORECASE)
            if scene_match:
                spine_parts.append(f"[Scene {scene_match.group(1)}]")
                continue
            # Extract dialogue content from voice tags
            voice_match = re.match(r'\[VOICE:\s*([^,\]]+)[^\]]*\]\s*(.+)$', line, re.IGNORECASE)
            if voice_match:
                speaker = voice_match.group(1).strip()
                dialogue = voice_match.group(2).strip()
                spine_parts.append(f"{speaker}: {dialogue}")

        # Truncate to ~50 words to keep Phase B prompt lean (~60 tokens)
        full_spine = " ".join(spine_parts)
        words = full_spine.split()
        if len(words) > 50:
            full_spine = " ".join(words[:50]) + "..."

        return full_spine if full_spine else "(middle events unavailable)"

    def _get_bookends(self, script_text):
        """Extract opening and closing dialogue blocks for the coherence pass."""
        # --- 1. OPENING BLOCK ---
        # Find Scene 1
        scene1_match = re.search(r'===\s*SCENE\s+1\s*===', script_text, re.IGNORECASE)
        if not scene1_match:
            return None

        # Focus on the first ~25 lines of Scene 1 to find dialogue
        body_start = script_text[scene1_match.end():]
        # Find all Voice tags in the first 4000 chars of Scene 1
        voices = list(re.finditer(r'\[VOICE:', body_start[:4000], re.IGNORECASE))
        if len(voices) < 4:
            return None

        # Opening block: from first voice to end of 8th voice (or last available)
        v_count = min(len(voices), 8)
        target_v = voices[v_count - 1]
        line_end = body_start.find("\n", target_v.end())
        if line_end == -1: line_end = len(body_start)
        opening_block = body_start[:line_end].strip()

        # --- 2. CLOSING BLOCK ---
        # Find the last scene (climax)
        # We look for the last SCENE marker before the EPILOGUE or Closing Music
        end_marker = re.search(r'===\s*EPILOGUE\s*===|\[MUSIC:\s*Closing theme\]', script_text, re.IGNORECASE)
        climax_boundary = end_marker.start() if end_marker else len(script_text)

        climax_area = script_text[:climax_boundary]
        scenes = list(re.finditer(r'===\s*SCENE\s+\d+\s*===', climax_area, re.IGNORECASE))
        if not scenes:
            return None

        last_scene_body = climax_area[scenes[-1].end():]
        # Find voice tags in the last scene
        last_voices = list(re.finditer(r'\[VOICE:', last_scene_body, re.IGNORECASE))
        if len(last_voices) < 3:
            return None

        # Closing block: pull at most the last 8 dialogue lines
        v_count_climax = min(len(last_voices), 8)
        start_idx = last_voices[-v_count_climax].start()
        closing_block = last_scene_body[start_idx:].strip()

        # Sanity check: ensure these blocks actually exist in the text (for later replace)
        if opening_block in script_text and closing_block in script_text:
            return opening_block, closing_block

        return None

    # Descriptor words that indicate a missing character name (Gemma dropped the NAME field)
    _GENDER_WORDS = frozenset([
        "male", "female", "man", "woman", "boy", "girl", "nonbinary",
        "young", "old", "older", "elderly", "middle", "teen",
    ])

    def _parse_script(self, text):
        """Parse raw script text into structured Canonical Audio Tokens.

        Robust against the most common Gemma formatting failure: omitting the
        character NAME as the first field in [VOICE:] tags, producing malformed
        tags like [VOICE: male, 40s, calm] that would silently create "MALE" as
        a character. Those are caught, logged, and assigned a positional fallback
        name (CHAR_A, CHAR_B, ...) so Bark still produces audio.
        """
        lines = []
        _fallback_counter = 0   # incremented in the for-loop below for CHAR_A / CHAR_B fallback names

        # OTR Canonical 1.0 RegEx Patterns
        # BUG-009 fix: accept both `=== SCENE N ===` and `=== SCENE N ***` (Gemma
        # occasionally uses asterisks as the closing delimiter, which silently
        # broke scene splitting and merged Act 3 into Act 2's last scene).
        scene_pat = re.compile(r'^===\s*SCENE\s+(.+?)\s*(?:===|\*\*\*)', re.IGNORECASE)
        env_pat   = re.compile(r'^\[ENV:\s*(.+?)\]',          re.IGNORECASE)
        sfx_pat   = re.compile(r'^\[SFX:\s*(.+?)\]',          re.IGNORECASE)
        beat_pat  = re.compile(r'^\(beat\)$', re.IGNORECASE)

        # Voice patterns — ordered from most to least specific:
        # v1 (canonical): [VOICE: NAME, traits] dialogue on same line
        voice_inline_pat = re.compile(r'^\[VOICE:\s*(.+?),\s*(.+?)\]\s*(.+)$', re.IGNORECASE)
        # v2 (no-traits): [VOICE: NAME] dialogue on same line
        voice_notrait_pat = re.compile(r'^\[VOICE:\s*([A-Z][A-Z0-9_ ]+?)\]\s*(.+)$', re.IGNORECASE)
        # v3 (tag only): [VOICE: NAME, traits] with dialogue on NEXT line
        voice_tagonly_pat = re.compile(r'^\[VOICE:\s*(.+?)(?:,\s*(.+?))?\]\s*$', re.IGNORECASE)
        # v4 (shorthand): [ANNOUNCER, traits] or [ANNOUNCER] as a standalone tag (Mistral Nemo style)
        voice_shorthand_pat = re.compile(r'^\[([A-Z][A-Z0-9_ ]{1,20})(?:,\s*(.+?))?\]\s*$', re.IGNORECASE)

        raw_lines = text.strip().splitlines()
        i = 0
        while i < len(raw_lines):
            raw_line = raw_lines[i]
            s = raw_line.strip()
            # v1.4 Markdown Bolding Hallucination Fix:
            # Gemma 2B often generates **[VOICE:...]** or **=== SCENE ===**
            # Strip outer asterisks before matching tags.
            s = re.sub(r'^\*+|(?<=\])\*+$', '', s).strip()
            # Also strip italic markers flanking a tag: *[VOICE:..]* or _[VOICE:..]_
            s = re.sub(r'^[_*]+|[_*]+$', '', s).strip()

            if not s:
                i += 1
                continue

            # v1.4 Theme B — Timeout fallback sentinel path.
            if s.startswith("[SYSTEM_SENTINEL:"):
                i += 1
                continue

            m = scene_pat.match(s)
            if m:
                lines.append({"type": "scene_break", "scene": m.group(1)})
                i += 1
                continue

            m = env_pat.match(s)
            if m:
                lines.append({"type": "environment", "description": m.group(1)})
                i += 1
                continue

            m = sfx_pat.match(s)
            if m:
                lines.append({"type": "sfx", "description": m.group(1)})
                i += 1
                continue

            m = beat_pat.match(s)
            if m:
                lines.append({"type": "pause", "kind": "beat", "duration_ms": 200})
                i += 1
                continue

            # ── VOICE TAG MATCHING (4 variants) ──────────────────────────

            # v1: [VOICE: NAME, traits] dialogue — inline
            m = voice_inline_pat.match(s)
            if m:
                raw_name     = m.group(1).strip()
                voice_traits = m.group(2).strip()
                dialogue     = m.group(3).strip().strip('"\u201c\u201d*_')
                if raw_name.lower() in self._GENDER_WORDS:
                    _fallback_counter += 1
                    fallback_name = f"CHAR_{chr(64 + _fallback_counter)}"
                    log.warning("[ScriptParser] Malformed VOICE tag — name field is a descriptor word '%s'. Assigning fallback '%s'.", raw_name, fallback_name)
                    voice_traits = f"{raw_name}, {voice_traits}"
                    character_name = fallback_name
                else:
                    character_name = raw_name.upper()
                lines.append({"type": "dialogue", "character_name": character_name, "voice_traits": voice_traits, "line": dialogue})
                i += 1
                continue

            # v2: [VOICE: NAME] dialogue — no traits, inline
            m = voice_notrait_pat.match(s)
            if m:
                character_name = m.group(1).strip().upper()
                dialogue = m.group(2).strip().strip('"\u201c\u201d*_')
                lines.append({"type": "dialogue", "character_name": character_name, "voice_traits": "", "line": dialogue})
                i += 1
                continue

            # v3: [VOICE: NAME, traits] tag-only — look ahead for dialogue on NEXT line
            m = voice_tagonly_pat.match(s)
            if m:
                raw_name     = m.group(1).strip()
                voice_traits = (m.group(2) or "").strip()
                # Skip non-VOICE bracket tags that could match (e.g. [MUSIC:...])
                # Only handle if the raw_name looks like a real character name (uppercase letters)
                if re.match(r'^[A-Z][A-Z0-9_ ]*$', raw_name, re.IGNORECASE) and raw_name.upper() not in (
                    "MUSIC", "SFX", "ENV", "BEAT", "PAUSE", "SYSTEM_SENTINEL"
                ):
                    # Peek at next non-empty line for dialogue
                    j = i + 1
                    while j < len(raw_lines) and not raw_lines[j].strip():
                        j += 1
                    next_s = raw_lines[j].strip() if j < len(raw_lines) else ""
                    next_s_clean = re.sub(r'^[*_]+|[*_]+$', '', next_s).strip()
                    # Accept as dialogue if next line is NOT a tag and not empty
                    if next_s_clean and not next_s_clean.startswith('[') and not next_s_clean.startswith('='):
                        dialogue = next_s_clean.strip('"\u201c\u201d*_')
                        if raw_name.lower() in self._GENDER_WORDS:
                            _fallback_counter += 1
                            character_name = f"CHAR_{chr(64 + _fallback_counter)}"
                        else:
                            character_name = raw_name.upper()
                        lines.append({"type": "dialogue", "character_name": character_name, "voice_traits": voice_traits, "line": dialogue})
                        i = j + 1  # consume both tag line and dialogue line
                        continue
                    # else: fall through as direction

            # v4: [CHARACTER, traits] shorthand (e.g. [ANNOUNCER, female, 50s, calm])
            # Used by Mistral Nemo when it omits the VOICE: prefix
            m = voice_shorthand_pat.match(s)
            if m:
                raw_name     = m.group(1).strip()
                voice_traits = (m.group(2) or "").strip()
                # Must look like a character name (not a known structural tag)
                upper_name = raw_name.upper()
                if upper_name not in ("ENV", "SFX", "MUSIC", "BEAT", "PAUSE", "ACT", "SCENE"):
                    j = i + 1
                    while j < len(raw_lines) and not raw_lines[j].strip():
                        j += 1
                    next_s = raw_lines[j].strip() if j < len(raw_lines) else ""
                    next_s_clean = re.sub(r'^[*_]+|[*_]+$', '', next_s).strip()
                    if next_s_clean and not next_s_clean.startswith('[') and not next_s_clean.startswith('='):
                        dialogue = next_s_clean.strip('"\u201c\u201d*_')
                        lines.append({"type": "dialogue", "character_name": upper_name, "voice_traits": voice_traits, "line": dialogue})
                        i = j + 1
                        continue

            # Fallback: treat as structural direction
            if s and not s.startswith("#") and not s.startswith("---"):
                lines.append({"type": "direction", "text": s})
            i += 1

        malformed = _fallback_counter
        if malformed:
            log.warning(
                "[ScriptParser] %d malformed VOICE tag(s) detected (missing character name). "
                "Update SCRIPT_SYSTEM_PROMPT Section 1 example if this recurs.", malformed
            )

        # BUG-010 fix: hard-abort if extraction produced an empty / no-dialogue
        # script. Previously this silently passed ghost data into SceneSequencer
        # which then crashed Bark / video assembly with cryptic errors.
        dialogue_count = sum(1 for ln in lines if ln.get("type") == "dialogue")
        
        # v1.4 Theme B - Failsafe for 2B models that strip [VOICE:] tags
        # If no dialogue was found but we see `NAME: dialogue` structure, attempt recovery
        if dialogue_count == 0 and len(lines) > 0:
            log.warning("[ScriptParser] Zero standard tags found. Attempting permissive 2B-fallback parse...")
            _recovered = 0
            for ln in lines:
                if ln.get("type") == "direction":
                    text = ln["text"]
                    # Match 'NAME: dialogue' or '**NAME:** dialogue' or 'NAME (angry): dialogue'
                    m = re.match(r'^(?:\*\*)?([A-Z\s]{2,20})(?:\*\*)?(?:\s*\([^)]*\))?\s*:\s*(.+)$', text)
                    if m:
                        ln["type"] = "dialogue"
                        ln["character_name"] = m.group(1).strip()
                        ln["voice_traits"] = "unspecified"
                        ln["line"] = m.group(2).strip()
                        _recovered += 1
            if _recovered > 0:
                log.info(f"[ScriptParser] Permissive fallback recovered {_recovered} dialogue lines!")
                dialogue_count = _recovered

        if not lines or dialogue_count == 0:
            log.critical(
                "[ScriptParser] FATAL: parsed %d structural lines but %d dialogue lines. "
                "Script extraction failed — refusing to pass empty data downstream.",
                len(lines), dialogue_count,
            )
            _runtime_log(
                f"PARSE_FATAL: lines={len(lines)} dialogue={dialogue_count} "
                f"raw_text_len={len(text)} — aborting"
            )
            with open("FAILED_SCRIPT_DUMP.txt", "w", encoding="utf-8") as f:
                f.write(text)
            raise ValueError(
                f"Script parser produced 0 dialogue lines from {len(text)}-char input. "
                "Aborting run to prevent silent audio failure."
            )

        # ── PRO QA: Enforce ANNOUNCER bookends ──
        # Guarantees the Announcer always opens and closes the show, even if the LLM forgets.
        # Only applies to full scripts (heuristically > 5 lines) to avoid polluting unit tests.
        dialogue_indices = [i for i, ln in enumerate(lines) if ln.get("type") == "dialogue"]
        if len(dialogue_indices) > 5:
            first_idx = dialogue_indices[0]
            last_idx = dialogue_indices[-1]
            
            if lines[first_idx]["character_name"] != "ANNOUNCER":
                log.warning("[ScriptParser] PRO QA: Missing ANNOUNCER opening. Auto-injecting.")
                _runtime_log("QA_REPAIR: Missing ANNOUNCER opening auto-injected")
                lines.insert(first_idx, {
                    "type": "dialogue",
                    "character_name": "ANNOUNCER",
                    "voice_traits": "male, 50s, authoritative, calm",
                    "line": "Welcome to Signal Lost. Tonight's broadcast takes us into the unknown."
                })
                # Re-evaluate indices
                dialogue_indices = [i for i, ln in enumerate(lines) if ln.get("type") == "dialogue"]
                last_idx = dialogue_indices[-1]

            if lines[last_idx]["character_name"] != "ANNOUNCER":
                log.warning("[ScriptParser] PRO QA: Missing ANNOUNCER closing. Auto-injecting.")
                _runtime_log("QA_REPAIR: Missing ANNOUNCER closing auto-injected")
                lines.insert(last_idx + 1, {
                    "type": "dialogue",
                    "character_name": "ANNOUNCER",
                    "voice_traits": "male, 50s, authoritative, calm",
                    "line": "And so the transmission ends. This has been Signal Lost. Stay safe."
                })

        return lines


# ─────────────────────────────────────────────────────────────────────────────
# NODE 2: DIRECTOR
# ─────────────────────────────────────────────────────────────────────────────

DIRECTOR_PROMPT = """You are the PRODUCTION DIRECTOR for the Canonical Audio Engine 1.0.
Your task is to take a raw script and compile it into a deterministic JSON production plan.

═══ 🧱 1. SCRIPT STRUCTURE (CANONICAL 1.0) ═══
The script follows these tokens:
- === SCENE X ===
- [ENV: description]
- [SFX: description]
- [VOICE: NAME, gender, age, tone, energy] Dialogue...
- (beat)

═══ 🧱 2. VOICE MAPPING RULES ═══
{voice_mapping_rules}

═══ 🧱 3. OUTPUT FORMAT (STRICT JSON) ═══
{{
  "episode_title": "...",
  "voice_assignments": {{
    "ANNOUNCER": {{
      "voice_preset": "v2/en_speaker_4",
      "notes": "Female, energetic, authoritative"
    }},
    "CHARACTER_A": {{
      "voice_preset": "v2/en_speaker_1",
      "notes": "Male, calm, 40s"
    }}
  }},
  "sfx_plan": [
    {{
      "cue_id": "sfx_001",
      "type": "sfx",
      "description": "Distant thunder rolling behind heavy rain",
      "generation_prompt": "Low rumble of distant thunder, heavy rain pattering on a tin roof, outdoor perspective, cinematic sound design"
    }}
  ],
  "music_plan": [
    {{
      "cue_id": "opening",
      "duration_sec": 12,
      "generation_prompt": "1940s old time radio opening theme, warm brass fanfare, upright bass, snare brushes, mono AM radio character, tube saturation, confident and mysterious, ends on a held chord"
    }},
    {{
      "cue_id": "closing",
      "duration_sec": 8,
      "generation_prompt": "1940s old time radio closing sting, brass and strings, resolving cadence, warm tube saturation, fades to silence"
    }},
    {{
      "cue_id": "interstitial",
      "duration_sec": 4,
      "generation_prompt": "short old time radio act-break stinger, single brass hit with cymbal swell, mono, tube warmth"
    }}
  ],
  "pacing": {{
    "beat_pause_ms": 100
  }}
}}

═══ 🔊 SFX PLAN RULES ═══
- Scan all [SFX:] tags in the script. Create one dictionary entry per tag in the sfx_plan list.
- Keep the `description` brief (for manual reference).
- The `generation_prompt` is for an AI Foley engine. Be highly descriptive about the textures, the environment, and the distance.
- Examples: 
  - "Footsteps crunching on dry autumn leaves, slow and deliberate, close-up perspective"
  - "A futuristic sliding door swoosh followed by a metallic latching sound"
  - "Old wooden floorboards creaking under weight in a silent room"
- Match the SFX to the story's setting (noir, sci-fi, etc.).
- Keep prompts under 25 words. Do NOT mention music or voices in SFX prompts.

═══ 🎵 MUSIC PLAN RULES ═══
- ALWAYS include exactly three music cues: opening, closing, interstitial. Cue ids are fixed strings.
- Tailor each generation_prompt to the TONE of THIS episode. A noir thriller gets minor-key brass and upright bass. A comedic episode gets brighter brass, a wink of pizzicato strings, a lighter tempo. A cosmic horror piece gets low drones and distant timpani. Match the story.
- Keep every prompt musically specific: name instruments, tempo feel, mood, era, and a recording character (mono AM radio, tube saturation). No generic "scary music" or "happy music."
- Keep prompts under 35 words each.
- duration_sec is fixed: opening=12, closing=8, interstitial=4. Do not change these numbers.
- The music model is instrumental-only. Never ask for vocals, lyrics, or singing.

CRITICAL RULES:
- Output ONLY the JSON block. No prose, no commentary, no markdown explanation.
- Do NOT copy, summarize, paraphrase, or include ANY dialogue from the script.
- Do NOT add a "script" or "dialogue" or "scenes" key to your JSON.
- Your ONLY job: extract character names, assign placeholder presets, list SFX cues, set pacing.
- Keep the JSON as MINIMAL as possible. Short notes, short descriptions.
- The procedural engine handles all voice casting — your presets are placeholders only.

SCRIPT:
{script_text}
"""


BARK_VOICE_RULES = """- Scan all [VOICE:] tags in the script. The FIRST FIELD (before the first comma) is the CHARACTER NAME.
- Collect every unique CHARACTER NAME. Map each to one UNIQUE voice preset.
- NOTE: Character names, voice presets, accents, and traits are PROCEDURALLY OVERRIDDEN after your JSON is generated.
  You only need to provide reasonable en_speaker_* placeholder presets so the JSON structure is valid.
  The procedural engine handles everything else. LEMMY always stays LEMMY with v2/en_speaker_8.
- The JSON key MUST be the CHARACTER NAME EXACTLY AS IT APPEARS (all caps, no descriptors).
- Use any en_speaker_* preset as a placeholder:
  v2/en_speaker_0 = Male, authoritative, deep
  v2/en_speaker_1 = Male, mid-range
  v2/en_speaker_2 = Female, neutral
  v2/en_speaker_3 = Male, younger
  v2/en_speaker_4 = Female, warmer
  v2/en_speaker_5 = Male, older
  v2/en_speaker_6 = Male, character voice
  v2/en_speaker_7 = Female, higher pitch
  v2/en_speaker_8 = Male, gravelly/raspy (reserved for LEMMY)
  v2/en_speaker_9 = Female, authoritative
- Each character gets ONE preset. No duplicates.
- LEMMY always gets v2/en_speaker_8."""

KOKORO_VOICE_RULES = """- Scan all [VOICE:] tags in the script. The FIRST FIELD (before the first comma) is the CHARACTER NAME.
- Collect every unique CHARACTER NAME. Map each to one UNIQUE voice preset from the Kokoro valid voices below.
- NOTE: Character names and traits are procedurally overridden, but the voice preset you choose WILL be used directly by the Kokoro engine.
  LEMMY always stays LEMMY with am_michael.
- The JSON key MUST be the CHARACTER NAME EXACTLY AS IT APPEARS (all caps, no descriptors).
- Use ONLY these exact valid Kokoro presets (1 per character, no duplicates):
  af_bella = Female, energetic
  af_sky = Female, neutral
  af_nicole = Female, whispery
  am_adam = Male, younger
  am_onyx = Male, deep
  am_michael = Male, older/authoritative (reserved for LEMMY)
- Each character gets ONE preset. No duplicates.
- LEMMY always gets am_michael."""


class LLMDirector:
    """Takes a script and generates a full production plan via LLM."""

    CATEGORY = "OldTimeRadio"
    FUNCTION = "direct"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("production_plan_json", "voice_map_json", "sfx_plan_json", "music_plan_json")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "script_text": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "temperature": ("FLOAT", {
                    "default": 0.4, "min": 0.1, "max": 1.0, "step": 0.05,
                    "tooltip": "Lower = more consistent JSON output"
                }),
                "tts_engine": (["bark (standard 8GB)", "kokoro (obsidian 4GB)"], {
                    "default": "bark (standard 8GB)",
                    "tooltip": "Bark for generative voices. Kokoro for low-VRAM neural voices."
                }),
                "vintage_intensity": (["subtle", "moderate", "heavy", "extreme"], {
                    "default": "subtle",
                    "tooltip": "How vintage/degraded should the final audio sound"
                }),
                # v1.4 Theme C — optional series bible, socket input only.
                "project_state": ("PROJECT_STATE", {
                    "tooltip": "Optional: Project State Loader output. When wired, series bible preamble is injected into the director prompt."
                }),
                "optimization_profile": (["Pro (Ultra Quality)", "Standard", "Obsidian (UNSTABLE/4GB)"], {
                    "default": "Standard",
                    "tooltip": "Consistency widget. Obsidian is unstable; on 4GB cards, ensure 'kokoro' is used for TTS below."
                }),
            },
        }

    def direct(self, script_text, temperature=0.4, tts_engine="bark (standard 8GB)", vintage_intensity="subtle",
               project_state=None, optimization_profile="Standard"):
        # ── MASTER SWITCH INHERITANCE ──
        # Inherently use the chosen model from ScriptWriter.
        global _CURRENT_LLM_MODEL
        model_id = _CURRENT_LLM_MODEL

        # Defer Bark health check until AFTER the script is written,
        # preventing Bark from hogging VRAM while Gemma writes the script.
        try:
            _bark_health_check()
        except Exception as e:
            _runtime_log(f"VOICE_HEALTH_SKIPPED: unexpected error {e}")

        # v1.4 Theme C — resolve series bible (read-only).
        try:
            if project_state is None:
                _director_state = ProjectState.load()
            else:
                _director_state = ProjectState.from_dict(project_state)
            _director_preamble = _director_state.prompt_preamble()
        except Exception as e:
            _runtime_log(f"Director: project_state load failed, continuing without preamble: {e}")
            _director_preamble = ""
        _runtime_log(f"Director: project_state_preamble_chars={len(_director_preamble)}")

        # v1.4 Theme C — director entry snapshot + peak reset.
        vram_reset_peak("director_entry")
        vram_snapshot("director_entry")

        if "kokoro" in tts_engine.lower():
            vrules = KOKORO_VOICE_RULES
        else:
            vrules = BARK_VOICE_RULES

        prompt = DIRECTOR_PROMPT.format(
            script_text=script_text[:6000],
            voice_mapping_rules=vrules
        )
        if _director_preamble:
            prompt = f"[SERIES BIBLE]\n{_director_preamble}\n\n{prompt}"

        vintage_map = {
            "subtle":   {"radio_static_amount": 0.05, "vinyl_crackle": 0.03, "tube_warmth": 0.4, "frequency_rolloff_hz": 8000, "hum_60hz": 0.02},
            "moderate": {"radio_static_amount": 0.15, "vinyl_crackle": 0.10, "tube_warmth": 0.7, "frequency_rolloff_hz": 6000, "hum_60hz": 0.05},
            "heavy":    {"radio_static_amount": 0.25, "vinyl_crackle": 0.20, "tube_warmth": 0.9, "frequency_rolloff_hz": 4500, "hum_60hz": 0.08},
            "extreme":  {"radio_static_amount": 0.40, "vinyl_crackle": 0.35, "tube_warmth": 1.0, "frequency_rolloff_hz": 3500, "hum_60hz": 0.12},
        }

        log.info(f"[LLMDirector] Generating production plan (vintage={vintage_intensity})")

        # Scale max_new_tokens to script length.
        # Director output: voice_assignments (placeholder presets, procedurally
        # overridden), sfx_plan, music_plan (3 fixed cues), pacing. No dialogue
        # duplication. A 5-character cast + 10 SFX cues + 3 music cues = ~600–800 tokens.
        # Budget: ~1 token per 10 chars of script (for SFX scanning) + 550 base.
        script_len = len(script_text)
        max_tokens = min(1700, max(650, 550 + script_len // 10))
        log.info(f"[LLMDirector] max_new_tokens={max_tokens} (script={script_len} chars)")

        raw = _generate_with_llm(
            prompt,
            model_id=model_id,
            max_new_tokens=max_tokens,
            temperature=temperature,
            optimization_profile=optimization_profile
        )

        # Extract JSON from response
        plan = self._extract_json(raw)

        # Procedural character names (except LEMMY stays LEMMY)
        # Use a deterministic seed based on script content hash
        if plan:
            import hashlib
            script_hash = hashlib.sha256(script_text.encode()).hexdigest()[:16]

            # BUG-004 fix: extract gender from each [VOICE: NAME, gender, ...] tag
            # so the procedural cast generator never assigns a male voice to a
            # female character or vice versa. First gender hint per name wins.
            gender_map = {}
            voice_tag_re = re.compile(
                r"\[VOICE:\s*([A-Z][A-Z0-9_ ]*?)\s*,\s*(male|female)\b",
                re.IGNORECASE,
            )
            for m in voice_tag_re.finditer(script_text):
                name_key = m.group(1).strip().upper()
                gender_val = m.group(2).strip().lower()
                gender_map.setdefault(name_key, gender_val)
            log.info("[LLMDirector] Parsed %d gender hints from script: %s",
                     len(gender_map), gender_map)

            plan = self._randomize_character_names(plan, script_hash, gender_map=gender_map)

        # Override vintage settings with user's intensity choice
        if plan:
            plan["vintage_settings"] = vintage_map.get(vintage_intensity, vintage_map["moderate"])

        plan_json = json.dumps(plan, indent=2)
        voice_json = json.dumps(plan.get("voice_assignments", {}), indent=2)
        sfx_json = json.dumps(plan.get("sfx_plan", []), indent=2)
        music_json = json.dumps(plan.get("music_plan", []), indent=2)

        log.info(f"[LLMDirector] Plan: {len(plan.get('voice_assignments', {}))} voices, "
                 f"{len(plan.get('sfx_plan', []))} SFX cues, "
                 f"{len(plan.get('music_plan', []))} music cues")

        # BUG-012 FIX: Explicitly unload Gemma from VRAM at the end of the director phase.
        # Otherwise it stays resident during the audio generation phases, causing VRAM OOM
        # or massive PCIe swapping on 4GB hardware.
        _unload_llm()
        _runtime_log("Director: Gemma unloaded — VRAM freed for Audio/TTS")

        # v1.4 Theme C — director exit snapshot.
        vram_snapshot("director_exit")

        return (plan_json, voice_json, sfx_json, music_json)

    def _extract_json(self, text):
        """Extract JSON object from LLM output (handles markdown fences, truncation)."""
        log.info(f"[LLMDirector] Raw output length: {len(text)} chars")
        log.info(f"[LLMDirector] Raw output preview: {text[:200]}...")

        # Try to find JSON block
        patterns = [
            re.compile(r'```json\s*\n(.*?)\n```', re.DOTALL),
            re.compile(r'```\s*\n(.*?)\n```', re.DOTALL),
            re.compile(r'(\{.*\})', re.DOTALL),
        ]
        for pat in patterns:
            m = pat.search(text)
            if m:
                try:
                    return json.loads(m.group(1))
                except json.JSONDecodeError:
                    # Try to repair: strip trailing commas, close unclosed braces
                    candidate = m.group(1)
                    candidate = re.sub(r',\s*([}\]])', r'\1', candidate)  # trailing commas
                    # If JSON was truncated, try closing it
                    open_braces = candidate.count('{') - candidate.count('}')
                    open_brackets = candidate.count('[') - candidate.count(']')
                    if open_braces > 0 or open_brackets > 0:
                        log.info(f"[LLMDirector] Attempting JSON repair: +{open_braces} braces, +{open_brackets} brackets")
                        candidate += ']' * open_brackets + '}' * open_braces
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError as e:
                            log.warning(f"[LLMDirector] JSON repair failed: {e}")
                    continue

        # Last resort: find the first { and try to build valid JSON from there
        brace_start = text.find('{')
        if brace_start >= 0:
            candidate = text[brace_start:]
            # Try progressively shorter substrings
            for end_offset in range(len(candidate), max(0, len(candidate) - 200), -1):
                try:
                    return json.loads(candidate[:end_offset])
                except json.JSONDecodeError:
                    continue

        log.critical("[LLMDirector] FATAL: Could not parse JSON from model output.")
        log.critical(f"[LLMDirector] Full raw output:\n{text[:1000]}")
        raise ValueError("Failed to parse production plan JSON. Aborting run to prevent silent audio failure.")

    def _randomize_character_names(self, plan: dict, episode_seed: str,
                                   gender_map: dict = None) -> dict:
        """Replace ALL character traits with procedural profiles. LEMMY stays LEMMY.

        For each character in voice_assignments:
          - LEMMY: Fixed profile (gravelly male, en_speaker_8)
          - ANNOUNCER: Random voice from balanced announcer pool
          - Everyone else: Full procedural profile — name, gender, age,
            demeanor, and best-fit en_speaker_* preset, all derived
            deterministically from the episode seed.

        The Director's original name/voice picks are fully overridden.
        English-only presets enforced at generation time.

        Args:
            plan: The parsed production plan dict
            episode_seed: A deterministic seed (script hash) for reproducibility

        Returns:
            Modified plan with procedural names, traits, and voice presets
        """
        if not plan or "voice_assignments" not in plan:
            return plan

        voice_assignments = plan.get("voice_assignments", {})
        if not voice_assignments:
            return plan

        # Track assigned presets to avoid duplicates across cast
        used_presets = set()
        new_voice_assignments = {}
        character_idx = 0

        # FIX: Process LEMMY and ANNOUNCER FIRST so their locked presets
        # are reserved in used_presets before regular characters draw from
        # the pool. Otherwise a regular char can grab v2/en_speaker_8
        # before Lemmy's branch runs, causing voice collision (Lemmy=Drake).
        all_keys = list(voice_assignments.keys())
        priority_keys = [k for k in all_keys if k.upper().strip() in ("LEMMY", "ANNOUNCER")]
        regular_keys  = [k for k in all_keys if k.upper().strip() not in ("LEMMY", "ANNOUNCER")]
        ordered_keys  = priority_keys + regular_keys

        for old_name in ordered_keys:
            upper_name = old_name.upper().strip()

            if upper_name == "LEMMY":
                # LEMMY — fixed iconic profile, never changes
                profile = _LEMMY_PROFILE.copy()
                new_voice_assignments["LEMMY"] = {
                    "voice_preset": profile["voice_preset"],
                    "notes": profile["notes"],
                }
                used_presets.add(profile["voice_preset"])
                log.info("[LLMDirector] LEMMY: locked → %s (%s)",
                         profile["voice_preset"], profile["notes"])

            elif upper_name == "ANNOUNCER":
                # ANNOUNCER — random from balanced pool, seeded per episode.
                # Respects gender_hint from script [VOICE: ANNOUNCER, gender, ...] tag.
                ann_gender = gender_map.get("ANNOUNCER") if gender_map else None
                ann = _generate_announcer_profile(episode_seed, gender_hint=ann_gender)
                new_voice_assignments["ANNOUNCER"] = {
                    "voice_preset": ann["voice_preset"],
                    "notes": ann["notes"],
                }
                used_presets.add(ann["voice_preset"])
                log.info("[LLMDirector] ANNOUNCER: procedural → %s (%s) [gender_hint=%s]",
                         ann["voice_preset"], ann["notes"], ann_gender or "none")

            else:
                # Regular character — full procedural profile.
                # BUG-004 fix: pull gender_hint from the script's [VOICE: NAME, gender, ...]
                # tag so we never assign a male voice to a female character (or vice versa).
                gender_hint = None
                if gender_map:
                    gender_hint = gender_map.get(upper_name)
                profile = _generate_character_profile(
                    character_idx, episode_seed, gender_hint=gender_hint
                )

                # De-duplicate voice presets: if this preset is already taken,
                # re-roll with offset seeds until we find an unused one in the
                # SAME gender pool (soft constraint — if pool exhausted, log and
                # accept duplicate).
                attempts = 0
                while profile["voice_preset"] in used_presets and attempts < 10:
                    attempts += 1
                    profile = _generate_character_profile(
                        character_idx + attempts * 100, episode_seed,
                        gender_hint=gender_hint,
                    )
                if profile["voice_preset"] in used_presets:
                    log.warning(
                        "[LLMDirector] CAST_GENDER_POOL_EXHAUSTED: %s (%s) "
                        "reusing preset %s — increase pool or accept duplicate",
                        upper_name, gender_hint or "unknown", profile["voice_preset"]
                    )

                used_presets.add(profile["voice_preset"])
                # FIX (v1.1): Use the ORIGINAL script name as the dict key so
                # BatchBark can match [VOICE: NAME ...] to the right preset.
                # The procedural name is stored in notes for the treatment file.
                new_voice_assignments[upper_name] = {
                    "voice_preset": profile["voice_preset"],
                    "notes": f"{profile['name']} — {profile['notes']}",
                }
                log.info("[LLMDirector] %s → voice: %s (profile: %s, %s, %s, %s)",
                         upper_name, profile["voice_preset"],
                         profile["name"], profile["gender"], profile["age"], profile["demeanor"])
                # BUG-004 telemetry — grep CAST_GENDER_MATCH to verify per-character matching
                _runtime_log(
                    f"CAST_GENDER_MATCH {upper_name}={gender_hint or 'unspecified'} "
                    f"→ {profile['voice_preset']} ({profile['gender']})"
                )
                character_idx += 1

        plan["voice_assignments"] = new_voice_assignments

        log.info("[LLMDirector] Procedural cast complete: %d characters "
                 "(%d unique presets)", len(new_voice_assignments), len(used_presets))

        return plan