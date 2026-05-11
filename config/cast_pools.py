"""
config/cast_pools.py -- canonical name / voice / trait pools for cast contract.

Lifted from v1.7:nodes/story_orchestrator.py:395-548 (the pre-LPL
procedural cast generator) and relocated here as the single source of
truth for the v2.0-alpha cast contract. The pre-LPL code is gone but
its pools were curated over many runs and many bug fixes -- this
module preserves every BUG-004 / FIX-3 / accent-ban comment verbatim
because that commentary IS the spec.

The cast contract LLM caller (nodes/_otr_casting.py) and the cast
assembler both import from this module. Do NOT inline these pools at
the call site. Do NOT lift them to JSON -- the inline bug-history
commentary is load-bearing and JSON would lose it.

Era-agnostic. Per the 2026-05-10 prompt-style rule, no hardcoded
period literals appear in pool entries; period flavor flows in via
the user's `style` + `news_seed` choices upstream.
"""
from __future__ import annotations

import random
from secrets import SystemRandom

# -----------------------------------------------------------------------------
# PROCEDURAL CHARACTER GENERATOR - name, age, gender, demeanor, accent, voice
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
#     in batch_bark_generator.py) - this prevents language drift
#   - Temperature capped at 0.55 for international presets (0.5 first lines)
# -----------------------------------------------------------------------------

# Sci-fi character name pools - contemporary, neutral, tech-aligned
# Omni-Retro 5-Pillar Naming Pool - short, punchy, Bark-optimized (1-2 syllables, hard consonants)
# Pillars: 1950s Americana Noir, Afrofuturism, Neo-Tokyo Cyberpunk, Thai Density, Russian Dieselpunk
FIRST_NAMES = [
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
    # The Office - generic character first names
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

LAST_NAMES = [
    "Stone", "Shaw", "Cross", "Wells", "Steele", "Frost", "Pierce", "Vaughn",
    "Black", "Drake", "Hayes", "Kane", "Voss", "Cranston", "Kendall", "Reeves",
    "Volkov", "Sato", "Tanaka", "Okafor", "Diallo", "Sirikit", "Petrov",
    # Generic last names (scrubbed franchise-specific)
    "Burns", "Hibbert", "Flanders", "Houten", "Smithers",
    "Terwilliger", "Bouvier", "Simpson", "Gordon", "Ming",
    "Carruthers", "Corben",
    # The Office - character last names (generic ones only)
    "Scott", "Halpert", "Beesly", "Howard", "Bernard", "Malone",
    "Kapoor", "Palmer", "Hudson", "Martin", "Flenderson", "Philbin", "Vance",
    # Ray Bradbury (generic)
    "Beatty", "Spender", "Stendahl", "Eckels", "Halloway",
    # Misc classic (generic)
    "Steiner",
]

# Trait pools for procedural character profiles
GENDERS = ["male", "female"]
AGE_BRACKETS = ["20s", "30s", "40s", "50s", "60s"]
DEMEANORS = [
    "calm", "intense", "warm", "sharp", "dry", "energetic",
    "measured", "wry", "stoic", "anxious", "confident", "weary",
]

# Accent pool - 100% English-native presets only.
# Foreign presets (de_speaker, fr_speaker, etc.) caused Bark hallucinations:
# the model generates foreign-language phonemes when given English text,
# producing gibberish instead of accented English. Until Bark's multilingual
# stability improves, all characters use en_speaker_* presets.
# See: v1.1 "Test Signal" critique - Lemmy (de_speaker_0) was unintelligible.
ACCENTS = [
    ("neutral",  "en", 1.00),   # English-only - no foreign presets
]

# Voice presets mapped by gender + vocal quality + language code.
# English-native presets (en_speaker_*) have known vocal qualities.
# International presets (xx_speaker_*) are grouped by speaker index tendencies.
# Each entry: (preset, gender, lang_code, quality_tags)
VOICE_PROFILES = [
    # -- English native (neutral accent) --
    ("v2/en_speaker_0", "male",   "en", {"authoritative", "deep", "50s", "60s", "announcer", "commander"}),
    ("v2/en_speaker_1", "male",   "en", {"calm", "measured", "30s", "40s", "technical", "pilot"}),
    ("v2/en_speaker_3", "male",   "en", {"energetic", "sharp", "20s", "30s", "rebel", "technician"}),
    ("v2/en_speaker_5", "male",   "en", {"warm", "weary", "wry", "50s", "60s", "doctor", "scientist"}),
    ("v2/en_speaker_6", "male",   "en", {"intense", "dry", "stoic", "40s", "officer", "android"}),
    ("v2/en_speaker_8", "male",   "en", {"gravelly", "anxious", "confident", "40s", "50s", "engineer", "mechanic"}),
    # English native (female)
    ("v2/en_speaker_2", "female", "en", {"clipped", "precise", "30s", "40s", "officer", "neutral-british"}),
    ("v2/en_speaker_4", "female", "en", {"warm", "energetic", "wry", "30s", "40s", "pilot", "explorer"}),
    ("v2/en_speaker_9", "female", "en", {"authoritative", "confident", "intense", "50s", "60s", "commander", "senator"}),
    # FIX-3 (v1.2): en_speaker_7 reclassified to female to prevent CAST_GENDER_POOL_EXHAUSTED
    # on 3-female episodes (was causing VEX/ZARA to share en_speaker_9 and sound identical).
    # Bark labels en_speaker_7 as androgynous - in English it reads soft/lighter so we
    # use it as the "younger" female slot (20s, anxious/sharp/technician).
    ("v2/en_speaker_7", "female", "en", {"sharp", "anxious", "nervous", "20s", "30s", "technician", "hacker"}),
    # -- DISABLED: Foreign accent presets ------------------------------
    # These caused Bark hallucinations - the model generates foreign-language
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

# ANNOUNCER voice pool - randomized per episode for gender balance (50/50 male/female)
# ANNOUNCER always uses neutral English (en_speaker_*) - no accent
ANNOUNCER_PRESETS = [
    ("v2/en_speaker_0", "Male, authoritative, deep"),
    ("v2/en_speaker_1", "Male, measured, calm"),
    ("v2/en_speaker_4", "Female, warm, energetic"),
    ("v2/en_speaker_9", "Female, mature, authoritative"),
]

# LEMMY fixed profile - always gravelly/raspy male, English-native preset
LEMMY_PROFILE = {
    "name": "LEMMY",
    "gender": "male",
    "age": "50s",
    "demeanor": "gravelly",
    "accent": "neutral",
    "voice_preset": "v2/en_speaker_8",  # English native - gravelly, confident, 40s-50s. Avoids Bark hallucination from de_speaker
    "character_description": "Grizzled wrench-wielding engineer, 50s, gravelly voice, gruff mechanic, anxious-confident demeanor",
    "notes": "Male, gravelly/raspy, 50s, gruff mechanic voice, iconic",
}

# Bark LEMMY 11% roll - SystemRandom is OS entropy, unaffected by random.seed().
# This ensures the 11% coin flip gives a true ~11% per run even when the rest
# of the pipeline is using a seeded RNG for reproducibility.
LEMMY_RATE = 0.11

# Module-level RNG for the LEMMY cameo coin flip. Lifted from
# v1.7:nodes/story_orchestrator.py:33 -- the SystemRandom usage is
# load-bearing (see commit history and tests/lemmy_rng_check.py).
_LEMMY_RNG = SystemRandom()


def roll_lemmy() -> bool:
    """Return True with probability LEMMY_RATE (~11%), False otherwise.

    Uses OS entropy (SystemRandom) so the same widget config does not
    freeze the roll to always-hit or always-miss.
    """
    return _LEMMY_RNG.random() < LEMMY_RATE


def pick_announcer(rng: random.Random) -> dict:
    """Return an announcer cast row.

    Picks one of the four ANNOUNCER_PRESETS at random using the given
    seeded RNG; the 50/50 gender split falls out naturally from the
    pool composition (2 male + 2 female).
    """
    voice_preset, vocal_desc = rng.choice(ANNOUNCER_PRESETS)
    gender = "male" if "Male" in vocal_desc else "female"
    return {
        "name": "ANNOUNCER",
        "gender": gender,
        "voice_preset": voice_preset,
        "character_description": (
            "Period radio announcer; reads the science story and "
            "frames the drama between beats."
        ),
        "notes": vocal_desc,
    }


def lemmy_row() -> dict:
    """Return a fully-populated LEMMY cast row in the same shape
    pick_announcer / casting LLM responses produce.
    """
    return {
        "name":                  LEMMY_PROFILE["name"],
        "gender":                LEMMY_PROFILE["gender"],
        "voice_preset":          LEMMY_PROFILE["voice_preset"],
        "character_description": LEMMY_PROFILE["character_description"],
    }


def open_voice_pool(taken: set[str]) -> list[tuple[str, str]]:
    """Return the (preset, short_description) list of voices NOT yet
    taken, suitable for inlining into the per-character casting prompt.

    short_description is a compact one-liner derived from the voice's
    quality tags. Keeps the prompt tight per the local-LLM brevity rule.
    """
    out: list[tuple[str, str]] = []
    for preset, gender, _lang, tags in VOICE_PROFILES:
        if preset in taken:
            continue
        # Compress quality tags to "gender + 1-2 vocal traits + age bracket"
        # so the LLM has just enough to pick. Drop role-shaped tags
        # (officer, pilot, etc) -- those bias selection without helping.
        vocal = [t for t in tags if t in {
            "warm", "weary", "wry", "calm", "measured", "energetic",
            "sharp", "anxious", "confident", "authoritative", "deep",
            "intense", "dry", "stoic", "gravelly", "clipped", "precise",
            "nervous",
        }][:2]
        age = next((t for t in tags if t in {"20s", "30s", "40s", "50s", "60s"}), "")
        short = " ".join([gender] + vocal + ([age] if age else "")).strip()
        out.append((preset, short))
    return out


def pick_first_last(rng: random.Random, taken_names: set[str]) -> str:
    """Roll a 'FIRSTNAME LASTNAME' from the curated pool, uppercase,
    avoiding any name already in `taken_names`.

    Falls back to retry up to 50 times before giving up and returning
    a raw roll (which the caller should accept). On a 110/50 pool, 50
    retries handles any plausible num_characters request.
    """
    for _ in range(50):
        first = rng.choice(FIRST_NAMES)
        last = rng.choice(LAST_NAMES)
        name = f"{first} {last}".upper()
        if name not in taken_names:
            return name
    # Cosmic-collision fallback -- accept whatever the last roll gave.
    return f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}".upper()
