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

# -----------------------------------------------------------------------------
# Name -> gender classification (S1, cast name<->gender<->voice coherence fix).
#
# FIRST_NAMES above is a flat, gender-blind pool: pick_first_last() rolls a
# name from it with no regard for the slot gender Python later binds, so a
# male-coded name (MALIK) can land on a female slot/voice and vice versa. The
# voice always followed gender correctly; only the NAME was wrong.
#
# FIRST_NAMES_BY_GENDER partitions the SAME names by gender so the repair pass
# (nodes/_otr_casting.py) can swap a mismatched first name for a same-gender one
# without ever touching the cast RNG. FIRST_NAMES stays byte-identical (its
# order is load-bearing for C7 -- pick_first_last draws rng.choice(FIRST_NAMES)),
# so these buckets are ADDITIVE: a partition view, never a replacement.
#
# Classification policy: only clearly male- or female-coded names are tagged
# male/female. Genuinely cross-cultural or ambiguous names (Quinn, Ren, Charlie,
# Dao, ...) are "unisex" -- unisex is coherent with EITHER binary slot gender,
# so it never triggers a (false) repair. This deliberately minimizes repairs to
# the unambiguous mismatches that motivated the fix.
# -----------------------------------------------------------------------------
FIRST_NAMES_BY_GENDER: dict[str, list[str]] = {
    "male": [
        "Vance", "Sully", "Mac", "Cole", "Drake", "Kane", "Malik", "Kael",
        "Tariq", "Kenji", "Jiro", "Hiro", "Lev", "Dmitri", "Sergei", "Volkov",
        "Yuri", "Nelson", "Martin", "Carl", "Lenny", "Montgomery", "Seymour",
        "Ned", "Barney", "Moe", "Kent", "Rod", "Todd", "Jimbo", "Dolph",
        "Kearney", "Tommy", "Allan", "Cavor", "Dracula", "Edward", "Griffin",
        "Gulliver", "Henry", "James", "John", "Karnacki", "Nemo", "Phileas",
        "Quasimodo", "Robinson", "Sherlock", "Smee", "Victor", "Watson",
        "Lawrence", "Reginald", "Anton", "Priam", "Maurice", "Alan", "Truman",
        "Fletcher", "Joel", "Stanley", "Walter", "Ace", "Lloyd", "Bruce",
        "Mork", "Sean", "Andrew", "Parry", "Malcolm", "Daniel", "Michael",
        "Ryan", "Kevin", "Toby", "Darryl", "Creed", "Oscar", "Steve", "Ed",
        "Doug", "Travis", "Will", "Faber", "Rick", "Glen", "Isidore", "Bob",
        "Manfred", "Leo", "Gus", "Monty", "Duane", "Rufus", "Leroy", "Skip",
        "Grover", "Peter",
        # --- from the RETIRED "unisex" bucket, 2026-08-15 ------------------
        # ADRIAN: male-coded English given name with its own female
        # counterpart (Adrienne) and no surname reading.
        "Adrian",
        # Surname-style given names, male-leaning in US given-name usage.
        # THE CLOSEST CALLS IN THE FILE -- a woman addressed as "Carter" or
        # "Stone" is ordinary, especially in the military and noir registers
        # this show works in. These are the first to flip if casts feel wrong.
        "Stone", "Hayes", "Carter", "Palmer",
        # English, male-leaning.
        "Blake", "Dale", "Chris", "Charlie", "Rainn",
        # Japanese, male-leaning usage.
        "Ren", "Akira", "Sora",
        # Thai, male.
        "Krit", "Niran",
        # Igbo; used for both, male-leaning in practice. CLOSE CALL.
        "Chidi",
        # Not personal names -- a sea monster, an invented sci-fi name, a
        # spirit. There is no gender fact to get right or wrong; each is
        # ASSIGNED rather than reasoned, and reads male in the register the
        # show uses them in.
        "Leviathan", "Tarkon", "Djinn",
    ],
    "female": [
        "Margot", "Nora", "Zuri", "Oya", "Nia", "Mali", "Anya", "Mira", "Edna",
        "Alice", "Ayesha", "Mina", "Wendy", "Pam", "Meredith", "Erin",
        "Phyllis", "Jenna", "Mindy", "Ellie", "Rashida", "Clarisse", "Donna",
        "Juliana",
        # --- from the RETIRED "unisex" bucket, 2026-08-15 ------------------
        # English/US usage has moved these decisively female in living memory.
        "Quinn", "Reese", "Kelly", "Sailor",
        # Japanese, female-leaning usage.
        "Yuki", "Rei",
        # Thai, female-leaning ("Dao" is star, "Som" is orange).
        "Sunan", "Dao", "Pim", "Som",
        # Yoruba; used for both, female-leaning standing alone. CLOSE CALL.
        "Ayo",
        # A nickname rather than a given name, female-leaning.
        "Pinky",
    ],
    # THE UNISEX BUCKET IS RETIRED, AND IT IS DELIBERATELY LEFT EMPTY RATHER
    # THAN DELETED -- `names_for_genre` and `_verify_name_buckets` both index
    # "unisex" by name, and the genre views iterate all three buckets.
    #
    # WHY IT WENT (operator ruling 2026-08-15: "I don't trust it"). A "unisex"
    # tag was the ONE value that told `_repair_ensemble_names` to stand down: it
    # exempted a name from the coherence check on EITHER binary slot gender. So
    # membership was a promise that the name genuinely works with a male or a
    # female voice -- and 30 of 153 names, one draw in five, were riding that
    # promise with nothing verifying it. ADRIAN sat there, and ADRIAN on a
    # female slot is the "Miss McFiggins" defect the gender work exists to stop.
    #
    # The existing coherence assertions could never have caught it. Both read
    # `assert gender_of_first_name(name) in (row["gender"], "unisex", "unknown")`
    # (`tests/test_cast_llm_naming.py:189,204`), where "unisex" is an
    # unconditional free pass -- the test encodes the very assumption that was
    # unsafe. `tests/test_cast_invariants.py` R10 replaces it with a
    # BEHAVIOURAL bar instead.
    #
    # Every name now carries a definite tag, so the repair always has an answer
    # and nothing is exempt. Assignments are best-judgment, not sourced, and
    # each is grouped above with its basis so a single call can be flipped
    # without re-litigating the rest. "unknown" still exists for names outside
    # the pool entirely (the LLM slot-fill path can invent one) and is still
    # treated as un-repairable -- that is a separate hole, and a real one.
    "unisex": [],
}

# Reverse index UPPER(first name) -> "male"|"female"|"unisex". Names are stored
# title-case in the buckets but the cast carries them upper-cased
# ("MALIK HIBBERT"), so the lookup key is upper-cased on both sides.
_FIRST_NAME_GENDER_INDEX: dict[str, str] = {}
for _bucket_gender, _bucket_names in FIRST_NAMES_BY_GENDER.items():
    for _bucket_name in _bucket_names:
        _FIRST_NAME_GENDER_INDEX[_bucket_name.upper()] = _bucket_gender


def _verify_name_buckets() -> None:
    """Fail fast at import if the gender buckets drift out of sync with
    FIRST_NAMES. A real RuntimeError (not assert) so ``python -O`` can't
    strip the guard -- a silent drift would let a mis-tagged name slip a
    bad repair into production.
    """
    union: set[str] = set()
    seen = 0
    for _g, _names in FIRST_NAMES_BY_GENDER.items():
        union |= set(_names)
        seen += len(_names)
    canonical = set(FIRST_NAMES)
    if union != canonical:
        missing = sorted(canonical - union)
        extra = sorted(union - canonical)
        raise RuntimeError(
            "FIRST_NAMES_BY_GENDER drift vs FIRST_NAMES: "
            f"missing={missing} extra={extra}"
        )
    if seen != len(union):
        raise RuntimeError(
            "FIRST_NAMES_BY_GENDER buckets must be disjoint "
            f"(total tagged {seen} != unique {len(union)})"
        )


_verify_name_buckets()


# Genre-biased name views (S1). OTR_CAST_GENRE selects a flavor subset; the
# default "auto" reproduces today's behavior exactly (the full FIRST_NAMES pool
# in its original order). Each non-auto genre is derived by intersecting a
# pillar name-set with the gender buckets, so every genre name is guaranteed
# already gender-classified -- no separate hand-curation to keep in sync.
_GENRE_NAME_SETS: dict[str, set[str]] = {
    # Hard-boiled 1950s Americana + public-domain classics + sitcom Americana.
    "scifi_1950s": {
        "Vance", "Stone", "Margot", "Nora", "Sully", "Mac", "Hayes", "Cole",
        "Drake", "Quinn", "Reese", "Kane", "Carter", "Blake", "Nelson",
        "Martin", "Carl", "Edna", "Ned", "Kent", "Alice", "Edward", "Henry",
        "James", "John", "Victor", "Wendy", "Walter", "Michael", "Pam",
        "Donna", "Bob", "Leo", "Peter",
    },
    "noir": {
        "Vance", "Stone", "Margot", "Nora", "Sully", "Mac", "Hayes", "Cole",
        "Drake", "Quinn", "Reese", "Kane", "Carter", "Blake", "Sherlock",
        "Watson", "Karnacki", "Victor", "Fletcher", "Rick", "Bob",
        "Travis", "Palmer", "Glen", "Manfred",
    },
    # Afrofuturism + Neo-Tokyo + Russian Dieselpunk + Thai density pillars.
    "space_opera": {
        "Malik", "Zuri", "Chidi", "Ayo", "Oya", "Kael", "Tariq", "Nia", "Ren",
        "Akira", "Kenji", "Yuki", "Sora", "Jiro", "Rei", "Hiro", "Krit",
        "Mali", "Niran", "Sunan", "Dao", "Pim", "Som", "Lev", "Anya", "Dmitri",
        "Sergei", "Volkov", "Mira", "Yuri", "Tarkon", "Leviathan", "Djinn",
    },
}

FIRST_NAMES_BY_GENRE: dict[str, dict[str, list[str]]] = {
    "auto": FIRST_NAMES_BY_GENDER,
}
for _genre, _nameset in _GENRE_NAME_SETS.items():
    FIRST_NAMES_BY_GENRE[_genre] = {
        _g: [_n for _n in FIRST_NAMES_BY_GENDER[_g] if _n in _nameset]
        for _g in ("male", "female", "unisex")
    }


# The legacy procedural trait pools (GENDERS / AGE_BRACKETS / DEMEANORS /
# ACCENTS) were removed 2026-08-28: zero in-repo readers. The ACCENTS lesson
# survives the list it sat above and still binds casting: foreign Bark presets
# (de_speaker, fr_speaker, ...) hallucinate foreign-language phonemes on
# English text -- v1.1 "Test Signal" Lemmy (de_speaker_0) was unintelligible --
# so every character uses en_speaker_* presets until Bark's multilingual
# stability improves.

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

# ANNOUNCER voice pool - randomized per episode for gender balance.
#
# THIS IS THE KOKORO POOL ONLY. The announcer's DEFAULT engine is Kokoro,
# and these voice IDs are Kokoro-namespaced (bm_* = British male, bf_* =
# British female) -- they CANNOT collide with the Bark VOICE_PROFILES above,
# since they live in separate TTS namespaces. Open-character voice picking
# only excludes Bark presets; a Kokoro announcer voice is never on that list.
#
# BARK IS ALSO A VALID ANNOUNCER ENGINE (2026-08-24, operator-selectable via
# announcer_voice_engine="bark"). It does NOT draw from this list -- there is
# no separate "Bark announcer pool" constant. `OTR_CastLock._assign_bark_
# announcer` draws directly from `VOICE_PROFILES` above (the same ten
# presets characters use), dynamically excluding whatever a character in the
# episode already took. `pick_announcer()` below always returns a Kokoro
# row; CastLock re-stamps it to Bark AFTER this function runs, when
# requested. Do not read "the announcer renders through Kokoro" as universal
# fact anymore -- it is the DEFAULT, not the only path.
#
# Pool composition (2 male + 2 female) drives the natural 50/50 announcer
# gender split per episode that Jeffrey called out 2026-05-10.
#
# Source of truth: this list mirrors `ANNOUNCER_VOICE_POOL` in
# nodes/_otr_audio_engines/eng_kokoro.py (the per_line kokoro engine, audio
# clean-break 1b). Keep them in sync; a follow-up is to import the pool from
# there so there is one canonical list.
ANNOUNCER_PRESETS = [
    ("bm_george", "BBC authoritative male"),
    ("bm_fable",  "documentary relaxed male"),
    ("bf_emma",   "BBC authoritative female"),
    ("bf_lily",   "documentary relaxed female"),
]

# LEMMY fixed profile - always gravelly/raspy male, English-native preset
LEMMY_PROFILE = {
    "name": "LEMMY",
    "gender": "male",
    "age": "50s",
    "demeanor": "gravelly",
    "accent": "cockney",
    "dialogue_orthography": "standard_english",
    "speech_signature": "Warm, quick-witted Cockney phrasing and rhythm using standard English spelling",
    "voice_preset": "v2/en_speaker_8",  # English native - gravelly, confident, 40s-50s. Avoids Bark hallucination from de_speaker
    "character_description": "Genial communications officer, 50s, broad friendly Cockney accent, quick-witted and humorous with a warm grin, brandishing a handheld brass communicator that looks like a polycorder crossed with a harmonica",
    "notes": "Male, gravelly/raspy, 50s, warm characterful voice, iconic",
}


#: WHICH CATALOGUE VOICE EACH RECURRING CHARACTER IS DELIVERED WITH, per engine.
#:
#: A recurring character is one a story lane writes on purpose and who should
#: sound like himself across episodes. Today that is exactly one, LEMMY. A later
#: character is ANOTHER KEY IN THIS TABLE -- not a new branch, not a new cameo,
#: not a second selector. The helper below and its callers iterate the registry;
#: nothing downstream may special-case a name.
#:
#: THESE IDS STAY SHARED. An assignment is not a reservation: `bm_george` is a
#: kokoro catalogue row tagged preferred_announcer, and the two cloud ids are
#: shared as well. ANNOUNCER still draws them through the announcer pool.
#: Reserving them to protect a recurring character would starve that pool --
#: reservation lives on the bank row (`reserved_for`) and covers only the clone
#: recordings of the character's own voice.
#:
#: The table owns a DELIVERED VOICE for a mapped engine. It does not own story
#: inclusion, a name, a description or a gender: those stay on the character
#: profile and on the cameo decision, which remain the only authorities for
#: whether the character appears at all.
#:
#: Engines absent from a character's map take the ordinary draw. There is
#: deliberately no entry for Bark (it keeps its preset path) or for the clone
#: engines, whose reserved rows are excluded from ordinary draws instead.
RECURRING_CHARACTER_VOICES = {
    "LEMMY": {
        "kokoro": "bm_george",
        "cloud_elevenlabs": "el_daniel",
        "google_tts": "gt_algenib",
    },
}


def recurring_character_key(entry, registry=None) -> str:
    """The canonical registry key this cast row is, or "" if it is none of them.

    Matches the row's ``name`` first and then its ``char_id``, each stripped and
    casefolded, against the registry's keys compared the same way. Returns the
    CANONICAL key as the table spells it, so callers index the table with the
    answer rather than re-deriving it.

    ``registry=None`` uses the configured table. AN EXPLICITLY EMPTY REGISTRY
    STAYS EMPTY -- `{}` means "no recurring characters", not "use the default",
    which is what lets a test inject a registry and get exactly it.

    Returns "" for anything that is not a dict, so a malformed row is an
    ordinary row rather than an exception inside casting. The caller's announcer
    check runs BEFORE this lookup, so a row named ANNOUNCER never reaches it
    even if someone adds that name to the table.
    """
    if registry is None:
        registry = RECURRING_CHARACTER_VOICES
    if not registry or not isinstance(entry, dict):
        return ""
    by_folded = {}
    for key in registry:
        by_folded[str(key).strip().casefold()] = key
    for field in ("name", "char_id"):
        value = str(entry.get(field) or "").strip().casefold()
        if value and value in by_folded:
            return by_folded[value]
    return ""


def recurring_character_voice(character_key, engine, registry=None) -> str:
    """The catalogue id `character_key` is delivered with on `engine`, or "".

    Separate from the key lookup so a caller that already knows the key does not
    re-match the row, and so the "no mapping for this engine" answer is one
    empty string rather than a KeyError the caller has to guard.
    """
    if registry is None:
        registry = RECURRING_CHARACTER_VOICES
    if not registry:
        return ""
    mapping = registry.get(character_key) or {}
    if not isinstance(mapping, dict):
        return ""
    return str(mapping.get(str(engine or "").strip()) or "").strip()


# -----------------------------------------------------------------------------
# VOICE_REGISTRY -- unified per-model voice catalog
# -----------------------------------------------------------------------------
#
# Single dict that exposes each TTS family's voices + allowed parameter spec
# in one place. Today the registry has two entries (bark, kokoro); when a new
# TTS model lands (Fish Speech, CosyVoice, the future period model) add an
# entry here.
#
# Schema per entry:
#   "presets":     list of (preset_id, short_description) tuples
#   "params_spec": dict mapping param-name -> (min, max, default) tuple
#                  for any model-tunable knob the LLM should pick per
#                  character. Empty today on both Bark and Kokoro --
#                  we'll fill these in when the casting LLM call is
#                  wired to ask for params (cast contract Phase 2;
#                  see project_cast_contract_p2_followups.md).
#
# Consumers should read THIS registry, not the standalone constants below,
# when they need a model-aware view. The standalone constants are kept for
# back-compat with code that already imports them directly.
VOICE_REGISTRY: dict[str, dict] = {
    "bark": {
        "presets": [
            # Lifted from VOICE_PROFILES; flattened to (preset, short)
            # so the registry shape is uniform across models.
            (p, " ".join(sorted(tags)))
            for p, _g, _lang, tags in VOICE_PROFILES
        ],
        "params_spec": {},  # placeholder -- e.g. "temperature": (0.5, 0.9, 0.7)
    },
    "kokoro": {
        "presets": list(ANNOUNCER_PRESETS),
        "params_spec": {},  # placeholder -- e.g. "speed": (0.8, 1.2, 1.0)
    },
}

# Convenience constant for callers that want to know which TTS models are
# currently supported by the registry.
KNOWN_TTS_MODELS: tuple[str, ...] = tuple(VOICE_REGISTRY.keys())


# LEMMY 11% cameo rate. Statistically held by tests/lemmy_rng_check.py.
LEMMY_RATE = 0.11

# Module-level SystemRandom (OS entropy). The LEMMY cameo roll ALWAYS
# uses this -- never a seeded RNG -- so the easter egg stays a genuine
# ~11% surprise, decoupled from the C7 byte-identity seed. See
# roll_lemmy() and BUG-LOCAL-260.
_LEMMY_RNG_SYSTEM = SystemRandom()


def roll_lemmy() -> bool:
    """Return True with probability LEMMY_RATE (~11%), False otherwise.

    Always rolls against the module-level SystemRandom (OS entropy),
    never a seeded RNG -- the LEMMY cameo is a genuine surprise,
    decoupled from the C7 byte-identity seed.

    History (BUG-LOCAL-260, 2026-05-23): a 2026-05-10 change routed
    this roll through the cast contract's seeded random.Random so an
    explicitly-seeded run was byte-reproducible end to end. But the
    writer's `seed` widget ships a fixed value, and a fixed seed
    reproduces ONE roll forever -- so a LEMMY-positive seed (42 was
    one) cast LEMMY on 100% of runs and a LEMMY-negative seed on 0%.
    A fixed seed can never yield the intended ~11%. Decoupling the
    roll from the seed restores the rare cameo; the deliberate
    trade-off is that LEMMY's hit is no longer reproducible from the
    seed. Cast names, the announcer pick, and the style picker stay
    fully seed-deterministic. Tests force a deterministic LEMMY via
    the `force_lemmy` knob on assemble_pre_locked_rows.
    """
    return _LEMMY_RNG_SYSTEM.random() < LEMMY_RATE


# Back-compat alias: some code expects `_LEMMY_RNG` as a module
# attribute. Today both story_orchestrator.py:33 and any external
# reference resolve to the SystemRandom one (the unseeded path).
_LEMMY_RNG = _LEMMY_RNG_SYSTEM


def pick_announcer(rng: random.Random) -> dict:
    """Return an announcer cast row using a Kokoro voice preset.

    Picks one of the four ANNOUNCER_PRESETS at random using the given
    seeded RNG; the 50/50 gender split falls out naturally from the
    pool composition (2 male + 2 female).

    Gender is derived from the Kokoro voice-ID prefix convention:
      bm_* = British male
      bf_* = British female
    (am_*, af_* = American male/female, not currently in our pool but
    handled defensively below.)

    The returned row stamps `tts_model="kokoro"` so downstream
    consumers can route by reading the field directly instead of
    pattern-matching the voice_preset prefix.
    """
    voice_preset, vocal_desc = rng.choice(ANNOUNCER_PRESETS)
    if voice_preset.startswith(("bm_", "am_")):
        gender = "male"
    elif voice_preset.startswith(("bf_", "af_")):
        gender = "female"
    else:
        # Fallback: parse the description text.
        gender = "male" if "male" in vocal_desc.lower() else "female"
    return {
        "name": "ANNOUNCER",
        "gender": gender,
        "tts_model": "kokoro",
        "voice_preset": voice_preset,
        # voice_params is a model-dependent dict (or None) where the
        # casting LLM stores per-character knobs it chose (e.g. Bark
        # "temperature", Kokoro "speed"). None today; Phase 2 will
        # populate when the LLM call learns to pick params.
        "voice_params": None,
        "character_description": (
            "Period radio announcer; reads the science story and "
            "frames the drama between beats."
        ),
        "notes": vocal_desc,
    }


def lemmy_row() -> dict:
    """Return a fully-populated LEMMY cast row in the same shape
    pick_announcer / casting LLM responses produce.

    `v2/en_speaker_8` is LEMMY's WRITER-STAGE identity, and the row stamps
    `tts_model="bark"` to say so. It is not a claim about delivery: CastLock
    may deliver him on another engine instead -- his own `reserved_for` bank
    row, or his `RECURRING_CHARACTER_VOICES` catalogue id -- and on the
    canonical graph it does. What CastLock's stamp does with the preset,
    measured 2026-09-20 rather than remembered: on a bark stamp it is left
    alone; on any stamp by another engine the leftover `v2/` is cleared,
    because no engine but bark reads it (Lime, 2026-09-17).
    """
    return {
        "name":                  LEMMY_PROFILE["name"],
        "gender":                LEMMY_PROFILE["gender"],
        "accent":                LEMMY_PROFILE["accent"],
        "dialogue_orthography":  LEMMY_PROFILE["dialogue_orthography"],
        "speech_signature":      LEMMY_PROFILE["speech_signature"],
        "tts_model":             "bark",
        "voice_preset":          LEMMY_PROFILE["voice_preset"],
        # voice_params: None today; Phase 2 populates when LLM picks
        # per-character knobs. LEMMY's fixed iconic profile may stay
        # at None permanently if we don't want per-episode variation
        # on his voice.
        "voice_params":          None,
        "character_description": LEMMY_PROFILE["character_description"],
    }


_VOCAL_TAGS = frozenset({
    "warm", "weary", "wry", "calm", "measured", "energetic",
    "sharp", "anxious", "confident", "authoritative", "deep",
    "intense", "dry", "stoic", "gravelly", "clipped", "precise",
    "nervous",
})
_AGE_TAGS = frozenset({"20s", "30s", "40s", "50s", "60s"})


def open_voice_pool(taken: set[str]) -> list[tuple[str, str]]:
    """Return the (preset, short_description) list of voices NOT yet
    taken, suitable for inlining into the per-character casting prompt.

    short_description is a compact one-liner derived from the voice's
    quality tags. Keeps the prompt tight per the local-LLM brevity rule.

    DETERMINISM (C7): VOICE_PROFILES tags is a Python set, and Python
    set iteration order is hash-randomization dependent across runs.
    Sort tags before slicing so the rendered short-description is
    byte-stable across processes -- otherwise the LLM prompt itself
    would drift between identical-input runs and break C7.
    """
    out: list[tuple[str, str]] = []
    for preset, gender, _lang, tags in VOICE_PROFILES:
        if preset in taken:
            continue
        # Compress quality tags to "gender + 1-2 vocal traits + age bracket"
        # so the LLM has just enough to pick. Drop role-shaped tags
        # (officer, pilot, etc) -- those bias selection without helping.
        # `sorted(tags)` is the C7 determinism guard.
        sorted_tags = sorted(tags)
        vocal = [t for t in sorted_tags if t in _VOCAL_TAGS][:2]
        age = next((t for t in sorted_tags if t in _AGE_TAGS), "")
        short = " ".join([gender] + vocal + ([age] if age else "")).strip()
        out.append((preset, short))
    return out


def pick_first_last(
    rng: random.Random,
    taken_names: set[str],
    genre: str = "auto",
) -> str:
    """Roll a 'FIRSTNAME LASTNAME' from the curated pool, uppercase,
    avoiding any name already in `taken_names`.

    Falls back to retry up to 50 times before giving up and returning
    a raw roll (which the caller should accept). On a 154/54 pool, 50
    retries handles any plausible num_characters request.

    genre: "auto" (default) draws from the full FIRST_NAMES pool in its
    original order -- byte-identical to the pre-S1 behavior, so a fixed
    OTR_CAST_SEED reproduces the exact same rolls (C7). A non-auto genre
    (see FIRST_NAMES_BY_GENRE) draws from that genre's flavor subset; this
    only takes effect when OTR_CAST_GENRE is explicitly set, so default
    runs are byte-identical.
    """
    genre_norm = (genre or "auto").strip().lower()
    # C7: the auto path MUST draw rng.choice(FIRST_NAMES) on the original
    # list object so the sequence is unchanged for a fixed seed.
    if genre_norm == "auto" or genre_norm not in FIRST_NAMES_BY_GENRE:
        first_pool = FIRST_NAMES
    else:
        first_pool = names_for_genre(genre_norm) or FIRST_NAMES
    for _ in range(50):
        first = rng.choice(first_pool)
        last = rng.choice(LAST_NAMES)
        name = f"{first} {last}".upper()
        if name not in taken_names:
            return name
    # Cosmic-collision fallback -- accept whatever the last roll gave.
    return f"{rng.choice(first_pool)} {rng.choice(LAST_NAMES)}".upper()


def gender_of_first_name(name: str) -> str:
    """Classify a first name as "male" | "female" | "unisex" | "unknown".

    Case-insensitive. Accepts either a bare first name ("Malik") or a full
    "FIRST LAST" cast name ("MALIK HIBBERT") -- only the first whitespace
    token is classified. "unknown" means the name is not in the curated
    pool; callers treat unknown like unisex (never force a repair on a name
    we cannot confidently gender).
    """
    if not name:
        return "unknown"
    head = name.strip().split()
    if not head:
        return "unknown"
    return _FIRST_NAME_GENDER_INDEX.get(head[0].upper(), "unknown")


def names_for_genre(genre: str, gender: str | None = None) -> list[str]:
    """Return the first-name list for a genre, optionally filtered to one
    gender bucket.

    "auto" (or any unknown genre) returns the full pool: FIRST_NAMES in its
    original order when unfiltered (byte-identical roll source), or the
    gender bucket when a gender is given. A known non-auto genre returns
    that genre's curated subset.
    """
    genre_norm = (genre or "auto").strip().lower()
    gender_norm = (gender or "").strip().lower() or None
    if genre_norm == "auto" or genre_norm not in FIRST_NAMES_BY_GENRE:
        if gender_norm in ("male", "female", "unisex"):
            return list(FIRST_NAMES_BY_GENDER[gender_norm])
        return list(FIRST_NAMES)
    buckets = FIRST_NAMES_BY_GENRE[genre_norm]
    if gender_norm in ("male", "female", "unisex"):
        return list(buckets[gender_norm])
    return [n for g in ("male", "female", "unisex") for n in buckets[g]]
