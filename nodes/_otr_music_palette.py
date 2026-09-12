"""nodes/_otr_music_palette.py -- what the music is PLAYED ON, per story.

Operator, 2026-09-11 (after listening to `moonlit_deception`): *"the 'music'
needs to be improved, doesn't sound like music ... sounds like radio hiss,
which is what we asked. So I am asking to go over ALL musical prompts and make
them more musical ... ideally it is relevant to the story."* And: it DID relate
to the story for a while (April / May) before drifting back to tape hiss.

A text-to-music model plays what it is told to play. Told "atmospheric, evokes
a moonlit wood, slow build, analog tape warmth" it renders texture; told
"lute, viol consort, recorder, Elizabethan consort music, minor key, tremolo
strings" it renders music. This module is the STORY -> ENSEMBLE step: a named
set of instruments and an idiom for the story's period and bank, and the
musical DEVICES the brief's mood words call for. `_otr_music_prompt` composes
them into the row text and the engine prompt; no engine prepends its own.

Keyed ONLY on ledger-stable facts -- `source_bank`, `source_meta` (the year,
read leniently: tonight's ledger says `"c. 1595"`), and the brief's
`music_mood_terms` -- so the same seed plays the same ensemble on replay. The
visual style roll is deliberately NOT an input: it is drawn from OS entropy
(`style_roll.seed_source`) and would make the instrumentation unreplayable.

PURE and total: no I/O, no engine imports, never raises on any meta shape --
a missing or malformed field is the house orchestra, never a dead render.
UTF-8 no BOM, ASCII-only.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Palette:
    """A named ensemble: ``key`` is the stable receipt label, ``instruments``
    the comma list the ENGINE hears first, ``idiom`` the period phrase the ROW
    text carries so a reader of the ledger knows what was asked for."""
    key: str
    instruments: str
    idiom: str


#: The house sound of the show: a 1940s radio-drama pit orchestra.
HOUSE_PALETTE = Palette(
    "radio_orchestra",
    "strings, muted brass, clarinet, vibraphone, upright bass",
    "1940s radio drama orchestra",
)

EARLY_CONSORT = Palette(
    "early_consort",
    "lute, viol consort, recorder, harpsichord",
    "Elizabethan consort music",
)
BAROQUE_CHAMBER = Palette(
    "baroque_chamber",
    "harpsichord, strings, oboe, bassoon",
    "baroque chamber music",
)
ROMANTIC_CHAMBER = Palette(
    "romantic_chamber",
    "piano, string quartet, clarinet, French horn",
    "Romantic-era chamber music",
)
ELECTRIC_COMBO = Palette(
    "electric_combo",
    "electric guitar, Hammond organ, bass guitar, drum kit",
    "1960s instrumental combo",
)
SCIFI_ORCHESTRA = Palette(
    "scifi_orchestra",
    "theremin, vibraphone, tremolo strings, timpani, low brass",
    "1950s science-fiction radio orchestra",
)

#: Period bands by the source's year: (exclusive upper bound, palette).
_PERIOD_BANDS = (
    (1650, EARLY_CONSORT),
    (1800, BAROQUE_CHAMBER),
    (1900, ROMANTIC_CHAMBER),
    (1960, HOUSE_PALETTE),
)
#: Past the last band.
_MODERN_PALETTE = ELECTRIC_COMBO

#: A bank with no usable year still has a period of its own.
_BANK_PALETTES = {
    "shakespeare": EARLY_CONSORT,
    "scifi_news_pro": SCIFI_ORCHESTRA,
}

#: Mood words -> the musical devices that express them. Matched at a word
#: start, first group wins per term; the short ambiguous stems are bounded so
#: "warm" never reads as "war" and "moonlight" never reads as "light".
_MOOD_DEVICES = (
    (r"\b(?:magic|enchant|dream|fairy|moonli|wonder|spell|fantas)",
     "celesta, harp glissandi, shimmering strings"),
    (r"\b(?:tense|tension|suspens|dread|fear|danger|menac|uneasy|anxi|nervous)",
     "minor key, tremolo strings, low brass swells"),
    (r"\b(?:sad\b|grief|loss\b|mourn|melanchol|sorrow|sombre|somber|elegiac|lament"
     r"|hopeless|loveless|despair|forlorn|lonel)",
     "slow cello line, muted piano, held minor chords"),
    (r"\b(?:dark\b|brood|sinister|ominous|grim\b|forebod|malevol)",
     "low strings, bass clarinet, slow dissonant chords"),
    (r"\b(?:urgent|urgency|chase|frantic|action|battle|conflict|war\b|fight|pursuit|storm)",
     "driving rhythm, staccato strings, snare accents"),
    (r"\b(?:grand\b|heroic|triumph|regal|majest|noble|royal|ceremon|epic\b)",
     "full brass fanfare, timpani, soaring strings"),
    (r"\b(?:myster|eerie|uncanny|strange|haunt|secret|shadow|deceit|decept|intrigue)",
     "sustained strings, harp arpeggios, sparse piano"),
    (r"\b(?:warm\b|warmth|tender|love\b|lovely|lover|loving|romanc|romantic|gentle"
     r"|hope\b|hopeful|affection|joy)",
     "major key, legato strings, soft woodwinds"),
    (r"\b(?:playful|comic|comed|whims|mischie|merry|jest|light-?hearted|witty)",
     "pizzicato strings, bright woodwinds, brushed drums"),
    (r"\b(?:calm\b|serene|peace|pastoral|quiet|tranquil|still\b|reflective)",
     "gentle flute melody, soft strings, slow tempo"),
)
_COMPILED_MOOD_DEVICES = tuple(
    (re.compile(pattern, re.IGNORECASE), device) for pattern, device in _MOOD_DEVICES)

#: When no mood word maps: still a musical instruction, never a texture.
DEFAULT_DEVICE = "a clear melody over steady harmony"

_FOUR_DIGITS = re.compile(r"(?<!\d)(\d{4})(?!\d)")


def year_of(source_meta) -> int | None:
    """The source's year as an int, read leniently from ``source_meta`` --
    ``1606``, ``"1606"``, ``"c. 1595"``, ``"1895-1897"`` all answer; anything
    without a plausible four-digit year answers ``None``. Never raises."""
    if not isinstance(source_meta, dict):
        return None
    for key in ("year", "date", "published", "first_performed", "period"):
        value = source_meta.get(key)
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            continue  # a nested object is not a date, whatever digits it holds
        match = _FOUR_DIGITS.search(str(value))
        if match:
            year = int(match.group(1))
            if 1000 <= year <= 2100:
                return year
    return None


def bank_of(meta) -> str:
    """The source bank name the ledger meta carries, lower-cased, or ``""``."""
    if not isinstance(meta, dict):
        return ""
    for value in (meta.get("source_bank"), meta.get("bank"),
                  (meta.get("story_routing") or {}).get("bank")
                  if isinstance(meta.get("story_routing"), dict) else None,
                  (meta.get("source_meta") or {}).get("bank")
                  if isinstance(meta.get("source_meta"), dict) else None):
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return ""


def story_palette(meta) -> Palette:
    """The ensemble for this story: by the source's year when it has one,
    else by bank, else the house orchestra. Total over every meta shape."""
    meta = meta if isinstance(meta, dict) else {}
    year = year_of(meta.get("source_meta"))
    if year is not None:
        for upper_bound, palette in _PERIOD_BANDS:
            if year < upper_bound:
                return palette
        return _MODERN_PALETTE
    return _BANK_PALETTES.get(bank_of(meta), HOUSE_PALETTE)


def mood_devices(mood_terms, *, limit: int = 2) -> list[str]:
    """The musical devices the brief's mood words call for -- at most
    ``limit`` distinct phrases, in mood order; the default device when none
    of the words map. Total over junk input."""
    devices: list[str] = []
    for term in (mood_terms or []):
        text = str(term or "")
        if not text.strip():
            continue
        for pattern, device in _COMPILED_MOOD_DEVICES:
            if pattern.search(text):
                if device not in devices:
                    devices.append(device)
                break
        if len(devices) >= max(1, int(limit)):
            break
    return devices or [DEFAULT_DEVICE]
