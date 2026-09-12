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


#: A cue is 4 to 12 seconds long and it is BACKGROUND. Two things follow,
#: and both were learned the hard way on 2026-09-12 when the music finally
#: became audible and turned out to be looping:
#:
#: * SUSTAINED INSTRUMENTS LEAD. A text-to-audio model reads the first
#:   tokens hardest, and a plucked or struck instrument (lute, harpsichord,
#:   pizzicato, any drum) has a fixed transient it can only repeat. Ask for
#:   those first in a 12-second window and the model writes a two-bar loop,
#:   which is exactly what Stable Audio Open is built to do best.
#: * ONE PACE PER CUE. Asking for "slow held chords" and "brushed drums"
#:   in the same breath is not a richer instruction, it is a contradictory
#:   one, and the model resolves it by picking the rhythmic half.
#:
#: Plucked instruments are not banned -- a lute IS the sound of 1595 -- they
#: are placed after the sustained ones and qualified ("gentle lute").
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
    "warm strings, muted brass, clarinet, soft upright bass",
    "1940s radio drama orchestra",
)

EARLY_CONSORT = Palette(
    "early_consort",
    "viol consort, recorders, soft bowed strings, gentle lute",
    "Elizabethan consort music",
)
BAROQUE_CHAMBER = Palette(
    "baroque_chamber",
    "bowed strings, oboe, bassoon, gentle harpsichord",
    "baroque chamber music",
)
ROMANTIC_CHAMBER = Palette(
    "romantic_chamber",
    "string quartet, clarinet, French horn, soft piano",
    "Romantic-era chamber music",
)
ELECTRIC_COMBO = Palette(
    "electric_combo",
    "Hammond organ, sustained electric guitar, soft bass guitar",
    "1960s instrumental combo",
)
SCIFI_ORCHESTRA = Palette(
    "scifi_orchestra",
    "theremin, sustained strings, low brass, distant french horn",
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

#: Mood words -> the musical devices that express them, each tagged with the
#: PACE it implies. Matched at a word start, first group wins per term; the
#: short ambiguous stems are bounded so "warm" never reads as "war" and
#: "moonlight" never reads as "light".
#:
#: THE PACE TAG IS LOAD-BEARING (2026-09-12). `mood_devices` may return two
#: devices, and before this tag its only filter was dedup -- so a brief
#: reading "melancholic, playful" asked one 12-second cue for "slow cello,
#: held minor chords" AND "pizzicato strings, brushed drums" at once. The
#: model resolved that contradiction by playing the rhythmic half, in a loop.
#: Two devices may now be returned only when their paces agree, or when one
#: of them is neutral.
#:
#: NO PERCUSSION ANYWHERE IN THIS TABLE, and that is deliberate rather than
#: squeamish: a drum in a 4-to-12-second background cue can only be a loop,
#: and every percussion phrase that used to live here ("brushed drums",
#: "snare accents", "timpani") was in a cue the operator heard as a tape
#: deck. The emotions they carried are expressed by register and articulation
#: instead, which is what an orchestrator would reach for at this length.
_MOOD_DEVICES = (
    (r"\b(?:magic|enchant|dream|fairy|moonli|wonder|spell|fantas)",
     "shimmering sustained strings, soft celesta colour", "slow"),
    (r"\b(?:tense|tension|suspens|dread|fear|danger|menac|uneasy|anxi|nervous)",
     "minor key, low brass swells, slow rising strings", "slow"),
    (r"\b(?:sad\b|grief|loss\b|mourn|melanchol|sorrow|sombre|somber|elegiac|lament"
     r"|hopeless|loveless|despair|forlorn|lonel)",
     "a slow cello line over held minor chords", "slow"),
    (r"\b(?:dark\b|brood|sinister|ominous|grim\b|forebod|malevol)",
     "low strings, bass clarinet, slow dissonant chords", "slow"),
    (r"\b(?:urgent|urgency|chase|frantic|action|battle|conflict|war\b|fight|pursuit|storm)",
     "urgent strings climbing over a restless bass line", "fast"),
    (r"\b(?:grand\b|heroic|triumph|regal|majest|noble|royal|ceremon|epic\b)",
     "broad brass and soaring strings, a noble melody", "neutral"),
    (r"\b(?:myster|eerie|uncanny|strange|haunt|secret|shadow|deceit|decept|intrigue)",
     "sustained strings, a sparse questioning melody", "slow"),
    (r"\b(?:warm\b|warmth|tender|love\b|lovely|lover|loving|romanc|romantic|gentle"
     r"|hope\b|hopeful|affection|joy)",
     "major key, legato strings, soft woodwinds", "slow"),
    (r"\b(?:playful|comic|comed|whims|mischie|merry|jest|light-?hearted|witty)",
     "a light dancing woodwind melody, bright major colour", "fast"),
    (r"\b(?:calm\b|serene|peace|pastoral|quiet|tranquil|still\b|reflective)",
     "gentle flute melody over soft strings", "slow"),
)
_COMPILED_MOOD_DEVICES = tuple(
    (re.compile(pattern, re.IGNORECASE), device, pace)
    for pattern, device, pace in _MOOD_DEVICES)

#: The tempo phrase each pace asks for. A text-to-audio model is given no
#: metre at all otherwise -- duration is a separate argument and never
#: reaches the text -- and an unpaced request at this length comes back as a
#: repeating figure. These are deliberately words, not a BPM number: a cue
#: is underscore and must not imply a click track.
_PACE_TEMPO = {
    "slow": "slow tempo, unhurried, expressive rubato",
    "fast": "moving tempo, flowing line",
    "neutral": "steady unhurried tempo",
}

#: When no mood word maps: still a musical instruction, never a texture, and
#: never "steady" -- that word asks for exactly the repetition this module
#: spent 2026-09-12 removing.
DEFAULT_DEVICE = "a clear unhurried melody over warm harmony"
DEFAULT_PACE = "slow"

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
    of the words map. Total over junk input.

    ONE PACE PER CUE. A second device joins only when its pace agrees with
    the first, or when either is neutral. Before that rule a brief reading
    "melancholic, playful" asked a 12-second cue to be slow and held AND
    pizzicato with brushed drums, and the model answered with a loop
    (operator, 2026-09-12: "a loop-a-loop tape deck").
    """
    return _mood_devices_with_pace(mood_terms, limit=limit)[0]


def mood_pace(mood_terms) -> str:
    """The pace the brief's mood words imply -- ``"slow"`` / ``"fast"`` /
    ``"neutral"``. Drives the tempo phrase, which is the only musical-time
    information the model receives.

    PUBLIC ON PURPOSE though `tempo_phrase` is its only caller today: the
    pace is the fact, the phrase is one rendering of it, and a caller that
    wants to branch on pace (a future per-pace cue length, say) should not
    have to parse English out of the phrase."""
    return _mood_devices_with_pace(mood_terms, limit=2)[1]


def tempo_phrase(mood_terms) -> str:
    """The words that tell the model how fast to play. Never a BPM number:
    a background cue must not imply a click track."""
    return _PACE_TEMPO.get(mood_pace(mood_terms), _PACE_TEMPO["slow"])


def _mood_devices_with_pace(mood_terms, *, limit: int = 2):
    """``(devices, pace)`` -- the shared body of the three readers above."""
    devices: list[str] = []
    paces: list[str] = []
    for term in (mood_terms or []):
        text = str(term or "")
        if not text.strip():
            continue
        for pattern, device, pace in _COMPILED_MOOD_DEVICES:
            if not pattern.search(text):
                continue
            if device in devices:
                break
            settled = next((p for p in paces if p != "neutral"), None)
            if devices and settled and pace != "neutral" and pace != settled:
                break          # one pace per cue -- the second device is dropped
            devices.append(device)
            paces.append(pace)
            break
        if len(devices) >= max(1, int(limit)):
            break
    if not devices:
        return [DEFAULT_DEVICE], DEFAULT_PACE
    return devices, next((p for p in paces if p != "neutral"), paces[0])
