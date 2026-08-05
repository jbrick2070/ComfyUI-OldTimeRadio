"""Derive an episode's sound world from the SOURCE, not from a dice roll.

The defect this exists to close
------------------------------
On the adaptation lanes the style is drawn deterministically from the cast
seed, and the drawn style carries a `sound_world` -- "a fire in the grate, a
mantel clock, a teacup" -- which is then rendered into the prompt grammar AND
stamped into the ledger AND turned into the episode canon's sound palette.
Nothing consults the source. So a Wells time-travel chapter was given a
domestic parlour soundscape, and the listener heard a world the author never
wrote. The arc-shape roll had exactly this defect and was already gated off
for adaptation lanes; the sound world is the same bug one field over.

This module derives the sound world from the work itself. It is deterministic
and model-free: the same body always yields the same line. It reports what the
SOURCE mentions, in the order the source first mentions it, and when the text
offers nothing recognizable it says so with an explicit neutral default rather
than inventing a world.

What it is NOT: a filter, a judgement, or a rewrite. It never rejects a source
and never edits the author's text.
"""
from __future__ import annotations

import re
from typing import Iterable

SOUND_WORLD_VERSION = "otr_source_sound_world_v1"

# When the text offers no recognizable environment, say so plainly. A neutral
# studio-period bed is honest; borrowing the catalog's parlour would be the
# original defect wearing a new hat.
NEUTRAL_PERIOD_SOUND_WORLD = (
    "a plain period recording space -- room tone, a chair, paper, footsteps; "
    "nothing the source does not give"
)

# How many distinct elements a derived line carries. Enough to place a scene,
# few enough to stay a sound world rather than an inventory.
_MAX_ELEMENTS = 5

# Deterministic environment vocabulary. Each row: (element key, detection
# pattern, the sound phrase it contributes). Patterns are word-boundary
# matched against the canonicalized body, case-insensitively. The phrases are
# period-neutral: they describe what the SOURCE names, so a source that names
# a telephone gets a telephone because its author put one there.
_ENVIRONMENT_VOCABULARY: tuple[tuple[str, str, str], ...] = (
    ("sea", r"\b(sea|ocean|tide|surf|harbou?r|ship|deck|sail|wave)s?\b",
     "water against a hull, rigging, gulls, a bell over open water"),
    ("storm", r"\b(storm|thunder|lightning|gale|tempest|rain|downpour)s?\b",
     "rain, wind pressing a window, thunder some way off"),
    ("wind", r"\b(wind|breeze|draught|draft)s?\b",
     "wind in the open, a loose shutter"),
    ("fire", r"\b(fire|hearth|grate|flame|ember|candle|lamp|lantern)s?\b",
     "a fire settling in the grate, a candle guttering"),
    ("clock", r"\b(clock|chime|bell|toll)s?\b",
     "a clock marking the hour, a bell at a distance"),
    ("horse", r"\b(horse|hoof|hooves|carriage|coach|cart|stable)s?\b",
     "hooves on stone, harness, a carriage drawing up"),
    # "train", "engine" and "whistle" are too polysemous to stand alone: a
    # "train of gloomy spectres" put a locomotive in Dumas' 1815 prison, and
    # Long John Silver "giving a little whistle" put one in 18th-century
    # piracy. Only unambiguous railway words qualify.
    ("rail", r"\b(railway|railroad|locomotive|steam engine|platform)s?\b",
     "an engine breathing at a platform, a whistle, iron on iron"),
    ("machine", r"\b(machine|machinery|lever|dial|apparatus|laboratory|workshop)s?\b",
     "small mechanisms, a lever thrown, glass and metal on a bench"),
    ("crowd", r"\b(crowd|street|market|tavern|inn|square|thron[gs])s?\b",
     "voices at a distance, a door, footsteps passing"),
    # "hall" belongs to the HOUSE below and nowhere else. It was in both, and
    # because this row sorts first at an equal position, an entrance hall in
    # a domestic story drew castle audio -- 17 of the 39 corpus works that
    # fired this row had no castle, throne, palace or banquet in them at all.
    ("court", r"\b(court|castle|throne|palace|banquet)s?\b",
     "a great room, footfalls carrying, cloth and iron"),
    ("battle", r"\b(sword|battle|army|soldier|drum|trumpet|banner)s?\b",
     "drums, a trumpet, armour and boots on ground"),
    ("wild", r"\b(forest|wood|moor|heath|field|hill|marsh|river|stream)s?\b",
     "open country -- birds, water moving, grass and branch"),
    # Added 2026-08-04 from a published leg: Twelfth Night 2.5 opens "In the
    # garden... Malvolio's coming down this walk" and derived NOTHING, so a
    # comic garden scene fell back to the neutral bed. Drama scenes are short
    # and stage their location in a line or two -- the vocabulary was
    # novel-shaped and missed the most common comedy setting there is.
    ("garden", r"\b(garden|orchard|arbou?r|bower|hedge|lawn|box tree)s?\b",
     "a garden -- birdsong, leaves, gravel underfoot, a gate"),
    ("night", r"\b(night|midnight|dark|moon|star)s?\b",
     "night quiet, an owl, something small in the dark"),
    ("house", r"\b(parlou?r|drawing[- ]room|study|kitchen|stair|hall(?:way)?|door)s?\b",
     "a house at rest -- a stair, a door, a clock in another room"),
    ("church", r"\b(church|chapel|bell tower|choir|hymn|priest|monk)s?\b",
     "a stone interior, a bell, voices in a vault"),
    ("weather_cold", r"\b(snow|frost|ice|winter|sleet)s?\b",
     "snow underfoot, a wind with an edge on it"),
)


class SourceWorldError(RuntimeError):
    """A sound world could not be derived (loud)."""


def _minimum_hits_for(body: str) -> int:
    """How many mentions an element needs before it counts as a real setting.

    A single mention is usually figurative in a full-length work -- "a storm
    of protest", "a train of gloomy spectres", "the tide of opinion" -- and
    because elements are ordered by FIRST APPEARANCE, one such phrase in an
    opening paragraph could lead the whole sound world. Requiring a second
    mention removes that without special-casing individual verbs.

    Short excerpts keep a threshold of one: in a few hundred words a single
    "he sat by the fire" IS the setting, not a figure of speech.
    """
    return 1 if len(body.split()) < 1000 else 2


def detect_environment_elements(body: str) -> tuple[str, ...]:
    """Element keys the SOURCE genuinely establishes, by first appearance.

    Ordering by position rather than hit count keeps the line loyal to how
    the work opens: a story that begins at sea and ends in a parlour leads
    with the sea. Qualifying by hit COUNT first keeps a passing metaphor from
    taking that lead. Deterministic for a given body.
    """
    if not isinstance(body, str):
        raise SourceWorldError(
            f"source body must be str, got {type(body).__name__}")
    minimum = _minimum_hits_for(body)
    found: list[tuple[int, str]] = []
    for key, pattern, _phrase in _ENVIRONMENT_VOCABULARY:
        matches = list(re.finditer(pattern, body, re.IGNORECASE))
        if len(matches) >= minimum:
            found.append((matches[0].start(), key))
    found.sort()
    return tuple(key for _position, key in found)


def _phrases_for(keys: Iterable[str]) -> list[str]:
    lookup = {key: phrase for key, _pattern, phrase in _ENVIRONMENT_VOCABULARY}
    return [lookup[key] for key in keys if key in lookup]


def derive_source_sound_world(body: str, *, max_elements: int = _MAX_ELEMENTS) -> str:
    """Compose a sound world from what the source actually names.

    Returns ``NEUTRAL_PERIOD_SOUND_WORLD`` when the text names nothing in the
    vocabulary. Never raises on ordinary prose; never returns an empty string,
    because an empty sound world silently re-opens the door for a caller to
    fall back to the drawn catalog palette.
    """
    keys = detect_environment_elements(body)[:max(1, int(max_elements))]
    phrases = _phrases_for(keys)
    if not phrases:
        return NEUTRAL_PERIOD_SOUND_WORLD
    return "; ".join(phrases)


def sound_world_receipt(body: str, *, source_ref: str = "") -> dict:
    """A body-free record of how the sound world was derived.

    Says which elements were detected and whether the neutral default was
    used, so a listener's "why does this episode sound like that" has an
    auditable answer that does not require the source text.
    """
    detected = detect_environment_elements(body)
    kept = detected[:_MAX_ELEMENTS]
    return {
        "sound_world_version": SOUND_WORLD_VERSION,
        "source_ref": source_ref,
        "elements_detected": list(detected),
        "elements_used": list(kept),
        "used_neutral_default": not kept,
        # What it took to qualify, so a reader can tell a short excerpt's
        # single-mention threshold from a full work's stricter one.
        "minimum_hits_required": _minimum_hits_for(body),
        "elements_dropped_by_cap": max(0, len(detected) - len(kept)),
    }
