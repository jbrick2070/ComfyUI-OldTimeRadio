"""The part of a cast description a picture model may read.

The writer authors ONE character description for every consumer, in the
casting format (``_otr_casting.py``)::

    40s, lead. Face: oval, deep-set eyes, ... Distinctive detail: ...
    Presence: carries the weight of ... Voice: warm, low register, ...

``Voice:`` is for TTS, and no picture model can draw a voice. On joint
audio-video lanes it is worse than dead weight: those models read one prompt
for picture AND sound, so voice words get VOCALIZED (measured before the first
fix: ``Voice:`` in 18 of 20 joint-AV prompts). Every picture prompt drops it.

A STILL drops more (2026-10-18, PBUG-20261018-03). FLUX 3 Klein's Qwen3-VL
encoder reads a labelled prose sentence as copy to typeset, and at cfg 1.0 no
negative can veto it: an 8 GB leg lettered "Presence. ..." and "Voice: Warm
... each word a carefully placed stone ..." into two stills of one character.
So a still keeps only what can be drawn -- the age/role lead, the face and the
distinctive detail -- with the ``Label:`` prefixes removed. ``Presence:`` is
mood prose, the same class as the atmosphere sentence the still tail already
dropped. A video prompt keeps ``Presence:``: behaviour is something a video
model can act on.
"""
from __future__ import annotations

import re

__all__ = ["picture_description"]

#: The labels the casting format and its writers use. Only these are stripped,
#: so ordinary prose with a colon ("She whispers: run.") is left alone.
_LABEL_WORDS = (r"(?:Face|Distinctive\s+detail|Detail|Presence|Voice|Hair|"
                r"Eyes|Build|Wardrobe|Clothing|Look|Appearance)")
#: A clause body runs to the end of its sentence, steps over a common
#: abbreviation ("St. Louis", "Dr. Ames"), and stops at the next label even
#: when the writer left out the period.
_BODY = (r"(?:\b(?:Dr|Mr|Mrs|Ms|St|Jr|Sr|Mt|Ft|Lt|Capt|Col|Gen|Prof|Rev|vs|etc)\.|"
         r"(?!\b" + _LABEL_WORDS + r"\s*:)[^.])*"
         r"(?:\.|$|(?=\b" + _LABEL_WORDS + r"\s*:))")
_VOICE = re.compile(r"\bVoice\s*:\s*" + _BODY, re.IGNORECASE)
_PRESENCE = re.compile(r"\bPresence\s*:\s*" + _BODY, re.IGNORECASE)
_LABEL = re.compile(r"\b" + _LABEL_WORDS + r"\s*:\s*", re.IGNORECASE)


def picture_description(text, *, still: bool) -> str:
    """The cast description with what a picture model cannot draw removed.

    ``still=False`` (video) drops the ``Voice:`` clause only. ``still=True``
    also drops ``Presence:`` and the casting labels (``Face:``, ``Distinctive
    detail:`` ...) wherever they sit, keeping the words after them. Never
    raises; an empty or non-string value returns ``""``.
    """
    if not text or not isinstance(text, str):
        return ""
    cleaned = _VOICE.sub(" ", text)
    if still:
        cleaned = _PRESENCE.sub(" ", cleaned)
        cleaned = _LABEL.sub("", cleaned)
    return " ".join(cleaned.split()).strip(" ,")
