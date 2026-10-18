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

SECTIONS, NOT SENTENCES (third attempt, after a panel). Two earlier cleaners
cut a clause at its first period, and every abbreviation ("St. Louis", "e.g.",
"a.m.") let the rest of a Voice clause back into the picture. The text is now
split at its SECTION HEADERS -- Face, Distinctive detail / Detail, Presence,
Voice -- and a section runs to the next header or the end, with no sentence
parsing at all. Voice is last in the format, so whatever follows it goes with
it. Inside a kept section a still also drops the inline sub-labels (Hair:,
Eyes:, Build: ...) where they open a phrase, so "her look: haunted" is prose
and stays.
"""
from __future__ import annotations

import re

__all__ = ["picture_description"]

#: The casting format's section headers. A header is recognised wherever it
#: stands as a whole word before a colon; the format never uses these words
#: with a colon in prose.
_SECTION = re.compile(
    r"\b(Face|Distinctive\s+detail|Detail|Presence|Demeanou?r|Manner|"
    r"Voice\s+cue|Vocal\s+quality|Voice|Speech|Tone)\s*[:：]\s*",
    re.IGNORECASE)
#: Header words that mean the voice section, or the presence section, under
#: another name (Sonnet QA on 04d1f6e2: the casting prompt asks for
#: Voice/Presence, but a writer can drift). A full-width colon counts too.
_VOICE_HEADERS = frozenset({"voice", "voice cue", "vocal quality", "speech", "tone"})
_PRESENCE_HEADERS = frozenset({"presence", "demeanor", "demeanour", "manner"})
#: Sub-labels some writers put inside the Face section. Stripped from a still
#: only where they open a phrase (start, or after , ; .), so prose keeps them.
_SUBLABEL = re.compile(
    r"(^|[,;.]\s*)(?:Hair|Eyes|Build|Wardrobe|Clothing|Look|Appearance)\s*[:：]\s*",
    re.IGNORECASE)


def _sections(text: str):
    """``[(header_or_None, label_text, body), ...]`` in order. The lead before
    the first header comes back with header ``None``."""
    out = []
    last_end, last_header, last_label = 0, None, ""
    for m in _SECTION.finditer(text):
        out.append((last_header, last_label, text[last_end:m.start()]))
        last_header = " ".join(m.group(1).lower().split())
        last_label, last_end = m.group(0), m.end()
    out.append((last_header, last_label, text[last_end:]))
    return out


def picture_description(text, *, still: bool) -> str:
    """The cast description with what a picture model cannot draw removed.

    ``still=False`` (video) drops the Voice section only and keeps the rest as
    written. ``still=True`` also drops Presence and every header and opening
    sub-label, keeping the words after them. Never raises; an empty or
    non-string value returns ``""``.
    """
    if not text or not isinstance(text, str):
        return ""
    kept = []
    for header, label, body in _sections(text):
        if header in _VOICE_HEADERS or (still and header in _PRESENCE_HEADERS):
            continue
        if still:
            kept.append(_SUBLABEL.sub(r"\1", body))
        else:
            kept.append(label + body)
    return " ".join(" ".join(kept).split()).strip(" ,")
