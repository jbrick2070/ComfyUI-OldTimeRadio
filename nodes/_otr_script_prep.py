"""Engine-neutral spoken-text preparation (the downstream per-engine hook).

The canonical script stays engine-neutral; each voice engine adapts a line to
how it "sees" text at render time via its ``prepare_text``. The shared base
here strips stage directions, bracket tags, and a leading speaker label down to
clean spoken words -- the audio direction those carry is preserved separately
in the per-line delivery vector (see ``_otr_delivery_vector``). Deterministic,
no LLM -> C7-safe. An engine that needs heavier rewriting can add an opt-in
LLM doctor pass on top (PD6 applies); the default path stays pure-Python.
"""
from __future__ import annotations

import re

_SPEAKER_PREFIX = re.compile(r"^[A-Z][A-Z .'\-]{1,30}:\s*")
_PAREN = re.compile(r"\([^)]{1,80}\)")
_BRACKET = re.compile(r"\[[^\]]{1,40}\]")
_WS = re.compile(r"\s+")


def clean_spoken_text(text: str) -> str:
    """Strip a leading speaker label, parenthetical stage directions, and
    bracket tags; collapse whitespace. Idempotent and deterministic.
    """
    t = text or ""
    t = _SPEAKER_PREFIX.sub("", t)
    t = _PAREN.sub(" ", t)
    t = _BRACKET.sub(" ", t)
    t = _WS.sub(" ", t).strip()
    return t


PREPARE_TEXT_VERSION = "1"

_ASTERISK = re.compile(r"\*+")
# ♩ ♪ ♫ ♬ + the two musical-note emoji, as ASCII-safe escapes.
_MUSIC_NOTE = re.compile(r"[♩♪♫♬\U0001F3B5\U0001F3B6]+")
_MULTIDOT = re.compile(r"\.{2,}")


def prepare_text(text: str) -> str:
    """Engine-neutral spoken text for a TTS forward (the per-engine hook default).

    Builds on ``clean_spoken_text`` (which strips a leading speaker label,
    parenthetical stage directions, and bracket tags), then additionally removes
    emphasis asterisks and music-note glyphs, normalizes any unicode ellipsis or
    run of dots to a single ``...`` pause, and KEEPS sentence punctuation
    ``. , ?``. Pure + deterministic -> C7-safe; versioned by
    ``PREPARE_TEXT_VERSION`` so it can participate in cache identity / IS_CHANGED.
    """
    t = clean_spoken_text(text)
    t = _ASTERISK.sub(" ", t)
    t = _MUSIC_NOTE.sub(" ", t)
    t = t.replace("…", "...")   # unicode ellipsis -> three dots
    t = _MULTIDOT.sub("...", t)      # any run of 2+ dots -> one pause
    t = _WS.sub(" ", t).strip()
    return t
