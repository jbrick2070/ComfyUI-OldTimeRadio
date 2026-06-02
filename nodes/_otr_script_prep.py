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
