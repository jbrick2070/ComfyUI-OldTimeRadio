"""nodes/_otr_line_hygiene.py -- story-quality Phase 1, A6 (clean delivery).

Two deterministic SCRUBS + one TRUNCATION DETECTOR for spoken lines. The
spine applies these: character defects are scrubbed in place (parenthetical
stage-directions, self-vocative) and TRUNCATION is REPAIRED BY RECOMPOSE
(never token-surgery) -- a character truncation routes to the spine recompose
seam, an announcer open/close truncation routes to the dedicated announcer
composer (the critic excludes announcer lines as locked structural content,
so a character reroll path cannot act on them).

  * scrub_parentheticals -- drop "(she translates a passage)" style stage
    directions, INCLUDING a dangling unclosed "(..." to end of line
    (leg_0013 "(she translates a passage,").
  * scrub_self_vocative -- drop the speaker's OWN cast name when it stands as
    a leading or trailing vocative ("Edna, ..." / "..., Edna."); never an
    in-line occurrence of the name.
  * is_truncated -- detect a mid-sentence cut (trailing comma / conjunction /
    dangling open paren) or a lone-stopword sentence ("The." in leg_0013
    "The. Stay with us.") so the caller can RECOMPOSE it. The music-beat dump
    text is left alone (callers only pass voiced character/announcer lines).

PURE module: stdlib only. Deterministic + idempotent (C7-safe). Never raises
(every function returns the input unchanged on any error / empty result).
UTF-8 no BOM. 4-space indentation.
"""

from __future__ import annotations

import re
from typing import Any

_WS = re.compile(r"\s+")
_PAREN_CLOSED = re.compile(r"\([^)]*\)")
_PAREN_DANGLING = re.compile(r"\([^)]*$")

# Connective / function words that should never END a finished line -- a line
# ending on one of these (with no terminal punctuation) was cut mid-thought.
_TRAIL_WORDS = frozenset({
    "and", "but", "or", "so", "the", "a", "an", "to", "of", "with", "for",
    "as", "that", "which", "who", "her", "his", "their", "its", "my", "your",
    "in", "on", "at", "from", "into", "than", "then", "if", "when", "while",
    "because", "about",
})

# A sentence that is JUST one of these + a period is a truncation stub.
_LONE_STOP = frozenset({
    "the", "a", "an", "and", "but", "or", "so", "then", "of", "to", "with",
    "for", "as", "that", "which", "when", "while", "because", "he", "she",
    "they", "it", "we", "i", "you", "this", "in", "on", "at",
})

_TERMINAL = ".!?\"')]…"  # sentence-final chars (incl. ellipsis)


def scrub_parentheticals(text: Any) -> str:
    """Strip closed and dangling-open parenthetical stage directions.
    Returns the original text if scrubbing would empty the line."""
    try:
        s = str(text or "")
        out = _PAREN_CLOSED.sub(" ", s)
        out = _PAREN_DANGLING.sub(" ", out)
        out = _WS.sub(" ", out).strip()
        return out or s.strip()
    except Exception:  # noqa: BLE001
        return str(text or "")


def scrub_self_vocative(text: Any, speaker_name: Any) -> str:
    """Drop the speaker's OWN name as a leading/trailing vocative only.
    Returns the original text if scrubbing would empty the line."""
    try:
        s = str(text or "")
        name = str(speaker_name or "").strip()
        if not name:
            return s
        variants = {name}
        first = name.split()[0] if name.split() else ""
        if first:
            variants.add(first)
        out = s
        for variant in variants:
            if not variant:
                continue
            v = re.escape(variant)
            # leading vocative: "Name, rest" / "Name: rest" / "Name - rest"
            out = re.sub(rf"^\s*{v}\s*[,:\-]\s+", "", out, flags=re.IGNORECASE)
            # trailing vocative: "rest, Name." / "rest, Name"
            out = re.sub(rf"\s*,\s*{v}\s*([.!?]?)\s*$", r"\1", out,
                         flags=re.IGNORECASE)
        out = _WS.sub(" ", out).strip()
        return out or s.strip()
    except Exception:  # noqa: BLE001
        return str(text or "")


def clean_spoken_character_line(text: Any, speaker_name: Any) -> str:
    """Parenthetical + self-vocative scrub for a spoken character line."""
    return scrub_self_vocative(scrub_parentheticals(text), speaker_name)


def is_truncated(text: Any) -> bool:
    """True when the line looks cut mid-thought (so the caller recomposes).

    Conservative -- only strong signals fire, to avoid recomposing a clean
    line: a dangling unclosed "(", a trailing clause-break char, a trailing
    connective word with no terminal punctuation, or a lone-stopword sentence.
    """
    try:
        s = _WS.sub(" ", str(text or "")).strip()
        if not s:
            return False
        # dangling unclosed parenthesis
        if "(" in s:
            after = s.split("(", 1)[1]
            if ")" not in after:
                return True
        last = s[-1]
        if last in ",;:-—":
            return True
        words = re.findall(r"[A-Za-z']+", s)
        if words and last not in _TERMINAL:
            if words[-1].lower() in _TRAIL_WORDS:
                return True
        # a sentence that is just one stopword + period ("The.")
        for sent in re.split(r"[.!?]+", s):
            toks = re.findall(r"[A-Za-z']+", sent)
            if len(toks) == 1 and toks[0].lower() in _LONE_STOP:
                return True
        return False
    except Exception:  # noqa: BLE001
        return False
