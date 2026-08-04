"""Select a VERBATIM passage from a play-form source, sized to an episode budget.

Operator ruling 2026-08-03, for the fidelity lanes: a play episode is "very
strict -- based on word count and random choice it hones in on a specific part of
a play to get real specific dialogue, no paraphrasing."

So the episode is not a compression of a scene. It is a contiguous WINDOW of
consecutive speeches, carried verbatim, chosen because it fits the operator's
word budget, the cast ceiling and the beat topology. Because the words are the
play's own, there is nothing for a model to drift away from: this is what stops
a Forest-of-Arden scene being narrated as if it were Verona.

FORM, NOT AUTHOR. Nothing here is Shakespeare-specific. Any public-domain source
that is already dialogue -- Wilde, Ibsen, Chekhov, Sheridan -- parses and selects
through the same path. Prose sources are a different problem and are NOT served
by this module: prose has no speech prefixes to slice, and pretending otherwise
is how a narrator's account becomes invented character dialogue.

THE CONSTRAINT PEOPLE MISS. A verbatim passage is performed one speech per VOICED
BEAT, and beats come from the act topology, not the word count
(``_otr_episode_budget.ACT_COUNT_CONFIG``): 30-120 target words buy exactly THREE
voiced beats, 150-200 buy six, 300-1200 buy fourteen. A seven-speech exchange
therefore cannot be performed at 120 words however neatly it fits the word
budget, which is why ``max_speeches`` is a required argument rather than a
courtesy.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

__all__ = [
    "PassageError",
    "Speech",
    "Passage",
    "parse_speeches",
    "eligible_windows",
    "select_passage",
]


class PassageError(RuntimeError):
    """A passage could not be parsed or selected. Never degrades to a guess."""


# A speech prefix is an all-caps character name, optionally carrying a stage
# qualifier, in one of two layouts and never with a colon:
#   verse -- the name alone on its line:      ORLANDO
#   prose -- the name inline, two spaces:     TOBY  Come thy ways, Signior Fabian.
# Single spaces inside the name allow FIRST WITCH / ANTIPHOLUS OF EPHESUS; a
# double space is what separates the name from the speech it introduces.
_SPEECH_RE = re.compile(
    r"^(?P<name>[A-Z][A-Z'’.\-]*(?: [A-Z][A-Z'’.\-]*)*)"
    r"(?:,\s*\[[^\]]*\])?"
    r"(?:\s*$|\s{2,}(?=\S))"
)

# All-caps shapes that are structure or performance direction, never speakers.
_NON_SPEAKER_TOKENS = frozenset({
    "ACT", "SCENE", "FINIS", "THE END", "EPILOGUE", "PROLOGUE", "EXIT", "EXEUNT",
})

# Folger writes stage business in brackets on its own line.
_STAGE_RE = re.compile(r"^\[.*\]$")


@dataclass(frozen=True)
class Speech:
    """One character's uninterrupted turn, exactly as the source has it."""

    index: int
    speaker: str
    text: str

    @property
    def word_count(self) -> int:
        return len(self.text.split())


@dataclass(frozen=True)
class Passage:
    """A contiguous run of speeches chosen to be performed verbatim."""

    speeches: tuple[Speech, ...]
    speakers: tuple[str, ...]
    word_count: int
    first_index: int
    last_index: int
    eligible_count: int

    @property
    def speech_count(self) -> int:
        return len(self.speeches)


def parse_speeches(source_text: str) -> tuple[Speech, ...]:
    """Split play-form text into ordered speeches.

    Stage directions are excluded from spoken text -- they are performance
    instruction, not dialogue, and speaking them aloud is a fidelity defect of
    its own. Their position is implicitly preserved by speech ordering.
    """
    speeches: list[tuple[str, list[str]]] = []
    for raw_line in str(source_text or "").splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or _STAGE_RE.match(stripped):
            continue
        # A prefix is never indented; indented lines continue the current speech.
        if line == line.lstrip():
            match = _SPEECH_RE.match(line)
            if match is not None and match.group("name") not in _NON_SPEAKER_TOKENS:
                remainder = line[match.end():].strip()
                speeches.append((match.group("name"), [remainder] if remainder else []))
                continue
        if speeches:
            speeches[-1][1].append(stripped)

    out: list[Speech] = []
    for speaker, lines in speeches:
        text = "\n".join(lines).strip()
        if text:
            out.append(Speech(index=len(out), speaker=speaker, text=text))
    return tuple(out)


def eligible_windows(
    speeches: tuple[Speech, ...],
    *,
    target_words: int,
    cast_ceiling: int,
    max_speeches: int,
    tolerance: float = 0.25,
    min_speakers: int = 2,
) -> tuple[tuple[int, int], ...]:
    """Every contiguous window that fits the word, cast and beat budgets.

    Returned as (first_index, last_index) pairs, inclusive.
    """
    if target_words <= 0:
        raise PassageError("target_words must be positive")
    if cast_ceiling < min_speakers:
        raise PassageError(
            f"cast_ceiling {cast_ceiling} cannot satisfy min_speakers {min_speakers}"
        )
    if max_speeches < min_speakers:
        raise PassageError(
            f"max_speeches {max_speeches} cannot hold {min_speakers} speakers -- "
            f"a passage needs one voiced beat per speech"
        )

    low = target_words * (1.0 - tolerance)
    high = target_words * (1.0 + tolerance)
    found: list[tuple[int, int]] = []
    for start in range(len(speeches)):
        words = 0
        speakers: list[str] = []
        for end in range(start, len(speeches)):
            if end - start + 1 > max_speeches:
                break
            speech = speeches[end]
            words += speech.word_count
            if speech.speaker not in speakers:
                speakers.append(speech.speaker)
            if len(speakers) > cast_ceiling or words > high:
                break
            if words >= low and len(speakers) >= min_speakers:
                found.append((start, end))
    return tuple(found)


def select_passage(
    source_text: str,
    *,
    target_words: int,
    cast_ceiling: int,
    max_speeches: int,
    seed: str,
    tolerance: float = 0.25,
    min_speakers: int = 2,
) -> Passage:
    """Choose one verbatim passage. Deterministic for a given seed.

    "Random choice" per the operator ruling, but replayable: the same seed always
    yields the same passage, so an episode can be re-rendered byte-identically
    and a receipt means something.

    Raises PassageError when no window fits, rather than relaxing a constraint --
    a passage that overruns its beats cannot be performed, and a passage stretched
    to fit is no longer the source's own words.
    """
    speeches = parse_speeches(source_text)
    if not speeches:
        raise PassageError(
            "no speeches parsed from the source text -- refusing to select a "
            "passage from something that is not play-form dialogue"
        )
    windows = eligible_windows(
        speeches,
        target_words=target_words,
        cast_ceiling=cast_ceiling,
        max_speeches=max_speeches,
        tolerance=tolerance,
        min_speakers=min_speakers,
    )
    if not windows:
        raise PassageError(
            f"no passage of ~{target_words} words fits {max_speeches} voiced "
            f"beat(s) and {cast_ceiling} cast slot(s) in a {len(speeches)}-speech "
            f"source. Raise the word budget (more words buy more beats) or widen "
            f"the tolerance; do not trim the source to fit."
        )
    digest = hashlib.sha256(str(seed).encode("utf-8")).digest()
    first, last = windows[int.from_bytes(digest[:8], "big") % len(windows)]
    chosen = speeches[first:last + 1]
    speakers: list[str] = []
    for speech in chosen:
        if speech.speaker not in speakers:
            speakers.append(speech.speaker)
    return Passage(
        speeches=chosen,
        speakers=tuple(speakers),
        word_count=sum(s.word_count for s in chosen),
        first_index=first,
        last_index=last,
        eligible_count=len(windows),
    )
