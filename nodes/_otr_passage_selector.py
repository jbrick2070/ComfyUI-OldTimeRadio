"""Select a VERBATIM passage from a play-form source, sized to an episode budget.

Operator ruling 2026-08-03, for the fidelity lanes: a play episode is "very
strict -- based on word count and random choice it hones in on a specific part of
a play to get real specific dialogue, no paraphrasing."

So the episode is not a compression of a scene. It is a contiguous WINDOW of
consecutive speeches, carried verbatim, chosen because it fits the operator's
word budget, the cast ceiling and the beat topology. Because the words are the
play's own, there is nothing for a model to drift away from: this is what stops
a Forest-of-Arden scene being narrated as if it were Verona.

FORM, NOT AUTHOR -- but only as far as it is PROVEN. The parser targets the
FOLGER plain-text layout specifically, and that is the only format with vendored
examples under test. Other already-dialogue public-domain sources (Wilde, Ibsen,
Chekhov) are plausible future callers because the selection logic is about form
rather than authorship, but their layouts are NOT verified here and must not be
assumed: a speculative multi-format grammar would widen the prefix match and
weaken the prose refusal below. Add an adapter per real sample, with corpus tests.

Prose sources are a different problem and are NOT served by this module: prose has
no speech prefixes to slice, and pretending otherwise is how a narrator's account
becomes invented character dialogue. The refusal is best-effort on text alone --
uppercase headings can in principle look like speakers -- so a caller that knows
the format should say so rather than relying on detection.

THE CONSTRAINT PEOPLE MISS. A passage is performed against VOICED BEATS, and beats
come from the act topology, not the word count
(``_otr_episode_budget.ACT_COUNT_CONFIG``): 30-120 target words buy exactly THREE
voiced beats, 150-200 buy six, 300-1200 buy fourteen. A long exchange therefore
cannot be performed at 120 words however neatly it fits the word budget, which is
why ``max_beats`` is a required argument rather than a courtesy.

A SPEECH IS NOT ALWAYS ONE BEAT. The Beat schema hard-caps a single voiced beat at
``BEAT_WORD_HARD_MAX`` (80 words), so a long speech spans CONSECUTIVE beats in the
same voice, split at line boundaries. That is pacing, not paraphrase -- the words
are untouched -- and without it the lane silently loses its best material: Banquo's
"Good sir, why do you start" (91 words) sits inside the Macbeth prophecy, and Lear's
love test, Prospero's history and Juliet's balcony speeches all exceed the cap.
Beat cost is therefore ``ceil(words / cap)`` per speech, never a flat one.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass

from ._otr_episode_budget import BEAT_WORD_HARD_MAX
from ._otr_text_metrics import canonical_word_count

__all__ = [
    "PassageError",
    "Speech",
    "Passage",
    "parse_speeches",
    "strip_stage_directions",
    "beats_for_words",
    "eligible_windows",
    "select_passage",
]


class PassageError(RuntimeError):
    """A passage could not be parsed or selected. Never degrades to a guess."""


# Folger writes ALL stage business in square brackets, and a line-level regex is
# not enough: real spans are multiline
#   "[Enter Luce above, unseen by Antipholus of Ephesus\nand his company.]"
# and inline, trailing or mid-sentence
#   "Monsieur Melancholy.\t[Jaques exits.]"   "with him. [As Ganymede.] Do you hear"
# Leaving them in spoken text means TTS reads "Jaques exits" aloud. `[^\]]`
# already matches a newline, so this one pattern removes every form.
_BRACKET_SPAN_RE = re.compile(r"\[[^\]]*\]")

# A speech prefix is an all-caps character name in one of two layouts, never with
# a colon:
#   verse -- the name alone on its line:   ORLANDO
#   prose -- the name inline, two spaces:  TOBY  Come thy ways, Signior Fabian.
# A stage qualifier may follow WITH OR WITHOUT a comma -- "ROSALIND, [as Ganymede]"
# but also "BOTTOM [sings]". Requiring the comma mis-parsed the latter, which
# handed Bottom's song to Titania. Bracket spans are stripped before matching, so
# only the optional trailing comma needs handling here. Single spaces inside the
# name allow FIRST WITCH / ANTIPHOLUS OF EPHESUS.
_SPEECH_RE = re.compile(
    r"^(?P<name>[A-Z][A-Z'’.\-]*(?: [A-Z][A-Z'’.\-]*)*),?"
    r"(?:\s*$|\s{2,}(?=\S))"
)

# All-caps shapes that are structure or performance direction, never speakers.
_NON_SPEAKER_TOKENS = frozenset({
    "ACT", "SCENE", "FINIS", "THE END", "EPILOGUE", "PROLOGUE", "EXIT", "EXEUNT",
})

# Collective turns, not cast identities. "ALL" is a real speaker label in the
# Macbeth text ("ALL, [dancing in a circle]"), but charging it a cast slot and a
# char_id would mint a phantom voice that no TTS speaker can own. Delivering a
# chorus needs its own member/voice-bus contract, so v1 refuses these windows
# rather than pretending.
_COLLECTIVE_SPEAKERS = frozenset({"ALL", "BOTH"})


def beats_for_words(word_count: int, *, beat_word_cap: int = BEAT_WORD_HARD_MAX) -> int:
    """Voiced beats one speech of this length needs.

    The Beat schema rejects a beat over ``BEAT_WORD_HARD_MAX`` words, so a long
    speech is carried across consecutive beats in the same voice rather than
    being cut or rewritten.
    """
    if beat_word_cap <= 0:
        raise PassageError("beat_word_cap must be positive")
    return max(1, math.ceil(max(0, int(word_count)) / beat_word_cap))


@dataclass(frozen=True)
class Speech:
    """One character's uninterrupted turn, exactly as the source has it."""

    index: int
    speaker: str
    text: str

    @property
    def word_count(self) -> int:
        # The canonical ledger counter, not len(split()). Two counters for one
        # quantity is a documented production defect (Bug Bible 12.67); a passage
        # sized by one and validated by the other would drift at the boundary.
        return canonical_word_count(self.text)

    @property
    def is_collective(self) -> bool:
        return self.speaker in _COLLECTIVE_SPEAKERS

    def beat_cost(self, *, beat_word_cap: int = BEAT_WORD_HARD_MAX) -> int:
        return beats_for_words(self.word_count, beat_word_cap=beat_word_cap)


@dataclass(frozen=True)
class Passage:
    """A contiguous run of speeches chosen to be performed verbatim."""

    speeches: tuple[Speech, ...]
    speakers: tuple[str, ...]
    word_count: int
    beat_cost: int
    first_index: int
    last_index: int
    eligible_count: int

    @property
    def speech_count(self) -> int:
        return len(self.speeches)


def strip_stage_directions(source_text: str) -> str:
    """Remove every bracketed stage-direction span, multiline or inline.

    Done over the WHOLE text rather than line by line, because Folger directions
    genuinely span lines and sit mid-sentence. A line-level rule left
    "[Enter Luce above, unseen by Antipholus of Ephesus / and his company.]"
    inside Dromio's spoken text.
    """
    return _BRACKET_SPAN_RE.sub(" ", str(source_text or ""))


def parse_speeches(source_text: str) -> tuple[Speech, ...]:
    """Split play-form text into ordered speeches.

    Stage directions are excluded from spoken text -- they are performance
    instruction, not dialogue, and speaking them aloud is a fidelity defect of
    its own. Their position is implicitly preserved by speech ordering.
    """
    cleaned = strip_stage_directions(source_text)
    speeches: list[tuple[str, list[str]]] = []
    for raw_line in cleaned.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
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
    max_beats: int,
    tolerance: float = 0.25,
    min_speakers: int = 2,
    beat_word_cap: int = BEAT_WORD_HARD_MAX,
) -> tuple[tuple[int, int], ...]:
    """Every contiguous window that fits the word, cast and beat budgets.

    Returned as (first_index, last_index) pairs, inclusive. A window's beat cost
    is the sum of each speech's own cost, so one long speech can consume several
    beats -- see ``beats_for_words``.
    """
    if target_words <= 0:
        raise PassageError("target_words must be positive")
    if cast_ceiling < min_speakers:
        raise PassageError(
            f"cast_ceiling {cast_ceiling} cannot satisfy min_speakers {min_speakers}"
        )
    if max_beats < min_speakers:
        raise PassageError(
            f"max_beats {max_beats} cannot hold {min_speakers} speakers -- "
            f"a passage needs at least one voiced beat per speech"
        )

    low = target_words * (1.0 - tolerance)
    high = target_words * (1.0 + tolerance)
    found: list[tuple[int, int]] = []
    for start in range(len(speeches)):
        words = 0
        beats = 0
        speakers: list[str] = []
        for end in range(start, len(speeches)):
            speech = speeches[end]
            if speech.is_collective:
                # A chorus turn cannot own a cast slot or a char_id; windows
                # containing one are refused rather than given a phantom voice.
                break
            beats += speech.beat_cost(beat_word_cap=beat_word_cap)
            if beats > max_beats:
                break
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
    max_beats: int,
    seed: str,
    tolerance: float = 0.25,
    min_speakers: int = 2,
    beat_word_cap: int = BEAT_WORD_HARD_MAX,
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
        max_beats=max_beats,
        tolerance=tolerance,
        min_speakers=min_speakers,
        beat_word_cap=beat_word_cap,
    )
    if not windows:
        raise PassageError(
            f"no passage of ~{target_words} words fits {max_beats} voiced "
            f"beat(s) and {cast_ceiling} cast slot(s) in a {len(speeches)}-speech "
            f"source. Raise the word budget -- more words buy more beats. Do not "
            f"trim the source to fit and do not loosen the bounds to force a "
            f"match; a passage stretched to fit is no longer the source's own."
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
        beat_cost=sum(s.beat_cost(beat_word_cap=beat_word_cap) for s in chosen),
        first_index=first,
        last_index=last,
        eligible_count=len(windows),
    )
