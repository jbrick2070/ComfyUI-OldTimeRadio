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
FOLGER plain-text layout and, since 2026-09-18, the COLON layout of the vendored
translations (``NAME: speech``, one speech per line -- see ``_COLON_SPEECH_RE``);
those are the only two formats with vendored examples under test. Other
already-dialogue public-domain sources (Wilde, Ibsen, Chekhov) are plausible
future callers because the selection logic is about form rather than authorship,
but their layouts are NOT verified here and must not be assumed: a speculative
multi-format grammar would widen the prefix match and weaken the prose refusal
below. Add an adapter per real sample, with corpus tests -- and gate it the way
the colon layout is gated (``detect_layout``), so the English corpus is provably
untouched by it.

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
Beat cost is therefore the number of line-packed chunks a speech needs
(``chunk_speech``), never a flat one -- and the CHUNKER is the one owner of
that number, so selection can never promise a plan execution cannot cut.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Mapping

from ._otr_episode_budget import BEAT_WORD_HARD_MAX
from ._otr_text_metrics import CJK_RUN_RE, canonical_word_count

__all__ = [
    "PassageError",
    "Speech",
    "SpeakerBinding",
    "Passage",
    "FOLGER_LAYOUT",
    "COLON_LAYOUT",
    "detect_layout",
    "parse_speeches",
    "strip_stage_directions",
    "chunk_speech",
    "BeatPlanEntry",
    "build_beat_plan",
    "render_passage_text",
    "CHUNKER_VERSION",
    "SELECTOR_VERSION",
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

# THE THIRD LAYOUT (2026-09-18): the vendored translations under
# config/source_banks/shakespeare/translations write `NAME: speech`, one speech
# per line, the name at column 0. Rusconi, Hugo and Marquez label people the
# ASCII rule above cannot see -- accented (CORDÉLIA, RÉGANE) and ordinal-led
# (1A STREGA, 2A STREGA, 3A STREGA) -- so the name class here is "starts with a
# letter or digit, runs to the first colon" and the all-caps test is made AFTER
# the match (`_is_upper_label`), where `str.islower` knows every alphabet.
# Measured before this existed: Hugo's Hamlet and Lear parsed to ZERO speeches,
# Rusconi's Macbeth and Marquez's As You Like It to ONE (the scene heading,
# swallowing the whole scene as its text), so `plan_verbatim_passage` reported
# unavailable and every native-language Shakespeare row fell back to the model
# translation in silence.
_COLON_SPEECH_RE = re.compile(r"^(?P<name>[^\W_][^:\n]{0,79}?):(?:\s+|$)")

# Heading words the colon editions carry, checked against the FIRST word of a
# colon-layout name only. `SCENA III.` / `ESCENA II.` / `SCÈNE I.` carry no
# colon and never match; this is for the edition that writes `ESCENA II: El
# bosque`. Never applied to the Folger layout, so no English file can change.
_HEADING_WORDS = frozenset({
    "ACT", "ACTE", "ATTO", "ACTO", "AKT",
    "SCENE", "SCÈNE", "SCENA", "ESCENA", "SZENE",
})

#: The two layouts `detect_layout` can name. A document has ONE layout.
FOLGER_LAYOUT = "folger"
COLON_LAYOUT = "colon"

# Collective turns, not cast identities. "ALL" is a real speaker label in the
# Macbeth text ("ALL, [dancing in a circle]"), but charging it a cast slot and a
# char_id would mint a phantom voice that no TTS speaker can own. Delivering a
# chorus needs its own member/voice-bus contract, so v1 refuses these windows
# rather than pretending.
_COLLECTIVE_SPEAKERS = frozenset({"ALL", "BOTH"})


# Bump when window SELECTION for an unchanged source and seed would change: a
# stored receipt's indices then name different speeches.
SELECTOR_VERSION = "otr_passage_selector_v1"

# Bump when chunk BOUNDARIES for an unchanged speech would move: a stored plan
# receipt names (speech_index, chunk_ordinal) pairs, so a v1 receipt is not
# comparable to a v2 one.
CHUNKER_VERSION = "otr_verbatim_chunker_v1"


def _line_units(text: str) -> list[str]:
    # One space between words inside a line: a stripped stage direction leaves
    # a run of spaces behind ("behind.   Thanks"), and a ledger line is not the
    # place to carry the parser's scars. Words untouched.
    #
    # AN IDEOGRAPHIC SPACE IS PART OF THE EDITION, NOT A PARSER'S SCAR.
    # `str.split()` treats U+3000 as whitespace, so this line turned every
    # one in a Japanese speech into an ASCII space before any cut was made
    # (agy located the mutation in `_line_tokens`; it was here, one function
    # earlier). A line that carries CJK collapses only ASCII spaces and tabs.
    units = []
    for ln in str(text or "").split("\n"):
        if not ln.strip():
            continue
        if CJK_RUN_RE.search(ln):
            units.append(_ASCII_WS.sub(" ", ln).strip(" \t\r"))
        else:
            units.append(" ".join(ln.split()))
    return units


#: A CJK LINE HAS NO SPACES TO CUT AT, SO IT IS CUT AT ITS PUNCTUATION. Every
#: cut in this module fell at a space -- `line.split()` -- which a Japanese or
#: Chinese speech never has. Tsubouchi's balcony scene planned or did not plan
#: depending on the seed: a window that drew Juliet's 180-word speech hit
#: `_halve`, which found one "word", could not cut it, and the whole vendored
#: translation fell back to the model's. Sentence marks first, then clause
#: marks, then -- only if a line has neither -- the characters themselves,
#: which at two characters per word is at worst a cut inside one word. The
#: pieces re-join with NOTHING between them, so the line's bytes survive
#: exactly; a space would be the one thing the edition never printed. A line
#: with no CJK in it never takes this path and cuts exactly as it always did.
#: A CLOSING QUOTE OR BRACKET STAYS WITH ITS SENTENCE. `「あゝ、ロミオ！」` cut
#: straight after the mark left `」` to open the next beat (agy, on
#: 96346118); the cut waits for the closing mark when one follows.
#: ESCAPED, BECAUSE THE FIRST CUT WAS NOT. The closers include `]`, and an
#: unescaped `]` inside a character class closes the class, so the lookahead
#: read "not followed by a closer AND a literal ]" -- never true -- and the
#: closer-keep never kept anything; its test was green only because the
#: packer's boundaries happened to fall on opening quotes (Cursor, on
#: f1f6a840). `re.escape` makes `]` and `)` literal inside the class.
#: CHUNKER_VERSION stays v1 on purpose: English boundaries are unchanged
#: (asserted), and no CJK plan ever succeeded before 96346118, so no stored
#: receipt names CJK chunk pairs that could move.
_CJK_CLOSERS = re.escape("」』”’）)]")
_CJK_SENTENCE_CUT = re.compile(
    r"(?<=[。！？!?])(?![" + _CJK_CLOSERS + r"])|(?<=[。！？!?][" + _CJK_CLOSERS + r"])")
_CJK_CLAUSE_CUT = re.compile(r"(?<=[、，；：;:])(?![" + _CJK_CLOSERS + r"])")
#: THE IDEOGRAPHIC SPACE IS PART OF THE EDITION, NOT A WORD BOUNDARY.
#: `str.split()` treats U+3000 as whitespace, so a Japanese line that
#: carried one took the whitespace path and came back re-joined with an
#: ASCII space -- 57 such lines in the vendored CJK texts (agy). Only an
#: ASCII space or tab routes a CJK line to the whitespace path.
_ASCII_WS = re.compile(r"[ \t]+")


def _line_tokens(line: str) -> tuple[list[str], str]:
    """``(pieces, joiner)``: what a line may be cut into, and what re-joins it."""
    if not CJK_RUN_RE.search(line) or len(_ASCII_WS.split(line.strip())) > 1:
        return line.split(), " "
    for cut in (_CJK_SENTENCE_CUT, _CJK_CLAUSE_CUT):
        pieces = [p for p in cut.split(line) if p]
        if len(pieces) > 1:
            return pieces, ""
    return list(line), ""


def _split_long_line(line: str, cap: int) -> list[str]:
    """A single line over the cap splits at word boundaries. Every word
    survives, in order. None in the Folger corpus; the colon-layout
    translations write a whole speech on one line, so their long speeches
    (Banquo's "Qual distanza v'ha ancora" is one) take this path."""
    pieces: list[str] = []
    piece: list[str] = []
    tokens, joiner = _line_tokens(line)
    for token in tokens:
        # CLOSE THE PIECE BEFORE THE TOKEN THAT WOULD OVERSHOOT, not after.
        # With one-word tokens the two are the same cut; with a CJK token
        # that is a whole sentence, closing afterwards handed a beat more
        # words than the cap the chunker promises never to exceed.
        if piece and canonical_word_count(joiner.join(piece + [token])) > cap:
            pieces.append(joiner.join(piece))
            piece = []
        if (not piece and canonical_word_count(token) > cap
                and len(_line_tokens(token)[0]) > 1):
            # ONE SENTENCE LONGER THAN A BEAT is cut again at its clause
            # marks, and a clause longer than a beat at its characters.
            pieces.extend(_split_long_line(token, cap))
            continue
        piece.append(token)
    if piece:
        pieces.append(joiner.join(piece))
    return pieces


def _pack_lines(units: list[str], cap: int) -> list[list[str]]:
    """Greedy whole-line packing into groups of at most ``cap`` words, in order.
    A single line over the cap becomes word-boundary pieces, one per group. This
    is the one packing rule; ``chunk_speech`` and the beat plan both cut from it,
    so selection can never cost a speech differently from how execution cuts it."""
    if cap <= 0:
        raise PassageError("cap must be positive")
    groups: list[list[str]] = []
    current: list[str] = []
    current_words = 0
    for line in units:
        words = canonical_word_count(line)
        if words > cap:
            if current:
                groups.append(current)
                current, current_words = [], 0
            groups.extend([piece] for piece in _split_long_line(line, cap))
            continue
        if current and current_words + words > cap:
            groups.append(current)
            current, current_words = [line], words
        else:
            current.append(line)
            current_words += words
    if current:
        groups.append(current)
    return groups


def chunk_speech(text: str, *, cap: int = BEAT_WORD_HARD_MAX) -> tuple[str, ...]:
    """Cut one speech into consecutive chunks of at most ``cap`` words, packing
    WHOLE LINES greedily in order.

    This is the ONE owner of "how many beats does a speech need": ``Speech.
    beat_cost`` calls it, so a selected passage can always be executed exactly
    as it was costed. The old ``ceil(words / cap)`` estimate disagreed with real
    line packing on one corpus speech (BENEDICK, Much Ado 2.3, 309 words: 4 vs
    5) -- an estimate the executor could not honour is a paraphrase waiting to
    happen. A chunk's lines are joined with ONE SPACE: the words are verbatim;
    the newline is a TTS segmentation choice (Kokoro splits synthesis on it),
    a transcript-line choice and a caption-wrap choice, none of them the
    play's. Never returns an empty chunk.
    """
    return tuple(" ".join(g) for g in _pack_lines(_line_units(text), cap))


@dataclass(frozen=True)
class SpeakerBinding:
    """How one edition label is carried through the episode.

    ``spoken`` is the name the episode CALLS the character -- the cast row, the
    captions, the credits -- in the translation's own language and in the same
    upper-case shape the Folger prefixes arrive in (``lock_cast`` upper-cases
    every source name anyway). ``roster`` is the ENGLISH sidecar name the gender
    ladder resolves instead of the spoken one: ``_otr_roster_gender`` is
    language-internal by construction, so ``1A STREGA`` cannot reach ``FIRST
    WITCH`` on any rung without this bridge. A ``roster`` naming a collective
    marker (``ALL``, ``BOTH``) makes the speech collective, exactly as the
    Folger ``ALL`` label is.
    """

    spoken: str
    roster: str


@dataclass(frozen=True)
class Speech:
    """One character's uninterrupted turn, exactly as the source has it.

    ``label`` is the prefix as the PAGE writes it and ``roster_name`` the English
    sidecar name it was bound to; both are empty on an unbound speech, where the
    speaker IS the page label. Defaults keep every existing construction site
    and the English parse byte-identical.
    """

    index: int
    speaker: str
    text: str
    label: str = ""
    roster_name: str = ""

    @property
    def page_label(self) -> str:
        return self.label or self.speaker

    @property
    def word_count(self) -> int:
        # The canonical ledger counter, not len(split()). Two counters for one
        # quantity is a documented production defect (Bug Bible 12.67); a passage
        # sized by one and validated by the other would drift at the boundary.
        return canonical_word_count(self.text)

    @property
    def is_collective(self) -> bool:
        # A bound speech is collective by its ROSTER identity: Rusconi's "TUTTE
        # LE STREGHE CANTANDO E DANZANDO" is Folger's "ALL, [dancing in a
        # circle]" with the stage direction fused into the label, and it must be
        # refused the same way -- a chorus cannot own a cast slot.
        return (self.roster_name or self.speaker) in _COLLECTIVE_SPEAKERS

    def beat_cost(self, *, beat_word_cap: int = BEAT_WORD_HARD_MAX) -> int:
        # The chunker is the owner: a speech costs exactly the beats the
        # executor will cut it into at this cap, never an estimate of them.
        return max(1, len(chunk_speech(self.text, cap=beat_word_cap)))


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
    # Labels anywhere in the SOURCE (not only this window) that the supplied
    # bindings did not name. Always empty when no bindings were supplied. An
    # unbound label is carried as written and degrades to the label-as-name
    # gender join; this is where a reader finds out that it did.
    unbound_labels: tuple[str, ...] = ()

    @property
    def speech_count(self) -> int:
        return len(self.speeches)

    @property
    def roster_names(self) -> dict:
        """spoken name -> the English roster name it was bound to (bound
        speakers only, in speaking order)."""
        out: dict = {}
        for speech in self.speeches:
            if speech.roster_name and speech.speaker not in out:
                out[speech.speaker] = speech.roster_name
        return out


def strip_stage_directions(source_text: str) -> str:
    """Remove every bracketed stage-direction span, multiline or inline.

    Done over the WHOLE text rather than line by line, because Folger directions
    genuinely span lines and sit mid-sentence. A line-level rule left
    "[Enter Luce above, unseen by Antipholus of Ephesus / and his company.]"
    inside Dromio's spoken text.
    """
    return _BRACKET_SPAN_RE.sub(" ", str(source_text or ""))


def _folger_prefix(line: str) -> tuple[str, str] | None:
    """``(name, remainder)`` when an unindented line opens a Folger speech."""
    match = _SPEECH_RE.match(line)
    if match is None or match.group("name") in _NON_SPEAKER_TOKENS:
        return None
    return match.group("name"), line[match.end():].strip()


#: AN ORDINAL INDICATOR IS CASE-NEUTRAL TYPOGRAPHY, NOT A LOWERCASE LETTER.
#: `ª` (U+00AA) and `º` (U+00BA) are how Spanish and Italian print a feminine or
#: masculine ordinal, and Python calls both `islower()` -- so `1ª FAT` and
#: `BRUJA 1.ª` failed the all-caps test and the runtime selector did not see
#: them as speakers at all.
#:
#: THE FILES WERE RIGHT AND THE CONSUMER COULD NOT READ THEM, which is the worst
#: shape this defect comes in. The vendored text carried the four fairies and
#: the three witches correctly labelled, the manifest bound every one of them to
#: a roster name, the corpus tests passed -- and `parse_speeches` skipped the
#: lines, so at RUNTIME each of those speeches merged into whoever spoke before
#: it. Rusconi's Macbeth escaped only because its transcriber wrote the ordinal
#: as `<sup>a</sup>`, which strips to a plain `A`; the same translator's
#: Midsummer prints the bare `ª` and broke.
#:
#: Found by a corpus-integrity check asking a question no test asked: does every
#: speaker_map KEY match a label the parser actually returns? A key that matches
#: nothing is silent -- it reads as a deliberately unbound label, which is a
#: legitimate outcome here.
_ORDINAL_INDICATORS = "ªº"


def _is_upper_label(name: str) -> bool:
    # At least one letter, and no lowercase one in ANY alphabet. `[A-Z]` in
    # `_SPEECH_RE` is what made CORDÉLIA invisible; this is the test that
    # regex could not express.
    letters = [ch for ch in name if ch not in _ORDINAL_INDICATORS]
    return (any(ch.isalpha() for ch in letters)
            and not any(ch.islower() for ch in letters))


def _colon_prefix(line: str) -> tuple[str, str] | None:
    """``(name, remainder)`` when an unindented line opens a colon-layout speech."""
    match = _COLON_SPEECH_RE.match(line)
    if match is None:
        return None
    name = " ".join(match.group("name").split())
    if not _is_upper_label(name) or name in _NON_SPEAKER_TOKENS:
        return None
    if name.split()[0].rstrip(".") in _HEADING_WORDS:
        return None
    return name, line[match.end():].strip()


def _detect_layout(cleaned: str) -> str:
    """Which of the two grammars this (direction-stripped) document is in.

    THE GATE THAT KEEPS THE ENGLISH CORPUS BYTE-IDENTICAL. A colon layout is
    NOT admitted line by line beside the Folger one: measured across all 81
    files under config/source_banks/*/sources, exactly ONE column-0 `ALLCAPS:`
    line exists -- `STAVE I:  MARLEY'S GHOST`, a chapter heading in the prose
    Christmas Carol -- and a per-line rule would have minted it as a speaker.
    So the layout is decided ONCE per document, and the colon layout needs
    what a dialogue needs and a heading never has: TWO DISTINCT labelled
    voices, outnumbering the Folger prefixes. `tests/test_passage_selector.py`
    pins the measurement and re-parses every English file both ways.
    """
    folger = 0
    colon_names: list[str] = []
    for raw_line in cleaned.splitlines():
        line = raw_line.rstrip()
        if not line or line != line.lstrip():
            continue
        if _folger_prefix(line) is not None:
            folger += 1
        hit = _colon_prefix(line)
        if hit is not None:
            colon_names.append(hit[0])
    if len(set(colon_names)) >= 2 and len(colon_names) > folger:
        return COLON_LAYOUT
    return FOLGER_LAYOUT


def detect_layout(source_text: str) -> str:
    """``FOLGER_LAYOUT`` or ``COLON_LAYOUT`` for a raw source text."""
    return _detect_layout(strip_stage_directions(source_text))


def _bind(speech: Speech, bindings) -> Speech:
    binding = bindings.get(speech.speaker) if bindings else None
    if binding is None:
        # Unbound: the page label is carried as written. Not an error here --
        # the Passage lists it under `unbound_labels` so the receipt says so.
        return speech
    return Speech(
        index=speech.index,
        speaker=str(binding.spoken or "").strip() or speech.speaker,
        text=speech.text,
        label=speech.speaker,
        roster_name=str(binding.roster or "").strip(),
    )


def parse_speeches(
    source_text: str,
    *,
    speaker_bindings: "Mapping[str, SpeakerBinding] | None" = None,
) -> tuple[Speech, ...]:
    """Split play-form text into ordered speeches.

    Stage directions are excluded from spoken text -- they are performance
    instruction, not dialogue, and speaking them aloud is a fidelity defect of
    its own. Their position is implicitly preserved by speech ordering.

    ``speaker_bindings`` (page label -> ``SpeakerBinding``) renames each bound
    speech to its spoken name and records its roster name; it is applied HERE,
    before any window is costed, so ``Speech.is_collective`` already sees the
    roster identity when ``eligible_windows`` refuses a chorus. ``None`` is the
    English path and touches nothing.
    """
    cleaned = strip_stage_directions(source_text)
    prefix_of = (
        _colon_prefix if _detect_layout(cleaned) == COLON_LAYOUT else _folger_prefix
    )
    speeches: list[tuple[str, list[str]]] = []
    for raw_line in cleaned.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        # A prefix is never indented; indented lines continue the current speech.
        if line == line.lstrip():
            hit = prefix_of(line)
            if hit is not None:
                name, remainder = hit
                speeches.append((name, [remainder] if remainder else []))
                continue
        if speeches:
            speeches[-1][1].append(stripped)

    out: list[Speech] = []
    for speaker, lines in speeches:
        text = "\n".join(lines).strip()
        if text:
            out.append(_bind(Speech(index=len(out), speaker=speaker, text=text),
                             speaker_bindings))
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
    min_words: int | None = None,
    max_words: int | None = None,
) -> tuple[tuple[int, int], ...]:
    """Every contiguous window that fits the word, cast and beat budgets.

    Returned as (first_index, last_index) pairs, inclusive. A window's beat cost
    is the sum of each speech's own cost, so one long speech can consume several
    beats -- see ``chunk_speech``.

    ``min_words`` / ``max_words`` override the tolerance band with explicit
    inclusive bounds; ``max_words=None`` with ``min_words=0`` means "words do not
    constrain this search at all", which is how the caller falls back when the
    requested length cannot be met. Beats and cast are never relaxed -- those are
    physical limits, not preferences.
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

    low = target_words * (1.0 - tolerance) if min_words is None else float(min_words)
    high = (
        target_words * (1.0 + tolerance)
        if max_words is None and min_words is None
        else (float("inf") if max_words is None else float(max_words))
    )
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
    speaker_bindings: "Mapping[str, SpeakerBinding] | None" = None,
) -> Passage:
    """Choose one verbatim passage. Deterministic for a given seed.

    "Random choice" per the operator ruling, but replayable: the same seed always
    yields the same passage, so an episode can be re-rendered byte-identically
    and a receipt means something.

    Raises PassageError when no window fits, rather than relaxing a constraint --
    a passage that overruns its beats cannot be performed, and a passage stretched
    to fit is no longer the source's own words.

    ``speaker_bindings`` is handed straight to ``parse_speeches``; see there.
    """
    speeches = parse_speeches(source_text, speaker_bindings=speaker_bindings)
    unbound: tuple[str, ...] = ()
    # The same sentinel as the lane's: `{}` means a vendored row with no map,
    # and it must report EVERY label as unbound. Fixing only the lane leaves an
    # `unbound_labels` key that is always empty, which never trips the writer's
    # warning -- so both layers test `is not None` or neither does.
    if speaker_bindings is not None:
        seen: list[str] = []
        for speech in speeches:
            if not speech.roster_name and speech.speaker not in seen:
                seen.append(speech.speaker)
        unbound = tuple(seen)
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
        # The word target is a REQUEST, not a gate (operator ruling): never refuse
        # a render because nothing landed inside the preferred band. Fall back to
        # the windows that satisfy the HARD limits -- beats and cast, which are
        # physical -- and take the one closest to the request. The passage is
        # still verbatim; only its length drifts from what was asked.
        windows = eligible_windows(
            speeches,
            target_words=target_words,
            cast_ceiling=cast_ceiling,
            max_beats=max_beats,
            min_speakers=min_speakers,
            beat_word_cap=beat_word_cap,
            min_words=0,
            max_words=None,
        )
        if windows:
            def _distance(window: tuple[int, int]) -> tuple[int, int]:
                first, last = window
                words = sum(s.word_count for s in speeches[first:last + 1])
                return (abs(words - target_words), first)

            best = min(_distance(w) for w in windows)[0]
            windows = tuple(w for w in windows if _distance(w)[0] == best)
    if not windows:
        # Nothing at all is performable: the source has no two-speaker exchange
        # that fits even one beat budget. That is a broken source, not a budget
        # disagreement, so it still raises.
        raise PassageError(
            f"no performable passage in a {len(speeches)}-speech source at "
            f"{max_beats} voiced beat(s) and {cast_ceiling} cast slot(s) -- no "
            f"window holds {min_speakers} speakers within the beat budget."
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
        unbound_labels=unbound,
    )


# ---------------------------------------------------------------------------
# Execution: the beat plan (cap-to-fill) and the passage renderer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BeatPlanEntry:
    """One voiced beat of a verbatim passage: which speech, which cut, the words."""

    speaker: str
    text: str            # this chunk's lines joined with one space; every word verbatim
    speech_index: int    # index into ``Passage.speeches``
    chunk_ordinal: int   # 0-based position of this chunk within its speech
    chunk_count: int     # how many chunks the speech was cut into


def _halve(group: list[str]) -> tuple[list[str], list[str]]:
    """Split one group at the line boundary nearest its middle (by words); a
    single line splits at its middle word. Either half may be empty only when
    there is nothing to split, which the caller treats as a miss."""
    if len(group) >= 2:
        weights = [max(1, canonical_word_count(u)) for u in group]
        total = sum(weights)
        cum = 0
        best, best_gap = 1, None
        for i in range(1, len(group)):
            cum += weights[i - 1]
            gap = abs(cum - total / 2)
            if best_gap is None or gap < best_gap:
                best, best_gap = i, gap
        return group[:best], group[best:]
    # A SPACELESS LINE HALVES AT ITS PUNCTUATION, not at a space it has not
    # got: `_line_tokens` hands back the pieces and the joiner that restores
    # the line's own bytes. An English line halves at its middle word as before.
    words, joiner = _line_tokens(group[0]) if group else ([], " ")
    mid = len(words) // 2
    return ([joiner.join(words[:mid])] if mid else []), ([joiner.join(words[mid:])] if words[mid:] else [])


def _cut_into(text: str, parts: int, cap: int) -> list[str]:
    """Cut a speech into exactly ``parts`` pieces, starting from the chunker's
    own groups (each already <= cap) and halving the longest piece at a line
    boundary until the count is reached. Halving only shrinks, so no piece can
    ever exceed the cap (codex r3: a 140-word line could before)."""
    groups = _pack_lines(_line_units(text), cap)
    if len(groups) > parts:
        raise PassageError(
            f"a speech that packs to {len(groups)} chunk(s) cannot fit {parts} beat(s)"
        )
    while len(groups) < parts:
        i = max(range(len(groups)),
                key=lambda k: (canonical_word_count(" ".join(groups[k])), -k))
        left, right = _halve(groups[i])
        if not left or not right:
            raise PassageError(
                f"a {canonical_word_count(' '.join(groups[i]))}-word piece cannot be "
                f"cut again to fill {parts} beats"
            )
        groups[i:i + 1] = [left, right]
    return [" ".join(g) for g in groups]


def build_beat_plan(passage: Passage, *, beat_count: int) -> tuple[BeatPlanEntry, ...]:
    """Cut the passage into EXACTLY ``beat_count`` consecutive voiced beats.

    The act topology is the operator's dial and stays the one topology owner
    (``_otr_episode_budget``): the executor never derives an act count from the
    passage. Instead the passage FILLS the beats the dial bought -- every speech
    takes at least one beat, and the remaining beats go, one at a time, to the
    speech whose beats are currently the longest, so a long speech spans several
    consecutive beats in the same voice (pacing, not paraphrase) while a rapid
    exchange keeps one beat per speech. There is no per-beat word floor anywhere
    in the tree, so a short chunk is legal.

    Raises ``PassageError`` when the passage cannot fill the beats -- fewer beats
    than speeches (selection never produces this: a speech costs at least one
    beat and the passage was chosen inside the beat budget), or fewer words than
    beats (no vendored scene at any legal dial; a caller treats it as a miss).
    Every entry is at most ``BEAT_WORD_HARD_MAX`` words: the plan cuts from the
    chunker's own groups and only ever halves them.
    """
    speeches = passage.speeches
    n = len(speeches)
    if n == 0:
        raise PassageError("cannot plan beats for an empty passage")
    if beat_count < n:
        raise PassageError(
            f"{beat_count} beat(s) cannot hold {n} speech(es) -- a speech never "
            f"shares a beat"
        )
    # Start every speech at the beats the chunker already needs for it (the
    # cost selection used), then hand out the remaining beats.
    counts = [len(_pack_lines(_line_units(s.text), BEAT_WORD_HARD_MAX))
              for s in speeches]
    words = [max(0, s.word_count) for s in speeches]
    if sum(counts) > beat_count:
        raise PassageError(
            f"{beat_count} beat(s) cannot hold a passage that packs to "
            f"{sum(counts)} chunk(s)"
        )
    for _ in range(beat_count - sum(counts)):
        # Only a speech with more words than beats can take another cut; ties go
        # to the earlier speech so the allocation is a pure function of the text.
        candidates = [k for k in range(n) if words[k] > counts[k]]
        if not candidates:
            raise PassageError(
                f"a {sum(words)}-word passage cannot fill {beat_count} beats"
            )
        k = max(candidates, key=lambda i: (words[i] / counts[i], -i))
        counts[k] += 1
    entries: list[BeatPlanEntry] = []
    for index, (speech, k) in enumerate(zip(speeches, counts)):
        pieces = _cut_into(speech.text, k, BEAT_WORD_HARD_MAX)
        for ordinal, piece in enumerate(pieces):
            entries.append(BeatPlanEntry(
                speaker=speech.speaker, text=piece, speech_index=index,
                chunk_ordinal=ordinal, chunk_count=len(pieces),
            ))
    if len(entries) != beat_count:  # pragma: no cover -- arithmetic guard
        raise PassageError(
            f"planned {len(entries)} beat(s) for a request of {beat_count}"
        )
    return tuple(entries)


def render_passage_text(passage: Passage) -> str:
    """The selected window in the parser's own layout, for the pre-outline
    authors that read the lane's ``full_text``. Words verbatim; the layout is
    one the parser reads, so the result parses back into the same speeches.

    An unbound passage renders in Folger's verse layout -- the speaker prefix
    on its own line, the speech beneath -- byte-identical to before. A BOUND
    passage renders in the colon layout, because a spoken name like CORDÉLIA
    is not a Folger prefix (`[A-Z]`) and the verse layout would glue her lines
    to the previous speaker on any re-parse.
    """
    if any(s.roster_name for s in passage.speeches):
        return "\n\n".join(f"{s.speaker}: {s.text}" for s in passage.speeches)
    return "\n\n".join(f"{s.speaker}\n{s.text}" for s in passage.speeches)
