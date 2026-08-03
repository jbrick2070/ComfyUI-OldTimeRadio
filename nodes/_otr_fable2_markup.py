"""Structural parser for the scifi_news_pro whole-play markup.

Python validates only delimiter order, nonempty fields, scene numbering, and
closed-roster identity. Ordinary authored casing, Unicode names, punctuation,
quotes, parentheses, brackets, line count, and coda style are preserved and
never influence acceptance. Transport whitespace around rows and delimiters is
not part of spoken prose.

The parser never repairs or rewrites a spoken word. A structurally malformed
artifact may receive a bounded same-story format repair upstream.
"""
from __future__ import annotations

import enum
import re
from dataclasses import dataclass

try:
    from ._otr_text_metrics import canonical_word_count
except ImportError:  # pragma: no cover -- flat test/standalone load
    from _otr_text_metrics import canonical_word_count  # type: ignore

__all__ = [
    "Fable2ParseDefect",
    "ParseDefect",
    "ParsedLine",
    "ParsedScene",
    "ParsedScript",
    "normalize_fable2_markup_text",
    "parse_fable2_markup",
    "render_defects",
]

ANNOUNCER_NAME = "ANNOUNCER"

# --- line classifiers (structural delimiters; first match wins) -------------
_RE_TITLE = re.compile(r"^TITLE:\s*(.+)$", re.IGNORECASE)
_RE_MUSIC = re.compile(r"^MUSIC:\s*(.+)$", re.IGNORECASE)
_RE_SCENE = re.compile(r"^SCENE\s+(\d{1,2}):\s*(.+)$", re.IGNORECASE)
_RE_CODA = re.compile(r"^CODA:\s*(.+)$", re.IGNORECASE)
_RE_END = re.compile(r"^END\.\s*$", re.IGNORECASE)
_RE_SPEAKER = re.compile(r"^([^:\r\n]+):\s*(\S(?:.*\S)?)$")


#: The ONLY emphasis markers this canonicalizer will remove, longest first so
#: ``**`` is tried before ``*``. A CLOSED grammar on purpose (kibitz r1): a
#: general Markdown sanitizer would be exactly the "silent massaging" the ladder
#: exists to prevent. Backticks, headings, bullets and HTML stay LOUD defects.
_TRANSPORT_MARKERS = ("**", "__", "*", "_")

#: The structural delimiters a WHOLE-LINE emphasis wrapper is allowed to hide.
#: Shape 4 (below) unwraps only when the result matches one of these, so the
#: rule stays "balanced emphasis around TRANSPORT is transport" rather than
#: becoming a general Markdown sanitizer. Ordered as the parser tries them.
_TRANSPORT_CLASSIFIERS = (_RE_TITLE, _RE_MUSIC, _RE_SCENE, _RE_CODA, _RE_END)


def _is_transport(line: str) -> bool:
    """True iff ``line`` is one of the structural delimiters."""
    return any(rx.match(line) for rx in _TRANSPORT_CLASSIFIERS)


def _family_balanced(inner: str, marker: str) -> bool:
    """Both LENGTHS of ``marker``'s family are balanced inside ``inner``.

    THE DEFECT THIS CLOSES (QA, 2026-08-03). The first shape-4 draft asked
    ``inner.count(marker) % 2 == 0``, which cannot tell ``*`` from ``**``
    because one is a substring of the other. A single unclosed ``**`` is TWO
    ``*`` characters, so it counted as EVEN and passed; a lone ``*`` inside a
    ``**`` wrapper was invisible to a ``"**"`` substring count and also passed.
    Measured escapes, each verified to reach an accepted field:

        *SCENE 5: **bold thing*            -> SCENE 5: **bold thing
        **SCENE 5: a *lonely marker here** -> SCENE 5: a *lonely marker here
        _SCENE 5: __vault door_            -> SCENE 5: __vault door

    Those strays land in ``parsed.title`` and ``scenes[].setting``, which feed
    the final title override and the shot-direction description -- exactly the
    "silently rewritten into accepted prose carrying a stray marker" outcome
    this module's docstring promises never happens.

    Count the DOUBLE form first, remove it, then count what single characters
    remain. A legitimately nested pair (``The **vault** deep``) still passes.
    """
    long_form = marker[0] * 2
    if inner.count(long_form) % 2:
        return False
    return inner.replace(long_form, "").count(marker[0]) % 2 == 0


def _canonicalize_transport_line(line: str) -> "tuple[str, tuple[str, ...]]":
    """Strip a BALANCED emphasis wrapper from a structural label. Report it.

    THE DEFECT THIS CLOSES (live, 2026-08-01). A local model emitted its markup
    wrapped in Markdown -- ``**TITLE:** ...``, ``**ANNOUNCER:** ...``,
    ``**END.**``. Every classifier here is ``^``-anchored, so a decorated line
    misses ``_RE_TITLE`` and then MATCHES the ``_RE_SPEAKER`` catch-all, whose
    group(1) becomes a character literally named ``**TITLE``. That is not in the
    cast, so each line raised UNKNOWN_SPEAKER *and* SKELETON_BREAK -- 106 of each
    across one campaign, and three of six episodes died in the writer.

    WHAT IS AND IS NOT AUTHORSHIP. ``_normalize_line`` already owns "transport"
    normalization, and emphasis wrapped around a DELIMITER is transport: it
    carries no story. Emphasis inside a spoken line IS authorship and is left
    byte-identical -- ``**BO NI:** Hello **world**`` canonicalizes to
    ``BO NI: Hello **world**``, keeping the payload's own markers.

    BALANCED ONLY. Four recognized shapes:

        <M>LABEL:<M> payload   ->  LABEL: payload      (wrapper spans the colon)
        <M>LABEL<M>: payload   ->  LABEL: payload      (wrapper before the colon)
        <M>TOKEN<M>            ->  TOKEN               (standalone, e.g. END.)
        <M>LABEL: payload<M>   ->  LABEL: payload      (wrapper spans the LINE)

    SHAPE 4 (2026-08-03, live). The 30-word sweep lost two legs to a model that
    wrote its structure as whole-line markdown -- ``**SCENE 5: The vault**``,
    ``**MUSIC**``, ``**CODA**``, ``**TITLE: ...**``. The wrapper spans label AND
    payload, so shapes 1-3 all miss: the body opens with a space after the colon
    (not the marker) and the label does not end with the marker. The line then
    fell to the ``_RE_SPEAKER`` catch-all and produced a character literally
    named ``**SCENE 5``, i.e. ``UNKNOWN_SPEAKER`` four attempts running until
    the markup ladder exhausted and the leg died in the writer having never
    reached a video engine.

    Shape 4 is ORDERED LAST and GATED ON TRANSPORT. Last, because checked
    earlier it would mangle the case this function exists to preserve:
    ``**BO NI:** Hello **world**`` also starts and ends with ``**``, and a naive
    outer strip yields ``BO NI:** Hello **world``. Gated, because unwrapping
    anything whose result is not a structural delimiter would turn this closed
    grammar into the general Markdown sanitizer the ladder exists to prevent --
    so ``**She turns: the room is empty**`` is left ALONE and stays loud.

    It also requires the REMAINDER to carry an EVEN number of that marker, so a
    line like ``**BO NI: Hello **world**`` (unbalanced interior) is never
    silently rewritten into accepted prose carrying a stray marker.

    An unbalanced, mixed or payload-internal marker is NOT touched and stays a
    loud defect. Roster-independent by design: it never consults the cast, so a
    genuinely unknown speaker still reaches UNKNOWN_SPEAKER after decoration is
    removed, and the closed-roster diagnostic keeps its meaning.

    Returns ``(line, notes)``; every removal is reported, never hidden.
    """
    s = str(line).strip()
    if not s:
        return s, ()
    for marker in _TRANSPORT_MARKERS:
        if not s.startswith(marker):
            continue
        body = s[len(marker):]
        colon = body.find(":")
        if colon < 0:
            # standalone token -- the whole line is wrapped (``**END.**``)
            if body.endswith(marker):
                inner = body[:-len(marker)].strip()
                if inner:
                    return inner, (
                        "removed balanced %s around standalone %r"
                        % (marker, inner),)
            return s, ()
        label, rest = body[:colon], body[colon + 1:]
        if rest.startswith(marker):                     # <M>LABEL:<M> payload
            out = (label + ":" + rest[len(marker):]).strip()
            return out, ("removed balanced %s around label %r (wrapper spanned "
                         "the colon)" % (marker, label.strip()),)
        if label.endswith(marker):                      # <M>LABEL<M>: payload
            out = (label[:-len(marker)] + ":" + rest).strip()
            return out, ("removed balanced %s around label %r"
                         % (marker, label[:-len(marker)].strip()),)
        # SHAPE 4, LAST and TRANSPORT-GATED -- see the docstring.
        if body.endswith(marker):                       # <M>LABEL: payload<M>
            inner = body[:-len(marker)].strip()
            if inner and _family_balanced(inner, marker) and _is_transport(inner):
                return inner, (
                    "removed balanced %s around whole transport line %r"
                    % (marker, inner),)
        return s, ()
    return s, ()


#: A speaker label that re-states the character's ROLE after the name --
#: "Commander Vance (Space Force Tactician)". The writer does this when the
#: cast card is fresh in its context, and every such line used to raise
#: UNKNOWN_SPEAKER against a roster holding the bare name.
_RE_ROLE_PARENTHETICAL = re.compile(r"\s*\([^()]*\)\s*$")


def _strip_role_parenthetical(name: str) -> str:
    """``"Vance (Tactician)"`` -> ``"Vance"``. Strips ONE trailing group.

    Only a TRAILING parenthetical, and only one, so a name that is entirely
    parenthesised or carries an interior group is left alone rather than being
    mangled into something that might collide with a different cast member. The
    caller keeps the exact-match attempt first; this is the fallback.
    """
    stripped = _RE_ROLE_PARENTHETICAL.sub("", str(name)).strip()
    return stripped or str(name).strip()


def _normalize_line(line: str) -> "tuple[str, tuple[str, ...]]":
    """Normalize transport whitespace and balanced label emphasis.

    Authored content is untouched: see :func:`_canonicalize_transport_line` for
    the exact, closed grammar and why a spoken word can never be rewritten."""
    return _canonicalize_transport_line(line)


def normalize_fable2_markup_text(text: str) -> str:
    """Return the parser's normalized proof artifact.

    SHARES ``_canonicalize_transport_line`` WITH THE PARSER ON PURPOSE (kibitz
    r1 MUST-FIX 3). This text is what ``normalized_source``, its sha256 and the
    proof map are built from, while ``_normalize_line`` decides classification.
    Two independent normalizers would let the accepted script diverge from the
    artifact hashed beside it -- a receipt describing a document nobody parsed."""
    return "\n".join(
        _canonicalize_transport_line(raw)[0] for raw in str(text).splitlines())


class Fable2ParseDefect(enum.Enum):
    """Defect classes for the markup ladder (doc section 6, exact set)."""

    MISSING_TITLE = "MISSING_TITLE"
    DUPLICATE_TITLE = "DUPLICATE_TITLE"
    # Missing END is structural. Every retry uses the same provider-capacity
    # contract; the parser never infers a word/token shortfall.
    MISSING_END = "MISSING_END"
    CONTENT_AFTER_END = "CONTENT_AFTER_END"
    BAD_LINE_SHAPE = "BAD_LINE_SHAPE"
    UNKNOWN_SPEAKER = "UNKNOWN_SPEAKER"
    SCENE_ORDER = "SCENE_ORDER"
    EMPTY_SCENE = "EMPTY_SCENE"
    SKELETON_BREAK = "SKELETON_BREAK"
    CAST_MEMBER_SILENT = "CAST_MEMBER_SILENT"
    MULTIPLE_CODA = "MULTIPLE_CODA"


@dataclass(frozen=True)
class ParseDefect:
    """One collected defect: class + human detail + 1-based source line."""

    code: Fable2ParseDefect
    detail: str = ""
    line_no: "int | None" = None

    def __str__(self) -> str:
        where = f" (line {self.line_no})" if self.line_no is not None else ""
        detail = f": {self.detail}" if self.detail else ""
        return f"{self.code.value}{detail}{where}"


@dataclass(frozen=True)
class ParsedLine:
    """One constituent spoken line with canonical roster identity."""

    speaker: str
    text: str


@dataclass(frozen=True)
class ParsedScene:
    n: int
    setting: str
    lines: "tuple[ParsedLine, ...]"


@dataclass(frozen=True)
class ParsedScript:
    title: str
    music_open: str
    music_inter: "tuple[tuple[int, str], ...]"  # (scene_after, cue_text)
    music_close: str
    announcer_intro: "tuple[str, ...]"
    scenes: "tuple[ParsedScene, ...]"
    announcer_outro: "tuple[str, ...]"
    coda: str
    character_word_count: int
    announcer_word_count: int
    # Retained for receipt compatibility; prose normalization is retired.
    normalizations: "tuple[str, ...]" = ()


def render_defects(defects: "tuple[ParseDefect, ...]") -> str:
    """Full defect list as structural-retry quotable text (one per line)."""
    return "\n".join(f"- {d}" for d in defects)


def _wc(text: str) -> int:
    return canonical_word_count(text)


# --- state machine states ----------------------------------------------------
_EXPECT_TITLE, _PREAMBLE, _SCENES, _POSTAMBLE, _DONE = range(5)


class _Parse:
    """Mutable walk state (module-private; the public surface is pure)."""

    def __init__(self, cast_names) -> None:
        self.state = _EXPECT_TITLE
        self.defects: "list[ParseDefect]" = []
        self.cast_names = tuple(str(name).strip() for name in cast_names)
        self.speaker_by_key = {
            self._speaker_key(ANNOUNCER_NAME): ANNOUNCER_NAME,
        }
        for name in self.cast_names:
            key = self._speaker_key(name)
            if not key:
                self.skeleton("cast roster contains a blank speaker label")
                continue
            prior = self.speaker_by_key.get(key)
            if prior is not None:
                self.skeleton(
                    f"cast roster labels {prior!r} and {name!r} are "
                    "ambiguous under case-insensitive identity"
                )
                continue
            self.speaker_by_key[key] = name
        self.title: "str | None" = None
        self.title_first = True
        self.music_open: "str | None" = None
        self.music_inter: "list[tuple[int, str]]" = []
        self.music_close: "str | None" = None
        self.intro: "list[str]" = []
        self.scenes: "list[tuple[int, str, list[ParsedLine]]]" = []
        self.outro: "list[str]" = []
        self.coda: "str | None" = None
        self.saw_end = False

    @staticmethod
    def _speaker_key(value: str) -> str:
        """Case/spacing-insensitive identity; display text stays canonical."""
        return " ".join(str(value).split()).casefold()

    def defect(self, code: Fable2ParseDefect, detail: str = "",
               line_no: "int | None" = None) -> None:
        self.defects.append(ParseDefect(code, detail, line_no))

    def skeleton(self, detail: str, line_no: "int | None" = None) -> None:
        self.defect(Fable2ParseDefect.SKELETON_BREAK, detail, line_no)

    # -- per-shape handlers ---------------------------------------------------

    def on_title(self, text: str, no: int) -> None:
        if self.title is not None:
            self.defect(Fable2ParseDefect.DUPLICATE_TITLE, text, no)
            return
        self.title = text
        if self.state != _EXPECT_TITLE:
            self.title_first = False
        self.state = max(self.state, _PREAMBLE)

    def on_music(self, text: str, no: int) -> None:
        if self.state in (_EXPECT_TITLE, _PREAMBLE):
            self.state = max(self.state, _PREAMBLE)
            if self.music_open is None:
                if self.intro:
                    self.skeleton(
                        "opening MUSIC must come before the announcer intro",
                        no)
                self.music_open = text
            else:
                self.skeleton("extra MUSIC line in the preamble", no)
        elif self.state == _SCENES:
            self.music_inter.append((self.scenes[-1][0], text))
        elif self.state == _POSTAMBLE:
            if self.coda is None:
                self.skeleton("closing MUSIC must come after the CODA", no)
            if self.music_close is None:
                self.music_close = text
            else:
                self.skeleton("extra MUSIC line in the postamble", no)

    def _close_scene(self, no: "int | None") -> None:
        if self.scenes and not self.scenes[-1][2]:
            self.defect(Fable2ParseDefect.EMPTY_SCENE,
                        f"SCENE {self.scenes[-1][0]} has no spoken lines", no)

    def _check_preamble_complete(self, no: int) -> None:
        if self.music_open is None:
            self.skeleton("opening MUSIC line missing", no)
        if not self.intro:
            self.skeleton("announcer intro missing", no)

    def on_scene(self, n: int, setting: str, no: int) -> None:
        if self.state == _POSTAMBLE:
            self.skeleton("SCENE header after the announcer outro began", no)
            return
        if self.state in (_EXPECT_TITLE, _PREAMBLE):
            self._check_preamble_complete(no)
            self.state = _SCENES
        else:
            self._close_scene(no)
        expected = (self.scenes[-1][0] + 1) if self.scenes else 1
        if n != expected:
            self.defect(Fable2ParseDefect.SCENE_ORDER,
                        f"SCENE {n} where SCENE {expected} was expected", no)
        self.scenes.append((n, setting, []))

    def on_coda(self, text: str, no: int) -> None:
        if self.coda is not None:
            self.defect(Fable2ParseDefect.MULTIPLE_CODA, text, no)
            return
        if self.state in (_EXPECT_TITLE, _PREAMBLE):
            self.skeleton("CODA before any scene", no)
            self.state = _PREAMBLE
        elif self.state == _SCENES:
            self._close_scene(no)
            self.skeleton("announcer outro missing before the CODA", no)
            self.state = _POSTAMBLE
        self.coda = text

    def on_end(self, no: int) -> None:
        self.saw_end = True
        if self.state == _SCENES:
            self._close_scene(no)
        if not self.scenes:
            self.skeleton("no scenes before END.", no)
        if not self.outro:
            self.skeleton("announcer outro missing", no)
        if self.coda is None:
            self.skeleton("CODA missing", no)
        if self.music_close is None:
            self.skeleton("closing MUSIC line missing", no)
        self.state = _DONE

    def on_speaker(self, name: str, text: str, no: int) -> None:
        supplied_name = " ".join(name.split())
        canonical_name = self.speaker_by_key.get(
            self._speaker_key(supplied_name)
        )
        if canonical_name is None:
            # THE ROLE PARENTHETICAL (2026-08-02, live wan_ti2v leg).
            # The writer re-states a speaker with the role it was cast in --
            # "Commander Vance (Space Force Tactician)" against a roster that
            # says "Commander Vance" -- and every one of that character's lines
            # then raised UNKNOWN_SPEAKER. Four repair attempts later the markup
            # ladder exhausted and the leg died in the writer, 2.7 minutes in,
            # having never reached a video engine at all.
            #
            # It is a restatement of the cast, not a different character, so it
            # resolves rather than fails. Deliberately a FALLBACK and never the
            # primary key: a roster that genuinely carries parentheses still
            # matches exactly first, and this cannot merge two distinct cast
            # members because it only rewrites the SUPPLIED name before looking
            # it up in the roster map that was already built from exact keys.
            bare = _strip_role_parenthetical(supplied_name)
            if bare != supplied_name:
                canonical_name = self.speaker_by_key.get(
                    self._speaker_key(bare)
                )
        if canonical_name is None:
            self.defect(
                Fable2ParseDefect.UNKNOWN_SPEAKER, supplied_name, no
            )
            canonical_name = supplied_name
        if canonical_name == ANNOUNCER_NAME:
            if self.state in (_EXPECT_TITLE, _PREAMBLE):
                self.state = max(self.state, _PREAMBLE)
                self.intro.append(text)
            elif self.state == _SCENES:
                self._close_scene(no)
                self.state = _POSTAMBLE
                self.outro.append(text)
            elif self.state == _POSTAMBLE:
                if self.coda is not None:
                    self.skeleton("ANNOUNCER line after the CODA", no)
                self.outro.append(text)
            return
        # character line
        if self.state in (_EXPECT_TITLE, _PREAMBLE):
            self.skeleton(
                f"character line ({canonical_name}) before SCENE 1", no
            )
            self.state = _PREAMBLE
        elif self.state == _SCENES:
            self.scenes[-1][2].append(
                ParsedLine(speaker=canonical_name, text=text)
            )
        elif self.state == _POSTAMBLE:
            self.skeleton(
                f"character line ({canonical_name}) after the last scene", no
            )


def parse_fable2_markup(text: str, cast_names) -> (
        "tuple[ParsedScript | None, tuple[ParseDefect, ...]]"):
    """Parse whole-play fable2 markup against the legal cast.

    ``cast_names`` is the treatment's canonical display-name roster;
    ANNOUNCER is implicitly legal. Matching ignores case and repeated spacing,
    while the returned artifact uses the canonical roster spelling.
    Returns ``(ParsedScript, ())`` on a clean parse or ``(None, defects)``
    with EVERY defect collected. Pure: no I/O, no mutation of arguments,
    never rewrites a spoken word.
    """
    p = _Parse(cast_names)
    normalizations: "list[str]" = []
    for no, raw in enumerate(str(text).splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        if p.state == _DONE:
            p.defect(Fable2ParseDefect.CONTENT_AFTER_END, line[:80], no)
            continue
        line, notes = _normalize_line(line)
        normalizations.extend(f"line {no}: {n}" for n in notes)
        if not line:
            continue  # the whole line was decoration
        m = _RE_TITLE.match(line)
        if m:
            p.on_title(m.group(1).strip(), no)
            continue
        m = _RE_MUSIC.match(line)
        if m:
            p.on_music(m.group(1).strip(), no)
            continue
        m = _RE_SCENE.match(line)
        if m:
            p.on_scene(int(m.group(1)), m.group(2).strip(), no)
            continue
        m = _RE_CODA.match(line)
        if m:
            p.on_coda(m.group(1).strip(), no)
            continue
        if _RE_END.match(line):
            p.on_end(no)
            continue
        m = _RE_SPEAKER.match(line)
        if m:
            p.on_speaker(m.group(1), m.group(2).strip(), no)
            continue
        p.defect(Fable2ParseDefect.BAD_LINE_SHAPE, line[:80], no)
        if p.state == _EXPECT_TITLE:
            p.state = _PREAMBLE

    # ---- end-of-text checks -------------------------------------------------
    if p.title is None:
        p.defect(Fable2ParseDefect.MISSING_TITLE)
    elif not p.title_first:
        p.skeleton("TITLE is not the first line")
    if not p.saw_end:
        # The missing delimiter is the actionable structural defect. Suppress
        # derivative postamble messages until an END line actually arrives.
        if p.state == _SCENES:
            p._close_scene(None)
        p.defect(Fable2ParseDefect.MISSING_END)
    spoken = {ln.speaker for _n, _s, lines in p.scenes for ln in lines}
    for name in p.cast_names:
        if name not in spoken:
            p.defect(Fable2ParseDefect.CAST_MEMBER_SILENT, name)

    if p.defects:
        return None, tuple(p.defects)

    character_words = sum(
        _wc(ln.text) for _n, _s, lines in p.scenes for ln in lines)
    announcer_words = (sum(_wc(t) for t in p.intro)
                       + sum(_wc(t) for t in p.outro)
                       + _wc(p.coda or ""))
    script = ParsedScript(
        title=p.title or "",
        music_open=p.music_open or "",
        music_inter=tuple(p.music_inter),
        music_close=p.music_close or "",
        announcer_intro=tuple(p.intro),
        scenes=tuple(ParsedScene(n=n, setting=s, lines=tuple(lines))
                     for n, s, lines in p.scenes),
        announcer_outro=tuple(p.outro),
        coda=p.coda or "",
        character_word_count=character_words,
        announcer_word_count=announcer_words,
        normalizations=tuple(normalizations),
    )
    return script, ()
