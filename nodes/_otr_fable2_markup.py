"""scifi_fable2 markup parser -- Stage S0 of the fable2 lane.

Parses the whole-play markup the P3/P5 script passes emit (grammar single
source of truth: pack seam ``fable2_script_system`` in
nodes/story_packs/scifi_fable2/scifi_fable2_v1.json, section 4 of
docs/2026-07-10-scifi-fable2-architecture.md). Pure stdlib functions, no
ComfyUI imports, no I/O -- property-tested in tests/test_fable2_markup.py.

Contract (architecture doc section 6):
- line classifiers, first match wins: TITLE / MUSIC / SCENE / CODA / END /
  speaker; anything else non-blank = BAD_LINE_SHAPE;
- state machine EXPECT_TITLE -> PREAMBLE -> SCENES -> POSTAMBLE -> DONE;
- ALL defects are COLLECTED (never first-fail): the runner quotes the full
  list in the markup-ladder reroll rung;
- ``parse_fable2_markup(text, cast_names) -> (ParsedScript | None, defects)``
  -- the script object is returned ONLY on a defect-free parse;
- UNKNOWN_SPEAKER is a HARD defect: no Levenshtein remap in S1-S3 (near-miss
  remap is deferred post-S3 behind live drift evidence, kibitz r1/CUT2);
- character_word_count and announcer_word_count are SEPARATE counters
  (r4/M4); the budget band downstream gates character words only. The CODA
  is announcer-spoken and counts as announcer words;
- lines are kept PER CONSTITUENT (one ParsedLine per markup line, never
  merged here) -- P7 assembly merges same-speaker runs and proves verbatim
  per constituent (r2/M4).

Python judges; the LLM writes: this module never rewrites, trims, or repairs
a spoken word -- a defective draft is rerolled upstream, never patched here.

DECORATION NORMALIZATION (S1b live-smoke hardening, 2026-07-10): the local
12B model persistently decorates otherwise-correct drafts with markdown
emphasis (``**TITLE:**``) and parenthetical delivery tags (``(softly)``)
through EVERY prompt-side ban including an assistant-turn few-shot (5 live
rolls of evidence). Both are NON-SPOKEN notation, and the repo already
ships the operator-ratified precedent for exactly this class (the legacy
composer's ``stage_dir_stripped`` compose flag). So the parser now applies
a DELETE-ONLY pre-lex normalization -- markdown emphasis markers stripped
everywhere; LINE-LEADING parenthetical/bracket delivery tags (letters
only, never digits) stripped from SPEAKER-line text only -- with every
strip COLLECTED into
``ParsedScript.normalizations`` (stamped to meta + logged by the runner;
the operator eyeball reviews them). Deletion never writes a word: every
surviving word is LLM-authored, so the proof-gate guarantee holds against
the same-normalized draft (``normalize_fable2_markup_text``). Parens and
brackets anywhere ELSE (TITLE/MUSIC/SCENE/CODA lines, unbalanced
leftovers, groups over the length cap) remain hard PAREN_OR_BRACKET
defects, and every other defect class stays reroll-strict.
"""
from __future__ import annotations

import enum
import re
from dataclasses import dataclass

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

# --- line classifiers (doc section 6; first match wins in _classify) --------
_RE_TITLE = re.compile(r"^TITLE:\s*(.+)$")
_RE_MUSIC = re.compile(r"^MUSIC:\s*(.+)$")
_RE_SCENE = re.compile(r"^SCENE\s+(\d{1,2}):\s*(.+)$")
_RE_CODA = re.compile(r"^CODA:\s*(.+)$")
_RE_END = re.compile(r"^END\.\s*$")
_RE_SPEAKER = re.compile(r"^([A-Z][A-Z0-9 .'-]{1,24}):\s*(.+)$")

_PAREN_OR_BRACKET_CHARS = frozenset("()[]")
_QUOTE_CHARS = frozenset('"“”„«»')

# --- decoration normalization (delete-only; see module docstring) -----------
_RE_MD_BOLD = re.compile(r"\*\*\*?(.+?)\*\*\*?")     # **bold** / ***both***
_RE_MD_STAR = re.compile(r"\*(.+?)\*")               # *italic*
# LINE-LEADING delivery tags on SPEAKER text only: "(softly)", "(over
# radio, quick)", "[beat]" -- letters/punctuation, NO digits, short.
# kibitz r2 M1 (2026-07-10): an EMBEDDED or digit-bearing group can be
# spoken/source content ("The sample (A17) moved.") -- stripping it
# against the same-normalized proof artifact would make the deletion
# invisible, so anything beyond a leading delivery tag stays a hard
# PAREN_OR_BRACKET defect.
_RE_LEADING_DELIVERY = re.compile(r"^[(\[][a-zA-Z ,.'\-]{1,40}[)\]]\s*")


def _strip_markdown(line: str) -> "tuple[str, bool]":
    out = _RE_MD_BOLD.sub(r"\1", line)
    out = _RE_MD_STAR.sub(r"\1", out)
    return out.strip(), out != line


def _normalize_line(line: str) -> "tuple[str, tuple[str, ...]]":
    """Delete-only decoration strip for ONE non-blank line. Returns the
    normalized line + note tags. Classification-order aware: the labeled
    shapes (TITLE/MUSIC/SCENE/CODA/END) only get the markdown strip;
    speaker-shaped lines also get delivery groups stripped from the TEXT
    part. Never adds, reorders, or rewrites a word."""
    notes: "list[str]" = []
    line, md = _strip_markdown(line)
    if md:
        notes.append("markdown_emphasis_stripped")
    coda = _RE_CODA.match(line)
    if coda:
        # The pivot colon is a STRUCTURAL seam marker (the splice into
        # the factual close), not a spoken word (15th live smoke
        # 2026-07-10: an otherwise-perfect draft died on a terminal
        # period). Terminal '.', '...' or missing punctuation normalizes
        # to ':' -- flagged; python owns structure, never words.
        text = coda.group(1).strip()
        fixed = text.rstrip().rstrip(".:…").rstrip() + ":"
        if fixed != text:
            notes.append("coda_pivot_colon_normalized")
            line = f"CODA: {fixed}"
        return line, tuple(notes)
    if (_RE_TITLE.match(line) or _RE_MUSIC.match(line)
            or _RE_SCENE.match(line) or _RE_END.match(line)):
        return line, tuple(notes)
    m = _RE_SPEAKER.match(line)
    if m:
        name, text = m.group(1), m.group(2)
        stripped_any = False
        while True:
            lead = _RE_LEADING_DELIVERY.match(text)
            if lead is None:
                break
            text = text[lead.end():].lstrip()
            stripped_any = True
        if stripped_any:
            notes.append("stage_direction_stripped")
            line = f"{name}: {text}".rstrip() if text else f"{name}:"
    return line, tuple(notes)


def normalize_fable2_markup_text(text: str) -> str:
    """The whole-draft form of the per-line normalization -- the runner
    stores THIS as the verbatim-proof artifact so proof spans and parsed
    constituents share one normalization (delete-only: every word in the
    result is LLM-authored)."""
    out: "list[str]" = []
    for raw in str(text).splitlines():
        line = raw.strip()
        if not line:
            out.append("")
            continue
        norm, _notes = _normalize_line(line)
        out.append(norm)
    return "\n".join(out)


class Fable2ParseDefect(enum.Enum):
    """Defect classes for the markup ladder (doc section 6, exact set)."""

    MISSING_TITLE = "MISSING_TITLE"
    DUPLICATE_TITLE = "DUPLICATE_TITLE"
    # MISSING_END = truncation; the runner retries ONCE with +25% tokens.
    MISSING_END = "MISSING_END"
    CONTENT_AFTER_END = "CONTENT_AFTER_END"
    BAD_LINE_SHAPE = "BAD_LINE_SHAPE"
    UNKNOWN_SPEAKER = "UNKNOWN_SPEAKER"
    PAREN_OR_BRACKET = "PAREN_OR_BRACKET"
    QUOTED_DIALOGUE = "QUOTED_DIALOGUE"
    SCENE_ORDER = "SCENE_ORDER"
    EMPTY_SCENE = "EMPTY_SCENE"
    SKELETON_BREAK = "SKELETON_BREAK"
    CAST_MEMBER_SILENT = "CAST_MEMBER_SILENT"
    CODA_SHAPE = "CODA_SHAPE"
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
    """One CONSTITUENT spoken line exactly as the LLM wrote it."""

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
    announcer_intro: "tuple[str, ...]"          # 1-2 lines
    scenes: "tuple[ParsedScene, ...]"
    announcer_outro: "tuple[str, ...]"          # 1-2 lines
    coda: str
    character_word_count: int
    announcer_word_count: int
    # Delete-only decoration strips applied pre-lex ("line N: tag"); the
    # runner stamps these to meta.fable2.parse + logs them LOUD.
    normalizations: "tuple[str, ...]" = ()


def render_defects(defects: "tuple[ParseDefect, ...]") -> str:
    """Full defect list as reroll-rung quotable text (one per line)."""
    return "\n".join(f"- {d}" for d in defects)


def _wc(text: str) -> int:
    return len(text.split())


# --- state machine states ----------------------------------------------------
_EXPECT_TITLE, _PREAMBLE, _SCENES, _POSTAMBLE, _DONE = range(5)


class _Parse:
    """Mutable walk state (module-private; the public surface is pure)."""

    def __init__(self, cast_names) -> None:
        self.legal_speakers = frozenset(cast_names) | {ANNOUNCER_NAME}
        self.cast_names = tuple(cast_names)
        self.state = _EXPECT_TITLE
        self.defects: "list[ParseDefect]" = []
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
        elif len(self.intro) > 2:
            self.skeleton("announcer intro exceeds 2 lines", no)

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
        # Terminal punctuation was normalized to ':' pre-lex
        # (coda_pivot_colon_normalized); what remains defect-worthy is a
        # coda that is not ONE clause (an inner sentence break) or that
        # somehow still lacks the pivot colon.
        stripped = text.rstrip().rstrip(":").rstrip()
        if not text.rstrip().endswith(":") or any(
                ch in stripped for ch in ".!?"):
            self.defect(Fable2ParseDefect.CODA_SHAPE,
                        "CODA must be ONE pivot clause (no inner "
                        "sentence breaks), ending with a colon",
                        no)

    def on_end(self, no: int) -> None:
        self.saw_end = True
        if self.state == _SCENES:
            self._close_scene(no)
        if not self.scenes:
            self.skeleton("no scenes before END.", no)
        if not self.outro:
            self.skeleton("announcer outro missing", no)
        elif len(self.outro) > 2:
            self.skeleton("announcer outro exceeds 2 lines", no)
        if self.coda is None:
            self.skeleton("CODA missing", no)
        if self.music_close is None:
            self.skeleton("closing MUSIC line missing", no)
        self.state = _DONE

    def on_speaker(self, name: str, text: str, no: int) -> None:
        name = name.strip()
        if name not in self.legal_speakers:
            self.defect(Fable2ParseDefect.UNKNOWN_SPEAKER, name, no)
        if any(ch in _QUOTE_CHARS for ch in text):
            self.defect(Fable2ParseDefect.QUOTED_DIALOGUE,
                        f"{name} line carries quotation marks", no)
        if name == ANNOUNCER_NAME:
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
            self.skeleton(f"character line ({name}) before SCENE 1", no)
            self.state = _PREAMBLE
        elif self.state == _SCENES:
            self.scenes[-1][2].append(ParsedLine(speaker=name, text=text))
        elif self.state == _POSTAMBLE:
            self.skeleton(f"character line ({name}) after the last scene", no)


def parse_fable2_markup(text: str, cast_names) -> (
        "tuple[ParsedScript | None, tuple[ParseDefect, ...]]"):
    """Parse whole-play fable2 markup against the legal cast.

    ``cast_names`` is the treatment's exact ALL-CAPS cast name set (the
    canonical name source, kibitz r2/M1); ANNOUNCER is implicitly legal.
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
        # Post-normalization leftovers -- parens/brackets on labeled
        # lines, unbalanced groups, over-cap asides -- stay HARD defects.
        if any(ch in _PAREN_OR_BRACKET_CHARS for ch in line):
            p.defect(Fable2ParseDefect.PAREN_OR_BRACKET, line[:80], no)
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
        # Truncation: the postamble-completeness skeleton checks live in
        # on_end -- suppressing them here keeps the reroll message about the
        # ACTUAL defect (the +25% token retry), not its cascade.
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
