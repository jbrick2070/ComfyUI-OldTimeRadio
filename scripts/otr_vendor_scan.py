# -*- coding: utf-8 -*-
"""Vendor a scene from a SCANNED volume that carries a text layer.

WHY THIS IS A SEPARATE SCRIPT FROM `otr_vendor_shakespeare.py`. That one fetches
an HTML page and reads the speakers off the EDITION'S OWN MARKUP -- a span
class, an italic tag, an indent class. A PDF text layer has none of that. The
page's structure is thrown away by `extract_text()` and a speaker name arrives
INLINE with the dialogue:

    ACTO III SCENA IV 95 MACBETH Onde ? LENNOX Aqui ,meu bom senhor

So the two share nothing at the speaker-marking step and forcing them together
would mean a second set of rules inside the first script's shape. They DO share
the scene-boundary step, which works on a PDF unchanged, so `extract` is
imported rather than reimplemented.

WHAT ANCHORS THE SPEAKERS. Layout is gone but CASE survives, so every all-caps
run is a candidate -- and a candidate is only accepted when it resolves to a
character in THAT SCENE'S English roster. The cast list does the work a layout
would have done. Two kinds of name resolve differently and both are needed:

  PROPER NOUNS mostly match on an accent-folded comparison. `MACBETH`,
  `BANQUO`, `GLOUCESTER`, `CORDELIA` are the same word in every edition here.

  FUNCTION NAMES never match, because they are translated. Folger writes
  `FIRST WITCH` where Domingos Ramos writes `1.ª FEITICEIRA` and Menendez y
  Pelayo writes `BRUJA 1.ª`. These need a small per-language vocabulary, which
  is bounded -- witch, fool, servant, messenger, all -- and is spelled out
  below rather than guessed.

ANYTHING THAT RESOLVES TO NOTHING IS NOT A SPEAKER. A running header, a folio
number, a translator's initials, a heading: all of them are all-caps and none is
in the roster, so the roster is the filter as well as the anchor.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import unicodedata
import urllib.parse

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _REPO)

import otr_vendor_shakespeare as V  # noqa: E402  (scene boundaries, shared)

BASE = os.path.join(_REPO, "config", "source_banks", "shakespeare", "translations")
SOURCES = os.path.join(_REPO, "config", "source_banks", "shakespeare", "sources")
CACHE = os.path.join(_REPO, "tmp", "scan_cache")
USER_AGENT = "OTR-corpus-vendor/1.0 (jbrick2070@gmail.com)"

#: Folger's function names, in the languages this corpus vendors. The KEY is
#: what the translation prints; the VALUE is the English roster name it stands
#: for. Proper nouns are deliberately absent -- they resolve by folding.
FUNCTION_NAMES = {
    # Portuguese (Domingos Ramos)
    "FEITICEIRA": "WITCH",
    "BOBO": "FOOL",
    "CRIADO": "SERVANT",
    "MENSAGEIRO": "MESSENGER",
    "CAVALHEIRO": "GENTLEMAN",
    "MEDICO": "DOCTOR",
    "PORTEIRO": "PORTER",
    "CAPITAO": "CAPTAIN",
    "TODAS": "ALL",
    "TODOS": "ALL",
    "AMBOS": "BOTH",
    # Spanish (Menendez y Pelayo, Jaime Clark)
    "BRUJA": "WITCH",
    "BUFON": "FOOL",
    "MENSAJERO": "MESSENGER",
    "CRIADA": "SERVANT",
    "MEDICO_ES": "DOCTOR",
    "TODAS_ES": "ALL",
}

#: The ordinal a translation prints in front of a function name, mapped to the
#: word Folger puts there. `1.ª FEITICEIRA` is `FIRST WITCH`.
ORDINALS = {
    "1": "FIRST", "2": "SECOND", "3": "THIRD", "4": "FOURTH",
    "I": "FIRST", "II": "SECOND", "III": "THIRD", "IV": "FOURTH",
    "PRIMEIRA": "FIRST", "SEGUNDA": "SECOND", "TERCEIRA": "THIRD",
    "PRIMERA": "FIRST", "SEGUNDO": "SECOND", "TERCERA": "THIRD",
}

#: A run of capitals, OPTIONALLY OPENED BY AN ORDINAL. Accented capitals are
#: included; the trailing guard stops it swallowing a capitalised word that
#: merely opens a sentence.
#:
#: THE LEADING ORDINAL IS NOT OPTIONAL POLISH -- IT IS MOST OF THE SCENE. The
#: first cut required a letter first, so `1.ª FEITICEIRA` never matched and
#: Macbeth 1.3 came back with 26 speeches and no witches, in the scene whose
#: entire opening is three witches talking to each other. The named parts
#: matched and the numbered ones vanished, which reads as a thin scene rather
#: than as a broken rule.
_CAPS_RUN = re.compile(
    r"(?<![A-Za-zÀ-ɏ0-9])"
    r"(?P<tok>(?:\d{1,2}\s*[.ªº]{0,2}\s*)?"
    r"[A-ZÀ-Þ][A-ZÀ-Þ'’.ªº]{1,24}"
    r"(?:\s+(?:\d{1,2}\s*[.ªº]{0,2}|"
    r"[A-ZÀ-Þ][A-ZÀ-Þ'’.ªº]{1,14})){0,3})"
    r"(?![a-zà-ɏ])")

#: Running headers and folio numbers repeat on every page and are all-caps.
_RUNNING_HEADER = re.compile(
    r"(?im)^.{0,6}(?:ACTO|ATTO|ACT)\s+[IVXLC]+\s*-?\s*"
    r"(?:SCENA|ESCENA|SCENE)\s+[IVXLC]+\s*\d*\s*$")

#: An act heading standing alone on a line -- the shape a recto running head
#: takes in these volumes. Scene headings are deliberately NOT included; see
#: `strip_running_titles`.
_ACT_LINE = re.compile(r"(?i)^(?:ACTO|ATTO|ACT)\b.{0,26}$")

#: Any heading-shaped line, act or scene. A HEADING NEVER EARNS ITS WAY INTO THE
#: FREQUENCY VOTE, because the vote's floor is a fifth of the volume and a
#: five-act play gives each act almost exactly that share -- so a recto running
#: head of `ACTO SEGUNDO` can reach the floor, and blanking every match would
#: take the act's own real heading with it and leave the scene unfindable.
#: Headings are handled by position instead: acts by `seen_act`, scenes by the
#: scene finder taking the first match.
_HEADING_SHAPED = re.compile(r"(?i)^(?:ACTO|ATTO|ACT|SCENA|ESCENA|SCENE)\b.{0,26}$")

#: A HEADING SPLIT ACROSS TWO LINES BY THE TEXT LAYER. The word and its numeral
#: are one line on the page and two in the extraction:
#:
#:     SCENA
#:     I
#:     Uma camara no palacio do REI LEAR
#:
#: so a search for `SCENA I` finds nothing and the scene reads as absent. This
#: is the extractor's line-breaking, not the book's, and it differs per volume:
#: the Macbeth prints `SCENA III` whole and the Rei Lear splits it. Rejoined
#: before anything looks for a heading.
#:
#: THE NUMERAL LINE CARRIES THE PRINTER'S PUNCTUATION, AND DEMANDING A BARE
#: NUMERAL MADE THE RULE FIRE ON NOTHING. A book sets its heading with a stop
#: -- `ESCENA` over `V .`, `ACTO` over `PRIMERO .`, `ACTO` over `QUINTO ,` --
#: and an end-anchored `[ \t]*$` rejects every one of them. Measured across the
#: corpus's scanned volumes, the old form rejoined ZERO headings in either
#: Spanish book, so every split heading in both was invisible to the scene
#: finder; tolerating the trailing stop rejoins 12 in the Clark volume and 19
#: in the Macpherson. Both Portuguese volumes are unaffected at 0 before and 0
#: after, which is what makes this safe to widen: the two scenes already
#: vendored from them cannot move.
#:
#: The numeral alternation is unchanged and is what keeps this narrow -- only a
#: roman, a small integer or a spelled ordinal may follow the heading word, so
#: a line of dialogue under a stray `ESCENA` is still not rejoined.
_SPLIT_HEADING = re.compile(
    r"(?im)^[ \t]*(ACTO|ATTO|ACT|SCENA|ESCENA|SCENE)[ \t]*\n[ \t]*"
    r"([IVXLC]{1,6}|\d{1,2}|PRIMEIR[OA]|SEGUND[OA]|TERCEIR[OA]|QUART[OA]|"
    r"QUINT[OA]|PRIMER[OA])[ \t.,;:]*$")

#: A BARE STAGE DIRECTION, WHICH THIS LAYOUT CANNOT DISTINGUISH FROM A SPEECH.
#: Entrances and exits are printed as their own line with no parentheses, so
#: once a label has been claimed the direction behind it looks exactly like
#: dialogue. It matters at the top of a scene: the location line reads `Uma
#: camara no palacio do REI LEAR`, the name in the SETTING resolves to a
#: character, and the entrance behind it was attributed to him -- Lear opening
#: the play by announcing "Enter Kent, Gloucester and Edmund".
#:
#: Dropped rather than reassigned, because a direction belongs to nobody. The
#: verb list is per-language and bounded, like the function names above.
_STAGE_OPENER = re.compile(
    r"(?i)^\s*(?:ENTRAM?|ENTRA|SAEM?|SAE|VAO-SE|RETIRAM|"
    r"ENTRAN|ENTRA\b|SALEN|SALE|VANSE|VASE|"
    r"ENTER|EXEUNT|EXIT)\b")

#: A SWALLOWED SPEAKER LABEL, BY ITS SHAPE. A lower-case letter running
#: straight into capitals inside one token is not a word in Spanish,
#: Portuguese or Italian -- it is a broken word welded onto the name of the
#: person who speaks next. See the refusal in `main`, which is where this is
#: enforced and where the two shipped instances are named.
#:
#: TWO SHAPES, BECAUSE THE EDITIONS DISAGREE ABOUT CASE. The 1912 Macbeth
#: sets its labels in capitals, so the weld reads `horriBANQUO` -- a
#: lower-case letter then a run of capitals. The 1919 Rei Lear sets them in
#: Title Case, so it reads `AfasKent` -- one capital, then lower-case resumes.
#: A pattern that demands two capitals catches the first and misses the
#: second, which is exactly how the second shipped after the first was
#: fixed. The capital must begin a real word (another capital, or two more
#: lower-case letters), so a lone stray capital inside OCR noise is not a
#: refusal.
_INTERIOR_WELD = re.compile(r"[a-zà-ÿ][A-ZÀ-Þ](?:[A-ZÀ-Þ]|[a-zà-ÿ]{2})")

#: A WORD CARRYING CYRILLIC IS A SCANNER ARTIFACT, NOT A WORD. The optical
#: reader on these volumes resolves some capitals to their Cyrillic lookalikes,
#: so the running title comes through as `МАСВЕТH` -- М А С В Е Т in Cyrillic
#: with a Latin H. It matches no heading pattern, survives every filter aimed at
#: Latin text, and lands inside a speech where a voice engine reads it aloud.
#: These volumes are Portuguese and Spanish; Cyrillic cannot legitimately occur,
#: so any word containing it is dropped whole rather than repaired -- repairing
#: would mean guessing which capital was meant.
_CYRILLIC_WORD = re.compile(r"\S*[Ѐ-ӿ]\S*")


def fold(text: str) -> str:
    """Accent-blind upper-case key. Never shown to anyone."""
    norm = unicodedata.normalize("NFD", str(text or ""))
    return "".join(c for c in norm if not unicodedata.combining(c)).upper()


#: A word broken across a printed line. Rejoined, because the hyphen is the
#: typesetter's and not the translator's -- `por-\ntos` is `portos`, one word,
#: and a voice engine handed the hyphen reads two.
#:
#: THE CONTINUATION MUST BE LOWER-CASE, AND THAT GUARD IS THE WHOLE RULE. A
#: typesetter breaks a word in its middle, so what follows the hyphen is always
#: the rest of a word. What ELSE can follow it is the next SPEAKER LABEL, when
#: the break happens to fall on the last line before one -- and rejoining there
#: destroys both of them at once. Measured in the shipped 1912 Macbeth, which
#: prints
#:
#:     MACBETH
#:     Nunca vi assim um dia tão bello e tão horri-
#:     BANQUO
#:     Que distancia fazem d'aqui a Forres ? ...
#:
#: and arrived in the corpus as the single token `horriBANQUO`: Banquo's label
#: was eaten, so his opening speech was filed under MACBETH, and the nonsense
#: word was left for a voice engine to read aloud. Requiring a lower-case
#: continuation refuses that join and leaves the label standing. It also
#: declines to weld a numbered label (`horri-` over `1.ª FEITICEIRA`) and
#: declines to swallow the author's own hyphen in a compound broken at its
#: hyphen (`Anglo-` over `Saxão`), both of which are the right answer for the
#: same reason: only a lower-case run is the rest of a word.
_LINE_BREAK_HYPHEN = re.compile(r"(\w)[-‐‑]\s*\n\s*([a-zà-ɏ])")


def pdf_text(url: str, reading_order: str = "flat") -> list[str]:
    """Every page's text layer, cached by url. Returns one string per page.

    PYMUPDF, NOT PYPDF, AND THE DIFFERENCE IS THE WHOLE LANE. Both read this
    text layer, and pypdf loses the word spacing inside a line:

        pypdf     tinhacastanhasno seu regaco;mastigava,mastigava
        pymupdf   tinha castanhas no seu regaco; mastigava, mastigava

    A day was spent recording that the scanned volumes "need a de-spacing pass"
    and sizing that as a model call per scene. They do not. They need a better
    extractor, and one was already installed. Measure the tool before designing
    around its output.
    """
    import pymupdf

    os.makedirs(CACHE, exist_ok=True)
    name = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24] + ".pdf"
    path = os.path.join(CACHE, name)
    if not os.path.exists(path) or os.path.getsize(path) < 5000:
        subprocess.run(["curl", "-sL", "-A", USER_AGENT, "-o", path, url],
                       check=True, timeout=900)
    doc = pymupdf.open(path)
    out = []
    for page in doc:
        try:
            if reading_order == "coordinates":
                text = "\n".join(rows_from_coordinates(page))
                # SAY WHERE THE ORDER CANNOT BE TRUSTED. A word whose own
                # glyphs straddle two printed rows carries two speakers, and
                # no reordering of intact words repairs it -- so the page is
                # named here rather than silently shipped.
                welded = fused_words(page)
                if welded:
                    print("[scan] page %d has %d word(s) welded across a row "
                          "boundary; speakers there are not trustworthy: %s"
                          % (page.number, len(welded), ", ".join(welded[:4])))
            else:
                text = page.get_text() or ""
            out.append(_LINE_BREAK_HYPHEN.sub(r"\1\2", text))
        except Exception:                      # noqa: BLE001 - a bad page is empty
            out.append("")
    doc.close()
    return out


#: A ROW IS A BASELINE, NOT A BOUNDING BOX. Measured on the Clark Tempest's
#: four scene pages: clustering on `y0` splits 73 image-verified printed rows,
#: because a comma and a capital do not share a box top. Clustering on the
#: MEDIAN CHARACTER BASELINE splits none and merges none.
#:
#: THE SPAN IS ADAPTIVE, AND THE FLOOR IS THE OLD FIXED VALUE. A hanging cue
#: can sit 3.29 points off its own dialogue (Clark page 133) or 3.66 (page
#: 182), and a fixed 3.0 leaves both cues on a row of their own. Measured:
#: `max(3.0, 0.45 * median word height)` is 4.37-4.50 on Clark, joins both
#: cues, changes NOTHING on the four Tempest pages (identical partitions,
#: 130 image-verified verse rows kept), and stays under the 5.445 nearest-row
#: gap. On Macpherson it is about 3.53 and still misses four image-verified
#: cue joins (pages 322, 346, 358, 431), so that volume is NOT certified by
#: this rule -- which is one more reason the reader stays opt-in.
_ROW_BASELINE_FLOOR = 3.0
_ROW_BASELINE_FRACTION = 0.45

#: A word whose own characters jump further than this is not one word: the
#: extractor fused the end of one printed row onto the start of the next.
#: Measured example, Clark Otelo page 176: `ramera.Mátame,` carries Othello's
#: `ramera.` and Desdemona's `Mátame,` in a single token, with a 7.6 point
#: baseline jump inside it. NOTHING that reorders intact words can repair that,
#: so it is REPORTED rather than silently reordered -- see `fused_words`.
_FUSED_WORD_JUMP = 4.5


def _median(values) -> float:
    """The statistical median: the mean of BOTH middles on an even count.

    Taking the upper middle alone looked harmless and was not. A word's
    characters are an even count about half the time, and on the qualified
    reconstruction that one-sided pick left 524 word baselines more than a
    point off across 13 pages -- `ramera.Mátame,` on Otelo 176 has fourteen
    glyph origins whose two middles are 269.76 and 277.40, and the upper one
    put it on the wrong row. With the true median every one of 302,837
    baselines matches the reference exactly and all 1,450 pages agree.
    """
    ordered = sorted(values)
    half = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[half]
    return (ordered[half - 1] + ordered[half]) / 2.0


def _word_baselines(page) -> tuple[list, list]:
    """``(words, baselines)`` -- PyMuPDF's own word tuples, each with the
    median baseline of ITS OWN characters and nothing else's.

    ONE TEXTPAGE, SO THE TWO VIEWS AGREE. Ask a page for `words` and for
    `rawdict` in two separate calls and each builds its own TextPage, and the
    two number their blocks differently -- on Clark page 59 the words started
    at block 0 while the raw blocks started at 1, which broke every join by
    index. Build the TextPage ONCE with the word flags and read both views
    from it: then the block, line and word numbers on a word tuple address
    exactly the characters that spell it. Measured across all five scanned
    volumes, 302,837 words, zero mismatches.

    WHY NOT GEOMETRY. The first shipped version of this looked up glyphs by
    the word's bounding box, and a word box is tall: it reaches into the
    printed row above. On Clark page 59 the word `que` (baseline 100.3)
    collected four characters of the line above (91.9) and took THEIR median,
    an 8.4 point error that moved it into the wrong row; on the same page
    `gusano,` moved out of Prospero's row into Miranda's. The page still had
    exactly 38 rows and every word survived, so neither the row count nor
    word conservation could see it. 14,689 words across 1,274 pages were
    off by more than a point. Ownership is by identity, never by proximity.
    """
    import pymupdf

    tp = page.get_textpage(flags=pymupdf.TEXTFLAGS_WORDS)
    words = page.get_text("words", textpage=tp)
    raw = page.get_text("rawdict", textpage=tp)
    # (block, line) -> [median baseline of each whitespace-separated run]
    per_line = {}
    for b_no, block in enumerate(raw.get("blocks", ())):
        for l_no, line in enumerate(block.get("lines", ())):
            runs, run = [], []
            # ACCUMULATE ACROSS SPANS. A word can change font or size in its
            # middle (an italic cue, a damaged glyph), and a run that resets
            # at each span boundary never sees the whole word.
            for span in line.get("spans", ()):
                for ch in span.get("chars", ()):
                    if ch.get("c", "").isspace():
                        if run:
                            runs.append(_median(run))
                            run = []
                    else:
                        run.append(ch["origin"][1])
            if run:
                runs.append(_median(run))
            per_line[(b_no, l_no)] = runs
    baselines = []
    for x0, y0, x1, y1, text, b_no, l_no, w_no in words:
        runs = per_line.get((b_no, l_no), ())
        baselines.append(runs[w_no] if w_no < len(runs) else (y0 + y1) / 2.0)
    return words, baselines


def rows_from_coordinates(page) -> list[str]:
    """The page's printed rows, in reading order, rebuilt from glyph positions.

    WHY THIS EXISTS AT ALL. `page.get_text()` does not return the printed
    order. On Clark page 59 the marginal cues `MIR.` and `FER.` are emitted at
    the BOTTOM of the page, detached from the lines they label, so Prospero
    absorbs two speeches that are not his. On page 36 of the 1912 Macbeth --
    a scene this repo already vendored -- the syllable `vel!` is thrown to the
    TOP of the page and lands inside the witches' speech, which is how
    `horrível` came to be broken and Banquo's opening speech came to be filed
    under Macbeth. A speaker label moved away from its dialogue cannot be
    matched by any table of names, so reading order is upstream of every other
    rule in this file.

    THE RULE. Take each word's MEDIAN CHARACTER BASELINE, from its own
    characters only (`_word_baselines`); sort by baseline, then by x, keeping
    source order for ties rather than sorting on the text; open a row at the
    first word and keep adding while the baseline stays within the adaptive
    span OF THE ROW'S FIRST BASELINE -- never of the previous word, because
    chaining near-neighbours walks a row down the page one small step at a
    time. On the Clark Tempest's four scene pages this splits zero printed
    rows and merges zero; the widest row spans 2.4 points and the closest
    neighbouring rows sit 5.4 apart.

    WHAT IT IS NOT. It is not certified across every volume, and the
    measurements say so plainly: Macpherson still splits four image-verified
    cue rows under the adaptive span, and it carries six pages of rotated
    tables. It is therefore OPT-IN -- `reading_order="coordinates"` -- and
    the flattened path remains the default until a volume is proven. Do not
    flip the default globally; prove a volume and pin it.

    A PDF "LINE" IS NOT A PRINTED ROW EITHER, which is why this works on words
    rather than on `get_text("dict")` lines: Clark page 42 puts Alonso's speech
    and Sebastian's reply in one PDF line whose word baselines are 177.6 and
    185.5 apart.
    """
    words, baselines = _word_baselines(page)
    if not words:
        return []
    heights = sorted(w[3] - w[1] for w in words)
    span = max(_ROW_BASELINE_FLOOR, _ROW_BASELINE_FRACTION * _median(heights))
    order = sorted(range(len(words)), key=lambda i: (baselines[i], words[i][0]))
    rows, row, anchor = [], [], None
    for i in order:
        baseline, x0, text = baselines[i], words[i][0], words[i][4]
        if anchor is None or baseline - anchor <= span:
            anchor = baseline if anchor is None else anchor
            row.append((x0, i, text))
        else:
            rows.append(" ".join(t for _, _, t in sorted(row)))
            row, anchor = [(x0, i, text)], baseline
    if row:
        rows.append(" ".join(t for _, _, t in sorted(row)))
    return rows


def fused_words(page) -> list[str]:
    """Words whose own characters straddle two printed rows.

    REPORTED, NEVER REPAIRED HERE. `ramera.Mátame,` on Clark Otelo page 176 is
    Othello's word welded to Desdemona's, and splitting it is a decision about
    whose speech a fragment belongs to -- which is the one judgement this lane
    refuses to make silently. A caller that sees these knows the page needs a
    human or an image check before its speakers can be trusted.

    THE RUN CROSSES SPAN BOUNDARIES. The first version reset at every span
    and therefore found none of the 32 measured fused tokens, because the
    fusion IS a font change: the second half of the word is set in the next
    row's face. It flagged nine pages of rotated tables instead. Rotated
    lines are skipped -- their glyph origins climb by design.
    """
    import pymupdf

    out = []
    tp = page.get_textpage(flags=pymupdf.TEXTFLAGS_WORDS)
    for block in page.get_text("rawdict", textpage=tp).get("blocks", ()):
        for line in block.get("lines", ()):
            direction = line.get("dir", (1, 0))
            if abs(direction[0]) < 0.9:            # rotated: origins climb by design
                continue
            run = []
            for span in line.get("spans", ()):
                for ch in span.get("chars", ()):
                    if ch.get("c", "").isspace():
                        run = []
                        continue
                    run.append(ch)
                    if len(run) > 1:
                        jump = abs(run[-1]["origin"][1] - run[-2]["origin"][1])
                        if jump > _FUSED_WORD_JUMP:
                            out.append("".join(c["c"] for c in run))
                            run = []
    return out


def roster_for(stem: str) -> set[str]:
    import nodes._otr_roster_gender as RG

    path = os.path.join(SOURCES, stem + ".txt")
    if not os.path.exists(path):
        return set()
    return {r["name"] for r in RG.load_roster_characters(path)}


#: A title in front of a name. `Duque de Borgonha` and `Rei de Franca` are the
#: Duke of Burgundy and the King of France, and Folger lists them by their place
#: alone. Stripped so the place is what gets matched.
_TITLE_PREFIX = re.compile(
    r"(?i)^\s*(?:DUQUE|DUQUESA|REI|RAINHA|CONDE|CONDESSA|DUQUE\s+DE|"
    r"REY|REINA|DUC|SENHOR|LORD)\s+(?:DE\s+|D[AEO]\s+)?")

#: A stage qualifier printed beside the label -- `Cordelia (aparte)`. It is
#: business, not part of the name, and the HTML path strips the same thing.
_LABEL_QUALIFIER = re.compile(r"\s*\([^()]{0,40}\)\s*$")

#: Place names Folger writes in English and an Iberian edition does not. Kept
#: tiny and explicit: every entry is a name a fold cannot reach.
PLACE_NAMES = {
    "BORGONHA": "BURGUNDY", "BORGONA": "BURGUNDY",
    "FRANCA": "FRANCE", "FRANCIA": "FRANCE",
    "ALBANIA": "ALBANY", "CORNOUAILLES": "CORNWALL", "CORNUALLA": "CORNWALL",
}


def resolve(token: str, roster: set[str]) -> str | None:
    """The English roster name this printed label stands for, or None.

    THE ROSTER IS THE FILTER AS WELL AS THE ANCHOR. A running header, a folio
    number and a translator's initials are all-caps too; none of them resolves,
    so none of them becomes a speaker.
    """
    folded = {fold(n): n for n in roster}
    key = fold(_LABEL_QUALIFIER.sub("", token)).strip(" .")
    if key in folded:
        return folded[key]

    # A TITLE IS NOT A NAME. `DUQUE DE BORGONHA` is Burgundy, whom Folger lists
    # by the place alone, so the title comes off before anything else is tried.
    bare = _TITLE_PREFIX.sub("", key).strip(" .")
    if bare and bare != key:
        if bare in folded:
            return folded[bare]
        if PLACE_NAMES.get(bare) in roster:
            return PLACE_NAMES[bare]
    if PLACE_NAMES.get(key) in roster:
        return PLACE_NAMES[key]

    # A TRANSLATED PROPER NOUN KEEPS ITS STEM AND CHANGES ITS ENDING --
    # `Edmundo` for Edmund, `Regane` for Regan, `Cordelia` for Cordelia. Matched
    # on a shared stem of at least five characters, in either direction, which
    # is long enough that no two names in one scene's roster collide. Anything
    # shorter is refused rather than guessed: putting a speech in the wrong
    # mouth is the failure this corpus cannot afford, and a near-miss on a short
    # name is exactly how that happens.
    for candidate in (bare or key, key):
        if len(candidate) < 5:
            continue
        for fkey, name in folded.items():
            if len(fkey) < 5:
                continue
            if candidate.startswith(fkey) or fkey.startswith(candidate):
                if abs(len(candidate) - len(fkey)) <= 2:
                    return name

    words = [w.strip(" .ªº") for w in key.split() if w.strip(" .")]
    if not words:
        return None

    # `1.ª FEITICEIRA` / `BRUJA 1.ª` -- an ordinal beside a function name, in
    # either order, because the two languages disagree about which comes first.
    ordinal = next((ORDINALS[w] for w in words if w in ORDINALS), None)
    function = next((FUNCTION_NAMES[w] for w in words if w in FUNCTION_NAMES), None)
    if function:
        if ordinal and fold("%s %s" % (ordinal, function)) in folded:
            return folded[fold("%s %s" % (ordinal, function))]
        if fold(function) in folded:
            return folded[fold(function)]
    return None


def clean_label(token: str) -> str:
    """The printed label, upper-cased, without its stage qualifier.

    `Cordelia (aparte)` is Cordelia speaking aside. The parenthetical is
    business, it is not part of who she is, and storing it would ship a
    character whose name contains a stage direction -- and a SECOND character
    the first time she speaks without one.
    """
    return _LABEL_QUALIFIER.sub("", token).strip(" .,;:").upper()



def _edge_folio_parts(text):
    """Comparison identity only: never alter a page's line topology."""
    text = text.strip()
    hit = re.fullmatch(r"([0-9]{1,3})\s+(.+)", text)
    if hit:
        folio, title = hit.groups()
    else:
        hit = re.fullmatch(r"(.+?)\s+([0-9]{1,3})", text)
        if not hit:
            return text, False
        title, folio = hit.groups()
    if title.strip(" .,:;").upper() in {"ACTO", "ATTO", "ACT", "SCENA", "ESCENA", "SCENE"}:
        return text, False
    if not (title.isupper() or _HEADING_SHAPED.match(title)):
        return text, False
    return title.strip(), True

def running_titles(pages: list[str]) -> set[str]:
    """Short lines that repeat across the volume: the book's own furniture.

    THE RUNNING TITLE CAN BE A CHARACTER'S NAME, which is why the roster filter
    does not catch it. Rei Lear prints `REI LEAR` at the top of every page, and
    that resolves to LEAR as surely as a real speech label does -- so eight page
    headers arrived as eight Lear speeches and cut his real ones in half.

    A heading appears once; furniture appears on a fifth of the pages. Counting
    the whole document is the only way to tell them apart, because inside one
    scene they look identical.

    POSITION IS THE DISCRIMINATOR, NOT FREQUENCY. Counting every occurrence in
    the book flags the BUSIEST CHARACTERS as furniture -- Lear, Kent and
    Gloucester each stand alone on a fifth of the pages simply by speaking a
    lot, and filtering on that took this scene from 86 speeches to 46 and
    deleted its three largest parts. A running title sits at the RIM of the
    page; a speaker label appears anywhere and repeatedly. So only the two edge
    bands are counted -- `_EDGE_DEPTH` lines at each end, both ends, because a
    volume is free to put its furniture at either -- and each page votes once
    per distinct string however many times it prints it.

    THE KNOWN LIMIT, AND IT IS THE COST OF THE LINE ABOVE: a label is judged by
    WHERE it lands, so an ordinary speaker whose name happens to sit in an edge
    band on a fifth of the pages is read as furniture and removed. The floor
    makes that unlikely rather than impossible -- real pagination moves a
    speech's start around the page, which is what keeps a character out of the
    band that consistently -- and neither measured volume trips it. There is no
    cheap way to separate the two cases, because the hard case is a header that
    IS a character: the 1912 Macbeth prints exactly that.

    THE JOINED FORM COUNTS TOO, because the extractor breaks a two-word title
    across two lines: `Rei Lear` arrives as `REI` then `LEAR`, and the caps rule
    downstream rejoins it into one token that matches neither line on its own.
    Left out, that token resolved to LEAR and split his speeches at every page
    boundary -- one continuous speech arriving as two, five times in one scene.
    """
    seen = collections.Counter()
    for page in pages:
        lines = page.splitlines()
        for edge in _page_edges(lines):
            texts = [_edge_folio_parts(lines[i])[0] for i in sorted(edge)]
            forms = set(texts)
            forms.update("%s %s" % pair for pair in zip(texts, texts[1:]))
            for form in forms:
                if 0 < len(form) <= 34 and not _HEADING_SHAPED.match(form):
                    seen[form] += 1
    floor = max(4, len(pages) // 5)
    return {line for line, n in seen.items() if n >= floor}


#: A number standing alone at the edge of a page: a folio, or the numeral half
#: of a heading the text layer tore off. The 1912 Macbeth breaks its header as
#: `ACTO I- SCENA` / `III` / `17`, and the orphaned `III` is not a heading by
#: shape -- it does not start with a heading word -- so it survived the band and
#: glued onto the label behind it, arriving as `III MACBETH` and costing Macbeth
#: a speech. Both kinds of number are furniture in this position and neither is
#: ever a line of dialogue.
_FOLIO = re.compile(r"^(?:\d{1,3}|[IVXLC]{1,6})$")
_EDGE_DEPTH = 3

#: A page holding no more than this many lines is the page ANNOUNCING something
#: -- a part-title or a half-title -- rather than a page of text with a header
#: on it. See `_strip_one_edge`: the act rule will not treat a line as a running
#: head on such a page, because there would be nothing for it to be heading.
_PART_TITLE_LINES = 2


#: A heading has to appear at the edge of this many pages before it can be read
#: as a running head. Three is low on purpose: one act's header covers only its
#: own pages, which is far under any share-of-the-volume floor.
_ECHO_FLOOR = 3


def recurring_headings(pages: list[str]) -> set[str]:
    """Heading-shaped lines the volume prints at a page edge again and again.

    THE FOLIO TEST ALONE IS NOT ENOUGH, and leaving it alone rebuilt the very
    bug that killed the rule before it. A running head is recognised here by
    carrying the page number, which is true of an echo -- and also true of a
    perfectly ordinary act-opening page that happens to print `ACTO PRIMEIRO`,
    a subtitle and a folio. That page's heading is the only copy in the book,
    and blanking it loses the act.

    So the file's own principle is put back where it had been dropped: furniture
    REPEATS and a heading does not. A line must clear both tests to be removed
    -- printed at a page edge on several pages AND sitting beside a folio on
    this one. The real heading of the 1919 Rei Lear fails the second, a
    one-time act page fails the first, and the echo fails neither.
    """
    seen = collections.Counter()
    for page in pages:
        lines = page.splitlines()
        # ONE VOTE PER PAGE, NOT PER EDGE. The head and foot windows overlap on
        # a short page, so counting each edge separately let ONE printed line be
        # counted twice -- and a heading on two short pages then cleared a floor
        # of three that the same heading on two ordinary pages could not.
        edges = _page_edges(lines)
        texts = {_edge_folio_parts(lines[i])[0] for edge in edges for i in edge}
        for text in texts:
            if _HEADING_SHAPED.match(text):
                seen[_normalise_heading(text)] += 1
    return {text for text, n in seen.items() if n >= _ECHO_FLOOR}


def _normalise_heading(text: str) -> str:
    """A heading compared the way every other lookup in this file compares.

    The refusal in `main` asks whether the caller's `--scene-label` is one of
    these, and asked it with a raw string match: a volume printing `Scena III`
    against the conventional `SCENA III` on the command line did not match, so
    the guard stayed silent on exactly the volume it exists for. Case and
    interior spacing are the scanner's, not the book's.
    """
    return re.sub(r"\s+", " ", text).strip(" .,;:").upper()


def _page_edges(lines: list[str]) -> list[list[int]]:
    """The two line windows a printer puts furniture in: the head and the foot.

    THE FOOT MATTERS AS MUCH AS THE HEAD. A volume is free to put its furniture
    at either end, and this corpus has both -- the 1919 Rei Lear alternates
    `2 / REI / LEAR` on the verso against `ACTO / PRIMEIRO / 3` on the recto, so
    the folio number leads on one side and trails on the other and the two words
    of each title land on separate lines.

    Each window is returned in READING ORDER OUTWARD-IN: the head as printed,
    the foot reversed, so a caller can walk either one from the page's edge
    inward and stop where the furniture stops.
    """
    filled = [i for i, line in enumerate(lines) if line.strip()]
    if not filled:
        return []
    return [filled[:_EDGE_DEPTH], filled[-_EDGE_DEPTH:][::-1]]


def _strip_one_edge(lines: list[str], edge: list[int], forms: set[str],
                    echoed: set[str], filled_count: int,
                    band_carries_folio: bool, title_taken: bool) -> bool:
    """Blank the furniture band at one edge of a page, in place.

    Returns whether this walk spent the page's one running-title budget, so an
    overlapping second walk can inherit it -- see `strip_running_titles`.

    `edge` runs from the page's rim inward. Each step decides what the line is
    and either blanks it and continues, or stops the walk:

    * a bare folio number is furniture wherever it falls in the band;
    * a line the volume repeats at this position is the running title;
    * two neighbouring lines whose JOINED text is repeated are one title the
      text layer broke in half -- `REI` over `LEAR` matches nothing alone;
    * an act heading SHARING ITS BAND WITH A FOLIO NUMBER is the running head
      quoting the act, while one that is not is the real heading and ends the
      walk;
    * anything else is the page's own text, and the band is over.

    THE FOLIO IS WHAT TELLS THE TWO ACT LINES APART, and picking the first
    occurrence instead was wrong. A volume is free to print `ACTO PRIMEIRO` in
    a table of contents, on a part-title, or in a half-title long before the
    act actually starts, and an order-based rule hands the slot to whichever
    came first and then blanks the genuine heading -- silently, because the
    scene simply stops being findable. What a running head actually IS, on
    every page of both measured volumes, is the act line set beside the page
    number; the real heading on page 24 of the 1919 Rei Lear carries `SCENA I`
    instead and no folio at all. That is a property of the page rather than a
    guess about document order, so it cannot be stolen by front matter.

    A PAGE CARRIES ONE RUNNING TITLE PER BAND, so the walk ends as soon as it
    has taken one -- folio numbers around it are free, a second title is not.
    Without that stop the Macbeth volume ate its own hero: its header is the
    single word `MACBETH`, and a page reading `MACBETH / 4 / MACBETH / Fala` had
    the header, the folio AND the genuine speech label blanked, because the
    header and the character are the same string and the walk had no reason to
    halt between them.

    THE BUDGET BELONGS TO THE BAND, NOT TO THE CALL, which is why it is passed
    in and handed back. On a short page the head and foot windows are the same
    physical band read from both ends, and a budget held per call gives that one
    band two: `MACBETH / 40 / MACBETH` had the head walk correctly SPARE the
    second label, and the foot walk -- starting fresh at exactly that line --
    delete it. Every page of six lost its real label that way. Sparing a line is
    a decision about the page, so the second reading has to inherit it.
    """
    step = 0
    band_carries_folio = band_carries_folio and filled_count > _PART_TITLE_LINES
    while step < len(edge):
        i = edge[step]
        text = lines[i].strip()
        identity, attached_folio = _edge_folio_parts(text)
        if _FOLIO.match(text):
            lines[i] = ""
            step += 1
            continue
        if attached_folio and identity in forms and not _HEADING_SHAPED.match(identity):
            lines[i] = ""
            step += 1
            title_taken = True
            continue
        # AN ACT LINE IS FREE LIKE A FOLIO, AND DOES NOT COUNT AS THE ONE TITLE,
        # because the 1912 Macbeth heads its pages with THREE pieces of one
        # running head -- `ACTO I`, `SCENA III`, `19` -- and spending the single
        # title on the act line stopped the walk before the rest of its own
        # header. The budget exists to protect a SPEAKER label that happens to
        # equal the header word; an act line is never a speaker.
        #
        # A SCENE LINE IS DELIBERATELY NOT FREE HERE, and that asymmetry is the
        # point. Blanking scene headings at page edges was tried and reverted
        # within the hour: it removes the running-head copies AND the real
        # heading that ends the scene, so Lear 1.1 ran on to 118 speeches and
        # Macbeth 1.3 picked up Duncan and Malcolm out of the scene after it.
        # A surviving header copy of the scene's own label is answered by
        # passing `--end-label`, which states the boundary instead of inferring
        # it -- see the note in `main`.
        if (_ACT_LINE.match(identity) and band_carries_folio
                and _normalise_heading(identity) in echoed):
            lines[i] = ""
            step += 1
            continue
        if title_taken:
            return True
        # THE PAIR IS TESTED BEFORE THE SINGLE LINE, because a split title's
        # halves usually vote on their own as well and the more specific match
        # has to win. Tested second, `REI` matched alone, the walk counted the
        # title as taken and stopped -- leaving `LEAR` standing as a speaker on
        # every verso, which is the exact defect the joined form exists to stop.
        if step + 1 < len(edge):
            first, second = sorted((i, edge[step + 1]))
            pair = "%s %s" % (lines[first].strip(), lines[second].strip())
            if pair in forms:
                lines[first] = lines[second] = ""
                step += 2
                title_taken = True
                continue
        if identity in forms and not _HEADING_SHAPED.match(identity):
            lines[i] = ""
            step += 1
            title_taken = True
            continue
        return title_taken
    return title_taken


def strip_running_titles(pages: list[str]) -> list[str]:
    """Delete the running title WHERE IT SITS, never wherever the string occurs.

    THE TITLE OF A PLAY IS USUALLY ALSO A CHARACTER IN IT, so the set above can
    never be used as a token blocklist. Domingos Ramos prints `MACBETH` at the
    head of all 240 pages of his Macbeth, and blocklisting that token reported
    `MACBETH x13 resolves to nobody` and handed the play's title character four
    speeches out of seventeen -- the same shape as the Lear/Kent/Gloucester
    deletion the position rule was written to fix, one layer further in.

    The header is furniture because of WHERE it is, not what it says, so it is
    removed by position -- only inside a page's own head and foot windows.
    Everywhere else that word is a man speaking.

    IT IS A RUN, NOT A FIXED DEPTH. Furniture occupies a contiguous band at the
    edge of the page and stops the moment the text begins, so each edge is
    WALKED INWARD and abandoned at the first line that is not furniture. A fixed
    three-line window instead reaches past the band into the page: Macbeth's
    header is one line, and a window deep enough for Lear's three-line verso
    deleted genuine `MACBETH` labels that happened to sit third, taking him from
    twelve speeches to nine.

    Blanking the line rather than deleting it keeps every later rule reading the
    same page shape this one did.

    KNOWN LIMIT, STATED RATHER THAN GUESSED AT: a title the text layer breaks
    into THREE lines is only partly handled -- the pair test rejoins two, and
    the third is caught only if it happens to vote on its own. No volume in this
    corpus prints one, so the machinery for it is not built.
    """
    # REJOIN SPLIT HEADINGS FIRST, AND VOTE ON THE REJOINED PAGES. Counting the
    # raw pages and then matching against rejoined ones let a bare `PRIMEIRO`
    # -- the orphaned second half of `ACTO / PRIMEIRO` -- reach the floor as an
    # ordinary word, because it is not heading-shaped on its own and so escaped
    # the heading exclusion. The two passes now read the same text.
    pages = [_SPLIT_HEADING.sub(r"\1 \2", page) for page in pages]
    forms = running_titles(pages)
    echoed = recurring_headings(pages)
    out: list[str] = []
    for page in pages:
        lines = page.splitlines()
        # WALK EACH LINE ONCE. On a page of five or fewer non-blank lines the
        # head and foot windows OVERLAP, and walking a shared line twice let the
        # second pass undo the first's decision.
        filled_count = sum(1 for line in lines if line.strip())
        edges = _page_edges(lines)
        # READ THE FOLIO OFF THE PAGE AS IT ARRIVED. The head walk blanks the
        # page number, so a flag computed inside the second call would see a
        # band that no longer has one and spare furniture the first walk had
        # already judged.
        folios = [any(_FOLIO.match(lines[i].strip()) or (
                          _edge_folio_parts(lines[i])[1] and (
                              _edge_folio_parts(lines[i])[0] in forms or
                              _normalise_heading(_edge_folio_parts(lines[i])[0]) in echoed))
                      for i in edge) for edge in edges]
        # ONE BAND, ONE BUDGET. Where the two windows overlap they are the same
        # physical band read from both ends, so the second reading inherits the
        # first's spend; where they are disjoint they are two bands and each
        # gets its own.
        shared = len(edges) == 2 and bool(set(edges[0]) & set(edges[1]))
        taken = False
        for edge, carries_folio in zip(edges, folios):
            # BOTH EDGES ALWAYS WALK, EVEN WHERE THEY OVERLAP. Skipping indices
            # the head pass had already seen was tried, to stop the second pass
            # undoing the first's decision -- but the skip broke the FOOTER page
            # outright. A page printed as text, then title, then folio is read
            # by the head walk as ordinary dialogue on its very first line, so
            # the walk returns having blanked nothing; with the window already
            # marked seen, the foot walk that would have recognised it never
            # ran, and the title survived on 25 pages of 25.
            taken = _strip_one_edge(lines, edge, forms, echoed, filled_count,
                                    carries_folio, taken if shared else False)
        out.append("\n".join(lines))
    return out


def slice_pages(pages: list[str], spec: str):
    """``(pages, why)`` for a `START-END` page window, or ``(None, why)``.

    THE PAGE IS THE ONLY STABLE ADDRESS IN A SCANNED VOLUME. A line index
    moves the moment a furniture rule or a PyMuPDF version changes, and a
    heading label does not identify a scene on its own: this corpus's volumes
    print `ESCENA PRIMERA .` five, six and ten times each, so asking for one
    by name returns whichever copy the matcher reaches first -- which is how a
    request for King Lear 1.1 came back with Act 2 Scene 1, and a request for
    the Tempest's act 3 came back spanning two plays.

    A BAD WINDOW FAILS LOUDLY AND THAT IS THE DESIGN. Ask for pages the
    volume does not have and the run stops; ask for the wrong ones and the
    scene heading is simply not in them, so `extract` refuses. Neither can
    store a wrong scene quietly, which is the failure this corpus cannot
    afford.
    """
    first, _, last = spec.partition("-")
    try:
        first = int(first)
        last = int(last) if last.strip() else first
    except ValueError:
        return None, ("--pages wants START-END as page indexes, e.g. 244-255; "
                      "got %r" % spec)
    if first > last:
        return None, "--pages %s runs backwards" % spec
    if not 0 <= first < len(pages) or last >= len(pages):
        return None, ("--pages %s is outside this volume's 0..%d"
                      % (spec, len(pages) - 1))
    return pages[first:last + 1], ("restricted to pages %d-%d of the volume"
                                   % (first, last))


def speeches_from_span(span: str, roster: set[str]) -> list[tuple[str, str]]:
    """``[(printed label, speech)]`` for every resolvable speaker in the span."""
    span = _RUNNING_HEADER.sub(" ", span)
    span = _CYRILLIC_WORD.sub(" ", span)
    # REMOVE A BARE DIRECTION BEFORE ANY LABEL IS CLAIMED, not after. Dropping
    # the (label, direction) PAIR afterwards was the first cut and it leaves the
    # speakers on either side ADJACENT -- Kent, [direction], Kent becomes Kent,
    # Kent -- which reads as a dropped speech and tripped the back-to-back
    # guard five times in one scene. Deleting the line means the label is never
    # created and the two real speeches stay separated by nothing at all.
    span = "\n".join("" if _STAGE_OPENER.match(l.strip()) else l
                     for l in span.splitlines())
    span = re.sub(r"(?m)^\s*\d{1,3}\s*$", " ", span)
    span = re.sub(r"[ \t]+", " ", span)

    # TWO SIGNALS, BECAUSE ONE EDITION USES BOTH. The names mostly arrive as
    # all-caps runs, and some arrive as a line of their own in TITLE case --
    # Domingos Ramos prints `ANGUS` and `Ross` in the same scene. A caps-only
    # rule silently glued Ross's two speeches onto whoever spoke before him,
    # which is the swallowing defect this corpus keeps re-learning. A line whose
    # WHOLE CONTENT resolves to a character is a speaker whatever its case.
    own_line = []
    for m in re.finditer(r"(?m)^[ \t]*(?P<tok>[^\n]{2,34}?)[ \t]*$", span):
        tok = m.group("tok").strip()
        if tok and resolve(tok, roster):
            own_line.append((m.start("tok"), m.end("tok"), tok))

    marks, rejected = [], collections.Counter()
    # ONE PRINTED SPELLING PER CHARACTER. A scanned page is not consistent about
    # its own punctuation -- Macbeth 1.3 prints `2.ª FEITICEIRA` five times and
    # `2. FEITICEIRA` once -- and two spellings of one witch would ship as two
    # characters with two voices. Everything that resolves to the same roster
    # name is rewritten to the spelling that appeared FIRST, which keeps the
    # translation's own words and unifies the variants. Same reasoning as
    # `canonicalise_labels` folding accents on the HTML path.
    canonical: dict[str, str] = {}
    for match in _CAPS_RUN.finditer(span):
        token = re.sub(r"\s+", " ", match.group("tok").strip())
        name = resolve(token, roster)
        if not name:
            rejected[token] += 1
            continue
        marks.append((match.start(), match.end(),
                      canonical.setdefault(name, clean_label(token))))

    # Merge the own-line hits in, dropping any that the caps pass already found
    # at the same place, and keep everything in document order.
    seen = {(s, e) for s, e, _ in marks}
    for start, end, tok in own_line:
        if (start, end) in seen or any(s <= start < e for s, e, _ in marks):
            continue
        name = resolve(tok, roster)
        # UPPER-CASE ON WRITE, ALWAYS. The corpus convention is an upper-case
        # label and `_is_upper_label` in the runtime selector enforces it: this
        # edition prints `ANGUS` and `Ross` in the same scene, and a title-case
        # label parses as no speaker at all, so Ross's lines would have merged
        # into whoever spoke before him at RUNTIME while the file looked fine.
        # Caught by the map-key guard rather than by reading.
        marks.append((start, end, canonical.setdefault(name, clean_label(tok))))
    marks.sort()

    if rejected:
        top = ", ".join("%s x%d" % (t, n) for t, n in rejected.most_common(8))
        print("[scan] all-caps runs that resolve to nobody (ignored): %s" % top)

    out = []
    for i, (start, end, token) in enumerate(marks):
        stop = marks[i + 1][0] if i + 1 < len(marks) else len(span)
        body = re.sub(r"\s+", " ", span[end:stop]).strip(" .,;:-")
        if body and not _STAGE_OPENER.match(body):
            out.append((token, body))

    # A BACK-TO-BACK RUN IS LEFT ALONE ON PURPOSE. Joining consecutive speeches
    # by one character was tried and reverted: it made the counts look clean and
    # DESTROYED THE ONLY SIGNAL that says a speaker was missed. When a label is
    # not recognised its text already belongs to whoever spoke last, so merging
    # cannot repair that -- it only hides the run that would have reported it.
    # Measured on Macbeth 1.3: merging took 49 speeches to 36 against 51 in the
    # French and Italian editions, which looked tidier and was further from the
    # truth. The scene-level guard in the suite reads these runs, and a scene
    # that trips it is held rather than smoothed.
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--iso", required=True)
    ap.add_argument("--play", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--stem", required=True,
                    help="English source stem, e.g. macbeth__act1_scene3")
    ap.add_argument("--act-label", default=None)
    ap.add_argument("--scene-label", required=True)
    ap.add_argument("--end-label", default=None)
    ap.add_argument("--reading-order", default="flat",
                    choices=("flat", "coordinates"),
                    help="how a page's text is ordered. `flat` is PyMuPDF's "
                         "own order and the default; `coordinates` rebuilds "
                         "each printed row from glyph baselines. The reader is "
                         "OPT-IN because it is proven per VOLUME, not globally: "
                         "an independent challenger accepted it for the two "
                         "Domingos Ramos volumes (zero differing speech bodies "
                         "against an image-verified reconstruction) and still "
                         "names four split cue rows in the Macpherson. Prove a "
                         "volume, then pin this on its rows.")
    ap.add_argument("--pages", default=None, metavar="START-END",
                    help="restrict the search to these PDF page indexes, "
                         "zero-based and inclusive, e.g. 244-255. THE PAGE IS "
                         "THE ONLY STABLE ADDRESS IN A SCANNED VOLUME: a line "
                         "index moves whenever a furniture rule or a PyMuPDF "
                         "version changes, and a heading label does not "
                         "identify a scene on its own -- this corpus's "
                         "volumes print `ESCENA PRIMERA .` five, six and ten "
                         "times each. Measured per scene from --probe.")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--probe", action="store_true",
                    help="list the act/scene headings this book actually prints "
                         "and stop. Run this FIRST for every new volume: the "
                         "wording is the transcriber's, not a convention, and "
                         "guessing it is how a scene search lands in the wrong "
                         "act. Macbeth prints `ACTO PRIMEIRO` as its heading "
                         "and `ACTO I - SCENA III` as its RUNNING HEADER, and "
                         "asking for the latter returned act two.")
    args = ap.parse_args(argv)

    leads = json.loads(io.open(os.path.join(BASE, "leads.json"),
                               encoding="utf-8").read())["leads"]
    key = (args.iso, args.play, args.scene)
    lead = next((r for r in leads
                 if (r.get("iso"), r.get("play"), r.get("scene")) == key
                 and not r.get("alt_of")), None)
    if lead is None:
        print("no lead row for %s" % (key,))
        return 2

    url = str(lead.get("url") or "").strip()
    print("[scan] %s" % urllib.parse.unquote(url)[:96])
    pages = strip_running_titles(pdf_text(url, reading_order=args.reading_order))
    # SLICE AFTER THE FURNITURE VOTE, NEVER BEFORE. `running_titles` decides
    # what is furniture by how often a line repeats across the WHOLE volume,
    # so a handful of pages cannot tell a running head from a speaker: on a
    # five-page window the floor is 4 and the play's lead clears it. Strip on
    # the whole book, then narrow.
    if args.pages:
        pages, why = slice_pages(pages, args.pages)
        if pages is None:
            print("[scan] %s" % why)
            return 2
        print("[scan] %s" % why)
    print("[scan] %d page(s) of text layer" % len(pages))

    if args.probe:
        flat = _RUNNING_HEADER.sub(" ", "\n".join(pages))
        flat = _SPLIT_HEADING.sub(r"\1 \2", flat)
        heads = collections.Counter()
        for m in re.finditer(
                r"(?im)^\s*((?:ACTO|ATTO|ACT|ESCENA|SCENA|SCENE)[^\n]{0,26})\s*$",
                flat):
            heads[re.sub(r"\s+", " ", m.group(1).strip())] += 1
        print("[scan] headings this volume prints, once the running header is out")
        print("[scan] a heading that appears ONCE is the real one; a repeat is "
              "furniture")
        for head, n in sorted(heads.items(), key=lambda kv: (-kv[1], kv[0])):
            print("        %-30s x%d%s" % (head, n, "   <-- likely real" if n == 1 else ""))
        return 0

    # STRIP THE RUNNING HEADER BEFORE LOCATING THE SCENE, NOT AFTER. Every page
    # of these volumes repeats `ACTO II - SCENA III 9` at the top, which is an
    # act heading and a scene heading on one line, dozens of times. Left in, the
    # heading search locks onto the first REPEAT rather than the real heading:
    # asked for Macbeth act 1 scene 3 it returned act 2 scene 3, the porter
    # scene, with Macduff and Lady Macbeth in it. That is the wrong-scene
    # failure, and only the roster filter downstream stopped it shipping --
    # three of nine labels resolved because the rest are not in this scene's
    # cast. A guard catching it is not a reason to leave the cause in.
    # REFUSE TO GUESS A BOUNDARY THIS VOLUME WILL LIE ABOUT. A scene heading is
    # never blanked as furniture, because the copy that ENDS the scene is the
    # same shape as the copies in the running head -- delete them all and the
    # scene runs on into the next one. The cost is that a volume printing its
    # scene label across the top of every page hands the scene finder a false
    # boundary on page two, and the result is a SHORT scene that looks entirely
    # healthy: no error, a plausible cast, and `alignment_confidence: 1.0`
    # written over the top of it. The length guard downstream only catches a
    # fragment of under four lines, so a fifty-line truncation ships silently.
    # The caller knows the real boundary, so ask for it rather than infer it.
    if (not args.end_label
            and _normalise_heading(args.scene_label) in recurring_headings(pages)):
        print("[scan] %s is this volume's RUNNING HEAD, printed at the edge of "
              "%d or more pages -- so the scene finder would stop at the next "
              "page rather than the next scene, and store the fragment as a "
              "whole scene.\n[scan] re-run with --end-label set to the heading "
              "that follows this one (--probe lists them)."
              % (args.scene_label.strip(), _ECHO_FLOOR))
        return 1

    flat = _RUNNING_HEADER.sub(" ", "\n".join(pages))
    flat = _SPLIT_HEADING.sub(r"\1 \2", flat)
    lines = flat.splitlines()
    body, reason = V.extract(lines, None, args.act_label, args.scene_label,
                             end_label=args.end_label)
    if not body:
        print("[scan] no scene: %s" % reason)
        return 1
    print("[scan] scene span: %d chars" % len(body))

    roster = roster_for(args.stem)
    if not roster:
        print("[scan] no English roster at %s" % args.stem)
        return 1
    pairs = speeches_from_span(body, roster)
    counts = collections.Counter(label for label, _ in pairs)
    print("[scan] %d speeches / %d distinct labels" % (len(pairs), len(counts)))
    for label, n in counts.most_common():
        print("        %-22s x%-3d -> %s" % (label, n, resolve(label, roster)))

    # A SWALLOWED LABEL IS THE ONE DEFECT THIS CORPUS HAS SHIPPED TWICE, AND
    # IT COSTS ONE REGEX TO REFUSE. Both instances were a word the typesetter
    # broke welded onto the speaker label that followed it -- `horriBANQUO` in
    # the 1912 Macbeth and `AfasKent` in the 1919 Rei Lear -- and each ate a
    # label, filed a whole speech under the wrong character, and left a
    # nonsense word for a voice engine to read aloud. Both shipped at
    # `alignment_confidence: 1.0` and neither was visible to any count.
    #
    # In the languages this lane vendors, a lower-case letter is never
    # followed directly by an upper-case one inside a token, so the weld has
    # a shape and the shape is refusable. Measured over all 41 vendored
    # scenes: two hits before the fixes, zero after. It runs on the SPEECH
    # BODIES only -- a label may legitimately carry capitals and punctuation
    # (`D . PED`, `1.ª FEITICEIRA`).
    welds = [(label, m.group(0)) for label, speech in pairs
             for m in [_INTERIOR_WELD.search(speech)] if m]
    if welds:
        print("[scan] REFUSING to write: %d speech(es) carry a swallowed "
              "label -- a lower-case letter running straight into capitals, "
              "which is how a speaker's name gets eaten and their speech "
              "filed under the previous character:" % len(welds))
        for label, hit in welds[:6]:
            print("        %-22s ...%s..." % (label, hit))
        print("[scan] read the page: the capitals are almost certainly the "
              "next speaker's cue.")
        return 1

    if not args.write:
        print("[scan] (no --write; nothing stored)")
        return 0
    text = "\n".join("%s: %s" % (label, speech) for label, speech in pairs)
    rel = "%s/%s_%s.txt" % (args.iso, args.play, args.scene.replace(".", "_"))
    dest = os.path.join(BASE, rel)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    stored = text + "\n"
    io.open(dest, "w", encoding="utf-8", newline="\n").write(stored)
    digest = hashlib.sha256(stored.encode("utf-8")).hexdigest()
    print("[scan] wrote %s (%d chars)" % (rel, len(stored)))

    # THE MANIFEST ROW IS WRITTEN HERE, not by hand. A transcribed row carries
    # exactly the fields a fetched one does -- `raw_sha256` signs the STORED
    # TEXT on both paths, which is what made this lane possible without a schema
    # change -- plus three that say where it came from. `source_url` already
    # points at the scan, so nothing about a fetched row changes meaning.
    man_path = os.path.join(BASE, "manifest.json")
    man = json.loads(io.open(man_path, encoding="utf-8").read())
    row = {
        "iso": args.iso, "play": args.play, "scene": args.scene, "file": rel,
        "translator": lead.get("translator", ""),
        "translator_death_date": lead.get("translator_death_date", ""),
        "translation_first_published": lead.get("translation_first_published", ""),
        "transcription_license": lead.get("transcription_license", ""),
        "source_url": url,
        "revision_id": "sha256:" + digest[:16],
        "raw_sha256": digest,
        "verdict": "READY",
        "alignment_confidence": 1.0,
        "edition_label": " / ".join(x for x in (args.act_label, args.scene_label) if x),
        "speaker_labels": len(pairs),
        "distinct_speakers": len(counts),
        "speaker_map": {label: {"spoken": resolve(label, roster),
                                "roster": resolve(label, roster)}
                        for label in counts},
        "transcription_method": "pdf_text_layer",
        "transcription_date": "2026-09-19",
        # THE READER IS PART OF THE PROVENANCE. A row vendored under the
        # coordinate reader is a different extraction from the same text
        # layer, and the string here is how a later window tells which one
        # produced the file it is looking at.
        "extractor": "scripts/otr_vendor_scan.py --reading-order %s"
                     % args.reading_order,
    }
    man["scenes"] = [s for s in man["scenes"]
                     if (s["iso"], s["play"], s["scene"]) != key]
    man["scenes"].append(row)
    man["scenes"].sort(key=lambda s: (s["iso"], s["play"], s["scene"]))
    io.open(man_path, "w", encoding="utf-8", newline="\n").write(
        json.dumps(man, ensure_ascii=False, indent=2) + "\n")
    print("[scan] manifest: %d scenes" % len(man["scenes"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
