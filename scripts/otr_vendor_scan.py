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
_LINE_BREAK_HYPHEN = re.compile(r"(\w)[-‐‑]\s*\n\s*(\w)")


def pdf_text(url: str) -> list[str]:
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
            out.append(_LINE_BREAK_HYPHEN.sub(r"\1\2", page.get_text() or ""))
        except Exception:                      # noqa: BLE001 - a bad page is empty
            out.append("")
    doc.close()
    return out


def roster_for(stem: str) -> set[str]:
    import nodes._otr_roster_gender as RG

    path = os.path.join(SOURCES, stem + ".txt")
    if not os.path.exists(path):
        return set()
    return {r["name"] for r in RG.load_roster_characters(path)}


def resolve(token: str, roster: set[str]) -> str | None:
    """The English roster name this printed label stands for, or None.

    THE ROSTER IS THE FILTER AS WELL AS THE ANCHOR. A running header, a folio
    number and a translator's initials are all-caps too; none of them resolves,
    so none of them becomes a speaker.
    """
    folded = {fold(n): n for n in roster}
    key = fold(token).strip(" .")
    if key in folded:
        return folded[key]

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


def speeches_from_span(span: str, roster: set[str]) -> list[tuple[str, str]]:
    """``[(printed label, speech)]`` for every resolvable speaker in the span."""
    span = _RUNNING_HEADER.sub(" ", span)
    span = _CYRILLIC_WORD.sub(" ", span)
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
                      canonical.setdefault(name, token.upper())))

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
        marks.append((start, end, canonical.setdefault(name, tok.upper())))
    marks.sort()

    if rejected:
        top = ", ".join("%s x%d" % (t, n) for t, n in rejected.most_common(8))
        print("[scan] all-caps runs that resolve to nobody (ignored): %s" % top)

    out = []
    for i, (start, end, token) in enumerate(marks):
        stop = marks[i + 1][0] if i + 1 < len(marks) else len(span)
        body = re.sub(r"\s+", " ", span[end:stop]).strip(" .,;:-")
        if body:
            out.append((token, body))
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
    ap.add_argument("--write", action="store_true")
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
    pages = pdf_text(url)
    print("[scan] %d page(s) of text layer" % len(pages))

    # STRIP THE RUNNING HEADER BEFORE LOCATING THE SCENE, NOT AFTER. Every page
    # of these volumes repeats `ACTO II - SCENA III 9` at the top, which is an
    # act heading and a scene heading on one line, dozens of times. Left in, the
    # heading search locks onto the first REPEAT rather than the real heading:
    # asked for Macbeth act 1 scene 3 it returned act 2 scene 3, the porter
    # scene, with Macduff and Lady Macbeth in it. That is the wrong-scene
    # failure, and only the roster filter downstream stopped it shipping --
    # three of nine labels resolved because the rest are not in this scene's
    # cast. A guard catching it is not a reason to leave the cause in.
    flat = _RUNNING_HEADER.sub(" ", "\n".join(pages))
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
        "extractor": "scripts/otr_vendor_scan.py",
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
