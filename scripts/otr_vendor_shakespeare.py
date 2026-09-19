#!/usr/bin/env python
"""Turn a located lead into a vendored scene file the writer can perform.

    python scripts/otr_vendor_shakespeare.py               # what is extractable
    python scripts/otr_vendor_shakespeare.py --write       # vendor them

WHAT THIS IS FOR. `leads.json` records where a translation LIVES; nothing in
the pipeline can read a URL. This opens the page, extracts the target scene,
normalises the speaker labels to `NAME:`, writes
`config/source_banks/shakespeare/translations/<iso>/<play>_<scene>.txt`, and
adds the manifest row `_otr_verbatim_corpus.load_manifest` validates.

Running it with no flag WRITES NOTHING. It reports, per lead, whether the target act and scene
are actually on the page -- because a lead can name the right work and the
wrong page. Measured on the first run: `it/tempest 3.1` pointed at
`La_tempesta_(Shakespeare-Maffei)/Atto_primo`, which holds Act I only, so
extracting "3.1" from it would have vendored Act I scene 1 under the name of
Act III scene 1. A wrong scene that looks right is the worst outcome this
corpus can produce, which is why the check runs first and separately.

EDITIONS DISAGREE ABOUT WHAT "3.2" MEANS. Three conventions are already in the
set: Hugo numbers scenes CONTINUOUSLY with no act headings at all, while
Rusconi and Marquez reset per act. So the scene is located by the edition's OWN
label, recorded on the lead by the operator's verification pass, and never by
counting headings down the page.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
import os
import re
import sys
import unicodedata
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from nodes import _otr_verbatim_corpus as CORPUS  # noqa: E402

LEADS = os.path.join(_REPO, "config", "source_banks", "shakespeare",
                     "translations", "leads.json")
OUT_ROOT = os.path.join(_REPO, "config", "source_banks", "shakespeare",
                        "translations")
CACHE = os.path.join(_REPO, "tmp", "corpus_cache")
USER_AGENT = ("OTR-corpus-vendor/1.0 (ComfyUI-OldTimeRadio; "
              "https://github.com/jbrick2070/ComfyUI-OldTimeRadio)")

#: Leads whose url serves readable text. A page scan needs transcription first
#: and is not this script's job.
TEXT_FORMS = ("whole_work_single_page", "act_page", "scene_page")


def fetch(url):
    """The page bytes, cached. Never re-scrapes during iteration."""
    os.makedirs(CACHE, exist_ok=True)
    name = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24] + ".txt"
    path = os.path.join(CACHE, name)
    if os.path.isfile(path):
        return io.open(path, encoding="utf-8").read()
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=45) as response:
        raw = response.read()
        charset = response.headers.get_content_charset() or "utf-8"
    body = raw.decode(charset, errors="replace")
    io.open(path, "w", encoding="utf-8", newline="\n").write(body)
    return body


def to_text(markup):
    """Rendered HTML to line-broken text.

    Block elements END THEIR LINE. A strip that does not break lines leaves
    every speaker label inline with its speech, and every line-anchored rule --
    the heading finder, the label counter -- as blind as it was on the raw HTML.

    The parameter is `markup`, not `html`: it used to be `html`, which SHADOWED
    the stdlib module of that name and turned `html.unescape(body)` into an
    attribute lookup on a str.
    """
    body = re.sub(r"(?is)<(script|style|template)\b.*?</\1>", " ", markup)
    body = re.sub(r"(?s)<!--.*?-->", " ", body)
    # READ WHO SPEAKS OFF THE PAGE BEFORE THE PAGE IS THROWN AWAY. Every edition
    # in the set marks its speakers; once the tags are gone that fact cannot be
    # recovered, only guessed at -- see `mark_speakers`.
    body = strip_direction_parentheticals(body)
    body = mark_speakers(body)
    body = re.sub(r"(?i)</(div|p|li|tr|h[1-6]|span)\s*>", "\n", body)
    body = re.sub(r"(?i)<br\s*/?>", "\n", body)
    body = re.sub(r"(?s)<[^>]+>", "", body)
    # UNESCAPE EVERYTHING, not a hand-list. The hand-list missed `&#91;` and
    # `&#93;` -- Wikisource's footnote brackets -- and they survived into the
    # vendored Macbeth as "(&#91; 9&#93; )", which the announcer would have
    # READ ALOUD as "ampersand hash ninety-one".
    body = html.unescape(body)
    # Footnote markers and page breaks are the EDITION's apparatus, not the
    # translator's words, and every one of them is READ ALOUD if it survives.
    # Three forms are in this corpus and the first cut caught only one:
    #   [9]                  -- the bracketed footnote
    #   (19)                 -- Hugo's parenthesised note, five of them in Lear
    #   [p. 39 modifica ]    -- Wikisource's page-break link, INSIDE a speech,
    #                           which the announcer would read as "p 39 modifica"
    body = re.sub(r"\[\s*\d{1,3}\s*\]", "", body)
    body = re.sub(r"\[\s*p\.\s*\d{1,4}\s*modifica\s*\]", "", body, flags=re.I)
    body = re.sub(r"\(\s*\d{1,3}\s*\)", "", body)
    body = re.sub(r"\(\s*\)", "", body)
    body = re.sub(r"[ \t ]+", " ", body)
    return re.sub(r"\n\s*\n+", "\n", body)


#: Wraps a speaker name the PUBLISHER marked, so the label rules below read a
#: fact off the page instead of guessing one from prose.
SPEAKER_MARK = "\x01"

#: A stage direction / sound cue. Every edition in the set sets its apparatus in
#: SMALLER TYPE than the dialogue; the exact percentage is the transcriber's
#: taste and nothing else.
#:
#: THE PERCENTAGE IS NOT A GATE (operator 2026-09-19: "I don't want font size
#: blocking our workflow"). This read `9[05]%` -- two hard-coded values, taken
#: from the two editions that happened to be vendored first. It found ZERO on
#: the transcribed Portuguese Hamlet, whose stage directions are `80%` spans
#: (`(Retira-se.)`, `(O gallo canta.)`), so that edition would have PERFORMED
#: every one of them. Discovering a third convention should not mean a code
#: change and a fourth should not mean another; "smaller than the body text" is
#: the rule the publishers are actually following.
#:
#: 60-99% is the whole band of small type. 100% and above is body text or a
#: heading and is never apparatus.
_DIRECTION_BLOCK = re.compile(
    r'(?is)<(div|span|p)\b[^>]*font-size:\s*[6-9]\d%[^>]*>(?P<body>.*?)</\1\s*>')

#: How each publisher marks a speaker. All three converge on small caps; only
#: the spelling of the markup differs.
_SPEAKER_SPANS = (
    # Wikisource (Hugo): class="sc", the name often rendered lowercase.
    re.compile(r'(?is)<span\b[^>]*class="sc"[^>]*>(?P<name>[^<]{1,40})</span>'),
    # Gutenberg (Marquez): an inline small-caps style, name carries its period.
    re.compile(r'(?is)<span\b[^>]*font-variant:\s*(?:all-)?small-caps[^>]*>'
               r'(?P<name>[^<]{1,40})</span>'),
    # Wikisource (Rusconi): an italic name, with the ordinal as a superscript --
    # `1<sup>a</sup> <i>Strega</i>` is "1a Strega", three distinct witches.
    # THE PARENTHETICAL IS PART OF THE LABEL, NOT THE SPEECH. An aside prints
    # as `<i>Macbeth</i> (<i>fra se</i>).`, and requiring the period to follow
    # the name directly missed every one of them -- Macbeth's "Thane di Glamis
    # e di Cawdor!" stayed inside Angus's speech. The qualifier is matched so
    # the label is found, and dropped because it is stage business.
    re.compile(r'(?is)(?P<ord>\d\s*<sup>\s*[ao]\s*</sup>\s*)?'
               r'<i>(?P<name>[^<]{1,40})</i>'
               r'\s*(?:\((?:<[^>]+>|[^()<])*\))?\s*\.'),
)


#: A parenthetical whose CONTENT IS WHOLLY ITALIC. Rusconi and Marquez set
#: stage business that way -- `(<i>entra Rosse</i>)`, `(<i>escono</i>)` -- and
#: the editions are consistent about it.
#:
#: THE BODY IS NOT LENGTH-CAPPED, AND THE CAP THAT USED TO BE HERE WAS A BUG.
#: It read `{1,80}`, which is a plausible-looking number and nothing more: a
#: stage direction is a sentence when the edition wants one, and Rusconi's
#: Macbeth has two past 80 -- the 89-character `rimane alcuni istanti assorto
#: in profonda meditazione, quindi si volge ad Angus e a Rosse`, which survived
#: into the vendored `it/macbeth 1.3` and would have been PERFORMED IN
#: MACBETH'S OWN VOICE mid-speech, and a 163-character scene setting. The
#: failure is silent in the worst direction: a direction too long for the cap
#: is not flagged, it is simply kept and spoken.
#:
#: What actually bounds this match is `[^<]`, which cannot cross a tag, so the
#: body is confined to a single text run inside one `<i>` that is immediately
#: wrapped in parentheses. That is the real anchor. A character count on top of
#: it bought nothing and cost a wrong mouth.
_ITALIC_PARENTHETICAL = re.compile(
    r"\(\s*<i>(?P<body>[^<]+)</i>\s*\)", re.I | re.S)


def strip_direction_parentheticals(markup):
    """Drop `(<i>entra Rosse</i>)`, keep `(car cette partie du monde...)`.

    A BLANKET `()` STRIP WOULD DELETE REAL DIALOGUE. The verbatim cleaner keeps
    parenthetical words on purpose, because Folger prints "(God shield us!)"
    INSIDE Bottom's line, and Hugo does the same in French -- Horatio's "(car
    cette partie du monde connu l'estimait pour tel)" is spoken. But Folger
    puts stage business in SQUARE brackets, which the selector already strips
    before parsing, while the colon editions put it in round ones. So the
    vendored lane inherited an assumption that is false for its own sources and
    would have performed `entrano Rosse e Angus` and `le streghe scompariscono`
    aloud.

    The page separates the two and we do not have to guess: stage business is
    ITALIC, dialogue is not. Same rule as the speaker marks, read at vendor
    time while the markup still exists -- once it is plain text the distinction
    is gone and only a word-list could stand in for it.
    """
    return _ITALIC_PARENTHETICAL.sub(" ", markup)


def mark_speakers(markup):
    """Tag the speaker names the EDITION marked, and drop its stage business.

    WHY THIS EXISTS. The first cut threw the markup away and then guessed the
    speakers back out of the plain text, with a rule that a name must RECUR.
    That guess cast sound cues as people (FANFARES got a voice in King Lear)
    and, far worse, MISSED Hugo's inline labels -- he prints them lowercase
    with a comma qualifier, `cordelia , a part.`, which the comma rule
    discarded, leaving Cordelia's aside INSIDE Goneril's speech. A speech
    count cannot see a wrong mouth; that is why this reads the page instead.

    ORDER MATTERS AND IS NOT OBVIOUS. A stage direction NAMES the people who
    enter, and marks them the same way: `<div style="font-size:90%">Fanfares.
    Entrent <span class="sc">Macbeth</span>...`. So the direction blocks are
    neutralised FIRST -- their text is kept as narration, their speaker markup
    is not -- and only the spans that survive are speakers. Reversing these two
    steps casts every entrance as a speaking part.
    """
    def _strip_direction(match):
        # DROP THE DIRECTION, DO NOT KEEP IT AS NARRATION. Keeping the words
        # put them back in a mouth: an unlabelled line merges into the PENDING
        # speech, so `GLOCESTER: ... Le roi vient. Fanfares. Entrent` had
        # Glocester announcing his own trumpets, and Lear reading "A Cordelia."
        # aloud. A vendored file is the translator's SPOKEN text labelled by
        # speaker; stage business is the edition's apparatus and is not
        # performed. Same rule for the qualifier on a label ("a part",
        # "montrant Edmond") -- it was never dialogue either.
        return "\n"

    def _mark(match):
        # `groupdict` rather than `group("ord")`: only the Rusconi pattern has
        # that group, and asking the others for it by name is an error.
        name = match.group("name").strip()
        ordinal = re.sub(r"(?s)<[^>]+>|\s+", "",
                         match.groupdict().get("ord") or "")
        full = ("%s %s" % (ordinal, name)).strip()
        return "\n%s%s%s " % (SPEAKER_MARK, full, SPEAKER_MARK)

    body = _DIRECTION_BLOCK.sub(_strip_direction, markup)
    for pattern in _SPEAKER_SPANS:
        body = pattern.sub(_mark, body)
    return body


def _fold(text):
    text = unicodedata.normalize("NFKD", str(text or "")).lower()
    return "".join(c for c in text if not unicodedata.combining(c))


def find_label(lines, label):
    """Index of the line matching the edition's OWN scene/act label, or -1.

    THE PREFIX FALLBACK MUST NOT EAT A LONGER NUMERAL. "SCENA II" is a prefix
    of "SCENA III", and the first cut of this matched the wrong one: asked for
    Tempest 1.2 it returned SCENA III, Ariel's entrance, which would have been
    vendored as scene 2 -- text that reads perfectly and is the wrong scene,
    the single failure this corpus cannot afford. So a prefix match is only
    allowed when the character that follows is NOT a letter or digit, i.e. the
    label ended and what follows is punctuation or chrome.
    """
    want = _fold(label).strip().rstrip(".")
    if not want:
        return -1
    for i, line in enumerate(lines):
        if _fold(line).strip().rstrip(".") == want:
            return i
    for i, line in enumerate(lines):          # tolerate trailing chrome
        folded = _fold(line).strip()
        if not folded.startswith(want):
            continue
        rest = folded[len(want):]
        if rest and (rest[0].isalnum() or rest[0] in "一-鿿"):
            continue                          # SCENA II must not match III
        return i
    return -1


#: What each edition calls the act and scene we want. Filled from the
#: operator's verification pass -- he opened these pages and read the labels
#: off them. A label here is a MEASUREMENT, never an inference from the
#: Folio's numbering, because the editions disagree with the Folio and with
#: each other.
#: (play_anchor, act_label, scene_label). The play anchor is REQUIRED on a
#: multi-play volume and None on a single-play page. Gutenberg 59686 is
#: "Dramas de Guillermo Shakespear", a COLLECTION: without an anchor the
#: first "ACTO III / ESCENA II" in the book belongs to JULIO CESAR, and it
#: was vendored as As You Like It on the first run.
EDITION_LABELS = {
    ("it", "macbeth", "1.3"): (None, "ATTO PRIMO", "SCENA III"),
    ("it", "tempest", "1.2"): (None, "ATTO PRIMO", "SCENA II"),
    ("es", "as_you_like_it", "3.2"): (None, "ACTO III", "ESCENA II"),
    # Hugo numbers scenes CONTINUOUSLY and prints no act headings at all, so
    # there is nothing to scope to above the scene.
    ("fr", "hamlet", "1.1"): (None, None, "SC\u00c8NE I"),
    ("fr", "king_lear", "1.1"): (None, None, "SC\u00c8NE I"),
    ("zh", "midsummer", "3.1"): (None, None, "\u7b2c\u4e00\u573a"),
}


def extract(lines, play_label, act_label, scene_label):
    """``(text, reason)`` -- the scene's lines, or "" and why not.

    Stops at the NEXT heading of the same shape. Returns "" rather than a
    guess: a wrong scene reads perfectly and is the one failure this corpus
    cannot afford.
    """
    start = 0
    if play_label:
        # A COLLECTION holds several plays and every one of them has an
        # ACTO III. Scope to the play before looking for the act, or the
        # first match in the book wins and it is the wrong play.
        play_at = find_label(lines, play_label)
        if play_at < 0:
            return "", "play heading %r not on the page" % play_label
        start = play_at
    if act_label:
        act_at = find_label(lines[start:], act_label)
        if act_at < 0:
            return "", "act heading %r not on the page" % act_label
        start += act_at
    scene_at = find_label(lines[start:], scene_label)
    if scene_at < 0:
        return "", "scene heading %r not found%s" % (
            scene_label, " under %s" % act_label if act_label else "")
    scene_at += start
    # The next heading of EITHER shape ends the scene.
    stem = re.match(r"^\s*(\S+)", scene_label)
    stem = _fold(stem.group(1)) if stem else ""
    end = len(lines)
    for j in range(scene_at + 1, len(lines)):
        folded = _fold(lines[j]).strip()
        if stem and folded.startswith(stem):
            end = j
            break
        if act_label and folded.startswith(_fold(act_label.split()[0])):
            end = j
            break
    body = [l.rstrip() for l in lines[scene_at:end]]
    body = [l for l in body if l.strip()]
    if len(body) < 4:
        return "", "only %d lines between headings" % len(body)
    return "\n".join(body), ""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true",
                        help="vendor the extractable scenes (default: check only)")
    parser.add_argument("--iso", action="append", default=None)
    args = parser.parse_args()

    leads = json.load(io.open(LEADS, encoding="utf-8"))["leads"]
    rows = []
    for lead in leads:
        key = (lead.get("iso"), lead.get("play"), lead.get("scene"))
        if lead.get("excluded") or lead.get("url_is") not in TEXT_FORMS:
            continue
        if lead.get("alt_of"):
            # One scene, one vendored file. An alternate writes the same
            # path and the LAST one wins silently -- on the first run a
            # Gutenberg collection overwrote a verified Wikisource
            # extraction with a different play. Alternates stay in
            # leads.json as backups and are vendored deliberately, never
            # as a side effect.
            print("  ALT   %-3s %-16s %-5s  alternate source, not vendored"
                  % key)
            continue
        if not str(lead.get("url") or "").strip():
            continue
        if args.iso and lead.get("iso") not in args.iso:
            continue
        if key not in EDITION_LABELS:
            print("  SKIP  %-3s %-16s %-5s  no edition label recorded"
                  % key)
            continue
        try:
            html = fetch(lead["url"])
        except Exception as exc:  # noqa: BLE001
            print("  FETCH %-3s %-16s %-5s  %s: %s"
                  % (key + (type(exc).__name__, str(exc)[:40])))
            continue
        text = to_text(html)
        lines = text.splitlines()
        body, reason = extract(lines, *EDITION_LABELS[key])
        # MEASURE THE ARTIFACT, NOT AN INTERMEDIATE. The gate counts `NAME:`
        # labels, and the extracted body does not carry them yet -- since the
        # speakers became publisher MARKS the raw body scores almost nothing,
        # and King Lear read as 27 speeches where the stored file has 87. Count
        # the normalised text, which is the text that will be written.
        labels, speakers = (CORPUS.speaker_label_stats(normalise_labels(body))
                            if body else (0, 0))
        if not body:
            print("  NO    %-3s %-16s %-5s  %s" % (key + (reason,)))
            continue
        ok = labels >= CORPUS.MIN_SPEAKER_LABELS and speakers >= 2
        print("  %-5s %-3s %-16s %-5s  %5d chars, %3d speeches / %2d speakers"
              % ("OK" if ok else "THIN", key[0], key[1], key[2],
                 len(body), labels, speakers))
        if ok:
            rows.append((lead, key, body, labels, speakers))

    print("\n  %d scene(s) extractable" % len(rows))
    if not args.write:
        print("  (--check only; nothing written)")
        return 0
    return write_rows(rows)



#: A vendored file uses ONE label shape so the passage selector stays one
#: rule: `NAME:` at line start, the spec's own advice. The editions use
#: `Orlando.\n—`, `kent.`, `1a Strega.`, `波　` -- normalised on the way in
#: rather than taught to the parser.
#: Punctuation that proves a candidate "name" is really a SENTENCE. A speaker
#: label does not contain a comma or a full stop; a line of dialogue does.
#:
#: THE ASCII HALF OF THIS LIST WAS THE WHOLE LIST, AND IT PUT BOTTOM'S SPEECH IN
#: SNOUT'S MOUTH. `_LABEL_SHAPES[0]` accepts the FULLWIDTH colon as a separator,
#: so on Zhu Shenghao's Chinese Midsummer it matched everything up to one --
#: `波　列位，你们得好好想一想` -- and offered that whole clause as a name. The
#: fullwidth comma inside it is the tell, and `,;--!?` could not see it, because
#: U+FF0C is not U+002C. The name then occurred once, the recurrence rule
#: demoted the line to continuation, and Bottom's entire speech was performed by
#: Snout:
#:     司: 咱担保她们一定会吓怕。 波　列位，你们得好好想一想：...
#: Rejecting the clause lets `_LABEL_SHAPES[3]` have the line, which reads the
#: ideographic space correctly and returns `波` -- the right speaker.
#:
#: The two length guards above cannot cover this and are near-useless on CJK:
#: `split()` finds no spaces in Chinese, so the word count is always 1, and 24
#: characters is a long sentence in a language that does not space its words.
#: Punctuation is the signal that survives the script change.
_NOT_IN_A_NAME = ",;—–!?，、；！？。…"

_LABEL_SHAPES = (
    re.compile(r"^(?P<name>[^\n:：]{1,40})[:：][ \t\u3000]?"),
    re.compile(r"^(?P<name>[A-Za-z\u00c0-\u024f0-9][^\n.]{0,38})\.\s*$"),
    re.compile(r"^(?P<name>[A-Za-z\u00c0-\u024f0-9][^\n.]{0,38})\.\s+(?=\S)"),
    re.compile(r"^(?P<name>[^\n\u3000]{1,12})\u3000"),
)


def _label_candidates(body):
    """How often each candidate name appears. Two passes, because one is not
    enough to tell a speaker from a stage direction.

    A SPEAKER RECURS; CHROME APPEARS ONCE. The length guard below cannot catch
    a SHORT stage direction -- "Un bosco" is two words and eight characters --
    and the first pass duly cast the Macbeth setting as a character called
    UN BOSCO, the Spanish one as EL BOSQUE, and Hamlet's "Francisco est en
    faction" as EST EN FACTION. Each would have been SPOKEN aloud with a voice
    assigned. Requiring a name to appear at least twice is the same rule the
    acceptance gate already uses to tell a scene from a cast list.

    The cost is a genuine one-line part demoted to prose. That is the safe
    direction: prose is read as narration, an invented character is read as a
    person who is not in the play.
    """
    counts = {}
    for raw in body.splitlines():
        name = _name_of(raw.strip())
        if name:
            counts[name] = counts.get(name, 0) + 1
    return counts


#: A name the PUBLISHER marked, sitting at the head of its line.
_MARKED_LABEL = re.compile(
    r"^%s(?P<name>[^%s\n]{1,40})%s\s*" % (SPEAKER_MARK, SPEAKER_MARK,
                                          SPEAKER_MARK))


def _marked_name(line):
    """The speaker name the edition itself marked on this line, or "".

    A MARKED NAME IS A FACT AND SKIPS EVERY GUESS BELOW. The heuristics exist
    only for a page whose markup we have not taught, and each of them costs a
    real speaker: the comma rule drops Hugo's `cordelia , a part.`, and the
    recurrence rule drops anyone who speaks once. Neither may veto the page.
    """
    match = _MARKED_LABEL.match(line)
    if not match:
        return ""
    name = match.group("name").strip().rstrip(".:　").strip()
    return name.upper() if name else ""


def _name_of(line):
    """The speaker name this line opens with, or "" -- shape rules only."""
    marked = _marked_name(line)
    if marked:
        return marked
    for shape in _LABEL_SHAPES:
        m = shape.match(line)
        if not m:
            continue
        name = m.group("name").strip().rstrip(".:　")
        if not name or CORPUS._NOT_A_NAME.search(name):
            continue
        if len(name) > 24 or len(name.split()) > 4:
            continue
        if any(ch in name for ch in _NOT_IN_A_NAME):
            continue
        return name.upper()
    return ""


def normalise_labels(body):
    """Rewrite every speaker label to `NAME:` and join it to its speech.

    WHEN THE PAGE MARKED ITS SPEAKERS, THE PAGE WINS AND THE GUESSES ARE OFF.
    The recurrence rule below is a stand-in for a fact, and it costs a real
    part every time it fires -- a character who speaks once is demoted to
    narration. It is worth paying only on a publisher whose markup we have not
    taught yet, so it applies to an UNMARKED page and never to a marked one.
    """
    marked = SPEAKER_MARK in body
    recurring = {n for n, c in _label_candidates(body).items() if c >= 2}
    out, pending = [], None
    for raw in body.splitlines():
        line = raw.strip()
        if not line:
            continue
        matched = None
        name = _name_of(line)
        if marked:
            # A marked page has exactly one kind of label, and a line without
            # the mark is narration however much it looks like a name.
            marked_name = _marked_name(line)
            if marked_name:
                rest = line[_MARKED_LABEL.match(line).end():].strip()
                matched = (marked_name, rest)
        elif name and name in recurring:
            # ONLY A RECURRING NAME IS A SPEAKER -- see `_label_candidates`.
            for shape in _LABEL_SHAPES:
                m = shape.match(line)
                if not m:
                    continue
                rest = line[m.end():].strip()
                matched = (name, rest)
                break
        if matched:
            if pending:
                out.append("%s: %s" % pending)
            pending = matched if matched[1] else (matched[0], "")
        elif pending:
            pending = (pending[0], (pending[1] + " " + line).strip())
        else:
            out.append(line)                  # heading / stage direction
    if pending:
        out.append("%s: %s" % pending)
    # The sentinel is scaffolding, never text. Any that survives here sat mid
    # line -- a name inside narration -- and must not reach a performer.
    body_out = "\n".join(l for l in out if l.strip())
    return body_out.replace(SPEAKER_MARK, "")


def write_rows(rows):
    manifest_path = os.path.join(OUT_ROOT, CORPUS.MANIFEST_NAME)
    existing = []
    if os.path.isfile(manifest_path):
        existing = json.load(io.open(manifest_path, encoding="utf-8")).get(
            "scenes", [])
    by_key = {(r.get("iso"), r.get("play"), r.get("scene")): r for r in existing}

    for lead, key, body, labels, speakers in rows:
        iso, play, scene = key
        text = normalise_labels(body)
        after, after_speakers = CORPUS.speaker_label_stats(text)
        rel = os.path.join(iso, "%s_%s.txt" % (play, scene.replace(".", "_")))
        dest = os.path.join(OUT_ROOT, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        # SIGN THE BYTES THAT ARE STORED, not the ones in hand. The first cut
        # wrote `text + "\n"` and hashed `text`, so the manifest signed a
        # different file than it named and `vendored_text` refused every scene
        # at the integrity check -- correctly. An off-by-one-newline is exactly
        # what that check is for; it just happened to be catching me.
        stored = text + "\n"
        io.open(dest, "w", encoding="utf-8", newline="\n").write(stored)
        digest = hashlib.sha256(stored.encode("utf-8")).hexdigest()
        # THE ROW IS REBUILT FROM SCRATCH, so a hand-curated key would be
        # deleted by a routine re-vendor -- the same way a re-fetch used to
        # wipe the gender roster off a sidecar (`STAMPER_OWNED_SIDECAR_KEYS`).
        # The speaker map is curated by hand against the edition's labels and
        # the English sidecar; carry it forward, and let the reader re-check
        # it against the new text via `unbound_labels` on the next plan.
        carried = {
            k: v for k, v in (by_key.get(key) or {}).items()
            if k == CORPUS.SPEAKER_MAP_FIELD
        }
        by_key[key] = {
            "iso": iso, "play": play, "scene": scene,
            "file": rel.replace("\\", "/"),
            "translator": lead.get("translator", ""),
            "translator_death_date": lead.get("translator_death_date", ""),
            "translation_first_published":
                lead.get("translation_first_published", ""),
            "transcription_license": lead.get("transcription_license", ""),
            "source_url": lead.get("url", ""),
            "revision_id": "sha256:" + digest[:16],
            "raw_sha256": digest,
            "verdict": CORPUS.READY,
            # Located by the edition's OWN label, verified by reading the
            # opening lines -- not by counting headings. 1.0 is the honest
            # figure for a label match confirmed against the scene's content.
            "alignment_confidence": 1.0,
            "edition_label": " / ".join(
                x for x in EDITION_LABELS[key] if x),
            "speaker_labels": after,
            "distinct_speakers": after_speakers,
            **carried,
        }
        print("  wrote %-46s %5d chars  %3d speeches / %2d speakers"
              % (rel, len(text), after, after_speakers))

    payload = {"schema_version": CORPUS.SCHEMA_VERSION,
               "note": ("Vendored scenes. Written by "
                        "scripts/otr_vendor_shakespeare.py; every row was "
                        "located by the EDITION'S OWN scene label and its "
                        "opening lines read before acceptance."),
               "scenes": sorted(by_key.values(),
                                key=lambda r: (r["iso"], r["play"], r["scene"]))}
    io.open(manifest_path, "w", encoding="utf-8", newline="\n").write(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print("\n  manifest: %s (%d scenes)" % (manifest_path, len(payload["scenes"])))
    CORPUS.load_manifest(manifest_path)
    print("  load_manifest validated it")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
