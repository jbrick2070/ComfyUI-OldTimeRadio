"""The vendored-translation corpus: legal tests, the acceptance gate, the manifest.

Operator ruling 2026-09-18 (*"I want the best pack available"*): a Shakespeare
scene ships a real translator's words when they clear BOTH the US test and
life+70 -- the set that can be published anywhere. Everything else keeps the
model translation that already ships (`_otr_verbatim_translation`), per scene,
per language. Nothing is ingested on the strength of a document: every lead is
opened and verdicted first.

THE DEPRECATED RULE. The first inventory used "translator died before 1944".
That is not a copyright test in any jurisdiction -- it was a conservative bound
that happened to clear everything it listed. The real tests are below, and they
are the reason Mandarin is a live phase rather than two plays.

WHY A PURE MODULE. The fetcher (`scripts/otr_shakespeare_corpus_gate.py`) is a
one-off that touches the network; the writer's plan step WILL read the manifest
on every fidelity render once leads are resolved to scenes (GO_FORWARD CODE 1).
Both need the same verdict rules, so the rules live here and neither owns them.
Nothing in this module opens a socket or a model.
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = "otr_verbatim_corpus_v1"

# --------------------------------------------------------------------------- #
# the legal tests -- spec `legal.tests`
# --------------------------------------------------------------------------- #

#: US: a work first published before this year is public domain there.
US_PUBLICATION_BEFORE = 1931

#: life+70 (EU, UK, most of Latin America): the translator died before this.
LIFE_PLUS_70_DEATH_BEFORE = 1956

#: life+50 (China; Japan pre-2018 works). Recorded because the spec records it;
#: the operator's pick does NOT rely on it -- a row that needs this and fails
#: the two above is not "publishable anywhere" and is refused.
LIFE_PLUS_50_DEATH_BEFORE = 1976


def clears_publication_anywhere(first_published: Any, translator_died: Any) -> bool:
    """True when a translation is public domain under BOTH shipped tests.

    The operator asked for the pack he can publish, not the largest pack: a
    row that clears only one jurisdiction is a row that is unsafe somewhere,
    so both must pass. An unknown year fails -- "not recorded" is not
    "cleared".
    """
    published = publication_year(first_published)
    died = _year(translator_died)
    if published is None or died is None:
        return False
    return published < US_PUBLICATION_BEFORE and died < LIFE_PLUS_70_DEATH_BEFORE


def publication_reasons(first_published: Any, translator_died: Any) -> list[str]:
    """Why a row does not clear, in the words the manifest should record."""
    out: list[str] = []
    published = publication_year(first_published)
    died = _year(translator_died)
    if published is None:
        out.append("translation_first_published is not a year")
    elif published >= US_PUBLICATION_BEFORE:
        out.append("first published %d; the US test needs before %d"
                   % (published, US_PUBLICATION_BEFORE))
    if died is None:
        out.append("translator_death_date is not a year")
    elif died >= LIFE_PLUS_70_DEATH_BEFORE:
        out.append("translator died %d; life+70 needs before %d"
                   % (died, LIFE_PLUS_70_DEATH_BEFORE))
    return out


def _years(value: Any) -> list:
    """Every plausible year in ``value``, in order."""
    if isinstance(value, bool):
        return []
    if isinstance(value, int):
        return [value] if 1000 <= value <= 2999 else []
    return [int(y) for y in re.findall(r"\b(1[0-9]{3}|20[0-9]{2})\b",
                                       str(value or ""))]


def _year(value: Any) -> "int | None":
    """The year in ``value``, or None. A person dies once, so the first
    year wins -- this is the DEATH reader."""
    found = _years(value)
    return found[0] if found else None


def publication_year(value: Any) -> "int | None":
    """The year a first-publication test must use: the LATEST in a span.

    A collected edition spanning "1865-1872" does not say which volume held
    this play, so the conservative reading is the last year -- if THAT clears,
    every volume clears. Taking the first year is the optimistic bound and
    would false-clear a span straddling the cutoff ("1929-1932" reads 1929 and
    passes, while the 1932 volume would not). Caught in review before any
    straddling row existed; the legal test may not be right by luck.
    """
    found = _years(value)
    return max(found) if found else None


# --------------------------------------------------------------------------- #
# the gate -- spec `acceptance_gate`
# --------------------------------------------------------------------------- #

READY = "READY"
PARTIAL = "PARTIAL"
EMPTY = "EMPTY"
BLOCKED = "BLOCKED"

#: A page that says the transcription has not happened. The spec names these
#: because v1 read the phrase and scored the row ready anyway.
TRANSCRIPTION_PENDING_MARKERS = (
    "a transcribir",          # es.wikisource: scan uploaded, no text
    "作業中",                  # Aozora: in progress
    "não revisado",           # pt.wikisource
    "not proofread",
    "text not available",
)

#: Minimum speeches for a scene to be performable as a passage. Below this the
#: page is a stub or the scene did not resolve; the passage selector needs
#: consecutive speeches, not a heading.
MIN_SPEAKER_LABELS = 6

#: Below this share of non-markup characters the page is scaffold, not text.
MIN_DIALOGUE_RATIO = 0.35


@dataclass
class LeadReport:
    """One (language, play, scene) lead, opened and measured."""

    iso: str
    play: str
    scene: str
    url: str = ""
    final_url: str = ""
    http_status: int = 0
    byte_length: int = 0
    encoding: str = ""
    headings_present: bool = False
    act_headings: int = 0
    speaker_labels: int = 0
    distinct_speakers: int = 0
    dialogue_ratio: float = 0.0
    pending_markers: list = field(default_factory=list)
    licence: str = ""
    revision_id: str = ""
    transport_error: str = ""
    is_scan: bool = False
    translator: str = ""
    translator_died: Any = ""
    first_published: Any = ""
    verdict: str = EMPTY
    reasons: list = field(default_factory=list)

    def as_row(self) -> dict:
        return {
            "iso": self.iso, "play": self.play, "scene": self.scene,
            "url": self.url, "final_url": self.final_url,
            "http_status": self.http_status, "byte_length": self.byte_length,
            "encoding": self.encoding, "headings_present": self.headings_present,
            "act_headings": self.act_headings,
            "speaker_labels": self.speaker_labels,
            "distinct_speakers": self.distinct_speakers,
            "dialogue_ratio": round(float(self.dialogue_ratio), 4),
            "pending_markers": list(self.pending_markers),
            "licence": self.licence, "revision_id": self.revision_id,
            "transport_error": self.transport_error,
            "is_scan": self.is_scan,
            "translator": self.translator,
            "translator_death_date": self.translator_died,
            "translation_first_published": self.first_published,
            "verdict": self.verdict, "reasons": list(self.reasons),
        }


def assess(report: LeadReport) -> LeadReport:
    """Set ``verdict`` and ``reasons`` from what the fetch measured.

    BLOCKED beats every other verdict: a rights failure is not a quality
    question and no amount of clean text fixes it.
    """
    reasons: list[str] = []

    rights = publication_reasons(report.first_published, report.translator_died)
    if rights:
        report.reasons = rights
        report.verdict = BLOCKED
        return report
    if not str(report.licence or "").strip():
        report.reasons = ["transcription_license not recorded"]
        report.verdict = BLOCKED
        return report

    if report.is_scan:
        # A page image has no text to measure, and decoding one as UTF-8
        # produces measurements that mean nothing. Say what it is: the work
        # is transcription, and the rights are already settled above.
        report.reasons = ["a page scan with no text layer -- needs "
                          "transcription, not another source"]
        report.verdict = EMPTY
        return report

    if report.pending_markers:
        reasons.append("transcription pending: %s"
                       % ", ".join(sorted(set(report.pending_markers))))
    if report.http_status and report.http_status >= 400:
        reasons.append("HTTP %d" % report.http_status)
    if report.transport_error:
        # NOT the same as an empty page, and the difference decides what to do
        # about it: a throttled host wants a re-run, an empty page wants a
        # different source. Conflating them is how a good lead gets cut.
        reasons.append("fetch failed (%s) -- re-run before judging this lead"
                       % report.transport_error)
    elif not report.byte_length:
        reasons.append("no bytes")
    if not report.headings_present:
        if report.act_headings >= 2:
            # The right WORK at the wrong granularity. Naming that is the
            # difference between "find another source" and "extract a range".
            reasons.append(
                "a whole work, not the target scene (%d act headings on the "
                "page) -- resolve this lead to its scene" % report.act_headings)
        else:
            reasons.append("target scene headings not found")
    if report.speaker_labels < MIN_SPEAKER_LABELS:
        reasons.append("%d speaker labels; a performable scene needs %d"
                       % (report.speaker_labels, MIN_SPEAKER_LABELS))
    elif report.distinct_speakers < MIN_DISTINCT_SPEAKERS:
        reasons.append("%d distinct speaker(s); a scene alternates"
                       % report.distinct_speakers)
    elif report.speaker_labels < report.distinct_speakers * MIN_LABELS_PER_SPEAKER:
        # A dramatis personae, a table of contents and a JSON blob all have
        # labels that never repeat -- and every one of them scored a
        # performable scene before this line existed.
        reasons.append(
            "%d labels across %d speakers; each speaks less than %.0f times, "
            "so this reads as a list rather than a scene"
            % (report.speaker_labels, report.distinct_speakers,
               MIN_LABELS_PER_SPEAKER))
    if report.dialogue_ratio < MIN_DIALOGUE_RATIO:
        reasons.append("dialogue-to-markup ratio %.2f below %.2f"
                       % (report.dialogue_ratio, MIN_DIALOGUE_RATIO))
    if not str(report.revision_id or "").strip():
        reasons.append("no revision id -- the bytes cannot be pinned")

    report.reasons = reasons
    if not reasons:
        report.verdict = READY
    elif report.byte_length and report.speaker_labels:
        report.verdict = PARTIAL
    else:
        report.verdict = EMPTY
    return report


def find_pending_markers(text: str) -> list:
    """Transcription-pending phrases present in ``text``, lowercased."""
    low = str(text or "").lower()
    return [m for m in TRANSCRIPTION_PENDING_MARKERS if m in low]


#: Markup CONSTRUCTS, not markup characters. Counting characters reads a
#: template-only page as 65% text, because `{{Index|page=1}}` is mostly
#: letters -- measured, and the reason this strips regions instead.
_MARKUP_REGIONS = (
    re.compile(r"\{\{[^{}]*\}\}", re.DOTALL),   # {{template|arg}}
    re.compile(r"\[\[[^\[\]]*\]\]", re.DOTALL),  # [[Link|label]]
    re.compile(r"<[^<>]{0,200}>", re.DOTALL),    # <noinclude>, HTML
    re.compile(r"\[\#\S+[^\]]*\]"),              # ［＃...］ Aozora notes
    re.compile(r"［＃[^］]*］"),
    re.compile(r"^[=*#:;|!].*$", re.MULTILINE),  # wiki headings, lists, tables
)


def dialogue_ratio(text: str) -> float:
    """Share of the page that survives markup removal.

    A near-empty Wikisource page is templates, links and table scaffold; a
    real scene is speeches. Regions are stripped repeatedly so a nested
    template collapses too. Deliberately crude: it separates "scaffold"
    from "text", and nothing finer is claimed.
    """
    raw = str(text or "")
    if not raw.strip():
        return 0.0
    body = raw
    for _ in range(4):                      # nested templates collapse inward
        before = body
        for pattern in _MARKUP_REGIONS:
            body = pattern.sub(" ", body)
        if body == before:
            break
    return max(0.0, min(1.0, len(body.strip()) / len(raw.strip())))


#: Speaker-label shapes the vendored sources actually use. The loader
#: normalises to ``NAME:`` at vendoring time so the parser stays one rule --
#: the spec's own advice, and cheaper than a per-language parser.
SPEAKER_LABEL_PATTERNS = (
    # `：` is not followed by a space in CJK typesetting, so the separator is
    # optional -- requiring `\s` counted zero speeches on Zhu Shenghao.
    re.compile(r"^(?P<name>[^\n:：]{1,40})[:：][ \t　]?", re.MULTILINE),
    re.compile(r"^(?P<name>[A-ZÀ-ÞĀ-Ž][^\n.]{0,38})\.\s", re.MULTILINE),  # Nom. speech
    re.compile(r"^(?P<name>[^\n]{1,40})--", re.MULTILINE),            # NAME--speech (hi)
    re.compile(r"^(?P<name>[^\n　]{1,20})　", re.MULTILINE),  # NAME<ideographic space>
)


#: A scene ALTERNATES. Below this many distinct speakers the page is a
#: monologue, a list, or prose that happens to carry a separator.
MIN_DISTINCT_SPEAKERS = 2

#: Each speaker must speak at least this often on average. A dramatis
#: personae, a table of contents and a JSON blob all have labels that never
#: repeat, and every one of them scored a performable scene before this.
MIN_LABELS_PER_SPEAKER = 2.0

#: Never a character name: URL, markup and code punctuation.
_NOT_A_NAME = re.compile(r"[/\\\"'=<>{}()@]|https?|www\.")


def speaker_label_stats(text: str) -> tuple:
    """``(labels, distinct_speakers)`` for the best-matching shape.

    Takes the MAXIMUM across shapes rather than the sum: a page uses one
    convention, and summing would let two weak partial matches look like a
    full scene. Prefixes carrying URL or code punctuation are dropped before
    counting -- an HTML page's `<link href="https:` is not a speech.
    """
    raw = str(text or "")
    best = (0, 0)
    for pattern in SPEAKER_LABEL_PATTERNS:
        names = [m.group("name").strip() for m in pattern.finditer(raw)]
        names = [n for n in names if n and not _NOT_A_NAME.search(n)]
        if len(names) > best[0]:
            best = (len(names), len({_fold(n) for n in names}))
    return best


def count_speaker_labels(text: str) -> int:
    """How many speeches this page seems to carry. See `speaker_label_stats`."""
    return speaker_label_stats(text)[0]


#: The word an edition puts before the number, in the seven shipped rows plus
#: English. Matching the NUMBER alone made "there are 2 witches" a scene
#: heading, and the bare roman `i` matched the English pronoun.
_ACT_WORDS = ("act", "acte", "atto", "ato", "akt", "acto", "幕", "अंक")
_SCENE_WORDS = ("scene", "scène", "scena", "cena", "escena", "場", "场", "दृश्य")

_ROMAN = {"1": "I", "2": "II", "3": "III", "4": "IV", "5": "V"}

#: THE ORDINAL WORD IS THE NORMAL 19th-CENTURY SPELLING, not the exception:
#: "ACTE PREMIER, SCENE III", "ATTO PRIMO", "acto primero", 第一幕. Matching
#: only digits and single-letter romans meant a real scene headed this way
#: could never reach READY -- and, worse, made "headings not found" ambiguous
#: between "wrong page" and "right page we cannot read". Caught in post-QA.
_ORDINAL_WORDS = {
    "1": ("first", "premier", "première", "premiere", "primo", "prima",
          "primero", "primera", "primeiro", "primeira", "一", "पहला", "प्रथम"),
    "2": ("second", "seconde", "deuxième", "deuxieme", "secondo", "seconda",
          "segundo", "segunda", "二", "दूसरा", "द्वितीय"),
    "3": ("third", "troisième", "troisieme", "terzo", "terza", "tercero",
          "tercera", "terceiro", "terceira", "三", "तीसरा", "तृतीय"),
    "4": ("fourth", "quatrième", "quatrieme", "quarto", "quarta", "cuarto",
          "cuarta", "四", "चौथा", "चतुर्थ"),
    "5": ("fifth", "cinquième", "cinquieme", "quinto", "quinta", "五",
          "पाँचवाँ", "पंचम"),
}

#: How far past the act token a real scene heading can sit. "ACTE PREMIER,
#: SCENE TROISIEME." is the longest shipped spelling and fits; a mention a
#: paragraph later does not.
_HEADING_WINDOW = 40


def _numberings(value: str) -> list:
    """Every spelling an edition may use for one small number: the digit, the
    roman numeral, and the ordinal WORD in the shipped languages."""
    out = [value]
    roman = _ROMAN.get(value)
    if roman:
        out.append(roman)
    out.extend(_ORDINAL_WORDS.get(value, ()))
    # Longest first: "premiere" must win before "premier" inside it.
    return [re.escape(v.lower())
            for v in sorted(set(out), key=len, reverse=True)]


def _labelled(raw: str, words: tuple, value: str) -> "re.Match | None":
    """`ACT I` / `atto 1` / `第1幕` -- a WORD beside the number, either order.

    Digit guards rather than `\\b`: in `第1幕` there is no word boundary
    between the kanji and the digit, so `\\b1` never matched and every CJK
    heading read as absent.
    """
    numbers = "|".join(_numberings(value))
    for word in words:
        w = re.escape(word.lower())
        for pattern in (r"%s[\s.,:\-]{0,4}(?<![0-9])(%s)(?![0-9])" % (w, numbers),
                        r"(?<![0-9])(%s)(?![0-9])[\s.,:\-]{0,4}%s" % (numbers, w)):
            match = re.search(pattern, raw)
            if match:
                return match
    return None


def act_headings_found(text: str) -> int:
    """How many DISTINCT act headings the page carries.

    Two or more means a whole work (or a collected edition), not the target
    scene -- a true and actionable answer that "headings not found" hid:
    measured 2026-09-18, the three Gutenberg leads are real play texts with
    `ACTE PREMIER` / `ACTO PRIMERO` / `ACTO PRIMEIRO` headings, and calling
    them heading-less sent a reader looking for the wrong fault.
    """
    raw = _fold(text)
    found = set()
    for number in _ROMAN:
        if _labelled(raw, _ACT_WORDS, number) is not None:
            found.add(number)
    return len(found)


def headings_present(text: str, scene: str) -> bool:
    """Does the page name the act AND the scene we are after, together?

    Both must be LABELLED (a word beside the number) and within one window of
    each other, because a heading reads "ACTE I, SCENE 3" -- two independent
    digit hits anywhere on a long page proved nothing and passed prose.

    Anchor matching (the spec's `alignment.do`) is the real resolver and is a
    separate step; this only answers "is the target plausibly on this page".
    """
    raw = _fold(text)
    act, _, sc = str(scene or "").partition(".")
    if not act or not sc:
        return False
    act_hit = _labelled(raw, _ACT_WORDS, act)
    if act_hit is None:
        return False
    # The scene heading follows its act within ONE HEADING's width. Wider and
    # "ACT 1 ... much later ... SCENE 3" reads as a heading it is not.
    window = raw[act_hit.start():act_hit.end() + _HEADING_WINDOW]
    return _labelled(window, _SCENE_WORDS, sc) is not None


def _fold(text: str) -> str:
    return unicodedata.normalize("NFKC", str(text or "")).lower()


def strip_tracking(url: str) -> str:
    """Drop `utm_*` and friends. The spec reads their presence as proof the
    row was never opened, so they never reach the manifest."""
    raw = str(url or "")
    if "?" not in raw:
        return raw
    head, _, query = raw.partition("?")
    keep = [p for p in query.split("&")
            if p and not p.split("=", 1)[0].lower().startswith(("utm_", "fbclid", "gclid"))]
    return head + ("?" + "&".join(keep) if keep else "")


# --------------------------------------------------------------------------- #
# the manifest -- what the writer's plan step reads
# --------------------------------------------------------------------------- #

MANIFEST_NAME = "manifest.json"

REQUIRED_MANIFEST_FIELDS = (
    "iso", "play", "scene", "file", "translator", "translator_death_date",
    "translation_first_published", "transcription_license", "source_url",
    "revision_id", "raw_sha256", "verdict", "alignment_confidence",
)


class CorpusError(ValueError):
    """A manifest that cannot be trusted. Never degraded around: a wrong
    vendored scene would publish one translator's words under another's name."""



def load_manifest(path: str) -> list:
    """Validated manifest rows, or [] when the file does not exist yet.

    A MISSING manifest is normal -- it means nothing is vendored for that
    language and the model translation runs. A malformed one is not.
    """
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:
        raise CorpusError("%s: unreadable manifest (%s)" % (path, exc)) from exc
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
        raise CorpusError("%s: schema_version must be %r" % (path, SCHEMA_VERSION))
    rows = data.get("scenes")
    if not isinstance(rows, list):
        raise CorpusError("%s: scenes must be a list" % path)
    for i, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise CorpusError("%s: scenes[%d] must be an object" % (path, i))
        missing = [k for k in REQUIRED_MANIFEST_FIELDS if k not in row]
        if missing:
            raise CorpusError("%s: scenes[%d] missing %s"
                              % (path, i, ", ".join(missing)))
        # PRESENT IS NOT FILLED. A row with `file=""` or `raw_sha256=""`
        # loaded cleanly and selected, which would vendor a scene nobody can
        # open and nobody can prove.
        blank = [k for k in REQUIRED_MANIFEST_FIELDS
                 if k != "alignment_confidence" and not str(row[k] or "").strip()]
        if blank:
            raise CorpusError("%s: scenes[%d] has empty %s"
                              % (path, i, ", ".join(blank)))
        try:
            confidence = float(row["alignment_confidence"])
        except (TypeError, ValueError):
            raise CorpusError("%s: scenes[%d] alignment_confidence is not a "
                              "number" % (path, i)) from None
        # NaN compares False against every bound, so `NaN < 0.8` is False and
        # a NaN row walked straight through the confidence floor.
        if not 0.0 <= confidence <= 1.0:
            raise CorpusError(
                "%s: scenes[%d] alignment_confidence %r is not within 0..1"
                % (path, i, row["alignment_confidence"]))
        if not clears_publication_anywhere(row["translation_first_published"],
                                           row["translator_death_date"]):
            raise CorpusError(
                "%s: scenes[%d] (%s %s %s) does not clear both publication "
                "tests: %s" % (path, i, row["iso"], row["play"], row["scene"],
                               "; ".join(publication_reasons(
                                   row["translation_first_published"],
                                   row["translator_death_date"]))))
    return [dict(row) for row in rows]


def select_scene(rows: Sequence[Mapping], *, iso: str, play: str, scene: str,
                 min_confidence: float = 0.8) -> "dict | None":
    """The READY, confidently-aligned row for this scene, or None.

    None is the normal answer and means "author it with the model" -- the
    lane never refuses an episode for want of a vendored scene.
    """
    want = (str(iso or "").strip().lower(), str(play or "").strip().lower(),
            str(scene or "").strip())
    for row in rows or ():
        got = (str(row.get("iso") or "").strip().lower(),
               str(row.get("play") or "").strip().lower(),
               str(row.get("scene") or "").strip())
        if got != want:
            continue
        if str(row.get("verdict") or "") != READY:
            continue
        try:
            confidence = float(row.get("alignment_confidence") or 0.0)
        except (TypeError, ValueError):
            continue
        # `NaN >= x` is False for every x, so the floor is written as a
        # POSITIVE test: a confidence that is not demonstrably high enough
        # does not select, and NaN is never demonstrably anything.
        if not confidence >= float(min_confidence):
            continue
        return dict(row)
    return None



__all__ = [
    "BLOCKED", "EMPTY", "LIFE_PLUS_50_DEATH_BEFORE", "LIFE_PLUS_70_DEATH_BEFORE",
    "MANIFEST_NAME", "MIN_DIALOGUE_RATIO", "MIN_SPEAKER_LABELS", "PARTIAL",
    "READY", "REQUIRED_MANIFEST_FIELDS", "SCHEMA_VERSION",
    "SPEAKER_LABEL_PATTERNS", "TRANSCRIPTION_PENDING_MARKERS",
    "US_PUBLICATION_BEFORE", "CorpusError", "LeadReport", "assess",
    "clears_publication_anywhere", "count_speaker_labels",
    "dialogue_ratio", "find_pending_markers", "headings_present",
    "load_manifest", "publication_reasons", "select_scene", "strip_tracking",
]
