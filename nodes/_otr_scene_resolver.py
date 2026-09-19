"""Cut one act/scene out of a whole-work Shakespeare translation.

The corpus gate (`scripts/otr_shakespeare_corpus_gate.py`) measured seven
translation leads that are the RIGHT PLAY at the WRONG GRANULARITY: a
Gutenberg ebook or a Wikisource play page carrying all five acts, not the one
scene a fidelity render needs. `_otr_verbatim_corpus.act_headings_found`
already says so -- "a whole work, not the target scene ... resolve this lead
to its scene" -- and this module is that resolution step.

It locates by HEADING, never by counting: find the target ACT heading, then
the target SCENE heading under it, then stop at the next SCENE heading (or
the next ACT heading, or the end of the text). It reuses the corpus module's
own heading grammar -- `_labelled`, `_ACT_WORDS`, `_SCENE_WORDS`,
`_numberings`, `_ORDINAL_WORDS`, `_ROMAN`, `speaker_label_stats` and its
`MIN_SPEAKER_LABELS` / `MIN_DISTINCT_SPEAKERS` / `MIN_LABELS_PER_SPEAKER`
floors -- rather than forking a second parser and a second set of speech
thresholds for the same eight languages. Those names are underscored as
module-private; they are imported anyway because a duplicate regex that
silently drifts from the original is a worse outcome than a documented
private import.

It refuses to guess. When a heading cannot be found, the result is empty
text at confidence 0.0 with a reason -- never "the first N speeches" or "the
middle of the book". A vendored scene that pins the wrong span would publish
one translator's words under a heading they never sat beside, which is the
exact failure `_otr_verbatim_corpus.CorpusError` exists to refuse downstream;
this module is the other half, refusing it at the source rather than at the
manifest.

Pure: no network, no model weights, no ComfyUI import, no file IO. UTF-8, no
BOM.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from nodes import _otr_verbatim_corpus as CORPUS

#: Below this many characters an "extracted scene" is a heading immediately
#: followed by another heading -- structurally located, but there is nothing
#: performable inside it.
MIN_SCENE_CHARS = 40

#: Above this share of the whole (boilerplate-stripped) work, one "scene" is
#: implausible -- a five-act play with a handful of scenes per act does not
#: have one scene that is half the book. Measured against the whole body
#: rather than the act's own span: an act that genuinely holds only one
#: scene should not be punished for occupying all of it.
MAX_SCENE_SHARE_OF_BODY = 0.5

#: Confidence penalties. Additive and floored at 0.0 -- several weak signals
#: stack rather than the worst one winning, because "found the act, found
#: the scene, but three other things look wrong" is genuinely less
#: trustworthy than any one of those alone.
_MULTI_ACT_PENALTY = 0.2
_MULTI_SCENE_PENALTY = 0.2
_SHORT_SPAN_PENALTY = 0.4
_LONG_SPAN_PENALTY = 0.3
_FEW_LABELS_PENALTY = 0.3
_FEW_SPEAKERS_PENALTY = 0.2

_GUTENBERG_START = re.compile(
    r"\*{3}\s*START OF (?:THIS |THE )?PROJECT GUTENBERG(?:'S)? EBOOK[^\n]*\*{3}",
    re.IGNORECASE,
)
_GUTENBERG_END = re.compile(
    r"\*{3}\s*END OF (?:THIS |THE )?PROJECT GUTENBERG(?:'S)? EBOOK[^\n]*\*{3}",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SceneResolution:
    """What `resolve_scene` found, or did not.

    `start`/`end` are character offsets into the TEXT THE CALLER PASSED IN --
    not the Gutenberg-stripped body used internally -- so
    ``text[r.start:r.end] == r.text`` holds whenever a scene was located.
    When it was not, `text` is empty, `start == end == 0`, and `confidence`
    is exactly 0.0: never a guess dressed as an answer.

    `act_candidates` / `scene_candidates` count every heading-shaped match
    for the requested number, not just the one used -- a caller that wants
    to know WHY confidence fell can read these instead of re-deriving them.
    """

    text: str
    start: int
    end: int
    confidence: float
    reasons: tuple = field(default_factory=tuple)
    act_candidates: int = 0
    scene_candidates: int = 0
    speaker_labels: int = 0
    distinct_speakers: int = 0


def _failure(reason: str, **extra) -> SceneResolution:
    return SceneResolution(text="", start=0, end=0, confidence=0.0,
                            reasons=(reason,), **extra)


def strip_gutenberg_envelope(text: str) -> "tuple[str, int]":
    """Cut the licence header/footer out of a Gutenberg ebook.

    Returns ``(body, offset)`` where `offset` is where `body` starts inside
    the ORIGINAL `text` -- every offset computed from `body` downstream adds
    this back, so a caller can always slice the text it actually handed in.
    A non-Gutenberg source (a Wikisource page has no envelope) is returned
    unchanged with `offset == 0`.
    """
    start = 0
    match = _GUTENBERG_START.search(text)
    if match:
        start = match.end()
    end = len(text)
    match = _GUTENBERG_END.search(text, start)
    if match:
        end = match.start()
    return text[start:end], start


def _index_preserving_fold(text: str) -> str:
    """Case/width-fold `text` one character at a time so the result is
    always the SAME LENGTH as the input -- offsets found in the folded copy
    then index the identical position in `text`.

    `_otr_verbatim_corpus._fold` does a whole-string NFKC pass instead,
    which is fine there because `headings_present` only answers yes/no. Here
    the fold result is used to locate a SPAN to slice out of the original,
    and a whole-string NFKC can change length (a ligature or a composed
    roman-numeral codepoint expands to more than one character), which would
    silently desynchronise every offset after it. Folding character-by-
    character and keeping any character whose fold is not exactly one
    character long, unfolded, costs those rare expansions their case-folding
    but keeps the position guarantee absolute.
    """
    out = []
    for ch in text:
        folded = unicodedata.normalize("NFKC", ch).lower()
        out.append(folded if len(folded) == 1 else ch.lower())
    return "".join(out)


def _all_labelled(folded: str, words: tuple, value: str) -> list:
    """Every occurrence of a WORD+number heading for this exact number, as
    ``(start, end)`` offsets into `folded`.

    Built from repeated `_otr_verbatim_corpus._labelled` calls -- which only
    ever returns the first match -- rather than a second regex assembled
    from `_numberings`: reuses the corpus module's own heading grammar
    verbatim instead of forking a copy that can drift from it. Re-running
    the digit-adjacency guard from the start of each remainder is a known,
    accepted imprecision: it can only matter at the exact character where a
    previous match ended, and no shipped heading shape lands there.
    """
    hits = []
    offset = 0
    while True:
        match = CORPUS._labelled(folded[offset:], words, value)
        if match is None:
            break
        hits.append((offset + match.start(), offset + match.end()))
        offset += match.end()
    return hits


def _next_heading(folded: str, words: tuple, start: int) -> "int | None":
    """The position of the next WORD+number heading, for ANY of the five
    numbers the corpus module knows ordinal spellings for, at or after
    `start`. This is the boundary a span stops at: a heading for a
    DIFFERENT number still ends the current act or scene.
    """
    best = None
    remainder = folded[start:]
    for number in CORPUS._ROMAN:  # "1" .. "5", in the corpus module's order
        match = CORPUS._labelled(remainder, words, number)
        if match is not None:
            pos = start + match.start()
            if best is None or pos < best:
                best = pos
    return best


def resolve_scene(text: str, *, scene: str) -> SceneResolution:
    """Extract one scene's text from a whole-work translation.

    `scene` is ``"act.scene"`` (``"1.3"`` for Act I Scene 3), matching the
    convention `_otr_verbatim_corpus.headings_present` already uses. The
    method is always: find the target ACT heading, find the target SCENE
    heading under it, extract from the scene heading to whichever comes
    first -- the next scene heading under the same act, the next act
    heading, or the end of the text. The heading line itself is INCLUDED in
    the extracted text; a caller that wants the bare dialogue can strip the
    first line, but a resolver that silently discarded it could not be
    checked against what it actually matched.

    Never falls back to a guess. If the act or the scene heading cannot be
    found, ``text`` is empty and ``confidence`` is 0.0 -- callers refuse
    below their own threshold, and a confident wrong span is worse than an
    honest failure.
    """
    act, _, sc = str(scene or "").partition(".")
    act, sc = act.strip(), sc.strip()
    if not act or not sc:
        return _failure("scene must be 'act.scene' (e.g. '1.3'), got %r"
                        % (scene,))

    body, envelope_offset = strip_gutenberg_envelope(text)
    folded = _index_preserving_fold(body)

    act_hits = _all_labelled(folded, CORPUS._ACT_WORDS, act)
    if not act_hits:
        return _failure("act %s heading not found" % act)

    act_heading_start, act_heading_end = act_hits[0]
    next_act = _next_heading(folded, CORPUS._ACT_WORDS, act_heading_end)
    act_region_end = next_act if next_act is not None else len(folded)
    act_region = folded[act_heading_end:act_region_end]

    scene_hits = _all_labelled(act_region, CORPUS._SCENE_WORDS, sc)
    if not scene_hits:
        return _failure(
            "scene %s heading not found under act %s heading" % (sc, act),
            act_candidates=len(act_hits))

    scene_start_rel, scene_end_rel = scene_hits[0]
    scene_heading_start = act_heading_end + scene_start_rel
    scene_heading_end = act_heading_end + scene_end_rel

    next_scene_rel = _next_heading(act_region, CORPUS._SCENE_WORDS, scene_end_rel)
    next_scene_abs = (act_heading_end + next_scene_rel
                      if next_scene_rel is not None else None)
    boundaries = [b for b in (next_scene_abs, act_region_end) if b is not None]
    scene_region_end = min(boundaries) if boundaries else len(folded)

    if scene_region_end <= scene_heading_start:
        return _failure(
            "scene %s heading is immediately followed by another heading "
            "under act %s -- nothing to extract" % (sc, act),
            act_candidates=len(act_hits), scene_candidates=len(scene_hits))

    extracted = body[scene_heading_start:scene_region_end]
    labels, distinct = CORPUS.speaker_label_stats(extracted)

    reasons = []
    confidence = 1.0

    if len(act_hits) > 1:
        confidence -= _MULTI_ACT_PENALTY
        reasons.append(
            "%d candidate act %s headings matched; used the first"
            % (len(act_hits), act))
    if len(scene_hits) > 1:
        confidence -= _MULTI_SCENE_PENALTY
        reasons.append(
            "%d candidate scene %s headings matched under act %s; used the "
            "first" % (len(scene_hits), sc, act))

    span_len = scene_region_end - scene_heading_start
    if span_len < MIN_SCENE_CHARS:
        confidence -= _SHORT_SPAN_PENALTY
        reasons.append(
            "extracted span is %d characters, implausibly short for a "
            "scene" % span_len)
    elif len(body) and span_len / len(body) > MAX_SCENE_SHARE_OF_BODY:
        confidence -= _LONG_SPAN_PENALTY
        reasons.append(
            "extracted span is %d of %d characters in the work -- "
            "implausibly long for one scene" % (span_len, len(body)))

    # Mirrors `_otr_verbatim_corpus.assess`'s own cascade, and the same
    # floors: a resolved span that could not pass the gate's own performable-
    # scene check is not evidence the resolver should trust either.
    if labels < CORPUS.MIN_SPEAKER_LABELS:
        confidence -= _FEW_LABELS_PENALTY
        reasons.append(
            "%d speaker labels in the extracted span; a performable scene "
            "needs %d" % (labels, CORPUS.MIN_SPEAKER_LABELS))
    elif distinct < CORPUS.MIN_DISTINCT_SPEAKERS:
        confidence -= _FEW_SPEAKERS_PENALTY
        reasons.append(
            "%d distinct speaker(s) in the extracted span; a scene "
            "alternates" % distinct)
    elif labels < distinct * CORPUS.MIN_LABELS_PER_SPEAKER:
        confidence -= _FEW_SPEAKERS_PENALTY
        reasons.append(
            "%d labels across %d speakers in the extracted span; reads as "
            "a list rather than a scene" % (labels, distinct))

    confidence = max(0.0, min(1.0, confidence))
    return SceneResolution(
        text=extracted,
        start=envelope_offset + scene_heading_start,
        end=envelope_offset + scene_region_end,
        confidence=confidence,
        reasons=tuple(reasons),
        act_candidates=len(act_hits),
        scene_candidates=len(scene_hits),
        speaker_labels=labels,
        distinct_speakers=distinct,
    )


__all__ = [
    "MAX_SCENE_SHARE_OF_BODY", "MIN_SCENE_CHARS", "SceneResolution",
    "resolve_scene", "strip_gutenberg_envelope",
]
