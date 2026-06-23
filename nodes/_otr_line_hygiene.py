"""nodes/_otr_line_hygiene.py -- story-quality Phase 1, A6 (clean delivery).

Two deterministic SCRUBS + one TRUNCATION DETECTOR for spoken lines. The
spine applies these: character defects are scrubbed in place (parenthetical
stage-directions, self-vocative) and TRUNCATION is REPAIRED BY RECOMPOSE
(never token-surgery) -- a character truncation routes to the spine recompose
seam, an announcer open/close truncation routes to the dedicated announcer
composer (the critic excludes announcer lines as locked structural content,
so a character reroll path cannot act on them).

  * scrub_parentheticals -- drop "(she translates a passage)" style stage
    directions, INCLUDING a dangling unclosed "(..." to end of line
    (leg_0013 "(she translates a passage,").
  * scrub_self_vocative -- drop the speaker's OWN cast name when it stands as
    a leading or trailing vocative ("Edna, ..." / "..., Edna."); never an
    in-line occurrence of the name.
  * is_truncated -- detect a mid-sentence cut (trailing comma / conjunction /
    dangling open paren) or a lone-stopword sentence ("The." in leg_0013
    "The. Stay with us.") so the caller can RECOMPOSE it. The music-beat dump
    text is left alone (callers only pass voiced character/announcer lines).

PURE module: stdlib only. Deterministic + idempotent (C7-safe). Never raises
(every function returns the input unchanged on any error / empty result).
UTF-8 no BOM. 4-space indentation.
"""

from __future__ import annotations

import re
from typing import Any

_WS = re.compile(r"\s+")
_PAREN_CLOSED = re.compile(r"\([^)]*\)")
_PAREN_DANGLING = re.compile(r"\([^)]*$")

# Connective / function words that should never END a finished line -- a line
# ending on one of these (with no terminal punctuation) was cut mid-thought.
_TRAIL_WORDS = frozenset({
    "and", "but", "or", "so", "the", "a", "an", "to", "of", "with", "for",
    "as", "that", "which", "who", "her", "his", "their", "its", "my", "your",
    "in", "on", "at", "from", "into", "than", "then", "if", "when", "while",
    "because", "about",
})

# A sentence that is JUST one of these + a period is a truncation stub.
_LONE_STOP = frozenset({
    "the", "a", "an", "and", "but", "or", "so", "then", "of", "to", "with",
    "for", "as", "that", "which", "when", "while", "because", "he", "she",
    "they", "it", "we", "i", "you", "this", "in", "on", "at",
})

_TERMINAL = ".!?\"')]…"  # sentence-final chars (incl. ellipsis)


def scrub_parentheticals(text: Any) -> str:
    """Strip closed and dangling-open parenthetical stage directions.
    Returns the original text if scrubbing would empty the line."""
    try:
        s = str(text or "")
        out = _PAREN_CLOSED.sub(" ", s)
        out = _PAREN_DANGLING.sub(" ", out)
        out = _WS.sub(" ", out).strip()
        return out or s.strip()
    except Exception:  # noqa: BLE001
        return str(text or "")


def scrub_self_vocative(text: Any, speaker_name: Any) -> str:
    """Drop the speaker's OWN name as a leading/trailing vocative only.
    Returns the original text if scrubbing would empty the line."""
    try:
        s = str(text or "")
        name = str(speaker_name or "").strip()
        if not name:
            return s
        variants = {name}
        first = name.split()[0] if name.split() else ""
        if first:
            variants.add(first)
        out = s
        for variant in variants:
            if not variant:
                continue
            v = re.escape(variant)
            # leading vocative: "Name, rest" / "Name: rest" / "Name - rest"
            out = re.sub(rf"^\s*{v}\s*[,:\-]\s+", "", out, flags=re.IGNORECASE)
            # trailing vocative: "rest, Name." / "rest, Name"
            out = re.sub(rf"\s*,\s*{v}\s*([.!?]?)\s*$", r"\1", out,
                         flags=re.IGNORECASE)
        out = _WS.sub(" ", out).strip()
        return out or s.strip()
    except Exception:  # noqa: BLE001
        return str(text or "")


def clean_spoken_character_line(text: Any, speaker_name: Any) -> str:
    """Parenthetical + self-vocative scrub for a spoken character line."""
    return scrub_self_vocative(scrub_parentheticals(text), speaker_name)


def is_stage_direction_only(text: Any) -> bool:
    """True when a spoken line has NO pronounceable content -- it is ENTIRELY a
    stage direction / cue (e.g. "(pauses, then flips the switch)", "(beat)").

    Root-cause detector (2026-06-22): `scrub_parentheticals` /
    `clean_spoken_character_line` deliberately KEEP such a line (they return the
    original rather than empty it, to avoid an empty character line), so the
    parenthetical would otherwise leak all the way to the voice worker -- where
    the TTS-side clean (`_otr_script_prep.clean_spoken_text`) empties it and the
    engine crashes / emits silence. This flags exactly those lines so the spine
    can RECOMPOSE them into real dialogue (a stage direction is not a line).

    Uses the SAME clean the voice path uses, so the writer-side detector and the
    TTS-side reality agree. Deterministic; never raises (a bad input -> False).
    """
    raw = str(text or "").strip()
    if not raw:
        return False  # an already-empty line is a separate (mechanical) defect
    try:
        try:
            from ._otr_script_prep import clean_spoken_text
        except ImportError:  # pragma: no cover - standalone / test load
            from _otr_script_prep import clean_spoken_text  # type: ignore
        return not str(clean_spoken_text(raw) or "").strip()
    except Exception:  # noqa: BLE001 -- detector must never raise on a line
        return False


# F7 (story-engine v1): narration / self-address detector. Fires ONLY when a
# spoken line narrates the SPEAKER's own physical action in third person, or
# opens with the speaker's own name as a 3rd-person subject + a narration verb.
# First-person lines and legitimate 3rd-person references to OTHERS ("They know
# the code", "He is lying") are NOT flagged -- that is craft, not breakage. The
# scan (scripts/story_quality_scan.py) imports THIS function so the engine and
# the measurement agree. Deterministic; never raises.
_NARRATION_VERBS = frozenset({
    "paces", "pacing", "stops", "stopping", "gazes", "gazing", "stares",
    "staring", "contemplates", "contemplating", "sighs", "sighing",
    "nods", "nodding", "shrugs", "shrugging", "turns", "turning", "walks",
    "walking", "leans", "leaning", "frowns", "frowning", "smiles",
    "smiling", "glances", "glancing", "reaches", "reaching", "stands",
    "standing", "sits", "sitting", "watches", "watching", "moves",
    "moving", "steps", "stepping", "looks", "looking",
    # D1 (2026-06-22, story-quality lift): EXPLICIT closed list grounded on the
    # "Chandra's Echo" leak corpus (b005/b010/b012/b015/b017). No "obvious
    # neighbours" -- only the verbs the real frozen ledger leaked.
    "adjusts", "clutches", "taps", "tightens", "overrides", "dances",
    "dancing",
})
_THIRD_PERSON_LEAD = re.compile(r"^\s*(he|she|they)\b", re.IGNORECASE)


def detect_narration_self_address(text: Any, speaker_name: Any = "") -> bool:
    """True when a spoken line narrates the speaker's OWN action in third
    person (or opens with the speaker's own name + a narration verb)."""
    try:
        s = " ".join(str(text or "").split())
        if not s:
            return False
        low = s.lower()
        words = re.findall(r"[a-z']+", low)
        if not words:
            return False
        # 3rd-person pronoun lead + a narration verb in the opening = self-narration
        if _THIRD_PERSON_LEAD.match(s):
            if any(w in _NARRATION_VERBS for w in words[:4]):
                return True
        # the speaker's own name as a 3rd-person subject + a narration verb
        name = str(speaker_name or "").strip().lower()
        first = name.split()[0] if name.split() else ""
        if first and len(first) > 1 and words and words[0] == first:
            if any(w in _NARRATION_VERBS for w in words[1:4]):
                return True
        return False
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# Bare (undelimited) leading stage-direction scrub + detector
# (2026-06-22, roundtable-converged: docs/2026-06-22-stage-direction-leak/).
#
# The writer LLM (esp. max-chaos) emits a BARE action clause as the leading part
# of a character line -- "twirls his pen nervously Look, Pinky..." -- with NO
# (), [], or ** delimiters, so every existing delimited scrub misses it and it
# reaches the frozen ledger text (Bark speaks it, captions show it).
#
# Design (panel-converged): a NARROW, fully-guarded DESTRUCTIVE strip used as a
# FREEZE-only floor + a slightly BROADER detector that drives a reroll. A naive
# verb-list strip is unsafe (false positives like "looks can be deceiving,
# John."), so the strip fires ONLY when a conjunction of guards all pass.
# ---------------------------------------------------------------------------

#: First word of a line whose SECOND token is one of these is a SUBJECT (real
#: dialogue: "looks can...", "pauses are..."), never a stage-action verb.
_COPULA_MODAL = frozenset({
    "is", "are", "was", "were", "be", "been", "being", "am", "can", "could",
    "will", "would", "shall", "should", "may", "might", "must", "has", "have",
    "had", "do", "does", "did",
})
#: A lowercase lead containing any of these is dialogue, not stage business
#: (kills "look, Pinky,...", "maybe we should ask John...").
_DIALOGUE_STARTER = frozenset({
    "yes", "no", "well", "oh", "maybe", "please", "now", "listen", "look",
    "hey", "okay", "fine", "sure",
})
#: 1st/2nd-person pronoun ROOTS (matched before an apostrophe: we've/you'll/i'm).
#: Their presence in the lead means it is dialogue, not an impersonal action.
_PRONOUN_ROOTS = frozenset({"i", "we", "you", "me", "us", "my", "your", "our"})
#: A capitalized token whose previous token is one of these is the OBJECT of the
#: action (skip it as the dialogue boundary): "glances at Pinky We..." -> "We...".
_OBJ_PREP = frozenset({
    "at", "to", "toward", "towards", "with", "of", "for", "by", "from", "over",
    "under", "through", "into", "onto", "upon", "on", "in", "inside", "behind",
    "past", "out", "about", "around", "against",
})
_ARTICLE = frozenset({"the", "a", "an"})
_POSS_ADJ = frozenset({"his", "her", "their", "my", "your", "our", "its"})
#: A capitalized token preceded by a conjunction is part of a COMPOUND object
#: chain ("looks at Pinky and Brain We...") -> too risky to parse -> ABORT (keep).
_CONJ = frozenset({"and", "or"})
#: A 1-word remainder is kept UNLESS it is one of these WITH terminal punctuation
#: ("sighs No." -> "No."; "sighs No" -> kept).
_SHORT_UTT = frozenset({
    "yes", "no", "wait", "stop", "never", "fine", "right", "okay", "ok", "go",
})

MAX_STAGE_PREFIX_WORDS = 6        # destructive floor: lead must be <= this
_DETECT_MAX_PREFIX_WORDS = 10     # detector is slightly broader (reroll is cheap)

#: Set by the Chunk-2 precision gate. True = the freeze floor strips; False =
#: detect-only (the floor reports but does not mutate). Default True; flip to
#: False if the corpus scan shows any false-positive mutation.
BARE_STAGE_FLOOR_ACTIVE = True

#: reroll directive handed to the line composer on a detector hit.
_BARE_STAGE_HINT = (
    "write only the spoken words; do not prefix the line with an action "
    "description (no stage directions)"
)

_LEAD_QUOTES = "\"'“”‘’"
_TERMINAL_PUNCT = (".", "!", "?")


def _norm_token(tok: str) -> str:
    """Lowercase a token stripped of surrounding punctuation/quotes."""
    return tok.strip(".,;:!?" + _LEAD_QUOTES).lower()


def _leading_stage_strip(text: Any, max_words: int) -> str:
    """Core: strip a BARE leading stage-direction clause iff ALL guards pass,
    else return the input UNCHANGED. Pure, idempotent, never raises.

    Guards (all must hold): the line starts lowercase (after an optional leading
    quote); the second token is not a copula/modal; the lead carries no
    1st/2nd-person pronoun or dialogue-starter; there is no terminal punctuation
    in the lead; a dialogue boundary (first capitalized non-object token) exists;
    the lead is <= ``max_words``; the remainder is >= 2 words OR a
    terminal-punctuated short utterance. A capitalized object after a
    preposition/article/possessive is skipped; one after a conjunction ABORTS.
    """
    try:
        s = "" if text is None else str(text)
        if not s.strip(" " + _LEAD_QUOTES):
            return s
        body = s.lstrip()
        # optional single leading quote, then re-strip
        if body[:1] in _LEAD_QUOTES:
            body = body[1:].lstrip()
        if not body or not body[0].islower():
            return s
        words = body.split()
        if len(words) < 2:
            return s
        # second token a copula/modal -> first word is a subject -> dialogue
        if _norm_token(words[1]) in _COPULA_MODAL:
            return s
        # find the dialogue boundary (first capitalized, non-object token)
        boundary = None
        for i in range(1, len(words)):
            first_alpha = next((c for c in words[i] if c.isalpha()), "")
            if not first_alpha or not first_alpha.isupper():
                continue
            prev = _norm_token(words[i - 1])
            prev_raw = words[i - 1]
            if prev in _CONJ:
                return s  # compound-object chain -> too risky, keep
            if (prev in _OBJ_PREP or prev in _ARTICLE or prev in _POSS_ADJ
                    or prev_raw.endswith("'s") or prev_raw.endswith("’s")):
                continue  # capitalized object of the action -> skip
            boundary = i
            break
        if not boundary:
            return s
        lead = words[:boundary]
        if len(lead) > max_words:
            return s
        # no terminal punctuation inside the lead (kills "looks like rain. We...")
        if any(p in " ".join(lead) for p in _TERMINAL_PUNCT):
            return s
        # no pronoun root / dialogue-starter in the lead
        for w in lead:
            nw = _norm_token(w)
            root = re.split(r"['’]", nw)[0]
            if nw in _DIALOGUE_STARTER or nw in _PRONOUN_ROOTS or root in _PRONOUN_ROOTS:
                return s
        remainder = " ".join(words[boundary:]).strip()
        if len(remainder.split()) < 2:
            # allow a single-word terminal-punctuated short utterance
            if remainder and remainder[-1] in _TERMINAL_PUNCT \
                    and _norm_token(remainder) in _SHORT_UTT:
                return remainder
            return s
        return remainder
    except Exception:  # noqa: BLE001 -- never raise on a spoken line
        return "" if text is None else str(text)


def scrub_leading_stage_direction(text: Any) -> str:
    """Destructive FREEZE-floor strip of a bare leading stage direction.
    Returns the input unchanged unless the narrow guard conjunction fires."""
    return _leading_stage_strip(text, MAX_STAGE_PREFIX_WORDS)


def detect_leading_stage_business(text: Any) -> "tuple[bool, str]":
    """Detector for the reroll gate (slightly broader than the destructive
    floor -- a false positive only costs one recompose). Returns
    ``(hit, reroll_hint)``; the hint is empty when no hit."""
    s = "" if text is None else str(text)
    hit = _leading_stage_strip(s, _DETECT_MAX_PREFIX_WORDS) != s
    return (hit, _BARE_STAGE_HINT if hit else "")


# ---------------------------------------------------------------------------
# D1 (2026-06-22, story-quality lift -- docs/2026-06-22-story-quality-lift/) --
# bare stage-direction leak AFTER / BETWEEN / WITHOUT quotes.
#
# The leading-only scrub above misses a stage direction that TRAILS a closing
# quote (b005 `"...this." adjusts dials on the console`), sits BETWEEN quotes on
# a malformed line (b015), or runs UNDELIMITED mid-line (b017). indextts2 then
# SPEAKS the direction. These helpers add: a shared double-quote segmenter (one
# source of truth so the composer reroll [Tier 2] and the freeze floor [Tier 3]
# parse identically, no drift), a third-person-action-clause classifier, a
# broad Tier-2 detector, and a narrow quote-anchored Tier-3 floor strip.
# ---------------------------------------------------------------------------

#: Curly -> straight double-quote normalization, applied BEFORE counting/splitting
#: so the freeze floor (which runs upstream of the scrub's smart-quote pass) and
#: the composer reroll segment identical input. Single quotes / apostrophes
#: ('The Chronicle', contractions, 'alive') are deliberately IGNORED.
_CURLY_DQUOTE = {"“": '"', "”": '"'}
_CURLY_DQUOTE_RE = re.compile("[“”]")

#: Word-count ceiling for a third-person action clause (a real stage direction is
#: short; a long 3rd-person sentence about OTHERS is dialogue, not a direction).
_ACTION_CLAUSE_MAX_WORDS = 12

#: Clause boundaries for the undelimited (no-quote) Tier-2 scan.
_CLAUSE_SPLIT_RE = re.compile(r"[.!?,;:]+")


def segment_double_quotes(text: Any) -> "tuple[str, list[tuple[str, bool]]]":
    """Shared SINGLE-SOURCE double-quote segmenter (Tier 2 + Tier 3 use it
    identically, no parse drift). Returns ``(normalized_text, segments)`` where
    ``normalized_text`` has curly double quotes folded to straight ``"`` and
    ``segments`` is an ordered list of ``(span_text, in_quote)`` tuples. With an
    EVEN number of ``"`` the spans alternate outside/in-quote starting OUTSIDE;
    an ODD count is unbalanced -- the caller checks
    ``normalized_text.count('"') % 2`` and treats odd as malformed. Single
    quotes/apostrophes are never counted. Pure; never raises."""
    try:
        s = "" if text is None else str(text)
        norm = _CURLY_DQUOTE_RE.sub(lambda m: _CURLY_DQUOTE[m.group(0)], s)
        parts = norm.split('"')
        segments = [(p, (idx % 2 == 1)) for idx, p in enumerate(parts)]
        return norm, segments
    except Exception:  # noqa: BLE001 -- never raise on a spoken line
        s = "" if text is None else str(text)
        return s, [(s, False)]


def is_third_person_action_clause(
    span: Any, max_words: int = _ACTION_CLAUSE_MAX_WORDS,
) -> bool:
    """True iff ``span`` reads as a THIRD-person stage-action clause (a leaked
    direction), not dialogue. Requires: (a) NO first/second-person pronoun --
    third-person (he/she/his/her/they) is PERMITTED, it is the subject of a
    direction; (b) the LEAD token is an extended ``_NARRATION_VERBS`` verb
    (directions are verb-led: "adjusts dials", "clutches her ring"); (c) the
    lead is not a ``_DIALOGUE_STARTER``; (d) word-count <= ``max_words``.
    Locked: "clutches her wedding ring tightly" -> True; "taps his cane
    impatiently" -> True; "I adjust the dial as I speak" -> False. Pure; never
    raises."""
    try:
        s = " ".join(str(span or "").split())
        if not s:
            return False
        words = re.findall(r"[a-z']+", s.lower())
        if not words or len(words) > max_words:
            return False
        # (a) no 1st/2nd-person pronoun anywhere in the clause
        for w in words:
            root = re.split(r"['’]", w)[0]
            if w in _PRONOUN_ROOTS or root in _PRONOUN_ROOTS:
                return False
        lead = words[0]
        # (c) lead not a dialogue starter
        if lead in _DIALOGUE_STARTER:
            return False
        # (b) lead is a narration verb (directions are verb-led)
        if lead not in _NARRATION_VERBS:
            return False
        return True
    except Exception:  # noqa: BLE001
        return False


def _contains_undelimited_action_clause(text: str) -> bool:
    """True when a no-quote / unbalanced line carries a 3rd-person action clause
    as one of its punctuation-delimited chunks (b017
    "...at once! overrides systems, fingers dancing on the console ...")."""
    for chunk in _CLAUSE_SPLIT_RE.split(text):
        if is_third_person_action_clause(chunk.strip()):
            return True
    return False


def detect_stage_business_for_reroll(
    text: Any, speaker_name: str = "",
) -> "tuple[bool, str, str]":
    """Tier-2 (composer reroll) detector -- BROAD on purpose (a false positive
    only costs one recompose). Returns ``(hit, hint, reason_code)`` with
    ``reason_code`` in {"leading", "trailing_after_quote",
    "embedded_between_quotes", "undelimited_action_clause"}; ``("", "")`` parts
    when no hit. Owns the malformed/undelimited cases (b015, b017) the freeze
    floor deliberately leaves alone. Pure; never raises."""
    try:
        s = "" if text is None else str(text)
        if not s.strip():
            return (False, "", "")
        # 1. bare LEADING direction (existing leading-only detector)
        lead_hit, _ = detect_leading_stage_business(s)
        if lead_hit:
            return (True, _BARE_STAGE_HINT, "leading")
        norm, segments = segment_double_quotes(s)
        nq = norm.count('"')
        if nq and nq % 2 == 0:
            # balanced quotes: an OUTSIDE-quote action span is a leaked direction
            for idx, (span, in_quote) in enumerate(segments):
                if in_quote:
                    continue
                if is_third_person_action_clause(span.strip(" ,;:-")):
                    reason = "leading" if idx == 0 else "trailing_after_quote"
                    return (True, _BARE_STAGE_HINT, reason)
            # malformed (b015): dialogue sits OUTSIDE, the direction got quoted
            for span, in_quote in segments:
                if not in_quote:
                    continue
                if is_third_person_action_clause(span.strip(" ,;:-")):
                    return (True, _BARE_STAGE_HINT, "embedded_between_quotes")
            return (False, "", "")
        # unbalanced or no quotes (b017): undelimited action clause anywhere
        if _contains_undelimited_action_clause(norm):
            return (True, _BARE_STAGE_HINT, "undelimited_action_clause")
        return (False, "", "")
    except Exception:  # noqa: BLE001
        return (False, "", "")


def _floor_well_formed(text: str) -> bool:
    """A floor-stripped line is well-formed iff (after trimming trailing
    separators) the last SPOKEN char -- ignoring one optional trailing closing
    structural ``"`` -- is terminal punctuation, the line is non-empty, and its
    double quotes are balanced. Else the floor ABORTS the strip."""
    candidate = (text or "").strip().strip(" ,;-")
    if not candidate:
        return False
    if candidate.count('"') % 2 != 0:
        return False
    core = candidate[:-1].rstrip() if candidate.endswith('"') else candidate
    if not core:
        return False
    return core[-1] in _TERMINAL_PUNCT


def strip_quote_anchored_stage_direction(text: Any) -> "tuple[str, bool, str]":
    """Tier-3 deterministic FREEZE floor for the BALANCED-QUOTE class only
    (b005/b010/b012 trailing-after-quote). Returns ``(text, changed, reason)``.

    Conservative + cannot route back to a reroll (it is downstream of the
    composer): an ODD ``"`` count -> ``(text, False, "")`` (leave unscrubbed; an
    odd-quote line that reaches the floor ships LOUD / CI-fails). With balanced
    quotes it removes an OUTSIDE-quote span that ``is_third_person_action_clause``
    and keeps the strip ONLY if the result is well-formed; otherwise it aborts
    and returns the original. No-quote (undelimited) lines are EXCLUDED -- those
    are Tier-2's job. Idempotent; pure; never raises."""
    try:
        s = "" if text is None else str(text)
        norm, segments = segment_double_quotes(s)
        nq = norm.count('"')
        if nq == 0 or nq % 2 != 0:
            return (s, False, "")
        changed = False
        reason = ""
        rebuilt: list[str] = []
        for idx, (span, in_quote) in enumerate(segments):
            if in_quote:
                rebuilt.append(span)
                continue
            if is_third_person_action_clause(span.strip(" ,;:-")):
                changed = True
                reason = "leading" if idx == 0 else "trailing_after_quote"
                rebuilt.append("")
            else:
                rebuilt.append(span)
        if not changed:
            return (s, False, "")
        out = '"'.join(rebuilt)
        out = _WS.sub(" ", out).strip()
        if not _floor_well_formed(out):
            return (s, False, "")  # ABORT -- never ship a malformed strip
        return (out, True, reason)
    except Exception:  # noqa: BLE001
        return ("" if text is None else str(text), False, "")


# ---------------------------------------------------------------------------
# S2 (story-quality R2) -- announcer close reads as a thesis/moral, not an image
# ---------------------------------------------------------------------------

#: Phrases that mark a close as a stated moral / lesson / news-summary instead
#: of a concrete final image. Case-insensitive, straight + curly apostrophe.
_BANNED_THESIS_RES = tuple(
    re.compile(p, re.IGNORECASE) for p in (
        r"tonight['’]s revelation",
        r"\bthe lesson is\b",
        r"\breminding us\b",
        r"\bproving \w+ right\b",
        r"\b\w+ is now shared\b",
        r"\bthis shows\b",
    )
)


def flag_thesis_close(text: Any) -> "tuple[bool, str]":
    """(flagged, reason): the announcer close states a moral / lesson /
    news-summary rather than showing a concrete final image. Pure; never
    raises."""
    try:
        s = str(text or "")
        for rx in _BANNED_THESIS_RES:
            m = rx.search(s)
            if m:
                return True, f"thesis/moral phrase {m.group(0)!r}"
        return False, ""
    except Exception:  # noqa: BLE001
        return False, ""


# ---------------------------------------------------------------------------
# S3 (story-quality R2) -- cliche + flat stage-business reject gates (FLAGS ONLY)
# A flagged VOICED line sets reroll_hint -> the existing compose_line reroll
# recomposes it. Lists are small + grounded; every phrase is one a strong (opus)
# writer would not produce, so the gate only ever lifts the weak end.
# ---------------------------------------------------------------------------

_CLICHE_RES = tuple(re.compile(p, re.IGNORECASE) for p in (
    r"\byou['’]re playing with fire\b",
    r"\bthis changes everything\b",
    r"\bwe['’]re not leaving anything to chance\b",
    r"\bleaving nothing to chance\b",
    r"\bthere['’]s no turning back\b",
    r"\bagainst all odds\b",
))

#: FLAT action-announce filler ("I'll go check") -- a character narrating an
#: errand instead of saying something with stakes. Distinct from the LEADING
#: stage-direction scrub (scrub_leading_stage_direction).
_STAGE_BUSINESS_RES = tuple(re.compile(p, re.IGNORECASE) for p in (
    r"\bI['’]ll go check\b",
    r"\bI['’]ll double[ -]check\b",
    r"\bI['’]ll lock (?:it |everything |the \w+ )?down\b",
    r"\bI['’]ve got this,? no need\b",
    r"\blet me handle (?:it|this)\b",
))


def flag_cliche(text: Any) -> "tuple[bool, str]":
    """(flagged, reason): the line leans on a worn cliche. Pure; never raises."""
    try:
        s = str(text or "")
        for rx in _CLICHE_RES:
            m = rx.search(s)
            if m:
                return True, f"cliche {m.group(0)!r}"
        return False, ""
    except Exception:  # noqa: BLE001
        return False, ""


#: C5 (story-quality R2) -- on-the-nose emotion: a character NAMING their feeling
#: or the stakes flatly ("I'm scared", "this is dangerous") instead of implying
#: it. A strong writer shows; this gate rerolls the weak end.
_ON_THE_NOSE_RES = tuple(re.compile(p, re.IGNORECASE) for p in (
    r"\bI['’]?m (?:so |really |very )?(?:scared|afraid|terrified|worried|nervous)\b",
    r"\bI am (?:so |really |very )?(?:scared|afraid|terrified|worried|nervous)\b",
    r"\bI feel (?:so |really |very )?(?:scared|afraid|terrified)\b",
    r"\bthis is (?:so |really |very )?(?:dangerous|terrifying|scary|serious)\b",
    r"\bI['’]?m feeling (?:scared|afraid|terrified)\b",
))


def flag_on_the_nose(text: Any) -> "tuple[bool, str]":
    """(flagged, reason): the line NAMES an emotion / the stakes on the nose
    instead of implying them. Pure; never raises."""
    try:
        s = str(text or "")
        for rx in _ON_THE_NOSE_RES:
            m = rx.search(s)
            if m:
                return True, f"on-the-nose emotion {m.group(0)!r}"
        return False, ""
    except Exception:  # noqa: BLE001
        return False, ""


def flag_stage_business(text: Any) -> "tuple[bool, str]":
    """(flagged, reason): the line is flat action-announce filler (an errand,
    not a beat with stakes). Pure; never raises."""
    try:
        s = str(text or "")
        for rx in _STAGE_BUSINESS_RES:
            m = rx.search(s)
            if m:
                return True, f"flat stage-business {m.group(0)!r}"
        return False, ""
    except Exception:  # noqa: BLE001
        return False, ""


# ---------------------------------------------------------------------------
# L1 (story-quality v2, R3 2026-06-22) -- objective-literal floor (FLAG ONLY).
#
# The universal weak-writer failure under pressure is a short, bald line that
# RESTATES the beat objective outright ("Marlowe, confess that you leaked the
# codes!") instead of playing it through action/subtext. This is a NARROW
# deterministic matcher: it fires ONLY when a SHORT line reuses most of the
# objective's distinctive content words. Tuned to favour false NEGATIVES over
# false positives -- a long, elaborated line that happens to contain the
# objective words is NOT a bald restatement and must not be flagged. Gated at
# the composer behind the story-quality-v2 flag; off by default.
# ---------------------------------------------------------------------------

#: Common words excluded from the objective<->line content-word overlap so the
#: ratio is driven by distinctive nouns/verbs, not function words.
_OBJLIT_STOPWORDS: frozenset = frozenset({
    "the", "and", "that", "this", "with", "from", "into", "your", "yours",
    "their", "them", "they", "have", "has", "had", "will", "would", "could",
    "should", "about", "what", "when", "where", "which", "while", "before",
    "after", "than", "then", "there", "here", "been", "being", "were", "was",
    "are", "for", "but", "not", "you", "him", "her", "his", "she", "out",
    "get", "got", "make", "made", "want", "wants", "wanted", "need", "needs",
    "tell", "tells", "told", "say", "says", "said", "ask", "asks", "asked",
    "let", "lets", "him", "all", "any", "can", "cant", "off", "over", "down",
    "back", "just", "more", "some", "such", "very", "onto", "upon", "amid",
})

#: A bald restatement is short. A long line has room to play the objective
#: indirectly, so it is NEVER flagged (false-positive guard).
_OBJLIT_MAX_LINE_WORDS: int = 18
#: Need at least this many distinctive objective words present AND this fraction
#: of the objective's content words, for a NARROW high-confidence match.
_OBJLIT_MIN_OVERLAP: int = 2
_OBJLIT_MIN_RATIO: float = 0.6

_OBJLIT_WORD_RE = re.compile(r"[A-Za-z][A-Za-z']{3,}")


def _objlit_content_words(s: str) -> "set[str]":
    return {
        w for w in (m.lower() for m in _OBJLIT_WORD_RE.findall(s or ""))
        if w not in _OBJLIT_STOPWORDS
    }


def flag_objective_literal(text: Any, beat_objective: Any) -> "tuple[bool, str]":
    """(flagged, hint): the spoken line baldly RESTATES the beat objective.

    NARROW + deterministic + pure (never raises). Fires only when ALL hold:
      * the objective carries >= ``_OBJLIT_MIN_OVERLAP`` distinctive content
        words (too thin to judge otherwise -> no flag);
      * the line is short (<= ``_OBJLIT_MAX_LINE_WORDS`` words -- a long line
        has room to imply the goal and is never flagged);
      * the line reuses >= ``_OBJLIT_MIN_OVERLAP`` of those words AND >=
        ``_OBJLIT_MIN_RATIO`` of the objective's content words.

    The hint is the actionable reroll instruction (imply the goal, don't state
    it). An empty objective / empty line -> (False, "")."""
    try:
        obj_words = _objlit_content_words(str(beat_objective or ""))
        if len(obj_words) < _OBJLIT_MIN_OVERLAP:
            return False, ""
        line = str(text or "")
        if len(re.findall(r"[A-Za-z']+", line)) > _OBJLIT_MAX_LINE_WORDS:
            return False, ""
        line_words = _objlit_content_words(line)
        overlap = obj_words & line_words
        if (len(overlap) >= _OBJLIT_MIN_OVERLAP
                and (len(overlap) / len(obj_words)) >= _OBJLIT_MIN_RATIO):
            return True, (
                "objective-literal: the line states the beat goal outright "
                f"({sorted(overlap)}) -- imply it through what the character "
                "DOES or deflects to, do not name the objective"
            )
        return False, ""
    except Exception:  # noqa: BLE001
        return False, ""


def is_truncated(text: Any) -> bool:
    """True when the line looks cut mid-thought (so the caller recomposes).

    Conservative -- only strong signals fire, to avoid recomposing a clean
    line: a dangling unclosed "(", a trailing clause-break char, a trailing
    connective word with no terminal punctuation, or a lone-stopword sentence.
    """
    try:
        s = _WS.sub(" ", str(text or "")).strip()
        if not s:
            return False
        # dangling unclosed parenthesis
        if "(" in s:
            after = s.split("(", 1)[1]
            if ")" not in after:
                return True
        last = s[-1]
        if last in ",;:-—":
            return True
        words = re.findall(r"[A-Za-z']+", s)
        if words and last not in _TERMINAL:
            if words[-1].lower() in _TRAIL_WORDS:
                return True
        # a sentence that is just one stopword + period ("The.")
        for sent in re.split(r"[.!?]+", s):
            toks = re.findall(r"[A-Za-z']+", sent)
            if len(toks) == 1 and toks[0].lower() in _LONE_STOP:
                return True
        return False
    except Exception:  # noqa: BLE001
        return False
