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
