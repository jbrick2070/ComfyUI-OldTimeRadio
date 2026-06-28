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
from dataclasses import dataclass, field
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
        # 3.3 (2026-06-27): a WHOLE-line impersonal action chain the per-chunk
        # clause check misses -- non-whitelisted verb leads joined by commas
        # ("snaps off pen's tip, jams it into the port, turning it to scrap")
        # that the heatwave-b008 leak rode through with EMPTY compose_flags.
        if is_whole_line_stage_action(norm):
            return (True, _BARE_STAGE_HINT, "whole_line_action")
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
# L7 (story-quality v2, R3 2026-06-22) -- dialogue|action SPLIT.
#
# The operator's idea (subsumes L6): instead of only STRIPPING a leaked stage
# direction, SPLIT a line into spoken dialogue + the extracted action so the
# action can be RECORDED (on the row's compose_flags) while the spoken text is
# kept clean. Scope: the BALANCED-QUOTE outside-span class (b005/b010/b012
# "<quoted dialogue>" <trailing 3rd-person action>) -- the same class the Tier-3
# freeze floor removes, so the dialogue this produces is byte-for-byte what the
# floor would leave (the strip that still runs after is then a no-op, never a
# double-strip). Leading-bare + undelimited classes stay the existing strip
# chain's job (unchanged -- no regression, no corruption). Reuses
# segment_double_quotes + is_third_person_action_clause (no regex dup).
# ---------------------------------------------------------------------------


def split_stage_business(text: Any) -> "tuple[str, str, str]":
    """Split a spoken line into ``(dialogue, action, reason)``.

    Returns ``action`` NON-EMPTY only on a confident extraction, and then
    ``dialogue`` is guaranteed non-empty + well-formed (``_floor_well_formed``).
    Nothing to split / not confident -> ``(normalized_text, "", "")`` -- the
    caller leaves the row to the normal strip chain. ``reason`` is one of
    ``"leading"`` / ``"trailing_after_quote"`` (matching the floor's reason
    codes). Conservative: ONLY the balanced-quote outside-span class; never
    corrupts the spoken text. Pure; never raises (the caller still wraps it and
    records ``action_split_failed`` on any unexpected error)."""
    try:
        s = "" if text is None else str(text)
        norm, segments = segment_double_quotes(s)
        nq = norm.count('"')
        if not nq or nq % 2 != 0:
            return (norm, "", "")
        actions: list[str] = []
        reason = ""
        rebuilt: list[str] = []
        for idx, (span, in_quote) in enumerate(segments):
            if in_quote:
                rebuilt.append(span)
                continue
            core = span.strip(" ,;:-")
            if core and is_third_person_action_clause(core):
                actions.append(core)
                reason = "leading" if idx == 0 else "trailing_after_quote"
                rebuilt.append("")
            else:
                rebuilt.append(span)
        if not actions:
            return (norm, "", "")
        dialogue = _WS.sub(" ", '"'.join(rebuilt)).strip()
        if dialogue and _floor_well_formed(dialogue):
            return (dialogue, " ".join(actions), reason)
        # extraction would corrupt the spoken text -> no split (keep it whole;
        # the normal strip chain / freeze floor still guards the row).
        return (norm, "", "")
    except Exception:  # noqa: BLE001 -- never raise on a spoken line
        return ("" if text is None else str(text), "", "")


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
    # 3.4 (2026-06-27): generalize the pronoun -- the shipped batch had
    # "You're playing with fire, Watson" reach the ledger past the you-only form.
    r"\b(?:you|we|i|they)['’]?re playing with fire\b",
    r"\bthis changes everything\b",
    r"\bwe['’]re not leaving anything to chance\b",
    r"\bleaving nothing to chance\b",
    r"\bthere['’]s no turning back\b",
    r"\bagainst all odds\b",
    # 3.4 expansion -- phrases the local writer shipped (dialing/compass/marked).
    r"\bhangs in the balance\b",
    r"\bover my dead body\b",
    r"\bnot on my watch\b",
    r"\bbest left buried\b",
    r"\brunning out of time\b",
    r"\bbefore it['’]?s too late\b",
    r"\bsafety first\b",
    r"\bgo(?:es)? up in smoke\b",
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


# ---------------------------------------------------------------------------
# 3.2 (story-quality, 2026-06-27) -- anchor-stuffing + one-breath gates (FLAG
# ONLY; v2-gated + character-only at the composer). The weak-writer failure is a
# line that crams every specificity anchor + every clause into one unsayable
# breath. These read the SAME injected "Specificity anchors (...)" block the
# writer appends to the canon_header (_otr_specificity.inject_anchors_into_header)
# so the engine and the scan agree. Pure, deterministic, never raise.
# ---------------------------------------------------------------------------

#: Prefix that marks the injected specificity-anchor block in a canon_header.
#: Kept as a lowercase substring so a minor wording drift in the writer's block
#: head does not silently zero the parse (matched case-insensitively).
_ANCHOR_BLOCK_PREFIX = "specificity anchors"


def extract_specificity_anchors_from_header(canon_header: Any) -> "list[str]":
    """Parse the bullet anchors out of the injected "Specificity anchors (...)"
    block of a canon_header, in HEADER ORDER (no `LineRequest` field). Returns
    ``[]`` when the block is absent / empty. Pure; deterministic; never raises."""
    try:
        s = "" if canon_header is None else str(canon_header)
        if not s:
            return []
        anchors: list[str] = []
        in_block = False
        for raw in s.splitlines():
            stripped = raw.strip()
            if not in_block:
                if stripped.lower().startswith(_ANCHOR_BLOCK_PREFIX):
                    in_block = True
                continue
            if stripped.startswith("- "):
                item = stripped[2:].strip()
                if item:
                    anchors.append(item)
            elif not stripped:
                break          # a blank line closes the block
            else:
                break          # a non-bullet line closes the block
        return anchors
    except Exception:  # noqa: BLE001
        return []


def flag_anchor_stuffing(
    text: Any, anchors: Any, *, threshold: int = 3,
) -> "tuple[bool, str]":
    """(flagged, hint): a single character line crams ``>= threshold`` DISTINCT
    specificity anchors. Plain casefold-substring match (MF-3: key_terms carry
    "41.3 degrees C", "837/835", hyphens -- they match literally). The hint lists
    the present anchors in header order. Pure; never raises."""
    try:
        s = str(text or "")
        if not s.strip():
            return False, ""
        low = s.casefold()
        seen: set = set()
        present: list[str] = []
        for a in (anchors or ()):
            a = str(a or "").strip()
            if not a:
                continue
            key = a.casefold()
            if key in seen:
                continue
            seen.add(key)
            if key in low:
                present.append(a)
        if len(present) >= threshold:
            return True, (
                "anchor-stuffing: this one line carries "
                f"{len(present)} concrete anchors ({', '.join(present)}) -- keep "
                "at most one and let the others live in other lines"
            )
        return False, ""
    except Exception:  # noqa: BLE001
        return False, ""


#: A soft word ceiling: above this AND with excessive clause nesting trips the
#: one-breath gate even before the hard ceiling.
_ONE_BREATH_SOFT_WORDS = 22
_CLAUSE_MARK_RE = re.compile(r"[,;]")
_CLAUSE_CONJ = frozenset({
    "and", "but", "or", "so", "then", "while", "because", "which", "that", "as",
    "though", "although", "whereas",
})


def flag_one_breath(
    text: Any, *, max_words: int = 28, max_clause_markers: int = 3,
) -> "tuple[bool, str]":
    """(flagged, hint): the line is too long / too nested to say in one spoken
    breath -- ``> max_words`` words, OR ``> _ONE_BREATH_SOFT_WORDS`` words with
    ``>= max_clause_markers`` comma/semicolon/conjunction breaks. Pure; never
    raises."""
    try:
        s = " ".join(str(text or "").split())
        if not s:
            return False, ""
        words = re.findall(r"[A-Za-z']+", s)
        n = len(words)
        if n > max_words:
            return True, (
                f"one-breath: {n} words is too long to speak in a single breath "
                "-- cut it to one short spoken clause"
            )
        if n > _ONE_BREATH_SOFT_WORDS:
            marks = len(_CLAUSE_MARK_RE.findall(s))
            conj = sum(1 for w in words if w.lower() in _CLAUSE_CONJ)
            if marks + conj >= max_clause_markers:
                return True, (
                    f"one-breath: {n} words across {marks + conj} clauses -- "
                    "break it into fewer, shorter spoken beats"
                )
        return False, ""
    except Exception:  # noqa: BLE001
        return False, ""


# ---------------------------------------------------------------------------
# 3.3 (story-quality, 2026-06-27) -- whole-line stage-action leak detector.
#
# Broadens the leak net beyond the verb-whitelisted is_third_person_action_clause
# (<=12 words, lead must be a _NARRATION_VERBS verb): a WHOLE character line that
# is an impersonal 3rd-person action chain ("snaps off pen's tip, jams it into
# the port, turning it into scrap metal") with NO 1st/2nd-person pronoun and no
# quote reads as a leaked stage direction. STRUCTURAL (not a fixed whitelist) but
# guarded by BN-1 so terse dialogue ("Looks like rain, Watson.") is never tripped.
# Drives a reroll (never silently strips). Pure; deterministic; never raises.
# ---------------------------------------------------------------------------

#: BN-1 dialogue-opener allowlist: a perception/copula opener is dialogue, not a
#: stage direction, even though it leads with a 3rd-person-singular verb.
_DIALOGUE_OPENER_ALLOW = (
    "looks like", "sounds like", "seems like", "seems ", "seem ",
    "feels like", "smells like", "looks ",
)
_THIRD_PERSON_SUBJECTS = frozenset({"he", "she", "they"})


def is_whole_line_stage_action(text: Any, *, max_words: int = 32) -> bool:
    """True iff the WHOLE line reads as an impersonal 3rd-person stage-action
    chain (a leaked direction), not spoken dialogue.

    Requires: no double quote; ``<= max_words`` words; NO 1st/2nd-person pronoun
    root anywhere (3rd-person he/she/they is permitted as the actor); an
    action-verb lead (in the extended ``_NARRATION_VERBS``, or an ``-ing``/
    ``-ed``/``-s`` inflection); and -- BN-1 false-positive guard -- when the lead
    is NOT whitelisted, the line must carry a comma-separated action chain OR an
    explicit 3rd-person subject. A perception/copula opener (``Looks like rain,
    Watson.``) is exempt. Pure; never raises."""
    try:
        s = " ".join(str(text or "").split())
        if not s:
            return False
        norm = _CURLY_DQUOTE_RE.sub(lambda m: _CURLY_DQUOTE[m.group(0)], s)
        if '"' in norm:
            return False
        low = norm.lower()
        # BN-1 allowlist: a perception/copula opener is dialogue.
        for opener in _DIALOGUE_OPENER_ALLOW:
            if low.startswith(opener):
                return False
        words = re.findall(r"[a-z']+", low)
        if not words or len(words) > max_words:
            return False
        # no 1st/2nd-person pronoun root anywhere -> impersonal / 3rd-person.
        for w in words:
            root = re.split(r"['’]", w)[0]
            if w in _PRONOUN_ROOTS or root in _PRONOUN_ROOTS:
                return False
        lead = words[0]
        if lead in _DIALOGUE_STARTER:
            return False
        third_subject = False
        verb = lead
        if lead in _THIRD_PERSON_SUBJECTS and len(words) > 1:
            third_subject = True
            verb = words[1]
        if verb in _COPULA_MODAL or verb in _DIALOGUE_STARTER:
            return False
        whitelisted = verb in _NARRATION_VERBS
        # A contraction lead ("it's", "that's", "here's") ends in -s but is
        # "<subject> is", not an action verb -> never verbish.
        verbish = bool(verb) and "'" not in verb and "’" not in verb and (
            verb.endswith("ing")
            or verb.endswith("ed")
            or (verb.endswith("s") and len(verb) > 2)
        )
        if not (whitelisted or verbish):
            return False
        # BN-1: a non-whitelisted lead needs a comma-separated action chain (the
        # real heatwave-b008 "snaps..., jams..., turning..." signature), so a
        # short non-action 3rd-person line ("She knows the code.") -- subject +
        # cognition verb, no chain -- is never swept up.
        if not whitelisted and "," not in norm:
            return False
        return True
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# 3.6 (story-quality, 2026-06-27) -- personal-cost boilerplate flag (BN-3).
#
# The L12 _enrich_tail cost clause was DROPPED from beat intents (commit
# c005f1e3), but the generic phrasing ("the trust they will lose either way")
# still leaks into a SPOKEN line when the writer echoes a prior beat intent. This
# dedicated detector (NOT the announcer-scoped _BANNED_THESIS_RES) flags it on a
# character line. Matches the three canonical _PERSONAL_COST phrases + light
# pronoun flex; a CONCRETE consequence ("loses access to the archive") is left
# alone. Pure; never raises.
# ---------------------------------------------------------------------------

_PERSONAL_COST_BOILERPLATE_RES = tuple(re.compile(p, re.IGNORECASE) for p in (
    r"what it costs (?:them|us|me|him|her|you|then)\b.{0,16}to be the one who",
    r"the trust .{0,20}\blose\b.{0,8}either way",
    r"the part of (?:themselves|ourselves|myself|himself|herself|yourself|"
    r"yourselves)\b.{0,20}to set down",
))


def flag_personal_cost_boilerplate(text: Any) -> "tuple[bool, str]":
    """(flagged, hint): the line leans on the generic L12 personal-cost
    boilerplate instead of a concrete consequence. Pure; never raises."""
    try:
        s = str(text or "")
        for rx in _PERSONAL_COST_BOILERPLATE_RES:
            m = rx.search(s)
            if m:
                return True, (
                    f"personal-cost boilerplate {m.group(0)!r} -- name the "
                    "concrete thing this character loses, not a generic cost"
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


# ---------------------------------------------------------------------------
# L3 (story-quality LIFT, 2026-06-23): ACTION-marker strip
# ---------------------------------------------------------------------------
# When OTR_COMPOSER_ACTION_STRIP is on, the composer prompt asks the model to
# put any non-spoken stage action on its OWN segment with a leading "ACTION:".
# This deterministically removes any such segment from the SHIPPED text (right
# after compose/polish, before persistence). Strips only what the model marked;
# L4 catches the rest. NEVER token-surgers dialogue: an ACTION: run goes from
# the marker to end-of-line / end-of-string, anywhere in the text.
_ACTION_MARKER_RE = re.compile(r"(?im)(?:^|(?<=[.!?\"'\)\]\s]))ACTION:\s*[^\n]*")


def strip_action_marker(text: Any) -> "tuple[str, str, int]":
    """Remove ACTION:-marked stage business. Returns (clean, action, n).

    ``clean`` is the spoken text with every ``ACTION: ...`` run removed and
    whitespace re-collapsed; ``action`` is the concatenated removed text (for
    audit, NOT persisted); ``n`` is how many markers were removed. Idempotent;
    never raises. A line that is ENTIRELY an ACTION: marker collapses to ""
    (the caller decides whether to keep the strip)."""
    try:
        s = str(text or "")
        if "action:" not in s.casefold():
            return s, "", 0
        removed: list[str] = []

        def _grab(m: "re.Match") -> str:
            removed.append(m.group(0))
            return " "

        clean = _ACTION_MARKER_RE.sub(_grab, s)
        clean = _WS.sub(" ", clean).strip()
        # tidy a space left before terminal punctuation
        clean = re.sub(r"\s+([.,!?;:])", r"\1", clean)
        return clean, " ".join(r.strip() for r in removed).strip(), len(removed)
    except Exception:  # noqa: BLE001
        return str(text or ""), "", 0


# ---------------------------------------------------------------------------
# L4 (story-quality LIFT, 2026-06-23): minimal transcript sanitizer
# ---------------------------------------------------------------------------
# Final line TEXT only, run after text is final + before freeze/TTS/hash. Strips
# prompt-leak / director-note patterns and balances a stray wrapper quote.
# CONSERVATIVE: never touches apostrophes, measurements, or identity fields, and
# never repairs mojibake (that is a verify-only build artifact -- see
# detect_mojibake). Returns (clean, reasons).
_PROMPTLEAK_PATTERNS = (
    re.compile(r"(?im)(?:^|(?<=[.!?]\s))(?:note|director|stage|voice|delivery|tone)\s*:\s*[^.!?\n]*[.!?]?"),
    re.compile(r"(?im)\b(?:the\s+)?voice\s+should\s+[^.!?\n]*[.!?]?"),
    re.compile(r"(?im)\bas\s+(?:the\s+)?ai[,]?\s+[^.!?\n]*[.!?]?"),
    re.compile(r"(?im)\(\s*(?:director|note|stage|voice)\b[^)]*\)"),
)
# Mojibake signatures (verify-only; NEVER mutated in v0).
_MOJIBAKE_RE = re.compile(r"[ÃÂâ€][\x80-\xBF–—‘-‟]")


def detect_mojibake(text: Any) -> bool:
    """True iff the text shows common UTF-8/cp1252 mojibake. VERIFY-ONLY --
    callers log/measure; v0 does NOT repair encoding."""
    try:
        return bool(_MOJIBAKE_RE.search(str(text or "")))
    except Exception:  # noqa: BLE001
        return False


def _balance_wrapper_quotes(s: str) -> "tuple[str, bool]":
    """Drop a single dangling wrapper double-quote (odd count). Conservative:
    only acts when there is exactly one straight double-quote at an edge."""
    if s.count('"') % 2 == 0:
        return s, False
    stripped = s
    changed = False
    if stripped.startswith('"') and stripped.count('"') % 2 == 1:
        stripped = stripped[1:].lstrip()
        changed = True
    elif stripped.endswith('"') and stripped.count('"') % 2 == 1:
        stripped = stripped[:-1].rstrip()
        changed = True
    return stripped, changed


def sanitize_transcript_text(text: Any) -> "tuple[str, list]":
    """Strip prompt-leak / director-note patterns + balance a stray wrapper
    quote. Returns (clean, reasons). Conservative + idempotent; never raises.
    Does NOT touch speaker labels or identity fields (caller passes TEXT only)
    and never repairs mojibake."""
    try:
        s = str(text or "")
        reasons: list[str] = []
        for rx in _PROMPTLEAK_PATTERNS:
            new = rx.sub(" ", s)
            if new != s:
                reasons.append(f"prompt_leak:{rx.pattern[:18]}")
                s = new
        s = _WS.sub(" ", s).strip()
        s = re.sub(r"\s+([.,!?;:])", r"\1", s)
        s, q = _balance_wrapper_quotes(s)
        if q:
            reasons.append("wrapper_quote_balanced")
            s = s.strip()
        if detect_mojibake(s):
            reasons.append("mojibake_detected")   # verify-only, no mutation
        return s, reasons
    except Exception:  # noqa: BLE001
        return str(text or ""), []


# ---------------------------------------------------------------------------
# leak-floor-v2 (2026-06-25, docs/2026-06-25-leaking-words/) -- ONE mandatory
# deterministic offline verifier over four named leak classes, using NARROW
# STRUCTURAL extract-or-fail rules (NOT verb-whitelist widening, NOT broad -ing
# scrubbing). PURE: stdlib only, deterministic, idempotent, never raises. The
# composer/writer gate every call behind ``_otr_config.leak_floor_v2_enabled()``
# so this layer ships DARK (default-OFF, byte-identical) until a live validation
# promotes it. The GROUNDED root cause of the shipped ``Gasping, "..."`` leak is
# ``_leading_stage_strip``'s lowercase-start guard (line 271), which never
# consults the verb whitelist -- so rule 1 is a NEW SIBLING, and we do NOT touch
# ``_leading_stage_strip`` or widen ``_NARRATION_VERBS``. Rule 4 (news-bleed)
# lives at ``_otr_line_composer.build_allowed_roster`` (fixed at the existing
# roster gate, not by a new detector); the verifier only ECHOES it as a
# line-level membership check over ``EntityPolicy.banned``.
# ---------------------------------------------------------------------------

#: A leading CAPITALISED PARTICIPLE wrapper ("Gasping," / "Frowned,") the model
#: emitted BEFORE the actual quoted dialogue. Anchored to be the WHOLE
#: outside-segment-0 (``$``) so "Running to the door, I shouted" (more after the
#: word) never matches, and ``"Running," she said`` (starts WITH the quote, so
#: outside-segment-0 is empty) never matches. ``[a-z]+`` before ``ing|ed`` needs
#: >=1 lead char, so short imperatives ("King," / "Red,") do not match.
_PARTICIPLE_WRAPPER_RE = re.compile(r"^[A-Z][a-z]+(?:ing|ed),\s*$")


def strip_participle_before_quote(text: Any) -> "tuple[str, bool]":
    """Rule 1. Extract the quoted dialogue when a line is a bare capitalised
    participle wrapper + a quoted span (``Gasping, "I can't believe it."`` ->
    ``I can't believe it.``). Returns ``(text, changed)``; unchanged unless the
    outside wrapper matches ``_PARTICIPLE_WRAPPER_RE`` AND the first quoted span
    is non-empty (the required quote is the false-positive guard). Reuses
    ``segment_double_quotes`` after curly->straight normalisation. Pure;
    idempotent (the result carries no wrapper / no double quotes); never
    raises."""
    try:
        s = "" if text is None else str(text)
        norm, segments = segment_double_quotes(s)
        nq = norm.count('"')
        # need one BALANCED quoted span: segment[0] outside, segment[1] in-quote
        if nq < 2 or nq % 2 != 0 or len(segments) < 2:
            return s, False
        seg0_text, seg0_in_quote = segments[0]
        seg1_text, seg1_in_quote = segments[1]
        if seg0_in_quote or not seg1_in_quote:
            return s, False
        if _PARTICIPLE_WRAPPER_RE.match(seg0_text) and seg1_text.strip():
            return seg1_text.strip(), True
        return s, False
    except Exception:  # noqa: BLE001 -- never raise on a spoken line
        return ("" if text is None else str(text)), False


def scrub_roster_vocative(text: Any, roster_fullnames: Any) -> str:
    """Rule 2. Drop an ALL-CAPS FULL-NAME vocative addressing another cast
    member (``YUKI MARTIN, no!`` -> ``no!``; ``Get out, YUKI MARTIN!`` ->
    ``Get out!``).

    ``scrub_self_vocative`` (above) only strips the SPEAKER's own name; this
    targets ANOTHER character's full name. Full names ONLY (a name contains a
    space), so a first-name vocative (``YUKI!``) is never touched. Matched
    longest-first, ALL-CAPS only (the shouty leaked form -- a normal-case
    ``Yuki Martin, hello`` is left to ordinary dialogue), at a vocative position
    only (leading ``^NAME[,!:-]`` or trailing ``[, ]NAME[.!?]?$``). DROPs the
    vocative (deterministic -- not title-case) and preserves terminal
    punctuation. Returns the original if scrubbing would empty the line. Pure;
    idempotent; never raises."""
    try:
        s = "" if text is None else str(text)
        names = sorted(
            {str(n).strip() for n in (roster_fullnames or ())
             if n and " " in str(n).strip()},
            key=len, reverse=True,
        )
        if not names:
            return s
        out = s
        for name in names:
            esc = re.escape(name.upper())  # ALL-CAPS literal, no IGNORECASE
            # leading vocative: ^NAME[,!:-] ...
            out = re.sub(rf"^\s*{esc}\s*[,!:\-]+\s*", "", out)
            # trailing vocative: [, ]NAME[.!?]?$ -> drop name, keep terminal punct
            out = re.sub(
                rf"[,\s]+{esc}\s*([.!?])?\s*$",
                lambda m: (m.group(1) or ""), out,
            )
        out = _WS.sub(" ", out).strip()
        return out or s.strip()
    except Exception:  # noqa: BLE001
        return ("" if text is None else str(text))


def has_malformed_internal_quote(text: Any) -> bool:
    """Rule 3. True iff the line has an ODD number of double quotes that is NOT
    a single EDGE wrapper -- an unclosed INTERNAL quote that must be recomposed.
    An edge wrapper (exactly one ``"`` at the start XOR the end) is NOT malformed
    (``sanitize_transcript_text`` balances it). Double quotes ONLY -- apostrophes
    / ``don't`` never trip it (``segment_double_quotes`` ignores single quotes).
    Pure; never raises."""
    try:
        norm, _segments = segment_double_quotes(text)
        cnt = norm.count('"')
        if cnt % 2 == 0:
            return False
        stripped = norm.strip()
        edge_wrapper = (
            cnt == 1
            and (stripped.startswith('"') != stripped.endswith('"'))
        )
        return not edge_wrapper
    except Exception:  # noqa: BLE001
        return False


@dataclass(frozen=True)
class EntityPolicy:
    """Transient per-episode entity policy for leak-floor-v2 (NOT persisted --
    the ledger schema stays frozen). ``allowed`` = cast + setting + fictional
    world nouns + legit news orgs (NASA/CERN); ``banned`` = real-person /
    political-figure source entities (President Trump, ...). Both UPPERCASE by
    convention. Invariant: ``banned & allowed == frozenset()``."""
    allowed: frozenset = field(default_factory=frozenset)
    banned: frozenset = field(default_factory=frozenset)


@dataclass(frozen=True)
class Defect:
    """One leak-floor-v2 finding. ``target_span`` is ``(start, end)`` for tests
    + telemetry; ``(-1, -1)`` when not span-anchored."""
    reason_code: str
    target_span: "tuple[int, int]" = (-1, -1)


@dataclass(frozen=True)
class VerificationResult:
    """Result of ``verify_and_repair_line``. ``text`` is the (possibly repaired)
    line; ``changed`` is True iff a rule mutated it; ``needs_recompose`` asks the
    composer for ONE reroll; ``failed`` is set only under ``strict`` once the
    recompose budget is spent; ``compose_flags`` are the per-line breadcrumbs."""
    text: str
    changed: bool = False
    defects: "tuple[Defect, ...]" = ()
    needs_recompose: bool = False
    failed: bool = False
    compose_flags: "tuple[str, ...]" = ()


def verify_and_repair_line(
    text: Any,
    req: Any = None,
    policy: Any = None,
    *,
    strict: bool = False,
    repair_budget: int = 1,
) -> "VerificationResult":
    """leak-floor-v2 entry point: run the four deterministic leak rules on ONE
    composed line BEFORE TTS/freeze.

    Rules 1-2 EXTRACT / scrub in place (the text is repaired); rule 3 (malformed
    internal quote) + rule 4 (a banned source entity survived into the line) set
    ``needs_recompose`` so the composer reroll handles them -- falling back to
    best-effort + telemetry, or ``failed`` under ``strict`` once the recompose
    budget is spent. ``policy`` (an ``EntityPolicy``) supplies the ALL-CAPS
    full-name roster (rule 2) + the banned-entity set (rule 4). ``req`` is
    accepted for signature parity / future per-request tuning (unused today).
    PURE -- no env reads (the caller passes ``strict`` from
    ``strict_local_clean_enabled()``); deterministic; idempotent; never raises.
    """
    try:
        s = "" if text is None else str(text)
        defects: list = []
        flags: list = []
        changed = False
        needs_recompose = False
        out = s

        # Rule 1: capitalised participle before a quote -> extract the dialogue.
        r1, c1 = strip_participle_before_quote(out)
        if c1:
            defects.append(Defect("capitalized_participle_before_quote"))
            flags.append("leak_floor:participle_before_quote")
            out = r1
            changed = True

        # Rule 2: ALL-CAPS full-name roster vocative -> drop.
        fullnames: tuple = ()
        if policy is not None:
            try:
                fullnames = tuple(
                    n for n in policy.allowed
                    if isinstance(n, str) and " " in n
                )
            except Exception:  # noqa: BLE001
                fullnames = ()
        r2 = scrub_roster_vocative(out, fullnames)
        if r2 != out:
            defects.append(Defect("roster_vocative"))
            flags.append("leak_floor:roster_vocative")
            out = r2
            changed = True

        # Rule 3: malformed INTERNAL double quote -> recompose (no in-place fix).
        if has_malformed_internal_quote(out):
            defects.append(Defect("malformed_quote"))
            flags.append("leak_floor:malformed_quote")
            needs_recompose = True

        # Rule 4 (line-level echo): a banned real-person / political entity
        # survived into the line. PRIMARY enforcement is upstream at
        # build_allowed_roster (the phantom/roster gate rerolls it); this
        # membership check forces a recompose even if it slipped the roster.
        if policy is not None:
            try:
                low = out.lower()
                for b in policy.banned:
                    bs = str(b).strip().lower()
                    if bs and bs in low:
                        defects.append(Defect("banned_source_entity"))
                        flags.append("leak_floor:banned_source_entity")
                        needs_recompose = True
                        break
            except Exception:  # noqa: BLE001
                pass

        failed = bool(strict and needs_recompose and repair_budget <= 0)
        return VerificationResult(
            text=out,
            changed=changed,
            defects=tuple(defects),
            needs_recompose=needs_recompose,
            failed=failed,
            compose_flags=tuple(flags),
        )
    except Exception:  # noqa: BLE001 -- the verifier must never break a render
        return VerificationResult(text=("" if text is None else str(text)))
