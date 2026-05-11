"""nodes/_otr_line_composer.py

Per-beat dialogue line generation for the v2.0 LedgerScriptWriter path.

Takes one Beat + EpisodeCanon header + last N ledger lines, generates ONE
raw dialogue string from the LLM, strips any leaked formatting (speaker
prefixes, brackets, markdown, wrapping quotes), returns the cleaned text
plus any compose-time flags (e.g. phantom-name detections).

The LLM is told to output only the spoken line. Python attaches the
[VOICE: NAME, traits] format tag deterministically at ledger-stamp time
(in OTR_LedgerScriptWriter, not here). This module never produces or
expects format markup.

Status: Phase 2 of v2.0 sprint, extended with Phase 0 name-roster gate
(2026-05-11). Companion to _otr_outline.py.

Public surface:
    LineRequest                   -- frozen dataclass: per-line input
    LineResult                    -- frozen dataclass: (text, compose_flags)
    LineCompositionFailedError    -- raised after 2 failed attempts
    compose_line(...)             -- main entrypoint, returns LineResult
    strip_line_formatting(...)    -- public for testing / one-shot use
    build_allowed_roster(...)     -- assemble UPPERCASE roster for the gate
    detect_phantom_names(...)     -- proper-noun extractor + roster check
    aggregate_compose_flags(...)  -- post-loop helper, stamps meta summary
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

log = logging.getLogger("OTR")


__all__ = [
    "LineRequest",
    "LineResult",
    "LineCompositionFailedError",
    "compose_line",
    "strip_line_formatting",
    "build_allowed_roster",
    "detect_phantom_names",
    "aggregate_compose_flags",
    # Phase 1 (2026-05-11)
    "render_outline_spine",
    "build_voice_card",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Generation params
_BASE_TEMPERATURE = 0.8
_MAX_NEW_TOKENS_PER_LINE = 200  # ~150 words max, generous for any beat
_MAX_OVERSIZE_RATIO = 3.0       # response > 3x target_words triggers retry

# Format-strip regexes (applied in order in strip_line_formatting)
_PREFIX_VOICE_TAG_RE = re.compile(
    r"^\s*\[\s*(?:VOICE\s*:\s*)?[A-Z][A-Z0-9_ .]{0,30}(?:\s*,\s*[^\]]+)?\s*\]\s*",
    re.IGNORECASE,
)
_PREFIX_SPEAKER_COLON_RE = re.compile(
    r"^\s*[A-Z][A-Z0-9_ .]{0,30}\s*[:\-—]\s*",
)
_MD_BOLD_ITALIC_RE = re.compile(r"(\*\*|__|\*|_|`)")
_QUOTES_WRAP_RE = re.compile(
    r'^\s*[“”‘’"\']\s*(.*?)\s*[“”‘’"\']\s*$',
    re.DOTALL,
)


# ---------------------------------------------------------------------------
# Format-strip pipeline (public for testability)
# ---------------------------------------------------------------------------


def strip_line_formatting(raw: str) -> str:
    """Remove leaked formatting from a raw LLM line response.

    Applies in order:
      1. Trim outer whitespace.
      2. Strip wrapping quotes (smart or straight, single or double).
      3. Strip leading [VOICE: NAME, traits] or [NAME, traits] tag.
      4. Strip leading SPEAKER: / SPEAKER - / SPEAKER -- prefix.
      5. Strip markdown bold/italic/code markers.
      6. Trim outer whitespace again.

    Returns the cleaned dialogue text. May return empty string if the
    response was nothing but formatting. Never raises.
    """
    if not raw:
        return ""
    s = raw.strip()
    # Step 2: wrapping quotes
    m = _QUOTES_WRAP_RE.match(s)
    if m:
        s = m.group(1).strip()
    # Step 3: leading bracket tag
    s = _PREFIX_VOICE_TAG_RE.sub("", s, count=1).strip()
    # Step 4: leading speaker colon/dash prefix
    s = _PREFIX_SPEAKER_COLON_RE.sub("", s, count=1).strip()
    # Step 5: markdown markers
    s = _MD_BOLD_ITALIC_RE.sub("", s).strip()
    # Second pass: markdown removal can expose previously-hidden speaker
    # tags (e.g. "**[ALICE]**" -> "[ALICE]" after step 5). Re-run the
    # bracket and colon-prefix strips to catch markdown-wrapped tags.
    s = _PREFIX_VOICE_TAG_RE.sub("", s, count=1).strip()
    s = _PREFIX_SPEAKER_COLON_RE.sub("", s, count=1).strip()
    return s


# ---------------------------------------------------------------------------
# Name-roster gate (Phase 0, 2026-05-11)
# ---------------------------------------------------------------------------
#
# The composer prompt tells Mistral-Nemo the speaker by name but does
# NOT (in v2.0-alpha) list the full cast. When a beat intent has ALICE
# reference another character or organization, the model invents one.
# Phantom names propagate silently to the ledger.
#
# Phase 0 fix: pass an UPPERCASE allowed_roster on every LineRequest,
# extract proper-noun candidates from each composed line via heuristic
# regex, flag any candidate not in the roster on the line row's
# compose_flags field. The composer does NOT reroll on a name violation
# (cast is locked; an LLM reroll cannot invent a different correct
# name). Phase 3's reviewer + Step 2.5 deterministic phantom-skip
# fallback handle repair downstream.
#
# Roster composition per §6.A (Option 1, strict):
#   - cast names (UPPERCASE from cast_rows)
#   - "ANNOUNCER" (always)
#   - key_terms from news_interpreter (uppercased)
# News-seed proper nouns are NOT widened in. The strict roster makes
# every undeclared name visible to the reviewer.

# ALL-CAPS tokens, ≥2 chars (catches "ALICE", "CERN", "JPL", "USA-CERN").
_ALL_CAPS_TOKEN_RE = re.compile(r"\b[A-Z]{2,}(?:[-_][A-Z0-9]+)*\b")

# Titled names ("Dr. Patel", "Sgt. Howard"). Captures the canonical
# title list the synthesis spec calls out plus a handful of common
# military / civic titles we've seen in soak output.
_TITLED_NAME_RE = re.compile(
    r"\b(?:Dr|Mr|Ms|Mrs|Prof|Lt|Capt|Cmdr|Adm|Sen|Sgt|Col|Gen)"
    r"\.\s+[A-Z][a-z]+\b"
)

# Title-Case bigrams ("Joe Smith", "New York"). Only flagged mid-
# sentence — sentence-start capitalization is orthography, not a
# proper-noun signal. _detect_phantom_names strips the first word
# of each sentence before scanning with this regex.
_TITLE_CASE_BIGRAM_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")

# Sentence boundary. Naive but sufficient for audio-drama dialogue
# (which doesn't carry initials like "Mr. J. R. R. Tolkien").
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_LEADING_WORD_RE = re.compile(r"^\W*\w+[\W]*")

# Common ALL-CAPS English tokens that are not names. Keep the list
# short and conservative — anything in news content (CERN, JPL, NASA,
# etc.) must be threaded via key_terms, not allowlisted here.
_COMMON_ALLCAPS_NON_NAMES: frozenset[str] = frozenset({
    "OK", "TV", "AI", "USA", "UK", "EU", "UN", "DNA", "RNA",
    "AM", "PM",
})


def build_allowed_roster(
    cast_rows: Iterable[object] = (),
    key_terms: Iterable[str] = (),
    *,
    include_announcer: bool = True,
) -> frozenset[str]:
    """Build the UPPERCASE allowed_roster for the Phase 0 name gate.

    Inputs:
      cast_rows         iterable of cast row dicts (each with "name")
                        OR (name, ...) tuples. Both shapes accepted so
                        unit tests can pass minimal fixtures without
                        constructing full ledger rows. Empty / falsy
                        rows are skipped.
      key_terms         iterable of journalistic terms from
                        news_interpreter's briefs. Uppercased and
                        merged into the roster so dialogue that
                        surfaces CERN / JPL / Voyager (etc.) does not
                        trigger phantom-name flags.
      include_announcer keep "ANNOUNCER" in the roster (default True).
                        Set False only in pure-test contexts.

    Returns a frozenset of UPPERCASE strings. Whitespace trimmed on
    each entry; empty entries dropped. Stable across calls — no RNG,
    no time, no os.

    Per §6.A (Option 1, strict): the roster does NOT widen with
    arbitrary proper nouns from news_seed. Names that legitimately
    belong in dialogue must arrive via key_terms.
    """
    roster: set[str] = set()
    if include_announcer:
        roster.add("ANNOUNCER")
    for row in cast_rows or ():
        name = ""
        if isinstance(row, dict):
            name = str(row.get("name") or "").strip()
        elif isinstance(row, (list, tuple)) and row:
            name = str(row[0] or "").strip()
        elif isinstance(row, str):
            name = row.strip()
        if name:
            roster.add(name.upper())
    for term in key_terms or ():
        term_s = str(term or "").strip()
        if term_s:
            roster.add(term_s.upper())
    return frozenset(roster)


def _strip_sentence_lead_word(sentence: str) -> str:
    """Drop the first word of a sentence (and any leading punctuation).

    Used to skip sentence-start capitalization when scanning for
    Title-Case bigrams. Returns the remainder of the sentence (may be
    empty if the sentence was one word).
    """
    if not sentence:
        return ""
    m = _LEADING_WORD_RE.match(sentence)
    if not m:
        return sentence
    return sentence[m.end():]


def detect_phantom_names(
    text: str,
    speaker: str,
    allowed_roster: frozenset[str],
) -> list[str]:
    """Return proper-noun candidates in `text` that are NOT in the roster.

    Run after `strip_line_formatting`. Three heuristics, in order:
      1. ALL-CAPS tokens (length ≥ 2)
      2. Titled names (Dr./Mr./Sgt./etc. + Capitalized word)
      3. Title-Case bigrams (mid-sentence only — sentence-start skipped)

    A candidate is a phantom iff its UPPERCASE form is NOT in
    `allowed_roster` and NOT the speaker's own name (the composer is
    told not to say its own name, but if it slips through,
    `strip_line_formatting` already removes it; flagging it as a
    phantom would be a false positive).

    Returns a list of phantoms in first-seen order, de-duplicated.
    Never raises.
    """
    if not text:
        return []
    speaker_u = (speaker or "").strip().upper()
    found: dict[str, None] = {}

    # 1. ALL-CAPS tokens — anywhere in text.
    for m in _ALL_CAPS_TOKEN_RE.finditer(text):
        tok = m.group(0).strip()
        if not tok:
            continue
        tok_u = tok.upper()
        if tok_u == speaker_u:
            continue
        if tok_u in allowed_roster:
            continue
        if tok_u in _COMMON_ALLCAPS_NON_NAMES:
            continue
        found.setdefault(tok, None)

    # 2. Titled names — anywhere.
    for m in _TITLED_NAME_RE.finditer(text):
        tok = m.group(0).strip()
        if not tok:
            continue
        tok_u = tok.upper()
        if tok_u == speaker_u:
            continue
        if tok_u in allowed_roster:
            continue
        found.setdefault(tok, None)

    # 3. Title-Case bigrams — mid-sentence only.
    sentences = _SENTENCE_SPLIT_RE.split(text.strip())
    for sentence in sentences:
        body = _strip_sentence_lead_word(sentence)
        for m in _TITLE_CASE_BIGRAM_RE.finditer(body):
            tok = m.group(0).strip()
            if not tok:
                continue
            tok_u = tok.upper()
            if tok_u == speaker_u:
                continue
            if tok_u in allowed_roster:
                continue
            # Skip if the bigram is itself a titled name already
            # caught by pass 2 (avoid double-reporting "Dr. Patel"
            # if its trailing surname happens to be Title-Case).
            if _TITLED_NAME_RE.fullmatch(tok):
                continue
            found.setdefault(tok, None)

    return list(found.keys())


def aggregate_compose_flags(ledger_data: dict) -> dict[str, int]:
    """Count compose_flags by kind across every line in the ledger.

    Walks `ledger_data["lines"]`, splits each flag at the first ":"
    to extract its kind, and returns the count map. Stamped to
    `meta.compose_flag_summary` by the orchestrator at end of run so
    Jeffrey can skim "did this run have 0 phantom flags or 12?"
    without grep-walking every line.

    Pure Python; no LLM cost. Never raises. Empty ledger → {}.
    """
    summary: dict[str, int] = {}
    if not isinstance(ledger_data, dict):
        return summary
    for line in ledger_data.get("lines", []) or []:
        if not isinstance(line, dict):
            continue
        for flag in line.get("compose_flags", []) or []:
            kind = str(flag).split(":", 1)[0].strip()
            if not kind:
                continue
            summary[kind] = summary.get(kind, 0) + 1
    return summary


# ---------------------------------------------------------------------------
# Request / Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LineRequest:
    """Per-beat input for compose_line.

    Fields are duplicated from Beat (rather than passing the Beat directly)
    to keep this module's import surface stdlib-only at module load. The
    caller maps Beat fields into LineRequest.

    Phase 0 (2026-05-11): `allowed_roster` field added for the name-
    gate check. Defaults to an empty frozenset for back-compat with
    early-stage tests; orchestrator MUST populate it on every real
    call (the roster is built once via build_allowed_roster after
    cast lock + news_interpreter).

    Phase 1 (2026-05-11): three new context fields replace the
    composer's previous "speaker + intent + mood + canon header +
    last 3 lines" diet:

      style_descriptor          full snake_case style for the episode
                                (from _otr_style_picker). Empty string
                                skips the STYLE block entirely.
      outline_spine             one-line-per-beat compact rendering
                                of the whole outline so the composer
                                can see the arc it is participating
                                in. Empty string skips the OUTLINE
                                block entirely. Renderer ships in this
                                module (`render_outline_spine`).
      character_voice_card      one-line `name (gender, traits)` blurb
                                for the speaker of this beat. Empty
                                string skips the CHARACTER block. Built
                                from cast_rows via `build_voice_card`.

    Prompt placement is static-first / variable-second (style +
    canon_header + outline_spine + allowed_roster are stable across
    every composer call in the episode; voice_card + last_lines +
    speaker + intent change per call). Once KV-cache reuse lands in
    the loader (deferred; tracked in the ADR), the cached prefix
    covers everything up to the CHARACTER block.

    All Phase 1 fields default to empty strings so unit tests and
    early-stage callers that don't have them yet keep working.
    """

    speaker: str
    intent: str
    mood: str
    target_words: int
    canon_header: str               # from render_episode_canon_header()
    last_lines: list[tuple[str, str]]  # [(speaker, text), ...] most recent last; empty for first beat
    allowed_roster: frozenset[str] = field(default_factory=frozenset)
    # Phase 1 (2026-05-11) -- composer prompt enrichment + sliding window.
    style_descriptor: str = ""
    outline_spine: str = ""
    character_voice_card: str = ""
    # Phase 2A (2026-05-11) -- arc_phase awareness. When non-empty,
    # the per-beat prompt grows by an ARC PHASE block carrying the
    # ARC_PHASE_GUIDANCE one-liner for the current phase so the
    # composer steers by narrative phase, not just mood.
    arc_phase: str = ""


@dataclass(frozen=True)
class LineResult:
    """compose_line return value.

    Phase 0 (2026-05-11): replaced the bare-string return so the
    composer can carry per-line diagnostic flags (phantom names,
    future format-leak counts) back to the orchestrator without
    coupling through globals or mutable side channels.

    Fields:
      text           cleaned dialogue text (post strip_line_formatting)
      compose_flags  tuple of `"kind:detail"` strings, empty when the
                     line had no detections. Currently emitted kinds:
                       "phantom_name:<token>" — Phase 0 gate flagged
                                                a proper noun not in
                                                allowed_roster
    """

    text: str
    compose_flags: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Phase 1 helpers (2026-05-11) -- outline spine + voice card rendering
# ---------------------------------------------------------------------------


def render_outline_spine(outline_or_beats) -> str:
    """Render the outline as a compact one-line-per-beat spine.

    Accepts EITHER:
      - a pydantic Outline (with .beats list) — usual orchestrator path
      - a plain iterable of Beat-like objects or dicts — testable path

    Each beat renders as:
        b002 ALICE (curious): hears unusual signal in lab
    Non-voiced beats (music_open/inter/close, sfx) drop the speaker
    and mood and render compactly:
        b001 [music_open]: cold open

    Used by the per-beat composer prompt so Mistral-Nemo can see the
    arc it is participating in. Stable across all composer calls in
    an episode (the spine doesn't change), so KV-cache reuse hits the
    prefix once it's wired in the loader.

    Phase 1 ships a flat spine (no arc_phase grouping). Phase 2A's
    arc_phase signal can later re-render this with per-phase
    subheadings (see synthesis §6.D).

    Never raises. Returns "" if no beats.
    """
    beats: list = []
    if outline_or_beats is None:
        return ""
    if hasattr(outline_or_beats, "beats"):
        beats = list(getattr(outline_or_beats, "beats") or [])
    else:
        try:
            beats = list(outline_or_beats)
        except TypeError:
            return ""
    if not beats:
        return ""
    lines: list[str] = ["OUTLINE:"]
    for b in beats:
        # Support both pydantic models and dict shapes for testability.
        def _g(key: str, default: str = "") -> str:
            if isinstance(b, dict):
                return str(b.get(key, default) or default)
            return str(getattr(b, key, default) or default)
        beat_id = _g("beat_id")
        speaker = _g("speaker")
        role = _g("speaker_role")
        mood = _g("mood")
        intent = _g("intent")
        if role in ("character", "announcer"):
            mood_blurb = f" ({mood})" if mood else ""
            lines.append(f"  {beat_id} {speaker}{mood_blurb}: {intent}")
        else:
            role_label = f"[{role}]" if role else "[beat]"
            lines.append(f"  {beat_id} {role_label}: {intent}")
    return "\n".join(lines)


def build_voice_card(cast_row) -> str:
    """Render one cast row as a compact voice card for the composer.

    Cast row is a dict (from production_ledger / _otr_casting). Fields
    consumed:
      name                     ALL-CAPS canonical name
      gender                   optional, "male" / "female" / ...
      character_description    optional, freeform trait line

    Returns a string like:
        ALICE (female, weary forensic engineer in her 40s, dry humor)
    Or for the ANNOUNCER stub (no gender/desc populated):
        ANNOUNCER (omniscient narrator)
    Or for a bare-name row:
        BOB

    Never raises. Returns "" on a row without a name.
    """
    if not cast_row:
        return ""
    if isinstance(cast_row, dict):
        name = str(cast_row.get("name") or "").strip()
        gender = str(cast_row.get("gender") or "").strip()
        desc = str(cast_row.get("character_description") or "").strip()
    else:
        # Best-effort attribute access for non-dict shapes (e.g.
        # CharacterEntry from _otr_cast_contract).
        name = str(getattr(cast_row, "name", "") or "").strip()
        gender = str(getattr(cast_row, "gender", "") or "").strip()
        desc = str(getattr(cast_row, "character_description", "") or "").strip()
    if not name:
        return ""
    if name == "ANNOUNCER" and not desc:
        return "ANNOUNCER (omniscient narrator)"
    bits: list[str] = []
    if gender:
        bits.append(gender)
    if desc:
        bits.append(desc)
    if bits:
        return f"{name} ({', '.join(bits)})"
    return name


# ---------------------------------------------------------------------------
# Error class
# ---------------------------------------------------------------------------


class LineCompositionFailedError(RuntimeError):
    """Raised after compose_line exhausts all retry attempts.

    Attributes:
        attempts: list of (raw_response, failure_reason) tuples
        request:  the LineRequest that was being processed
    """

    def __init__(
        self,
        attempts: list[tuple[str, str]],
        request: LineRequest,
    ) -> None:
        self.attempts = attempts
        self.request = request
        last = attempts[-1][1] if attempts else "no attempts"
        super().__init__(
            f"Line composition failed after {len(attempts)} attempts. "
            f"Last failure: {last}"
        )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You write a single line of dialogue for an audio drama.

Output ONLY the line the character speaks. Do not include the character name. Do not include stage directions. Do not wrap the line in quotes. No prefix, no suffix, no formatting markup.

Match the requested word count approximately. Match the requested mood. Speak in the voice of the character given the recent dialogue context and the episode setting.

If you have nothing the character should say, output one short natural-sounding line that fits the moment. Never refuse, never explain, never apologize, never output meta commentary. Just the spoken line.
"""


def _format_last_lines(last_lines: list[tuple[str, str]]) -> str:
    if not last_lines:
        return "(no prior dialogue -- this is the first line of the episode)"
    rows = [f"[{spk}]: {txt}" for spk, txt in last_lines]
    return "\n".join(rows)


def _build_user_prompt(req: LineRequest) -> str:
    """Render the per-beat user prompt for the composer.

    Phase 1 (2026-05-11): static-first layout. Blocks ordered so a
    future KV-cache reuse in the loader covers everything up to the
    CHARACTER block. The static prefix is shared across every
    composer call in an episode; the variable suffix (CHARACTER /
    RECENT DIALOGUE / WRITE LINE) changes per call.

    Optional blocks (STYLE, OUTLINE, ALLOWED NAMES, CHARACTER) are
    dropped entirely when their LineRequest field is empty so early-
    stage callers and unit tests that haven't populated them don't
    see "STYLE: " with an empty value. The roster block fires only
    when allowed_roster is non-empty (the gate already skips itself
    in that case).
    """
    parts: list[str] = []

    # ----- STATIC (all-episode-stable) -----
    if req.style_descriptor:
        parts.append(f"STYLE: {req.style_descriptor}")
        parts.append("")
    parts.append("EPISODE CONTEXT")
    parts.append(req.canon_header)
    if req.outline_spine:
        parts.append("")
        parts.append(req.outline_spine)
    if req.allowed_roster:
        # Sort for stability across calls (frozenset iteration order
        # is implementation-defined; we want the cached prefix to be
        # byte-identical run to run).
        names_sorted = ", ".join(sorted(req.allowed_roster))
        parts.append("")
        parts.append(
            "ALLOWED NAMES (do not invent any name outside this list; "
            "characters outside the cast or news-relevant terms will "
            "be flagged): " + names_sorted
        )

    # ----- VARIABLE (per-call) -----
    if req.character_voice_card:
        parts.append("")
        parts.append(f"CHARACTER: {req.character_voice_card}")
    # Phase 2A (2026-05-11): arc_phase awareness. Lazy import of
    # ARC_PHASE_GUIDANCE so this module stays importable without the
    # episode-budget module being present (back-compat with tests).
    if req.arc_phase:
        guidance = ""
        try:
            from . import _otr_episode_budget as _OTRB  # type: ignore
            guidance = _OTRB.ARC_PHASE_GUIDANCE.get(req.arc_phase, "")
        except Exception:  # noqa: BLE001
            guidance = ""
        parts.append("")
        if guidance:
            parts.append(f"ARC PHASE: {req.arc_phase}")
            parts.append(f"  {guidance}")
        else:
            parts.append(f"ARC PHASE: {req.arc_phase}")
    parts.append("")
    parts.append("RECENT DIALOGUE (most recent at bottom):")
    parts.append(_format_last_lines(req.last_lines))
    parts.append("")
    parts.append("WRITE LINE")
    parts.append(f"  Speaker: {req.speaker}")
    parts.append(f"  This line accomplishes: {req.intent}")
    parts.append(f"  Mood: {req.mood}")
    parts.append(f"  Target word count: ~{req.target_words}")
    parts.append("")
    parts.append("Write the line. Output only the spoken text.")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# compose_line -- main entrypoint
# ---------------------------------------------------------------------------


def compose_line(
    generate_fn,                # same GenerateFn contract as _otr_outline
    req: LineRequest,
    *,
    max_attempts: int = 2,
    base_temperature: float = _BASE_TEMPERATURE,
) -> LineResult:
    """Compose one cleaned dialogue line for a beat.

    Retry strategy (UNCHANGED in Phase 0 -- name-violation does NOT
    trigger reroll, per §6.A):
      Attempt 1: temperature = base_temperature (0.8).
      Attempt 2: temperature = base_temperature + 0.1 (0.9).

    Failure conditions that trigger retry:
      - generate_fn raises.
      - cleaned response is empty.
      - cleaned response is more than _MAX_OVERSIZE_RATIO * target_words long.

    Phase 0 (2026-05-11): on success, before returning, run
    `detect_phantom_names` against `req.allowed_roster`. Any matched
    phantom is recorded on the returned LineResult.compose_flags as
    `"phantom_name:<token>"`. The line is committed unchanged; Phase 3
    auditor + deterministic Step 2.5 fallback own repair.

    Return:
      LineResult(text=<cleaned dialogue>, compose_flags=<tuple of flags>)

    Raises LineCompositionFailedError after all attempts exhausted.
    """
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")
    if not callable(generate_fn):
        raise ValueError("generate_fn must be callable")

    system = _SYSTEM_PROMPT
    user = _build_user_prompt(req)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    attempts: list[tuple[str, str]] = []
    word_cap = max(15, int(req.target_words * _MAX_OVERSIZE_RATIO))

    for attempt_idx in range(max_attempts):
        temp = base_temperature + (0.1 * attempt_idx)
        log.info(
            "[OTR_LineComposer] attempt %d/%d for %s (temp=%.2f, target=%d words)",
            attempt_idx + 1, max_attempts, req.speaker, temp, req.target_words,
        )

        try:
            raw = generate_fn(
                messages,
                temperature=temp,
                max_new_tokens=_MAX_NEW_TOKENS_PER_LINE,
            )
        except Exception as exc:  # noqa: BLE001
            err_msg = f"generate_fn raised: {type(exc).__name__}: {exc}"
            log.warning("[OTR_LineComposer] %s", err_msg)
            attempts.append(("", err_msg))
            continue

        cleaned = strip_line_formatting(raw or "")

        if not cleaned:
            err_msg = "empty after format-strip"
            log.warning("[OTR_LineComposer] attempt %d failed: %s (raw=%r)",
                        attempt_idx + 1, err_msg, raw)
            attempts.append((raw or "", err_msg))
            continue

        word_count = len(cleaned.split())
        if word_count > word_cap:
            err_msg = f"oversize: {word_count} words > cap {word_cap}"
            log.warning("[OTR_LineComposer] attempt %d failed: %s",
                        attempt_idx + 1, err_msg)
            attempts.append((raw or "", err_msg))
            continue

        # Phase 0 name-roster gate. Detect-and-flag only -- the line
        # commits regardless. Empty roster skips the gate entirely so
        # early-stage callers / unit tests that don't populate it pay
        # zero cost.
        compose_flags: tuple[str, ...] = ()
        if req.allowed_roster:
            phantoms = detect_phantom_names(
                cleaned, req.speaker, req.allowed_roster,
            )
            if phantoms:
                compose_flags = tuple(
                    f"phantom_name:{p}" for p in phantoms
                )
                log.warning(
                    "[OTR_LineComposer] %d phantom name(s) on %s line: %s",
                    len(phantoms), req.speaker, phantoms,
                )

        log.info(
            "[OTR_LineComposer] success on attempt %d/%d: %d words for %s "
            "(flags=%d)",
            attempt_idx + 1, max_attempts, word_count, req.speaker,
            len(compose_flags),
        )
        return LineResult(text=cleaned, compose_flags=compose_flags)

    raise LineCompositionFailedError(attempts=attempts, request=req)


# ---------------------------------------------------------------------------
# Self-test (run as `python nodes/_otr_line_composer.py`)
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("=== _otr_line_composer.py self-test ===")

    # Test 1: strip_line_formatting handles each formatting type.
    print("\n[Test 1] strip_line_formatting")
    cases = [
        ("Hello there.", "Hello there."),
        ('"Hello there."', "Hello there."),
        ("'Hello there.'", "Hello there."),
        ("“Hello there.”", "Hello there."),
        ("ALICE: Hello there.", "Hello there."),
        ("ALICE - Hello there.", "Hello there."),
        ("[ALICE] Hello there.", "Hello there."),
        ("[VOICE: ALICE] Hello there.", "Hello there."),
        ("[ALICE, female, 30s, calm] Hello there.", "Hello there."),
        ("**Hello there.**", "Hello there."),
        ("*Hello there.*", "Hello there."),
        ("ALICE: *Hello there.*", "Hello there."),
        ('  "ALICE: Hello there."  ', "Hello there."),
        ("**[ALICE]**", ""),
        ("*[ALICE]*", ""),
        ("**ALICE:**", ""),
        ("**[ALICE] Hello there.**", "Hello there."),
        ("", ""),
        ("   ", ""),
    ]
    for raw, expected in cases:
        got = strip_line_formatting(raw)
        marker = "PASS" if got == expected else "FAIL"
        print(f"  {marker}: {raw!r:50} -> {got!r}")

    # Test 2: _format_last_lines empty + populated.
    print("\n[Test 2] _format_last_lines")
    assert "no prior dialogue" in _format_last_lines([])
    populated = _format_last_lines([("ALICE", "Hi."), ("BOB", "Hello.")])
    assert "[ALICE]: Hi." in populated
    assert "[BOB]: Hello." in populated
    print("  PASS")

    # Test 3: _build_user_prompt structure.
    print("\n[Test 3] _build_user_prompt")
    req = LineRequest(
        speaker="ALICE",
        intent="reveal the signal",
        mood="tense",
        target_words=15,
        canon_header="TITLE: x\nSETTING: y\nTIME: z\nPREMISE: w",
        last_lines=[("BOB", "What did you find?")],
    )
    user_prompt = _build_user_prompt(req)
    # Phase 1 (2026-05-11): prompt header changed from "NEXT LINE"
    # to "WRITE LINE" to match the synthesis §6.D layout.
    for required in ("EPISODE CONTEXT", "RECENT DIALOGUE", "WRITE LINE",
                     "Speaker: ALICE", "Mood: tense", "~15"):
        assert required in user_prompt, f"missing {required!r}"
    # Bare-bones request omits STYLE / OUTLINE / ALLOWED NAMES blocks.
    assert "STYLE:" not in user_prompt
    assert "OUTLINE:" not in user_prompt
    assert "ALLOWED NAMES" not in user_prompt
    print("  PASS")

    # Test 4: compose_line happy path with mock generate_fn.
    print("\n[Test 4] compose_line happy path")
    def mock_ok(messages, *, temperature, max_new_tokens):
        return "ALICE: I found something I cannot explain."
    result = compose_line(mock_ok, req)
    assert isinstance(result, LineResult)
    assert result.text == "I found something I cannot explain."
    assert result.compose_flags == ()
    print(f"  PASS (cleaned: {result.text!r})")

    # Test 5: compose_line retries on empty.
    print("\n[Test 5] compose_line retries on empty response")
    call_count = {"n": 0}
    def mock_empty_then_ok(messages, *, temperature, max_new_tokens):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return "**[ALICE]**"  # strips to empty
        return "I see it now."
    result = compose_line(mock_empty_then_ok, req)
    assert result.text == "I see it now."
    assert call_count["n"] == 2
    print("  PASS")

    # Test 6: compose_line retries on oversize.
    print("\n[Test 6] compose_line retries on oversize response")
    call_count2 = {"n": 0}
    def mock_oversize_then_ok(messages, *, temperature, max_new_tokens):
        call_count2["n"] += 1
        if call_count2["n"] == 1:
            return " ".join(["word"] * 200)  # way over cap
        return "Short reply."
    result = compose_line(mock_oversize_then_ok, req)
    assert result.text == "Short reply."
    print("  PASS")

    # Test 7: compose_line raises after exhausting attempts.
    print("\n[Test 7] LineCompositionFailedError after exhaustion")
    def mock_always_empty(messages, *, temperature, max_new_tokens):
        return ""
    try:
        compose_line(mock_always_empty, req)
        print("  FAIL: should have raised")
    except LineCompositionFailedError as e:
        assert len(e.attempts) == 2
        assert e.request.speaker == "ALICE"
        assert "2 attempts" in str(e)
        print("  PASS")

    # Test 8: compose_line propagates generate_fn exceptions through retry.
    print("\n[Test 8] generate_fn exceptions are caught and retried")
    call_count3 = {"n": 0}
    def mock_raise_then_ok(messages, *, temperature, max_new_tokens):
        call_count3["n"] += 1
        if call_count3["n"] == 1:
            raise RuntimeError("simulated CUDA OOM")
        return "Recovered line."
    result = compose_line(mock_raise_then_ok, req)
    assert result.text == "Recovered line."
    print("  PASS")

    # Test 9 (Phase 0): build_allowed_roster + detect_phantom_names.
    print("\n[Test 9] Phase 0 roster + phantom detection")
    roster = build_allowed_roster(
        cast_rows=[{"name": "ALICE"}, {"name": "BOB"}],
        key_terms=("CERN", "Voyager"),
    )
    assert "ALICE" in roster
    assert "BOB" in roster
    assert "ANNOUNCER" in roster
    assert "CERN" in roster
    assert "VOYAGER" in roster
    # ALICE's own line never flags herself.
    assert detect_phantom_names("Alice waits.", "ALICE", roster) == []
    # CERN is in roster.
    assert detect_phantom_names("The CERN team is ready.", "ALICE", roster) == []
    # Dr. Patel is a phantom.
    flagged = detect_phantom_names(
        "Dr. Patel can confirm the readings.", "ALICE", roster,
    )
    assert flagged == ["Dr. Patel"], f"expected ['Dr. Patel'], got {flagged!r}"
    # CARLA is a phantom (uppercase, not in roster).
    assert detect_phantom_names(
        "CARLA knows the truth.", "ALICE", roster,
    ) == ["CARLA"]
    # "The radio crackles." -- "The radio" at sentence start is not a phantom.
    assert detect_phantom_names(
        "The radio crackles.", "ALICE", roster,
    ) == []
    print("  PASS")

    # Test 10 (Phase 0): compose_line stamps flags on LineResult.
    print("\n[Test 10] compose_line stamps compose_flags for phantoms")
    req_with_roster = LineRequest(
        speaker="ALICE", intent="reveal", mood="tense", target_words=15,
        canon_header="TITLE: x\nSETTING: y\nTIME: z\nPREMISE: w",
        last_lines=[],
        allowed_roster=roster,
    )
    def mock_phantom(messages, *, temperature, max_new_tokens):
        return "Dr. Patel insists this is real."
    res = compose_line(mock_phantom, req_with_roster)
    assert res.compose_flags == ("phantom_name:Dr. Patel",), \
        f"expected 1 phantom flag, got {res.compose_flags!r}"
    print("  PASS")

    # Test 11 (Phase 0): aggregate_compose_flags counts kinds.
    print("\n[Test 11] aggregate_compose_flags rolls up flag kinds")
    fake_ledger = {
        "lines": [
            {"line_id": "b001", "compose_flags": ["phantom_name:Dr. Patel"]},
            {"line_id": "b002", "compose_flags": ["phantom_name:CARLA",
                                                   "phantom_name:Dr. Patel"]},
            {"line_id": "b003", "compose_flags": []},
            {"line_id": "b004"},  # missing field entirely
        ]
    }
    summary = aggregate_compose_flags(fake_ledger)
    assert summary == {"phantom_name": 3}, f"got {summary!r}"
    assert aggregate_compose_flags({}) == {}
    assert aggregate_compose_flags({"lines": []}) == {}
    print("  PASS")

    print("\n=== Task 3 + Phase 0 self-tests passed ===")
