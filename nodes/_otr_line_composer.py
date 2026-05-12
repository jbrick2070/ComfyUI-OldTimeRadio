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
    # Phase 4 v4 (2026-05-11)
    "render_current_beat",
    "needs_polish",
    "polish_line",
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
    # Phase 4 v4 (2026-05-11) -- prompt revision pass. All defaults
    # are empty so every existing test / caller keeps working; each
    # block in `_build_user_prompt` gates on the corresponding field
    # being non-empty.
    #
    #   allowed_people / allowed_things  Split roster for prompt
    #       rendering. Cast names ("ALICE") and journalistic terms
    #       ("CERN") render in distinct buckets. `allowed_roster`
    #       remains the union and stays the input to the phantom gate;
    #       these two fields are render-only. When both are empty the
    #       composer falls back to the legacy combined ALLOWED NAMES
    #       block driven by `allowed_roster`.
    #   prev_speaker  Name of the character who spoke the immediately
    #       preceding line. Renders in the WRITE LINE role-induction
    #       block as "You are responding to <name>." Empty drops that
    #       sentence (first line of a scene, post-music marker).
    #   current_beat_block  Pre-rendered CURRENT BEAT block (one
    #       outline-spine row for the beat we are writing now). The
    #       writer computes this once per beat via
    #       `render_current_beat(outline, beat.beat_id)`. Keeping the
    #       outline_spine itself plain (no arrow) lets the static
    #       prefix stay byte-stable across every call in an episode
    #       so a future KV-cache reuse pass lands without re-encoding
    #       the spine.
    #   theme  One-sentence theme from `meta.news.script_brief`
    #       (Commit 2 in the v4 plan). Optional flavor, not the
    #       structural-direction outline.
    #   all_voice_cards  Newline-joined voice cards for the whole
    #       cast (Commit 2). When set, replaces single-speaker
    #       CHARACTER block with CAST. Falls back to
    #       `character_voice_card` when empty.
    #   sfx_cue  `beat.sfx_cue` for this beat (Commit 2). Renders as
    #       SOUND IN THE ROOM in the per-beat tail.
    #   position  "<phase>, beat N of M. Next phase: <next>." string
    #       (Commit 4). Replaces the generic per-phase ARC_PHASE_GUIDANCE
    #       one-liner with a position-specific directive. Falls back
    #       to the legacy ARC PHASE block driven by `arc_phase` when
    #       empty.
    allowed_people: frozenset[str] = field(default_factory=frozenset)
    allowed_things: frozenset[str] = field(default_factory=frozenset)
    prev_speaker: str = ""
    current_beat_block: str = ""
    theme: str = ""
    all_voice_cards: str = ""
    sfx_cue: str = ""
    position: str = ""


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


def render_current_beat(outline_or_beats, current_beat_id: str) -> str:
    """Render ONE row from the outline (the beat we are writing now).

    Used by `_build_user_prompt` to emit a CURRENT BEAT block in the
    per-call tail of the prompt without modifying the outline-spine
    string (which lives in the static prefix and must stay byte-stable
    across every composer call in the episode for KV-cache reuse to
    land).

    Returns:
      "CURRENT BEAT\n  bNNN SPEAKER (mood): intent"
        for character / announcer beats
      "CURRENT BEAT\n  bNNN [role]: intent"
        for music / sfx beats
      "" when:
        - outline_or_beats is None / empty
        - current_beat_id is empty or does not match any row

    Never raises.
    """
    if not current_beat_id:
        return ""
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
    target = str(current_beat_id).strip()
    for b in beats:
        if isinstance(b, dict):
            beat_id = str(b.get("beat_id", "") or "")
            speaker = str(b.get("speaker", "") or "")
            role = str(b.get("speaker_role", "") or "")
            mood = str(b.get("mood", "") or "")
            intent = str(b.get("intent", "") or "")
        else:
            beat_id = str(getattr(b, "beat_id", "") or "")
            speaker = str(getattr(b, "speaker", "") or "")
            role = str(getattr(b, "speaker_role", "") or "")
            mood = str(getattr(b, "mood", "") or "")
            intent = str(getattr(b, "intent", "") or "")
        if beat_id != target:
            continue
        if role in ("character", "announcer"):
            mood_blurb = f" ({mood})" if mood else ""
            return f"CURRENT BEAT\n  {beat_id} {speaker}{mood_blurb}: {intent}"
        role_label = f"[{role}]" if role else "[beat]"
        return f"CURRENT BEAT\n  {beat_id} {role_label}: {intent}"
    return ""


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
You write one spoken line for a character in a radio drama.

OUTPUT FORMAT - strict:
- Only the words the character speaks out loud.
- No character name, no colon, no quotation marks.
- No stage directions. No actions in parentheses or brackets.
- No "he said" / "she added" / narration of any kind.
- Output the single line and stop. Nothing before it, nothing after.

CRAFT:
- Imply more than you state. People rarely say what they mean.
- Push the scene forward by one small step.
- Follow naturally from the last thing said.
- Stay in the speaker's voice - their job, their pressure, their habits.
- Inhabit the mood without naming it.
- Use only proper nouns listed under NAMED ENTITIES. Generic roles
  ("the tech", "the lab", "mission control") are fine.

Short and charged beats long and explanatory. Within plus or minus
30% of the requested word count.
"""


def _format_last_lines(last_lines: list[tuple[str, str]]) -> str:
    """Render the rolling-window of recent dialogue.

    Phase 4 v4 (2026-05-11): label moved from "RECENT DIALOGUE" to
    "LAST SPOKEN (this scene)" at the call site. Empty window now
    emits a more descriptive placeholder so the model knows whether
    it is writing the first spoken line of a scene vs the first line
    of the whole episode (Commit 3 cleared the window at music
    markers so this state is reachable mid-episode too).
    """
    if not last_lines:
        return "(scene just opened - no one has spoken yet)"
    rows = [f"[{spk}]: {txt}" for spk, txt in last_lines]
    return "\n".join(rows)


def _build_user_prompt(req: LineRequest) -> str:
    """Render the per-beat user prompt for the composer.

    Phase 4 v4 (2026-05-11): block order tightened for future KV-cache
    reuse. Every block that stays byte-identical across all composer
    calls in an episode lives in the STATIC PREFIX:

        STYLE
        THEME                (Commit 2: meta.news first-sentence theme)
        EPISODE CONTEXT      (canon_header)
        NAMED ENTITIES       (people + things, sorted)
        CAST                 (full voice cards, all characters)
        OUTLINE              (full spine, plain - no per-call arrow)

    Blocks that change per call live in the PER-BEAT TAIL:

        CURRENT BEAT         (single spine row for the beat we write)
        POSITION             (Commit 4: phase, beat N of M, next phase)
        SOUND IN THE ROOM    (Commit 2: beat.sfx_cue)
        LAST SPOKEN          (last_lines rolling window; scene-local
                              via Commit 3)
        WRITE LINE           (role induction + beat + mood + word count
                              + "Speak now.")

    Optional blocks are dropped entirely when their LineRequest field
    is empty so early-stage callers and unit tests that haven't
    populated them keep working. NAMED ENTITIES fires only when
    allowed_people OR allowed_things is non-empty; back-compat callers
    that only set the legacy `allowed_roster` get an ALLOWED NAMES
    block in the same slot.

    The role-induction sentence "You are <SPEAKER>." (plus optional
    "You are responding to <PREV_SPEAKER>.") sits immediately above
    the generation target. Small instruct-tuned LLMs in the 7B-14B
    class hold a per-call role much more reliably when the directive
    is one block above the response slot vs upstream in the system
    prompt.
    """
    parts: list[str] = []

    # ===== STATIC PREFIX (byte-stable across an episode) =====

    if req.style_descriptor:
        parts.append(f"STYLE: {req.style_descriptor}")
        parts.append("")

    # THEME emits when the writer threads a non-empty theme via
    # `LineRequest.theme` (Commit 2 in the v4 plan).
    if req.theme:
        parts.append(f"THEME: {req.theme}")
        parts.append("")

    parts.append("EPISODE CONTEXT")
    parts.append(req.canon_header)

    # NAMED ENTITIES split (Commit 1 in the v4 plan). When the writer
    # populates allowed_people / allowed_things separately, render
    # them in distinct buckets. Otherwise fall back to the legacy
    # combined ALLOWED NAMES block driven by allowed_roster so old
    # callers and unit tests still get the gate signal.
    if req.allowed_people or req.allowed_things:
        parts.append("")
        parts.append("NAMED ENTITIES IN THIS WORLD")
        if req.allowed_people:
            parts.append(
                "  People: " + ", ".join(sorted(req.allowed_people))
            )
        if req.allowed_things:
            parts.append(
                "  Places, agencies, things: "
                + ", ".join(sorted(req.allowed_things))
            )
        parts.append(
            'Generic roles ("the tech", "the lab", "mission control") '
            "are fine. Do not invent any other proper name."
        )
    elif req.allowed_roster:
        # Legacy combined roster path.
        names_sorted = ", ".join(sorted(req.allowed_roster))
        parts.append("")
        parts.append(
            "ALLOWED NAMES (do not invent any name outside this list; "
            "characters outside the cast or news-relevant terms will "
            "be flagged): " + names_sorted
        )

    # CAST replaces single-speaker CHARACTER when all_voice_cards is
    # threaded. Falls back to the speaker-only voice card on legacy
    # callers (Commit 2 wires the full-cast path in the writer).
    if req.all_voice_cards:
        parts.append("")
        parts.append("CAST")
        parts.append(req.all_voice_cards)
    elif req.character_voice_card:
        parts.append("")
        parts.append(f"CHARACTER: {req.character_voice_card}")

    if req.outline_spine:
        parts.append("")
        parts.append(req.outline_spine)

    # ===== PER-BEAT TAIL (changes every call) =====

    # CURRENT BEAT — single spine row for the beat we are writing
    # right now. The outline above stays plain (no arrow) for KV
    # stability; this block names which row we are on. Writer
    # pre-renders the string via `render_current_beat(outline,
    # beat.beat_id)` and threads it on `req.current_beat_block`.
    if req.current_beat_block:
        parts.append("")
        parts.append(req.current_beat_block)

    # POSITION supersedes the old generic ARC PHASE block (Commit 4
    # in the v4 plan). Emits the position string verbatim. Legacy
    # arc_phase-only callers still get a fallback ARC PHASE block so
    # this commit does not regress them in isolation.
    if req.position:
        parts.append("")
        parts.append(f"POSITION: {req.position}")
    elif req.arc_phase:
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

    # SOUND IN THE ROOM — Commit 2 in the v4 plan. Threaded from
    # beat.sfx_cue so the line can react to the sound environment.
    if req.sfx_cue:
        parts.append("")
        parts.append(f"SOUND IN THE ROOM: {req.sfx_cue}")

    parts.append("")
    parts.append("LAST SPOKEN (this scene):")
    parts.append(_format_last_lines(req.last_lines))

    parts.append("")
    parts.append("WRITE LINE")
    # Role induction one block above the generation target. Empty
    # prev_speaker drops the "responding to" clause cleanly (first
    # line of a scene / post-music marker / first line of episode).
    if req.prev_speaker and req.prev_speaker.strip().upper() != req.speaker.strip().upper():
        parts.append(
            f"You are {req.speaker}. You are responding to {req.prev_speaker}."
        )
    else:
        parts.append(f"You are {req.speaker}.")
    parts.append(f"Mood: {req.mood}.")
    parts.append(f"Beat: {req.intent}.")
    parts.append(f"Word count target: {req.target_words}.")
    parts.append("Speak now.")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# compose_line -- main entrypoint
# ---------------------------------------------------------------------------


_DEFAULT_STOP_STRINGS: tuple[str, ...] = ("\n\n", "\n[", "\n(")
"""Default stop substrings for compose_line + polish_line. `\n\n`
catches "line + stage direction on next paragraph"; `\n[` / `\n(`
catch leaked bracketed/parenthesized directions on a new line.
Do not stop on a bare `\n` -- some legitimate lines have a soft
break. Forwarded through generate_fn's stop= kwarg; loader falls
back to a substring-matching StoppingCriteria when the underlying
generate call doesn't natively support stop strings."""


# ---------------------------------------------------------------------------
# Phase 4 v4 (2026-05-11) — optional polish pass (regex-gated)
# ---------------------------------------------------------------------------
#
# After the composer's retry ladder closes, optionally check the
# generated line against a small narration-leak regex set. If the
# line trips any pattern, fire ONE targeted polish LLM call with a
# tight cleanup prompt and replace the line. Default OFF — keeps the
# composer hot-path at 1 call per voiced beat. Opt-in via the
# `enable_polish_pass` widget on OTR_LedgerScriptWriter.
#
# Cost when on: typically +1-2 calls per 15-line episode (~30s),
# NOT +15 calls (~3-5 min) — the regex gate filters down to lines
# that actually leaked. Targeted Script Doctor at the end of the
# writer still catches anything the polish pass misses.

_NARRATION_LEAK_PATTERNS: tuple[str, ...] = (
    # "he said" / "she replied" / "they whispered" — narration verbs
    # attached to a pronoun, mid-sentence or end-of-sentence.
    r"\b(?:he|she|they)\s+(?:said|replied|added|asked|whispered|"
    r"shouted|paused|continued|murmured|exclaimed)\b",
    # Opens with a quote mark (smart or straight).
    r'^["“‘]',
    # Markdown / asterisk wrapped action ("*sighs*").
    r"\*[^*]+\*",
    # Bracket stage direction ("[pauses]" / "[looks away]").
    r"\[[^\]]+\]",
    # Parenthesized cue verb ("(sighs)", "(pause)", "(laughs)").
    r"\([^)]*(?:sigh|pause|beat|laughs?|smiles?|gestures?|nods?|"
    r"shrugs?|cough)[^)]*\)",
)

_NARRATION_LEAK_REGEXES: tuple = tuple(
    re.compile(p, re.IGNORECASE) for p in _NARRATION_LEAK_PATTERNS
)


def needs_polish(line: str) -> bool:
    """Return True if `line` matches any narration-leak pattern.

    Cheap regex check (no LLM call). Used by `compose_line` to gate
    the optional polish pass. Empty / falsy input returns False.
    Never raises.
    """
    if not line:
        return False
    return any(rx.search(line) for rx in _NARRATION_LEAK_REGEXES)


_POLISH_SYSTEM_PROMPT = """\
You are a script editor cleaning one line of radio drama dialogue.
The line below leaked narration or stage direction. Rewrite it as
pure spoken dialogue.

OUTPUT RULES - strict:
- Only the words the character speaks out loud.
- No name, no colon, no quotes, no brackets, no parentheses.
- No "he said" / "she replied" / narration of any kind.
- Preserve the character's intent. Preserve the speaker's voice.
- Keep within plus or minus 20% of the original word count.

Output the cleaned line and stop. Nothing else.
"""


_POLISH_BASE_TEMPERATURE = 0.4
_POLISH_MAX_TOKENS_MULTIPLIER = 3  # ~3 tokens/word target ceiling


def polish_line(
    generate_fn,
    leaked_line: str,
    speaker_voice_card: str,
    *,
    temperature: float = _POLISH_BASE_TEMPERATURE,
    stop_strings: tuple[str, ...] = _DEFAULT_STOP_STRINGS,
) -> str:
    """Run ONE polish LLM call against `leaked_line`.

    Targeted edit (low temperature). Returns the cleaned line on
    success. On any failure (generate raise, empty result, polish
    still trips the regex), returns the original `leaked_line`
    unchanged — polish here is a quality nicety, not a correctness
    requirement.
    """
    if not (leaked_line or "").strip():
        return leaked_line
    user = (
        f"CHARACTER: {speaker_voice_card or 'unspecified speaker'}\n"
        f"ORIGINAL LINE: {leaked_line}\n"
    )
    messages = [
        {"role": "system", "content": _POLISH_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    orig_word_count = max(4, len(leaked_line.split()))
    mnt = max(40, orig_word_count * _POLISH_MAX_TOKENS_MULTIPLIER)
    try:
        try:
            raw = generate_fn(
                messages,
                temperature=temperature,
                max_new_tokens=mnt,
                stop=list(stop_strings) if stop_strings else None,
            )
        except TypeError:
            raw = generate_fn(
                messages, temperature=temperature, max_new_tokens=mnt,
            )
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "[OTR_LineComposer] polish_line generate_fn raised: %s",
            exc,
        )
        return leaked_line
    cleaned = strip_line_formatting(raw or "")
    if not cleaned:
        log.debug("[OTR_LineComposer] polish_line produced empty output")
        return leaked_line
    return cleaned


def compose_line(
    generate_fn,                # same GenerateFn contract as _otr_outline
    req: LineRequest,
    *,
    max_attempts: int = 2,
    base_temperature: float = _BASE_TEMPERATURE,
    max_new_tokens_cap: int = _MAX_NEW_TOKENS_PER_LINE,
    stop_strings: tuple[str, ...] = _DEFAULT_STOP_STRINGS,
    enable_polish_pass: bool = False,
) -> LineResult:
    """Compose one cleaned dialogue line for a beat.

    Retry strategy (UNCHANGED in Phase 0 -- name-violation does NOT
    trigger reroll, per §6.A):
      Attempt 1: temperature = base_temperature (0.8).
      Attempt 2: temperature = base_temperature + 0.1 (0.9).

    Phase 4 v4 (2026-05-11): max_new_tokens scales with target_words
    on attempt 1 (`min(max_new_tokens_cap, target_words * 4)`) so a
    short line does not get a profligate token budget that invites
    drift. Attempt 2 uses the full cap as retry headroom.

    Stop strings are passed through to generate_fn via the optional
    `stop=` kwarg. Loaders that don't accept `stop=` swallow the
    kwarg silently (the writer's _build_truncating_generate_fn does).

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
    # Attempt-1 max_new_tokens scaled to target line length; attempt 2
    # uses the full cap. ~4 tokens per English word is the textbook
    # transformers heuristic.
    attempt_tokens = (
        min(int(max_new_tokens_cap), max(40, int(req.target_words) * 4)),
        int(max_new_tokens_cap),
    )

    for attempt_idx in range(max_attempts):
        temp = base_temperature + (0.1 * attempt_idx)
        # Pick from attempt_tokens by index, falling back to the cap
        # on any extra attempt past the table.
        if attempt_idx < len(attempt_tokens):
            mnt = attempt_tokens[attempt_idx]
        else:
            mnt = int(max_new_tokens_cap)
        log.info(
            "[OTR_LineComposer] attempt %d/%d for %s "
            "(temp=%.2f, max_new_tokens=%d, target=%d words)",
            attempt_idx + 1, max_attempts, req.speaker, temp, mnt,
            req.target_words,
        )

        try:
            # Try with stop= first; older generate_fn signatures
            # without the kwarg fall back to the no-stop path.
            try:
                raw = generate_fn(
                    messages,
                    temperature=temp,
                    max_new_tokens=mnt,
                    stop=list(stop_strings) if stop_strings else None,
                )
            except TypeError:
                raw = generate_fn(
                    messages,
                    temperature=temp,
                    max_new_tokens=mnt,
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

        # Phase 4 v4 (2026-05-11): optional polish pass. Regex-gated
        # so polish only fires on lines that actually leaked narration
        # / stage direction. Default OFF; per-episode opt-in via the
        # `enable_polish_pass` widget on OTR_LedgerScriptWriter.
        if enable_polish_pass and needs_polish(cleaned):
            log.info(
                "[OTR_LineComposer] polish_line firing on %s "
                "(narration-leak detected)",
                req.speaker,
            )
            polished = polish_line(
                generate_fn,
                cleaned,
                req.character_voice_card,
                stop_strings=stop_strings,
            )
            # Re-strip in case the polish prompt produced a fresh
            # speaker tag at the head (defensive — polish's prompt
            # forbids it but small models occasionally slip).
            polished_clean = strip_line_formatting(polished or "")
            if polished_clean:
                cleaned = polished_clean
                # Update word_count for the success log below.
                word_count = len(cleaned.split())

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
    # v4: placeholder phrasing updated to "scene just opened".
    assert "scene just opened" in _format_last_lines([])
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
    # v4 (2026-05-11): block labels updated for the prompt-revision pass.
    for required in ("EPISODE CONTEXT", "LAST SPOKEN (this scene):",
                     "WRITE LINE", "You are ALICE.", "Mood: tense.",
                     "15", "Speak now."):
        assert required in user_prompt, f"missing {required!r}"
    # Bare-bones request omits STYLE / THEME / OUTLINE / NAMED ENTITIES
    # / CAST / CURRENT BEAT / POSITION / SOUND IN THE ROOM blocks.
    for missing in ("STYLE:", "THEME:", "OUTLINE:", "NAMED ENTITIES",
                    "ALLOWED NAMES", "CAST", "CURRENT BEAT", "POSITION:",
                    "SOUND IN THE ROOM"):
        assert missing not in user_prompt, f"unexpected {missing!r}"
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
