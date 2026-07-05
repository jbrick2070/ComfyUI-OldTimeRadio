"""Sprint 10A step 5 -- Stage 3 per-line validators.

Pure-Python validators that run AFTER a dialogue line lands. Each
returns ValidationFinding | None. validate_line() runs all six on a
single line; validate_episode() runs all six across every line in a
plan + rendered ledger.

Per docs/story-generator-final-plan.md Stage 3:
  1. length            -- word count in [target * 0.5, target * 1.7]
  2. pronoun_consistency -- speaker references match Stage 1 pronouns
  3. speaker_leak      -- no stage directions / narrator voice
  4. banned_phrases    -- no purple-prose markers
  5. continuity        -- line does not contradict running_facts
  6. on_beat           -- line at least loosely matches beat.intent

The validators are MEASUREMENT only for step 5. Integration into the
writer pipeline (or critic call) happens later -- step 5's job is
to land the surface + prove each validator catches its target
failure mode on historical bad-line fixtures.

Module is PURE: no LLM, no I/O, no transformers imports.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from ._otr_stage1_plan import Stage1Beat, Stage1CastMember, Stage1Plan


# ---------------------------------------------------------------------------
# Constants -- finding severity + codes
# ---------------------------------------------------------------------------

_SEVERITY_ERROR: str = "error"
_SEVERITY_WARN: str = "warn"

CODE_LENGTH_DRIFT: str = "length_drift"
CODE_PRONOUN_MISMATCH: str = "pronoun_mismatch"
CODE_SPEAKER_LEAK: str = "speaker_leak"
CODE_BANNED_PHRASE: str = "banned_phrase"
CODE_CONTINUITY_BREAK: str = "continuity_break"
CODE_OFF_BEAT: str = "off_beat"
# BUG-LOCAL-288 (2026-05-27): SFW post-validator. PD4 says "Safe
# for work. No profanity. Non-violent. Good narrative arc." The
# writer occasionally emits mild profanity ("Damn it",
# "What the hell") that BUG_BIBLE-261 already caught at the
# whole-episode level but that slipped through compose_line. This
# code lets the Stage 3 line-level gate flag the same patterns
# so the operator sees the violation before Bark ever renders.
CODE_SFW_VIOLATION: str = "sfw_violation"


# ---------------------------------------------------------------------------
# Length validator bounds
# ---------------------------------------------------------------------------

# Per the final plan: word count in [target * 0.5, target * 1.7].
# The 0.5 / 1.7 multipliers are deliberately asymmetric -- a too-short
# line is a bigger quality hit than a slightly-long line.
_LENGTH_LOWER_MULTIPLIER: float = 0.5
_LENGTH_UPPER_MULTIPLIER: float = 1.7


# ---------------------------------------------------------------------------
# Speaker-leak regexes
# ---------------------------------------------------------------------------

# Each pattern matches a known leak shape. The list is ordered by
# specificity so the first match wins on overlap.

_SPEAKER_LEAK_PATTERNS: List[tuple[str, re.Pattern]] = [
    # Narrator-voice: "X said", "X replied", "X muttered"
    ("narrator_attribution", re.compile(
        r"\b(?:he|she|they|the announcer)\s+(?:said|replied|muttered|whispered|asked|answered|shouted|exclaimed)\b",
        re.IGNORECASE,
    )),
    # Self-naming meta-leak: "As <NAME>, I would..."
    # Case-insensitive on the leading 'as' so sentence-initial 'As'
    # also matches. The NAME tokens stay uppercase-only since cast
    # names are ALL CAPS by OTR convention.
    ("self_naming", re.compile(
        r"\bas\s+[A-Z][A-Z\s]+,\s+I\b",
        re.IGNORECASE,
    )),
    # Bracketed stage direction: [walks across]
    ("bracketed_direction", re.compile(r"\[[^\]]{2,80}\]")),
    # Parenthesized cue: (sighs)
    ("parenthesized_cue", re.compile(r"\([^\)]{2,80}\)")),
    # Asterisk action: *sighs* or *walks across*
    ("asterisk_action", re.compile(r"\*[^*]{2,80}\*")),
]


# ---------------------------------------------------------------------------
# Banned-phrase initial list
# ---------------------------------------------------------------------------

# Sourced from the operator's documented purple-prose findings on
# shipped episodes. The list is configurable per-call -- this module
# constant is just the default starting point that consumers can
# extend.

DEFAULT_BANNED_PHRASES: List[str] = [
    # Operator-flagged from signal_lost_bioluminescent_trench_descent
    "forfeit to the void",
    "expected toll",
    # Documented purple-prose markers
    "the void",
    "dance with destiny",
    "ancient secrets",
    "darkness incarnate",
    "shadows whisper",
    "destiny calls",
    "the abyss",
    "primordial",
    # Filler intensifiers
    "absolutely",
    "literally",
    "utterly",
]


# ---------------------------------------------------------------------------
# BUG-LOCAL-288: SFW profanity terms (PD4)
# ---------------------------------------------------------------------------
# PD4 ships a hard rule: "Safe for work. No profanity. Non-violent.
# Good narrative arc." validate_sfw catches the family of mild
# profanity Mistral-Nemo and Gemma occasionally emit on dialogue.
# Each entry is matched as a whole word (word-boundary regex) so
# 'hell' fires on "go to hell" but NOT on "hello". Case-insensitive.
# Operator-extensible per-call; this constant is the starting set.
#
# Sourced from live runs (pending_20260527_153942 "Space Force /
# SpaceX targeting" shipped a 'damn it' first-line that violated
# PD4 and still made it through to ledger persistence).

DEFAULT_PROFANITY_TERMS: List[str] = [
    # Mild profanity that the writer occasionally emits on tense
    # beats.
    "damn",
    "damnit",
    "dammit",
    "hell",
    "goddamn",
    "godammit",
    "goddammit",
    "shit",
    "bullshit",
    "bitch",
    "bastard",
    "ass",
    "asshole",
    "fuck",
    "fucking",
    "fucker",
    "piss",
    # Common substitutions
    "screw you",
    "screw this",
    "screw that",
]


# ---------------------------------------------------------------------------
# Pronoun lookups
# ---------------------------------------------------------------------------

# Maps a Stage 1 pronoun choice to the words a SELF-reference would
# use. Used by validate_pronoun_consistency to detect when a line
# references the speaker with the wrong pronoun form (e.g. she/her
# speaker referring to themselves with "he" or "him").

_SELF_PRONOUN_TOKENS: dict[str, set[str]] = {
    "he/him":     {"he", "him", "his", "himself"},
    "she/her":    {"she", "her", "hers", "herself"},
    "they/them":  {"they", "them", "their", "theirs", "themselves", "themself"},
}

# All recognized self-pronoun tokens across all variants -- used to
# detect ANY self-reference for the "wrong pronoun" check.
_ALL_PRONOUN_TOKENS: set[str] = (
    _SELF_PRONOUN_TOKENS["he/him"]
    | _SELF_PRONOUN_TOKENS["she/her"]
    | _SELF_PRONOUN_TOKENS["they/them"]
)


# ---------------------------------------------------------------------------
# Finding + result types
# ---------------------------------------------------------------------------


@dataclass
class ValidationFinding:
    """One Stage 3 validator finding on one line."""

    severity: str        # "error" | "warn"
    code: str            # one of the CODE_* constants
    beat_id: str         # bNNN this line belongs to
    speaker: str         # cast member name for this line
    message: str         # human-readable description
    expected: Optional[str] = None
    got: Optional[str] = None


@dataclass
class ValidationResult:
    """Aggregate result for a set of validator runs."""

    findings: List[ValidationFinding] = field(default_factory=list)

    @property
    def errors(self) -> List[ValidationFinding]:
        return [f for f in self.findings if f.severity == _SEVERITY_ERROR]

    @property
    def warns(self) -> List[ValidationFinding]:
        return [f for f in self.findings if f.severity == _SEVERITY_WARN]

    @property
    def ok(self) -> bool:
        """True iff no error-level findings."""
        return len(self.errors) == 0

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "error_count": len(self.errors),
            "warn_count": len(self.warns),
            "findings": [
                {
                    "severity": f.severity,
                    "code": f.code,
                    "beat_id": f.beat_id,
                    "speaker": f.speaker,
                    "message": f.message,
                    "expected": f.expected,
                    "got": f.got,
                }
                for f in self.findings
            ],
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _word_count(text: str) -> int:
    """Count whitespace-separated tokens, ignoring punctuation-only
    artifacts. Simple enough for length validation."""
    if not isinstance(text, str):
        return 0
    return len([t for t in text.split() if any(c.isalnum() for c in t)])


def _tokenize_words_lower(text: str) -> set[str]:
    """Lowercase set of alphabetic word tokens. Strips punctuation."""
    if not isinstance(text, str):
        return set()
    return {
        "".join(c for c in tok.lower() if c.isalpha())
        for tok in text.split()
    } - {""}


# Common English stopwords -- removed from on-beat overlap so the
# similarity score isn't dominated by 'the' / 'and' / 'a'.
_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the",
    "and", "or", "but", "if", "then", "so", "because",
    "of", "in", "on", "at", "to", "for", "with", "by", "from",
    "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "done",
    "have", "has", "had", "having",
    "i", "you", "he", "she", "it", "we", "they",
    "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their",
    "this", "that", "these", "those",
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
    "not", "no", "yes",
    "as", "than", "just",
    "will", "would", "should", "could", "can", "may", "might", "must",
})


# ---------------------------------------------------------------------------
# Individual validators
# ---------------------------------------------------------------------------


def validate_length(
    beat: Stage1Beat,
    line_text: str,
    speaker_name: str,
) -> Optional[ValidationFinding]:
    """Length validator: word count must lie in
    [target * 0.5, target * 1.7]. Lines outside this band trigger
    a length_drift finding at severity 'error'.
    """
    target = beat.length_target_words
    lower = max(1, int(target * _LENGTH_LOWER_MULTIPLIER))
    upper = max(lower + 1, int(target * _LENGTH_UPPER_MULTIPLIER))
    actual = _word_count(line_text)
    if actual < lower or actual > upper:
        return ValidationFinding(
            severity=_SEVERITY_ERROR,
            code=CODE_LENGTH_DRIFT,
            beat_id=beat.beat_id,
            speaker=speaker_name,
            message=(
                f"Line is {actual} words; beat target is "
                f"{target} (band [{lower}..{upper}])."
            ),
            expected=f"[{lower}..{upper}]",
            got=str(actual),
        )
    return None


def validate_pronoun_consistency(
    beat: Stage1Beat,
    line_text: str,
    speaker: Stage1CastMember,
) -> Optional[ValidationFinding]:
    """Pronoun-consistency validator: if the line uses an
    out-of-palette self-pronoun (e.g. a she/her speaker referring
    to themselves with 'he'), flag a pronoun_mismatch error.

    Conservative implementation:
      * Tokenize the line.
      * If it contains pronoun tokens from a DIFFERENT pronoun set
        than the speaker's, AND no tokens from the speaker's own
        set, flag it. (Mixed pronouns may legitimately reference
        another character; we only flag when the speaker's own
        set is absent AND a wrong set is present.)
    """
    expected_set = _SELF_PRONOUN_TOKENS.get(speaker.pronouns)
    if expected_set is None:
        return None
    tokens = _tokenize_words_lower(line_text)
    other_sets = {
        pronouns: token_set
        for pronouns, token_set in _SELF_PRONOUN_TOKENS.items()
        if pronouns != speaker.pronouns
    }
    has_own = bool(tokens & expected_set)
    wrong_pronoun_found: Optional[str] = None
    for other_pronouns, other_tokens in other_sets.items():
        if tokens & other_tokens:
            wrong_pronoun_found = other_pronouns
            break
    if wrong_pronoun_found and not has_own:
        return ValidationFinding(
            severity=_SEVERITY_ERROR,
            code=CODE_PRONOUN_MISMATCH,
            beat_id=beat.beat_id,
            speaker=speaker.name,
            message=(
                f"Speaker {speaker.name!r} has pronouns "
                f"{speaker.pronouns!r} but the line uses "
                f"{wrong_pronoun_found!r} pronouns with no own-"
                f"pronoun reference."
            ),
            expected=speaker.pronouns,
            got=wrong_pronoun_found,
        )
    return None


def validate_speaker_leak(
    beat: Stage1Beat,
    line_text: str,
    speaker_name: str,
) -> Optional[ValidationFinding]:
    """Speaker-leak validator: detect narrator voice, stage
    directions, bracketed cues, asterisk actions, and self-naming
    meta-leaks. Any match fires a speaker_leak error.
    """
    if not isinstance(line_text, str):
        return None
    for pattern_name, rx in _SPEAKER_LEAK_PATTERNS:
        m = rx.search(line_text)
        if m is not None:
            return ValidationFinding(
                severity=_SEVERITY_ERROR,
                code=CODE_SPEAKER_LEAK,
                beat_id=beat.beat_id,
                speaker=speaker_name,
                message=(
                    f"Speaker-leak ({pattern_name}) detected in line: "
                    f"{m.group(0)!r}"
                ),
                expected="dialogue only",
                got=pattern_name,
            )
    return None


def validate_banned_phrases(
    beat: Stage1Beat,
    line_text: str,
    speaker_name: str,
    banned: "Optional[Sequence[str]]" = None,
) -> Optional[ValidationFinding]:
    """Banned-phrase validator: simple substring check (case-
    insensitive) against the configurable list. First match fires a
    banned_phrase error.

    Stage 4: production supplies ``banned`` from the bank's story_rules
    pack (the writer's stage3_banned_phrases producer); ``None`` = the
    DEFAULT_BANNED_PHRASES extraction fixture (tests / legacy callers).
    Accepts any Sequence (the rules pack hands a tuple).
    """
    if not isinstance(line_text, str):
        return None
    phrases = banned if banned is not None else DEFAULT_BANNED_PHRASES
    low = line_text.lower()
    for phrase in phrases:
        if phrase.lower() in low:
            return ValidationFinding(
                severity=_SEVERITY_ERROR,
                code=CODE_BANNED_PHRASE,
                beat_id=beat.beat_id,
                speaker=speaker_name,
                message=(
                    f"Banned phrase detected: {phrase!r}. Grown "
                    f"from operator-flagged purple-prose findings; "
                    f"see DEFAULT_BANNED_PHRASES in "
                    f"_otr_stage3_validators."
                ),
                expected="no banned phrase",
                got=phrase,
            )
    return None


def validate_sfw(
    beat: Stage1Beat,
    line_text: str,
    speaker_name: str,
    terms: Optional[List[str]] = None,
) -> Optional[ValidationFinding]:
    """BUG-LOCAL-288 (2026-05-27): SFW profanity validator.

    Word-boundary regex check against the configurable profanity
    list. First match fires an `sfw_violation` ERROR so compose_line's
    single-repair branch fires (Wave 1 Agent B contract). The
    distinction from `validate_banned_phrases` is intentional --
    profanity is a HARD PD4 ship-stop ("Safe for work. No
    profanity."), purple-prose is taste. Codes stay separate so the
    operator can filter the meta.stage3_findings_per_beat trail by
    cause without grepping prose.

    Match shape: case-insensitive, whole-word ONLY. "hell" fires on
    "go to hell" but NOT on "hello" / "hellish" / "shellfish".
    Multi-word entries ("screw you") match exact phrases as
    substrings (still bounded -- the writer is unlikely to slip
    "screw you" inside another word).

    Reserved speakers (ANNOUNCER, MUSIC) are subject to the same
    rule -- the announcer's bookends are part of the shippable
    transcript and PD4 applies uniformly.
    """
    if not isinstance(line_text, str):
        return None
    terms = terms if terms is not None else DEFAULT_PROFANITY_TERMS
    low = line_text.lower()
    for term in terms:
        t = (term or "").strip().lower()
        if not t:
            continue
        if " " in t:
            # Multi-word phrase -- substring match.
            if t in low:
                return ValidationFinding(
                    severity=_SEVERITY_ERROR,
                    code=CODE_SFW_VIOLATION,
                    beat_id=beat.beat_id,
                    speaker=speaker_name,
                    message=(
                        f"SFW violation (PD4): profanity phrase "
                        f"{term!r} detected in line. PD4 ships as "
                        f"a hard rule -- rewrite without the term."
                    ),
                    expected="no profanity",
                    got=term,
                )
        else:
            # Single word -- word-boundary regex so 'hell' does
            # NOT match 'hello'.
            pattern = re.compile(
                rf"\b{re.escape(t)}\b",
                re.IGNORECASE,
            )
            if pattern.search(line_text):
                return ValidationFinding(
                    severity=_SEVERITY_ERROR,
                    code=CODE_SFW_VIOLATION,
                    beat_id=beat.beat_id,
                    speaker=speaker_name,
                    message=(
                        f"SFW violation (PD4): profanity word "
                        f"{term!r} detected in line. PD4 ships as "
                        f"a hard rule -- rewrite without the word."
                    ),
                    expected="no profanity",
                    got=term,
                )
    return None


def validate_continuity(
    beat: Stage1Beat,
    line_text: str,
    speaker_name: str,
    running_facts: List[str],
) -> Optional[ValidationFinding]:
    """Continuity validator: detect lines that contradict an
    established running_fact.

    Conservative implementation -- without an NLI model we can't do
    true semantic contradiction. This validator catches a narrow
    family of explicit contradictions:
      * Each running_fact is tokenized into non-stopword content
        words.
      * If the line contains a 'not <content_word>' or '<content_word>
        not' phrasing AND the fact asserts that word positively, we
        flag a soft warn.
      * If the line contains a directly-opposite numeric statement
        (e.g. fact says 'on Mars', line says 'on Venus'), we do NOT
        catch that today -- that's step-7 critic territory.

    Returns at most ONE finding per call (first contradiction wins).
    Severity is WARN -- continuity is hard to nail with substring
    matching, so we lean conservative to keep the false-positive
    rate low.
    """
    if not isinstance(line_text, str) or not running_facts:
        return None
    line_low = line_text.lower()
    for fact in running_facts:
        if not isinstance(fact, str):
            continue
        fact_low = fact.lower()
        # Extract content words from the fact.
        fact_words = {
            "".join(c for c in tok if c.isalpha())
            for tok in fact_low.split()
        } - _STOPWORDS - {""}
        # Look for "not <content_word>" in the line, where
        # content_word also appears in the fact.
        for word in fact_words:
            if len(word) < 4:
                continue  # too short to be reliable
            if word not in line_low:
                continue
            if f"not {word}" in line_low or f"no {word}" in line_low:
                return ValidationFinding(
                    severity=_SEVERITY_WARN,
                    code=CODE_CONTINUITY_BREAK,
                    beat_id=beat.beat_id,
                    speaker=speaker_name,
                    message=(
                        f"Line negates {word!r} but running_fact "
                        f"establishes it: {fact!r}"
                    ),
                    expected=fact,
                    got=line_text[:200],
                )
    return None


def validate_on_beat(
    beat: Stage1Beat,
    line_text: str,
    speaker_name: str,
    min_overlap_ratio: float = 0.10,
) -> Optional[ValidationFinding]:
    """On-beat validator: check that the line's content-word set
    overlaps the beat.intent's content-word set above
    min_overlap_ratio.

    Soft warn only. The threshold is conservative (10%) so a wide
    swing of legitimate dialogue passes; lines that score 0 overlap
    on a substantive beat are the ones we want to surface for
    forensic review.
    """
    if not isinstance(line_text, str):
        return None
    intent_words = _tokenize_words_lower(beat.intent) - _STOPWORDS - {""}
    line_words = _tokenize_words_lower(line_text) - _STOPWORDS - {""}
    if not intent_words or not line_words:
        return None
    overlap = intent_words & line_words
    ratio = len(overlap) / max(1, len(intent_words))
    if ratio < min_overlap_ratio:
        return ValidationFinding(
            severity=_SEVERITY_WARN,
            code=CODE_OFF_BEAT,
            beat_id=beat.beat_id,
            speaker=speaker_name,
            message=(
                f"Line has {len(overlap)} content-word overlap with "
                f"beat.intent (ratio={ratio:.2f}, threshold "
                f"={min_overlap_ratio:.2f}). May be off-topic."
            ),
            expected=f">={min_overlap_ratio:.2f}",
            got=f"{ratio:.2f}",
        )
    return None


# ---------------------------------------------------------------------------
# Top-level orchestrators
# ---------------------------------------------------------------------------


def validate_line(
    plan: Stage1Plan,
    beat: Stage1Beat,
    line_text: str,
    *,
    banned_phrases: "Optional[Sequence[str]]" = None,
    min_on_beat_ratio: float = 0.10,
) -> ValidationResult:
    """Run all six Stage 3 validators on a single line.

    Resolves the speaker (cast member) via beat.speaker == cast.name.
    Reserved speakers (ANNOUNCER, MUSIC) get a subset of the
    validators since they don't have a cast row with pronouns.
    """
    result = ValidationResult()

    # Resolve speaker: cast row OR a reserved speaker.
    speaker_member: Optional[Stage1CastMember] = next(
        (c for c in plan.cast if c.name == beat.speaker),
        None,
    )
    speaker_name = beat.speaker  # always available even for reserved

    # Length validator applies to ALL beats (including ANNOUNCER /
    # MUSIC) -- a 0-word line is unrenderable regardless.
    finding = validate_length(beat, line_text, speaker_name)
    if finding is not None:
        result.findings.append(finding)

    # Speaker-leak applies to all beats too.
    finding = validate_speaker_leak(beat, line_text, speaker_name)
    if finding is not None:
        result.findings.append(finding)

    # BUG-LOCAL-288: SFW profanity (PD4 hard rule). Runs BEFORE
    # banned phrases so an SFW-violating line that ALSO contains a
    # banned phrase surfaces the SFW finding first -- profanity is
    # the more serious ship-stop. The compose_line repair branch
    # treats both shapes the same way (one regenerate attempt).
    finding = validate_sfw(beat, line_text, speaker_name)
    if finding is not None:
        result.findings.append(finding)

    # Banned phrases.
    finding = validate_banned_phrases(
        beat, line_text, speaker_name, banned=banned_phrases,
    )
    if finding is not None:
        result.findings.append(finding)

    # On-beat.
    finding = validate_on_beat(
        beat, line_text, speaker_name, min_overlap_ratio=min_on_beat_ratio,
    )
    if finding is not None:
        result.findings.append(finding)

    # Pronoun + continuity need a real cast member.
    if speaker_member is not None:
        finding = validate_pronoun_consistency(
            beat, line_text, speaker_member,
        )
        if finding is not None:
            result.findings.append(finding)

        finding = validate_continuity(
            beat, line_text, speaker_name, plan.running_facts,
        )
        if finding is not None:
            result.findings.append(finding)

    return result


def validate_episode(
    plan: Stage1Plan,
    rendered_lines: List[dict],
    *,
    banned_phrases: "Optional[Sequence[str]]" = None,
    min_on_beat_ratio: float = 0.10,
) -> dict[str, ValidationResult]:
    """Run all validators across every line of a rendered episode.

    Args:
        plan: the Stage 1 plan the episode was rendered from.
        rendered_lines: list of dicts shaped like an L3 ledger line
            row (must have at least 'beat_id' and 'text' keys).

    Returns:
        dict mapping beat_id -> ValidationResult for that line. Beats
        without a matching rendered line are skipped silently (caller
        decides whether missing-line is itself a finding).
    """
    by_beat_id = {b.beat_id: b for b in plan.beats}
    results: dict[str, ValidationResult] = {}
    for line in rendered_lines:
        if not isinstance(line, dict):
            continue
        bid = line.get("beat_id")
        if not isinstance(bid, str):
            continue
        beat = by_beat_id.get(bid)
        if beat is None:
            continue
        text = line.get("text") or ""
        results[bid] = validate_line(
            plan, beat, text,
            banned_phrases=banned_phrases,
            min_on_beat_ratio=min_on_beat_ratio,
        )
    return results
