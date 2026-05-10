"""nodes/_otr_line_composer.py

Per-beat dialogue line generation for the v2.0 LedgerScriptWriter path.

Takes one Beat + EpisodeCanon header + last N ledger lines, generates ONE
raw dialogue string from the LLM, strips any leaked formatting (speaker
prefixes, brackets, markdown, wrapping quotes), returns the cleaned text.

The LLM is told to output only the spoken line. Python attaches the
[VOICE: NAME, traits] format tag deterministically at ledger-stamp time
(in OTR_LedgerScriptWriter, not here). This module never produces or
expects format markup.

Status: Phase 2 of v2.0 sprint. Companion to _otr_outline.py.

Public surface:
    LineRequest                   -- frozen dataclass: per-line input
    LineCompositionFailedError    -- raised after 2 failed attempts
    compose_line(...)             -- main entrypoint
    strip_line_formatting(...)    -- public for testing / one-shot use
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger("OTR")


__all__ = [
    "LineRequest",
    "LineCompositionFailedError",
    "compose_line",
    "strip_line_formatting",
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
# Request dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LineRequest:
    """Per-beat input for compose_line.

    Fields are duplicated from Beat (rather than passing the Beat directly)
    to keep this module's import surface stdlib-only at module load. The
    caller maps Beat fields into LineRequest.
    """

    speaker: str
    intent: str
    mood: str
    target_words: int
    canon_header: str               # from render_episode_canon_header()
    last_lines: list[tuple[str, str]]  # [(speaker, text), ...] most recent last; empty for first beat


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
    return (
        f"EPISODE CONTEXT\n"
        f"{req.canon_header}\n\n"
        f"RECENT DIALOGUE (most recent at bottom):\n"
        f"{_format_last_lines(req.last_lines)}\n\n"
        f"NEXT LINE\n"
        f"Speaker: {req.speaker}\n"
        f"This line accomplishes: {req.intent}\n"
        f"Mood: {req.mood}\n"
        f"Target word count: ~{req.target_words}\n\n"
        f"Write the line. Output only the spoken text."
    )


# ---------------------------------------------------------------------------
# compose_line -- main entrypoint
# ---------------------------------------------------------------------------


def compose_line(
    generate_fn,                # same GenerateFn contract as _otr_outline
    req: LineRequest,
    *,
    max_attempts: int = 2,
    base_temperature: float = _BASE_TEMPERATURE,
) -> str:
    """Compose one cleaned dialogue line for a beat.

    Retry strategy:
      Attempt 1: temperature = base_temperature (0.8).
      Attempt 2: temperature = base_temperature + 0.1 (0.9).

    Failure conditions that trigger retry:
      - generate_fn raises.
      - cleaned response is empty.
      - cleaned response is more than _MAX_OVERSIZE_RATIO * target_words long.

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

        log.info(
            "[OTR_LineComposer] success on attempt %d/%d: %d words for %s",
            attempt_idx + 1, max_attempts, word_count, req.speaker,
        )
        return cleaned

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
    for required in ("EPISODE CONTEXT", "RECENT DIALOGUE", "NEXT LINE",
                     "Speaker: ALICE", "Mood: tense", "~15"):
        assert required in user_prompt, f"missing {required!r}"
    print("  PASS")

    # Test 4: compose_line happy path with mock generate_fn.
    print("\n[Test 4] compose_line happy path")
    def mock_ok(messages, *, temperature, max_new_tokens):
        return "ALICE: I found something I cannot explain."
    result = compose_line(mock_ok, req)
    assert result == "I found something I cannot explain."
    print(f"  PASS (cleaned: {result!r})")

    # Test 5: compose_line retries on empty.
    print("\n[Test 5] compose_line retries on empty response")
    call_count = {"n": 0}
    def mock_empty_then_ok(messages, *, temperature, max_new_tokens):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return "**[ALICE]**"  # strips to empty
        return "I see it now."
    result = compose_line(mock_empty_then_ok, req)
    assert result == "I see it now."
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
    assert result == "Short reply."
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
    assert result == "Recovered line."
    print("  PASS")

    print("\n=== Task 3 self-tests passed ===")
