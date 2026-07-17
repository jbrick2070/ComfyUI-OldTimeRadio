"""Bank-aware genre / spoken-text boundary guard (v4 campaign, P1(iii)).

WHY. Each non-sci-fi v4 lane (shakespeare, public_domain, media_archive) must
write in its OWN idiom, never bleed into sci-fi; original_radio must hold its
noir register. The lanes already carry a per-lane ``story_rules.banned_phrases``
list, but today those gate only the PROMPT (soft) -- nothing scans the spoken
text that actually ships. And the legacy ``_otr_stage3_validators.
validate_banned_phrases`` is a raw case-insensitive SUBSTRING check: banning
``gun`` would fire on ``begun``, ``RSS`` inside another word, etc.

WHAT. Two cooperating parts, BOTH opt-in per bank via the validated scalar
default ``defaults.genre_guard_spoken`` (bool, default False -> INERT for every
current bank; a v4 bank flips it True in Phase 2):

  1. ``find_banned_phrase_hits(text, phrases)`` -- a NEW casefolded, Unicode-aware
     token/phrase-BOUNDARY matcher. ``gun`` never fires on ``begun`` / ``shotgun``;
     a single-word entry also matches a simple English plural (+s/+es) so banning
     ``spaceship`` catches ``spaceships``; a multi-word entry matches with flexible
     internal whitespace, bounded at both ends. Pure; never raises.

  2. ``apply_genre_spoken_repair(...)`` -- the WRITER-boundary AUTHORED repair: for
     each spoken line hitting a banned phrase, request a bounded re-author from
     the creative slot; keep the rewrite only if it is non-empty AND clean; one
     retry; else leave the original untouched. Never raises -- a repair that fails
     is not a ship-stop.

THE LAW (operator 2026-07-13). An audit may IMPROVE a story, it may never FAIL
one; only a DETERMINISTIC validator may end an episode. So the authored repair
here only improves; the terminal is the read-only Phase-10 scan
``_otr_ledger_freeze._check_g10_genre_spoken_text`` (inside ``run_gap_audit`` --
the one path EVERY execution family crosses: codex's ``phase_10_gap_audit_post_
and_freeze`` finalizer, the inline ``run_freeze_cascade`` node, and fable2's
finalizer). It reads ``meta['genre_banned_phrases']`` (stamped by the writer only
when the bank opts in) and raises ``FreezeAssertionError`` on a surviving hit,
exactly like the G9 SFW gate.

Guarded boundaries by path:
  * inline lanes -- writer ``OTR_LedgerScriptWriter.run()`` last creative pass,
    just after the self-vocative attribution repair, before the J rollup + tail.
  * codex lane -- deterministic terminal covers it now; its own authored repair
    lands with ``scifi_codex_v4`` (Phase 2), at its structured slot seam.

Self-contained (no ledger/rules imports -> no circular import from the freeze
module). UTF-8 no BOM, SFW.
"""
from __future__ import annotations

import re
from typing import Any, Callable, List, Optional, Sequence, Tuple

__all__ = [
    "find_banned_phrase_hits",
    "is_spoken_role",
    "scan_spoken_lines",
    "apply_genre_spoken_repair",
]


# ---------------------------------------------------------------------------
# 1. Boundary matcher
# ---------------------------------------------------------------------------

# `\w` is Unicode-aware for `str` patterns in Python 3, so the word-boundary
# lookarounds treat accented letters as word characters (cafe vs cafe-au-lait).
_PLURAL_SUFFIX = r"(?:es|s)?"


def _compile_phrase(phrase: str) -> "Optional[re.Pattern[str]]":
    """Compile ONE banned entry into a casefolded boundary pattern.

    Single token -> ``(?<!\\w)<tok>(?:es|s)?(?!\\w)`` (word-bounded, simple
    plural). Multi-word -> tokens joined by ``\\s+`` and bounded at both ends.
    Returns None for an empty / whitespace-only entry.
    """
    p = (phrase or "").strip().casefold()
    if not p:
        return None
    if any(ch.isspace() for ch in p):
        body = r"\s+".join(re.escape(tok) for tok in p.split())
        return re.compile(rf"(?<!\w){body}(?!\w)")
    return re.compile(rf"(?<!\w){re.escape(p)}{_PLURAL_SUFFIX}(?!\w)")


def _compiled_phrases(
    phrases: "Sequence[str]",
) -> "List[Tuple[str, re.Pattern[str]]]":
    out: "List[Tuple[str, re.Pattern[str]]]" = []
    for ph in phrases or ():
        if not isinstance(ph, str):
            continue
        pat = _compile_phrase(ph)
        if pat is not None:
            out.append((ph, pat))
    return out


def find_banned_phrase_hits(
    text: Any, phrases: "Sequence[str]",
) -> "List[str]":
    """Return the banned entries (original casing, de-duped, input order) whose
    boundary pattern matches ``text``. Casefolded + Unicode-aware. Pure; a
    non-string ``text`` or empty ``phrases`` yields ``[]``. Never raises."""
    if not isinstance(text, str) or not text:
        return []
    try:
        hay = text.casefold()
        seen: set = set()
        hits: "List[str]" = []
        for original, pat in _compiled_phrases(phrases):
            if original in seen:
                continue
            if pat.search(hay):
                seen.add(original)
                hits.append(original)
        return hits
    except Exception:  # noqa: BLE001 -- matching is never fatal
        return []


# ---------------------------------------------------------------------------
# Spoken-row scope (exchange / inline / repaired / announcer / outro -- NOT music)
# ---------------------------------------------------------------------------

def is_spoken_role(role: Any) -> bool:
    """True for a spoken row (announcer / character / outro / untyped); False
    for a music bookend / interstitial. Music rows carry a generation prompt,
    not spoken text, and must not be scanned for genre bleed."""
    r = str(role or "").strip().lower()
    if not r:
        return True  # untyped row -> conservatively in scope
    return not r.startswith("music")


def scan_spoken_lines(
    ledger_data: Any, phrases: "Sequence[str]",
) -> "List[Tuple[str, List[str]]]":
    """Return ``[(line_id, [hits...]), ...]`` for every spoken line whose text
    hits a banned phrase. Read-only; never raises. Shared by the deterministic
    terminal and the authored repair so they scan identically."""
    result: "List[Tuple[str, List[str]]]" = []
    if not isinstance(ledger_data, dict):
        return result
    lines = ledger_data.get("lines")
    if not isinstance(lines, list) or not phrases:
        return result
    for line in lines:
        if not isinstance(line, dict):
            continue
        if not is_spoken_role(line.get("speaker_role")):
            continue
        text = line.get("text")
        if not isinstance(text, str) or not text:
            continue
        hits = find_banned_phrase_hits(text, phrases)
        if hits:
            result.append((str(line.get("line_id") or "<no line_id>"), hits))
    return result


# ---------------------------------------------------------------------------
# 2. Writer-boundary authored repair (creative slot; keep-if-clean)
# ---------------------------------------------------------------------------

def _append_flag(line: dict, flag: str) -> None:
    flags = line.get("compose_flags")
    if not isinstance(flags, list):
        flags = []
        line["compose_flags"] = flags
    flags.append(flag)


def _repair_messages(text: str, hits: "Sequence[str]") -> list:
    """OpenAI-style chat messages for a single-line, in-idiom re-author."""
    banned = ", ".join(hits)
    return [
        {
            "role": "system",
            "content": (
                "You revise a single line of radio dialogue. Keep the speaker, "
                "meaning, and register identical. Remove the listed forbidden "
                "words/phrases and any synonym for them. Do not add narration, "
                "quotation marks, a speaker name, or stage directions. Return "
                "ONLY the revised line as plain prose."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Forbidden (must not appear, in any casing or plural): {banned}\n"
                f"Line: {text}"
            ),
        },
    ]


def _reauthor_line(
    text: str,
    hits: "Sequence[str]",
    creative_fn: "Callable[..., Any]",
    phrases: "Sequence[str]",
    attempts: int,
) -> "Optional[str]":
    """Ask the creative slot for a clean rewrite. Return the first non-empty
    candidate that carries NO banned hit, else None. Never raises."""
    budget = min(256, max(32, len(text.split()) * 3 + 32))
    for _ in range(max(1, attempts)):
        try:
            # LLM slot: creative -- the writer's creative slot re-authors the
            # spoken line; the deterministic Phase-10 G10 terminal is the gate.
            raw = creative_fn(
                _repair_messages(text, hits),
                temperature=0.7,
                max_new_tokens=budget,
                stop=None,
            )
        except Exception:  # noqa: BLE001 -- a failed slot call is not a ship-stop
            return None
        cand = str(raw or "").strip().strip('"').strip()
        if cand and not find_banned_phrase_hits(cand, phrases):
            return cand
    return None


def apply_genre_spoken_repair(
    ledger_data: Any,
    phrases: "Sequence[str]",
    creative_fn: "Optional[Callable[..., Any]]",
    *,
    max_attempts: int = 2,
) -> int:
    """Re-author spoken lines hitting a banned genre phrase. Returns the count
    of lines actually rewritten.

    Keeps a rewrite only when it is non-empty AND clean; otherwise leaves the
    original for the deterministic Phase-10 G10 terminal to catch. Stamps a
    ``genre_repair:removed=...`` compose-flag breadcrumb on each rewritten row
    (no silent mutation). Never raises; a no-op when disabled/empty."""
    if creative_fn is None or not isinstance(ledger_data, dict) or not phrases:
        return 0
    lines = ledger_data.get("lines")
    if not isinstance(lines, list):
        return 0
    fixed = 0
    for line in lines:
        if not isinstance(line, dict):
            continue
        if not is_spoken_role(line.get("speaker_role")):
            continue
        text = line.get("text")
        if not isinstance(text, str) or not text:
            continue
        hits = find_banned_phrase_hits(text, phrases)
        if not hits:
            continue
        new_text = _reauthor_line(text, hits, creative_fn, phrases, max_attempts)
        if new_text and new_text != text:
            line["text"] = new_text
            line["word_count"] = len(new_text.split())
            _append_flag(line, f"genre_repair:removed={','.join(hits)}")
            fixed += 1
    return fixed
