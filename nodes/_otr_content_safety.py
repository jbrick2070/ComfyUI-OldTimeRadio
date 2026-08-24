"""The retired spoken-content vocabulary -- READ-ONLY, and it stays that way.

Three word lists (profanity, explicit weapons, explicit sexual/nudity) and a
whole-word matcher. Nothing here judges, blocks, rewrites or refuses anything.

THE REWRITE HALF IS GONE (2026-08-23, the lean-mean OPEN item). This module used
to carry `propose_safety_patches` and `apply_safety_cleanup`: an LLM-driven pass
that took a delivered spoken row matching one of these lists and REWROTE IT.
That is precisely what the operator's 2026-08-03 directive forbids -- *"no
violence or swearing guardrails, they just cause problems"* -- and on an
adaptation lane it meant editing Shakespeare. The pass had already been unwired
at its caller (`_otr_ledger_cleanup` stamps `safety.status = "retired"` on every
path, so the ledger field keeps its owner) and `validate_sfw` had already been
gutted to `return None`. What remained here was 165 lines of dormant rewrite
machinery that anything could have re-armed, plus two bare `RuntimeError`s that
would have killed a render if it ever ran.

WHY THE VOCABULARY SURVIVES THE MACHINERY. The directive bans FILTERING, not
knowing the words. `tests/test_bug_local_288_sfw_validator.py` keeps the whole
retired list green on purpose -- every term must PASS a line -- and says why: a
deleted test is silence, and silence is how a policy creeps back. Keeping the
tuples means a future NON-blocking use (a report, a tag, an advisory the
operator asked for) does not have to re-derive them, and it keeps that guard
able to enumerate what must never block again.

**If you are here to wire this into the generation path: don't.** A blocking
caller is the thing that was removed, twice.

Pure and stdlib-only: no pydantic, no LLM, no ledger writes.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import re
from typing import Any, Mapping


PROFANITY_TERMS: tuple[str, ...] = (
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
    "screw you",
    "screw this",
    "screw that",
)

EXPLICIT_WEAPON_TERMS: tuple[str, ...] = (
    "gun",
    "guns",
    "handgun",
    "handguns",
    "pistol",
    "pistols",
    "revolver",
    "revolvers",
    "rifle",
    "rifles",
    "shotgun",
    "shotguns",
    "firearm",
    "firearms",
    "knife",
    "knives",
    "dagger",
    "daggers",
    "switchblade",
    "switchblades",
    "weapon",
    "weapons",
)

EXPLICIT_NUDITY_TERMS: tuple[str, ...] = (
    "nude",
    "nudity",
    "naked",
    "porn",
    "pornographic",
    "pornography",
    "explicit sex",
    "sexual intercourse",
)

SPOKEN_ROLES = frozenset({"character", "announcer"})


@dataclass(frozen=True)
class SafetyHit:
    line_id: str
    line_index: int
    category: str
    term: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def profanity_terms() -> tuple[str, ...]:
    """Return the global narrow profanity vocabulary."""
    return PROFANITY_TERMS


@lru_cache(maxsize=16)
def _pattern(terms: tuple[str, ...]) -> re.Pattern[str]:
    ordered = sorted(set(terms), key=len, reverse=True)
    if not ordered:
        return re.compile(r"(?!x)x")
    body = "|".join(re.escape(term) for term in ordered)
    return re.compile(rf"(?<!\w)(?:{body})(?!\w)", re.IGNORECASE)


def find_text_hits(text: Any) -> tuple[tuple[str, str], ...]:
    """Return unique category/matched-term pairs in stable order."""
    surface = str(text or "")
    if not surface:
        return ()
    categories = (
        ("profanity", profanity_terms()),
        ("weapon", EXPLICIT_WEAPON_TERMS),
        ("sexual_nudity", EXPLICIT_NUDITY_TERMS),
    )
    found: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for category, terms in categories:
        for match in _pattern(tuple(terms)).finditer(surface):
            row = (category, match.group(0).casefold())
            if row not in seen:
                seen.add(row)
                found.append(row)
    return tuple(found)


def scan_spoken_ledger(ledger_data: Mapping[str, Any]) -> tuple[SafetyHit, ...]:
    """Scan the exact non-skipped character/announcer projection."""
    hits: list[SafetyHit] = []
    lines = ledger_data.get("lines") if isinstance(ledger_data, Mapping) else ()
    for index, row in enumerate(lines or ()):
        if not isinstance(row, Mapping):
            continue
        if bool(row.get("skip")):
            continue
        if str(row.get("speaker_role") or "") not in SPOKEN_ROLES:
            continue
        line_id = str(row.get("line_id") or f"lines[{index}]")
        for category, term in find_text_hits(row.get("text")):
            hits.append(SafetyHit(line_id, index, category, term))
    return tuple(hits)


def format_safety_hits(hits: Iterable[SafetyHit], *, limit: int = 8) -> str:
    rows = list(hits)
    sample = rows[: max(1, int(limit))]
    rendered = "; ".join(
        f"{hit.line_id}: {hit.category}={hit.term!r}"
        for hit in sample
    )
    if len(rows) > len(sample):
        rendered += f" (+{len(rows) - len(sample)} more)"
    return rendered


__all__ = [
    "EXPLICIT_NUDITY_TERMS",
    "PROFANITY_TERMS",
    "EXPLICIT_WEAPON_TERMS",
    "SPOKEN_ROLES",
    "SafetyHit",
    "find_text_hits",
    "format_safety_hits",
    "profanity_terms",
    "scan_spoken_ledger",
]
