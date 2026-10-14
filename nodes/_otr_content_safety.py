"""The retired spoken-content vocabulary -- READ-ONLY, and it stays that way.

Three word lists (profanity, explicit weapons, explicit sexual/nudity) and a
whole-word matcher. Nothing here judges, blocks, rewrites or refuses anything.

There is no rewrite or filtering pass: the operator's 2026-08-03 directive --
*"no violence or swearing guardrails, they just cause problems"* -- forbids one,
and on an adaptation lane it would mean editing Shakespeare.
`_otr_ledger_cleanup` stamps `safety.status = "retired"` on every path, so the
ledger field keeps its owner.

WHY THE VOCABULARY IS KEPT. The directive bans FILTERING, not
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

from functools import lru_cache
import re
from typing import Any


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


__all__ = [
    "EXPLICIT_NUDITY_TERMS",
    "PROFANITY_TERMS",
    "EXPLICIT_WEAPON_TERMS",
    "find_text_hits",
    "profanity_terms",
]
