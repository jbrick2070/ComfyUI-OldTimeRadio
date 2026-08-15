"""Read a character's gender out of the author's own pronouns. Offline, pure.

WHY THIS EXISTS. `cast_source_contract.gender_by_name` was `{}` on every
public_domain episode, because 64 of 65 units carry no provenance sidecar and
the sidecar is where the fact lives. With the field empty the caster fell back
to a blind 40/40/20 roll, and the roll inverted real characters three times on
three different sources: GERTRUDE (a woman) male on all six lines, LORD RONALD
(a man) female on all six -- the script makes it audible, the male-voiced
character is addressed *"Miss McFiggins"* -- and then AHAB, of Moby-Dick, in a
woman's voice.

THE QUESTION WAS NEVER HARD. Melville writes "he" of Ahab throughout; Leacock
writes "she" of Gertrude. The fact is sitting in the source text the vendor
already downloaded. What was missing was the plumbing to carry it, so this
module is deliberately the DUMBEST thing that can work: count the author's
pronouns near each mention of the name, and require a decisive margin.

IT DECLINES RATHER THAN GUESSES, and that is the whole design. A wrong pin is
worse than no pin: no pin leaves today's roll, which is a coin flip nobody
promised anything about, while a wrong pin is a confident lie that silences the
roll and ships a man speaking as a woman. So the floor is a MARGIN, not a
majority -- below it this returns no verdict and the caller is free to reach for
a later tier or leave the roll alone.

Vendor-time only. Nothing on the render path imports this; the sidecar it helps
write is what the render path reads. It lives beside `_otr_character_roster`
because it borrows that module's period-tuned title and relation vocabulary
verbatim -- restating those lists here is exactly the two-copies-of-one-fact
mistake that `BUG_BIBLE.yaml` 12.86 is about.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

try:  # package import (normal)
    from ._otr_character_roster import (
        _FEMALE_RELATION, _FEMALE_TITLE, _MALE_RELATION, _MALE_TITLE,
    )
except ImportError:  # pragma: no cover -- flat test/standalone load
    from _otr_character_roster import (  # type: ignore
        _FEMALE_RELATION, _FEMALE_TITLE, _MALE_RELATION, _MALE_TITLE,
    )

__all__ = [
    "MENTION_WINDOW_CHARS",
    "SCORE_FLOOR",
    "DOMINANCE_RATIO",
    "PronounVerdict",
    "mention_forms",
    "scan_gender",
]

#: Characters after a mention that still plausibly refer to it. Long enough to
#: catch "Ahab said nothing for a moment, then he ..." and short enough that the
#: next character's paragraph is usually out of reach.
MENTION_WINDOW_CHARS = 240

#: A decision needs BOTH a floor and a margin. The floor stops a single stray
#: pronoun from deciding a barely-mentioned name; the ratio stops a genuinely
#: mixed scene (two people talking about each other) from being read as
#: evidence. Calibration is tunable; DECLINING BELOW THE FLOOR IS NOT.
SCORE_FLOOR = 4
DOMINANCE_RATIO = 3.0

#: A gendered word carried by the name itself ("Lord Ronald", "the Governess")
#: or standing immediately in front of a mention ("Mrs. Sappleton") is the
#: author naming the role outright, so it outweighs a lone pronoun. Counted ONCE
#: for the name itself -- otherwise a frequently-mentioned title would swamp
#: every pronoun in the text and the scan would stop being a scan.
NAMING_WEIGHT = 3
PRONOUN_WEIGHT = 1

_MALE_PRONOUNS = frozenset({"he", "him", "his", "himself"})
_FEMALE_PRONOUNS = frozenset({"she", "her", "hers", "herself"})

_MALE_RELATION_SET = frozenset(w.lower() for w in _MALE_RELATION)
_FEMALE_RELATION_SET = frozenset(w.lower() for w in _FEMALE_RELATION)
_MALE_TITLE_SET = frozenset(w.lower() for w in _MALE_TITLE)
_FEMALE_TITLE_SET = frozenset(w.lower() for w in _FEMALE_TITLE)

_MALE_WORDS = _MALE_RELATION_SET | _MALE_TITLE_SET
_FEMALE_WORDS = _FEMALE_RELATION_SET | _FEMALE_TITLE_SET

_ARTICLES = ("the ", "a ", "an ")
_WORD_RE = re.compile(r"[a-z']+")


@dataclass(frozen=True)
class PronounVerdict:
    """What the author's own words say, and how strongly."""

    gender: str          # "male" | "female" | "" when it declined
    male_score: int
    female_score: int
    mentions: int
    evidence: str

    @property
    def decided(self) -> bool:
        return self.gender in ("male", "female")


def _strip_article(name: str) -> str:
    low = name.lower()
    for article in _ARTICLES:
        if low.startswith(article):
            return name[len(article):]
    return name


def mention_forms(name: str) -> Tuple[str, ...]:
    """Every string in the text that plausibly refers to this character.

    The full name, the same without a leading article ("the water ghost" ->
    "water ghost"), and for a multi-token name its final token, because prose
    introduces "Captain Ahab" once and then says "Ahab" four hundred times.
    """
    base = str(name or "").strip()
    if not base:
        return ()
    forms = [base]
    bare = _strip_article(base)
    if bare and bare not in forms:
        forms.append(bare)
    tokens = [t for t in bare.split() if t]
    if len(tokens) > 1 and tokens[-1] not in forms:
        forms.append(tokens[-1])
    seen = []
    for form in forms:
        low = form.lower().strip()
        if low and low not in seen:
            seen.append(low)
    return tuple(seen)


def _shared_forms(name: str, other_names: Iterable[str]) -> frozenset:
    """Mention forms this name shares with another candidate in the same unit.

    `monkeys_paw` has TWO Whites. A window opened by "White" belongs to neither
    of them, so a shared form is counted for NOBODY rather than for both --
    cross-contamination would produce a confident verdict built from the other
    character's pronouns, which is the worst possible failure here.
    """
    mine = set(mention_forms(name))
    shared = set()
    for other in other_names or ():
        if str(other).strip().lower() == str(name).strip().lower():
            continue
        shared |= mine & set(mention_forms(other))
    return frozenset(shared)


def _self_designation(name: str) -> str:
    """The gender the name states about ITSELF, if it states one.

    Relation words before titles, matching `_otr_character_roster.infer_gender`
    -- "daughter to Duke Senior" is a woman despite the Duke, and the two
    modules must not disagree about a name they may both see.
    """
    words = _WORD_RE.findall(str(name or "").lower())
    for word in words:
        if word in _MALE_RELATION_SET:
            return "male"
        if word in _FEMALE_RELATION_SET:
            return "female"
    for word in words:
        if word in _MALE_TITLE_SET:
            return "male"
        if word in _FEMALE_TITLE_SET:
            return "female"
    return ""


def _preceding_word(text: str, start: int) -> str:
    """The word standing immediately before a mention ('Mrs.' in 'Mrs. White')."""
    lead = text[max(0, start - 24):start].lower()
    words = _WORD_RE.findall(lead)
    return words[-1] if words else ""


def scan_gender(
    name: str, text: str, *, other_names: Sequence[str] = (),
) -> PronounVerdict:
    """Count the author's pronouns around every mention, and decide or decline.

    Returns a verdict whose `gender` is "" when the evidence did not clear both
    the floor and the margin. That is a real answer, not a failure: the caller
    leaves the existing roll in place rather than pinning something it cannot
    support.
    """
    body = str(text or "")
    excluded = _shared_forms(name, other_names)
    forms = [f for f in mention_forms(name) if f not in excluded]

    # A NAME THAT STATES ITS OWN GENDER IS DECIDED BY THAT STATEMENT, before any
    # counting. This began life as merely the heaviest signal, and LORD RONALD
    # is why it is not: measured on the real text he scored male 78 / female 36,
    # which clears the floor and FAILS the 3x margin, so the scan declined a man
    # whose name is the word "Lord". The cause is structural, not a calibration
    # miss -- `gertrude_governess` is a romance, so the two leads share nearly
    # every mention window and each one's pronouns pollute the other's count. No
    # ratio fixes that without also deciding cases that genuinely are ambiguous.
    # The author naming the role outranks a tally of nearby pronouns, which is
    # what the SPEC meant by "lets the author's own naming of a role decide it
    # deterministically before any model is consulted".
    stated = _self_designation(name)
    if stated:
        return PronounVerdict(
            gender=stated, male_score=0, female_score=0, mentions=0,
            evidence=(
                "the source's own name for this character states the gender: "
                "%r contains a gendered title or relation word" % (str(name),)
            ),
        )

    male = female = 0
    mentions = 0
    reasons = []

    for form in forms:
        pattern = re.compile(r"\b" + re.escape(form) + r"\b", re.IGNORECASE)
        for match in pattern.finditer(body):
            mentions += 1
            lead = _preceding_word(body, match.start())
            if lead in _MALE_WORDS:
                male += NAMING_WEIGHT
            elif lead in _FEMALE_WORDS:
                female += NAMING_WEIGHT
            window = body[match.end():match.end() + MENTION_WINDOW_CHARS]
            for word in _WORD_RE.findall(window.lower()):
                if word in _MALE_PRONOUNS:
                    male += PRONOUN_WEIGHT
                elif word in _FEMALE_PRONOUNS:
                    female += PRONOUN_WEIGHT

    winner, loser = max(male, female), min(male, female)
    decided = ""
    if mentions and winner >= SCORE_FLOOR and winner >= DOMINANCE_RATIO * loser:
        decided = "male" if male > female else "female"

    evidence = (
        "pronoun scan: he/him/his %d vs she/her/hers %d across %d mention "
        "window(s) for %s" % (
            male, female, mentions,
            ", ".join(repr(f) for f in forms) or "no usable mention form",
        )
    )
    if reasons:
        evidence += " (" + "; ".join(reasons) + ")"
    if not decided:
        evidence += "; DECLINED -- below the floor of %d or the %.1fx margin" % (
            SCORE_FLOOR, DOMINANCE_RATIO,
        )
    if excluded:
        evidence += "; ignored shared form(s) %s" % (sorted(excluded),)

    return PronounVerdict(
        gender=decided, male_score=male, female_score=female,
        mentions=mentions, evidence=evidence,
    )
