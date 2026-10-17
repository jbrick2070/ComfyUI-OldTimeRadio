"""Every cast row ends male or female -- the one ladder, for every lane.

OPERATOR RULING (2026-10-17): "unisex names is ok but we need to clamp male or
female in the ledger and voice and descriptions for rendering prompts, stills
and video". A name may stay unisex; the GENDER on the ledger cast row may not.
Voice casting picks from that stamped gender, and the still and video prompts
anchor on it, so nothing downstream ever meets an unresolved "other".

WHERE IT RUNS. The producers resolve gender BEFORE any model writes a bio or a
line (`_otr_casting` no longer deals an "other" share; My Story settles every
role before its acts; Sci-Fi News Pro's casting schema is already binary).
`production_ledger.Ledger.set_cast` calls :func:`clamp_cast_rows` as the
safety net every lane passes through, and says so LOUDLY when it fires,
because a clamp there means a producer let a non-binary gender through.

THE LADDER, strongest evidence first (each rung declines rather than guess):
  1. a courtesy title the name LEADS with ("Mrs. Vance", "Sir Toby");
  2. a relation word in the role ("her brother"), possessives skipped;
  3. our name pool, honorifics stripped (unisex / unknown names decline);
  4. subject/reflexive pronouns in the description -- only when the
     description names no other cast member and they all agree;
  5. a coin seeded on the normalized name, so the same name casts the same way.

Pure, deterministic, stdlib + repo modules only.
"""
from __future__ import annotations

import logging
import random
import re
from typing import Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

try:
    from . import _otr_roster_gender as _RG
    from ._otr_character_roster import _FEMALE_RELATION, _MALE_RELATION, _scan
except ImportError:  # pragma: no cover -- flat load
    import _otr_roster_gender as _RG  # type: ignore
    from _otr_character_roster import (  # type: ignore
        _FEMALE_RELATION, _MALE_RELATION, _scan)

try:
    from ..config import cast_pools as _POOLS
except ImportError:  # pragma: no cover -- flat load
    from config import cast_pools as _POOLS  # type: ignore

log = logging.getLogger("OTR")

BINARY = ("male", "female")

_SUBJECT_PRONOUNS = {
    "she": "female", "herself": "female", "he": "male", "himself": "male",
}
#: A pronoun IMMEDIATELY after one of these opens a reported clause, usually
#: about someone else ("doubts that she saw lights").
_SUBORDINATORS = frozenset({"that", "whether"})
#: Words a cast name can contain that are not that person ("The Doctor").
_NAME_ARTICLES = frozenset({"the", "a", "an", "of", "and", "de", "la", "le",
                            "el", "los", "las", "du", "von", "van"})


def binary(raw) -> str:
    """``male``/``female`` for any spelling of them, else ``""``."""
    g = _RG.canonical_bank_gender(raw)
    return g if g in BINARY else ""


def name_forms(name: str) -> "tuple[tuple[str, ...], ...]":
    """A name's word forms: honorifics dropped, and as typed."""
    words = tuple(re.findall(r"[^\W_]+(?:['\-][^\W_]+)*",
                             str(name or "").casefold()))
    kept = tuple(w for w in words if w not in _RG._HONORIFICS)
    return tuple(dict.fromkeys(f for f in (kept, words) if f))


def name_tokens(name: str) -> "tuple[str, ...]":
    """Casefolded words of a name, honorifics dropped (as typed if that
    leaves nothing)."""
    forms = name_forms(name)
    return forms[0] if forms else ()


def description_gender(description: str, others: Sequence[str] = ()) -> str:
    """The gender a description's own subject/reflexive pronouns state, or "".

    Object and possessive pronouns are no evidence ("her brother"). A pronoun
    right after "that"/"whether" opens a reported clause and is skipped. A
    description that names another cast member (articles aside) gives no
    pronoun evidence at all -- no word window can tell whose pronoun it is.
    """
    words = re.findall(r"[^\W_]+", str(description or "").lower())
    names = {w for name in others for form in name_forms(name)
             for w in form} - _NAME_ARTICLES
    if names.intersection(words):
        return ""
    seen = {_SUBJECT_PRONOUNS[w] for i, w in enumerate(words)
            if w in _SUBJECT_PRONOUNS
            and not (i and words[i - 1] in _SUBORDINATORS)}
    return next(iter(seen)) if len(seen) == 1 else ""


def settle_binary_gender(name: str, role: str = "", description: str = "",
                         others: Sequence[str] = ()) -> "tuple[str, str]":
    """``(gender, source)`` -- always ``male`` or ``female``. See the ladder
    in the module docstring."""
    gender = _RG.slot_honorific_gender(name)
    if gender in BINARY:
        return gender, "courtesy_title"
    role_text = re.sub(r"\b[a-z]+'s\b", " ", str(role or "").lower())
    gender = _scan(re.findall(r"[a-z]+", role_text),
                   _MALE_RELATION, _FEMALE_RELATION)
    if gender in BINARY:
        return gender, "role_relation"
    gender = _POOLS.gender_of_first_name(" ".join(name_tokens(name)))
    if gender in BINARY:
        return gender, "name_pool"
    gender = description_gender(description, others)
    if gender:
        return gender, "description_pronouns"
    key = " ".join(str(name or "").split()).casefold()
    coin = random.Random("otr_cast_gender:%s" % key)
    return ("male" if coin.random() < 0.5 else "female"), "seeded_coin"


def clamp_cast_rows(rows: Sequence[MutableMapping]) -> "list[dict]":
    """Clamp every row's gender to male/female IN PLACE; return receipts.

    A row already male/female (any spelling) is canonicalized and otherwise
    untouched. Mutating the caller's dicts is deliberate: a writer that keeps
    reading its own rows after `set_cast` must see the same gender the ledger
    holds.
    """
    receipts: List[dict] = []
    names = [str(r.get("name") or "") for r in rows]
    for i, row in enumerate(rows):
        before = row.get("gender")
        g = binary(before)
        if g:
            if before != g:
                row["gender"] = g
            continue
        others = [n for j, n in enumerate(names) if j != i]
        g, source = settle_binary_gender(
            names[i], str(row.get("role") or ""),
            str(row.get("character_description") or ""), others)
        row["gender"] = g
        receipts.append({"char_id": str(row.get("char_id") or ""),
                         "name": names[i], "gender": g, "source": source,
                         "was": before})
        log.warning(
            "[cast_gender] CLAMPED %r (%s) from %r to %s (%s) -- a producer "
            "let a non-binary gender reach the ledger", names[i],
            row.get("char_id"), before, g, source)
    return receipts


__all__ = [
    "BINARY", "binary", "clamp_cast_rows", "description_gender",
    "name_forms", "name_tokens", "settle_binary_gender",
]
